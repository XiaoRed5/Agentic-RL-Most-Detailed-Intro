#!/usr/bin/env python3
"""Wait for a complete local Qwen snapshot, upload it, verify it, and launch training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def run(command: list[str], *, timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, text=True, capture_output=True, timeout=timeout)


def log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}\n")


def is_complete(root: Path, manifest: Path) -> bool:
    values = json.loads(manifest.read_text(encoding="utf-8"))
    for item in values["files"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["size"]):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    os.chdir(repo)
    manifest = repo / "artifacts/qwen3_8b_official_manifest.json"
    log_path = repo / "artifacts/qwen3_8b_watch.log"
    download_root = repo.parent.parent / "work/models"
    candidates = [
        download_root / "Qwen3-8B-msfast",
        download_root / "Qwen3-8B",
        download_root / "Qwen3-8B-modelscope",
        download_root / "Qwen3-8B-hpxet",
    ]
    remote_root = "/home/hadoop-aipnlp/agenticqwen-paper-debug-20260804"
    remote_model = remote_root + "/models/Qwen3-8B"
    remote_manifest = remote_root + "/artifacts/qwen3_8b_official_manifest.json"
    while True:
        complete = next((path for path in candidates if is_complete(path, manifest)), None)
        if complete is None:
            sizes = ", ".join(f"{path.name}={sum(item.stat().st_size for item in path.rglob('*') if item.is_file())/1024**3:.2f}GiB" for path in candidates if path.exists())
            log(log_path, f"waiting_for_complete_snapshot {sizes}")
            time.sleep(args.poll_seconds)
            continue

        verify_local = run([
            os.environ.get("PYTHON", "python3"), "scripts/verify_snapshot.py",
            "--model-dir", str(complete), "--manifest", str(manifest),
            "--output", str(repo / "artifacts/qwen3_8b_local_verification.json"),
        ], timeout=1800)
        log(log_path, f"local_snapshot_verified source={complete} output={verify_local.stdout.strip()[-200:]}")
        run(["ssh", "my-codelab", f"mkdir -p {remote_model} {remote_root}/artifacts"], timeout=120)
        # Safetensors are already compressed/entropy-dense; disabling rsync's
        # CPU-heavy zlib keeps the resumable transfer limited by the link.
        run(
            [
                "rsync",
                "-a",
                "--partial",
                "--inplace",
                "--exclude=*.safetensors_*",
                "--exclude=*.incomplete",
                "--exclude=*.manual",
                "--exclude=.cache/",
                str(complete) + "/",
                f"my-codelab:{remote_model}/",
            ],
            timeout=7200,
        )
        run(["rsync", "-az", str(manifest), f"my-codelab:{remote_manifest}"], timeout=120)
        # The remote verifier is part of this repository.  Sync scripts before
        # invoking it; otherwise a fresh CodeLab project can fail with
        # "verify_snapshot.py: No such file" even though every model shard is
        # already present.
        run(["rsync", "-az", "scripts/", f"my-codelab:{remote_root}/scripts/"], timeout=300)
        remote_verify = run([
            "ssh", "my-codelab", "env", "PYTHONPATH=src",
            "/home/hadoop-aipnlp/.conda/envs/agenticqwen-py310/bin/python",
            "scripts/verify_snapshot.py", "--model-dir", remote_model, "--manifest", remote_manifest,
            "--output", remote_root + "/artifacts/qwen3_8b_remote_verification.json",
        ], timeout=1800)
        log(log_path, f"remote_snapshot_verified output={remote_verify.stdout.strip()[-200:]}")
        run(["rsync", "-az", "src/agentic_repro/", f"my-codelab:{remote_root}/src/agentic_repro/"], timeout=300)
        run(["rsync", "-az", "configs/", f"my-codelab:{remote_root}/configs/"], timeout=300)
        run(["rsync", "-az", "scripts/", f"my-codelab:{remote_root}/scripts/"], timeout=300)
        smoke = run([
            "ssh", "my-codelab", "env",
            f"PYTHONPATH={remote_root}/src",
            f"AGENTICQWEN_MODEL_PATH={remote_model}",
            "/home/hadoop-aipnlp/.conda/envs/agenticqwen-py310/bin/python",
            f"{remote_root}/scripts/model_load_smoke.py",
            "--model-path", remote_model,
            "--output", remote_root + "/artifacts/qwen3_8b_model_load_smoke.json",
        ], timeout=1800)
        log(log_path, f"model_load_smoke_passed output={smoke.stdout.strip()[-200:]}")
        launch = (
            f"cd {remote_root}; mkdir -p logs artifacts/agenticqwen_codelab_real_run1; "
            f"if [ ! -f logs/agenticqwen_codelab_real_run1.pid ] || ! kill -0 $(cat logs/agenticqwen_codelab_real_run1.pid) 2>/dev/null; then "
            f"nohup env PYTHONPATH=src AGENTICQWEN_MODEL_PATH={remote_model} CUDA_VISIBLE_DEVICES=0 "
            f"/home/hadoop-aipnlp/.conda/envs/agenticqwen-py310/bin/python -m agentic_repro.paper_grpo_train "
            f"--config configs/agenticqwen_codelab_real.json --output-root artifacts/agenticqwen_codelab_real_run1 --mode train "
            f"> logs/agenticqwen_codelab_real_run1.log 2>&1 < /dev/null & echo $! > logs/agenticqwen_codelab_real_run1.pid; fi"
        )
        launched = run(["ssh", "my-codelab", launch], timeout=180)
        log(log_path, f"training_launch {launched.stdout.strip()}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
