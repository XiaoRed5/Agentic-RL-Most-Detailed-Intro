#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_IMPORTS = (
    "torch",
    "transformers",
    "ray",
    "hydra",
    "tensordict",
    "sglang",
    "flash_attn",
    "flashinfer",
    "verl",
    "my_script.reward_function",
    "my_script.tools.mock_tool",
    "verl.experimental.agent_loop",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    imports = {}
    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
            imports[name] = {
                "ok": True,
                "version": str(getattr(module, "__version__", "present")),
                "file": str(getattr(module, "__file__", "")),
            }
        except Exception as exc:  # pragma: no cover - remote diagnostic path
            imports[name] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    torch = importlib.import_module("torch") if imports["torch"]["ok"] else None
    smoke = {"cuda_available": False}
    if torch is not None:
        smoke = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": str(torch.version.cuda),
            "gpu_count": int(torch.cuda.device_count()),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "bf16_supported": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
        }

    trainer = subprocess.run(
        [sys.executable, "-m", "verl.trainer.main_ppo", "--help"],
        cwd=args.upstream,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    run_script = args.upstream / "my_script/scripts/run_virtual_tool.sh"
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS"
        if all(item["ok"] for item in imports.values()) and smoke["cuda_available"] and trainer.returncode == 0
        else "FAIL",
        "python": sys.version,
        "platform": platform.platform(),
        "imports": imports,
        "cuda": smoke,
        "official_main_ppo_help": {
            "returncode": trainer.returncode,
            "stdout_tail": trainer.stdout[-2000:],
            "stderr_tail": trainer.stderr[-2000:],
        },
        "official_run_script": {
            "path": str(run_script),
            "exists": run_script.is_file(),
            "sha256": sha256(run_script) if run_script.is_file() else None,
        },
        "environment": {
            "python_executable": sys.executable,
            "pythonpath_configured": bool(os.getenv("PYTHONPATH")),
            "conda_prefix": os.getenv("CONDA_PREFIX"),
            "cc": os.getenv("CC"),
            "cxx": os.getenv("CXX"),
            "ld_library_path_configured": bool(os.getenv("LD_LIBRARY_PATH")),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
