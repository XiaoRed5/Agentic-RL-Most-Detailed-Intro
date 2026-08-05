from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "output": completed.stdout,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Record CodeLab verification for the AgenticQwen paper path")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    artifact_root = args.artifact_root.resolve()
    flywheel_root = artifact_root / "flywheel"
    manifest_path = flywheel_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    hash_checks: list[dict[str, Any]] = []
    for round_item in manifest["rounds"]:
        for kind, key in (
            ("tree", "tree_sha256"),
            ("tasks", "tasks_sha256"),
            ("validation", "validation_sha256"),
        ):
            path = flywheel_root / round_item["paths"][kind]
            actual = sha256(path)
            hash_checks.append(
                {
                    "round": round_item["round_index"],
                    "kind": kind,
                    "expected": round_item[key],
                    "actual": actual,
                    "passed": actual == round_item[key],
                }
            )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    tests = run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=project_root,
        env=env,
    )
    test_count_match = re.search(r"Ran (\d+) tests?", tests["output"])
    tests["test_count"] = int(test_count_match.group(1)) if test_count_match else None
    tests["passed"] = tests["returncode"] == 0 and "OK" in tests["output"]

    gpu = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        cwd=project_root,
    )
    source_files = (
        "src/agentic_repro/paper_flywheel_schema.py",
        "src/agentic_repro/paper_flywheel.py",
        "src/agentic_repro/paper_flywheel_env.py",
        "src/agentic_repro/paper_grpo_train.py",
        "configs/agenticqwen_paper_micro.json",
        "tests/test_paper_flywheel.py",
    )
    result = {
        "schema_version": 1,
        "status": "PASS" if tests["passed"] and all(item["passed"] for item in hash_checks) else "FAIL",
        "scope": "CodeLab CPU contract/debug validation; no model weights downloaded and no GRPO optimizer run",
        "runtime": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "gpu_probe": gpu,
        },
        "tests": tests,
        "flywheel_hash_checks": hash_checks,
        "source_sha256": {
            name: sha256(project_root / name)
            for name in source_files
        },
        "artifact_manifest_sha256": sha256(manifest_path),
        "paper_scale_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "test_count": tests["test_count"],
        "hash_checks": len(hash_checks),
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
