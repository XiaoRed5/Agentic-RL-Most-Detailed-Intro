#!/usr/bin/env python3
"""Verify a Qwen snapshot using size and SHA-256, without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(model_dir: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = []
    for item in manifest["files"]:
        path = model_dir / item["path"]
        present = path.is_file()
        size_ok = present and path.stat().st_size == int(item["size"])
        digest = sha256(path) if size_ok else None
        hash_ok = digest == item["sha256"]
        checks.append(
            {
                "path": item["path"],
                "expected_size": item["size"],
                "actual_size": path.stat().st_size if present else None,
                "sha256": digest,
                "status": "PASS" if present and size_ok and hash_ok else "FAIL",
            }
        )
    return {
        "model_id": manifest.get("model_id"),
        "model_dir": str(model_dir.resolve()),
        "file_count": len(checks),
        "passed": sum(item["status"] == "PASS" for item in checks),
        "overall_status": "PASS" if checks and all(item["status"] == "PASS" for item in checks) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.model_dir, args.manifest)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
