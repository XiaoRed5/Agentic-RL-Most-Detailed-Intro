#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


SECRET_PATTERNS = {
    "generic_api_key": re.compile(rb"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_./+\-=]{16,}"),
    "openai_like_key": re.compile(rb"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "huggingface_token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
}


def secret_findings(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in paths:
        if path.stat().st_size > 32 * 1024 * 1024:
            continue
        data = path.read_bytes()
        if b"\x00" in data[:8192]:
            continue
        for name, pattern in SECRET_PATTERNS.items():
            matches = list(pattern.finditer(data))
            if matches:
                findings.append({"path": str(path.relative_to(root)), "pattern": name, "count": len(matches)})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and package an AutoDL run directory")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.run_root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Run root is missing: {root}")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "artifact_inventory.json"
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink() and path != manifest_path)
    findings = secret_findings(files, root)
    if findings:
        raise RuntimeError("Secret-like material found; refusing to package: " + json.dumps(findings, ensure_ascii=False))
    inventory: list[dict[str, Any]] = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "status": "ready",
        "run_root": str(root),
        "file_count": len(inventory),
        "total_bytes": sum(row["bytes"] for row in inventory),
        "files": inventory,
        "secret_scan": {"status": "PASS", "patterns": sorted(SECRET_PATTERNS)},
        "inventory_self_excluded": True,
        "archive_includes_inventory": True,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with tarfile.open(output, "w:gz") as archive:
        archive.add(root, arcname=root.name, recursive=True)
    result = {
        "status": "packaged",
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256(output),
        "inventory": str(manifest_path),
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
