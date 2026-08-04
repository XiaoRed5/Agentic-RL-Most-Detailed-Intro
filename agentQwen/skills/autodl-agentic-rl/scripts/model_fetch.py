#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path, marker: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == marker or ".cache" in path.parts:
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def download(model_id: str, target: Path, source: str, revision: str | None) -> str:
    if source == "modelscope":
        from modelscope import snapshot_download

        kwargs: dict[str, Any] = {"local_dir": str(target)}
        if revision:
            kwargs["revision"] = revision
        return str(snapshot_download(model_id, **kwargs))

    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": model_id,
        "local_dir": str(target),
    }
    if revision:
        kwargs["revision"] = revision
    return str(snapshot_download(**kwargs))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and audit a model snapshot")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--source", choices=("modelscope", "huggingface"), default="modelscope")
    parser.add_argument("--revision")
    parser.add_argument("--force-audit", action="store_true")
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".autodl_model_complete.json"
    if marker.is_file() and not args.force_audit:
        cached = json.loads(marker.read_text(encoding="utf-8"))
        if (
            cached.get("status") == "completed"
            and cached.get("source_model_id") == args.model_id
            and (args.revision is None or cached.get("revision") == args.revision)
        ):
            invalid: list[dict[str, str]] = []
            rows = cached.get("files", [])
            for row in rows:
                path = target / row["path"]
                if not path.is_file():
                    invalid.append({"path": row["path"], "reason": "missing"})
                elif path.stat().st_size != int(row.get("bytes", -1)):
                    invalid.append({"path": row["path"], "reason": "size_mismatch"})
                elif row.get("sha256") and sha256(path) != row["sha256"]:
                    invalid.append({"path": row["path"], "reason": "sha256_mismatch"})
            if rows and not invalid:
                print(json.dumps({**cached, "cache_hit": True, "marker": str(marker)}, ensure_ascii=False, indent=2))
                return 0
            if invalid:
                print(json.dumps({"status": "cache_invalid", "invalid": invalid}, ensure_ascii=False))
                for item in invalid:
                    damaged = target / item["path"]
                    if damaged.is_file():
                        damaged.unlink()

    started = time.perf_counter()
    resolved = download(args.model_id, target, args.source, args.revision)
    rows = inventory(target, marker)
    if not rows:
        raise RuntimeError(f"Model snapshot is empty: {target}")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "source_model_id": args.model_id,
        "source_hub": args.source,
        "revision": args.revision,
        "resolved": str(Path(resolved).resolve()),
        "seconds": round(time.perf_counter() - started, 3),
        "bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
