#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_model(root: Path) -> dict[str, Any]:
    marker = root / ".autodl_model_complete.json"
    result: dict[str, Any] = {
        "path": str(root),
        "exists": root.is_dir(),
        "marker": str(marker),
        "marker_valid": False,
        "missing_shards": [],
    }
    if not marker.is_file():
        return result
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    result["marker_valid"] = manifest.get("status") == "completed"
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weights = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        shards = sorted(set(weights.values()))
        result["shard_count"] = len(shards)
        result["missing_shards"] = [name for name in shards if not (root / name).is_file()]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoDL CUDA, dependency, disk, and model preflight")
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("--min-vram-gib", type=float, default=0.0)
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--basic-only", action="store_true", help="Check disk and nvidia-smi before installing dependencies")
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    disk = shutil.disk_usage(args.data_root)
    free_gib = disk.free / 1024**3
    add("data disk free", free_gib >= args.min_free_gib, {"free_gib": round(free_gib, 3), "minimum_gib": args.min_free_gib})

    if args.basic_only:
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
            )
            rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
            gpu_rows = []
            for row in rows:
                name, memory_mib = row.rsplit(",", 1)
                gpu_rows.append({"name": name.strip(), "vram_gib": round(float(memory_mib.strip()) / 1024, 3)})
            add("nvidia-smi GPU", bool(gpu_rows), gpu_rows)
            if gpu_rows:
                add("GPU VRAM", gpu_rows[0]["vram_gib"] >= args.min_vram_gib, {"gpu": gpu_rows[0], "minimum_gib": args.min_vram_gib})
        except Exception as exc:
            add("nvidia-smi GPU", False, repr(exc))
    else:
        try:
            import torch

            cuda = torch.cuda.is_available()
            add("CUDA available", cuda, {"torch": torch.__version__, "cuda_available": cuda})
            if cuda:
                props = torch.cuda.get_device_properties(0)
                vram = props.total_memory / 1024**3
                add("GPU VRAM", vram >= args.min_vram_gib, {"name": props.name, "vram_gib": round(vram, 3), "minimum_gib": args.min_vram_gib, "capability": list(torch.cuda.get_device_capability(0))})
        except Exception as exc:
            add("CUDA available", False, repr(exc))

    packages: dict[str, Any] = {}
    for name in [] if args.basic_only else args.require:
        try:
            importlib.import_module(name.replace("-", "_"))
            packages[name] = version(name)
            add(f"import {name}", True, packages[name])
        except Exception as exc:
            add(f"import {name}", False, repr(exc))

    model = check_model(args.model_dir.resolve()) if args.model_dir and not args.basic_only else None
    if model is not None:
        add("model snapshot marker", bool(model["marker_valid"]), model)
        add("model shards complete", not model["missing_shards"], model["missing_shards"])

    result = {
        "schema_version": 1,
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "python": sys.version,
        "checks": checks,
        "packages": packages,
        "model": model,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
