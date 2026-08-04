from __future__ import annotations

import json
import hashlib
import os
import time
from pathlib import Path

from modelscope import snapshot_download


target = Path(os.getenv("MODEL_DIR", "/root/autodl-tmp/models/Qwen3-8B"))
started = time.perf_counter()
resolved = snapshot_download("Qwen/Qwen3-8B", local_dir=str(target))
elapsed = time.perf_counter() - started


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


files = [path for path in sorted(target.rglob("*")) if path.is_file()]
manifest = {
    "schema_version": 1,
    "status": "completed",
    "source_model_id": "Qwen/Qwen3-8B",
    "source_hub": "ModelScope",
    "resolved": str(Path(resolved).resolve()),
    "seconds": round(elapsed, 3),
    "bytes": sum(path.stat().st_size for path in files),
    "files": [
        {
            "path": str(path.relative_to(target)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ],
}
marker = target / ".modelscope_complete.json"
temporary = marker.with_suffix(".tmp")
temporary.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(marker)
print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
