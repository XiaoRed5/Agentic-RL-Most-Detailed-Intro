from __future__ import annotations

import json
import time
from pathlib import Path

from modelscope import snapshot_download


target = Path("/root/autodl-tmp/modelscope-probe/Qwen3-8B")
started = time.perf_counter()
resolved = snapshot_download(
    "Qwen/Qwen3-8B",
    local_dir=str(target),
    allow_patterns=["config.json", "generation_config.json", "tokenizer_config.json"],
)
elapsed = time.perf_counter() - started
files = sorted(
    {
        "name": str(path.relative_to(target)),
        "bytes": path.stat().st_size,
    }
    for path in target.rglob("*")
    if path.is_file()
)
print(json.dumps({"resolved": resolved, "seconds": elapsed, "files": files}, indent=2))
