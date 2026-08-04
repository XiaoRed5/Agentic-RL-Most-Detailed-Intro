from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .environment import make_scenarios
from .flywheel import train_flywheel
from .local_qwen import run_qwen_inference


def _jsonable_metrics(metrics: dict) -> dict:
    clean = dict(metrics)
    clean["baseline"] = {
        "overall": metrics["baseline"]["overall"],
        "by_level": metrics["baseline"]["by_level"],
    }
    clean["final"] = {
        "overall": metrics["final"]["overall"],
        "by_level": metrics["final"]["by_level"],
    }
    return clean


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(config: dict, project_dir: Path, qwen_model_path: str | None = None) -> dict:
    artifacts = project_dir / config["paths"]["artifacts"]
    artifacts.mkdir(parents=True, exist_ok=True)

    seed = int(config["run"]["seed"])
    rounds = int(config["run"]["rounds"])
    tasks = [
        task
        for level in range(rounds)
        for task in make_scenarios(level, int(config["run"]["train_tasks_per_round"]), seed, "catalog")
    ]
    tasks_path = artifacts / "tasks.jsonl"
    tasks_path.write_text("\n".join(json.dumps(asdict(task), ensure_ascii=False) for task in tasks) + "\n", encoding="utf-8")

    policy, metrics = train_flywheel(config)
    metrics_path = artifacts / "metrics.json"
    metrics_path.write_text(json.dumps(_jsonable_metrics(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    policy.save(artifacts / "smoke_policy.json")

    trajectories = metrics["final"]["trajectories"]
    selected = {}
    for trajectory in trajectories:
        if trajectory.scenario not in selected:
            selected[trajectory.scenario] = trajectory
    (artifacts / "smoke_trajectories.jsonl").write_text(
        "\n".join(json.dumps(item.to_dict(), ensure_ascii=False) for item in selected.values()) + "\n",
        encoding="utf-8",
    )

    qwen_path = artifacts / "qwen3_8b_inference.json"
    if qwen_model_path:
        qwen_results = run_qwen_inference(Path(qwen_model_path), config)
    else:
        qwen_results = {
            "status": "skipped",
            "reason": "QWEN_MODEL_PATH was not supplied",
            "model_id": config["qwen_inference"]["model_id"],
            "network_fallback": False,
        }
    qwen_path.write_text(json.dumps(qwen_results, ensure_ascii=False, indent=2), encoding="utf-8")

    files = [tasks_path, metrics_path, artifacts / "smoke_policy.json", artifacts / "smoke_trajectories.jsonl", qwen_path]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "files": {path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size} for path in files},
    }
    (artifacts / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"metrics": _jsonable_metrics(metrics), "qwen": qwen_results, "manifest": manifest}

