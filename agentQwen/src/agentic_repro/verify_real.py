from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .real_grpo import DecisionTask, evaluate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _restore_task(data: dict[str, Any]) -> DecisionTask:
    return DecisionTask(
        task_id=data["task_id"],
        policy_excerpt=data["policy_excerpt"],
        user_request=data["user_request"],
        rubric=data["rubric"],
        candidates=tuple(data["candidates"]),
        correct_index=data["correct_index"],
        normal_path=tuple(data["normal_path"]),
        hack_path=tuple(data["hack_path"]),
    )


def verify(config_path: Path) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import load_adapters

    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_dir = config_path.parent.parent
    artifacts_dir = project_dir / "artifacts" / "real_qwen3_8b"
    adapter_dir = artifacts_dir / "adapter"
    summary = json.loads((artifacts_dir / "summary.json").read_text(encoding="utf-8"))
    expected_eval = json.loads((artifacts_dir / "final_eval.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifacts_dir / "task_manifest.json").read_text(encoding="utf-8"))
    tasks = [_restore_task(task) for task in manifest["tasks"]]
    train_count = config["train_task_count"]
    train_tasks = tasks[:train_count]
    unseen_tasks = tasks[train_count:]

    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "status": "PASS" if condition else "FAIL", "detail": detail})

    model_file = Path(config["model_path"]) / "model.safetensors"
    final_adapter = adapter_dir / "adapters.safetensors"
    initial_adapter = adapter_dir / "adapters_initial.safetensors"
    check(
        "official model weight hash",
        _sha256(model_file) == summary["model"]["weights_sha256"],
        summary["model"]["weights_sha256"],
    )
    check(
        "official dataset hash",
        _sha256(Path(config["dataset_path"])) == summary["dataset"]["parquet_sha256"],
        summary["dataset"]["parquet_sha256"],
    )
    initial_hash = _sha256(initial_adapter)
    final_hash = _sha256(final_adapter)
    check(
        "adapter changed on disk",
        initial_hash != final_hash and final_hash == summary["training"]["adapter_final_sha256"],
        f"initial={initial_hash}; final={final_hash}",
    )
    check(
        "non-zero learned LoRA delta",
        summary["training"]["lora_b_norm_after"] > 0.0,
        f"lora_b_norm_after={summary['training']['lora_b_norm_after']}",
    )
    check(
        "real group-relative updates",
        summary["training"]["updated_groups"] > 0 and summary["training"]["optimizer_steps"] > 0,
        f"updated_groups={summary['training']['updated_groups']}; steps={summary['training']['optimizer_steps']}",
    )

    started = time.perf_counter()
    model, tokenizer = load(config["model_path"])
    load_adapters(model, str(adapter_dir))
    replay = {
        "train": evaluate(
            model,
            tokenizer,
            train_tasks,
            split="train",
            variant="train",
            max_prompt_tokens=config["max_prompt_tokens"],
        ),
        "holdout": evaluate(
            model,
            tokenizer,
            train_tasks,
            split="same_tasks_prompt_holdout",
            variant="holdout",
            max_prompt_tokens=config["max_prompt_tokens"],
        ),
        "unseen": evaluate(
            model,
            tokenizer,
            unseen_tasks,
            split="unseen_tasks",
            variant="unseen",
            max_prompt_tokens=config["max_prompt_tokens"],
        ),
    }
    replay_seconds = time.perf_counter() - started
    for split in ("train", "holdout", "unseen"):
        accuracy_equal = replay[split]["accuracy"] == expected_eval[split]["accuracy"]
        probability_delta = abs(
            replay[split]["mean_correct_action_probability"]
            - expected_eval[split]["mean_correct_action_probability"]
        )
        check(
            f"fresh-process checkpoint replay: {split}",
            accuracy_equal and probability_delta < 1e-7,
            (
                f"accuracy={replay[split]['accuracy']}; "
                f"mean_correct_probability={replay[split]['mean_correct_action_probability']}; "
                f"delta={probability_delta}"
            ),
        )
    check(
        "scope label rejects paper-scale claim",
        summary["paper_scale_claimed"] is False,
        summary["scope"],
    )

    result = {
        "overall_status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "summary": {
            "pass": sum(item["status"] == "PASS" for item in checks),
            "fail": sum(item["status"] == "FAIL" for item in checks),
        },
        "fresh_process_replay_seconds": round(replay_seconds, 4),
        "peak_memory_gib": round(mx.get_peak_memory() / (1024**3), 3),
        "checks": checks,
        "replay_metrics": {
            split: {
                "accuracy": replay[split]["accuracy"],
                "mean_correct_action_probability": replay[split]["mean_correct_action_probability"],
            }
            for split in ("train", "holdout", "unseen")
        },
    }
    (artifacts_dir / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently reload and verify the real Qwen3-8B checkpoint")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.config.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
