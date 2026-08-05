#!/usr/bin/env python3
"""Run and audit the real strong-model side of the AgenticQwen flywheel."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from agentic_repro.paper_flywheel import linear_seed_tree, write_flywheel_artifacts
from agentic_repro.paper_flywheel_env import execute_path
from agentic_repro.paper_grpo_train import _flywheel_from_config, _teacher_from_config


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["AGENTICQWEN_TEACHER_AUDIT_FILE"] = str(output / "teacher_api_audit.jsonl")
    events = output / "events.jsonl"

    def event(name: str, **fields: Any) -> None:
        row = {"event": name, "unix_time": time.time(), **fields}
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    started = time.perf_counter()
    try:
        teacher = _teacher_from_config(config)
        flywheel = _flywheel_from_config(config, teacher)
        tree = linear_seed_tree()
        event("seed_generation_started", teacher=teacher.model_name)
        seed = flywheel.seed_task(tree)
        seed_rollout = execute_path(seed, seed.normal_path)
        event(
            "seed_generation_completed",
            task_id=seed.task_id,
            reward=seed_rollout.reward,
            branch_hit=seed_rollout.intended_branch_hit,
            actions=list(seed.normal_actions),
        )
        event("round1_evolution_started", rollout_count=1)
        evolution = flywheel.evolve_one(
            tree=tree,
            policy_rollouts=(seed_rollout,),
            round_index=1,
        )
        manifest = write_flywheel_artifacts(output / "flywheel", (evolution,))
        result = {
            "status": "completed",
            "teacher": teacher.model_name,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "seed": {
                "task_id": seed.task_id,
                "actions": list(seed.normal_actions),
                "reward": seed_rollout.reward,
                "branch_hit": seed_rollout.intended_branch_hit,
            },
            "evolution": {
                "revision": evolution.output_tree_revision,
                "branches": [list(branch.actions) for branch in evolution.tree.branches()],
                "candidates": evolution.candidate_count,
                "retained": evolution.retained_count,
                "validations": [
                    {
                        "task_id": value.task_id,
                        "reward": value.reward,
                        "branch_hit": value.intended_branch_hit,
                        "retained": value.retained,
                        "reason": value.reason,
                    }
                    for value in evolution.validations
                ],
            },
            "manifest_sha256": manifest["manifest_sha256"],
        }
        write_json(output / "summary.json", result)
        event("round1_evolution_completed", retained=evolution.retained_count)
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "traceback": traceback.format_exc(),
        }
        write_json(output / "failure.json", failure)
        event("teacher_smoke_failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
