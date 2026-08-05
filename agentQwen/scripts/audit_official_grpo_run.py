#!/usr/bin/env python3
"""Audit an official AgenticQwen/verl run from persisted artifacts only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: {exc}")
                continue
            if not isinstance(value, dict):
                errors.append(f"{path}:{line_number}: row is not an object")
                continue
            rows.append(value)
    return rows, errors


def extract_grad_norms(path: Path) -> list[float]:
    """Read persisted verl console metrics without trusting checkpoint presence alone."""
    if not path.is_file():
        return []
    pattern = re.compile(r"actor/grad_norm:([-+0-9.eE]+)")
    return [float(value) for value in pattern.findall(path.read_text(encoding="utf-8", errors="replace"))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--environment-manifest", required=True, type=Path)
    parser.add_argument("--expected-steps", required=True, type=int)
    parser.add_argument(
        "--train-log",
        type=Path,
        help="Persisted verl console log; when supplied, positive actor/grad_norm is required for a learning PASS",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    dataset = load_json(args.dataset_manifest.resolve())
    environment = load_json(args.environment_manifest.resolve())
    teacher_probe_path = run_dir / "teacher_api_probe.json"
    teacher_probe = load_json(teacher_probe_path) if teacher_probe_path.is_file() else {}
    latest_file = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    latest_step = int(latest_file.read_text().strip()) if latest_file.is_file() else None
    checkpoint_dir = (
        run_dir / "checkpoints" / f"global_step_{latest_step}" / "actor"
        if latest_step is not None
        else None
    )
    checkpoint_files = (
        sorted(path for path in checkpoint_dir.rglob("*") if path.is_file())
        if checkpoint_dir and checkpoint_dir.is_dir()
        else []
    )
    rollout_files = sorted((run_dir / "rollouts").glob("*.jsonl"))
    validation_files = sorted((run_dir / "validation").glob("*.jsonl"))
    rollout_rows, rollout_errors = load_jsonl(rollout_files)
    validation_rows, validation_errors = load_jsonl(validation_files)
    scores = [float(row["score"]) for row in rollout_rows if isinstance(row.get("score"), (int, float))]
    unique_scores = sorted(set(scores))
    train_log = args.train_log.resolve() if args.train_log else run_dir / "train.log"
    grad_norms = extract_grad_norms(train_log)
    positive_grad_norms = [value for value in grad_norms if value > 0]

    checks = {
        "environment_pass": environment.get("status") == "PASS",
        "teacher_api_reachable": teacher_probe.get("status") in {"PASS", "PARTIAL"},
        "official_data_present": dataset.get("sources", {}).get("official", {}).get("rows", 0) > 0,
        "official_train_rows_present": dataset.get("splits", {}).get("train", {}).get("official_rows", 0) > 0,
        "group_disjoint_holdout": bool(
            dataset.get("checks", {}).get("train_validation_base_id_disjoint")
        ),
        "synthetic_within_cap": bool(dataset.get("checks", {}).get("synthetic_within_cap")),
        "expected_step_checkpointed": latest_step is not None and latest_step >= args.expected_steps,
        "checkpoint_actor_nonempty": bool(checkpoint_files),
        "rollout_jsonl_valid": bool(rollout_rows) and not rollout_errors,
        "rollout_scores_present": len(scores) == len(rollout_rows) and bool(scores),
        "validation_jsonl_valid": bool(validation_rows) and not validation_errors,
    }
    execution_pass = all(checks.values())
    learning_signal = {
        "nonzero_reward_observed": any(score > 0 for score in scores),
        "zero_reward_observed": any(score == 0 for score in scores),
        "reward_variance_observed": len(unique_scores) > 1,
        "train_log_present": train_log.is_file(),
        "positive_grad_norm_observed": bool(positive_grad_norms),
    }
    learning_pass = learning_signal["reward_variance_observed"] and learning_signal[
        "positive_grad_norm_observed"
    ]
    payload = {
        "schema_version": 1,
        "status": "PASS" if execution_pass else "FAIL",
        "claim_scope": (
            "official AgenticQwen/verl execution is verified"
            if execution_pass
            else "official run artifacts are incomplete"
        ),
        "learning_signal_status": "PASS" if learning_pass else "PARTIAL",
        "expected_steps": args.expected_steps,
        "latest_checkpointed_step": latest_step,
        "checks": checks,
        "learning_signal": learning_signal,
        "dataset": {
            "official_pool_rows": dataset.get("sources", {}).get("official", {}).get("rows"),
            "official_train_rows": dataset.get("splits", {}).get("train", {}).get("official_rows"),
            "official_validation_rows": dataset.get("splits", {}).get("validation", {}).get("official_rows"),
            "synthetic_train_rows": dataset.get("splits", {}).get("train", {}).get("synthetic_rows"),
            "synthetic_cap": dataset.get("policy", {}).get("synthetic_cap"),
        },
        "artifacts": {
            "rollout_files": [str(path) for path in rollout_files],
            "rollout_rows": len(rollout_rows),
            "rollout_errors": rollout_errors,
            "validation_files": [str(path) for path in validation_files],
            "validation_rows": len(validation_rows),
            "validation_errors": validation_errors,
            "checkpoint_actor_dir": str(checkpoint_dir) if checkpoint_dir else None,
            "checkpoint_file_count": len(checkpoint_files),
            "checkpoint_manifest_sha256": hashlib.sha256(
                json.dumps(
                    [(str(path.relative_to(checkpoint_dir)), path.stat().st_size) for path in checkpoint_files],
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if checkpoint_files and checkpoint_dir
            else None,
            "environment_manifest_sha256": sha256_file(args.environment_manifest.resolve()),
            "dataset_manifest_sha256": sha256_file(args.dataset_manifest.resolve()),
            "teacher_probe_sha256": sha256_file(teacher_probe_path)
            if teacher_probe_path.is_file()
            else None,
            "train_log": str(train_log) if train_log.is_file() else None,
            "train_log_sha256": sha256_file(train_log) if train_log.is_file() else None,
        },
        "reward": {
            "count": len(scores),
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "mean": sum(scores) / len(scores) if scores else None,
            "unique": unique_scores,
        },
        "optimizer": {
            "grad_norm_count": len(grad_norms),
            "positive_grad_norm_count": len(positive_grad_norms),
            "max_grad_norm": max(grad_norms) if grad_norms else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "learning_signal": payload["learning_signal_status"], "output": str(args.output)}))
    return 0 if execution_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
