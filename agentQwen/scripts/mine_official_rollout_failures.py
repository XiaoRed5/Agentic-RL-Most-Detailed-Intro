#!/usr/bin/env python3
"""Map low-reward verl rollouts back to official tasks for the next curriculum round."""

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


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_tool_calls(output: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for payload in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", output, re.DOTALL | re.IGNORECASE):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            calls.append(value)
    return calls


def expected_normal_path(task: dict[str, Any]) -> list[dict[str, Any]]:
    expected = task.get("tool_return_expected", {})
    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except json.JSONDecodeError:
            return []
    return list(expected.get("normal_path", [])) if isinstance(expected, dict) else []


def call_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    actual_name = str(actual.get("name", actual.get("tool_name", "")))
    expected_name = str(expected.get("tool_name", expected.get("name", "")))
    actual_args = actual.get("arguments", actual.get("input", {}))
    expected_args = expected.get("input", expected.get("arguments", {}))
    return (
        actual_name == expected_name
        and isinstance(actual_args, dict)
        and isinstance(expected_args, dict)
        and all(actual_args.get(key) == value for key, value in expected_args.items())
    )


def classify(output: str, task: dict[str, Any] | None = None) -> list[str]:
    lowered = output.lower()
    labels = []
    if "###stop" not in lowered and "###transfer_to_human" not in lowered:
        labels.append("missing_terminal_marker")
    if "<tool_response>" not in lowered:
        labels.append("missing_tool_observation")
    if "timeout" in lowered or "try again later" in lowered:
        labels.append("transient_tool_error_seen")
    if "<tool_call>" not in lowered:
        labels.append("no_tool_call")
    calls = extract_tool_calls(output)
    signatures = [
        (
            str(call.get("name", call.get("tool_name", ""))),
            json.dumps(call.get("arguments", call.get("input", {})), sort_keys=True),
        )
        for call in calls
    ]
    if len(set(signatures)) < len(signatures):
        labels.append("duplicate_tool_call")
    normal_path = expected_normal_path(task or {})
    if normal_path:
        position = 0
        completed = True
        for expected in normal_path:
            while position < len(calls) and not call_matches(calls[position], expected):
                position += 1
            if position >= len(calls):
                completed = False
                break
            position += 1
        if not completed:
            labels.append("normal_path_call_missing_or_args_mismatch")
            if "###transfer_to_human" in lowered:
                labels.append("premature_transfer_to_human")
    return labels or ["rubric_or_policy_failure"]


def mine(
    *,
    task_path: Path,
    rollout_dir: Path,
    max_candidates: int,
) -> dict[str, Any]:
    if not 1 <= max_candidates <= 10:
        raise ValueError("max_candidates must be between 1 and 10")
    tasks = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("task file must contain a JSON list")
    question_index: list[tuple[str, dict[str, Any]]] = []
    for task in tasks:
        prompt = task.get("prompt", [])
        if len(prompt) < 2:
            continue
        question = normalize(str(prompt[1].get("content", "")))
        if question:
            question_index.append((question, task))
    question_index.sort(key=lambda item: len(item[0]), reverse=True)

    rollout_files = sorted(rollout_dir.glob("*.jsonl"))
    rollout_rows: list[dict[str, Any]] = []
    for path in rollout_files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            value["_rollout_file"] = str(path.resolve())
            value["_rollout_line"] = line_number
            rollout_rows.append(value)

    failures: dict[str, dict[str, Any]] = {}
    unmatched = 0
    ordered_rollouts = sorted(rollout_rows, key=lambda row: float(row.get("score", 0.0)))
    has_zero_reward = any(float(row.get("score", 0.0)) == 0.0 for row in ordered_rollouts)
    for rollout in ordered_rollouts:
        score = float(rollout.get("score", 0.0))
        if has_zero_reward and score > 0.0:
            continue
        normalized_input = normalize(str(rollout.get("input", "")))
        matched = next((task for question, task in question_index if question in normalized_input), None)
        if matched is None:
            unmatched += 1
            continue
        task_id = str(matched["id"])
        if task_id in failures:
            continue
        output = str(rollout.get("output", ""))
        labels = classify(output, matched) if score == 0.0 else ["frontier_expansion_from_success"]
        failures[task_id] = {
            "task_id": task_id,
            "score": score,
            "failure_labels": labels,
            "failed_output": output,
            "rollout_provenance": {
                "file": rollout["_rollout_file"],
                "line": rollout["_rollout_line"],
            },
            "source_task": matched,
        }
        if len(failures) >= max_candidates:
            break

    candidates = list(failures.values())
    return {
        "schema_version": 1,
        "status": "PASS" if candidates else "PARTIAL",
        "policy": {
            "selection": (
                "unique official tasks with score == 0; if none fail, lowest-reward successful "
                "trajectories seed structure-driven frontier expansion"
            ),
            "max_candidates": max_candidates,
            "new_synthesis_cap": 10,
        },
        "source": {
            "task_path": str(task_path.resolve()),
            "task_sha256": sha256_file(task_path),
            "task_rows": len(tasks),
            "rollout_dir": str(rollout_dir.resolve()),
            "rollout_files": [str(path.resolve()) for path in rollout_files],
            "rollout_rows": len(rollout_rows),
            "unmatched_zero_reward_rollouts": unmatched,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--rollout-dir", required=True, type=Path)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = mine(
        task_path=args.tasks.resolve(),
        rollout_dir=args.rollout_dir.resolve(),
        max_candidates=args.max_candidates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "candidates": result["candidate_count"]}))
    return 0 if result["candidate_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
