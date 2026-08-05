from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mine_official_rollout_failures.py"
SPEC = importlib.util.spec_from_file_location("mine_official_rollout_failures", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def task(task_id: str, question: str) -> dict:
    return {
        "id": task_id,
        "prompt": [
            {"role": "system", "content": "<policy>p</policy><tools>[]</tools>"},
            {"role": "user", "content": question},
        ],
        "tool_return_expected": {
            "normal_path": [{"tool_name": "query", "input": {"id": task_id}}],
            "hack_path": [],
        },
    }


def test_mines_unique_zero_reward_tasks(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([task("a", "Book Beijing"), task("b", "Book Shanghai")]))
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    rows = [
        {"input": "system Book Beijing", "output": "<tool_call>x</tool_call>", "score": 0.0},
        {"input": "system Book Beijing", "output": "second", "score": 0.0},
        {"input": "system Book Shanghai", "output": "###STOP <tool_response>x</tool_response>", "score": 1.0},
    ]
    (rollout_dir / "1.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    result = MODULE.mine(task_path=tasks, rollout_dir=rollout_dir, max_candidates=10)

    assert result["status"] == "PASS"
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["task_id"] == "a"
    assert "missing_terminal_marker" in result["candidates"][0]["failure_labels"]


def test_uses_successful_frontier_when_no_failures_exist(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([task("a", "Book Beijing")]))
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    (rollout_dir / "1.jsonl").write_text(
        json.dumps({"input": "Book Beijing", "output": "###STOP", "score": 1.0}) + "\n"
    )

    result = MODULE.mine(task_path=tasks, rollout_dir=rollout_dir, max_candidates=10)

    assert result["candidate_count"] == 1
    assert result["candidates"][0]["failure_labels"] == ["frontier_expansion_from_success"]


def test_classifies_argument_mismatch_transfer_and_duplicate_call() -> None:
    source = task("PRT-1234", "status")
    output = (
        '<tool_call>{"name":"query","arguments":{"id":"wrong"}}</tool_call>'
        '<tool_call>{"name":"query","arguments":{"id":"wrong"}}</tool_call>'
        "###TRANSFER_TO_HUMAN"
    )

    labels = MODULE.classify(output, source)

    assert "duplicate_tool_call" in labels
    assert "normal_path_call_missing_or_args_mismatch" in labels
    assert "premature_transfer_to_human" in labels
