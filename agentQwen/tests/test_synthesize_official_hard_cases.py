from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "synthesize_official_hard_cases.py"
SPEC = importlib.util.spec_from_file_location("synthesize_official_hard_cases", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def candidate() -> dict:
    return {
        "tool_return_expected": {
            "normal_path": [
                {"tool_name": "query", "input": {"id": "x"}},
                {"tool_name": "write", "input": {"id": "x", "confirmed": True}},
            ],
            "hack_path": [],
        }
    }


def test_extract_tagged_json() -> None:
    assert MODULE.extract_json("prefix <task_json>{\"x\": 1}</task_json>", "task_json") == {"x": 1}


def test_solve_prompt_requires_complete_episode_and_terminal() -> None:
    rendered = MODULE.SOLVE_PROMPT.format(candidate="{}", gate_feedback="missing second call")
    assert "COMPLETE" in rendered
    assert "###STOP" in rendered
    assert "missing second call" in rendered


def test_branch_hit_requires_order_arguments_and_terminal() -> None:
    solution = {
        "solvable": True,
        "terminal": "###STOP",
        "tool_calls": [
            {"name": "query", "arguments": {"id": "x"}},
            {"name": "write", "arguments": {"id": "x", "confirmed": True}},
        ],
    }
    assert MODULE.branch_hit(candidate(), solution) is True
    hit, diagnostics = MODULE.branch_hit_diagnostics(candidate(), solution)
    assert hit is True
    assert diagnostics["matched_expected_calls"] == 2
    assert diagnostics["reason"] == "matched"
    solution["tool_calls"].reverse()
    assert MODULE.branch_hit(candidate(), solution) is False
    hit, diagnostics = MODULE.branch_hit_diagnostics(candidate(), solution)
    assert hit is False
    assert diagnostics["reason"] == "expected_call_not_found_in_order"
    assert diagnostics["first_unmatched_expected_call"]["tool_name"] == "write"
