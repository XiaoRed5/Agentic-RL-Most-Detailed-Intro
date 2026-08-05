#!/usr/bin/env python3
"""Generate and teacher-gate at most ten official-schema curriculum tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_official_agentic_data import validate_rows


GENERATION_PROMPT = """You are expanding an AgenticQwen tool-use behavior tree from a failed policy rollout.

Create exactly one harder, still solvable task by branch-to-task inversion:
1. Pick a concrete branch the policy failed to handle.
2. Back-translate that branch into environment state, user instruction, and agent policy/SOP.
3. Add a meaningful condition such as a decoy record, transient tool error, confirmation gate, or policy constraint.
4. Make the compliant normal_path the uniquely justified path. Preserve an adversarial hack_path when relevant.
5. Keep the original tool schemas inside the system message's <tools> block and policy inside <policy>.

Return one JSON object inside <task_json>...</task_json> with exactly these fields:
id, prompt, task_background, rubrics, test_policy, user_escape_strategy, tool_return_expected.
prompt must begin with system and user messages. tool_return_expected must contain normal_path and hack_path.

Failure labels:
{failure_labels}

Failed policy output:
{failed_output}

Source task:
{source_task}
"""


SOLVE_PROMPT = """Solve this proposed AgenticQwen task as the strong-model filter.
Read the policy, tools, world state, user instruction, normal_path, and hack_path. Simulate the COMPLETE
multi-turn episode, using each expected tool output to decide the next action. Do not stop after merely
planning the first tool call. Return JSON inside <solve_json>...</solve_json> with keys: solvable
(boolean), terminal (string), tool_calls (array of every tool call in execution order, each with name
and arguments), and explanation. terminal must be exactly ###STOP or ###TRANSFER_TO_HUMAN when solvable;
it must never be a boolean or the string "false". Follow only the compliant branch; never use hack_path.

Gate feedback from a previous solve attempt, if any:
{gate_feedback}

Candidate task:
{candidate}
"""


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def extract_json(text: str, tag: str) -> dict[str, Any]:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    candidates = [match.group(1)] if match else []
    candidates.extend(re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE))
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        stripped = candidate.strip()
        try:
            value = json.loads(stripped)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        for index, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError(f"teacher response does not contain a {tag} JSON object")


def call_teacher(
    *,
    client: httpx.Client,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    audit: list[dict[str, Any]],
    phase: str,
) -> str:
    request_hash = sha256_json({"model": model, "prompt": prompt, "max_tokens": max_tokens})
    last_error: Exception | None = None
    for attempt in range(1, 4):
        started = time.monotonic()
        try:
            response = client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"] or ""
            if not content:
                raise ValueError("empty teacher response")
            audit.append(
                {
                    "phase": phase,
                    "attempt": attempt,
                    "ok": True,
                    "status_code": response.status_code,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "request_sha256": request_hash,
                    "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "response_chars": len(content),
                }
            )
            return content
        except Exception as exc:  # pragma: no cover - live endpoint path
            last_error = exc
            audit.append(
                {
                    "phase": phase,
                    "attempt": attempt,
                    "ok": False,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "request_sha256": request_hash,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"teacher failed after retries: {last_error}")


def path_signature(task: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    expected = task["tool_return_expected"]
    if isinstance(expected, str):
        expected = json.loads(expected)
    return tuple(
        (
            str(call.get("tool_name", call.get("name", ""))),
            json.dumps(call.get("input", call.get("arguments", {})), sort_keys=True),
        )
        for call in expected.get("normal_path", [])
    )


def branch_hit_diagnostics(candidate: dict[str, Any], solution: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "solvable": bool(solution.get("solvable")),
        "terminal": str(solution.get("terminal", "")),
        "matched_expected_calls": 0,
    }
    if not bool(solution.get("solvable")):
        diagnostics["reason"] = "teacher_marked_unsolvable"
        return False, diagnostics
    terminal = str(solution.get("terminal", "")).upper()
    if "###STOP" not in terminal and "###TRANSFER_TO_HUMAN" not in terminal:
        diagnostics["reason"] = "missing_terminal_marker"
        return False, diagnostics
    expected = candidate["tool_return_expected"]
    if isinstance(expected, str):
        expected = json.loads(expected)
    calls = solution.get("tool_calls", [])
    diagnostics["expected_path"] = expected.get("normal_path", [])
    diagnostics["actual_tool_calls"] = calls
    position = 0
    for target in expected.get("normal_path", []):
        target_name = str(target.get("tool_name", target.get("name", "")))
        target_args = target.get("input", target.get("arguments", {}))
        found = False
        while position < len(calls):
            actual = calls[position]
            position += 1
            actual_name = str(actual.get("name", actual.get("tool_name", "")))
            actual_args = actual.get("arguments", actual.get("input", {}))
            if actual_name == target_name and all(actual_args.get(key) == value for key, value in target_args.items()):
                found = True
                break
        if not found:
            diagnostics["reason"] = "expected_call_not_found_in_order"
            diagnostics["first_unmatched_expected_call"] = target
            return False, diagnostics
        diagnostics["matched_expected_calls"] += 1
    hit = bool(expected.get("normal_path"))
    diagnostics["reason"] = "matched" if hit else "empty_normal_path"
    return hit, diagnostics


def branch_hit(candidate: dict[str, Any], solution: dict[str, Any]) -> bool:
    hit, _ = branch_hit_diagnostics(candidate, solution)
    return hit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-candidates", required=True, type=Path)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--max-new-tasks", type=int, default=10)
    parser.add_argument("--max-solve-attempts", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--rejected-output",
        type=Path,
        help="Persist rejected candidate/solution pairs for gate diagnostics (default: beside manifest)",
    )
    args = parser.parse_args()
    if not 1 <= args.max_new_tasks <= 10:
        raise ValueError("max_new_tasks must be between 1 and 10")
    if not 1 <= args.max_solve_attempts <= 3:
        raise ValueError("max_solve_attempts must be between 1 and 3")

    source = json.loads(args.failure_candidates.read_text(encoding="utf-8"))
    candidates = list(source.get("candidates", []))[: args.max_new_tasks]
    api_key = args.key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("teacher key file is empty")
    url = f"{args.api_base.rstrip('/')}/chat/completions"
    retained: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    with httpx.Client(timeout=180) as client:
        for item in candidates:
            source_task = item["source_task"]
            validation: dict[str, Any] = {"source_task_id": item["task_id"], "retained": False}
            try:
                generation = call_teacher(
                    client=client,
                    url=url,
                    api_key=api_key,
                    model=args.model,
                    prompt=GENERATION_PROMPT.format(
                        failure_labels=json.dumps(item["failure_labels"], ensure_ascii=False),
                        failed_output=str(item["failed_output"])[-12000:],
                        source_task=json.dumps(source_task, ensure_ascii=False),
                    ),
                    max_tokens=12000,
                    audit=audit,
                    phase="branch_to_task_inversion",
                )
                candidate = extract_json(generation, "task_json")
                candidate["id"] = f"curriculum-{item['task_id']}-{sha256_json(candidate)[:10]}"
                validate_rows([candidate], label="teacher_candidate", allow_repeated_ids=False)
                if path_signature(candidate) == path_signature(source_task):
                    raise ValueError("candidate normal_path does not add a new branch")
                solution: dict[str, Any] = {}
                hit = False
                hit_diagnostics: dict[str, Any] = {"reason": "not_attempted"}
                gate_feedback = "None; this is the first solve attempt."
                solve_attempts = 0
                for solve_attempts in range(1, args.max_solve_attempts + 1):
                    solve_response = call_teacher(
                        client=client,
                        url=url,
                        api_key=api_key,
                        model=args.model,
                        prompt=SOLVE_PROMPT.format(
                            candidate=json.dumps(candidate, ensure_ascii=False),
                            gate_feedback=gate_feedback,
                        ),
                        max_tokens=6000,
                        audit=audit,
                        phase=f"teacher_solve_{solve_attempts}",
                    )
                    solution = extract_json(solve_response, "solve_json")
                    hit, hit_diagnostics = branch_hit_diagnostics(candidate, solution)
                    if hit:
                        break
                    gate_feedback = json.dumps(hit_diagnostics, ensure_ascii=False)
                validation.update(
                    {
                        "candidate_id": candidate["id"],
                        "schema_valid": True,
                        "teacher_solvable": bool(solution.get("solvable")),
                        "intended_branch_hit": hit,
                        "branch_hit_diagnostics": hit_diagnostics,
                        "teacher_solve_attempts": solve_attempts,
                        "retained": hit,
                        "candidate_sha256": sha256_json(candidate),
                        "solution_sha256": sha256_json(solution),
                    }
                )
                if hit:
                    retained.append(candidate)
                else:
                    rejected.append(
                        {
                            "source_task_id": item["task_id"],
                            "candidate": candidate,
                            "teacher_solution": solution,
                            "branch_hit_diagnostics": hit_diagnostics,
                        }
                    )
            except Exception as exc:
                validation.update({"error_type": type(exc).__name__, "error": str(exc)[:1000]})
            validations.append(validation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(retained, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    rejected_output = (
        args.rejected_output.resolve()
        if args.rejected_output
        else args.manifest.resolve().with_name(f"{args.manifest.stem}.rejected.json")
    )
    rejected_output.parent.mkdir(parents=True, exist_ok=True)
    rejected_output.write_text(json.dumps(rejected, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if retained else "PARTIAL",
        "teacher_model": args.model,
        "teacher_substitution": "DeepSeek-v4-flash replaces Qwen3-235B for this bounded reproduction",
        "source_failure_manifest": str(args.failure_candidates.resolve()),
        "source_failure_sha256": hashlib.sha256(args.failure_candidates.read_bytes()).hexdigest(),
        "requested_candidates": len(candidates),
        "retained_tasks": len(retained),
        "max_new_tasks": args.max_new_tasks,
        "cap_respected": len(retained) <= 10,
        "teacher_solved_and_branch_hit_required": True,
        "validations": validations,
        "api_audit": audit,
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "rejected_output": str(rejected_output),
        "rejected_output_sha256": hashlib.sha256(rejected_output.read_bytes()).hexdigest(),
        "rejected_tasks": len(rejected),
        "secret_persisted": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "retained": len(retained), "requested": len(candidates)}))
    return 0 if retained else 2


if __name__ == "__main__":
    raise SystemExit(main())
