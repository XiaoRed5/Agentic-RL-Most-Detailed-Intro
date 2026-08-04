from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ProcessEvent:
    """Observable, deterministic signals for one agent turn."""

    tool_name: str
    schema_valid: bool = True
    placeholder_argument: bool = False
    repeated_tool: bool = False
    tool_error: bool = False
    repeated_error: bool = False
    unsafe_action: bool = False
    skipped_required_read: bool = False
    recovered_from_error: bool = False
    data_dependency_satisfied: bool = False
    distinct_read: bool = False
    policy_confirmation: bool = False
    reasoning_tokens: int = 0
    task_complete: bool = False
    tool_count_so_far: int = 1


@dataclass(frozen=True)
class RuleContribution:
    rule: str
    value: float
    fired: bool


def prm_lite_score(event: ProcessEvent) -> tuple[float, list[RuleContribution]]:
    """Return an interpretable process score clipped to [-0.5, 0.5].

    The rules intentionally use only trajectory facts. They never call an LLM,
    so the same trace always receives the same score and can be audited for
    reward hacking.
    """

    rules = [
        RuleContribution("P1_placeholder_argument", -0.05, event.placeholder_argument),
        RuleContribution("P2_redundant_tool", -0.03, event.repeated_tool),
        RuleContribution("P3_tool_error", -0.04, event.tool_error),
        RuleContribution("P4_repeated_error", -0.06, event.repeated_error),
        RuleContribution("P5_malformed_schema", -0.08, not event.schema_valid),
        RuleContribution("P6_unsafe_action", -0.15, event.unsafe_action),
        RuleContribution("P7_skipped_required_read", -0.08, event.skipped_required_read),
        RuleContribution("P8_excessive_tools", -0.02, event.tool_count_so_far > 8),
        RuleContribution("B1_error_recovery", 0.05, event.recovered_from_error),
        RuleContribution("B2_data_dependency", 0.08, event.data_dependency_satisfied),
        RuleContribution("B3_distinct_read", 0.02, event.distinct_read),
        RuleContribution("B4_valid_schema", 0.02, event.schema_valid),
        RuleContribution("B5_policy_confirmation", 0.03, event.policy_confirmation),
        RuleContribution(
            "B6_bounded_reasoning",
            0.03,
            8 <= event.reasoning_tokens <= 512 and not event.tool_error,
        ),
        RuleContribution("B7_task_completion", 0.12, event.task_complete),
    ]
    raw = sum(item.value for item in rules if item.fired)
    return max(-0.5, min(0.5, raw)), rules


def combined_reward(outcome: float, process_score: float, *, weight: float = 0.3) -> float:
    if not 0.0 <= outcome <= 1.0:
        raise ValueError("outcome must be in [0, 1]")
    if not -0.5 <= process_score <= 0.5:
        raise ValueError("process_score must be in [-0.5, 0.5]")
    return outcome + weight * process_score


def group_relative_advantages(rewards: Sequence[float], *, eps: float = 1e-6) -> list[float]:
    if not rewards:
        raise ValueError("rewards cannot be empty")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    if variance <= eps**2:
        return [0.0 for _ in rewards]
    std = math.sqrt(variance)
    return [(reward - mean) / (std + eps) for reward in rewards]


def length_aware_credit(
    advantage: float,
    token_counts: Sequence[int],
    *,
    mode: str,
    turn_discount_alpha: float = 1.05,
) -> list[list[float]]:
    """Expand a trajectory advantage into per-token credit.

    ``linear`` implements A/L, ``sqrt_length`` implements LATA A/sqrt(L), and
    ``turn_discount`` applies a normalized early-turn multiplier. The returned
    list preserves turn boundaries for direct alignment with response masks.
    """

    if not token_counts or any(length <= 0 for length in token_counts):
        raise ValueError("token_counts must contain positive integers")
    if mode not in {"linear", "sqrt_length", "turn_discount"}:
        raise ValueError(f"unknown credit mode: {mode}")

    if mode == "linear":
        return [[advantage / length] * length for length in token_counts]
    if mode == "sqrt_length":
        return [[advantage / math.sqrt(length)] * length for length in token_counts]

    raw_weights = [
        turn_discount_alpha ** (len(token_counts) - 1 - index)
        for index in range(len(token_counts))
    ]
    normalizer = sum(raw_weights) / len(raw_weights)
    turn_weights = [value / normalizer for value in raw_weights]
    return [
        [advantage * turn_weights[index] / length] * length
        for index, length in enumerate(token_counts)
    ]


def _variance(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def offline_reward_diagnostic(
    training_log: Path,
    task_manifest: Path,
    *,
    process_weight: float = 0.3,
) -> dict:
    """Replay logged action groups under outcome and PRM-Lite rewards.

    This is a counterfactual reward diagnostic, not a trained-model result.
    It answers whether the deterministic process rules would have introduced
    within-group variance on the already observed rollout actions.
    """

    manifest = json.loads(task_manifest.read_text(encoding="utf-8"))
    task_by_id = {str(item["task_id"]): item for item in manifest["tasks"]}
    groups: list[dict] = []
    with training_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task = task_by_id[str(row["task_id"])]
            candidates = task["candidates"]
            correct_index = int(task["correct_index"])
            hack_path = set(task.get("hack_path", []))
            outcome_rewards: list[float] = []
            shaped_rewards: list[float] = []
            process_rows: list[dict] = []
            for action_index in row["rollout_actions"]:
                action_name = candidates[action_index]
                outcome = 1.0 if action_index == correct_index else 0.0
                event = ProcessEvent(
                    tool_name=action_name,
                    schema_valid=True,
                    unsafe_action=action_name in hack_path,
                    task_complete=outcome == 1.0,
                    reasoning_tokens=0,
                )
                process_score, contributions = prm_lite_score(event)
                outcome_rewards.append(outcome)
                shaped_rewards.append(
                    combined_reward(outcome, process_score, weight=process_weight)
                )
                process_rows.append(
                    {
                        "action": action_name,
                        "outcome": outcome,
                        "process_score": process_score,
                        "fired_rules": [item.rule for item in contributions if item.fired],
                    }
                )
            groups.append(
                {
                    "task_id": row["task_id"],
                    "outcome_variance": _variance(outcome_rewards),
                    "shaped_variance": _variance(shaped_rewards),
                    "outcome_rewards": outcome_rewards,
                    "shaped_rewards": shaped_rewards,
                    "process_rows": process_rows,
                }
            )

    outcome_active = sum(group["outcome_variance"] > 0 for group in groups)
    shaped_active = sum(group["shaped_variance"] > 0 for group in groups)
    return {
        "status": "OFFLINE_COUNTERFACTUAL_DIAGNOSTIC",
        "paper_comparable": False,
        "group_count": len(groups),
        "process_reward_weight": process_weight,
        "outcome_nonzero_variance_groups": outcome_active,
        "shaped_nonzero_variance_groups": shaped_active,
        "newly_activated_groups": shaped_active - outcome_active,
        "outcome_zero_variance_rate": 1.0 - outcome_active / len(groups),
        "shaped_zero_variance_rate": 1.0 - shaped_active / len(groups),
        "interpretation": (
            "Re-scores previously sampled actions only. It does not update the model "
            "and must not be reported as an ablation accuracy result."
        ),
        "rules": [asdict(item) for item in prm_lite_score(ProcessEvent("noop"))[1]],
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic GRPO ablation utilities")
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--process-weight", type=float, default=0.3)
    args = parser.parse_args()
    result = offline_reward_diagnostic(
        args.training_log.resolve(),
        args.task_manifest.resolve(),
        process_weight=args.process_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "groups"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
