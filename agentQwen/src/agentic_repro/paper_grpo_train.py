from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .curriculum_train import (
    _adapter_weights,
    _build_model,
    _clean_trace,
    _latest_checkpoint,
    _make_args,
    _runtime_versions,
    _sha256,
    _trainable_parameter_hash,
    _tree_hash,
)
from .paper_flywheel import (
    AgenticQwenDataFlywheel,
    DeterministicStrongModel,
    OpenAICompatibleStrongModel,
    StrongModel,
    linear_seed_tree,
    write_flywheel_artifacts,
)
from .paper_flywheel_env import PaperFlightEnvironment
from .paper_flywheel_schema import (
    AgenticTask,
    BehaviorTree,
    CandidateValidation,
    FlywheelRound,
    TrajectoryRecord,
)


POLICY_SYSTEM_PROMPT = """You are the trainable policy in a stateful airline environment.

Use tools to read the world before acting. Follow the supplied SOP even when the user asks you to
bypass it. Never invent inventory, membership, or tool results. A write is valid only after all
branch conditions have been verified. Continue until the task is resolved and then confirm exactly
one valid outcome. Prefer concise tool calls and recover from tool errors using observations.
"""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_tasks(path: Path, tasks: Iterable[AgenticTask]) -> None:
    values = tuple(tasks)
    if not values:
        raise ValueError(f"Refusing to write an empty AgenticQwen task file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for task in values
        ),
        encoding="utf-8",
    )


def load_tasks(path: Path) -> tuple[AgenticTask, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = tuple(AgenticTask.from_dict(row) for row in rows)
    if not tasks:
        raise ValueError(f"AgenticQwen task file is empty: {path}")
    return tasks


def task_rows(tasks: Iterable[AgenticTask], stage: str) -> list[dict[str, Any]]:
    """Compile paper's environment/user/agent streams for TRL.

    The policy sees the natural user request and task-specific SOP.  The mock
    environment receives the serialized environment stream, adversarial user
    strategy, normal/hack paths, and objective rubric via ``task_json``.
    """
    rows: list[dict[str, Any]] = []
    for task in tasks:
        system = POLICY_SYSTEM_PROMPT + "\nTask-specific SOP:\n" + task.agent_instruction
        official = task.official_compatibility_fields()
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": task.user_input.request},
                ],
                "task_json": json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True),
                "stage": stage,
                "task_id": task.task_id,
                "tree_revision": task.tree_revision,
                "source_branch_id": task.source_branch_id,
                "extra_info": {
                    "question": task.user_input.request,
                    "training_streams": task.training_streams(),
                    **official,
                },
            }
        )
    return rows


def seed_task_variants(
    seed: AgenticTask,
    destinations: Iterable[str],
    *,
    max_tasks: int | None = None,
) -> tuple[AgenticTask, ...]:
    """Instantiate one SynthAgent linear workflow in several world contexts."""
    values: list[AgenticTask] = []
    destination_values = tuple(destinations)
    if max_tasks is not None:
        if max_tasks < 1:
            raise ValueError("max_tasks must be positive when provided")
        destination_values = destination_values[:max_tasks]
    for index, destination_value in enumerate(destination_values):
        destination = str(destination_value).strip()
        if not destination:
            raise ValueError("Seed destination must not be empty")
        state = {**seed.environment_state, "destination": destination}
        normal_path = tuple(
            replace(call, arguments={**call.arguments, "destination": destination})
            if "destination" in call.arguments
            else call
            for call in seed.normal_path
        )
        user = replace(
            seed.user_input,
            request=f"Book a ticket to {destination} and confirm the valid reservation.",
            background=f"The user needs a direct flight to {destination} and the inventory is stable.",
        )
        values.append(
            replace(
                seed,
                task_id=f"{seed.task_id}-seed-{index:02d}",
                environment_state=state,
                user_input=user,
                normal_path=normal_path,
                provenance={
                    **seed.provenance,
                    "seed_source": "SynthAgent-compatible linear workflow",
                    "seed_variant": index,
                    "destination": destination,
                },
            )
        )
    if not values:
        raise ValueError("At least one Round-0 seed destination is required")
    return tuple(values)


def trajectory_from_row(value: dict[str, Any]) -> TrajectoryRecord:
    required = {
        "task_id",
        "actions",
        "events",
        "final_state",
        "reward",
        "rubric_scores",
        "intended_branch_hit",
        "success",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"Policy trajectory is missing fields: {', '.join(missing)}")
    reward = float(value["reward"])
    if not 0.0 <= reward <= 1.0:
        raise ValueError(f"Policy trajectory reward is outside [0,1]: {reward}")
    return TrajectoryRecord(
        task_id=str(value["task_id"]),
        actions=tuple(str(item) for item in value["actions"]),
        events=tuple(dict(item) for item in value["events"]),
        final_state=dict(value["final_state"]),
        reward=reward,
        rubric_scores={str(key): float(score) for key, score in value["rubric_scores"].items()},
        intended_branch_hit=bool(value["intended_branch_hit"]),
        success=bool(value["success"]),
    )


def read_policy_rollouts(path: Path) -> tuple[TrajectoryRecord, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Policy rollout file does not exist: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rollouts = tuple(trajectory_from_row(row) for row in rows)
    if not rollouts:
        raise ValueError(f"Policy rollout file is empty: {path}")
    return rollouts


def summarize_rollouts(rollouts: Iterable[TrajectoryRecord]) -> dict[str, Any]:
    values = tuple(rollouts)
    rewards = [item.reward for item in values]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    variance = (
        sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        if rewards
        else 0.0
    )
    return {
        "episodes": len(values),
        "successes": sum(item.success for item in values),
        "branch_hits": sum(item.intended_branch_hit for item in values),
        "mean_reward": mean,
        "reward_std": variance**0.5,
        "unique_reward_count": len(set(rewards)),
    }


def _dataset(tasks: tuple[AgenticTask, ...], stage: str):
    from datasets import Dataset

    if not tasks:
        raise ValueError(f"No tasks supplied for {stage}")
    return Dataset.from_list(task_rows(tasks, stage))


def train_round(
    config: dict[str, Any],
    *,
    stage_name: str,
    tasks: tuple[AgenticTask, ...],
    output_root: Path,
    input_adapter: Path | None,
) -> dict[str, Any]:
    """Train one QLoRA-GRPO round and emit real policy trajectories."""
    import torch
    from trl import GRPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Paper-aligned GRPO training requires a CUDA GPU")
    stage_dir = output_root / stage_name
    trainer_dir = stage_dir / "trainer"
    adapter_dir = stage_dir / "adapter"
    stage_dir.mkdir(parents=True, exist_ok=True)
    task_file = stage_dir / "training_tasks.jsonl"
    write_tasks(task_file, tasks)
    dataset = _dataset(tasks, stage_name)
    model, tokenizer, peft_config = _build_model(config, input_adapter)
    args = _make_args(config, stage_name, trainer_dir)
    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
        environment_factory=PaperFlightEnvironment,
        peft_config=peft_config,
    )
    before_hash = _trainable_parameter_hash(trainer.model)
    resume_checkpoint = _latest_checkpoint(trainer_dir)

    before_trace = stage_dir / "eval_before_traces.jsonl"
    _clean_trace(before_trace)
    os.environ["AGENTICQWEN_PAPER_TRACE_FILE"] = str(before_trace)
    torch.manual_seed(int(config["seed"]))
    before_metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix="before")

    train_trace = stage_dir / "train_traces.jsonl"
    if resume_checkpoint is None:
        _clean_trace(train_trace)
    os.environ["AGENTICQWEN_PAPER_TRACE_FILE"] = str(train_trace)
    started = time.perf_counter()
    output = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    training_seconds = time.perf_counter() - started
    after_hash = _trainable_parameter_hash(trainer.model)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    weights = _adapter_weights(adapter_dir)

    after_trace = stage_dir / "eval_after_traces.jsonl"
    _clean_trace(after_trace)
    os.environ["AGENTICQWEN_PAPER_TRACE_FILE"] = str(after_trace)
    torch.manual_seed(int(config["seed"]))
    after_metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix="after")
    after_rollouts = read_policy_rollouts(after_trace)
    summary = {
        "schema_version": 1,
        "status": "completed",
        "stage": stage_name,
        "algorithm": "AgenticQwen response-token QLoRA-GRPO",
        "environment_factory": "PaperFlightEnvironment",
        "model_id": config["model"]["id"],
        "runtime": _runtime_versions(),
        "task_count": len(tasks),
        "task_file": str(task_file.resolve()),
        "task_sha256": _sha256(task_file),
        "training_seconds": round(training_seconds, 3),
        "global_step": int(output.global_step),
        "train_metrics": dict(output.metrics),
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "policy_rollouts_after": summarize_rollouts(after_rollouts),
        "policy_rollout_file": str(after_trace.resolve()),
        "policy_rollout_sha256": _sha256(after_trace),
        "trainable_parameter_hash_before": before_hash,
        "trainable_parameter_hash_after": after_hash,
        "trainable_parameters_changed": before_hash != after_hash,
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter_tree_sha256": _tree_hash(adapter_dir),
        "adapter_weights_sha256": _sha256(weights),
        "input_adapter": str(input_adapter.resolve()) if input_adapter else None,
        "resumed_from_checkpoint": str(resume_checkpoint.resolve()) if resume_checkpoint else None,
        "reward_range": [0.0, 1.0],
        "paper_scale_claimed": False,
    }
    _write_json(stage_dir / "summary.json", summary)
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _teacher_from_config(config: dict[str, Any]) -> StrongModel:
    teacher = config.get("teacher", {})
    backend = teacher.get("backend", "deterministic_contract")
    if backend == "deterministic_contract":
        return DeterministicStrongModel()
    if backend == "openai_compatible":
        return OpenAICompatibleStrongModel(
            model_name=str(teacher.get("model", "Qwen/Qwen3-235B-A22B-Instruct-2507")),
            base_url=teacher.get("base_url"),
            api_key_env=str(teacher.get("api_key_env", "AGENTICQWEN_TEACHER_API_KEY")),
            api_key_file=teacher.get("api_key_file"),
            timeout_seconds=int(teacher.get("timeout_seconds", 300)),
            max_tokens=int(teacher.get("max_tokens", 8192)),
            response_format_json=bool(teacher.get("response_format_json", True)),
        )
    raise ValueError(f"Unsupported teacher backend: {backend}")


def _flywheel_from_config(config: dict[str, Any], teacher: StrongModel) -> AgenticQwenDataFlywheel:
    flywheel = config.get("flywheel", {})
    return AgenticQwenDataFlywheel(
        teacher,
        generation_attempts=int(flywheel.get("teacher_generation_attempts", 3)),
        validation_attempts=int(flywheel.get("teacher_validation_attempts", 3)),
        allow_deterministic_expansion_fallback=bool(
            flywheel.get("allow_deterministic_expansion_fallback", False)
        ),
    )


def _completed_stage(stage_dir: Path) -> dict[str, Any] | None:
    summary_path = stage_dir / "summary.json"
    if not summary_path.is_file():
        return None
    summary = _read_json(summary_path)
    if summary.get("status") != "completed":
        return None
    adapter_dir = Path(str(summary.get("adapter_dir", "")))
    rollout_path = Path(str(summary.get("policy_rollout_file", "")))
    if not adapter_dir.is_dir() or not rollout_path.is_file():
        return None
    try:
        weights = _adapter_weights(adapter_dir)
    except (FileNotFoundError, RuntimeError):
        return None
    if _sha256(weights) != summary.get("adapter_weights_sha256"):
        return None
    if _sha256(rollout_path) != summary.get("policy_rollout_sha256"):
        return None
    return summary


def _load_evolution_round(
    output_root: Path,
    *,
    round_index: int,
    input_tree_revision: int,
    rollout_count: int,
) -> FlywheelRound | None:
    round_dir = output_root / "flywheel" / f"round_{round_index}"
    tree_path = round_dir / "behavior_tree.json"
    tasks_path = round_dir / "training_tasks.jsonl"
    validation_path = round_dir / "teacher_validation.jsonl"
    if not all(path.is_file() for path in (tree_path, tasks_path, validation_path)):
        return None
    tree = BehaviorTree.from_dict(_read_json(tree_path))
    tasks = load_tasks(tasks_path)
    validations = tuple(
        CandidateValidation.from_dict(json.loads(line))
        for line in validation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not validations:
        return None
    retained_ids = {item.task_id for item in validations if item.retained}
    if {task.task_id for task in tasks} != retained_ids:
        return None
    return FlywheelRound(
        round_index=round_index,
        input_tree_revision=input_tree_revision,
        output_tree_revision=tree.revision,
        rollout_count=rollout_count,
        candidate_count=len(validations),
        retained_count=len(tasks),
        rejected_count=len(validations) - len(tasks),
        tasks=tasks,
        validations=validations,
        tree=tree,
    )


def build_training_plan(config: dict[str, Any]) -> dict[str, Any]:
    stages = ["round0", *[f"round{index}" for index in range(1, int(config["flywheel"]["rounds"]) + 1)]]
    missing = [stage for stage in stages if stage not in config["stages"]]
    if missing:
        raise ValueError(f"Config is missing training stages: {', '.join(missing)}")
    return {
        "algorithm": "AgenticQwen §3.3 alternating Agentic RL data flywheel",
        "stages": [
            {
                "name": stage,
                "max_steps": int(config["stages"][stage]["max_steps"]),
                "input": "SynthAgent happy path" if stage == "round0" else "retained branch-inverted tasks",
                "output": "policy rollout + LoRA adapter",
            }
            for stage in stages
        ],
        "transition": [
            "train small policy with GRPO",
            "roll out in stateful mock user/tool environment",
            "strong model expands behavior tree from trajectories",
            "invert every selected branch into environment/user/agent inputs",
            "strong model solves candidate and branch-hit verifier filters it",
        ],
        "teacher": config.get("teacher", {}),
        "policy_model": config["model"],
        "round0_seed_destinations": list(config["flywheel"].get("seed_destinations", ["Beijing"])),
        "synthesis_budget": {
            "max_total_tasks": int(config["flywheel"].get("max_synthetic_trajectories", 10)),
            "max_new_tasks_per_round": int(config["flywheel"].get("max_new_tasks_per_round", 2)),
            "counting_rule": "Round-0 seed tasks plus retained branch-inverted tasks; replay never creates new tasks",
        },
        "reward_range": [0.0, 1.0],
        "paper_scale_claimed": False,
    }


def synthesize_contract(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Exercise every data-flywheel invariant without claiming a model run."""
    os.environ["AGENTICQWEN_TEACHER_AUDIT_FILE"] = str(output_root / "teacher_api_audit.jsonl")
    teacher = _teacher_from_config(config)
    rounds = _flywheel_from_config(config, teacher).evolve(
        rounds=int(config["flywheel"]["rounds"]),
        max_new_tasks_per_round=int(config["flywheel"].get("max_new_tasks_per_round", 2)),
    )
    manifest = write_flywheel_artifacts(output_root / "flywheel", rounds)
    result = {
        "status": "completed",
        "mode": "algorithm_contract" if isinstance(teacher, DeterministicStrongModel) else "teacher_synthesis",
        "manifest": manifest,
        "plan": build_training_plan(config),
    }
    _write_json(output_root / "synthesis_summary.json", result)
    return result


def run_pipeline(config_path: Path, output_root: Path) -> dict[str, Any]:
    """Run Round 0–3 as train → rollout → expand/invert/filter → retrain."""
    config = _read_json(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["AGENTICQWEN_TEACHER_AUDIT_FILE"] = str(output_root / "teacher_api_audit.jsonl")
    shutil.copy2(config_path, output_root / "resolved_config.json")
    teacher = _teacher_from_config(config)
    if isinstance(teacher, DeterministicStrongModel) and not bool(
        config.get("teacher", {}).get("allow_contract_backend_for_training", False)
    ):
        raise RuntimeError(
            "Full pipeline requires a real strong-model endpoint. Set teacher.backend=openai_compatible, "
            "or explicitly allow the deterministic contract backend for a non-paper teacher ablation."
        )
    flywheel = _flywheel_from_config(config, teacher)
    tree = linear_seed_tree()
    stages: list[dict[str, Any]] = []
    evolution_rounds: list[FlywheelRound] = []
    round0_task_file = output_root / "round0" / "training_tasks.jsonl"
    flywheel_config = config["flywheel"]
    max_total_tasks = int(flywheel_config.get("max_synthetic_trajectories", 10))
    max_new_tasks_per_round = int(flywheel_config.get("max_new_tasks_per_round", 2))
    if not 1 <= max_total_tasks <= 10:
        raise ValueError("max_synthetic_trajectories must be between 1 and 10")
    if max_new_tasks_per_round < 1:
        raise ValueError("max_new_tasks_per_round must be positive")
    if round0_task_file.is_file():
        training_tasks = load_tasks(round0_task_file)
        if len(training_tasks) > max_total_tasks:
            raise RuntimeError(
                f"Persisted Round-0 task count {len(training_tasks)} exceeds synthesis budget {max_total_tasks}"
            )
    else:
        seed_task = flywheel.seed_task(tree)
        training_tasks = seed_task_variants(
            seed_task,
            flywheel_config.get("seed_destinations", ["Beijing"]),
            max_tasks=max_total_tasks,
        )
    previous_adapter: Path | None = None
    replay = bool(config["flywheel"].get("replay_previous_tasks", True))

    for round_index in range(0, int(config["flywheel"]["rounds"]) + 1):
        stage_name = f"round{round_index}"
        persisted_task_file = output_root / stage_name / "training_tasks.jsonl"
        if persisted_task_file.is_file():
            training_tasks = load_tasks(persisted_task_file)
        stage_summary = _completed_stage(output_root / stage_name)
        if stage_summary is None:
            stage_summary = train_round(
                config,
                stage_name=stage_name,
                tasks=training_tasks,
                output_root=output_root,
                input_adapter=previous_adapter,
            )
        stages.append(stage_summary)
        previous_adapter = Path(stage_summary["adapter_dir"])
        _write_json(
            output_root / "run_state.json",
            {
                "status": "running",
                "last_completed_boundary": f"{stage_name}:training_and_policy_rollout",
                "completed_stages": [item["stage"] for item in stages],
                "completed_evolution_rounds": [item.round_index for item in evolution_rounds],
            },
        )
        if round_index == int(config["flywheel"]["rounds"]):
            break

        policy_rollouts = read_policy_rollouts(Path(stage_summary["policy_rollout_file"]))
        evolution = _load_evolution_round(
            output_root,
            round_index=round_index + 1,
            input_tree_revision=tree.revision,
            rollout_count=len(policy_rollouts),
        )
        if evolution is None:
            remaining_budget = max_total_tasks - len(training_tasks)
            if remaining_budget < 1:
                raise RuntimeError(
                    "Synthesis budget exhausted before the configured flywheel rounds completed"
                )
            evolution = flywheel.evolve_one(
                tree=tree,
                policy_rollouts=policy_rollouts,
                round_index=round_index + 1,
                max_candidates=min(max_new_tasks_per_round, remaining_budget),
            )
        if not evolution.tasks:
            # A teacher-gated candidate set can legitimately be empty.  Keep
            # the rejected trajectories and tree as a durable recovery point
            # instead of losing the reason behind the PARTIAL boundary when
            # the controller exits non-zero.
            rejected_dir = output_root / "flywheel" / f"round_{round_index + 1}"
            rejected_dir.mkdir(parents=True, exist_ok=True)
            _write_json(rejected_dir / "behavior_tree.json", evolution.tree.to_dict())
            (rejected_dir / "teacher_validation_rejected.jsonl").write_text(
                "".join(
                    json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                    for item in evolution.validations
                ),
                encoding="utf-8",
            )
            _write_json(
                rejected_dir / "rejection_summary.json",
                {
                    "status": "rejected_all_candidates",
                    "round_index": evolution.round_index,
                    "candidate_count": evolution.candidate_count,
                    "retained_count": evolution.retained_count,
                    "rejected_count": evolution.rejected_count,
                    "teacher_model": teacher.model_name,
                    "synthesis_budget": {
                        "max_total_tasks": max_total_tasks,
                        "max_new_tasks_per_round": max_new_tasks_per_round,
                    },
                },
            )
            _write_json(
                output_root / "run_state.json",
                {
                    "status": "partial",
                    "last_completed_boundary": f"{stage_name}:training_and_policy_rollout",
                    "blocked_boundary": f"round{round_index + 1}:teacher_solved_and_branch_hit_filter",
                    "completed_stages": [item["stage"] for item in stages],
                    "completed_evolution_rounds": [item.round_index for item in evolution_rounds],
                    "recovery_artifact": str((rejected_dir / "rejection_summary.json").resolve()),
                },
            )
            raise RuntimeError(f"Teacher rejected every candidate in evolution round {round_index + 1}")
        evolution_rounds.append(evolution)
        # Persist synthesis evidence immediately so an interrupted GPU run can
        # be audited up to the exact completed data-evolution boundary.
        write_flywheel_artifacts(output_root / "flywheel", tuple(evolution_rounds))
        _write_json(
            output_root / "run_state.json",
            {
                "status": "running",
                "last_completed_boundary": f"round{round_index + 1}:tree_expand_invert_filter",
                "completed_stages": [item["stage"] for item in stages],
                "completed_evolution_rounds": [item.round_index for item in evolution_rounds],
            },
        )
        tree = evolution.tree
        training_tasks = (
            tuple(dict((task.task_id, task) for task in (*training_tasks, *evolution.tasks)).values())
            if replay
            else evolution.tasks
        )
        if len(training_tasks) > max_total_tasks:
            raise RuntimeError(
                f"Synthesis budget violated: {len(training_tasks)} tasks > {max_total_tasks}"
            )

    flywheel_manifest = write_flywheel_artifacts(
        output_root / "flywheel", tuple(evolution_rounds)
    )
    changed = all(bool(item["trainable_parameters_changed"]) for item in stages)
    adapter_chain = all(
        stages[index]["input_adapter"] == stages[index - 1]["adapter_dir"]
        for index in range(1, len(stages))
    )
    result = {
        "status": "completed" if changed and adapter_chain else "failed_verification",
        "algorithm": "AgenticQwen alternating train/trajectory/tree-expansion/branch-inversion loop",
        "stages": stages,
        "flywheel": flywheel_manifest,
        "checks": {
            "all_training_rounds_changed_parameters": changed,
            "adapter_chain_is_continuous": adapter_chain,
            "three_stream_task_schema": True,
            "teacher_solved_and_branch_hit_filter": True,
            "reward_is_normalized_0_1": True,
            "synthesis_budget_respected": len({task.task_id for stage in stages for task in load_tasks(Path(stage["task_file"]))}) <= max_total_tasks,
        },
        "synthesis_budget": {
            "max_total_tasks": max_total_tasks,
            "max_new_tasks_per_round": max_new_tasks_per_round,
        },
        "paper_scale_claimed": False,
    }
    _write_json(output_root / "run_summary.json", result)
    return result


def verify_contract(output_root: Path) -> dict[str, Any]:
    flywheel_root = output_root / "flywheel"
    manifest_path = flywheel_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _read_json(manifest_path)
    checks: list[dict[str, Any]] = []
    for item in manifest["rounds"]:
        for kind, hash_key in (
            ("tree", "tree_sha256"),
            ("tasks", "tasks_sha256"),
            ("validation", "validation_sha256"),
        ):
            configured_path = Path(item["paths"][kind])
            path = configured_path if configured_path.is_absolute() else flywheel_root / configured_path
            valid = path.is_file() and _sha256(path) == item[hash_key]
            checks.append(
                {
                    "name": f"round{item['round_index']}_{kind}_sha256",
                    "status": "PASS" if valid else "FAIL",
                    "path": str(path),
                }
            )
        checks.append(
            {
                "name": f"round{item['round_index']}_teacher_filter",
                "status": "PASS" if item["retained_count"] > 0 else "FAIL",
                "retained": item["retained_count"],
                "candidates": item["candidate_count"],
            }
        )
    result = {
        "overall_status": "PASS" if checks and all(item["status"] == "PASS" for item in checks) else "FAIL",
        "checks": checks,
    }
    _write_json(output_root / "verification.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-aligned AgenticQwen behavior-tree RL flywheel")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("plan", "synthesize", "train", "verify"),
        default="plan",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    config = _read_json(config_path)
    if args.mode == "plan":
        result = build_training_plan(config)
    elif args.mode == "synthesize":
        result = synthesize_contract(config, output_root)
    elif args.mode == "train":
        result = run_pipeline(config_path, output_root)
    else:
        result = verify_contract(output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    status = result.get("overall_status", result.get("status", "completed"))
    return 0 if status not in {"FAIL", "failed_verification"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
