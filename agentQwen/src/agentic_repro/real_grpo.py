from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DecisionTask:
    task_id: str
    policy_excerpt: str
    user_request: str
    rubric: str
    candidates: tuple[str, str, str, str]
    correct_index: int
    normal_path: tuple[str, ...]
    hack_path: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def _tool_names_from_system(system: str) -> list[str]:
    names: list[str] = []
    patterns = (
        r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"',
        r"<name>\s*([A-Za-z_][A-Za-z0-9_]*)\s*</name>",
        r"function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    )
    for pattern in patterns:
        for match in re.findall(pattern, system):
            if match not in names:
                names.append(match)
    return names


def _compact_policy(system: str, max_chars: int) -> str:
    policy = system
    for marker in ("<tools>", "<tool", "# Tools", "## Tools", "Available tools"):
        if marker in policy:
            policy = policy.split(marker, 1)[0]
            break
    policy = re.sub(r"\s+", " ", policy).strip()
    return policy[:max_chars]


def build_tasks(
    parquet_path: Path,
    *,
    count: int,
    seed: int,
    policy_chars: int,
) -> tuple[list[DecisionTask], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the official AgenticQwen parquet") from exc

    columns = [
        "id",
        "system",
        "user",
        "rubrics",
        "tool_return_expected_json",
    ]
    table = pq.read_table(parquet_path, columns=columns)
    rows = table.to_pylist()
    rng = random.Random(seed)
    row_order = list(range(len(rows)))
    rng.shuffle(row_order)
    tasks: list[DecisionTask] = []

    for row_idx in row_order:
        row = rows[row_idx]
        try:
            expected = _json_loads(row["tool_return_expected_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        normal = [step.get("tool_name") for step in expected.get("normal_path", [])]
        hack = [step.get("tool_name") for step in expected.get("hack_path", [])]
        normal = [name for name in normal if isinstance(name, str) and name]
        hack = [name for name in hack if isinstance(name, str) and name]
        if not normal:
            continue

        target = normal[0]
        pool: list[str] = []
        for name in hack + normal[1:] + _tool_names_from_system(row["system"]):
            if name != target and name not in pool:
                pool.append(name)
        for fallback in (
            "ask_user_for_missing_information",
            "refuse_request",
            "transfer_to_human_reviewer",
            "confirm_completion",
        ):
            if fallback != target and fallback not in pool:
                pool.append(fallback)
        if len(pool) < 3:
            continue

        distractors = pool[:3]
        target_position = len(tasks) % 4
        candidates = distractors[:]
        candidates.insert(target_position, target)
        tasks.append(
            DecisionTask(
                task_id=str(row["id"]),
                policy_excerpt=_compact_policy(row["system"], policy_chars),
                user_request=re.sub(r"\s+", " ", row["user"]).strip()[:900],
                rubric=re.sub(r"\s+", " ", row["rubrics"]).strip()[:1200],
                candidates=tuple(candidates),
                correct_index=target_position,
                normal_path=tuple(normal),
                hack_path=tuple(hack),
            )
        )
        if len(tasks) >= count:
            break

    if len(tasks) < count:
        raise RuntimeError(f"Only {len(tasks)} usable tasks were found; requested {count}")
    metadata = {
        "source": "alibaba-pai/AgenticQwen-Data",
        "parquet_path": str(parquet_path.resolve()),
        "parquet_sha256": _sha256(parquet_path),
        "source_rows": int(table.num_rows),
        "selected_rows": len(tasks),
        "selection_seed": seed,
    }
    return tasks, metadata


def render_prompt(task: DecisionTask, tokenizer: Any, *, variant: str, max_tokens: int) -> tuple[str, list[int]]:
    if variant == "train":
        instruction = "Choose the single best next tool action under the policy."
    elif variant == "holdout":
        instruction = "Which action should a compliant agent take next?"
    else:
        instruction = "Select the safest policy-valid next operation."
    action_lines = "\n".join(f"{idx + 1}. {name}" for idx, name in enumerate(task.candidates))
    user_content = (
        f"{instruction}\n\n"
        f"User request: {task.user_request}\n\n"
        f"Candidate actions:\n{action_lines}\n\n"
        "Return only one digit: 1, 2, 3, or 4."
    )
    policy = task.policy_excerpt
    while True:
        messages = [
            {"role": "system", "content": policy},
            {"role": "user", "content": user_content},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            prompt = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, **kwargs)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return prompt, tokens
        if len(policy) <= 400:
            return prompt, tokens[-max_tokens:]
        policy = policy[: int(len(policy) * 0.82)]


def _action_token_ids(tokenizer: Any) -> list[int]:
    token_ids: list[int] = []
    for digit in ("1", "2", "3", "4"):
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"Expected {digit!r} to be one token, got {encoded}")
        token_ids.append(encoded[0])
    return token_ids


def _policy_vector(model: Any, mx: Any, token_ids: list[int], action_ids: list[int], temperature: float) -> tuple[Any, Any]:
    tokens = mx.array([token_ids])
    logits = model(tokens)[0, -1]
    selected = logits[mx.array(action_ids)].astype(mx.float32) / temperature
    log_probs = selected - mx.logsumexp(selected)
    return tokens, log_probs


def evaluate(
    model: Any,
    tokenizer: Any,
    tasks: list[DecisionTask],
    *,
    split: str,
    variant: str,
    max_prompt_tokens: int,
    temperature: float = 1.0,
) -> dict[str, Any]:
    import mlx.core as mx

    model.eval()
    action_ids = _action_token_ids(tokenizer)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for task in tasks:
        prompt, ids = render_prompt(task, tokenizer, variant=variant, max_tokens=max_prompt_tokens)
        _, log_probs = _policy_vector(model, mx, ids, action_ids, temperature)
        probs_array = mx.exp(log_probs)
        mx.eval(probs_array)
        probs = [float(x) for x in probs_array.tolist()]
        predicted = max(range(4), key=probs.__getitem__)
        rows.append(
            {
                "task_id": task.task_id,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_tokens": len(ids),
                "candidates": list(task.candidates),
                "correct_index": task.correct_index,
                "correct_action": task.candidates[task.correct_index],
                "predicted_index": predicted,
                "predicted_action": task.candidates[predicted],
                "correct": predicted == task.correct_index,
                "action_probabilities": [round(value, 8) for value in probs],
                "correct_action_probability": round(probs[task.correct_index], 8),
            }
        )
        mx.clear_cache()
    elapsed = time.perf_counter() - started
    return {
        "split": split,
        "prompt_variant": variant,
        "task_count": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "mean_correct_action_probability": sum(row["correct_action_probability"] for row in rows) / len(rows),
        "elapsed_seconds": round(elapsed, 4),
        "rows": rows,
    }


def _lora_b_norm(model: Any, mx: Any, tree_flatten: Any) -> float:
    squares = []
    for name, value in tree_flatten(model.trainable_parameters()):
        if name.endswith("lora_b"):
            squares.append(mx.sum(value.astype(mx.float32) ** 2))
    if not squares:
        return 0.0
    total = sum(squares, mx.array(0.0))
    mx.eval(total)
    return math.sqrt(float(total.item()))


def run(config_path: Path) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_dir = config_path.parent.parent
    artifacts_dir = project_dir / "artifacts" / "real_qwen3_8b"
    adapters_dir = artifacts_dir / "adapter"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(config["model_path"]).resolve()
    parquet_path = Path(config["dataset_path"]).resolve()
    model_file = model_path / "model.safetensors"
    if not model_file.exists() or model_file.stat().st_size < 4_000_000_000:
        raise RuntimeError(f"Complete Qwen3-8B weights not found at {model_file}")

    all_tasks, dataset_meta = build_tasks(
        parquet_path,
        count=config["task_count"],
        seed=config["seed"],
        policy_chars=config["policy_chars"],
    )
    train_count = config["train_task_count"]
    train_tasks = all_tasks[:train_count]
    unseen_tasks = all_tasks[train_count:]
    manifest = {
        "dataset": dataset_meta,
        "train_task_ids": [task.task_id for task in train_tasks],
        "unseen_task_ids": [task.task_id for task in unseen_tasks],
        "tasks": [asdict(task) for task in all_tasks],
    }
    (artifacts_dir / "task_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    mx.random.seed(config["seed"])
    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    load_started = time.perf_counter()
    model, tokenizer = load(str(model_path))
    load_seconds = time.perf_counter() - load_started
    model.freeze()
    linear_to_lora_layers(
        model,
        config["lora_num_layers"],
        {
            "rank": config["lora_rank"],
            "scale": config["lora_scale"],
            "dropout": 0.0,
            "keys": config.get("lora_keys", ["self_attn.q_proj", "self_attn.v_proj"]),
        },
    )
    if config.get("gradient_checkpointing", True):
        grad_checkpoint(model.layers[0])
    print_trainable_parameters(model)

    initial_weights = dict(tree_flatten(model.trainable_parameters()))
    initial_adapter = adapters_dir / "adapters_initial.safetensors"
    mx.save_safetensors(str(initial_adapter), initial_weights)
    initial_norm = _lora_b_norm(model, mx, tree_flatten)

    baseline = {
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
    (artifacts_dir / "baseline_eval.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    action_ids = _action_token_ids(tokenizer)
    rollout_temperature = float(config["rollout_temperature"])
    clip_epsilon = float(config["clip_epsilon"])
    entropy_beta = float(config["entropy_beta"])

    def grpo_loss(model: Any, tokens: Any, actions: Any, old_log_probs: Any, advantages: Any):
        logits = model(tokens)[0, -1]
        action_logits = logits[mx.array(action_ids)].astype(mx.float32) / rollout_temperature
        log_probs = action_logits - mx.logsumexp(action_logits)
        chosen = log_probs[actions]
        ratio = mx.exp(chosen - old_log_probs)
        unclipped = ratio * advantages
        clipped = mx.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -mx.mean(mx.minimum(unclipped, clipped))
        entropy = -mx.sum(mx.exp(log_probs) * log_probs)
        return policy_loss - entropy_beta * entropy

    value_and_grad = nn.value_and_grad(model, grpo_loss)
    optimizer = optim.Adam(learning_rate=float(config["learning_rate"]))
    rng = random.Random(config["seed"] + 17)
    log_rows: list[dict[str, Any]] = []
    train_started = time.perf_counter()
    model.train()
    step_number = 0

    for repeat in range(config["training_repeats"]):
        order = list(range(len(train_tasks)))
        rng.shuffle(order)
        for task_index in order:
            task = train_tasks[task_index]
            _, ids = render_prompt(
                task,
                tokenizer,
                variant="train",
                max_tokens=config["max_prompt_tokens"],
            )
            tokens, log_probs = _policy_vector(model, mx, ids, action_ids, rollout_temperature)
            probs_mx = mx.exp(log_probs)
            mx.eval(probs_mx)
            probs = [float(x) for x in probs_mx.tolist()]

            sampled: list[int] = []
            rewards: list[float] = []
            for _ in range(config["group_size"]):
                action = rng.choices(range(4), weights=probs, k=1)[0]
                sampled.append(action)
                rewards.append(1.0 if action == task.correct_index else 0.0)
            reward_mean = sum(rewards) / len(rewards)
            reward_variance = sum((reward - reward_mean) ** 2 for reward in rewards) / len(rewards)
            if reward_variance == 0.0:
                log_rows.append(
                    {
                        "step": step_number,
                        "repeat": repeat,
                        "task_id": task.task_id,
                        "skipped": True,
                        "reason": "zero within-group reward variance",
                        "rollout_actions": sampled,
                        "rewards": rewards,
                        "policy_probabilities": probs,
                    }
                )
                continue

            reward_std = math.sqrt(reward_variance)
            advantages = [(reward - reward_mean) / (reward_std + 1e-6) for reward in rewards]
            sampled_mx = mx.array(sampled)
            old_selected_mx = log_probs[sampled_mx]
            advantages_mx = mx.array(advantages, dtype=mx.float32)
            mx.eval(old_selected_mx)

            losses: list[float] = []
            for _ in range(config["ppo_epochs"]):
                loss, grads = value_and_grad(
                    model,
                    tokens,
                    sampled_mx,
                    old_selected_mx,
                    advantages_mx,
                )
                optimizer.update(model, grads)
                mx.eval(model.state, optimizer.state, loss)
                losses.append(float(loss.item()))
                step_number += 1
                mx.clear_cache()

            log_rows.append(
                {
                    "step": step_number,
                    "repeat": repeat,
                    "task_id": task.task_id,
                    "skipped": False,
                    "rollout_actions": sampled,
                    "rollout_action_names": [task.candidates[index] for index in sampled],
                    "rewards": rewards,
                    "reward_mean": reward_mean,
                    "advantages": advantages,
                    "policy_probabilities": probs,
                    "losses": losses,
                    "correct_action": task.candidates[task.correct_index],
                }
            )
            print(
                f"[grpo] step={step_number} task={task.task_id[:8]} "
                f"reward={reward_mean:.3f} loss={losses[-1]:.5f}",
                flush=True,
            )

    training_seconds = time.perf_counter() - train_started
    with (artifacts_dir / "training_log.jsonl").open("w", encoding="utf-8") as handle:
        for row in log_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    final_adapter = adapters_dir / "adapters.safetensors"
    mx.save_safetensors(str(final_adapter), dict(tree_flatten(model.trainable_parameters())))
    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": config["lora_num_layers"],
        "lora_parameters": {
            "rank": config["lora_rank"],
            "scale": config["lora_scale"],
            "dropout": 0.0,
            "keys": config.get("lora_keys", ["self_attn.q_proj", "self_attn.v_proj"]),
        },
        "training_algorithm": "action-masked LoRA-GRPO",
    }
    (adapters_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    final_norm = _lora_b_norm(model, mx, tree_flatten)

    final_eval = {
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
    (artifacts_dir / "final_eval.json").write_text(
        json.dumps(final_eval, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    attempted_groups = len(log_rows)
    updated_groups = sum(not row["skipped"] for row in log_rows)
    summary = {
        "status": "completed",
        "scope": "real Qwen3-8B 4-bit, local action-masked small-scale GRPO reproduction",
        "paper_scale_claimed": False,
        "model": {
            "id": "Qwen/Qwen3-8B-MLX-4bit",
            "path": str(model_path),
            "weights_bytes": model_file.stat().st_size,
            "weights_sha256": _sha256(model_file),
            "load_seconds": round(load_seconds, 4),
            "quantization": "4-bit MLX",
        },
        "dataset": dataset_meta,
        "training": {
            "algorithm": "group-relative policy optimization over executable next-tool rewards",
            "action_space": "four task-specific candidate tool calls, emitted as one token",
            "train_tasks": len(train_tasks),
            "unseen_tasks": len(unseen_tasks),
            "group_size": config["group_size"],
            "attempted_groups": attempted_groups,
            "updated_groups": updated_groups,
            "optimizer_steps": step_number,
            "training_seconds": round(training_seconds, 4),
            "lora_b_norm_before": initial_norm,
            "lora_b_norm_after": final_norm,
            "adapter_initial_sha256": _sha256(initial_adapter),
            "adapter_final_sha256": _sha256(final_adapter),
        },
        "metrics": {
            split: {
                "accuracy_before": baseline[split]["accuracy"],
                "accuracy_after": final_eval[split]["accuracy"],
                "mean_correct_probability_before": baseline[split]["mean_correct_action_probability"],
                "mean_correct_probability_after": final_eval[split]["mean_correct_action_probability"],
            }
            for split in ("train", "holdout", "unseen")
        },
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "peak_memory_gib": round(mx.get_peak_memory() / (1024**3), 3),
        },
        "limitations": [
            "Action-masked four-way next-tool selection replaces unrestricted JSON/function-call generation.",
            "Only a deterministic subset of the official synthetic dataset is used.",
            "No 7B-task online flywheel, multi-node verl training, or production environment is reproduced.",
            "Unseen-task metrics are reported separately from same-task prompt holdout metrics.",
        ],
    }
    (artifacts_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real local Qwen3-8B action-level GRPO reproduction")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    summary = run(args.config.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
