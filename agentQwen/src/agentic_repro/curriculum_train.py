from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .curriculum_env import (
    StatefulRefundEnvironment,
    base_task_bank,
    load_tasks,
    read_jsonl,
    summarize_failures,
    synthesize_hard_tasks,
    task_rows,
    write_tasks,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(file)))
    return digest.hexdigest()


def _trainable_parameter_hash(model: Any) -> str:
    """Hash optimizer-visible tensors without persisting another adapter."""
    import torch

    digest = hashlib.sha256()
    trainable = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            trainable += parameter.numel()
            digest.update(name.encode("utf-8"))
            value = parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
            digest.update(value.numpy().tobytes())
            del value
    if trainable == 0:
        raise RuntimeError("No trainable parameters found while hashing LoRA state")
    return digest.hexdigest()


def _adapter_weights(path: Path) -> Path:
    preferred = path / "adapter_model.safetensors"
    if preferred.is_file():
        return preferred
    candidates = sorted(path.glob("*.safetensors"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one adapter safetensors file under {path}, found {candidates}"
        )
    return candidates[0]


def _clean_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def _latest_checkpoint(trainer_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in trainer_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def _completed_stage(output_root: Path, stage_name: str) -> dict[str, Any] | None:
    summary_path = output_root / stage_name / "summary.json"
    if not summary_path.exists():
        return None
    summary = _read_json(summary_path)
    adapter_value = summary.get("adapter_dir")
    if not adapter_value:
        return None
    adapter_dir = Path(adapter_value)
    if (
        summary.get("status") != "completed"
        or not adapter_dir.is_dir()
        or not summary.get("adapter_weights_sha256")
    ):
        return None
    return summary


def _task_hashes(path: Path) -> dict[str, Any]:
    tasks = load_tasks(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "count": len(tasks),
        "task_ids": [task.task_id for task in tasks],
    }


def prepare_data(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    task_dir = output_root / "tasks"
    banks = base_task_bank()
    paths = {
        "stage1_train": task_dir / "stage1_train.jsonl",
        "probe": task_dir / "curriculum_probe.jsonl",
        "holdout": task_dir / "final_holdout.jsonl",
    }
    for key, path in paths.items():
        write_tasks(path, banks[key])
    ids = {key: {task.task_id for task in values} for key, values in banks.items()}
    if ids["stage1_train"] & ids["holdout"] or ids["probe"] & ids["holdout"]:
        raise RuntimeError("Final holdout task IDs overlap train/probe data")
    manifest = {
        "schema_version": 1,
        "seed": int(config["seed"]),
        "ground_truth_provenance": "deterministic environment code, never model output",
        "datasets": {key: _task_hashes(path) for key, path in paths.items()},
        "contamination_check": "PASS",
    }
    _write_json(output_root / "data_manifest.json", manifest)
    return manifest


def synthesize_stage2(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    # Prefer residual probe failures. If the tiny probe reaches 100%, keep the
    # flywheel grounded in observed failures by mining Stage-1 training traces,
    # then the pre-training probe. Only fall back to the configured frontier
    # taxonomy when every observed trajectory succeeds.
    candidates = (
        ("post_train_probe_failures", output_root / "stage1" / "eval_probe_traces.jsonl"),
        ("stage1_training_failures", output_root / "stage1" / "train_traces.jsonl"),
        ("pre_train_probe_failures", output_root / "stage1" / "eval_before_traces.jsonl"),
    )
    source_name = "frontier_taxonomy_fallback"
    failure_path = candidates[0][1]
    rows: list[dict[str, Any]] = []
    available_sources: list[dict[str, Any]] = []
    for candidate_name, candidate_path in candidates:
        candidate_rows = read_jsonl(candidate_path)
        if candidate_rows:
            available_sources.append(
                {
                    "name": candidate_name,
                    "path": str(candidate_path.resolve()),
                    "sha256": _sha256(candidate_path),
                    "episodes": len(candidate_rows),
                    "failures": sum(
                        row.get("failure_type") != "SUCCESS" for row in candidate_rows
                    ),
                }
            )
        failed_rows = [row for row in candidate_rows if row.get("failure_type") != "SUCCESS"]
        if failed_rows and not rows:
            source_name = candidate_name
            failure_path = candidate_path
            rows = failed_rows
    if not available_sources:
        raise RuntimeError("No Stage-1 traces found for curriculum synthesis")
    hard_tasks, manifest = synthesize_hard_tasks(
        rows,
        count=int(config["curriculum"]["hard_task_count"]),
        seed=int(config["seed"]) + 1,
    )
    hard_path = output_root / "tasks" / "stage2_hard.jsonl"
    write_tasks(hard_path, hard_tasks)
    replay_count = int(config["curriculum"].get("replay_task_count", 4))
    replay = load_tasks(output_root / "tasks" / "stage1_train.jsonl")[:replay_count]
    stage2_train = hard_tasks + replay
    train_path = output_root / "tasks" / "stage2_train.jsonl"
    write_tasks(train_path, stage2_train)
    manifest.update(
        {
            "source_trace": str(failure_path.resolve()),
            "source_trace_sha256": _sha256(failure_path),
            "source_selection": source_name,
            "source_failure_rows": len(rows),
            "available_sources": available_sources,
            "hard_tasks": _task_hashes(hard_path),
            "replay_tasks": [task.task_id for task in replay],
            "stage2_train": _task_hashes(train_path),
        }
    )
    _write_json(output_root / "stage2" / "synthesis_manifest.json", manifest)
    return manifest


def _runtime_versions() -> dict[str, Any]:
    import accelerate
    import bitsandbytes
    import datasets
    import peft
    import torch
    import transformers
    import trl

    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "capability": list(torch.cuda.get_device_capability(0)),
            "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3),
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "datasets": datasets.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
    }


def _model_load_path(config: dict[str, Any]) -> str:
    """Return a local snapshot when the orchestrator has pre-downloaded one.

    The immutable source model ID remains in the resolved config and summaries;
    this override changes transport/location only, not the declared base model.
    """

    return os.getenv("AGENTICQWEN_MODEL_PATH", str(config["model"]["id"]))


def _capture_model_snapshot(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    load_path = _model_load_path(config)
    source = {
        "source_model_id": str(config["model"]["id"]),
        "load_path": load_path,
        "transport": "hub_runtime",
        "manifest_path": None,
        "manifest_sha256": None,
    }
    local_dir = Path(load_path)
    marker = local_dir / ".modelscope_complete.json"
    if local_dir.is_dir() and marker.is_file():
        manifest = _read_json(marker)
        if manifest.get("status") != "completed":
            raise RuntimeError(f"Incomplete local model snapshot marker: {marker}")
        destination = output_root / "model_snapshot_manifest.json"
        shutil.copy2(marker, destination)
        source.update(
            {
                "transport": "modelscope_local_snapshot",
                "manifest_path": str(destination.resolve()),
                "manifest_sha256": _sha256(destination),
                "snapshot_bytes": int(manifest.get("bytes", 0)),
                "snapshot_file_count": len(manifest.get("files", [])),
            }
        )
    return source


def _build_model(config: dict[str, Any], input_adapter: Path | None):
    import torch
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = _model_load_path(config)
    dtype = torch.bfloat16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        torch_dtype=dtype,
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
        device_map={"": int(os.getenv("LOCAL_RANK", "0"))},
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if input_adapter:
        model = PeftModel.from_pretrained(model, str(input_adapter), is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=int(config["model"]["lora_rank"]),
            lora_alpha=int(config["model"]["lora_alpha"]),
            lora_dropout=float(config["model"].get("lora_dropout", 0.0)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(config["model"]["target_modules"]),
        )
    return model, tokenizer, peft_config


def _make_args(config: dict[str, Any], stage_name: str, output_dir: Path):
    from trl import GRPOConfig

    common = config["training"]
    stage = config["stages"][stage_name]
    return GRPOConfig(
        output_dir=str(output_dir),
        run_name=f"agenticqwen-{stage_name}",
        max_steps=int(stage["max_steps"]),
        learning_rate=float(stage.get("learning_rate", common["learning_rate"])),
        per_device_train_batch_size=int(common["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(common.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(common["gradient_accumulation_steps"]),
        num_generations=int(common["num_generations"]),
        num_generations_eval=int(common.get("num_generations_eval", 1)),
        max_completion_length=int(common["max_completion_length"]),
        max_tool_calling_iterations=int(common["max_tool_calling_iterations"]),
        temperature=float(common["temperature"]),
        top_p=float(common["top_p"]),
        repetition_penalty=float(common.get("repetition_penalty", 1.0)),
        beta=float(common["beta"]),
        epsilon=float(common.get("epsilon", 0.2)),
        loss_type=str(common.get("loss_type", "dapo")),
        mask_truncated_completions=True,
        scale_rewards="group",
        gradient_checkpointing=True,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        max_grad_norm=float(common.get("max_grad_norm", 1.0)),
        warmup_ratio=float(common.get("warmup_ratio", 0.05)),
        logging_steps=1,
        logging_first_step=True,
        log_completions=True,
        num_completions_to_print=1,
        save_strategy="steps",
        save_steps=int(stage.get("save_steps", max(1, int(stage["max_steps"]) // 2))),
        save_total_limit=2,
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        chat_template_kwargs={"enable_thinking": False},
        disable_dropout=True,
        use_vllm=False,
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
    )


def _dataset(path: Path, stage: str):
    from datasets import Dataset

    tasks = load_tasks(path)
    if not tasks:
        raise RuntimeError(f"Task file is empty: {path}")
    return Dataset.from_list(task_rows(tasks, stage))


def train_stage(
    config: dict[str, Any],
    *,
    stage_name: str,
    train_tasks_path: Path,
    eval_tasks_path: Path,
    output_root: Path,
    input_adapter: Path | None = None,
) -> dict[str, Any]:
    import torch
    from trl import GRPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Curriculum GRPO training requires a CUDA GPU")
    stage_dir = output_root / stage_name
    adapter_dir = stage_dir / "adapter"
    trainer_dir = stage_dir / "trainer"
    stage_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = _dataset(train_tasks_path, stage_name)
    eval_dataset = _dataset(eval_tasks_path, f"{stage_name}_eval")
    model, tokenizer, peft_config = _build_model(config, input_adapter)
    args = _make_args(config, stage_name, trainer_dir)
    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        environment_factory=StatefulRefundEnvironment,
        peft_config=peft_config,
    )
    process_weight = float(config["training"]["process_reward_weight"])
    os.environ["AGENTICQWEN_PROCESS_REWARD_WEIGHT"] = str(process_weight)
    resume_checkpoint = _latest_checkpoint(trainer_dir)
    trainable_hash_before = _trainable_parameter_hash(trainer.model)

    baseline_trace = stage_dir / "eval_before_traces.jsonl"
    _clean_trace(baseline_trace)
    os.environ["AGENTICQWEN_TRACE_FILE"] = str(baseline_trace)
    torch.manual_seed(int(config["seed"]))
    baseline_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="before")

    train_trace = stage_dir / "train_traces.jsonl"
    if resume_checkpoint is None:
        _clean_trace(train_trace)
    os.environ["AGENTICQWEN_TRACE_FILE"] = str(train_trace)
    started = time.perf_counter()
    train_output = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    training_seconds = time.perf_counter() - started
    trainable_hash_after = _trainable_parameter_hash(trainer.model)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    adapter_weights = _adapter_weights(adapter_dir)

    eval_trace = stage_dir / ("eval_probe_traces.jsonl" if stage_name == "stage1" else "eval_final_traces.jsonl")
    _clean_trace(eval_trace)
    os.environ["AGENTICQWEN_TRACE_FILE"] = str(eval_trace)
    torch.manual_seed(int(config["seed"]))
    final_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="after")

    train_rows = read_jsonl(train_trace)
    before_rows = read_jsonl(baseline_trace)
    after_rows = read_jsonl(eval_trace)
    summary = {
        "schema_version": 1,
        "status": "completed",
        "stage": stage_name,
        "model_id": config["model"]["id"],
        "model_load_path": _model_load_path(config),
        "algorithm": "multi-turn response-token QLoRA-GRPO with stateful tools",
        "environment_factory": "StatefulRefundEnvironment",
        "runtime": _runtime_versions(),
        "training_seconds": round(training_seconds, 3),
        "global_step": int(train_output.global_step),
        "trainable_parameter_hash_before": trainable_hash_before,
        "trainable_parameter_hash_after": trainable_hash_after,
        "trainable_parameters_changed": trainable_hash_before != trainable_hash_after,
        "resumed_from_checkpoint": str(resume_checkpoint.resolve()) if resume_checkpoint else None,
        "train_output": dict(train_output.metrics),
        "metrics_before": baseline_metrics,
        "metrics_after": final_metrics,
        "failure_summary_before": summarize_failures(before_rows),
        "failure_summary_after": summarize_failures(after_rows),
        "training_rollout_summary": summarize_failures(train_rows),
        "train_tasks": _task_hashes(train_tasks_path),
        "eval_tasks": _task_hashes(eval_tasks_path),
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter_tree_sha256": _tree_hash(adapter_dir),
        "adapter_weights_file": str(adapter_weights.resolve()),
        "adapter_weights_sha256": _sha256(adapter_weights),
        "input_adapter": str(input_adapter.resolve()) if input_adapter else None,
        "input_adapter_tree_sha256": _tree_hash(input_adapter) if input_adapter else None,
        "input_adapter_weights_sha256": (
            _sha256(_adapter_weights(input_adapter)) if input_adapter else None
        ),
        "trace_files": {
            "before": str(baseline_trace.resolve()),
            "train": str(train_trace.resolve()),
            "after": str(eval_trace.resolve()),
        },
        "paper_scale_claimed": False,
    }
    _write_json(stage_dir / "summary.json", summary)
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _evaluate_adapter_fresh_process(
    *,
    output_root: Path,
    adapter_dir: Path,
    tasks_path: Path,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "agentic_repro.curriculum_train",
        "--config",
        str(output_root / "resolved_config.json"),
        "--output-root",
        str(output_root),
        "--mode",
        "evaluate",
        "--adapter-dir",
        str(adapter_dir),
        "--tasks-path",
        str(tasks_path),
        "--eval-output-dir",
        str(output_dir),
        "--label",
        label,
    ]
    env = os.environ.copy()
    env["AGENTICQWEN_PARENT_PID"] = str(os.getpid())
    log_path = output_dir / "process.log"
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Fresh-process evaluation failed with exit code {completed.returncode}; see {log_path}"
        )
    result = _read_json(output_dir / "summary.json")
    result["command"] = command
    result["log"] = str(log_path.resolve())
    return result


def verify_run(config: dict[str, Any], output_root: Path, replay: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "status": "PASS" if condition else "FAIL", "detail": detail})

    stage1 = _read_json(output_root / "stage1" / "summary.json")
    stage2 = _read_json(output_root / "stage2" / "summary.json")
    manifest = _read_json(output_root / "data_manifest.json")
    synth = _read_json(output_root / "stage2" / "synthesis_manifest.json")
    stage1_adapter = Path(stage1["adapter_dir"])
    stage2_adapter = Path(stage2["adapter_dir"])
    check("stage1 adapter exists", stage1_adapter.exists(), str(stage1_adapter))
    check("stage2 adapter exists", stage2_adapter.exists(), str(stage2_adapter))
    check(
        "stage2 adapter differs from stage1",
        stage1["adapter_weights_sha256"] != stage2["adapter_weights_sha256"],
        f"stage1={stage1['adapter_weights_sha256']}; stage2={stage2['adapter_weights_sha256']}",
    )
    check(
        "both GRPO stages changed trainable parameters",
        bool(stage1.get("trainable_parameters_changed"))
        and bool(stage2.get("trainable_parameters_changed")),
        f"stage1={stage1.get('trainable_parameters_changed')}; "
        f"stage2={stage2.get('trainable_parameters_changed')}",
    )
    stage1_reward_std = float(stage1.get("training_rollout_summary", {}).get("reward_std", 0.0))
    stage2_reward_std = float(stage2.get("training_rollout_summary", {}).get("reward_std", 0.0))
    check(
        "training reward variance observed",
        stage1_reward_std > 0.0 and stage2_reward_std > 0.0,
        f"stage1_std={stage1_reward_std}; stage2_std={stage2_reward_std}",
    )
    holdout_ids = set(manifest["datasets"]["holdout"]["task_ids"])
    train_ids = set(manifest["datasets"]["stage1_train"]["task_ids"])
    stage2_ids = set(synth["stage2_train"]["task_ids"])
    check(
        "final holdout isolation",
        not holdout_ids.intersection(train_ids | stage2_ids),
        f"overlap={sorted(holdout_ids.intersection(train_ids | stage2_ids))}",
    )
    check(
        "failure-driven synthesis has provenance",
        bool(synth.get("source_trace_sha256")) and synth.get("task_count", 0) > 0,
        synth.get("generator", "missing"),
    )
    check(
        "stage2 consumed stage1 adapter",
        stage2.get("input_adapter_weights_sha256") == stage1.get("adapter_weights_sha256"),
        str(stage2.get("input_adapter_weights_sha256")),
    )
    check(
        "all planned optimizer steps completed",
        int(stage1.get("global_step", 0)) == int(config["stages"]["stage1"]["max_steps"])
        and int(stage2.get("global_step", 0)) == int(config["stages"]["stage2"]["max_steps"]),
        f"stage1={stage1.get('global_step')}/{config['stages']['stage1']['max_steps']}; "
        f"stage2={stage2.get('global_step')}/{config['stages']['stage2']['max_steps']}",
    )
    check(
        "evaluation traces contain complete episode sets",
        int(stage1.get("failure_summary_after", {}).get("episodes", 0))
        >= int(manifest["datasets"]["probe"]["count"])
        and int(stage2.get("failure_summary_after", {}).get("episodes", 0))
        >= int(manifest["datasets"]["holdout"]["count"]),
        f"stage1_probe={stage1.get('failure_summary_after', {}).get('episodes')}; "
        f"stage2_holdout={stage2.get('failure_summary_after', {}).get('episodes')}",
    )
    replay_result: dict[str, Any] | None = None
    if replay:
        replay_dir = output_root / "fresh_replay"
        replay_result = _evaluate_adapter_fresh_process(
            output_root=output_root,
            adapter_dir=stage2_adapter,
            tasks_path=output_root / "tasks" / "final_holdout.jsonl",
            output_dir=replay_dir,
            label="fresh_replay",
        )
        process_info = replay_result.get("process", {})
        check(
            "fresh-process adapter replay",
            replay_result["episodes"] >= int(manifest["datasets"]["holdout"]["count"])
            and process_info.get("is_fresh_process") is True,
            f"episodes={replay_result['episodes']}; success_rate={replay_result['success_rate']}; "
            f"parent_pid={process_info.get('parent_pid')}; child_pid={process_info.get('pid')}",
        )
    result = {
        "schema_version": 1,
        "overall_status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "checks": checks,
        "fresh_replay": replay_result,
    }
    _write_json(output_root / "verification.json", result)
    return result


def evaluate_adapter(
    config: dict[str, Any],
    *,
    adapter_dir: Path,
    tasks_path: Path,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    import torch
    from trl import GRPOTrainer

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _dataset(tasks_path, label)
    model, tokenizer, _ = _build_model(config, adapter_dir)
    args = _make_args(config, "stage2", output_dir / "trainer")
    args.max_steps = 1
    args.num_generations_eval = 1
    args.per_device_train_batch_size = 1
    args.gradient_accumulation_steps = 1
    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
        environment_factory=StatefulRefundEnvironment,
    )
    trace_path = output_dir / "traces.jsonl"
    _clean_trace(trace_path)
    os.environ["AGENTICQWEN_TRACE_FILE"] = str(trace_path)
    os.environ["AGENTICQWEN_PROCESS_REWARD_WEIGHT"] = str(
        config["training"]["process_reward_weight"]
    )
    torch.manual_seed(int(config["seed"]))
    metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=label)
    rows = read_jsonl(trace_path)
    summary = summarize_failures(rows)
    result = {"label": label, "metrics": metrics, **summary, "trace": str(trace_path.resolve())}
    parent_pid = int(os.getenv("AGENTICQWEN_PARENT_PID", "0"))
    result["process"] = {
        "pid": os.getpid(),
        "parent_pid": parent_pid or None,
        "is_fresh_process": bool(parent_pid and parent_pid != os.getpid()),
        "python": sys.executable,
    }
    _write_json(output_dir / "summary.json", result)
    return result


def run_pipeline(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / "resolved_config.json")
    model_snapshot = _capture_model_snapshot(config, output_root)
    prepare = prepare_data(config, output_root)
    stage1 = _completed_stage(output_root, "stage1") or train_stage(
        config,
        stage_name="stage1",
        train_tasks_path=output_root / "tasks" / "stage1_train.jsonl",
        eval_tasks_path=output_root / "tasks" / "curriculum_probe.jsonl",
        output_root=output_root,
    )
    synthesis_path = output_root / "stage2" / "synthesis_manifest.json"
    stage2_tasks = output_root / "tasks" / "stage2_train.jsonl"
    synthesis = (
        _read_json(synthesis_path)
        if synthesis_path.exists() and stage2_tasks.exists()
        else synthesize_stage2(config, output_root)
    )
    stage2 = _completed_stage(output_root, "stage2") or train_stage(
        config,
        stage_name="stage2",
        train_tasks_path=stage2_tasks,
        eval_tasks_path=output_root / "tasks" / "final_holdout.jsonl",
        output_root=output_root,
        input_adapter=Path(stage1["adapter_dir"]),
    )
    verification = verify_run(config, output_root, replay=bool(config.get("fresh_replay", True)))
    result = {
        "status": "completed" if verification["overall_status"] == "PASS" else "failed_verification",
        "evidence_class": (
            "cloud_gpu_observed"
            if os.getenv("AGENTICQWEN_ORCHESTRATOR") in {"modal", "autodl"}
            and bool(stage2.get("runtime", {}).get("gpu"))
            else "gpu_observed"
        ),
        "execution": {
            "orchestrator": os.getenv("AGENTICQWEN_ORCHESTRATOR", "local"),
            "run_id": os.getenv("AGENTICQWEN_RUN_ID", output_root.name),
            "hostname": socket.gethostname(),
            "modal_task_id": os.getenv("MODAL_TASK_ID"),
            "modal_function_id": os.getenv("MODAL_FUNCTION_ID"),
            "autodl_instance_id": os.getenv("AUTODL_INSTANCE_ID"),
        },
        "config": str(config_path.resolve()),
        "output_root": str(output_root.resolve()),
        "model_snapshot": model_snapshot,
        "data": prepare,
        "stage1": stage1,
        "synthesis": synthesis,
        "stage2": stage2,
        "verification": verification,
    }
    _write_json(output_root / "run_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real two-stage multi-turn AgenticQwen GRPO curriculum")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("pipeline", "prepare", "synthesize", "verify", "evaluate"),
        default="pipeline",
    )
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--tasks-path", type=Path)
    parser.add_argument("--eval-output-dir", type=Path)
    parser.add_argument("--label", default="fresh_replay")
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    config = _read_json(config_path)
    if args.mode == "pipeline":
        result = run_pipeline(config_path, output_root)
    elif args.mode == "prepare":
        result = prepare_data(config, output_root)
    elif args.mode == "synthesize":
        result = synthesize_stage2(config, output_root)
    elif args.mode == "verify":
        result = verify_run(config, output_root, replay=not args.no_replay)
    else:
        required = {
            "--adapter-dir": args.adapter_dir,
            "--tasks-path": args.tasks_path,
            "--eval-output-dir": args.eval_output_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit(f"evaluate mode requires: {', '.join(missing)}")
        result = evaluate_adapter(
            config,
            adapter_dir=args.adapter_dir.absolute(),
            tasks_path=args.tasks_path.absolute(),
            output_dir=args.eval_output_dir.absolute(),
            label=args.label,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("overall_status", result.get("status", "completed")) not in {"FAIL", "failed_verification"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
