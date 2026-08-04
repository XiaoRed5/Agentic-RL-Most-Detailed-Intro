from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal


PROJECT_DIR = Path(__file__).resolve().parent
REQUIREMENTS = [
    line.strip()
    for line in (PROJECT_DIR / "requirements-cloud.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
BENCHMARK_REQUIREMENTS = [
    line.strip()
    for line in (PROJECT_DIR / "requirements-benchmark-cloud.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
BFCL_REQUIREMENTS = [
    line.strip()
    for line in (PROJECT_DIR / "requirements-bfcl-cloud.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

app = modal.App("agenticqwen-curriculum-grpo")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*REQUIREMENTS)
    .add_local_dir(PROJECT_DIR, remote_path="/workspace/project", copy=False)
)
benchmark_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("git", "curl")
    .uv_pip_install(*BENCHMARK_REQUIREMENTS)
    .run_commands(
        "python -m venv /opt/bfcl-venv",
        "/opt/bfcl-venv/bin/pip install --no-cache-dir " + " ".join(BFCL_REQUIREMENTS),
    )
    .add_local_dir(PROJECT_DIR, remote_path="/workspace/project", copy=False)
)
model_cache = modal.Volume.from_name("agenticqwen-model-cache", create_if_missing=True)
result_volume = modal.Volume.from_name("agenticqwen-curriculum-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    timeout=60 * 210,
    volumes={"/models": model_cache, "/results": result_volume},
)
def train(run_id: str) -> str:
    os.environ.update(
        {
            "HF_HOME": "/models/huggingface",
            "HF_HUB_CACHE": "/models/huggingface/hub",
            "TRANSFORMERS_CACHE": "/models/huggingface/hub",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": "/workspace/project/src",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "AGENTICQWEN_ORCHESTRATOR": "modal",
            "AGENTICQWEN_RUN_ID": run_id,
        }
    )
    output_root = Path("/results") / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "agentic_repro.curriculum_train",
        "--config",
        "/workspace/project/configs/curriculum_qwen3_8b.json",
        "--output-root",
        str(output_root),
        "--mode",
        "pipeline",
    ]
    subprocess.run(command, cwd="/workspace/project", check=True)
    model_cache.commit()
    result_volume.commit()
    summary_path = output_root / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return json.dumps(
        {
            "status": summary["status"],
            "run_id": run_id,
            "result_volume": "agenticqwen-curriculum-results",
            "remote_path": f"/{run_id}",
            "verification": summary["verification"]["overall_status"],
            "stage1_after": summary["stage1"]["failure_summary_after"],
            "stage2_after": summary["stage2"]["failure_summary_after"],
        },
        ensure_ascii=False,
    )


@app.function(
    image=benchmark_image,
    gpu="L40S",
    cpu=4,
    memory=49152,
    timeout=60 * 50,
    volumes={"/models": model_cache, "/results": result_volume},
)
def benchmark(run_id: str, tasks_per_category: int = 2) -> str:
    os.environ.update(
        {
            "HF_HOME": "/models/huggingface",
            "HF_HUB_CACHE": "/models/huggingface/hub",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": "/workspace/project/src",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "AGENTICQWEN_ORCHESTRATOR": "modal",
            "AGENTICQWEN_RUN_ID": run_id,
        }
    )
    output_root = Path("/results") / run_id
    command = [
        sys.executable,
        "-m",
        "agentic_repro.cloud_bfcl",
        "--run-root",
        str(output_root),
        "--tasks-per-category",
        str(tasks_per_category),
        "--model-cache",
        "/models/huggingface/hub",
        "--bfcl-python",
        "/opt/bfcl-venv/bin/python",
        "--bfcl-bin",
        "/opt/bfcl-venv/bin/bfcl",
    ]
    subprocess.run(command, cwd="/workspace/project", check=True)
    model_cache.commit()
    result_volume.commit()
    manifest = json.loads(
        (output_root / "benchmarks" / "bfcl_smoke" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return json.dumps(
        {
            "status": manifest["overall_status"],
            "run_id": run_id,
            "scope": manifest["scope"],
            "tasks_per_category": manifest["tasks_per_category"],
            "variants": {
                name: {
                    "status": value["status"],
                    "episodes": value["result_episodes"],
                }
                for name, value in manifest["variants"].items()
            },
        },
        ensure_ascii=False,
    )


@app.local_entrypoint()
def main(
    run_id: str = "",
    max_budget_usd: float = 15.0,
    training_gpu: str = "A100-80GB",
    estimated_minutes: int = 180,
    training_timeout_minutes: int = 210,
    benchmark_minutes: int = 35,
    run_benchmark: bool = True,
):
    # GPUs are billed per second. This guard is deliberately conservative and
    # supplements, rather than replaces, the workspace spending limit.
    # Current public rates plus 4 physical cores and 48 GiB RAM.
    gpu_hourly = {
        "L40S": 1.9512,
        "A100-40GB": 2.0988,
        "A100-80GB": 2.4984,
        "H100": 3.9492,
    }
    training_gpu = training_gpu.upper()
    if training_gpu not in gpu_hourly:
        raise SystemExit(
            f"Unsupported --training-gpu {training_gpu}; choose one of {sorted(gpu_hourly)}"
        )
    if training_timeout_minutes < estimated_minutes:
        raise SystemExit("--training-timeout-minutes must be >= --estimated-minutes")
    cpu_memory_hourly = 4 * 0.04716 + 48 * 0.007992
    estimated_hourly = gpu_hourly[training_gpu] + cpu_memory_hourly
    benchmark_hourly = gpu_hourly["L40S"] + cpu_memory_hourly
    estimated_cost = estimated_hourly * estimated_minutes / 60.0
    if run_benchmark:
        estimated_cost += benchmark_hourly * benchmark_minutes / 60.0
    if estimated_cost > max_budget_usd:
        raise SystemExit(
            f"Refusing launch: estimated ${estimated_cost:.2f} exceeds --max-budget-usd ${max_budget_usd:.2f}"
        )
    if not run_id:
        run_id = "qwen3-8b-curriculum-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    print(
        json.dumps(
            {
                "cost_guard": {
                    "gpu": training_gpu,
                    "estimated_minutes": estimated_minutes,
                    "benchmark_gpu": "L40S 48GB" if run_benchmark else None,
                    "benchmark_minutes": benchmark_minutes if run_benchmark else 0,
                    "estimated_cost_usd": round(estimated_cost, 2),
                    "max_budget_usd": max_budget_usd,
                    "hard_timeout_minutes": training_timeout_minutes,
                },
                "run_id": run_id,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    result = train.with_options(
        gpu=training_gpu,
        timeout=60 * training_timeout_minutes,
    ).remote(run_id)
    print(result)
    if run_benchmark:
        benchmark_result = benchmark.remote(run_id, 2)
        print(benchmark_result)
    print(
        "Download with: modal volume get agenticqwen-curriculum-results "
        f"/{run_id} ./artifacts/cloud_curriculum/{run_id}"
    )
