from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def build_plan(config: dict, project_dir: Path) -> dict:
    run = config["run"]
    plan = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper": config["paper"],
        "reproduction_claim": {
            "target": "AgenticQwen agentic data flywheel and agentic tool-use behavior",
            "local_scope": "structural smoke RL plus real Qwen3-8B inference",
            "excluded": "paper-scale 8B/30B RL training and benchmark-number replication",
        },
        "dag": [
            {
                "id": "P0",
                "owner": "planner",
                "task": "Lock paper identity, method components, evidence levels and gates",
                "depends_on": [],
                "outputs": ["artifacts/plan.json", "artifacts/paper_spec.json"],
                "acceptance": ["arXiv ID is 2604.21590", "claim boundary is explicit"],
            },
            {
                "id": "E1",
                "owner": "executor",
                "task": "Generate executable linear, branch and adversarial airline tasks",
                "depends_on": ["P0"],
                "outputs": ["artifacts/tasks.jsonl"],
                "acceptance": ["levels 0, 1 and 2 present", "each task has a verifiable target action"],
            },
            {
                "id": "E2",
                "owner": "executor",
                "task": "Run multi-round GRPO-style smoke policy optimization",
                "depends_on": ["E1"],
                "outputs": ["artifacts/metrics.json", "artifacts/smoke_policy.json"],
                "acceptance": ["reward in [0,1]", "fixed seed", "round metrics recorded"],
            },
            {
                "id": "E3",
                "owner": "executor",
                "task": "Run local Qwen3-8B tool-use rollouts through MLX",
                "depends_on": ["E1"],
                "outputs": ["artifacts/qwen3_8b_inference.json"],
                "acceptance": ["actual model identifier recorded", "raw outputs and tool events retained"],
            },
            {
                "id": "V1",
                "owner": "verifier",
                "task": "Audit artifacts, determinism, reward semantics, inference provenance and gates",
                "depends_on": ["E2", "E3"],
                "outputs": ["artifacts/verification.json"],
                "acceptance": ["no hidden cloud fallback", "all hard gates pass"],
            },
            {
                "id": "R1",
                "owner": "reporter",
                "task": "Build evidence-linked HTML report and presentation",
                "depends_on": ["V1"],
                "outputs": ["agenticqwen_report/index.html", "slides/AgenticQwen_LongHorizon_Lab.pptx"],
                "acceptance": ["paper results and local results are visually separated", "sources traceable"],
            },
        ],
        "runtime": {
            "seed": run["seed"],
            "rounds": run["rounds"],
            "group_size": run["group_size"],
            "max_steps": run["max_steps"],
        },
    }
    return plan


def write_plan(config: dict, project_dir: Path) -> Path:
    artifacts = project_dir / config["paths"]["artifacts"]
    artifacts.mkdir(parents=True, exist_ok=True)
    path = artifacts / "plan.json"
    path.write_text(json.dumps(build_plan(config, project_dir), ensure_ascii=False, indent=2), encoding="utf-8")
    spec = {
        "paper_claims": [
            "reasoning RL uses final-answer verifiability",
            "agentic RL uses rubric-completion reward in [0,1]",
            "linear workflows are expanded into behavior trees",
            "branch-to-task inversion turns a chosen branch into state, user and agent instructions",
            "adversarial users push the policy toward an incorrect branch",
            "paper reports about 100K training samples and three flywheel rounds",
        ],
        "local_mapping": {
            "mock environment": "deterministic airline state machine",
            "behavior tree": "available / sold-out / delayed membership branches",
            "rubric reward": "five executable subgoals",
            "GRPO": "group-normalized reward policy gradient on a transparent smoke policy",
            "policy inference": "Qwen3-8B 4-bit via MLX, no cloud endpoint",
        },
        "non_equivalences": [
            "the smoke policy is not a language model",
            "Qwen3-8B is used for inference, not locally RL-trained",
            "the local task suite is diagnostic and not TAU-2/BFCL-V4",
        ],
    }
    (artifacts / "paper_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
