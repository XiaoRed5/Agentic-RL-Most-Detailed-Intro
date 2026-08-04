# AgenticQwen Reproduction

> **Evidence-first Agentic RL curriculum with Qwen3-8B.**  
> A real two-stage response-token QLoRA-GRPO run on a stateful multi-tool environment: train → diagnose failure/saturation → synthesize harder frontier tasks → retrain → reload the Stage-2 adapter in a fresh process.

**Deliverables:** `agenticqwen_report/TECHNICAL_REPORT.md` is the canonical report source and `agenticqwen_report/index.html` is its yyhdbl-style view; `slides/AgenticQwen_Curriculum_Cloud.pptx` is the project presentation; `artifacts/cloud_runs/qwen3-8b-qlora-20260804-v2/evidence/` contains compact, machine-readable cloud evidence.

## 🔥 Key Results

The cloud micro-run proves that the two-stage optimization loop, curriculum handoff, parameter updates, and fresh-process reload are real. It does **not** reproduce the paper's benchmark claim.

| Metric | Before | After | Δ / evidence |
|---|---:|---:|---:|
| Stage-1 frozen probe | 3 / 4 (75.0%) | 4 / 4 (100.0%) | +25.0 pt |
| Stage-2 untouched holdout | 4 / 6 (66.7%) | 6 / 6 (100.0%) | +33.3 pt |
| Stage-1 training rollouts | — | 40 / 48 success | reward std 0.5268 · 27 unique rewards |
| Stage-2 training rollouts | — | 46 / 48 success | reward std 0.2673 · 33 unique rewards |
| Optimizer steps | 12 (Stage 1) | 12 (Stage 2) | both trainable-parameter hashes changed |
| Fresh-process replay | — | 6 / 6 success | parent PID 5643 → child PID 6913 |

> **Failure → fix:** V1 produced 48/48 successes at the identical reward 1.3 in both stages, zero policy loss, and identical Stage-1/Stage-2 adapter hashes. V2 redesigned process rewards and stress profiles; the verifier then observed non-zero reward variance, non-zero policy losses, two parameter-changing stages, different adapter hashes, and a successful fresh-process replay.

## 🎯 Problem & Motivation

The AgenticQwen paper reports that AgenticQwen-8B improves the seven-task TAU-2 / BFCL-V4 average from **23.8 to 47.4** after multi-round agentic RL and dual data flywheels. Reproducing that claim requires much more than loading a Qwen model:

1. Multi-turn stateful user/tool interaction, not one-step classification.
2. Roughly 100K training examples and Round 0–3 data evolution.
3. Qwen3-235B as synthesizer, user simulator, tool simulator, and reward judge.
4. Official TAU-2 Avg@4 and BFCL-V4 exact-match evaluators.
5. Independent checkpoints, trajectories, failure analysis, and ablations.

This project separates those requirements into **COMPLETE**, **PARTIAL_RUN**, **CODE_READY**, and **BLOCKED_RESOURCE** instead of calling an unfinished scaffold a reproduction.

## 💡 Method

### What actually ran

- Official `Qwen/Qwen3-8B` base snapshot, 16 files / 16.40 GB, loaded with NF4 runtime quantization and BF16 compute.
- A seven-tool stateful refund environment with identity, order, payment, policy, exact confirmation, retry, and idempotent write constraints.
- Stage 1: 8 tasks, group size 4, 12 optimizer steps, 48 training rollouts, QLoRA on attention and MLP projections.
- Frozen curriculum probe: 3/4 before training and 4/4 after training.
- Deterministic frontier synthesis: 8 harder tasks with decoys/transient errors plus 4 Stage-1 replay tasks.
- Stage 2: 12 tasks, 12 optimizer steps, 48 training rollouts, starting from the Stage-1 adapter.
- Untouched final holdout: 4/6 before Stage 2 and 6/6 after Stage 2.
- Fresh-process reload: the saved Stage-2 adapter reproduced 6/6 successes in a separate child process.
- An earlier V1 run that failed integrity verification because reward saturation produced zero policy loss and identical Stage-1/Stage-2 adapter hashes.
- Additional local evidence: one DashScope `qwen3.7-flash` 8-turn/5-tool trajectory and an earlier one-token MLX LoRA-GRPO diagnostic. These are retained as supporting experiments, not confused with the cloud response-token run.

### What is deliberately scaled down

The primary cloud run does train full assistant response tokens and native tool calls, but it uses only 8 Stage-1 tasks, 8 generated hard tasks, 4 replay tasks, 6 final-holdout tasks, one seed, and 24 total optimizer steps. It validates the causal training/data-flywheel chain; it does not estimate paper-scale performance or statistical significance.

### Cloud curriculum path (real micro-run complete)

`curriculum_train.py` connects those two layers using TRL's stateful environment interface. Qwen3-8B generates assistant response tokens and native tool calls; the environment executes each call, preserves episode state, and returns an outcome-plus-process reward. The two-stage pipeline is:

```text
Stage 1 QLoRA-GRPO
  → frozen curriculum-probe rollouts
  → deterministic failure clustering
  → harder tasks with decoy orders / transient tool errors
  → replay mix
  → Stage 2 QLoRA-GRPO from Stage 1 adapter
  → untouched final holdout
  → fresh-process adapter reload
```

This small-scale curriculum ran on one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB). Qwen3-8B was loaded in NF4 at runtime and both stages trained LoRA adapters. `verification.json` is PASS across adapter existence/difference, parameter change, reward variance, Stage-2 consumption of Stage-1, split isolation, planned steps, synthesis provenance, complete traces, and fresh-process reload. The run is **COMPLETE at micro scale**, not a claim to the paper's 100K-example/eight-H100 recipe.

## 🧭 Long-Horizon Trajectory

The runnable mini-environment models a duplicate-charge refund:

```text
User request
  → ask for identity
  → lookup_customer
  → list_orders
  → get_payment_history
  → get_refund_policy
  → explicit user confirmation
  → create_refund with idempotency key
  → verifier reads final state
```

The model cannot mutate state directly. `long_horizon_env.py` validates identity, read-before-write ordering, the exact duplicate charge, refund amount, explicit confirmation, and idempotency. The completed artifact reports outcome reward `1.0`, process reward `0.85`, zero tool errors, and zero unsafe attempts.

## 🧪 Ablation System

The code includes a fixed 5×3×6 matrix:

| Experiment | Process reward | Credit assignment | Hypothesis |
|---|---:|---|---|
| Vanilla | 0.0 | `A/L` | exposes group saturation and length dilution |
| Turn-Discount | 0.0 | normalized early-turn weight | protects early planning |
| PRM-Lite | 0.3 | `A/L` | supplies local quality signals |
| LATA | 0.0 | `A/√L` | preserves long-response credit |
| Joint | 0.3 | `A/√L` | combines signal source and transmission path |

PRM-Lite implements 15 auditable rules. LATA and Turn-Discount kernels preserve response-turn boundaries and have unit tests.

### Offline PRM-Lite diagnostic

Re-scoring the existing 24 groups changed the number of active groups from **4 to 4**. This is a useful negative result: the single-step action task does not expose recovery, data-chain, read-diversity, or long-reasoning events. PRM-Lite must be evaluated in a real multi-turn rollout; offline re-scoring is not reported as an accuracy ablation.

## 📊 Official Benchmark Paths

### BFCL-V4 Multi-Turn

- Official `bfcl-eval==2026.3.23` package.
- `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`.
- Smoke profile: 5 deterministic IDs per category, 20 total.
- Paper profile: 200 per category, 800 total.
- Base and adapter use separate result/score roots.

### TAU-2

- Official repository pinned to `v0.2.0`.
- `airline`, `retail`, and `telecom`.
- Smoke profile: 5 tasks/domain × 1 trial, local Qwen3-8B user simulator.
- Paper profile: all tasks/domain × 4 trials, explicit external user simulator.
- Smoke results are interface checks and are never compared to paper Avg@4.

The evaluator code and dry-run manifest are complete, but no benchmark score is claimed in this version.

## 🏗️ Project Structure

```text
agenticqwen-reproduction/
├── configs/
│   ├── real_qwen3_8b.json
│   ├── curriculum_qwen3_8b.json
│   ├── long_horizon_demo.json
│   ├── benchmarks.json
│   └── ablation_matrix.json
├── src/agentic_repro/
│   ├── real_grpo.py
│   ├── curriculum_env.py
│   ├── curriculum_train.py
│   ├── long_horizon_env.py
│   ├── dashscope_policy.py
│   ├── trajectory_runner.py
│   ├── verify_real.py
│   ├── benchmark_runner.py
│   ├── ablations.py
│   ├── blog_report.py
│   └── blog_renderer.py
├── artifacts/
│   ├── real_qwen3_8b/
│   ├── long_horizon/
│   ├── ablations/
│   └── benchmarks/
├── docs/
│   ├── benchmark_protocol.md
│   ├── experiment_design.md
│   ├── failure_diagnosis.md
│   ├── curriculum_protocol.md
│   └── claims_and_evidence.md
├── tests/
├── run_real_qwen3_8b.sh
├── run_curriculum_modal.sh
├── modal_curriculum.py
├── run_long_horizon_demo.sh
├── run_ablations.sh
├── run_benchmarks.sh
└── agenticqwen_report/
    ├── TECHNICAL_REPORT.md
    └── index.html
```

## 🚀 Quick Start

### Rebuild the completed local run

```bash
./run_real_qwen3_8b.sh
```

This runs training, independent verification, offline reward diagnosis, benchmark dry-run, tests, HTML generation, and PPT generation. If `DASHSCOPE_API_KEY` is present it also refreshes the multi-turn trajectory; otherwise it reuses the checked-in verified trajectory artifact.

### Re-run the real multi-turn trajectory

```bash
DASHSCOPE_API_KEY=<rotated-key> ./run_long_horizon_demo.sh
```

The key is read only from the process environment and is rejected if it appears in the trajectory artifact.

### Run the two-stage Qwen3-8B curriculum on AutoDL

Upload or sync the project to `/root/autodl-tmp/agenticqwen-reproduction`, then run:

```bash
./run_curriculum_autodl_remote.sh
```

The launcher reuses the retained standard Qwen3-8B snapshot, loads it as NF4 at runtime, trains LoRA adapters, resumes Trainer checkpoints, saves Stage-1/Stage-2 traces and hashes, and refuses to turn a missing run into a completed report.

### Modal alternative

```bash
../../work/modalenv/bin/modal setup
TRAINING_GPU=A100-80GB MAX_BUDGET_USD=15 ESTIMATED_MINUTES=180 \
  TRAINING_TIMEOUT_MINUTES=210 BENCHMARK_MINUTES=35 ./run_curriculum_modal.sh
```

The launcher defaults to an A100-80GB for training and an L40S for official BFCL smoke; refuses an estimate above the supplied budget; has hard timeouts; persists model and result volumes; resumes Trainer checkpoints; downloads the completed run; and then relies on `verification.json` to decide whether optimizer steps, adapter chaining, split isolation, synthesis provenance, fresh reload, and official evaluator outputs actually passed. Use `TRAINING_GPU=H100 ESTIMATED_MINUTES=120 TRAINING_TIMEOUT_MINUTES=145` for the faster path. See `docs/curriculum_protocol.md`.

### Validate code without model inference

```bash
./run_ablations.sh
DRY_RUN=1 ./run_benchmarks.sh
../../work/mlxenv312/bin/python -m pytest -q
```

### Install and run official evaluator smoke tests

```bash
./scripts/setup_benchmarks.sh
PROFILE=smoke VARIANTS=base,adapter ./run_benchmarks.sh
```

### Full benchmark profile

```bash
TAU2_USER_MODEL=openai/<model> \
TAU2_USER_API_BASE=https://<endpoint>/v1 \
TAU2_USER_API_KEY=<key> \
PROFILE=paper ./run_benchmarks.sh
```

## ✅ Completion Summary

| Component | Status | Evidence boundary |
|---|---|---|
| Stage-1 response-token QLoRA-GRPO | COMPLETE (micro) | 12 steps; reward std 0.5268; trainable hash changed |
| V1 failure diagnosis → V2 repair | COMPLETE | zero-variance/identical-hash failure preserved; V2 verifier PASS |
| Frontier hard-task synthesis + replay | COMPLETE (micro) | frozen trace hash → 8 hard + 4 replay tasks |
| Stage-2 continuation from Stage 1 | COMPLETE (micro) | input hash matches Stage 1; output hash differs |
| Untouched holdout + fresh reload | COMPLETE (micro) | 4/6 → 6/6; new-process replay 6/6 |
| BFCL-V4 / TAU-2 | CODE_READY / NOT_RUN | no official score is claimed |
| Three seeds and online ablations | NOT_RUN | required for statistical and mechanism claims |
| ≈100K / 235B / 8×H100 paper recipe | BLOCKED_RESOURCE | outside this reproduction budget |

`docs/claims_and_evidence.md` is the current claim ledger. The older
`artifacts/real_qwen3_8b/completion_matrix.json` describes the retained local
one-token diagnostic only; it is not the ledger for the cloud curriculum run.

## 🧾 Claim Boundary

- Supported: a real Qwen3-8B LoRA-GRPO update occurred and replays from disk.
- Supported: a real Qwen3.7-Flash policy completed a stateful 8-turn, 5-tool trajectory with 9/9 verifier checks.
- Supported: same-task prompt holdout improved on this tiny sample.
- Supported: a real two-stage multi-turn response-token QLoRA-GRPO run completed on a cloud GPU.
- Supported: V1 saturation was caught by the verifier; V2 produced non-zero reward variance and changed trainable parameters in both stages.
- Supported: the untouched six-task holdout improved from 4/6 to 6/6 in this single micro-run and replayed 6/6 after a fresh reload.
- Not supported: unseen-task generalization improved.
- Not supported: PRM-Lite solved saturation in this environment.
- Not tested: official BFCL-V4 / TAU-2 scores or the paper's 47.4 average benchmark score.
- Not tested: the paper's ≈100K, Qwen3-235B, Round 0–3 data flywheel.

## 📚 References

- [AgenticQwen paper](https://arxiv.org/abs/2604.21590)
- [Official AgenticQwen collection](https://huggingface.co/collections/alibaba-pai/agenticqwen)
- [Official data synthesis / RL code](https://github.com/haruhi-sudo/data_synth_and_rl)
- [Official TAU-2 evaluator](https://github.com/sierra-research/tau2-bench/tree/v0.2.0)
- [Official BFCL evaluator](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [Agentic-GRPO-LongHorizon style and diagnosis reference](https://github.com/qiqihezh/agentic-grpo-longhorizon)
- [DashScope Qwen Function Calling](https://help.aliyun.com/en/model-studio/qwen-function-calling)
