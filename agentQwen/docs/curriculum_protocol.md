# Two-stage multi-turn Qwen3-8B curriculum protocol

## What this experiment tests

This experiment tests whether a small response-token Agentic RL run can close a measurable failure loop:

1. train Qwen3-8B with grouped policy optimization in a stateful tool environment;
2. evaluate on a frozen curriculum probe;
3. classify terminal failures from environment state rather than model prose;
4. synthesize harder but verifier-grounded tasks from those categories;
5. continue training the same adapter with hard-task/replay mixing;
6. evaluate on an untouched final holdout and reload the saved adapter in a fresh process.

It does not claim to reproduce the paper's data volume, Qwen3-235B synthesis stack, eight-H100 training recipe, or reported benchmark average.

## Environment and action space

`StatefulRefundEnvironment` exposes seven typed tools:

- `request_identity`
- `lookup_customer`
- `list_orders`
- `get_payment_history`
- `get_refund_policy`
- `request_confirmation`
- `create_refund`

TRL constructs one environment per rollout, forwards the dataset's `task_json` and `stage` fields into `reset`, and registers the public methods as tools. The model generates ordinary assistant response tokens plus structured tool calls. Tool observations are appended to the same conversation until completion or the ten-call limit.

The irreversible write is accepted only when identity, orders, payment evidence, policy, exact confirmation, duplicate charge ID, amount, reason, and idempotency key all pass deterministic checks.

## Reward

The environment returns:

```text
combined_reward = outcome_success + 0.30 × process_score
```

`outcome_success` is one only when the final refund state is exactly correct. `process_score` rewards verified progress and recovery from a configured transient error, while penalizing unsafe writes, schema mistakes, redundant calls, and long loops. The group-relative baseline is still computed across four sampled rollouts; process variation is intended to reduce the all-zero/all-one saturation observed in the local one-token run.

## Curriculum construction

Stage 1 uses eight deterministic training tasks and four separate probe tasks. Probe traces are classified into:

- identity failure;
- orders/payment/policy not read;
- missing confirmation;
- wrong charge or amount;
- transient tool error not recovered;
- timeout/loop;
- refund not created.

Stage 2 creates eight new tasks from the observed categories and mixes four Stage 1 replay tasks. Hard tasks add multiple decoy orders, tighter language, and category-specific transient tool failures. The generator may change surface form later, but IDs, amounts, transitions, and verifier targets must remain deterministic code outputs; model output is never trusted as ground truth.

The final holdout contains six distinct task IDs, two decoy orders per task, unseen wording, and policy-service failures on a subset. Its IDs are hashed before training and rejected if they overlap either training stage.

## Training recipe

- Base: `Qwen/Qwen3-8B`
- Quantization: NF4 4-bit with double quantization
- Adapter: rank-16 LoRA over Q/K/V/O and MLP projections
- Optimizer: fused AdamW
- Objective: DAPO-style GRPO loss, group size four
- Precision: bfloat16 with TF32 and gradient checkpointing
- Stage 1: 12 optimizer steps at `5e-6`
- Stage 2: 12 optimizer steps at `3e-6`
- Generation: temperature 0.8, top-p 0.9, maximum 1,400 completion tokens and ten tool iterations

The small step count is a budgeted integration experiment. It is enough to establish whether response-token gradients, adapter persistence, failure mining, curriculum data flow, and fresh reload work; it is not enough to establish paper-level performance.

## Cloud and recovery

The Modal launcher defaults to one A100-80GB for training and one L40S for the sequential official BFCL smoke run, each with four physical CPU cores and 48 GiB RAM. The model, algorithm, steps, generation limits, and benchmark remain unchanged. A local cost guard estimates three training hours plus 35 benchmark minutes at about USD 10.68 and refuses a request above `MAX_BUDGET_USD`; the cloud functions have 210-minute and 50-minute hard timeouts. Set `TRAINING_GPU=H100 ESTIMATED_MINUTES=120 TRAINING_TIMEOUT_MINUTES=145` for the faster path, or `TRAINING_GPU=L40S` for a lower hourly rate with greater runtime/OOM risk. A Modal workspace spending limit remains the authoritative cost control.

Model cache, Trainer checkpoints, adapters, traces, and summaries live in named persistent volumes. On retry, a stage with a completed adapter is skipped; an interrupted stage resumes from the largest `checkpoint-N`, and its prior training trace is retained.

## Acceptance gates

`verification.json` passes only when:

1. both adapter directories exist;
2. both stages recorded a positive optimizer step count;
3. Stage 2 started from the exact Stage 1 adapter tree hash;
4. Stage 1 and Stage 2 adapter hashes differ;
5. final-holdout IDs never enter training;
6. hard tasks record the exact Stage 1 trace hash and synthesis provenance;
7. the Stage 2 adapter loads in a fresh process and produces scored holdout episodes.

Performance deltas are reported separately from integrity gates. A run can be technically valid while failing to improve success rate; that is a real negative result, not grounds to hide or relabel it.

## Run command

```bash
../../work/modalenv/bin/modal setup
TRAINING_GPU=A100-80GB MAX_BUDGET_USD=15 ESTIMATED_MINUTES=180 \
  TRAINING_TIMEOUT_MINUTES=210 BENCHMARK_MINUTES=35 ./run_curriculum_modal.sh
```

Downloaded artifacts are stored under `artifacts/cloud_curriculum/<run-id>/`. Do not report a curriculum result until the remote `run_summary.json`, both stage summaries, trace JSONL files, synthesis manifest, adapters, and `verification.json` are present locally.
