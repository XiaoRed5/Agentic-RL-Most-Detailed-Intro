# AgenticQwen Reproduction

![失败驱动的 Agentic RL 课程学习闭环](agenticqwen_report/images/figure4_curriculum_stage_flow.png)

> **工业级执行入口：** [`run_industrial_agenticqwen.sh`](run_industrial_agenticqwen.sh) 是面向官方
> `haruhi-sudo/data_synth_and_rl` 的可恢复控制平面：官方数据生成 → teacher/mock-user
> solve → rubric filter → verl parquet → SGLang multi-turn GRPO → 独立评测。它不会把
> 4-bit 单步 action demo 或 deterministic contract backend 冒充成论文训练。

工业控制器默认指向 `configs/official_ds_v4_bounded10_*.yaml`。**最多 10 条只约束新合成的困难任务，不约束官方已发布数据**；官方 parquet 作为主数据池，可按预算选取课程层级或扩展到完整训练。

## ✅ 官方数据 + 官方 verl/SGLang 两轮实跑

这条路径直接使用 [`haruhi-sudo/data_synth_and_rl`](https://github.com/haruhi-sudo/data_synth_and_rl) 的 RL 代码和已发布 parquet，而不是自建退款 toy environment。下载并校验的官方数据池有 **37,401** 行、4,782 个 base task；本次单卡 H200 先从中选择可学习的单工具课程层级，再根据真实失败追加 1 条困难任务。`≤10` 的合成预算没有限制官方 replay。

| 环节 | 数据与执行 | 可审计结果 |
|---|---|---|
| Round 0 | 2 条官方训练任务 + 1 条 group-disjoint 官方 holdout；Qwen3-8B、BF16、FSDP、SGLang async、GRPO | 8 条 rollout，1 成功 / 7 失败；reward mean 0.125；`grad_norm=1.484375`；审计 **PASS** |
| 失败审计 | 将 rollout 精确映射回官方任务和预期工具路径 | 识别出参数改写后不匹配、过早转人工、重复工具调用、缺少终止标记 |
| 定向合成 | DeepSeek-v4-flash 替代论文 Qwen3-235B，做 branch-to-task inversion 和 teacher-solved/branch-hit gate | 请求 2 条，保留 1 条；另一条因没有新增行为分支被拒；合成用量 1/10 |
| Round 1 | 2 条官方 replay + 1 条门控困难任务；从 Round-0 合并权重继续 GRPO | 12 条 rollout，4 成功 / 8 失败；reward mean 0.3333；`grad_norm=1.2734375`；审计 **PASS** |
| 独立交付 | 合并 FSDP checkpoint，逐分片比较 Stage 1/2，另起 Python 进程重载 | 4/4 权重分片哈希变化；H200 fresh reload **PASS**；隔离任务首个工具调用正确 |
| 泛化边界 | 固定官方 holdout 的完整多轮 reward | 0/1；**不声称性能提升或论文 benchmark 复现** |

机器可读证据位于 [`artifacts/official_agenticqwen_h200/`](artifacts/official_agenticqwen_h200/)：两个 round 的 dataset manifest、rollout、reward/gradient 审计、教师筛选清单、分片哈希和 fresh-process reload 均已落盘。训练使用的是完整 Qwen3-8B BF16/FSDP 权重更新；它与下文较早完成的 NF4/LoRA 退款课程实验是两条独立证据链。

## 🚀 AutoDL 一键实验 Skill

本项目把完整的云端实验操作沉淀为 [`autodl-agentic-rl`](skills/autodl-agentic-rl/SKILL.md)。它覆盖 GPU 预算与实例确认、代码和模型准备、CUDA/TRL/bitsandbytes 预检、可恢复训练、状态监控、结果归档、SHA-256 校验和安全关机，可直接复用于 QLoRA-GRPO、SFT、GRPO 与 curriculum 实验。

配套资源包括：

- `references/autodl-browser.md`：AutoDL 浏览器操作和敏感信息边界；
- `references/job-contract.md`：任务契约、终态和恢复协议；
- `scripts/prepare_job.py`：生成 payload、launch/status/collect 四件套；
- `scripts/model_fetch.py`、`remote_preflight.py`、`package_results.py`：模型下载、远端预检和结果安全打包。
- `scripts/monitor-codelab-bfcl.sh`：只读监控 CodeLab BFCL 进程、GPU、结果文件和 score；控制面失联时返回明确的非成功状态。

## 🌳 AgenticQwen 原文行为树实现

仓库现在新增了一条与旧退款课程实验隔离的原文算法路径：从 SynthAgent 风格的线性 happy path 出发，每轮先训练小策略并在 mock user / stateful tool 环境里生成真实 rollout，再由强模型扩展行为树；随后对每个分支执行 `b → (environment, user, agent)` 反演，生成 `normal_path`、`hack_path` 和 `[0,1]` objective rubric，只有“强模型能解出且轨迹命中预设分支”的候选才进入下一轮 GRPO。

核心入口：

- [`paper_flywheel.py`](src/agentic_repro/paper_flywheel.py)：行为树扩展、分支反演、Qwen3-235B 接口和教师过滤；
- [`paper_flywheel_env.py`](src/agentic_repro/paper_flywheel_env.py)：mock user、状态化工具、normal/hack path 与客观奖励；
- [`paper_grpo_train.py`](src/agentic_repro/paper_grpo_train.py)：Round 0–3 的“训练 → rollout → 树扩展 → 再训练”交替编排；
- [`agenticqwen_paper_micro.json`](configs/agenticqwen_paper_micro.json)：Qwen3-8B NF4 + LoRA-GRPO 微型配置；
- [`原文算法逐项对齐说明`](docs/agenticqwen_paper_implementation.md)：实现映射、运行方式和未完成边界。
- [`CodeLab 独立验证证据`](artifacts/agenticqwen_paper_micro/codelab_verification.json)：Python 3.10 全量测试与三轮证据哈希复核。

真实云端 profile 默认启用合成预算硬门：`max_synthetic_trajectories=10`，当前每轮最多新增 1 条（4 条 Round-0 种子 + 3 轮最多 1 条 = 7 条；回放旧任务不计为新合成）。教师候选必须通过 solved + branch-hit 校验才会落盘；全候选被拒时会保存 `flywheel/round_N/teacher_validation_rejected.jsonl` 和 `rejection_summary.json`，状态保持 PARTIAL，不会把失败候选当成训练数据。

无需 GPU 的算法契约检查：

```bash
./run_agenticqwen_paper.sh
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_paper_flywheel.py' -v
```

契约模式明确使用 `deterministic-contract-backend`，用于验证算法不变量，不能冒充 Qwen3-235B 生成结果。真正训练模式要求把配置中的 teacher 切换为本地 vLLM/SGLang 的 Qwen3-235B endpoint，然后执行 `MODE=train ./run_agenticqwen_paper.sh`。本项目对微型实现与论文 100K 数据/分布式规模严格分开陈述。

> **Evidence-first Agentic RL curriculum with Qwen3-8B.**  
> A real two-stage response-token QLoRA-GRPO run on a stateful multi-tool environment: train → diagnose failure/saturation → synthesize harder frontier tasks → retrain → reload the Stage-2 adapter in a fresh process.

**Deliverables:** [`TECHNICAL_REPORT.md`](agenticqwen_report/TECHNICAL_REPORT.md) is the canonical report source, [`index.html`](agenticqwen_report/index.html) is its yyhdbl-style view, and [`AgenticQwen_Curriculum_Cloud.pptx`](slides/AgenticQwen_Curriculum_Cloud.pptx) is the project presentation. [`artifacts/cloud_codelab/`](artifacts/cloud_codelab/) contains the machine-readable H200 run, model SHA gate, fresh replay, and paper-flywheel audit.

## 🧭 失败驱动的 Agentic RL 数据飞轮

项目先在基础退款任务上学习受约束的线性工具链，再由轨迹审计器读取真实环境状态与事件账本，将失败归因到缺失读取、身份核对、临时错误恢复、用户确认、写入时机和幂等控制等具体环节。数据合成器据此增加相似订单、干扰订单、一次性超时和确认门等压力条件，生成带分支的困难工作流，并进入下一阶段持续训练。

## 🔥 Key Results

The cloud micro-run proves that the two-stage optimization loop, curriculum handoff, parameter updates, and fresh-process reload are real. It does **not** reproduce the paper's benchmark claim.

| Metric | Before | After | Δ / evidence |
|---|---:|---:|---:|
| Stage-1 frozen probe | 3 / 4 (75.0%) | 4 / 4 (100.0%) | +25.0 pt |
| Stage-2 untouched holdout | 6 / 6 (100.0%) | 6 / 6 (100.0%) | saturated; no claim of gain |
| Stage-1 training rollouts | — | 36 / 48 success | reward std 0.6144 · 20 unique rewards |
| Stage-2 training rollouts | — | 45 / 48 success | reward std 0.3297 · 36 unique rewards |
| Optimizer steps | 12 (Stage 1) | 12 (Stage 2) | both trainable-parameter hashes changed |
| Fresh-process replay | — | 6 / 6 success | independent child process; verification PASS |

> **Failure → fix:** the real H200 run keeps the earlier saturation diagnostic as a negative control. In the completed curriculum, the verifier observed non-zero reward variance, non-zero policy losses, two parameter-changing stages, different adapter hashes, and a successful fresh-process replay. The final holdout was already saturated before Stage 2, so this run does not claim a holdout improvement.

### Paper-style flywheel status

官方路径现已完成 Round 0 → 失败审计 → 定向合成 → Round 1 的一次真实闭环。DS-v4-flash 曾出现 60/180 秒尾延迟，但重试和落盘审计使本轮完成；因此它是稳定性风险，不再是“完全阻塞”。官方 `verl`、SGLang、FSDP、reward function、mock user/tool 均已在 H200 实际执行。

本次 active curriculum 只消耗 3 条官方 variant 和 1 条新合成任务，目的是先验证梯度、检查点交接和失败驱动数据流。37,401 行官方池已经下载、校验并可供扩展，但没有把“数据可用”写成“全量训练已完成”。

| Paper path gate | Status |
|---|---|
| 官方数据池下载、schema、去重与 group-disjoint split | PASS（37,401 行） |
| Qwen3-8B Round 0 / Round 1 verl-GRPO 与非零梯度 | PASS |
| 失败归因、branch inversion、teacher solve + branch-hit | PASS（保留 1，拒绝 1） |
| Stage 1 → Stage 2 权重交接、分片变化、fresh reload | PASS |
| 固定 holdout 提升 | NOT OBSERVED（0/1） |
| Round 2/3、官方全量训练、TAU-2/BFCL 论文分数 | NOT RUN |

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
- Untouched final holdout: 6/6 before Stage 2 and 6/6 after Stage 2 (saturated).
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

This small-scale curriculum ran on one NVIDIA H200 (139.7 GiB visible memory). Qwen3-8B was loaded in NF4 at runtime and both stages trained LoRA adapters. `verification.json` is PASS across adapter existence/difference, parameter change, reward variance, Stage-2 consumption of Stage-1, split isolation, planned steps, synthesis provenance, complete traces, and fresh-process reload. The run is **COMPLETE at micro scale**, not a claim to the paper's 100K-example/eight-H100 recipe.

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
- Smoke profile: 1 deterministic ID per category, 4 total（云端已实际运行）。
- Paper profile: 200 per category, 800 total.
- Base and adapter use separate result/score roots.

云端 `bfcl-eval==2026.3.23` 已完成 base 与 Stage-2 adapter 的四类 multi-turn smoke。官方生成、执行与评分 artifact 均齐全，流水线审计为 **PASS**；但这不是质量通过：两个 variant 的 `Overall Acc` / `Multi Turn Acc` 均为 **0.00%**。因此这里能写“官方 smoke pipeline 跑通”，不能写“BFCL 已提升”或“论文 benchmark 已复现”。

### TAU-2

- Official repository pinned to `v0.2.0`.
- `airline`, `retail`, and `telecom`.
- Smoke profile: 5 tasks/domain × 1 trial, local Qwen3-8B user simulator.
- Paper profile: all tasks/domain × 4 trials, explicit external user simulator.
- Smoke results are interface checks and are never compared to paper Avg@4.

The evaluator code and cloud smoke artifacts are complete. The smoke score is reported as observed (base/adapter both 0.00%); no paper-scale benchmark score or improvement is claimed.

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
├── skills/
│   └── autodl-agentic-rl/   # 一键租卡→上传→训练→收集→关机
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
| Stage-1 response-token QLoRA-GRPO | COMPLETE (micro) | 12 steps; reward std 0.6144; trainable hash changed |
| V1 failure diagnosis → V2 repair | COMPLETE | zero-variance/identical-hash failure preserved; V2 verifier PASS |
| Frontier hard-task synthesis + replay | COMPLETE (micro) | frozen trace hash → 8 hard + 4 replay tasks |
| Stage-2 continuation from Stage 1 | COMPLETE (micro) | input hash matches Stage 1; output hash differs |
| Untouched holdout + fresh reload | COMPLETE (micro) | 6/6 → 6/6 (saturated); fresh-process replay 6/6 |
| Paper-style behavior-tree flywheel | PARTIAL_RUN | Round 0/1 complete; Round 2 blocked by teacher timeout/truncation |
| BFCL-V4 multi-turn smoke | PASS (pipeline) / 0.00% model score | official 4-episode base + adapter artifacts |
| BFCL-V4 paper profile / TAU-2 | CODE_READY / NOT_RUN | no paper-scale score is claimed |
| Three seeds and online ablations | NOT_RUN | required for statistical and mechanism claims |
| ≈100K / 235B / 8×H100 paper recipe | BLOCKED_RESOURCE | outside this reproduction budget |

`docs/claims_and_evidence.md` is the current claim ledger. The older
`artifacts/real_qwen3_8b/completion_matrix.json` describes the retained local
one-token diagnostic only; it is not the ledger for the cloud curriculum run.

## 🧾 Claim Boundary

- Supported: a real Qwen3-8B LoRA-GRPO update occurred and replays from disk.
- Supported: a real Qwen3.7-Flash policy completed a stateful 8-turn, 5-tool trajectory with 9/9 verifier checks.
- Supported: the Stage-1 frozen probe improved from 3/4 to 4/4; the Stage-2 final holdout was already 6/6 before training and stayed 6/6.
- Supported: a real two-stage multi-turn response-token QLoRA-GRPO run completed on a cloud GPU.
- Supported: saturation is preserved as a negative diagnostic; this H200 run produced non-zero reward variance and changed trainable parameters in both stages.
- Supported: the untouched six-task holdout replayed 6/6 after a fresh reload; this is not a Stage-2 improvement claim.
- Not supported: unseen-task generalization improved.
- Not supported: PRM-Lite solved saturation in this environment.
- Tested: official BFCL-V4 four-category smoke pipeline; observed base and Stage-2 scores are both 0.00%.
- Not tested: BFCL-V4 paper profile, TAU-2 Avg@4, or the paper's 47.4 average benchmark score.
- Not tested: the paper's ≈100K, Qwen3-235B, Round 0–3 data flywheel.

## 📚 References

- [AgenticQwen paper](https://arxiv.org/abs/2604.21590)
- [Official AgenticQwen collection](https://huggingface.co/collections/alibaba-pai/agenticqwen)
- [Official data synthesis / RL code](https://github.com/haruhi-sudo/data_synth_and_rl)
- [Official TAU-2 evaluator](https://github.com/sierra-research/tau2-bench/tree/v0.2.0)
- [Official BFCL evaluator](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [Agentic-GRPO-LongHorizon style and diagnosis reference](https://github.com/qiqihezh/agentic-grpo-longhorizon)
- [DashScope Qwen Function Calling](https://help.aliyun.com/en/model-studio/qwen-function-calling)
