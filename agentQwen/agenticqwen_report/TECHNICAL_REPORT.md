# AgenticQwen Long-Horizon Curriculum：多证据链可审计复现报告

> ✅ **结论先行。** 最新官方路径已在 `NVIDIA H200` 上直接运行 AgenticQwen 发布的 `verl + SGLang + mock user/tool + completion reward`：Round 0 使用官方数据训练，基于真实失败做定向合成，再从 Round-0 权重继续 Round 1。两轮均出现非零奖励方差、非零 policy loss 和正梯度，审计均为 **PASS**。固定官方 holdout 最终仍为 0/1，因此本报告证明“算法链路和参数更新真实发生”，**不声称泛化提升、100K 全量训练或论文 47.4 分已复现**。

**最新官方 Run:** `official_round0_single_tool_smoke_v3` → `official_round1_curriculum_smoke_v1`
**模型:** `Qwen/Qwen3-8B` · BF16 · full-parameter FSDP
**训练:** Round 0 `1` audited step → failure audit / teacher gate → Round 1 `1` audited step
**状态:** official execution `PASS` · learning signal `PASS` · holdout improvement `NOT OBSERVED` · paper-scale claim `false`

## 0. 最新官方路径：37,401 行发布数据作为主池

这次没有把“最多合成 10 条”错误地套到官方数据上。已下载和校验的 `agenticqwen_synthetic_data.parquet` 包含 **37,401** 行、4,782 个 base task；`≤10` 只限制新生成的困难任务。为了在单张 H200 上先验证 curriculum 的因果链路，本轮从官方池精确选取 3 个单工具 variant：2 个进入训练，1 个作为按 base-id 隔离的 holdout。

| 环节 | 真实输入 | 真实结果 |
|---|---|---|
| Round 0 | 2 条官方 train，1 条官方 holdout；4 generations / prompt | 8 条 rollout：1 成功、7 失败；reward mean `0.125`；policy loss `0.01707`；`grad_norm=1.484375` |
| 轨迹审计 | rollout 对齐官方 `normal_path`、参数和终止协议 | 参数改写不匹配、过早转人工、重复调用、缺少终止标记 |
| 教师扩展 | DeepSeek-v4-flash 替代 Qwen3-235B；branch inversion → schema → solve → branch-hit | 请求 2 条，保留 1 条；另一条因未新增分支被拒；合成用量 `1/10` |
| Round 1 | 2 条官方 replay + 1 条门控困难任务；从 Round-0 合并权重启动 | 12 条 rollout：4 成功、8 失败；reward mean `0.3333`；policy loss `0.02850`；`grad_norm=1.2734375` |
| 权重与部署 | 两轮 FSDP checkpoint 合并为 Hugging Face 格式 | Stage 1/2 的 4/4 分片哈希全部变化；fresh-process reload `PASS` |
| 固定 holdout | 完整多轮工具任务 | 最终 `0/1`；独立重载能生成正确首个工具调用，但没有通过完整 episode reward |

官方执行证据保存在 `artifacts/official_agenticqwen_h200/`。审计器不仅检查 checkpoint 和 reward，还解析持久化训练日志，只有同时观察到 reward variance 与 `actor/grad_norm > 0` 才把 learning signal 标为 `PASS`。

## 0.1 较早的 QLoRA 退款课程证据链

下面原有章节记录的是另一条较早完成的自建退款环境实验：NF4 + LoRA、每阶段 12 steps，并跑通 BFCL smoke 流水线。它用于解释长程状态机、过程奖励和 curriculum 工程，不与最新的官方 FSDP 路径合并计数。

### 模型与运行时 gate

模型不是在训练进程里临时下载的：远端 15 个文件已逐项 SHA-256 校验，随后才启动训练。独立 smoke 重新加载 NF4 base + LoRA，并在同一张 GPU 上生成短 completion。

| Gate | 状态 | 证据 |
|---|---|---|
| Remote snapshot SHA-256 | **PASS** | `15 files` |
| NF4 + LoRA model load | **PASS** | `NVIDIA H200`, peak `9.148 GiB` |
| Transformers / bitsandbytes / torch | **observed** | `5.14.1 / 0.50.0 / 2.6.0+cu124` |
| Trainable LoRA parameters | **observed** | `43646976` |


## 1. 这次到底复现了什么

这不是“写好一个框架”或“调用一次 API”的展示。训练过程中，Qwen3-8B 自己生成 assistant response token 和结构化工具调用；TRL 为每条 rollout 创建独立的 stateful environment；工具执行后把 observation 继续写回对话；episode 结束时由环境最终状态计算 outcome/process reward；四条同 prompt rollout 形成 GRPO group 并反向传播到 LoRA 参数。

第二阶段的数据不是一份与运行无关的固定训练集。训练完成后先运行冻结 probe：若仍有 terminal failure，就按实际失败类型生成难例；若小 probe 已全对，就转入预定义的 frontier taxonomy，继续增加 decoy orders、瞬时失败和更紧的确认约束。两条路径都会保存 source trace hash，并混入旧任务 replay 再训练。本次实际走的是 `stage1_training_failures`。

| 层次 | 本次真实证据 | 不能据此声称 |
|---|---|---|
| 多轮 policy | Qwen3-8B response tokens + native tool calls | 不是单 token 四选一 |
| Environment | 身份、订单、支付、政策、确认、退款状态机 | 不是模型自报“成功” |
| RL update | 两阶段 optimizer steps、adapter hash、checkpoint | 不是只做 inference |
| Curriculum | Stage-1 trace hash → failure/frontier selection → hard tasks → Stage 2 | 当前选择：`stage1_training_failures` |
| Evaluation | untouched final holdout + fresh reload | BFCL smoke 当前为 `PASS` |

## 2. 从论文目标拆成 Plan → Execute → Verify

### Plan

1. 锁定 final holdout，再生成训练数据，阻断 task-ID 泄漏。
2. Stage 1 只训练基础安全退款任务；probe 专门暴露 decoy 与工具失败。
3. 失败诊断只读环境 state/event ledger，不让 LLM 充当 ground truth。
4. Stage 2 从原 adapter 继续训练，混入 replay 避免只适配难例。
5. 独立进程重载 adapter；最后再进入官方 BFCL evaluator。

### Execute

```text
Stage-1 train (8 tasks, group=4)
    ↓
Frozen probe rollouts (4 tasks)
    ↓
Failure taxonomy / frontier fallback + trace SHA-256
    ↓
Hard synthesis (8 tasks) + replay (4 tasks)
    ↓
Stage-2 continue training
    ↓
Untouched holdout (6 tasks)
    ↓
Fresh-process reload
    ↓
BFCL base/adapter smoke（本次状态：PASS）
```

### Verify

| 审计门 | 状态 | 证据 |
|---|---|---|
| stage1 adapter exists | **PASS** | `/home/hadoop-aipnlp/agenticqwen-paper-debug-20260804/artifacts/agenticqwen_codelab_curriculum_run1/stage1/adapter` |
| stage2 adapter exists | **PASS** | `/home/hadoop-aipnlp/agenticqwen-paper-debug-20260804/artifacts/agenticqwen_codelab_curriculum_run1/stage2/adapter` |
| stage2 adapter differs from stage1 | **PASS** | `stage1=f0fad889000fb98e7a80de61a30615f35d3ae1a763cca60c2830e33981104a9c; stage2=ea699445f74d835df8d49a7a8839069faceee7f5c0c51f0085504d2433264cfb` |
| both GRPO stages changed trainable parameters | **PASS** | `stage1=True; stage2=True` |
| training reward variance observed | **PASS** | `stage1_std=0.6143729119528745; stage2_std=0.32974885162106293` |
| final holdout isolation | **PASS** | `overlap=[]` |
| failure-driven synthesis has provenance | **PASS** | `deterministic_failure_driven_curriculum` |
| stage2 consumed stage1 adapter | **PASS** | `f0fad889000fb98e7a80de61a30615f35d3ae1a763cca60c2830e33981104a9c` |
| all planned optimizer steps completed | **PASS** | `stage1=12/12; stage2=12/12` |
| evaluation traces contain complete episode sets | **PASS** | `stage1_probe=4; stage2_holdout=6` |
| fresh-process adapter replay | **PASS** | `episodes=6; success_rate=1.0; parent_pid=69452; child_pid=72418` |

## 3. Stateful tool environment

环境有七个公开工具：`request_identity`、`lookup_customer`、`list_orders`、`get_payment_history`、`get_refund_policy`、`request_confirmation`、`create_refund`。它们不是一串无状态 mock：每一步读取或改变 episode state，后续工具会检查先决条件。

写操作 `create_refund` 只有在以下条件同时成立时才通过：身份已验证、订单已读、支付历史已读、政策已读、用户对精确 order/charge/amount 明确确认、目标是 duplicate charge、金额精确、reason 合法、idempotency key 合法。模型无法绕过代码直接修改 state。

奖励为：

$$R = \mathbb{1}[\text{exact refund success}] + 0.30 \cdot R_{process}$$

`R_process` 奖励正确读取链、从 transient error 中恢复、任务绑定的幂等键、精确退款理由与参数简洁度；惩罚 premature write、schema error、重复调用与超出预期的工具步数。V1 把所有成功轨迹裁成同一个 `1.3`，导致 advantage 为零；V2 保留成功轨迹之间的可解释质量差异。

## 4. 真正的 response-token GRPO

每个 prompt 采样 $G=4$ 条完整轨迹。第 $i$ 条轨迹的 group-relative advantage 为：

$$A_i = \frac{R_i - \mu(R_1,\ldots,R_G)}{\sigma(R_1,\ldots,R_G)+10^{-4}}$$

优化使用 DAPO-style clipped objective；loss mask 只覆盖 assistant completion tokens，tool observations 与 prompt 不参与梯度。LoRA 覆盖 Q/K/V/O 与 gate/up/down projections，base 权重以 NF4 4-bit 加载，计算使用 bfloat16。

| 超参数 | 值 |
|---|---:|
| Group size | 4 |
| Gradient accumulation | 4 |
| Max completion | 1400 tokens |
| Max tool iterations | 10 |
| Stage-1 LR / steps | 5e-06 / 12 |
| Stage-2 LR / steps | 3e-06 / 12 |
| Loss | `dapo` |

## 5. Stage 1：先训练，再诚实看失败

| 指标 | Before | After | Δ |
|---|---:|---:|---:|
| Probe success rate | 75.0% | 100.0% | +25.0 pt |
| Mean combined reward | 0.9041 | 1.2502 | +0.3461 |
| Optimizer global step | — | 12 | — |
| Training seconds | — | 425.942 | — |
| Training rollout reward std | — | 0.6144 | 20 unique rewards |
| Trainable parameters changed | — | `True` | train loss `0.022327` |

| 失败类型 | Before | After |
|---|---:|---:|
| `POLICY_NOT_INSPECTED` | 1 | 0 |

这里不因为“训练跑完”就自动写成进步。如果 success rate 或 mean reward 下降，报告仍保留负号；curriculum 的价值要由 Stage-2 unseen holdout 与 failure composition 判断，而不是挑一条漂亮轨迹。

### Stage-1 probe 代表轨迹

任务 `refund-curriculum_probe-0100`；成功：`True`；失败分类：`SUCCESS`；combined reward：`1.243`。

| Turn | Tool | OK | Observation / state evidence |
|---:|---|:---:|---|
| 1 | `request_identity` | ✅ | `{"email": "customer0100@example.com", "phone_last4": "8521"}` |
| 2 | `lookup_customer` | ✅ | `{"customer_id": "CUS-1100", "verified": true}` |
| 3 | `list_orders` | ✅ | `{"orders": [{"amount": 349.0, "currency": "CNY", "order_id": "ORD-2026-4100"}, {"amount": 478.13, "currency": "CNY", "order_id": "ORD-DECOY-100-A"}]}` |
| 4 | `get_payment_history` | ✅ | `{"charges": [{"amount": 349.0, "captured_at": "2026-07-25T09:31:08+08:00", "charge_id": "CHG-7200", "currency": "CNY", "duplicate": false}, {"amount": 349.0, "captured_at": "202…` |
| 5 | `get_refund_policy` | ✅ | `{"arrival_window": "3-5 business days", "currency": "CNY", "eligible": true, "max_refund": 349.0, "requires_confirmation": true, "target": "duplicate charge only"}` |
| 6 | `request_confirmation` | ✅ | `{"confirmed": true, "user_message": "I confirm refunding only CHG-7201 for CNY 349.00."}` |
| 7 | `create_refund` | ✅ | `{"amount": 349.0, "arrival_window": "3-5 business days", "currency": "CNY", "refund_id": "RF-55B751695E"}` |

## 6. 失败驱动的困难数据合成

困难数据 manifest 记录 source trace 的绝对路径与 SHA-256：`7388b93f024dac4bd8842fbe6b6239ef2ea29e4a7a051dafd8ddfa23540f9a9f`。本次 source selection 为 `stage1_training_failures`。当 post-train probe 仍有失败时优先按残余失败合成；若小 probe 已全对，则升级到预定义 frontier taxonomy。生成器不把模型自然语言当答案，所有 customer/order/charge/amount、状态转移与 verifier target 都由代码确定。

| 合成项 | 证据 |
|---|---|
| Generator | `deterministic_failure_driven_curriculum` |
| Source selection | `stage1_training_failures` |
| Hard tasks | 8 |
| Replay task IDs | `refund-stage1_train-0000, refund-stage1_train-0001, refund-stage1_train-0002, refund-stage1_train-0003` |
| Stage-2 train SHA-256 | `222bd26ad52470d25ea3a0c3d190af9636edc981c57276d4951283b6597fcdc7` |
| Ground-truth policy | All identifiers, amounts, transitions, and verifier targets are generated deterministically from code; no model output is used as ground truth. |

任务增强规则把 failure 转成可训练扰动：未读支付历史 → 增加第三个 decoy；policy failure → policy service 首次 timeout；identity failure → lookup 首次 timeout；loop/tool-error → 在读链或 write 上加入一次 retryable failure；confirmation missing → 保留精确确认硬门。

## 7. Stage 2：从同一个 adapter 继续训练

| 指标 | Before | After | Δ |
|---|---:|---:|---:|
| Final-holdout success rate | 100.0% | 100.0% | +0.0 pt |
| Mean combined reward | 1.2526 | 1.2525 | -0.0001 |
| Optimizer global step | — | 12 | — |
| Training seconds | — | 402.884 | — |
| Training rollout reward std | — | 0.3297 | 36 unique rewards |
| Trainable parameters changed | — | `True` | train loss `-0.030853` |

| 失败类型 | Before | After |
|---|---:|---:|
| 无失败类别 | 0 | 0 |

Stage-2 summary 中的 `input_adapter_weights_sha256` 必须等于 Stage-1 `adapter_model.safetensors` hash；Stage-2 权重 hash 又必须不同。这两个条件分别排除“第二阶段从 base 重开”和“第二阶段根本没有更新”。

### Final holdout 代表轨迹

任务 `refund-final_holdout-0204`；成功：`True`；失败分类：`SUCCESS`；combined reward：`1.248`。

| Turn | Tool | OK | Observation / state evidence |
|---:|---|:---:|---|
| 1 | `request_identity` | ✅ | `{"email": "customer0204@example.com", "phone_last4": "2369"}` |
| 2 | `lookup_customer` | ✅ | `{"customer_id": "CUS-1204", "verified": true}` |
| 3 | `list_orders` | ✅ | `{"orders": [{"amount": 349.0, "currency": "CNY", "order_id": "ORD-2026-4204"}, {"amount": 478.13, "currency": "CNY", "order_id": "ORD-DECOY-204-A"}, {"amount": 478.13, "currency…` |
| 4 | `get_payment_history` | ✅ | `{"charges": [{"amount": 349.0, "captured_at": "2026-07-25T09:31:08+08:00", "charge_id": "CHG-7408", "currency": "CNY", "duplicate": false}, {"amount": 349.0, "captured_at": "202…` |
| 5 | `get_refund_policy` | ✅ | `{"arrival_window": "3-5 business days", "currency": "CNY", "eligible": true, "max_refund": 349.0, "requires_confirmation": true, "target": "duplicate charge only"}` |
| 6 | `request_confirmation` | ✅ | `{"confirmed": true, "user_message": "I confirm refunding only CHG-7409 for CNY 349.00."}` |
| 7 | `create_refund` | ✅ | `{"amount": 349.0, "arrival_window": "3-5 business days", "currency": "CNY", "refund_id": "RF-27582FA0BC"}` |

## 8. 独立评测：fresh reload 与官方 BFCL-V4 smoke

已完成的独立性来自 fresh-process reload：父训练进程释放模型后启动新的 Python 子进程，记录父/子 PID，再重新加载 base + Stage-2 PEFT adapter，在未触碰的 final holdout 上重新生成并计分。官方 BFCL-V4 multi-turn evaluator 使用官方数据、官方生成器和官方 checker；当前状态由下表中的 manifest 决定，未通过时不会被包装成成功。

| Fresh replay evidence | 值 |
|---|---|
| Parent PID | `69452` |
| Child PID | `72418` |
| Is fresh process | `True` |
| Episodes | `6` |

| Variant | 状态 | Episodes | Official score row |
|---|---|---:|---|
| `base` | **PASS** | 4 | `{"Overall Acc": "0.00%", "Multi Turn Acc": "0.00%", "Model": "Qwen3-8B (FC)"}` |
| `stage2_adapter` | **PASS** | 4 | `{"Overall Acc": "0.00%", "Multi Turn Acc": "0.00%", "Model": "Qwen3-8B (FC)"}` |

BFCL 状态为 `PASS`。本次 manifest、result JSON 和 score CSV 已齐全，因此官方接口、tool-call 格式、multi-turn execution 与 score pipeline 确实跑通；但这组 smoke 的 base 与 Stage-2 adapter `Overall Acc`/`Multi Turn Acc` 都是 **0.00%**，说明当前模型没有通过所选的 4 个 BFCL 题目，不能把“流水线 PASS”误读为模型质量 PASS。它仍不能替代每类 200 个任务的论文 profile，更不能与论文平均分直接比较。

## 9. 实验完整性与可恢复性

- 数据先落盘再训练；train/probe/final-holdout ID 分离并写入 manifest。
- 每个 JSONL trace 同时记录 task hash、stage、failure type、reward 与完整 event ledger。
- Trainer 每半程保存 checkpoint；远程中断后从最大 `checkpoint-N` 恢复，旧 trace 不清空。
- 已完成的 stage 只有在 summary 状态和 adapter 目录同时存在时才跳过。
- 模型缓存、checkpoint、adapter、benchmark 与报告源分别持久化；下载后再次计算 SHA-256。
- 日志写入前扫描疑似 `sk-` secret；API key 永远不进入 artifact。

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `benchmarks/bfcl_smoke/base/result/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_base_result.json` | 232015 | `b13788f58cd00635…` |
| `benchmarks/bfcl_smoke/base/result/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_long_context_result.json` | 283413 | `ea02c9bb19514c09…` |
| `benchmarks/bfcl_smoke/base/result/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_result.json` | 362205 | `fcf22aff512f9f1c…` |
| `benchmarks/bfcl_smoke/base/result/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_miss_param_result.json` | 236252 | `a9b71303c99f35b9…` |
| `benchmarks/bfcl_smoke/base/score/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_base_score.json` | 231911 | `9067ea7eb346518c…` |
| `benchmarks/bfcl_smoke/base/score/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_long_context_score.json` | 282395 | `2ee93cf52609ce37…` |
| `benchmarks/bfcl_smoke/base/score/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_score.json` | 363378 | `562529f06a1c9012…` |
| `benchmarks/bfcl_smoke/base/score/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_miss_param_score.json` | 237216 | `01808a881e5dfcc8…` |
| `benchmarks/bfcl_smoke/base/score/data_agentic.csv` | 207 | `165122540b141ba8…` |
| `benchmarks/bfcl_smoke/base/score/data_format_sensitivity.csv` | 2723 | `dab92466fbcb2369…` |
| `benchmarks/bfcl_smoke/base/score/data_live.csv` | 218 | `25620e00c0de91b5…` |
| `benchmarks/bfcl_smoke/base/score/data_multi_turn.csv` | 118 | `10541ca2bfb14b86…` |
| `benchmarks/bfcl_smoke/base/score/data_non_live.csv` | 238 | `80fe0a5937eb5bae…` |
| `benchmarks/bfcl_smoke/base/score/data_overall.csv` | 884 | `45423faf5c75d34d…` |
| `benchmarks/bfcl_smoke/base/test_case_ids_to_generate.json` | 249 | `a96fa16dd4d8ce2e…` |
| `benchmarks/bfcl_smoke/manifest.json` | 31575 | `3a731a7327147478…` |
| `benchmarks/bfcl_smoke/stage2_adapter/result/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_base_result.json` | 233343 | `c16ff07e0e981727…` |
| `benchmarks/bfcl_smoke/stage2_adapter/result/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_long_context_result.json` | 214101 | `19549c6f5e8d3bba…` |
| `benchmarks/bfcl_smoke/stage2_adapter/result/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_result.json` | 229027 | `3e33fe331b69443e…` |
| `benchmarks/bfcl_smoke/stage2_adapter/result/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_miss_param_result.json` | 203534 | `8afd21c9a9aae00b…` |
| `benchmarks/bfcl_smoke/stage2_adapter/score/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_base_score.json` | 233122 | `4ec02e609cb2ccb3…` |
| `benchmarks/bfcl_smoke/stage2_adapter/score/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_long_context_score.json` | 214096 | `4802020fd3c60351…` |
| `benchmarks/bfcl_smoke/stage2_adapter/score/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_score.json` | 229620 | `832c549ee5734c75…` |
| `benchmarks/bfcl_smoke/stage2_adapter/score/Qwen_Qwen3-8B-FC/multi_turn/BFCL_v4_multi_turn_miss_param_score.json` | 206416 | `8b46bd9ce325d549…` |
| `benchmarks/bfcl_smoke/stage2_adapter/score/data_agentic.csv` | 207 | `165122540b141ba8…` |
| `benchmarks/bfcl_smoke/stage2_adapter/score/data_format_sensitivity.csv` | 2723 | `dab92466fbcb2369…` |
| `benchmarks/bfcl_smoke/stage2_adapter/score/data_live.csv` | 218 | `25620e00c0de91b5…` |
| `benchmarks/bfcl_smoke/stage2_adapter/score/data_multi_turn.csv` | 118 | `10541ca2bfb14b86…` |
| `benchmarks/bfcl_smoke/stage2_adapter/score/data_non_live.csv` | 238 | `80fe0a5937eb5bae…` |
| `benchmarks/bfcl_smoke/stage2_adapter/score/data_overall.csv` | 884 | `111dce47512dea3b…` |
| `benchmarks/bfcl_smoke/stage2_adapter/test_case_ids_to_generate.json` | 249 | `a96fa16dd4d8ce2e…` |
| `benchmarks/bfcl_smoke/verification.json` | 5779 | `3c5305cc87100285…` |
| `data_manifest.json` | 1743 | `c3659f3731eed782…` |
| `fresh_replay/summary.json` | 1750 | `7e2c65225444589d…` |
| `fresh_replay/traces.jsonl` | 23771 | `555799d996d56640…` |
| `resolved_config.json` | 1940 | `ee814212f548ae3d…` |
| `run_summary.json` | 26077 | `6dbcb6b01a818e0d…` |
| `stage1/adapter/adapter_config.json` | 1200 | `2d2df16f2493c568…` |
| `stage1/adapter/adapter_model.safetensors` | 87361592 | `f0fad889000fb98e…` |
| `stage1/adapter/tokenizer.json` | 11422650 | `be75606093db2094…` |

## 9.5 论文式行为树飞轮：真实执行与失败恢复

最新官方路径不再使用下面较早的 deterministic frontier fallback。它读取 Qwen3-8B 的真实 verl rollout，对照官方 `normal_path` 做失败归因，再把失败任务交给 DS-v4-flash 做 branch-to-task inversion。候选必须依次通过 schema、与源路径不同、教师可解、完整工具序列命中预设分支四个门。

| Gate | Round 0 | Round 1 |
|---|---:|---:|
| official train / synthetic train | 2 / 0 | 2 / 1 |
| rollout rows | 8 | 12 |
| reward mean / unique | 0.125 / {0,1} | 0.3333 / {0,1} |
| policy loss | 0.01707 | 0.02850 |
| actor grad norm | 1.484375 | 1.2734375 |
| execution / learning audit | PASS / PASS | PASS / PASS |
| merged model | PASS | PASS + fresh-process reload |

失败审计不是只搜关键词：Maramureș 任务实际把 `"Maramureș County, Romania"` 改写成了 `"Maramureș County, RO"` 并过早转人工；PRT-1234 任务重复执行同一查询且没有终止标记。合成器据此产生带干扰删除路径的候选；保留 1 条，另一条因 normal path 没有新增分支而被拒。这里的 `1/10` 是新合成用量，官方数据池 37,401 行不受该 cap 限制。

### 9.5.1 较早的 paper_flywheel 试运行（历史负对照）

下面这条路径与退款 curriculum 分开计数：它从 SynthAgent-compatible 的线性航班 workflow 出发，让教师模型做 branch expansion、branch-to-task inversion 和 teacher-solved/branch-hit filtering，再把新任务交给同一个 Qwen3-8B policy 继续训练。论文中的 Qwen3-235B 在本次实验中由配置的 DeepSeek-v4-flash endpoint 替代，因此不声称论文规模或教师等价性。该旧 run 的合成预算同样只约束新任务，不约束官方 replay。

| 项目 | 证据 |
|---|---|
| run status | **partial_teacher_endpoint_blocked** |
| flywheel verification | **PASS** |
| completed boundary | `round1:training_and_policy_rollout` |
| teacher audit rows | `96` |
| teacher operation/status counts | `Analyze policy rollouts and expand the existing linear or partial workflow into a deeper multi-branch behavior tree/ok=10, Perform AgenticQwen branch-to-task inversion/error=5, Perform AgenticQwen branch-to-task inversion/ok=23, Solve this simulated agent task/error=10, Solve this simulated agent task/ok=48` |
| deterministic frontier recovery observed | `False` |

| Round | Optimizer steps | Policy episodes | Successes | Reward std | Parameters changed |
|---|---:|---:|---:|---:|:---:|
| `round0` | 4 | 4 | 4 | 0.0 | False |
| `round1` | 4 | 5 | 5 | 0.0 | False |

真实运行中，Round 0 的 policy 训练与 rollout 已落盘；后续扩展由教师 API 返回 502/读取超时以及 branch-hit 校验拒绝，控制器保持 PARTIAL 并保留 API audit，不把未验证候选写入训练集。恢复配置已经写入并启动，但在 saturation fallback 产物落盘前因 teacher endpoint 的 timeout/截断停止；因此不能把 fallback 说成已完成。 设计上的 fallback 是：如果五条允许的 micro 行为分支已饱和，就增加 revision 并重放可验证分支，不伪造第六条分支。
### 最新 bounded continuation（独立 run2）

该次恢复明确执行 `max_synthetic_trajectories=10`、`max_new_tasks_per_round=1`，不会合成超过 10 条。Round 0 的 Qwen3-8B LoRA-GRPO 产物已保留；教师 audit 共 `18` 条，其中状态计数为 `Analyze policy rollouts and expand the existing linear or partial workflow into a deeper multi-branch behavior tree/error=1, Analyze policy rollouts and expand the existing linear or partial workflow into a deeper multi-branch behavior tree/ok=3, Perform AgenticQwen branch-to-task inversion/error=1, Perform AgenticQwen branch-to-task inversion/ok=6, Solve this simulated agent task/error=1, Solve this simulated agent task/ok=6`。终态为 **PARTIAL**：教师请求出现读取超时，随后 executable branch-hit gate 拒绝候选，因此没有把未验证任务写入下一轮。


## 10. 失败模式：项目为什么可能“训练了却没学会”

### Group saturation

如果同一 group 四条 rollout 都成功或都失败，标准差接近零，relative advantage 变成零。过程奖励只能在轨迹真正产生不同读取/恢复行为时打破平局；若所有样本在第一个 token 就走同一路径，仍然没有信号。

### Reward hacking

模型可能反复调用读工具累积 progress。环境因此对 redundant call 和超过十步的序列施加惩罚，并且 outcome success 仍要求精确写状态；只刷过程分无法得到完整奖励。

### Curriculum forgetting

只训练难例会破坏基础路径，因此 Stage 2 固定加入四个 Stage-1 replay tasks。更完整的实验应增加 replay ratio 消融与至少三 seeds；本次单次预算运行不能支持“普遍优于”的统计主张。

### Benchmark mismatch

自建退款环境验证 causal curriculum，但 BFCL 测工具调用格式、缺函数/缺参数和长上下文；TAU-2 又依赖 user simulator。不同指标回答不同问题，不能用自建 success rate 替换官方 benchmark。

## 11. 一键复现与运行成本

```bash
# 官方 AgenticQwen/verl 路径
./scripts/run_official_agenticqwen_h200.sh

# AutoDL 实例中：复用 /root/autodl-tmp/models/Qwen3-8B
./run_curriculum_autodl_remote.sh

# 本地：把下载的 run root 渲染成 HTML + PPT
./finalize_curriculum_project.sh /path/to/qwen3-8b-qlora-20260804-v2
```

两条路径都在单张 `NVIDIA H200`（显存约 `139.7` GiB）上运行。最新官方路径使用 BF16 full-parameter FSDP，峰值 allocated 约 76.35 GiB、reserved 约 89.25 GiB；较早退款路径才是 NF4 + LoRA。模型、venv、run、日志和证据归档分离落盘，终态按 SHA-256 收集。

## 12. 简历与面试怎么讲

### 简历 bullet（只有本报告 PASS 后才能使用）

> 基于 AgenticQwen 官方 verl/SGLang 与 Qwen3-8B 实现两轮 stateful multi-turn GRPO：从 37,401 行发布数据池构建 group-disjoint curriculum，审计真实 rollout 的参数错配、过早转人工、重复调用和终止失败，使用强模型完成 branch inversion + solved/branch-hit 门控后继续训练；两轮均验证非零梯度，Stage 1/2 四个权重分片全部变化，并完成 fresh-process reload。

### 面试问题 1：这和 SFT 有什么本质区别？

SFT 复制给定答案；这里模型自己采样完整 trajectory，reward 来自工具执行后的最终 state，同 prompt 的多个 trajectory 做 group-relative baseline。困难数据也由实际失败类型触发，而不是预先固定一份 instruction 数据。

### 面试问题 2：为什么过程奖励不会替代最终正确性？

combined reward 的主项仍是 exact outcome；process 只占 0.30，而且对越权、schema error、循环有负项。最终 verifier 独立检查 charge ID、amount、确认与 refund state，所以“步骤看起来很努力”不能冒充成功。

### 面试问题 3：怎么证明第二阶段真的用了第一阶段？

Stage-2 summary 保存 input adapter tree hash；审计要求它与 Stage-1 adapter hash 完全相等。同时 Stage-2 output hash 必须不同、global step 必须大于零，三条证据合在一起排除重开、未训练和只改 metadata。

### 面试问题 4：如果 Stage 2 指标没提升怎么办？

把它作为负结果：比较 failure composition、group variance、hard/replay ratio 和 transient recovery；扩大 seeds。不能删除失败 run 或只展示一条成功轨迹。工程完整性 PASS 与算法效果提升是两个独立结论。

## 13. 自我检查：仍未完成什么

| 项目 | 当前状态 | 补完条件 |
|---|---|---|
| 官方数据 + verl/SGLang 两轮 GRPO | **PASS（微型）** | 两轮 execution / learning audit 均 PASS |
| 失败审计 → 定向合成 → 再训练 | **PASS（1 条新任务）** | solved + branch-hit，合成用量 1/10 |
| 固定官方 holdout 提升 | **NOT OBSERVED** | 当前 0/1；扩大任务、steps、seed |
| 较早退款 QLoRA curriculum | **COMPLETE** | verification 全门通过 |
| 官方 BFCL multi-turn smoke | **PIPELINE PASS / SCORE 0%** | base/adapter result 与 score 齐全，但无质量提升 |
| BFCL-V4 paper profile | **CODE_READY / NOT_RUN** | 4 categories × 200，固定 decoding |
| TAU-2 Avg@4 | **CODE_READY / NOT_RUN** | 全域任务 × 4，论文一致 user simulator |
| 3 seeds + ablation matrix | **NOT_RUN** | Vanilla/PRM/LATA/Joint × 3 seeds |
| 37,401 行官方发布池 | **AVAILABLE / VERIFIED** | 当前 active curriculum 只使用 3 个 variant |
| 论文约 100K 总训练规模 | **NOT RUN** | 扩大官方池 + reasoning 数据 + 多轮扩展 |
| 论文八 H100 配方 | **BLOCKED_RESOURCE** | 同版本 veRL/SGLang 与多卡预算 |

## 14. Claim boundary

本报告支持：“一次真实 Qwen3-8B 官方 AgenticQwen/verl 两轮 curriculum 在 H200 执行；官方数据参与训练，真实失败驱动一条门控困难任务进入下一轮；两轮均观察到 reward variance、非零 policy loss、正梯度与独立 checkpoint，Stage-2 合并模型可在新进程重载并生成正确首个工具调用。”

本报告不支持：“完整复现论文 47.4、100K 全量训练、Qwen3-235B 教师等价性、八 H100 recipe、TAU-2 Avg@4、BFCL 提升或统计显著的普遍提升。”

## 参考资料

- [AgenticQwen paper](https://arxiv.org/abs/2604.21590)
- [Official data synthesis / RL code](https://github.com/haruhi-sudo/data_synth_and_rl)
- [TRL GRPOTrainer stateful environments](https://huggingface.co/docs/trl/en/grpo_trainer)
- [Official BFCL repository](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [Official TAU-2 v0.2.0](https://github.com/sierra-research/tau2-bench/tree/v0.2.0)
- [Agentic-GRPO-LongHorizon style reference](https://github.com/qiqihezh/agentic-grpo-longhorizon)
- [Modal GPU documentation](https://modal.com/docs/guide/gpu)
- [Modal pricing](https://modal.com/pricing)
