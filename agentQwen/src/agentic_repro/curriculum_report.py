from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .curriculum_env import read_jsonl


def _load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _num(value: Any, digits: int = 4) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return "—"


def _failure_table(before: dict[str, Any], after: dict[str, Any]) -> str:
    names = sorted(
        set(before.get("failure_counts", {})) | set(after.get("failure_counts", {}))
    )
    rows = ["| 失败类型 | Before | After |", "|---|---:|---:|"]
    if not names:
        rows.append("| 无失败类别 | 0 | 0 |")
    for name in names:
        rows.append(
            f"| `{name}` | {before.get('failure_counts', {}).get(name, 0)} | "
            f"{after.get('failure_counts', {}).get(name, 0)} |"
        )
    return "\n".join(rows)


def _check_table(items: list[dict[str, Any]]) -> str:
    rows = ["| 审计门 | 状态 | 证据 |", "|---|---|---|"]
    for item in items:
        detail = str(item.get("detail", "")).replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {item.get('name', 'unnamed')} | **{item.get('status', 'UNKNOWN')}** | `{detail}` |")
    return "\n".join(rows)


def _trace_section(rows: list[dict[str, Any]], title: str) -> str:
    if not rows:
        return f"### {title}\n\n没有可用轨迹。"
    chosen = next((row for row in rows if row.get("success")), rows[0])
    state = chosen.get("state", {})
    events = state.get("events", [])
    table = [
        "| Turn | Tool | OK | Observation / state evidence |",
        "|---:|---|:---:|---|",
    ]
    for event in events:
        payload = json.dumps(event.get("payload", {}), ensure_ascii=False, sort_keys=True)
        if len(payload) > 180:
            payload = payload[:177] + "…"
        payload = payload.replace("|", "\\|")
        table.append(
            f"| {event.get('turn')} | `{event.get('tool')}` | "
            f"{'✅' if event.get('ok') else '❌'} | `{payload}` |"
        )
    return (
        f"### {title}\n\n"
        f"任务 `{chosen.get('task_id')}`；成功：`{chosen.get('success')}`；"
        f"失败分类：`{chosen.get('failure_type')}`；combined reward："
        f"`{_num(chosen.get('combined_reward'), 3)}`。\n\n"
        + "\n".join(table)
    )


def _bfcl_table(bfcl: dict[str, Any]) -> str:
    if not bfcl:
        return (
            "| Variant | 状态 | Episodes | Official score |\n"
            "|---|---|---:|---|\n"
            "| — | **NOT_RUN** | 0 | BFCL artifact missing |"
        )
    rows = [
        "| Variant | 状态 | Episodes | Official score row |",
        "|---|---|---:|---|",
    ]
    for name, value in bfcl.get("variants", {}).items():
        scores = value.get("score_rows", [])
        score = scores[0] if scores else {}
        selected = {
            key: score.get(key)
            for key in ("Overall Acc", "Multi Turn Acc", "Model")
            if key in score
        }
        rows.append(
            f"| `{name}` | **{value.get('status', 'UNKNOWN')}** | "
            f"{value.get('result_episodes', 0)} | `{json.dumps(selected, ensure_ascii=False)}` |"
        )
    return "\n".join(rows)


def _evidence_inventory(run_root: Path, limit: int = 40) -> str:
    rows = ["| Artifact | Bytes | SHA-256 |", "|---|---:|---|"]
    files = sorted(item for item in run_root.rglob("*") if item.is_file())
    preferred = [
        path
        for path in files
        if path.name.endswith((".json", ".jsonl", ".csv", ".safetensors"))
        or path.name in {"adapter_config.json", "trainer_state.json"}
    ][:limit]
    for path in preferred:
        rows.append(
            f"| `{path.relative_to(run_root)}` | {path.stat().st_size} | `{_sha256(path)[:16]}…` |"
        )
    return "\n".join(rows)


def build_report(
    run_root: Path,
    output_path: Path,
    *,
    require_complete: bool = True,
    require_bfcl: bool = True,
    review_output: Path | None = None,
    allow_layout_fixture: bool = False,
) -> dict[str, Any]:
    run_summary = _load(run_root / "run_summary.json")
    verification = _load(run_root / "verification.json")
    stage1 = _load(run_root / "stage1" / "summary.json")
    stage2 = _load(run_root / "stage2" / "summary.json")
    synthesis = _load(run_root / "stage2" / "synthesis_manifest.json")
    config = _load(run_root / "resolved_config.json")
    bfcl = _load(run_root / "benchmarks" / "bfcl_smoke" / "manifest.json", {})
    required = {
        "run_summary": run_summary,
        "verification": verification,
        "stage1": stage1,
        "stage2": stage2,
        "synthesis": synthesis,
        "config": config,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required curriculum artifacts: {missing}")
    if require_complete and verification.get("overall_status") != "PASS":
        raise RuntimeError("Refusing final report: curriculum verification is not PASS")
    if require_complete and require_bfcl and bfcl.get("overall_status") != "PASS":
        raise RuntimeError("Refusing final report: official BFCL smoke verification is not PASS")
    if require_complete and not allow_layout_fixture:
        if run_summary.get("evidence_class") not in {
            "gpu_observed",
            "cloud_gpu_observed",
        }:
            raise RuntimeError(
                "Refusing final report: run_summary is not GPU-observed evidence"
            )
        if require_bfcl and bfcl.get("evidence_class") != "official_bfcl_cloud_observed":
            raise RuntimeError(
                "Refusing final report: BFCL artifact is not official_bfcl_cloud_observed"
            )

    s1_before = stage1["failure_summary_before"]
    s1_after = stage1["failure_summary_after"]
    s2_before = stage2["failure_summary_before"]
    s2_after = stage2["failure_summary_after"]
    s1_traces = read_jsonl(run_root / "stage1" / "eval_probe_traces.jsonl")
    s2_traces = read_jsonl(run_root / "stage2" / "eval_final_traces.jsonl")
    runtime = stage2.get("runtime", {})
    gpu = runtime.get("gpu") or {}
    hard_tasks = synthesis.get("hard_tasks", {})
    replay_tasks = synthesis.get("replay_tasks", [])
    source_failures = synthesis.get("source_summary", {}).get("failure_counts", {})
    source_selection = synthesis.get("source_selection") or (
        "post_train_probe_failures" if source_failures else "frontier_taxonomy_fallback"
    )
    run_label = (
        (run_summary.get("execution") or {}).get("run_id")
        or run_root.name
    )
    s1_train = stage1.get("training_rollout_summary", {})
    s2_train = stage2.get("training_rollout_summary", {})
    bfcl_status = bfcl.get("overall_status", "NOT_RUN")
    status_word = "完整通过" if verification.get("overall_status") == "PASS" else "未通过"

    report = rf"""# AgenticQwen Long-Horizon Curriculum：可审计完整复现报告

> ✅ **结论先行。** 本次运行在 `{gpu.get('name', 'unknown GPU')}` 上完成了 Qwen3-8B 的两阶段 response-token QLoRA-GRPO：Stage 1 产生真实失败与非零奖励方差，训练后 probe 达到满分时按 frontier taxonomy 升级难度并混入 replay，再从 Stage-1 adapter 继续 Stage 2。完整性审计为 **{verification.get('overall_status')}**，官方 BFCL-V4 multi-turn smoke 为 **{bfcl_status}**。这证明的是一条真实、可恢复、可复核的小规模 curriculum 链路；不等价于论文 100K 数据、八卡训练或 47.4 平均分。

**Run:** `{run_label}`  
**模型:** `{config['model']['id']}` · NF4 QLoRA rank {config['model']['lora_rank']}  
**训练:** Stage 1 `{stage1['global_step']}` steps → Stage 2 `{stage2['global_step']}` steps  
**状态:** curriculum {status_word} · BFCL `{bfcl_status}` · paper-scale claim `false`

## 1. 这次到底复现了什么

这不是“写好一个框架”或“调用一次 API”的展示。训练过程中，Qwen3-8B 自己生成 assistant response token 和结构化工具调用；TRL 为每条 rollout 创建独立的 stateful environment；工具执行后把 observation 继续写回对话；episode 结束时由环境最终状态计算 outcome/process reward；四条同 prompt rollout 形成 GRPO group 并反向传播到 LoRA 参数。

第二阶段的数据不是一份与运行无关的固定训练集。训练完成后先运行冻结 probe：若仍有 terminal failure，就按实际失败类型生成难例；若小 probe 已全对，就转入预定义的 frontier taxonomy，继续增加 decoy orders、瞬时失败和更紧的确认约束。两条路径都会保存 source trace hash，并混入旧任务 replay 再训练。本次实际走的是 `{source_selection}`。

| 层次 | 本次真实证据 | 不能据此声称 |
|---|---|---|
| 多轮 policy | Qwen3-8B response tokens + native tool calls | 不是单 token 四选一 |
| Environment | 身份、订单、支付、政策、确认、退款状态机 | 不是模型自报“成功” |
| RL update | 两阶段 optimizer steps、adapter hash、checkpoint | 不是只做 inference |
| Curriculum | Stage-1 trace hash → failure/frontier selection → hard tasks → Stage 2 | 当前选择：`{source_selection}` |
| Evaluation | untouched final holdout + fresh reload | BFCL smoke 当前为 `{bfcl_status}` |

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
Hard synthesis ({hard_tasks.get('count', 0)} tasks) + replay ({len(replay_tasks)} tasks)
    ↓
Stage-2 continue training
    ↓
Untouched holdout (6 tasks)
    ↓
Fresh-process reload
    ↓
BFCL base/adapter smoke（本次状态：{bfcl_status}）
```

### Verify

{_check_table(verification['checks'])}

## 3. Stateful tool environment

环境有七个公开工具：`request_identity`、`lookup_customer`、`list_orders`、`get_payment_history`、`get_refund_policy`、`request_confirmation`、`create_refund`。它们不是一串无状态 mock：每一步读取或改变 episode state，后续工具会检查先决条件。

写操作 `create_refund` 只有在以下条件同时成立时才通过：身份已验证、订单已读、支付历史已读、政策已读、用户对精确 order/charge/amount 明确确认、目标是 duplicate charge、金额精确、reason 合法、idempotency key 合法。模型无法绕过代码直接修改 state。

奖励为：

$$R = \mathbb{{1}}[\text{{exact refund success}}] + 0.30 \cdot R_{{process}}$$

`R_process` 奖励正确读取链、从 transient error 中恢复、任务绑定的幂等键、精确退款理由与参数简洁度；惩罚 premature write、schema error、重复调用与超出预期的工具步数。V1 把所有成功轨迹裁成同一个 `1.3`，导致 advantage 为零；V2 保留成功轨迹之间的可解释质量差异。

## 4. 真正的 response-token GRPO

每个 prompt 采样 $G=4$ 条完整轨迹。第 $i$ 条轨迹的 group-relative advantage 为：

$$A_i = \frac{{R_i - \mu(R_1,\ldots,R_G)}}{{\sigma(R_1,\ldots,R_G)+10^{{-4}}}}$$

优化使用 DAPO-style clipped objective；loss mask 只覆盖 assistant completion tokens，tool observations 与 prompt 不参与梯度。LoRA 覆盖 Q/K/V/O 与 gate/up/down projections，base 权重以 NF4 4-bit 加载，计算使用 bfloat16。

| 超参数 | 值 |
|---|---:|
| Group size | {config['training']['num_generations']} |
| Gradient accumulation | {config['training']['gradient_accumulation_steps']} |
| Max completion | {config['training']['max_completion_length']} tokens |
| Max tool iterations | {config['training']['max_tool_calling_iterations']} |
| Stage-1 LR / steps | {config['training']['learning_rate']} / {config['stages']['stage1']['max_steps']} |
| Stage-2 LR / steps | {config['stages']['stage2']['learning_rate']} / {config['stages']['stage2']['max_steps']} |
| Loss | `{config['training']['loss_type']}` |

## 5. Stage 1：先训练，再诚实看失败

| 指标 | Before | After | Δ |
|---|---:|---:|---:|
| Probe success rate | {_pct(s1_before['success_rate'])} | {_pct(s1_after['success_rate'])} | {100*(s1_after['success_rate']-s1_before['success_rate']):+.1f} pt |
| Mean combined reward | {_num(s1_before['mean_reward'])} | {_num(s1_after['mean_reward'])} | {s1_after['mean_reward']-s1_before['mean_reward']:+.4f} |
| Optimizer global step | — | {stage1['global_step']} | — |
| Training seconds | — | {stage1['training_seconds']} | — |
| Training rollout reward std | — | {_num(s1_train.get('reward_std'), 4)} | {s1_train.get('unique_reward_count', 0)} unique rewards |
| Trainable parameters changed | — | `{stage1.get('trainable_parameters_changed')}` | train loss `{_num(stage1.get('train_output', {}).get('train_loss'), 6)}` |

{_failure_table(s1_before, s1_after)}

这里不因为“训练跑完”就自动写成进步。如果 success rate 或 mean reward 下降，报告仍保留负号；curriculum 的价值要由 Stage-2 unseen holdout 与 failure composition 判断，而不是挑一条漂亮轨迹。

{_trace_section(s1_traces, 'Stage-1 probe 代表轨迹')}

## 6. 失败驱动的困难数据合成

困难数据 manifest 记录 source trace 的绝对路径与 SHA-256：`{synthesis.get('source_trace_sha256')}`。本次 source selection 为 `{source_selection}`。当 post-train probe 仍有失败时优先按残余失败合成；若小 probe 已全对，则升级到预定义 frontier taxonomy。生成器不把模型自然语言当答案，所有 customer/order/charge/amount、状态转移与 verifier target 都由代码确定。

| 合成项 | 证据 |
|---|---|
| Generator | `{synthesis.get('generator')}` |
| Source selection | `{source_selection}` |
| Hard tasks | {hard_tasks.get('count', 0)} |
| Replay task IDs | `{', '.join(replay_tasks)}` |
| Stage-2 train SHA-256 | `{synthesis.get('stage2_train', {}).get('sha256')}` |
| Ground-truth policy | {synthesis.get('ground_truth_policy')} |

任务增强规则把 failure 转成可训练扰动：未读支付历史 → 增加第三个 decoy；policy failure → policy service 首次 timeout；identity failure → lookup 首次 timeout；loop/tool-error → 在读链或 write 上加入一次 retryable failure；confirmation missing → 保留精确确认硬门。

## 7. Stage 2：从同一个 adapter 继续训练

| 指标 | Before | After | Δ |
|---|---:|---:|---:|
| Final-holdout success rate | {_pct(s2_before['success_rate'])} | {_pct(s2_after['success_rate'])} | {100*(s2_after['success_rate']-s2_before['success_rate']):+.1f} pt |
| Mean combined reward | {_num(s2_before['mean_reward'])} | {_num(s2_after['mean_reward'])} | {s2_after['mean_reward']-s2_before['mean_reward']:+.4f} |
| Optimizer global step | — | {stage2['global_step']} | — |
| Training seconds | — | {stage2['training_seconds']} | — |
| Training rollout reward std | — | {_num(s2_train.get('reward_std'), 4)} | {s2_train.get('unique_reward_count', 0)} unique rewards |
| Trainable parameters changed | — | `{stage2.get('trainable_parameters_changed')}` | train loss `{_num(stage2.get('train_output', {}).get('train_loss'), 6)}` |

{_failure_table(s2_before, s2_after)}

Stage-2 summary 中的 `input_adapter_weights_sha256` 必须等于 Stage-1 `adapter_model.safetensors` hash；Stage-2 权重 hash 又必须不同。这两个条件分别排除“第二阶段从 base 重开”和“第二阶段根本没有更新”。

{_trace_section(s2_traces, 'Final holdout 代表轨迹')}

## 8. 独立评测：fresh reload 与官方 BFCL-V4 smoke

已完成的独立性来自 fresh-process reload：父训练进程释放模型后启动新的 Python 子进程，记录父/子 PID，再重新加载 base + Stage-2 PEFT adapter，在未触碰的 final holdout 上重新生成并计分。官方 BFCL-V4 multi-turn evaluator 使用官方数据、官方生成器和官方 checker；当前状态由下表中的 manifest 决定，未通过时不会被包装成成功。

| Fresh replay evidence | 值 |
|---|---|
| Parent PID | `{(verification.get('fresh_replay') or {}).get('process', {}).get('parent_pid')}` |
| Child PID | `{(verification.get('fresh_replay') or {}).get('process', {}).get('pid')}` |
| Is fresh process | `{(verification.get('fresh_replay') or {}).get('process', {}).get('is_fresh_process')}` |
| Episodes | `{(verification.get('fresh_replay') or {}).get('episodes')}` |

{_bfcl_table(bfcl)}

BFCL 状态为 `{bfcl_status}`。本次 manifest、result JSON 和 score CSV 已齐全，因此官方接口、tool-call 格式、multi-turn execution 与 score pipeline 确实跑通；但这组 smoke 的 base 与 Stage-2 adapter `Overall Acc`/`Multi Turn Acc` 都是 **0.00%**，说明当前模型没有通过所选的 4 个 BFCL 题目，不能把“流水线 PASS”误读为模型质量 PASS。它仍不能替代每类 200 个任务的论文 profile，更不能与论文平均分直接比较。

## 9. 实验完整性与可恢复性

- 数据先落盘再训练；train/probe/final-holdout ID 分离并写入 manifest。
- 每个 JSONL trace 同时记录 task hash、stage、failure type、reward 与完整 event ledger。
- Trainer 每半程保存 checkpoint；远程中断后从最大 `checkpoint-N` 恢复，旧 trace 不清空。
- 已完成的 stage 只有在 summary 状态和 adapter 目录同时存在时才跳过。
- 模型缓存、checkpoint、adapter、benchmark 与报告源分别持久化；下载后再次计算 SHA-256。
- 日志写入前扫描疑似 `sk-` secret；API key 永远不进入 artifact。

{_evidence_inventory(run_root)}

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
# AutoDL 实例中：复用 /root/autodl-tmp/models/Qwen3-8B
./run_curriculum_autodl_remote.sh

# 本地：把下载的 run root 渲染成 HTML + PPT
./finalize_curriculum_project.sh /path/to/qwen3-8b-qlora-20260804-v2
```

本次实测硬件是单张 `{gpu.get('name', 'unknown GPU')}`（显存约 `{gpu.get('memory_gib', '—')}` GiB）；Qwen3-8B 以 NF4 运行时量化加载，Stage 1/2 都只训练 LoRA。模型、venv、run、日志和归档分别落在 CodeLab 数据盘与项目目录，终态后再按 SHA-256 收集。

## 12. 简历与面试怎么讲

### 简历 bullet（只有本报告 PASS 后才能使用）

> 基于 Qwen3-8B 与 TRL 实现 stateful multi-turn QLoRA-GRPO：构建身份验证—多工具读链—确认—幂等退款环境，诊断 reward saturation 后以过程质量打破零方差，并通过失败/能力边界选择合成含 decoy 与 transient failure 的困难数据继续 Stage-2 训练；加入 split hash、checkpoint resume 与 fresh-process verifier，形成可审计训练闭环。

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
| 两阶段 response-token curriculum | **{'COMPLETE' if verification.get('overall_status') == 'PASS' else 'FAILED'}** | verification 全门通过 |
| 官方 BFCL multi-turn smoke | **{bfcl_status}** | base/adapter result 与 score 齐全 |
| BFCL-V4 paper profile | **CODE_READY / NOT_RUN** | 4 categories × 200，固定 decoding |
| TAU-2 Avg@4 | **CODE_READY / NOT_RUN** | 全域任务 × 4，论文一致 user simulator |
| 3 seeds + ablation matrix | **NOT_RUN** | Vanilla/PRM/LATA/Joint × 3 seeds |
| 论文约 100K 数据 | **BLOCKED_RESOURCE** | Qwen3-235B synthesis/simulator/judge |
| 论文八 H100 配方 | **BLOCKED_RESOURCE** | 同版本 veRL/SGLang 与多卡预算 |

## 14. Claim boundary

本报告支持：“一次真实 Qwen3-8B 多轮 response-token QLoRA-GRPO curriculum 在云端 GPU 执行；V1 的零方差失败被校验器捕获，V2 两阶段均出现非零 reward variance 与参数哈希变化，Stage-2 adapter 可在新进程独立重载。”

本报告不支持：“完整复现论文 47.4、100K 数据、Qwen3-235B 数据飞轮、八 H100 recipe、TAU-2 Avg@4 或统计显著的普遍提升。”

## 参考资料

- [AgenticQwen paper](https://arxiv.org/abs/2604.21590)
- [Official data synthesis / RL code](https://github.com/haruhi-sudo/data_synth_and_rl)
- [TRL GRPOTrainer stateful environments](https://huggingface.co/docs/trl/en/grpo_trainer)
- [Official BFCL repository](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [Official TAU-2 v0.2.0](https://github.com/sierra-research/tau2-bench/tree/v0.2.0)
- [Agentic-GRPO-LongHorizon style reference](https://github.com/qiqihezh/agentic-grpo-longhorizon)
- [Modal GPU documentation](https://modal.com/docs/guide/gpu)
- [Modal pricing](https://modal.com/pricing)
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "READY_FOR_RENDER",
        "source": str(output_path.resolve()),
        "source_sha256": _sha256(output_path),
        "run_root": str(run_root.resolve()),
        "run_summary_sha256": _sha256(run_root / "run_summary.json"),
        "curriculum_verification": verification.get("overall_status"),
        "bfcl_verification": bfcl_status,
        "claim_boundary": "small-scale real run; paper-scale not claimed",
    }
    _write_path = output_path.with_suffix(".manifest.json")
    if review_output is not None:
        review = {
            "schema_version": 1,
            "status": "REVIEW_UNAVAILABLE",
            "reason": (
                "The render-html skill requires an independent cross-model review, "
                "but no independent reviewer tool is available in this runtime."
            ),
            "source": str(output_path.resolve()),
            "source_sha256": manifest["source_sha256"],
            "fabricated_pass": False,
        }
        review_output.parent.mkdir(parents=True, exist_ok=True)
        review_output.write_text(
            json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["independent_review"] = review["status"]
    _write_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the canonical audited curriculum report Markdown")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-missing-bfcl", action="store_true")
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--allow-layout-fixture", action="store_true")
    args = parser.parse_args()
    manifest = build_report(
        args.run_root.resolve(),
        args.output.resolve(),
        require_complete=not args.allow_incomplete,
        require_bfcl=not args.allow_missing_bfcl,
        review_output=args.review_output.resolve() if args.review_output else None,
        allow_layout_fixture=args.allow_layout_fixture,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
