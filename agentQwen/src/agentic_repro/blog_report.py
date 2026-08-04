"""Build the canonical long-form AgenticQwen reproduction report."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .blog_renderer import render_file


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _delta(before: float, after: float) -> str:
    return f"{(after - before) * 100:+.1f} pt"


def _excerpt(
    path: Path,
    start_marker: str,
    end_marker: str,
    *,
    end_offset: int = 0,
) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(
        (index for index, line in enumerate(lines) if start_marker in line),
        None,
    )
    if start is None:
        raise RuntimeError(f"Excerpt start marker not found in {path}: {start_marker}")
    end = next(
        (index for index in range(start, len(lines)) if end_marker in lines[index]),
        None,
    )
    if end is None:
        raise RuntimeError(f"Excerpt end marker not found in {path}: {end_marker}")
    return "\n".join(lines[start : end + end_offset + 1])


def _trajectory_steps(trajectory: dict[str, Any]) -> list[str]:
    steps: list[str] = []
    for event in trajectory["events"]:
        if event["event_type"] == "tool_call":
            tool = event["tool_name"]
            arguments = json.dumps(event["arguments"], ensure_ascii=False, sort_keys=True)
            steps.append(
                f"{len(steps) + 1}. **{tool}** — 参数 {arguments}；本步 reward Δ "
                f"{event.get('reward_delta', 0):.2f}"
            )
    return steps


def build_report(project: Path) -> tuple[str, dict[str, Any]]:
    artifacts = project / "artifacts"
    summary = _load(artifacts / "real_qwen3_8b" / "summary.json")
    verification = _load(artifacts / "real_qwen3_8b" / "verification.json")
    trajectory = _load(artifacts / "long_horizon" / "trajectory_qwen3.7_flash.json")
    diagnostic = _load(artifacts / "ablations" / "offline_reward_diagnostic.json")
    benchmark = _load(artifacts / "benchmarks" / "planned_manifest.json")
    curriculum_config = _load(project / "configs" / "curriculum_qwen3_8b.json")

    train = summary["metrics"]["train"]
    holdout = summary["metrics"]["holdout"]
    unseen = summary["metrics"]["unseen"]
    training = summary["training"]
    runtime = trajectory["runtime"]
    model = summary["model"]
    dataset = summary["dataset"]
    local_verify = verification["summary"]
    trajectory_verification = trajectory["verification"]
    trajectory_verdict = (
        "PASS"
        if trajectory_verification.get("success")
        and all(check.get("passed") for check in trajectory_verification.get("checks", []))
        else "FAIL"
    )
    tool_steps = "\n".join(_trajectory_steps(trajectory))
    cloud_dirs = sorted((artifacts / "cloud_curriculum").glob("*"))
    cloud_artifacts = [
        path
        for directory in cloud_dirs
        for path in directory.rglob("*")
        if path.is_file()
    ]
    cfg_training = curriculum_config["training"]
    cfg_model = curriculum_config["model"]
    fence = chr(96) * 3
    environment_excerpt = _excerpt(
        project / "src" / "agentic_repro" / "long_horizon_env.py",
        "def _create_refund",
        'return self._unsafe("WRONG_AMOUNT"',
    )
    grpo_excerpt = _excerpt(
        project / "src" / "agentic_repro" / "real_grpo.py",
        "sampled: list[int]",
        "advantages = [",
    )
    verification_excerpt = _excerpt(
        project / "src" / "agentic_repro" / "curriculum_train.py",
        'check("stage1 adapter exists"',
        'str(stage2.get("input_adapter_weights_sha256"))',
        end_offset=1,
    )

    report = f"""# 我把 AgenticQwen 拆成了一条可审计的长程 Agentic RL 工程链

从读论文、搭状态化工具环境、跑一条真实多轮轨迹，到本地 Qwen3-8B LoRA-GRPO、失败诊断和两阶段 curriculum 代码。这不是一份把所有框都涂绿的“完整复现”，而是一篇把**真跑过什么、失败在哪里、下一步怎么跑**逐项讲清楚的项目复盘。

[论文：AgenticQwen](https://arxiv.org/abs/2604.21590) · [项目代码](../README.md) · [真实训练摘要](../artifacts/real_qwen3_8b/summary.json) · [完整多轮轨迹](../artifacts/long_horizon/trajectory_qwen3.7_flash.md)

> **先给结论。** 当前真正完成的是：一条 qwen3.7-flash 驱动的 8-turn、5-tool 状态化轨迹；一次本地 Qwen3-8B MLX 4-bit 的受限 LoRA group-relative 训练；以及 checkpoint 哈希和 fresh-process replay。据当时 AutoDL 终端观察，云端标准权重已下载并关机保留，但远端 marker 尚未取回本地；response-token GRPO、失败驱动 Stage 2、SFT、BFCL-V4 和 TAU-2 都还没有真实结果。

| 证据状态 | 我在这篇报告里的定义 |
|---|---|
| **COMPLETE** | 实际运行，留下模型、轨迹、哈希或独立验证证据 |
| **PARTIAL_RUN** | 实际运行，但规模、动作空间或任务语义弱于论文 |
| **CODE_READY** | 代码、配置、测试或 dry-run 已完成，没有模型结果 |
| **NOT_RUN / BLOCKED** | 尚未执行，或需要当前环境之外的算力与数据生成器 |

## 一、我为什么想做这个项目

我一开始以为，Agentic RL 最难的地方是把 Qwen3-8B 放到显卡上，然后接一个 GRPO Trainer。真正做起来以后才发现，训练反而只是最里面的一层。外面还套着用户、工具、环境状态、Reward、数据演化和评测器。

只要其中一层是假的，最后的“训练成功”就很可能只是日志里多了一行 loss。模型说“退款已经处理”，不代表环境真的多了一笔退款；LoRA 文件写到了磁盘，也不代表参数发生了变化；Benchmark 命令能够生成，更不代表论文分数已经复现。

所以我把目标换成了一个更严格的问题：**能不能把论文拆成一条每一步都有验收门的工程链？** 哪一层真跑了，就给出轨迹、模型哈希和独立重放；哪一层只有代码，就明确写 CODE_READY；需要 235B 合成器和多卡训练的部分，也不再用架构图冒充结果。

这篇文章的主线因此不是“我们追平了 47.4”，而是：

1. 论文为什么要用数据飞轮解决 Agentic RL 的学习信号问题；
2. 一个长程工具任务怎样变成可以训练和验证的环境；
3. 本地最小训练到底更新了什么，又为什么出现 group saturation；
4. 如何把真实失败转成下一轮更难的数据；
5. 哪些云端训练和官方 benchmark 仍然欠着。

## 二、论文真正解决的不是一个 loss，而是数据会不会继续变难

AgenticQwen 的动机很现实：工业场景需要大量、低延迟的工具调用，把每次请求都交给 235B 模型并不经济。论文希望通过合成数据与多轮 RL，把 agentic 能力训练到 8B 和 30B-A3B 这类小模型上，同时避免静态数据很快被学完。

论文设计了两条数据飞轮。Reasoning flywheel 从模型答错的问题出发，做 self-instruct、persona injection，并用多次求解的一致性过滤保留可验证难题。Agentic flywheel 则从线性 workflow 开始，把它扩成 behavior tree，再把分支反向变成环境状态、用户指令和 agent 指令，最后加入 adversarial mock user 给交互施压。

![论文 Figure 1：Reasoning 与 Agentic 两条数据飞轮](images/figure1_dual_flywheel.png)

这里最关键的不是“多合成一点数据”，而是新数据来自模型当前不会的地方。它试图形成一个闭环：

**训练 → rollout 暴露失败 → 把失败结构化 → 生成更难但仍可验证的任务 → 再训练。**

论文使用 Qwen3-235B 充当合成器、用户与工具模拟器、以及 rubric reward judge，训练数据约 100K。报告中的论文结果也必须和本项目结果分开：Qwen3-8B 在论文表 1 中从平均 23.8 提升到 AgenticQwen-8B 的 47.4；这两个数字是**论文目标和参考上限**，不是本项目复现出来的分数。

![论文 Figure 2：Round 0 到 Round 3 的论文飞轮结果，不是本项目曲线](images/figure2_flywheel_results.png)

| 维度 | 论文 AgenticQwen | 本项目目前 |
|---|---|---|
| 数据规模 | 约 100K | 官方数据 37,401 行中固定选 16 行做本地实验 |
| 强模型角色 | Qwen3-235B 合成、模拟、评审 | 一条 API 轨迹使用 qwen3.7-flash；训练模型是 Qwen3-8B |
| 训练循环 | Round 0–3，持续扩数据 | 本地单阶段最小实验；两阶段代码就绪但未跑 |
| Benchmark | TAU-2、BFCL-V4 Multi-Turn | 只有 dry-run manifest，没有分数 |
| 可宣称结果 | 论文 8B 平均 47.4 | 本地动作选择实验与 1 条状态化轨迹 |

## 三、先把“复现”拆成 Plan → Execute → Verify

我没有把 Plan、Executor、Verifier 包装成三个神奇的自主 Agent。它们更像三层职责分离：计划先定义什么叫完成，执行器负责留下产物，验证器不相信执行器自己说“completed”。

**Plan** 锁定论文 ID、目标、排除项、依赖和验收条件。**Executor** 运行数据准备、训练、轨迹和报告任务，并为 artifact 记录 SHA-256。**Verifier** 再独立检查 reward、数据来源、adapter 是否真的变化、split 是否隔离，以及保存后的 checkpoint 能否在新进程里重放。

Reporter 只应该消费 Verifier 接受的证据。这里的原则很简单：**训练完整性 PASS 与算法效果提升是两个不同结论。**

| 阶段 | 回答的问题 | 主要代码 |
|---|---|---|
| Plan | 论文哪些主张要复现？通过条件是什么？ | [planner.py](../src/agentic_repro/planner.py) |
| Execute | 任务、训练、轨迹和哈希如何落盘？ | [executor.py](../src/agentic_repro/executor.py) |
| Verify | 模型真的更新了吗？环境真的成功了吗？ | [verifier.py](../src/agentic_repro/verifier.py) |
| Report | 哪些证据允许进入最终长文？ | [blog_report.py](../src/agentic_repro/blog_report.py) |

早期的 Plan–Execute–Verify 是结构化 smoke pipeline；真正的两阶段训练由 [curriculum_train.py](../src/agentic_repro/curriculum_train.py) 承担。后者增加了 adapter 链接、untouched holdout、optimizer step、失败轨迹 provenance 和 fresh-process gate。

## 四、先把一笔“重复扣款退款”做成真正的环境

为了让新手能看到 Agentic RL 到底在训练什么，我选了一条非常具体的任务：用户说机械键盘被扣了两次款，但不记得订单号，希望退掉重复的 199 元。

这不是一问一答。Agent 必须先追问身份，再按依赖读取订单、支付历史和退款政策；退款属于写操作，必须等用户明确确认；最后 charge ID、金额和 idempotency key 都要由环境检查。模型最后说得再漂亮，只要环境状态没有生成正确的 RF-2026-00081，就算失败。

真正阻止模型“跳步骤”的不是 prompt，而是环境里的硬门：

{fence}python
{environment_excerpt}
{fence}

这段代码把身份、read-before-write、明确确认、正确 charge 和正确金额全部变成可执行条件。它也解释了为什么 final answer 不能当 ground truth：写操作只有通过这些门，环境状态才会改变。

真实保存下来的工具序列是：

{tool_steps}

这条轨迹由 DashScope 的 **{trajectory['policy']['model']}** 实际驱动，共 {runtime['agent_turns']} 个 agent turns、{runtime['tool_calls']} 次工具调用、{runtime['events']} 个事件和 {runtime['user_messages']} 条用户消息。它最终只退款 CHG-9002，金额 CNY 199，且在写操作前获得明确确认。

![论文 Figure 3：论文中的工业分析案例；本项目用退款任务实现同类状态化约束](images/figure3_case_study.png)

这里要分清两套环境：

- [long_horizon_env.py](../src/agentic_repro/long_horizon_env.py) 是五工具 demo，已经跑出上面的真实 API 轨迹；
- [curriculum_env.py](../src/agentic_repro/curriculum_env.py) 是七工具训练环境，支持 decoy、瞬时工具失败、错误恢复和 failure taxonomy，目标是给 response-token GRPO 使用，但云端训练尚未发生。

## 五、为什么我把评测放在训练前面

长程任务的终局失败，可能来自完全不同的原因：身份没核对、参数填错、漏读政策、没有确认就写入、工具报错后没有恢复、陷入循环，或者 final answer 与环境状态不一致。如果只看最后一句话，很多失败会被误判为成功。

所以我把评测拆成三层：

| 评测层 | 检查什么 | 防止的假阳性 |
|---|---|---|
| State verifier | 最终 charge、amount、refund state | 模型口头声称已经完成 |
| Process ledger | 读链、用户确认、工具错误与恢复 | 碰巧写对最终状态 |
| Fresh reload | 保存 adapter 后在独立进程重放 | 内存残留、空 checkpoint 或错加载 |

真实多轮轨迹经过 9 项环境检查，9/9 PASS：身份已验证、支付记录和政策已读取、明确确认已出现、只退款重复扣款、金额正确、环境生成退款记录、最终答案包含退款凭证和预计到账时间。

本地 LoRA checkpoint 也不是靠训练脚本自证。验证器重新加载权重和 adapter，在新进程中复放 train、same-task prompt holdout 与 unseen split，并检查模型权重 hash、数据 hash、adapter hash 和非零 LoRA delta。结果为 {local_verify['pass']} PASS / {local_verify['fail']} FAIL。

> 这两组 PASS 只证明 harness 与 artifact 完整：它们不自动推出“算法取得了可迁移提升”，更不推出“复现了论文 47.4”。

## 六、Reward 到底在奖励什么

论文、已运行的本地实验、以及代码就绪的多轮 curriculum，用的不是同一种 reward。

| 场景 | Reward 语义 | 当前证据 |
|---|---|---|
| 论文 AgenticQwen | Qwen3-235B 按 rubric 子目标完成比例给 0 到 1 | 论文方法 |
| 本地 action-masked GRPO | 四个候选工具中，下一工具是否选对，0 或 1 | 已运行 |
| 多轮 curriculum | outcome success + {cfg_training['process_reward_weight']:.2f} × process score | 代码就绪，未训练 |

多轮环境中的 outcome 仍是主项：身份、read-before-write、确认、正确 charge 和正确金额全部满足，退款才算成功。Process reward 用来区分过程质量，例如奖励数据依赖、错误恢复和合法 schema，惩罚越权写入、重复工具、跳过必需读取和超长循环。

为什么需要过程信号？因为二元 outcome 很容易让整组 rollout 全对或全错。一旦组内没有方差，GRPO 看到的 advantage 全是零。过程 reward 只有在轨迹真的暴露出不同决策路径时才可能补信号；离线给同一个动作换个分数，并不会凭空创造多轮能力。

## 七、数据怎么切，以及 SFT 到底完成了没有

本地实验读取 [alibaba-pai/AgenticQwen-Data](https://huggingface.co/datasets/alibaba-pai/AgenticQwen-Data) 的 {dataset['source_rows']:,} 行，并用固定 seed {dataset['selection_seed']} 选择 {dataset['selected_rows']} 行：{training['train_tasks']} 个训练任务、{training['unseen_tasks']} 个完全 unseen 任务。same-task prompt holdout 只是同一个工具目标换一种问法，它不是 OOD。

模型侧使用的是 **{model['id']}**，权重文件 {model['weights_bytes'] / 1024**3:.2f} GiB，SHA-256 为 {model['weights_sha256'][:16]}…。训练时只给模型四个 task-specific 候选工具，让它输出一个 action token。这保留了 group sampling、reward、advantage、clipped update 和 LoRA 参数更新，但没有训练 JSON arguments，也没有把多轮 state transition 放进 optimizer。

我原本设想过一条标准的 **LoRA SFT → LoRA GRPO** 后训练链，但当前仓库没有 SFTTrainer、SFT 配置、SFT adapter 或训练日志。因此这篇报告不会写“SFT 已完成”。

未来正确的 SFT 阶段应该是：

1. 从成功 teacher trajectory 构造 assistant/tool-call token mask；
2. 用 LoRA 做监督微调，先学会合法 schema 和基础读写顺序；
3. 在冻结环境上通过 eval gate；
4. 再用同一个 adapter 初始化多轮 GRPO。

这是一项明确的 **NOT IMPLEMENTED**，不是隐藏在“后训练”三个字里的已完成工作。

## 八、Qwen3-8B 到底怎么加载：MLX 4-bit 与 NF4 QLoRA

这个项目先后出现过两套 4-bit 路线，很容易混在一起。

| 实验 | 权重形态 | 更新方式 | 状态 |
|---|---|---|---|
| Mac 本地最小实验 | 已预量化 Qwen3-8B-MLX-4bit | MLX LoRA | 已运行 |
| AutoDL curriculum | 标准 Qwen/Qwen3-8B checkpoint | BitsAndBytes 加载时转 NF4，只训练 LoRA | 终端观察已下载；本地证据未取回；训练未跑 |

NF4 是 QLoRA 加载时的量化表示，不要求在 Hub 上另找一个独立的“小型 NF4 模型文件”。云端下载标准 checkpoint；[curriculum_train.py](../src/agentic_repro/curriculum_train.py) 通过 BitsAndBytesConfig 设置 load-in-4bit、NF4、BF16 compute 和 double quant，再冻结 base 4-bit 权重、更新 LoRA adapter。

云端配置使用 LoRA rank {cfg_model['lora_rank']}、alpha {cfg_model['lora_alpha']}，目标模块覆盖 q/k/v/o projection 与 MLP 的 gate/up/down projection。每个阶段计划 {curriculum_config['stages']['stage1']['max_steps']} 个 step，group 内 {cfg_training['num_generations']} 个 generation，最大 {cfg_training['max_tool_calling_iterations']} 次工具迭代。

据当时 AutoDL 终端观察，标准 Qwen3-8B snapshot 显示 16/16 文件、约 16.40 GB，实例随后关机保留数据盘。远程 .modelscope_complete.json 尚未取回本地并纳入本报告 manifest，因此这只是 **operator-observed** 基础设施状态，不是可独立验证的训练证据。本地 cloud_curriculum 目录当前只有 {len(cloud_dirs)} 个空 run 目录、{len(cloud_artifacts)} 个运行文件。**下载模型不是 optimizer step，也不是训练完成。**

## 九、GRPO 如何更新，以及本地结果到底说明了什么

先用大白话说 GRPO：同一道题让模型采样多条候选，不问“绝对有多好”，而看它在这一组里比同伴更好还是更差。

对同一个 prompt 的第 i 条 rollout，先计算组内标准化 advantage：

$$A_i = (r_i - mean(r)) / (std(r) + epsilon)$$

举一个四条 rollout 的例子。reward = [1, 1, 0, 0] 时，均值是 0.5，成功轨迹的 advantage 为正，失败轨迹为负，模型能学会提高前者概率；如果 reward = [1, 1, 1, 1]，标准差为零，整组没有相对学习信号。

代码中的零方差门和 advantage 计算就是这个逻辑：

{fence}python
{grpo_excerpt}
{fence}

再用 PPO 风格的 ratio clip 限制单次更新幅度。ratio 可以理解为“当前策略给这个动作的概率 ÷ rollout 时旧策略的概率”，clip 防止一次更新走得太远。LoRA 参数接收梯度，base 4-bit 权重保持冻结。已运行的 MLX 版本用四选一 action token 做这个闭环；代码就绪的 TRL 版本则让模型生成普通 assistant tokens、JSON arguments，并由 stateful environment 接管多轮工具循环。

本地训练一共尝试 {training['attempted_groups']} 个 group，每组 {training['group_size']} 个 rollout。只有 {training['updated_groups']} 个 group 有非零 reward 方差，因此只执行了 {training['optimizer_steps']} 次 optimizer update，用时 {training['training_seconds']:.1f} 秒，峰值内存 {summary['runtime']['peak_memory_gib']:.2f} GiB。

| Split | Base | Adapter | 变化 |
|---|---:|---:|---:|
| Train | {_pct(train['accuracy_before'])} | {_pct(train['accuracy_after'])} | {_delta(train['accuracy_before'], train['accuracy_after'])} |
| Same-task prompt holdout | {_pct(holdout['accuracy_before'])} | {_pct(holdout['accuracy_after'])} | {_delta(holdout['accuracy_before'], holdout['accuracy_after'])} |
| Fully unseen | {_pct(unseen['accuracy_before'])} | {_pct(unseen['accuracy_after'])} | {_delta(unseen['accuracy_before'], unseen['accuracy_after'])} |

LoRA-B norm 从 {training['lora_b_norm_before']:.3f} 变成 {training['lora_b_norm_after']:.3f}，adapter SHA-256 从 {training['adapter_initial_sha256'][:12]}… 变成 {training['adapter_final_sha256'][:12]}…。这证明参数确实更新并写入磁盘。

但效果结论必须保守：模型学到了一部分小样本工具映射，并在同任务换措辞时有所改善；4 个 fully unseen 任务仍停在 {_pct(unseen['accuracy_after'])}，没有形成可报告的泛化证据。

## 十、20/24 个 group 没有梯度，是这次最重要的负结果

本地训练最值得写进项目复盘的，不是 +16.7 pt 的 same-task holdout，而是 **{training['attempted_groups'] - training['updated_groups']}/{training['attempted_groups']} 个 rollout group 饱和**。

当一个 group 的 16 次采样全对或全错时，所有 reward 相同，标准化 advantage 就是零。这不是“loss 看起来小”，而是这一组根本无法贡献学习信号。只有 {training['updated_groups']} 个非零方差组能够更新，零方差率达到 {diagnostic['outcome_zero_variance_rate'] * 100:.2f}%。

我也做了一个 PRM-Lite 离线反事实诊断：在已经采样好的动作上追加 0.3 权重的过程分。结果 active group 仍然是 {diagnostic['outcome_nonzero_variance_groups']} → {diagnostic['shaped_nonzero_variance_groups']}，新激活组为 {diagnostic['newly_activated_groups']}。

> 这是一个负结果，不是消融提升。它只说明：单步四选一环境没有暴露身份、读链、确认、错误恢复等过程差异，所以给旧动作重新打分也无法解决 saturation。要真正验证 process reward，必须在多 token、多 turn 的在线 rollout 中重新训练。

这个发现反过来决定了下一阶段设计：不再只调 learning rate，而是让任务、环境和 rollout diversity 一起变。

## 十一、失败如何变成下一轮更难的数据

两阶段 curriculum 的代码已经实现，但没有云端 optimizer 证据。完整链路是：

1. Stage 1 在基础退款任务上做 response-token QLoRA-GRPO；
2. 用冻结 probe rollout 读取真实 environment state，而不是让模型自己解释失败；
3. 把终局分成 IDENTITY_ERROR、ORDERS_NOT_READ、PAYMENT_NOT_INSPECTED、POLICY_NOT_INSPECTED、CONFIRMATION_MISSING、WRONG_CHARGE、WRONG_AMOUNT、TOOL_ERROR_NOT_RECOVERED、TIMEOUT_OR_LOOP、REFUND_NOT_CREATED；
4. 按失败类型生成 decoy order、瞬时工具错误、更紧确认约束等 hard task；
5. 混入旧任务 replay，避免只适配难例；
6. Stage 2 从 Stage-1 adapter 继续训练；
7. 在 untouched holdout 上评估，再做 fresh-process reload。

| 论文 flywheel | 本项目的小规模工程映射 |
|---|---|
| 235B 扩 behavior tree | 确定性代码按 terminal failure 合成难例 |
| 235B 模拟 user/tool | typed tools + state machine |
| 约 100K、Round 0–3 | 计划 8 个 hard task + 4 个 replay，两阶段 |
| 强模型验证可解性 | deterministic ground truth + state verifier |

例如，TOOL_ERROR_NOT_RECOVERED 会触发一条带可重试 payment-history timeout 的难例；模型必须在同一 episode 里恢复，而不是碰巧走通无错误路径。

关键 provenance 已经写进设计：Stage-2 synthesis manifest 必须记录 Stage-1 probe trace 的路径和 SHA-256；Stage 2 必须保存 input adapter hash；Verifier 必须检查 task split isolation、非零 optimizer step、adapter 链接和独立进程重载。下面这段门禁专门防止“Stage 2 其实没继承 Stage 1”或 holdout 泄漏：

{fence}python
{verification_excerpt}
{fence}

这使它成为论文思想的一条**可执行缩小版映射**，但在 run_summary、stage1/stage2 summary、adapter、trace 和 verification 全部出现以前，它的状态只能是 CODE_READY / NOT_RUN。

## 十二、Benchmark 代码写好了，不代表 Benchmark 跑了

论文主要看 TAU-2 和 BFCL-V4 Multi-Turn。当前项目已经把官方 evaluator 的版本、类别、base/adapter 变体和命令写进 [benchmark_runner.py](../src/agentic_repro/benchmark_runner.py)，也生成了固定 manifest。

当前 smoke 计划是：

- BFCL：4 个 multi-turn category × {benchmark['scope']['bfcl_tasks_per_category']} 个任务，共 {benchmark['scope']['bfcl_total_tasks']} 个；
- TAU-2：airline、retail、telecom 各 {benchmark['scope']['tau2_num_tasks_per_domain']} 个任务，1 trial；
- 分别评估 base 与 adapter；
- 计划固定的 BFCL evaluator 版本 {benchmark['official_versions']['bfcl_eval']}，计划固定的 TAU-2 tag {benchmark['official_versions']['tau2_git_tag']}。

但 [planned_manifest.json](../artifacts/benchmarks/planned_manifest.json) 的状态就是 **{benchmark['status']}**，artifact_inventory 为空。没有 inference result、score CSV 或 TAU-2 episode，就没有任何可报告的 benchmark 分数。

另外，即使 smoke 跑完，它也不自动等价于论文表格：TAU-2 的 user simulator、trial 数和 Avg@4 口径必须对齐；本地 adapter 又是在 action-masked 任务上训练的，不一定能直接改善 unrestricted function calling。

## 复现账本

下面是这次改写后采用的事实矩阵。它故意没有“完成率 90%”这种容易误导的总分。

| 项目 | 状态 | 已有证据 | 还缺什么 |
|---|---|---|---|
| 论文方法与结果边界 | COMPLETE | arXiv 2604.21590、论文图和表 | 论文实验本身不由本项目复验 |
| 真实多轮工具轨迹 | COMPLETE | 8 turns、5 tools、16 events、9/9 verifier | 扩成任务集和多 seed |
| 本地 Qwen3-8B LoRA-GRPO | PARTIAL_RUN | 24×16 rollout、8 updates、adapter hash、fresh reload | unrestricted JSON 和多轮 state |
| Fully unseen 泛化 | PARTIAL_RUN | 4 tasks，25.0% → 25.0% | 更大分层集、至少 3 seeds |
| Process reward 诊断 | PARTIAL_RUN | 离线 4 → 4 active groups | 在线多轮训练消融 |
| Stateful response-token GRPO | CODE_READY / NOT_RUN | TRL environment_factory、NF4 LoRA、单测 | 真实云端 stage artifacts |
| 失败驱动 Stage 2 | CODE_READY / NOT_RUN | taxonomy、synthesis、replay、adapter chaining | Stage-1 failure 与 Stage-2 checkpoint |
| SFT LoRA | NOT IMPLEMENTED | 仅有设计说明 | 数据、trainer、adapter、eval |
| BFCL-V4 | PLANNED_NOT_RUN | 官方命令与固定 task IDs | inference 与 score artifact |
| TAU-2 | PLANNED_NOT_RUN | 三域命令与 profile | episode、Avg@4 |
| 论文约 100K 与 Round 0–3 | BLOCKED_RESOURCE | 论文参考 | 235B 合成/模拟/评审与多卡训练 |
| 论文 47.4 | NOT REPRODUCED | 只作为论文参考值 | 论文同口径完整训练与评测 |

测试目录中的 curriculum_deck_only fixture 含有人造排版指标，DO_NOT_REPORT.md 已标明不可作为实验结果。新版报告不会读取它；生成正式 cloud 报告时，[curriculum_report.py](../src/agentic_repro/curriculum_report.py) 也会拒绝缺少 cloud GPU、verification PASS 或官方 BFCL artifact 的 run。

## 十三、如果明天重新开卡，什么才算“训练完成”

AutoDL 开机以后，最短路径不是继续改报告，而是让以下证据全部落盘：

1. GPU、CUDA、PyTorch、TRL、Transformers 和 BitsAndBytes 的 runtime manifest；
2. Stage-1 baseline/probe/train trace、optimizer step 和 adapter SHA；
3. 由 Stage-1 failure 生成的 synthesis manifest 与 source trace SHA；
4. Stage-2 input adapter hash、final adapter、untouched holdout；
5. fresh-process reload 的独立 PID 和结果；
6. base/adapter BFCL smoke inference 与 score artifact；
7. 如果加入 SFT，单独保存 SFT data hash、token mask、adapter 和 eval gate。

只有这些门通过，报告里的“stateful response-token curriculum”才能从 CODE_READY 升级为 COMPLETE。即使 Stage 2 没有提升，也应把它保留为负结果，分析 failure composition、group variance、hard/replay ratio 和工具错误恢复，而不是删除 run。

对应的一键入口已经拆开：

- [run_curriculum_autodl_remote.sh](../run_curriculum_autodl_remote.sh)：云端环境、自举、断点模型检查与两阶段训练；
- [finalize_curriculum_project.sh](../finalize_curriculum_project.sh)：取回 artifact 后生成报告与 PPT；
- [run_benchmarks.sh](../run_benchmarks.sh)：官方 evaluator 入口；
- [configs/curriculum_qwen3_8b.json](../configs/curriculum_qwen3_8b.json)：唯一训练配方。

## 面试怎么讲

现在可以安全写进简历的一句话是：

> 基于 Qwen3-8B/MLX 实现小规模 LoRA-GRPO 工具决策训练闭环，完成 24×16 group rollout、adapter 参数哈希审计与 fresh-process replay；诊断 20/24 group 的二元奖励饱和，并实现 stateful multi-turn environment、failure taxonomy、困难样本 curriculum，以及官方 BFCL/TAU-2 evaluator 的 dry-run 接入路径（云端 curriculum 与 benchmark 尚待运行）。

这句话故意没有说“完整复现 AgenticQwen”，也没有说“跑完 SFT + GRPO”或“BFCL 提升”。面试时可以按 90 秒讲六件事：

1. 论文为什么需要数据飞轮，而不是无限重复静态任务；
2. 如何把客服退款做成有 typed tools、不可逆写操作和环境真值的状态机；
3. 如何组合 outcome、process ledger 与 fresh-process verifier；
4. 本地 LoRA-GRPO 真实发生的参数更新与 20/24 saturation；
5. 如何据此设计 failure-driven curriculum；
6. 哪些已经运行，哪些仍待云端实验和官方 benchmark。

如果被追问“最大的技术判断是什么”，答案不是某个超参，而是：**发现同一组 rollout 没有 reward 方差时，继续堆训练 step 没有意义；必须改变任务难度、轨迹空间和评测粒度，让模型真的产生可比较的行为差异。**

## 结语

做到这里，项目离论文意义上的完整复现还有明显距离。

Qwen3-8B 在本地确实发生了 LoRA 参数更新，一条真实长程轨迹也确实完成了身份验证、数据读取、用户确认和退款写入。但它们目前还是两条分开的证据链：本地训练是四选一 action token，多轮轨迹没有真正进入在线 GRPO；BFCL 和 TAU-2 也只有可执行路径，没有分数。

不过这次最有价值的结果，反而是 20/24 group 没有学习信号。它让我看到二元 Reward、任务粒度和 rollout diversity 如何彼此限制，也直接推动下一阶段变成“训练—发现失败—合成更难数据—再训练”的 curriculum。

下一次开机，真正要验证的不是脚本能不能启动，而是 Stage 1 的失败能不能形成可追溯的新任务，Stage 2 是否从原 adapter 继续更新，以及这些变化能不能在 untouched holdout 和官方 evaluator 上留下证据。

如果最后没有提升，也应该原样写下来。一个能解释为什么失败、知道下一步该补什么的 Agentic RL 项目，比一份只有漂亮架构图和虚假完成度的“完整复现”更值得写进简历。

---

资料： [AgenticQwen paper](https://arxiv.org/abs/2604.21590) · [官方数据与训练代码](https://github.com/haruhi-sudo/data_synth_and_rl) · [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) · [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) · [TAU-2](https://github.com/sierra-research/tau2-bench) · [视觉与叙事参考](https://yyhdbl.github.io)
"""

    evidence = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": {
            "paper_scale_claimed": False,
            "cloud_curriculum_completed": False,
            "sft_completed": False,
            "bfcl_completed": False,
            "tau2_completed": False,
        },
        "actual_runs": {
            "local_qwen3_8b_action_masked": {
                "status": "PARTIAL_RUN",
                "summary": str((artifacts / "real_qwen3_8b" / "summary.json").resolve()),
                "verification": verification["overall_status"],
            },
            "qwen3_7_flash_trajectory": {
                "status": "COMPLETE",
                "trajectory_id": trajectory["trajectory_id"],
                "verifier": trajectory_verdict,
            },
        },
        "not_run": {
            "cloud_run_file_count": len(cloud_artifacts),
            "sft": "NOT_IMPLEMENTED",
            "bfcl_tau2": benchmark["status"],
            "paper_round_0_3": "BLOCKED_RESOURCE",
        },
        "sources": {},
    }
    for relative in [
        "artifacts/real_qwen3_8b/summary.json",
        "artifacts/real_qwen3_8b/verification.json",
        "artifacts/long_horizon/trajectory_qwen3.7_flash.json",
        "artifacts/ablations/offline_reward_diagnostic.json",
        "artifacts/benchmarks/planned_manifest.json",
        "configs/curriculum_qwen3_8b.json",
    ]:
        source = project / relative
        evidence["sources"][relative] = {
            "sha256": _sha256(source),
            "bytes": source.stat().st_size,
        }
    return report, evidence


def write_report(project: Path, output_dir: Path) -> dict[str, Any]:
    report, evidence = build_report(project)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "TECHNICAL_REPORT.md"
    html_path = output_dir / "index.html"
    manifest_path = output_dir / "report_manifest.json"
    markdown_path.write_text(report, encoding="utf-8")
    rendered = render_file(
        markdown_path,
        html_path,
        title="AgenticQwen：从失败轨迹到下一轮训练",
        brand="AgenticQwen/notes",
    )
    evidence["report"] = {
        "markdown": str(markdown_path.resolve()),
        "markdown_sha256": _sha256(markdown_path),
        "html": str(html_path.resolve()),
        "renderer": "agentic_repro.blog_renderer",
        "rendered": rendered,
    }
    manifest_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_project = Path(__file__).resolve().parents[2]
    parser.add_argument("--project", type=Path, default=default_project)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project / "agenticqwen_report"
    )
    result = write_report(project, output_dir)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
