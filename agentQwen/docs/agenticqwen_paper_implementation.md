# AgenticQwen 原文算法实现说明

这条代码路径专门实现论文 §3.3 的 **Agentic Data Flywheel**，与仓库里已经跑过的退款课程实验相互独立。退款实验仍是有效的 QLoRA-GRPO 工程证据，但它不是论文所述的行为树生成算法；不能把两者混写成同一项结果。

## 1. 算法闭环

```text
SynthAgent 线性 happy path（Round 0）
  → 小策略 QLoRA-GRPO
  → 小策略在 mock user / stateful tool 环境中 rollout
  → 强模型读取轨迹并扩展行为树
  → 对选定分支做 branch-to-task inversion
       b ↦ environment state + user instruction + agent instruction
  → 生成 normal_path / hack_path / rubric
  → 强模型试做候选任务
  → 仅保留“解题成功 且 命中预设分支”的样本
  → 从上一轮 LoRA adapter 继续 GRPO
  → 重复三轮
```

`paper_grpo_train.run_pipeline` 按这个顺序真正交替执行，而不是预先把三轮困难样本一次性生成好。第 `k+1` 轮行为树扩展的输入是第 `k` 轮保存的 `eval_after_traces.jsonl`。

## 2. 与原文的逐项对应

| 原文机制 | 代码位置 | 当前状态 |
|---|---|---|
| SynthAgent 线性工作流种子 | `paper_flywheel.linear_seed_tree` | 已实现微型航旅 happy path |
| 从当前 rollout 扩展行为树 | `StrongModel.expand_behavior_tree`、`evolve_one` | 已实现；训练模式强制非空 policy rollout |
| branch-to-task inversion | `StrongModel.invert_branch` | 已实现环境、用户、Agent 三路反演 |
| mock user 对抗诱导 | `UserInput`、`ask_user` | 已实现多轮诱导与 follow-up；对抗任务至少经历一次用户回合 |
| mock tool / 可变环境状态 | `PaperFlightEnvironment` | 已实现状态化工具、读后写约束和非法动作阻断 |
| normal / hack path | `AgenticTask.normal_path` / `hack_path` | 已实现并写入任务证据 |
| objective rubric reward | `RubricItem`、`_score_rubric` | 已实现，严格归一化到 `[0,1]` |
| 强模型先试做再过滤 | `validate_candidate` | 同时检查 reward 达标和 intended branch hit |
| Round 0–3 持续训练 | `paper_grpo_train.run_pipeline` | Qwen3-8B NF4 + LoRA-GRPO 入口已实现 |
| Qwen3-235B 多角色强模型 | `OpenAICompatibleStrongModel` | vLLM/SGLang 接口已实现，尚未在本次本地 contract run 中启动 235B |
| 约 100K 数据及论文分布式规模 | 配置边界 | 未执行、未声称 |

## 3. 一条生成任务包含什么

每条 `AgenticTask` 同时包含：

1. `environment_state`：航班状态、替代交通、会员等级等，只传给 mock tool；
2. `user_input`：自然请求、背景、对抗策略和诱导成功后的 follow-up，只传给 mock user；
3. `agent_instruction`：待训练策略必须遵守的 SOP；
4. `normal_path`：选定行为树分支对应的合规轨迹；
5. `hack_path`：被用户诱导后可能出现的违规轨迹；
6. `rubric`：必做动作、禁做动作、最终状态、分支命中、无非法写、用户交互等客观检查。

序列化时还会通过 `agenticqwen_official_compat` 输出官方仓库使用的字段名：`test_policy`、`task_background`、`user_escape_strategy`、`hack_success_user_background`、`tool_return_expected.normal_path/hack_path` 和 `rubrics`。因此本实现的强类型数据可以直接转换到官方 verl `extra_info` 结构，而不是另造一套含义不同的训练格式。

策略 prompt 不包含 `normal_path`、`hack_path` 或 `source_branch_id`。它们只存在于环境/评测侧，避免答案泄漏。环境的审计函数也全部为私有方法，不会被 TRL 暴露成模型工具。

## 4. 两种运行模式

### 算法契约模式（不需要模型）

```bash
./run_agenticqwen_paper.sh
```

它使用明确标注的 `deterministic-contract-backend`，完整检查三轮树增长、任务反演、mock user、工具状态、rubric、强模型过滤和证据 SHA-256。此模式不能被描述为“运行了 Qwen3-235B”。

### 真正的交替训练模式

先用 vLLM 或 SGLang 提供 Qwen3-235B 的 OpenAI-compatible endpoint，然后把配置改为：

```json
{
  "teacher": {
    "backend": "openai_compatible",
    "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "base_url": "http://127.0.0.1:8000/v1"
  }
}
```

再运行：

```bash
MODE=train CONFIG_PATH=configs/agenticqwen_paper_micro.json \
  OUTPUT_ROOT=artifacts/agenticqwen_paper_train ./run_agenticqwen_paper.sh
```

训练策略是 `Qwen/Qwen3-8B`，以 NF4 运行时量化加载，在注意力和 MLP 投影层训练 LoRA；四个训练阶段依次为 Round 0、1、2、3。Round 0 会把同一条 SynthAgent 线性工作流实例化到多个目的地上下文，后续轮次再加入反演得到的行为树分支。微型配置每轮只跑少量 step，用于验证算法链路，不用于复述论文的 100K 数据或公开 benchmark 数字。

## 5. 证据与停止条件

算法契约产物位于 `artifacts/agenticqwen_paper_micro/flywheel/`：每轮都有行为树、训练任务、教师验证轨迹和 SHA-256。完整训练还会为每轮保存 adapter、训练前后参数 hash、训练/评测轨迹和 adapter 链接关系。

`artifacts/agenticqwen_paper_micro/codelab_verification.json` 是第二环境验证证据：代码在 CodeLab Python 3.10 环境中重新编译并运行全量单测，再逐一核对三轮 tree/tasks/validation 的 9 个 SHA-256。该文件明确标注为 CPU contract/debug validation；虽然实例可见 H200，它没有下载模型、也没有把这次调试写成 GRPO 训练结果。

只有同时满足以下条件，训练运行才标记为 `completed`：

- 每轮 LoRA 可训练参数都发生变化；
- Round 1–3 都从上一轮 adapter 继续训练；
- 行为树扩展消费的是真实 policy rollout；
- 候选任务经过“教师解出 + 命中目标分支”双门过滤；
- reward 始终在 `[0,1]`。
