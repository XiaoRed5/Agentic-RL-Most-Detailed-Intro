"""多轮 agentic rollout loop —— agentic RL 的心脏。

对齐 slime example 的 asolve：反复 [policy 生成 → 解析动作 → env.step → 观测],
直到 terminated / truncated / aborted。产出一条 Trajectory，其中：

    每个 Turn 的 assistant_token_ids = policy 生成的文本 token   (loss_mask=1)
                 obs_token_ids       = 环境/工具/用户返回的 token (loss_mask=0)

这就是 token-level loss masking：**只对 policy 自己生成的 token 算 loss**，环境
返回的 token 一律 mask 掉。masking 错了(把 observation 也算进 loss)是 agentic RL
最常见、最隐蔽的 bug —— 会让模型去"拟合"检索/工具返回的内容，训练直接跑偏。
"""
from __future__ import annotations

from typing import Any, Optional

from agentic_rl.env.tau_env import TauEnv
from agentic_rl.rollout.policy import Policy, parse_tool_call
from agentic_rl.rollout.tokenizer import ByteTokenizer
from agentic_rl.utils.types import (
    ActionType,
    Status,
    Trajectory,
    Turn,
)

# 属于 BAO 的 𝒜_u(user-involved 动作集)：问用户 / 请求核验
USER_INVOLVED_ACTIONS = {"respond", "ask_user", "request_verification"}


def classify_action(name: str) -> ActionType:
    if name == "respond":
        return ActionType.RESPOND
    if name == "answer":
        return ActionType.ANSWER
    if name in ("", None):
        return ActionType.OTHER
    return ActionType.TOOL_CALL


def rollout(
    env: TauEnv,
    policy: Policy,
    tokenizer: ByteTokenizer,
    max_steps: int = 16,
) -> Trajectory:
    """跑一条完整轨迹。返回带 token 级 masking 的 Trajectory。"""
    task = env.task
    obs = env.reset()

    # 初始 prompt：system + 首轮 user。整段 loss_mask=0(不是 policy 生成的)。
    prompt_text = f"[SYSTEM] {task.system_prompt}\n[USER] {obs}\n"
    traj = Trajectory(
        task_id=task.task_id,
        domain=task.domain,
        prompt_token_ids=tokenizer.encode(prompt_text),
    )
    traj.metadata["reward_basis"] = task.reward_basis

    for step_i in range(max_steps):
        messages = _render_messages(traj, prompt_text)
        action, raw_text = policy.act(messages)

        # 解析失败 → ABORTED
        if action is None:
            parsed = parse_tool_call(raw_text)
            if parsed is None:
                turn = Turn(
                    index=step_i, action_type=ActionType.OTHER,
                    assistant_token_ids=tokenizer.encode(raw_text),
                )
                traj.turns.append(turn)
                traj.status = Status.ABORTED
                break
            action = parsed

        name = action.get("name", "")
        atype = classify_action(name)
        assistant_ids = tokenizer.encode(raw_text)

        # 执行环境
        result = env.step(action)
        # 环境返回文本 → observation token(loss_mask=0)
        obs_text = f"\n[OBS] {result.observation}\n"
        obs_ids = tokenizer.encode(obs_text)

        turn = Turn(
            index=step_i,
            action_type=atype,
            action_name=name,
            action_args=action.get("arguments", {}),
            assistant_token_ids=assistant_ids,
            obs_token_ids=obs_ids,
            immediate_reward=result.reward,
            is_user_involved=(name in USER_INVOLVED_ACTIONS),
            info=dict(result.info),
        )
        traj.turns.append(turn)

        if result.terminated:
            traj.status = Status.COMPLETED
            traj.outcome_reward = result.reward
            traj.reward_info = {"end_reason": env.end_reason,
                                "reward_breakdown": result.info.get("reward_breakdown", {})}
            break
        if result.truncated:
            traj.status = Status.TRUNCATED
            traj.outcome_reward = result.reward
            traj.reward_info = {"end_reason": env.end_reason}
            break
    else:
        # for 正常结束(没 break)= 达到 max_steps 循环上限
        traj.status = Status.TRUNCATED
        traj.outcome_reward = 0.0

    return traj


def _render_messages(traj: Trajectory, prompt_text: str) -> list[dict[str, Any]]:
    """把已发生的轨迹渲染成 messages(供 policy 参考)。离线 policy 多为脚本化，
    但保留此接口以对齐真机(真机 policy 需要完整对话历史做条件生成)。"""
    messages: list[dict[str, Any]] = [{"role": "system", "content": prompt_text}]
    for t in traj.turns:
        messages.append({"role": "assistant", "content": t.action_name or ""})
        if t.obs_token_ids:
            messages.append({"role": "user", "content": "[obs]"})
    return messages
