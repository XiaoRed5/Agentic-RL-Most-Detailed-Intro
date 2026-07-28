"""InfoPO 运行时：在真实模型上计算反事实信息增益(counterfactual info-gain)。

credit/credit.py 里的 counterfactual_info_gain 只做纯 KL 计算(model-agnostic)；
这里负责**从模型真实产出** factual / masked 两个下一步动作分布，再喂给它。

核心机制(对齐 survey ④ InfoPO)：
  对第 t 轮，它的 observation(环境反馈) o_t 到底有没有用？
  - factual 分支：prompt + turns[0..t](含 o_t) → 模型在"下一步"位置的动作分布 P_真
  - masked 分支：同样序列，但把 o_t 的 token 换成 pad → 分布 P_masked
  gain_t = KL(P_真 ‖ P_masked)。变化越大 = o_t 越关键。

这个 gain 在**零方差组**(所有轨迹 reward 相同、GRPO advantage 全 0)时顶上来当训练信号，
让"奖励学不动"的组靠信息增益继续供梯度 —— 正是 survey 说的 InfoPO 主战场。
"""
from __future__ import annotations

import torch

from agentic_rl.credit.credit import counterfactual_info_gain
from agentic_rl.utils.types import Trajectory


def _next_token_dist(model, token_ids: list[int], at_pos: int) -> torch.Tensor:
    """模型在序列位置 at_pos 处对 next-token 的分布(softmax over vocab)。

    at_pos = 已知 token 的最后一个下标；返回预测 token[at_pos+1] 的分布。
    """
    ids = torch.tensor([token_ids[: at_pos + 1]], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(ids)
    return torch.softmax(logits[0, at_pos, :], dim=-1)


def trajectory_info_gains(
    model,
    traj: Trajectory,
    pad_id: int,
) -> list[float]:
    """对轨迹每一轮算反事实 info-gain。返回长度 = n_turns 的列表(与 turn 对齐)。

    turn t 的 gain 衡量"turn t 的 observation 对 turn t+1 的动作选择有多大影响"。
    最后一轮没有"下一步动作"，gain=0。
    """
    n = traj.n_turns
    gains = [0.0] * n
    if n < 2:
        return gains

    # 预计算每个 turn 在完整序列中的 token 区间(prompt 之后)
    full = traj.all_token_ids()
    prompt_len = len(traj.prompt_token_ids)

    # 各 turn 的 [assistant起, obs起, obs止) 绝对下标
    spans = []
    cur = prompt_len
    for t in traj.turns:
        a0 = cur
        o0 = a0 + len(t.assistant_token_ids)
        o1 = o0 + len(t.obs_token_ids)
        spans.append((a0, o0, o1))
        cur = o1

    for t in range(n - 1):
        a0_next = spans[t + 1][0]           # 下一轮第一个 assistant token 的位置
        if a0_next == 0:
            continue
        pred_pos = a0_next - 1              # 预测 token[a0_next] 的位置

        # factual：原始序列
        p_fact = _next_token_dist(model, full, pred_pos)

        # masked：把 turn t 的 obs token 换成 pad
        _, o0, o1 = spans[t]
        if o1 <= o0:                        # 该轮没有 obs(纯 respond 且 user 无回填)
            gains[t] = 0.0
            continue
        masked = list(full)
        for i in range(o0, o1):
            masked[i] = pad_id
        p_mask = _next_token_dist(model, masked, pred_pos)

        gains[t] = counterfactual_info_gain(p_fact, p_mask)

    return gains


def group_infogain_advantages(
    model,
    group: list[Trajectory],
    pad_id: int,
) -> list[list[float]]:
    """对一个 group 算每条轨迹每个 response token 的 info-gain advantage。

    turn t 的 gain 广播到 turn t 的所有 assistant token(obs token 用 0)；
    然后 group 级归一化(与外部奖励 advantage 同尺度，便于融合)。
    """
    from agentic_rl.credit.credit import normalize_turn_values

    per_traj_gains: list[list[float]] = []
    pooled: list[float] = []
    for traj in group:
        g = trajectory_info_gains(model, traj, pad_id)
        per_traj_gains.append(g)
        pooled.extend(g)

    normed = normalize_turn_values(pooled) if len(pooled) > 1 else [0.0] * len(pooled)

    out: list[list[float]] = []
    cursor = 0
    for traj, g in zip(group, per_traj_gains):
        n = len(g)
        turn_adv = normed[cursor:cursor + n]
        cursor += n
        token_adv: list[float] = []
        for turn, a in zip(traj.turns, turn_adv):
            token_adv.extend([a] * len(turn.assistant_token_ids))
            token_adv.extend([0.0] * len(turn.obs_token_ids))
        out.append(token_adv)
    return out
