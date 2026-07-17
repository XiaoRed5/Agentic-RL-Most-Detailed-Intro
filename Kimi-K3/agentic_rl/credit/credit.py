"""信用分配(credit assignment) —— survey 这条线的核心深水区。三种打法并存。

多轮 agent 的核心难题：成功信号只在末尾出现一个，但功劳可能来自前面某个澄清轮。
到底把功劳记给哪一步？三种打法：

1. outcome —— 只用最终奖励，中间轮全 0(最朴素，Search-R1 式)。
2. R2G (UserRL) —— Reward-to-Go 折扣前传：把终点奖励按 γ 一路倒着折扣分给早期轮。
3. InfoPO —— 反事实信息增益：拿掉某轮反馈，看 agent 下一步决策分布变多少(KL)。

三者在**同一条 rollout** 上可切换对比 —— 这正是本项目"做深"的地方：你能亲眼看到
survey 说的"零奖励澄清轮"被不同方法救活的差异。
"""
from __future__ import annotations

from typing import Optional

import torch

from agentic_rl.utils.types import Trajectory


# ============ 1. Reward-to-Go (UserRL 折扣前传) ============
def reward_to_go(immediate_rewards: list[float], gamma: float = 0.8) -> list[float]:
    """R2G_t = r_t + γ·R2G_{t+1}(反向递推)。

    对齐 survey TravelGym 算例(γ=0.8)：把终点的成功奖励沿时间倒着折扣分给铺路的每一步，
    让"当时看着没用的那句澄清"也分到非零功劳。R2G 值可 >1(是折扣累加不是概率)。
    """
    n = len(immediate_rewards)
    r2g = [0.0] * n
    running = 0.0
    for t in reversed(range(n)):
        running = immediate_rewards[t] + gamma * running
        r2g[t] = running
    return r2g


def apply_r2g_to_trajectory(traj: Trajectory, gamma: float = 0.8) -> list[float]:
    """把 R2G 写进每个 Turn 的 .r2g 字段，返回 r2g 列表。"""
    imm = [t.immediate_reward for t in traj.turns]
    r2g = reward_to_go(imm, gamma)
    for turn, v in zip(traj.turns, r2g):
        turn.r2g = v
    return r2g


# ============ 2. 组内归一化(喂给 GRPO 前) ============
def normalize_turn_values(values: list[float], eps: float = 1e-6) -> list[float]:
    """survey：真正喂给 GRPO 前还会再做组内归一化 (值-μ)/σ 把尺度拉回。"""
    t = torch.tensor(values, dtype=torch.float32)
    if t.numel() <= 1:
        return [0.0 for _ in values]
    mean = t.mean()
    std = t.std(unbiased=False)
    return ((t - mean) / (std + eps)).tolist()


# ============ 3. InfoPO 反事实信息增益 ============
def counterfactual_info_gain(
    p_factual: torch.Tensor,       # [V] 看到真反馈后 agent 下一步动作分布 P(a|s,o)
    p_masked: torch.Tensor,        # [V] 盖住该反馈后的动作分布 P(a|s,mask)
    eps: float = 1e-8,
) -> float:
    """gain_t ≈ KL(P_真 ‖ P_masked)：拿掉这轮反馈，agent 下一步决策变了多少。

    变化越大 = 这轮反馈越关键，功劳越大(归因给触发它的动作)。稀疏场景也能精确定位
    哪轮有用，而不是像 R2G 只按位置摊分。
    """
    p = p_factual.clamp_min(eps)
    q = p_masked.clamp_min(eps)
    p = p / p.sum()
    q = q / q.sum()
    kl = (p * (p.log() - q.log())).sum()
    return float(kl.clamp_min(0.0))


def infopo_turn_gains(
    factual_dists: list[torch.Tensor],
    masked_dists: list[torch.Tensor],
) -> list[float]:
    """对每一轮算反事实 info-gain。两个列表一一对应每轮的(事实/反事实)下一步分布。"""
    assert len(factual_dists) == len(masked_dists)
    return [counterfactual_info_gain(f, m) for f, m in zip(factual_dists, masked_dists)]


# ============ 4. 方差门控(InfoPO 治零方差组) ============
def variance_gate_weight(
    group_rewards: torch.Tensor,
    threshold: float = 1e-6,
    gain_weight: float = 1.0,
) -> float:
    """组内奖励方差≈0(零方差组，GRPO 学不动)时 → 返回大权重让 info-gain 主导；
    方差正常时 → 返回 0，交回外部奖励主导。

    这是 survey 里 InfoPO 的关键：奖励学不动时靠信息增益继续供梯度。
    """
    var = float(group_rewards.var(unbiased=False))
    if var < threshold:
        return gain_weight        # 开大 info-gain
    return 0.0                     # 退回外部奖励


def fuse_advantage(
    reward_advantage: torch.Tensor,   # 来自 GRPO/RLOO 的外部奖励 advantage
    info_gain_advantage: torch.Tensor,
    gate_weight: float,
) -> torch.Tensor:
    """方差门控自适应融合：A = A_reward + gate·A_infogain。

    零方差组时 A_reward≈0，全靠 gate·A_infogain；正常组时 gate=0，退回 A_reward。
    """
    return reward_advantage + gate_weight * info_gain_advantage


# ============ 统一入口：按 config.method 选打法 ============
def compute_turn_advantages(
    traj: Trajectory,
    method: str = "r2g",
    gamma: float = 0.8,
    normalize: bool = True,
) -> list[float]:
    """按信用分配打法算每轮 advantage，写入 Turn.advantage 并返回。

    - outcome：每轮 advantage = 轨迹最终 reward(常数广播)。
    - r2g：折扣前传后(可选)组内归一化。
    - infopo：需外部提供 info-gain(此处退化为 r2g + 由训练器融合，见 train)。
    """
    if method == "outcome":
        vals = [traj.outcome_reward for _ in traj.turns]
    elif method in ("r2g", "infopo"):
        vals = apply_r2g_to_trajectory(traj, gamma)
    else:
        raise ValueError(f"unknown credit method: {method}")

    if normalize and len(vals) > 1:
        vals = normalize_turn_values(vals)
    for turn, v in zip(traj.turns, vals):
        turn.advantage = v
    return vals
