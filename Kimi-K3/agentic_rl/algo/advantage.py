"""Advantage estimators：GRPO / RLOO / PPO+GAE。全部纯 torch，CPU 可验，对拍手算。

术语统一：一个 "group" = 同一个 prompt 采样的 G 条轨迹(rollout)。GRPO/RLOO 用组内
其它轨迹当 baseline 估 advantage，省掉 value model；PPO 用学出来的 value + GAE。

这些是 RL 的"信号来源"：advantage 决定每条轨迹(每个 token)被强化还是被压制。
"""
from __future__ import annotations

import torch


def grpo_group_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """GRPO：组内标准化 advantage。A_i = (r_i - mean) / (std + eps)。

    rewards: [G] 一个组内 G 条轨迹的标量 reward。
    返回:    [G] 每条轨迹的 advantage(轨迹级，之后广播到该轨迹所有 token)。
    这就是 survey 里"喂给 GRPO 前再做组内归一化 (值-μ)/σ"。
    """
    assert rewards.dim() == 1
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def grpo_group_advantage_no_std(rewards: torch.Tensor) -> torch.Tensor:
    """GRPO 的 mean-only 变体(Dr.GRPO 风格)：只减均值不除 std，避免长度/难度偏置。"""
    return rewards - rewards.mean()


def rloo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """RLOO(Leave-One-Out)：每条轨迹的 baseline = 组内**其它** G-1 条的均值。

    A_i = r_i - mean_{j≠i}(r_j) = r_i - (sum - r_i)/(G-1)。
    比 GRPO 更无偏(baseline 不含自己)。
    """
    assert rewards.dim() == 1
    G = rewards.shape[0]
    assert G >= 2, "RLOO 需要组内至少 2 条轨迹"
    total = rewards.sum()
    loo_baseline = (total - rewards) / (G - 1)
    return rewards - loo_baseline


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(Generalized Advantage Estimation)，用于带 value model 的 PPO。

    rewards: [T] 每一步(turn 或 token)的即时 reward。
    values:  [T] value model 对每一步的估值 V(s_t)。
    返回: (advantages [T], returns [T])，其中 returns = advantages + values(TD target)。

    δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
    A_t = δ_t + γλ·A_{t+1}   (反向递推)
    """
    T = rewards.shape[0]
    adv = torch.zeros(T, dtype=rewards.dtype)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


def broadcast_advantage_to_tokens(
    traj_advantage: float,
    n_response_tokens: int,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """把轨迹级 advantage 广播到该轨迹每个 response token(GRPO/RLOO 的标准做法)。

    loss_mask(可选)：mask=0 的位置 advantage 也置 0(那些 token 不参与 loss)。
    """
    adv = torch.full((n_response_tokens,), float(traj_advantage))
    if loss_mask is not None:
        adv = adv * loss_mask.to(adv.dtype)
    return adv
