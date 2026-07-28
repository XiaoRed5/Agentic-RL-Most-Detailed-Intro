"""DAPO 数据层的两件套：Dynamic Sampling + Overlong Reward Shaping。

这两个不在 loss 公式里，而在"喂什么数据/什么 reward"层面，故独立成模块。
"""
from __future__ import annotations

import torch


def is_zero_variance_group(rewards: torch.Tensor, eps: float = 1e-6) -> bool:
    """② Dynamic Sampling 的判据：组内奖励方差≈0(全对或全错) → advantage 全 0 → 无梯度。

    这正是 survey 里 InfoPO 说的"零方差组 GRPO 学不动"。DAPO 的做法是直接丢弃这种组；
    InfoPO 的做法是换 info-gain 信号救活它(见 credit/infopo.py)。两种思路并存。
    """
    return float(rewards.var(unbiased=False)) < eps


def filter_dynamic_sampling(groups: list[torch.Tensor], eps: float = 1e-6):
    """丢弃零方差组，返回 (保留的组下标, 保留的组列表)。"""
    keep_idx, keep = [], []
    for i, g in enumerate(groups):
        if not is_zero_variance_group(g, eps):
            keep_idx.append(i)
            keep.append(g)
    return keep_idx, keep


def overlong_reward_shaping(
    reward: float,
    response_len: int,
    max_len: int,
    buffer: int,
    penalty_scale: float = 1.0,
) -> float:
    """④ Overlong Reward Shaping：软惩罚过长响应，避免"想太多/话太多"(呼应 BAO)。

    - 长度 <= max_len - buffer：不惩罚。
    - 进入 buffer 区间：线性递增惩罚。
    - 超过 max_len：满额惩罚。
    返回 shaped reward。
    """
    soft_cap = max_len - buffer
    if response_len <= soft_cap:
        return reward
    if response_len >= max_len:
        return reward - penalty_scale
    frac = (response_len - soft_cap) / max(buffer, 1)
    return reward - penalty_scale * frac
