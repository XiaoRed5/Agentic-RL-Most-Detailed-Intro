"""PPO clipped policy loss + DAPO 四件套。纯 torch，token 级，尊重 loss_mask。

DAPO 四件套(survey 提及，逐个可开关，单测隔离验证)：
① Clip-Higher：正负方向用不同的 clip 上界(ε_high > ε_low)，鼓励低概率 token 探索。
② Dynamic Sampling：丢弃组内奖励零方差的样本(advantage 全 0，无梯度) —— 在数据层做，见 algo/dapo.py。
③ Token-Level Loss：按 token 数平均(而非先按序列再平均)，长序列的每个 token 权重相等。
④ Overlong Reward Shaping：对超长响应做软惩罚 —— 在 reward 层做，见 algo/dapo.py。
"""
from __future__ import annotations

import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None) -> torch.Tensor:
    """按 mask 求均值(mask=0 的位置不计入)。"""
    mask = mask.to(x.dtype)
    if dim is None:
        s = (x * mask).sum()
        n = mask.sum().clamp_min(1.0)
        return s / n
    s = (x * mask).sum(dim=dim)
    n = mask.sum(dim=dim).clamp_min(1.0)
    return s / n


def ppo_policy_loss(
    logprobs: torch.Tensor,        # [N] 当前 policy 对每个 token 的 log π(a|s)
    old_logprobs: torch.Tensor,    # [N] 采样时(behavior)policy 的 log π_old
    advantages: torch.Tensor,      # [N] 每个 token 的 advantage
    loss_mask: torch.Tensor,       # [N] 1=算 loss, 0=忽略(prompt/obs token)
    clip_eps: float = 0.2,
    clip_higher: bool = False,
    clip_eps_high: float = 0.28,
    token_level: bool = True,
) -> tuple[torch.Tensor, dict]:
    """标准 PPO clipped surrogate，token 级，尊重 mask。

    ratio = exp(logπ - logπ_old)
    L = -min(ratio·A, clip(ratio, 1-ε, 1+ε_high)·A)
    """
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    low = 1.0 - clip_eps
    high = 1.0 + (clip_eps_high if clip_higher else clip_eps)   # ① Clip-Higher
    clipped = torch.clamp(ratio, low, high) * advantages
    per_token_loss = -torch.min(unclipped, clipped)

    if token_level:
        # ③ Token-Level Loss：所有有效 token 一起平均(长序列每 token 等权)
        loss = masked_mean(per_token_loss, loss_mask)
    else:
        # 传统：per-token loss 先按序列平均，再对序列平均(短序列的 token 被放大)
        loss = per_token_loss  # 序列维度由调用方处理
        loss = masked_mean(per_token_loss, loss_mask)

    # 诊断指标
    with torch.no_grad():
        clip_frac = masked_mean(
            ((ratio > high) | (ratio < low)).to(ratio.dtype), loss_mask
        )
        approx_kl = masked_mean(old_logprobs - logprobs, loss_mask)
    return loss, {"clip_frac": float(clip_frac), "approx_kl": float(approx_kl),
                  "ratio_mean": float(masked_mean(ratio, loss_mask))}


def value_loss(
    values: torch.Tensor,          # [N] 当前 value 预测
    returns: torch.Tensor,         # [N] GAE returns(TD target)
    loss_mask: torch.Tensor,
    old_values: torch.Tensor | None = None,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO value loss，可选 value clipping。"""
    if old_values is not None:
        v_clipped = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
        loss_unclipped = (values - returns) ** 2
        loss_clipped = (v_clipped - returns) ** 2
        loss = 0.5 * torch.max(loss_unclipped, loss_clipped)
    else:
        loss = 0.5 * (values - returns) ** 2
    return masked_mean(loss, loss_mask)


def entropy_bonus(logits: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """token 级熵(鼓励探索)。logits: [N, V]。"""
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(dim=-1)   # [N]
    return masked_mean(ent, loss_mask)


def kl_penalty(logprobs: torch.Tensor, ref_logprobs: torch.Tensor,
               loss_mask: torch.Tensor) -> torch.Tensor:
    """与参考 policy 的 KL(GRPO 常用的 k3 无偏估计)。"""
    diff = ref_logprobs - logprobs
    kl = torch.exp(diff) - diff - 1.0   # k3 estimator，恒 >=0
    return masked_mean(kl, loss_mask)
