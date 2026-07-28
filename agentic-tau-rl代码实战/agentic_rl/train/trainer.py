"""端到端 GRPO 训练器：把前面所有模块串成一个训练 step。

数据流(一个 step)：
  1. 对每个 task 采样 G 条 rollout(一个 group)。
  2. credit assignment：按 config.credit.method 算每轮 advantage(outcome / R2G / InfoPO)。
  3. shaping(可选)：BAO/PPP 正则修正 reward。
  4. advantage：组内标准化(GRPO)。
  5. token 级 PPO loss(尊重 loss_mask，只对 policy 生成的 token 算) + backward。

真机迁移：把 rollout 后端换 SGLang、模型换 Qwen3-4B、加分布式，本文件的算法逻辑不变。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from agentic_rl.algo.advantage import grpo_group_advantage, rloo_advantage
from agentic_rl.algo.dapo import is_zero_variance_group
from agentic_rl.algo.loss import kl_penalty, ppo_policy_loss
from agentic_rl.credit.credit import (
    apply_r2g_to_trajectory,
    normalize_turn_values,
    variance_gate_weight,
)
from agentic_rl.credit.infopo_runtime import group_infogain_advantages
from agentic_rl.shaping.shaping import apply_bao_shaping
from agentic_rl.train.model import TinyTransformer, token_logprobs
from agentic_rl.utils.config import Config
from agentic_rl.utils.types import Trajectory


@dataclass
class StepStats:
    loss: float
    mean_reward: float
    mean_response_len: float
    n_groups: int
    n_dropped_zero_var: int
    clip_frac: float
    approx_kl: float
    n_infogain_gated: int = 0


def _expand_turns_to_tokens(traj: Trajectory, per_turn_adv: list[float]) -> list[float]:
    """把每轮 advantage 展开到该轮的 token：assistant token 用该轮 advantage，obs token 用 0。"""
    token_adv: list[float] = []
    for turn, a in zip(traj.turns, per_turn_adv):
        token_adv.extend([a] * len(turn.assistant_token_ids))
        token_adv.extend([0.0] * len(turn.obs_token_ids))
    return token_adv


def compute_group_token_advantages(
    group: list[Trajectory], config: Config
) -> list[list[float]]:
    """在**整个 group 层面**算每条轨迹每个 response token 的 advantage。

    - outcome：GRPO/RLOO 轨迹级 advantage，广播到该轨迹所有 assistant token。
    - r2g/infopo(UserRL 式)：对 group 内**所有轨迹的所有 turn** 的 R2G 统一做组内归一化，
      每个 turn 的 advantage = 其归一化 R2G。这样成功轨迹的高-R2G 轮得正、失败轨迹得负，
      且成功轨迹的早期澄清轮(R2G 仍高于组均值)也拿到正信用 —— 正是"救活零奖励澄清轮"。
    """
    method = config.credit.method
    rewards = torch.tensor([t.outcome_reward for t in group], dtype=torch.float32)

    if method == "outcome":
        traj_adv = compute_group_advantages(rewards, config).tolist()
        return [_expand_turns_to_tokens(t, [a] * t.n_turns)
                for t, a in zip(group, traj_adv)]

    # r2g / infopo：先给每条轨迹算 turn-level R2G，再 group 级统一归一化。
    per_traj_r2g: list[list[float]] = []
    pooled: list[float] = []
    for t in group:
        r2g = apply_r2g_to_trajectory(t, config.credit.gamma)
        per_traj_r2g.append(r2g)
        pooled.extend(r2g)

    normed = normalize_turn_values(pooled) if len(pooled) > 1 else [0.0] * len(pooled)
    # 切回每条轨迹
    out: list[list[float]] = []
    cursor = 0
    for t, r2g in zip(group, per_traj_r2g):
        n = len(r2g)
        per_turn = normed[cursor:cursor + n]
        cursor += n
        out.append(_expand_turns_to_tokens(t, per_turn))
    return out


def compute_group_advantages(rewards: torch.Tensor, config: Config) -> torch.Tensor:
    if config.algo.estimator == "rloo" and rewards.numel() >= 2:
        return rloo_advantage(rewards)
    return grpo_group_advantage(rewards)


def train_step(
    model: TinyTransformer,
    optimizer: torch.optim.Optimizer,
    groups: list[list[Trajectory]],     # 每个 task 一个 group，每 group G 条 rollout
    config: Config,
    old_logprobs_batch: Optional[torch.Tensor] = None,   # 多步 PPO 用(采样时 behavior policy)
    ref_logprobs_batch: Optional[torch.Tensor] = None,   # ref policy KL 用
) -> StepStats:
    """一个 GRPO 训练 step。groups 已由 rollout 阶段产出(轨迹带 token + mask)。

    单步 on-policy(默认)：old_logprobs=当前 → ratio≡1，loss=-mean(adv)。
    多步 PPO：传入 old_logprobs_batch(采样时冻结的 logprob)，可对同一批数据多次更新，
              ratio≠1，真正发挥 clip 的作用。见 multi_step_ppo_update()。
    """
    model.train()

    all_rewards: list[float] = []
    all_lens: list[float] = []
    seq_batch: list[tuple[list[int], list[int], list[float]]] = []  # (tokens, loss_mask, token_adv)
    n_dropped = 0
    n_infogain_gated = 0
    pad_id = model.vocab_size - 1

    for group in groups:
        rewards = torch.tensor([t.outcome_reward for t in group], dtype=torch.float32)
        all_rewards.extend(rewards.tolist())

        zero_var = is_zero_variance_group(rewards, config.credit.infopo_variance_threshold)

        # DAPO ② Dynamic Sampling：丢弃零方差组(无梯度)。
        # 注意：InfoPO 与 Dynamic Sampling 是对零方差组的两种**对立**处理 ——
        # Dynamic Sampling 丢弃它，InfoPO 用 info-gain 救活它。若同时开，InfoPO 优先
        # (先尝试救；只有非 InfoPO 模式才丢弃)。
        if config.algo.dynamic_sampling and zero_var and config.credit.method != "infopo":
            n_dropped += 1
            continue

        # shaping(可选)：BAO 正则修正 —— 修正后用 shaped_reward 覆盖 immediate
        if config.shaping.bao_info_seeking or config.shaping.bao_over_thinking:
            for t in group:
                apply_bao_shaping(
                    t, config.shaping.lambda_ans, config.shaping.lambda_think,
                    config.shaping.turn_budget,
                    config.shaping.bao_info_seeking, config.shaping.bao_over_thinking,
                )
                for turn in t.turns:
                    turn.immediate_reward = turn.shaped_reward

        group_token_advs = compute_group_token_advantages(group, config)

        # InfoPO：方差门控融合。零方差组时(reward advantage≈0)用反事实 info-gain 顶上。
        if config.credit.method == "infopo":
            gate = variance_gate_weight(
                rewards,
                threshold=config.credit.infopo_variance_threshold,
                gain_weight=config.credit.infopo_gain_weight,
            )
            if gate > 0.0:
                n_infogain_gated += 1
                infogain_advs = group_infogain_advantages(model, group, pad_id)
                fused = []
                for base, ig in zip(group_token_advs, infogain_advs):
                    fused.append([b + gate * g for b, g in zip(base, ig)])
                group_token_advs = fused

        for traj, resp_adv in zip(group, group_token_advs):
            tokens = traj.all_token_ids()
            full_mask = traj.full_loss_mask()
            # token 级 advantage：prompt 段补 0，response 段按 credit 分配
            token_adv = [0.0] * len(traj.prompt_token_ids) + resp_adv
            assert len(token_adv) == len(tokens) == len(full_mask)
            seq_batch.append((tokens, full_mask, token_adv))
            all_lens.append(traj.response_length)

    if not seq_batch:
        return StepStats(0.0, float(sum(all_rewards) / max(len(all_rewards), 1)),
                         0.0, len(groups), n_dropped, 0.0, 0.0, n_infogain_gated)

    # 组 batch(pad 到等长)
    input_ids, loss_mask, adv = _pad_batch(seq_batch, pad_id=pad_id)

    logits, _ = model(input_ids)
    logprobs = token_logprobs(logits, input_ids)          # [B, T-1]
    # 对齐：token t 的 logprob 预测 token t+1 → advantage/mask 取 [1:] 位置
    adv_shift = adv[:, 1:]
    mask_shift = loss_mask[:, 1:]

    # old_logprobs：多步 PPO 时由外部传入(采样时的 behavior policy)；否则单步 on-policy(ratio=1)
    if old_logprobs_batch is not None:
        old_lp = old_logprobs_batch[:, 1:]
    else:
        old_lp = logprobs.detach()
    loss, info = ppo_policy_loss(
        logprobs.reshape(-1), old_lp.reshape(-1),
        adv_shift.reshape(-1), mask_shift.reshape(-1),
        clip_eps=config.algo.clip_eps,
        clip_higher=config.algo.clip_higher,
        clip_eps_high=config.algo.clip_eps_high,
        token_level=config.algo.token_level_loss,
    )

    # 与参考 policy 的 KL 惩罚(GRPO 常用，防漂移)。ref_logprobs 由外部提供。
    if ref_logprobs_batch is not None and config.algo.kl_coef > 0:
        ref_lp = ref_logprobs_batch[:, 1:]
        kl = kl_penalty(logprobs.reshape(-1), ref_lp.reshape(-1), mask_shift.reshape(-1))
        loss = loss + config.algo.kl_coef * kl

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return StepStats(
        loss=float(loss),
        mean_reward=float(sum(all_rewards) / max(len(all_rewards), 1)),
        mean_response_len=float(sum(all_lens) / max(len(all_lens), 1)),
        n_groups=len(groups),
        n_dropped_zero_var=n_dropped,
        clip_frac=info["clip_frac"],
        approx_kl=info["approx_kl"],
        n_infogain_gated=n_infogain_gated,
    )


def _pad_batch(seq_batch, pad_id: int):
    B = len(seq_batch)
    T = max(len(s[0]) for s in seq_batch)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    loss_mask = torch.zeros(B, T)
    adv = torch.zeros(B, T)
    for i, (toks, m, a) in enumerate(seq_batch):
        L = len(toks)
        input_ids[i, :L] = torch.tensor(toks, dtype=torch.long)
        loss_mask[i, :L] = torch.tensor(m, dtype=torch.float32)
        adv[i, :L] = torch.tensor(a, dtype=torch.float32)
    return input_ids, loss_mask, adv


def build_batch_advantages(groups: list[list[Trajectory]], model: TinyTransformer,
                           config: Config):
    """把 groups 展开成 padded batch(input_ids, loss_mask, token_adv)。

    供多步 PPO 用：先固定这批 advantage 和 old_logprobs，再对同一批数据多轮更新。
    这里不做 dynamic sampling / infopo(简化，多步路径专注演示 ratio≠1 的 clip 行为)。
    """
    seq_batch = []
    for group in groups:
        advs = compute_group_token_advantages(group, config)
        for traj, resp_adv in zip(group, advs):
            tokens = traj.all_token_ids()
            token_adv = [0.0] * len(traj.prompt_token_ids) + resp_adv
            seq_batch.append((tokens, traj.full_loss_mask(), token_adv))
    return _pad_batch(seq_batch, pad_id=model.vocab_size - 1)


def multi_step_ppo_update(
    model: TinyTransformer,
    optimizer: torch.optim.Optimizer,
    groups: list[list[Trajectory]],
    config: Config,
    ppo_epochs: int = 4,
    ref_model: Optional[TinyTransformer] = None,
) -> list[dict]:
    """真正的多步 PPO：捕获 old_logprobs(采样时冻结)，对同一批数据更新 ppo_epochs 次。

    与单步 train_step 的区别：第 2 个 epoch 起 ratio≠1，clip 真正起作用 —— 这才是
    PPO 的核心机制。ref_model(可选)提供 KL 锚点，防策略漂移(GRPO 常用)。
    返回每个 epoch 的诊断。
    """
    model.train()
    input_ids, loss_mask, adv = build_batch_advantages(groups, model, config)
    adv_shift = adv[:, 1:]
    mask_shift = loss_mask[:, 1:]

    # 捕获 old_logprobs(behavior policy，采样时刻，全程冻结)
    with torch.no_grad():
        logits0, _ = model(input_ids)
        old_lp = token_logprobs(logits0, input_ids).detach()
        ref_lp = None
        if ref_model is not None:
            rlogits, _ = ref_model(input_ids)
            ref_lp = token_logprobs(rlogits, input_ids).detach()

    history = []
    for epoch in range(ppo_epochs):
        logits, _ = model(input_ids)
        logprobs = token_logprobs(logits, input_ids)
        loss, info = ppo_policy_loss(
            logprobs.reshape(-1), old_lp.reshape(-1),
            adv_shift.reshape(-1), mask_shift.reshape(-1),
            clip_eps=config.algo.clip_eps,
            clip_higher=config.algo.clip_higher,
            clip_eps_high=config.algo.clip_eps_high,
            token_level=config.algo.token_level_loss,
        )
        if ref_lp is not None and config.algo.kl_coef > 0:
            kl = kl_penalty(logprobs.reshape(-1), ref_lp.reshape(-1), mask_shift.reshape(-1))
            loss = loss + config.algo.kl_coef * kl
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss),
                        "clip_frac": info["clip_frac"], "approx_kl": info["approx_kl"]})
    return history
