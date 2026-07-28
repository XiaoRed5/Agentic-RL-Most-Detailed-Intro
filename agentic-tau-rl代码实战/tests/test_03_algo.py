"""algo 测试：每个 advantage / loss 对拍手算，验证数学正确性(不是'跑通就行')。"""
import math

import pytest
import torch

from agentic_rl.algo.advantage import (
    broadcast_advantage_to_tokens,
    compute_gae,
    grpo_group_advantage,
    grpo_group_advantage_no_std,
    rloo_advantage,
)
from agentic_rl.algo.dapo import (
    is_zero_variance_group,
    filter_dynamic_sampling,
    overlong_reward_shaping,
)
from agentic_rl.algo.loss import (
    entropy_bonus,
    kl_penalty,
    masked_mean,
    ppo_policy_loss,
    value_loss,
)


# ========== GRPO advantage ==========
def test_grpo_advantage_hand_computed():
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])   # 1对3错
    adv = grpo_group_advantage(r)
    mean = 0.25
    std = math.sqrt(((1-0.25)**2 + 3*(0-0.25)**2) / 4)   # 有偏 std
    expected = torch.tensor([(1-mean)/std, -mean/std, -mean/std, -mean/std])
    assert torch.allclose(adv, expected, atol=1e-5)
    # 归一化后均值≈0
    assert abs(float(adv.mean())) < 1e-5


def test_grpo_no_std_is_centering():
    r = torch.tensor([1.0, 2.0, 3.0])
    adv = grpo_group_advantage_no_std(r)
    assert torch.allclose(adv, torch.tensor([-1.0, 0.0, 1.0]))


def test_grpo_zero_variance_gives_zero_advantage():
    """全对(或全错)→ advantage 全 0(零方差组无梯度)。"""
    r = torch.tensor([1.0, 1.0, 1.0])
    adv = grpo_group_advantage(r)
    assert torch.allclose(adv, torch.zeros(3), atol=1e-5)


# ========== RLOO advantage ==========
def test_rloo_leave_one_out_hand_computed():
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])
    adv = rloo_advantage(r)
    # A_0 = 1 - (0+0+0)/3 = 1 ; A_1 = 0 - (1+0+0)/3 = -1/3
    expected = torch.tensor([1.0, -1/3, -1/3, -1/3])
    assert torch.allclose(adv, expected, atol=1e-6)


def test_rloo_needs_two():
    with pytest.raises(AssertionError):
        rloo_advantage(torch.tensor([1.0]))


# ========== GAE ==========
def test_gae_hand_computed_gamma1_lambda1():
    """γ=1,λ=1 时 GAE 退化为 (return - value)，最易手验。"""
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    adv, returns = compute_gae(rewards, values, gamma=1.0, lam=1.0, last_value=0.0)
    # returns(蒙特卡洛): t2=1, t1=1, t0=1 ; adv = returns - values = 0.5 each
    assert torch.allclose(returns, torch.tensor([1.0, 1.0, 1.0]), atol=1e-6)
    assert torch.allclose(adv, torch.tensor([0.5, 0.5, 0.5]), atol=1e-6)


def test_gae_single_step_delta():
    rewards = torch.tensor([2.0])
    values = torch.tensor([1.0])
    adv, returns = compute_gae(rewards, values, gamma=0.9, lam=0.95, last_value=0.0)
    # δ = 2 + 0.9*0 - 1 = 1.0
    assert float(adv[0]) == pytest.approx(1.0)
    assert float(returns[0]) == pytest.approx(2.0)


def test_broadcast_advantage_respects_mask():
    mask = torch.tensor([1.0, 1.0, 0.0])
    adv = broadcast_advantage_to_tokens(0.7, 3, mask)
    assert torch.allclose(adv, torch.tensor([0.7, 0.7, 0.0]))


# ========== masked_mean ==========
def test_masked_mean():
    x = torch.tensor([1.0, 2.0, 100.0])
    mask = torch.tensor([1.0, 1.0, 0.0])
    assert float(masked_mean(x, mask)) == pytest.approx(1.5)


def test_masked_mean_empty_mask_no_nan():
    x = torch.tensor([1.0, 2.0])
    mask = torch.tensor([0.0, 0.0])
    assert not math.isnan(float(masked_mean(x, mask)))


# ========== PPO loss ==========
def test_ppo_loss_on_policy_equals_neg_adv():
    """logπ == logπ_old(ratio=1, 未 clip) → loss = -mean(A)。"""
    lp = torch.tensor([-1.0, -2.0, -0.5])
    adv = torch.tensor([1.0, -1.0, 2.0])
    mask = torch.ones(3)
    loss, info = ppo_policy_loss(lp, lp.clone(), adv, mask)
    assert float(loss) == pytest.approx(-adv.mean().item())
    assert info["ratio_mean"] == pytest.approx(1.0)


def test_ppo_clip_caps_positive_advantage():
    """ratio 很大 + 正 advantage → 被 clip 到 (1+ε)。"""
    old = torch.tensor([-1.0])
    new = torch.tensor([1.0])       # ratio = exp(2) ≈ 7.39
    adv = torch.tensor([1.0])
    mask = torch.ones(1)
    loss, info = ppo_policy_loss(new, old, adv, mask, clip_eps=0.2)
    # min(7.39*1, 1.2*1) = 1.2 → loss = -1.2
    assert float(loss) == pytest.approx(-1.2, abs=1e-5)
    assert info["clip_frac"] == pytest.approx(1.0)


def test_clip_higher_uses_higher_bound():
    old = torch.tensor([-1.0])
    new = torch.tensor([1.0])
    adv = torch.tensor([1.0])
    mask = torch.ones(1)
    loss_lo, _ = ppo_policy_loss(new, old, adv, mask, clip_eps=0.2, clip_higher=False)
    loss_hi, _ = ppo_policy_loss(new, old, adv, mask, clip_eps=0.2,
                                 clip_higher=True, clip_eps_high=0.28)
    # clip_higher 允许更大 ratio → 正 adv 下 loss 更小(更负)
    assert float(loss_hi) < float(loss_lo)
    assert float(loss_hi) == pytest.approx(-1.28, abs=1e-5)


def test_ppo_loss_ignores_masked_tokens():
    lp = torch.tensor([-1.0, -1.0])
    adv = torch.tensor([1.0, 999.0])   # 第二个 token 巨大但被 mask
    mask = torch.tensor([1.0, 0.0])
    loss, _ = ppo_policy_loss(lp, lp.clone(), adv, mask)
    assert float(loss) == pytest.approx(-1.0)   # 只算第一个


# ========== value loss / entropy / kl ==========
def test_value_loss_mse():
    v = torch.tensor([1.0, 2.0])
    ret = torch.tensor([1.0, 0.0])
    mask = torch.ones(2)
    # 0.5*mean((0)^2, (2)^2) = 0.5*2 = 1.0
    assert float(value_loss(v, ret, mask)) == pytest.approx(1.0)


def test_kl_penalty_nonnegative_and_zero_when_equal():
    lp = torch.tensor([-1.0, -2.0])
    mask = torch.ones(2)
    assert float(kl_penalty(lp, lp.clone(), mask)) == pytest.approx(0.0, abs=1e-6)
    assert float(kl_penalty(lp, lp + 0.5, mask)) > 0


def test_entropy_uniform_is_log_v():
    logits = torch.zeros(1, 4)   # 均匀分布 → 熵 = ln(4)
    mask = torch.ones(1)
    assert float(entropy_bonus(logits, mask)) == pytest.approx(math.log(4), abs=1e-5)


# ========== DAPO 数据层 ==========
def test_zero_variance_detection():
    assert is_zero_variance_group(torch.tensor([1.0, 1.0, 1.0]))
    assert not is_zero_variance_group(torch.tensor([1.0, 0.0]))


def test_dynamic_sampling_filters_flat_groups():
    groups = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0])]
    keep_idx, keep = filter_dynamic_sampling(groups)
    assert keep_idx == [0, 2]
    assert len(keep) == 2


def test_overlong_shaping():
    # 未超软上界：不惩罚
    assert overlong_reward_shaping(1.0, 400, max_len=1024, buffer=512) == 1.0
    # 超过 max_len：满额惩罚
    assert overlong_reward_shaping(1.0, 1024, max_len=1024, buffer=512) == pytest.approx(0.0)
    # buffer 区间线性
    r = overlong_reward_shaping(1.0, 768, max_len=1024, buffer=512)  # soft_cap=512, frac=(768-512)/512=0.5
    assert r == pytest.approx(0.5)
