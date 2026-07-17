"""credit assignment 测试：核心是对拍 survey 的 TravelGym R2G 算例(第7轮=1.44)。"""
import math

import pytest
import torch

from agentic_rl.credit.credit import (
    apply_r2g_to_trajectory,
    compute_turn_advantages,
    counterfactual_info_gain,
    fuse_advantage,
    infopo_turn_gains,
    normalize_turn_values,
    reward_to_go,
    variance_gate_weight,
)
from agentic_rl.utils.types import ActionType, Trajectory, Turn


# ========== R2G：对拍 survey TravelGym 算例 ==========
def test_r2g_matches_survey_travelgym_example():
    """survey 折叠块给出的反向递推(γ=0.8)：
       轮9 R2G=1.0；轮8 = 0 + 0.8*1.0 = 0.8；轮7 = 0.8 + 0.8*0.8 = 1.44。
    对应 9 轮的即时 reward 序列(取自 survey 表格 ①原始 reward 列)。"""
    # 即时 reward(survey 表 ①原始列)：轮1..9
    immediate = [0.0, 0.2, 0.2, 0.0, 0.2, 0.0, 0.8, 0.0, 1.0]
    r2g = reward_to_go(immediate, gamma=0.8)
    # 关键锚点(survey 明确给出的三个数)
    assert r2g[8] == pytest.approx(1.0)        # 轮9
    assert r2g[7] == pytest.approx(0.8)        # 轮8 = 0 + 0.8*1.0
    assert r2g[6] == pytest.approx(1.44)       # 轮7 = 0.8 + 0.8*0.8  ← survey 核心算例


def test_r2g_recursion_property():
    """R2G_t = r_t + γ·R2G_{t+1} 严格成立。"""
    imm = [0.1, 0.0, 0.5, 1.0]
    g = 0.9
    r2g = reward_to_go(imm, g)
    for t in range(len(imm) - 1):
        assert r2g[t] == pytest.approx(imm[t] + g * r2g[t + 1])
    assert r2g[-1] == pytest.approx(imm[-1])


def test_r2g_rescues_zero_reward_turn():
    """survey 核心洞见：即时 reward=0 的澄清轮，经 R2G 后拿到非零功劳。"""
    imm = [0.0, 0.0, 1.0]     # 前两轮白问，最后成功
    r2g = reward_to_go(imm, gamma=0.8)
    assert r2g[0] > 0 and r2g[1] > 0   # 零奖励轮被"救活"
    assert r2g[0] == pytest.approx(0.64)   # 0 + 0.8*(0 + 0.8*1) = 0.64


def test_r2g_can_exceed_one():
    """survey 明确：R2G 值可以 >1(折扣累加不是概率)。"""
    imm = [0.8, 0.2, 1.0]
    r2g = reward_to_go(imm, gamma=0.8)
    assert r2g[0] > 1.0


def test_apply_r2g_writes_to_turns():
    traj = _mk_traj([0.0, 0.2, 1.0])
    r2g = apply_r2g_to_trajectory(traj, gamma=0.8)
    assert [t.r2g for t in traj.turns] == pytest.approx(r2g)


# ========== 组内归一化 ==========
def test_normalize_zero_mean():
    vals = normalize_turn_values([1.44, 0.8, 1.0, 0.5])
    assert abs(sum(vals) / len(vals)) < 1e-5
    t = torch.tensor(vals)
    assert float(t.std(unbiased=False)) == pytest.approx(1.0, abs=1e-4)


# ========== InfoPO 反事实信息增益 ==========
def test_info_gain_zero_when_identical():
    """事实与反事实分布相同 → 该轮反馈无信息增益(gain=0)。"""
    p = torch.tensor([0.25, 0.25, 0.25, 0.25])
    assert counterfactual_info_gain(p, p.clone()) == pytest.approx(0.0, abs=1e-6)


def test_info_gain_large_when_decisive():
    """事实分布很确定(直奔溺水)、反事实还在乱猜 → gain 大(呼应 survey 图)。"""
    p_factual = torch.tensor([0.95, 0.02, 0.02, 0.01])   # 很确定
    p_masked = torch.tensor([0.25, 0.25, 0.25, 0.25])    # 乱猜
    gain = counterfactual_info_gain(p_factual, p_masked)
    assert gain > 0.5


def test_info_gain_nonnegative():
    for _ in range(5):
        a = torch.rand(6); b = torch.rand(6)
        assert counterfactual_info_gain(a, b) >= 0.0


def test_infopo_turn_gains_batch():
    fac = [torch.tensor([0.9, 0.1]), torch.tensor([0.5, 0.5])]
    msk = [torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5])]
    gains = infopo_turn_gains(fac, msk)
    assert gains[0] > 0        # 第一轮有信息
    assert gains[1] == pytest.approx(0.0, abs=1e-6)   # 第二轮无信息


# ========== 方差门控(治零方差组) ==========
def test_variance_gate_opens_on_zero_variance():
    """零方差组(全对/全错)→ 门控开大 info-gain 权重。"""
    flat = torch.tensor([0.5, 0.5, 0.5, 0.5])
    assert variance_gate_weight(flat, gain_weight=1.0) == 1.0


def test_variance_gate_closes_on_normal_variance():
    """方差正常 → 门控关闭(=0)，交回外部奖励主导。"""
    varied = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert variance_gate_weight(varied) == 0.0


def test_fuse_advantage_zero_variance_uses_infogain():
    """零方差组：A_reward≈0，融合后全靠 info-gain。"""
    a_reward = torch.tensor([0.0, 0.0, 0.0])   # 零方差组 advantage 全 0
    a_info = torch.tensor([0.7, 0.1, 0.3])
    fused = fuse_advantage(a_reward, a_info, gate_weight=1.0)
    assert torch.allclose(fused, a_info)


def test_fuse_advantage_normal_ignores_infogain():
    a_reward = torch.tensor([0.5, -0.5])
    a_info = torch.tensor([9.0, 9.0])
    fused = fuse_advantage(a_reward, a_info, gate_weight=0.0)
    assert torch.allclose(fused, a_reward)


# ========== 统一入口：三种打法切换 ==========
def test_method_outcome_broadcasts_final_reward():
    traj = _mk_traj([0.0, 0.0, 0.0])
    traj.outcome_reward = 1.0
    vals = compute_turn_advantages(traj, method="outcome", normalize=False)
    assert vals == [1.0, 1.0, 1.0]


def test_method_r2g_differs_from_outcome():
    """同一条轨迹，R2G 给早期轮的信用与 outcome(常数)不同 —— 这是'做深'的可视化。"""
    imm = [0.0, 0.0, 1.0]
    traj_o = _mk_traj(imm); traj_o.outcome_reward = 1.0
    traj_r = _mk_traj(imm); traj_r.outcome_reward = 1.0
    v_out = compute_turn_advantages(traj_o, method="outcome", normalize=False)
    v_r2g = compute_turn_advantages(traj_r, method="r2g", normalize=False)
    assert v_out[0] == v_out[1] == v_out[2]     # outcome：常数
    assert v_r2g[0] < v_r2g[2]                  # r2g：越靠终点信用越大


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        compute_turn_advantages(_mk_traj([1.0]), method="bogus")


# ========== helper ==========
def _mk_traj(immediate_rewards):
    traj = Trajectory(task_id="t", domain="retail")
    for i, r in enumerate(immediate_rewards):
        traj.turns.append(Turn(index=i, action_type=ActionType.TOOL_CALL,
                               immediate_reward=r, assistant_token_ids=[1, 2]))
    return traj
