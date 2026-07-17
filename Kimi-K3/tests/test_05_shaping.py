"""shaping 测试：BAO 两条正则(对拍 survey 算例 4.33) + PPP 三目标。"""
import pytest

from agentic_rl.shaping.shaping import (
    Effort,
    apply_bao_shaping,
    bao_information_seeking_penalty,
    bao_over_thinking_penalty,
    ppp_normalized_advantage,
    ppp_personalization_reward,
    ppp_proactivity_reward,
    ppp_total_reward,
)
from agentic_rl.utils.types import ActionType, Status, Trajectory, Turn


def _mk_traj(action_names, immediate=None, outcome=0.0):
    traj = Trajectory(task_id="t", domain="turtle")
    imm = immediate or [0.0] * len(action_names)
    for i, name in enumerate(action_names):
        traj.turns.append(Turn(
            index=i,
            action_type=ActionType.RESPOND if name == "respond" else ActionType.TOOL_CALL,
            action_name=name,
            is_user_involved=(name == "respond"),
            immediate_reward=imm[i],
        ))
    traj.outcome_reward = outcome
    return traj


# ========== BAO ① Information-Seeking ==========
def test_info_seeking_penalizes_consecutive_asks():
    """连问两轮(respond→respond)→ 第二轮被扣。"""
    traj = _mk_traj(["respond", "respond", "get_x"])
    pen = bao_information_seeking_penalty(traj, lambda_ans=0.1)
    assert pen == [0.0, -0.1, 0.0]


def test_info_seeking_no_penalty_when_interleaved():
    """问完用户去调工具再问 → 不连续，不罚。"""
    traj = _mk_traj(["respond", "get_x", "respond"])
    pen = bao_information_seeking_penalty(traj, lambda_ans=0.1)
    assert pen == [0.0, 0.0, 0.0]


def test_info_seeking_multiple_runs():
    traj = _mk_traj(["respond", "respond", "respond"])
    pen = bao_information_seeking_penalty(traj, lambda_ans=0.2)
    assert pen == [0.0, -0.2, -0.2]   # 轮2、轮3 都连着上一轮


# ========== BAO ② Over-Thinking：对拍 survey 算例 ==========
def test_over_thinking_matches_survey_example_433():
    """survey 举例：失败轨迹只走 3 轮就自爆 → 惩罚 = (16-3)/3 = 4.33。"""
    traj = _mk_traj(["respond", "respond", "respond"], outcome=0.0)
    pen = bao_over_thinking_penalty(traj, lambda_think=1.0, turn_budget=16)
    assert pen == pytest.approx(-13/3)          # -4.333...
    assert pen == pytest.approx(-4.3333, abs=1e-3)


def test_over_thinking_lighter_when_late_failure():
    """survey：走到 12 轮才失败 → (16-12)/12 ≈ 0.33，轻很多。"""
    traj = _mk_traj(["x"] * 12, outcome=0.0)
    pen = bao_over_thinking_penalty(traj, lambda_think=1.0, turn_budget=16)
    assert pen == pytest.approx(-4/12, abs=1e-4)   # -0.333
    assert abs(pen) < 0.5


def test_over_thinking_no_penalty_on_success():
    """成功轨迹不罚(只惩罚失败且提前终止)。"""
    traj = _mk_traj(["x", "x", "x"], outcome=1.0)
    assert bao_over_thinking_penalty(traj, lambda_think=1.0, turn_budget=16) == 0.0


def test_over_thinking_no_penalty_when_budget_used():
    """用满预算才失败 → 不算'提前终止'，不罚。"""
    traj = _mk_traj(["x"] * 16, outcome=0.0)
    assert bao_over_thinking_penalty(traj, lambda_think=1.0, turn_budget=16) == 0.0


def test_apply_bao_writes_shaped_reward():
    traj = _mk_traj(["respond", "respond"], immediate=[0.0, 0.0], outcome=0.0)
    apply_bao_shaping(traj, lambda_ans=0.1, lambda_think=1.0, turn_budget=16)
    # 轮1: 0 ; 轮2: 连问-0.1 + over-thinking(-(16-2)/2=-7) = -7.1
    assert traj.turns[0].shaped_reward == pytest.approx(0.0)
    assert traj.turns[1].shaped_reward == pytest.approx(-0.1 - 7.0)


# ========== PPP 三目标 ==========
def test_ppp_proactivity_all_low_gets_bonus():
    r = ppp_proactivity_reward([Effort.LOW, Effort.LOW])
    assert r == pytest.approx(0.05)


def test_ppp_proactivity_high_heavily_penalized():
    """甩活给用户(high)→ -0.5 重罚。"""
    r = ppp_proactivity_reward([Effort.LOW, Effort.HIGH])
    assert r == pytest.approx(-0.5)   # 非全 low → 无 bonus；一个 high → -0.5


def test_ppp_proactivity_medium():
    r = ppp_proactivity_reward([Effort.MEDIUM, Effort.MEDIUM])
    assert r == pytest.approx(-0.2)


def test_ppp_personalization():
    r = ppp_personalization_reward([True, True, False], violated_penalties=[0.3])
    assert r == pytest.approx(0.05 + 0.05 - 0.3)


def test_ppp_total_no_weights():
    """survey：三块无权重相加，Prod 是大头。"""
    ppp = ppp_total_reward(
        productivity=1.0,
        efforts=[Effort.LOW],
        pref_satisfied=[True],
        pref_violated_pen=[],
    )
    assert ppp.total == pytest.approx(1.0 + 0.05 + 0.05)
    assert ppp.productivity == 1.0   # 大头


def test_ppp_clip_normalization_absorbs_small_penalty():
    """survey 关键坑：先 clip 到[0,1]再标准化，负罚被边界吃掉 → 必须压小量级。"""
    # 两条轨迹：一条干成(1.0-0.5=0.5)，一条没干成(0.0-0.5=-0.5)
    adv = ppp_normalized_advantage([0.5, -0.5])
    # -0.5 被 clip 到 0 → [0.5, 0.0] 标准化 → 均值 0.25
    assert adv[0] > adv[1]          # 干成的仍占优
    # 验证 clip 生效：-0.5 和 -100 归一化结果应相同(都被 clip 到 0)
    adv2 = ppp_normalized_advantage([0.5, -100.0])
    assert adv2 == pytest.approx(adv)
