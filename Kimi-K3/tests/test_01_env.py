"""env 测试：
1. reward 聚合逻辑与真实 tau2 gold reward_info 对拍(最硬的正确性验证)。
2. 离线环境能真跑到 solved / transfer / truncated，且 reward 语义正确。
"""
import pytest

from agentic_rl.env.offline_tasks import make_retail_task, make_telecom_task
from agentic_rl.env.reward import (
    RewardComponents,
    RewardType,
    compute_reward,
    reconstruct_from_reward_info,
)
from agentic_rl.env.tau_env import STOP_TOKEN, TRANSFER_TOKEN, TauEnv


# ========== A. reward 聚合对拍真实 gold ==========
def test_reward_aggregation_matches_gold(all_fixtures):
    """对每条真实 fixture：用我们的 compute_reward 从 gold 各 component 重算，
    结果必须等于 tau2 官方算出的 gold reward。"""
    for name, fx in all_fixtures.items():
        ri = fx["gold"]["reward_info"]
        res = reconstruct_from_reward_info(ri)
        assert res.reward == pytest.approx(ri["reward"]), (
            f"{name}: recomputed {res.reward} != gold {ri['reward']}"
        )


def test_reward_basis_multiplication_semantics():
    """核心语义：reward = ∏ component[c] for c in basis。只有 basis 里的才乘。"""
    # DB=1, COMMUNICATE=0, 但 basis 只含 DB → reward=1
    comp = RewardComponents(db=1.0, communicate=0.0)
    assert compute_reward(comp, ["DB"]).reward == 1.0
    # basis 含两者 → 1*0 = 0
    assert compute_reward(comp, ["DB", "COMMUNICATE"]).reward == 0.0
    # 缺失的 component 视为 0
    assert compute_reward(RewardComponents(db=1.0), ["ENV_ASSERTION"]).reward == 0.0


def test_telecom_uses_env_assertion_basis(telecom_success):
    """telecom 用 ENV_ASSERTION basis(dual-control 断言)，不是 DB。"""
    ri = telecom_success["gold"]["reward_info"]
    assert ri["reward_basis"] == ["ENV_ASSERTION"]
    assert reconstruct_from_reward_info(ri).reward == 1.0


def test_partial_fail_still_zero_final(telecom_fail):
    """telecom_fail：partial_components 有 0.5，但 final reward 必须是 0(门控是乘积)。"""
    gold = telecom_fail["gold"]
    assert gold["partial_score"] == 0.0 or gold["reward"] == 0.0
    assert reconstruct_from_reward_info(gold["reward_info"]).reward == 0.0


# ========== B. 离线环境真跑 ==========
def test_retail_env_can_be_solved():
    """retail：正确的动作序列(cancel + refund + 提到 refund) → reward=1, STOP。"""
    task, tools = make_retail_task()
    env = TauEnv(task, tools, max_steps=16)
    obs = env.reset()
    assert "cancel" in obs.lower()

    env.step({"name": "cancel_order", "arguments": {"order_id": "#W001"}})
    env.step({"name": "process_refund", "arguments": {"order_id": "#W001"}})
    # 向用户确认并提到 refund(满足 COMMUNICATE)
    res = env.step({"name": "respond",
                    "arguments": {"content": "Your order is cancelled and the refund is processed."}})
    assert res.terminated
    assert res.reward == 1.0
    assert env.end_reason == STOP_TOKEN


def test_retail_env_transfer_path():
    """retail：agent 反复援引 policy 拒绝 → 用户坚持后 TRANSFER，reward=0。"""
    task, tools = make_retail_task()
    env = TauEnv(task, tools, max_steps=16)
    env.reset()
    env.step({"name": "respond", "arguments": {"content": "Sorry, our policy does not allow this."}})
    res = env.step({"name": "respond", "arguments": {"content": "I cannot do that per policy."}})
    assert res.terminated
    assert env.end_reason == TRANSFER_TOKEN
    assert res.reward == 0.0


def test_retail_missing_communicate_fails():
    """cancel+refund 都做了，但 agent 没提 refund → COMMUNICATE=0 → 卡在未达成。"""
    task, tools = make_retail_task()
    env = TauEnv(task, tools, max_steps=4)
    env.reset()
    env.step({"name": "cancel_order", "arguments": {"order_id": "#W001"}})
    env.step({"name": "process_refund", "arguments": {"order_id": "#W001"}})
    # DB 达标但没说 refund 关键词
    assert env._goal_reached() is False


def test_telecom_dualcontrol_solved():
    """telecom dual-control：agent 指导用户关飞行模式+开数据 → user 改 user_db → ENV_ASSERTION=1。"""
    task, tools = make_telecom_task()
    env = TauEnv(task, tools, max_steps=16)
    env.reset()
    # agent 无法自己按用户手机，只能 respond 指导
    res = env.step({"name": "respond",
                    "arguments": {"content": "Please turn off airplane mode and enable mobile data."}})
    assert res.terminated
    assert res.reward == 1.0
    assert env.end_reason == STOP_TOKEN
    # 验证是 user 侧状态被改(dual-control 核心)
    assert env.state.user_db["mobile_data"] is True
    assert env.state.user_db["airplane_mode"] is False


def test_telecom_truncation_when_no_progress():
    """agent 一直做无关操作 → 到 max_steps 截断，reward=0。"""
    task, tools = make_telecom_task()
    env = TauEnv(task, tools, max_steps=3)
    env.reset()
    r = None
    for _ in range(3):
        r = env.step({"name": "get_customer_by_phone", "arguments": {}})
    assert r.truncated
    assert r.reward == 0.0
    assert env.end_reason == "max_steps"


def test_agent_cannot_stop_itself():
    """对齐 τ-bench：agent 无 stop 动作，结束权在 user。agent 发普通 respond 不结束(未达标时)。"""
    task, tools = make_telecom_task()
    env = TauEnv(task, tools, max_steps=16)
    env.reset()
    res = env.step({"name": "respond", "arguments": {"content": "Let me check your account."}})
    assert not res.terminated  # 未达标，user 不会 STOP
