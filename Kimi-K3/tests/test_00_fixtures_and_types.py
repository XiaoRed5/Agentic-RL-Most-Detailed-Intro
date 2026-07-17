"""冒烟测试：验证 golden fixtures 完整、可解析，且核心类型能承载真实轨迹。

这是整个测试套件的地基 —— 如果这些不过，后面 env/rollout/credit 全无从谈起。
"""
import json

import pytest

from agentic_rl.utils.fixtures import (
    fixture_names,
    is_tool_result_message,
    parse_assistant_tool_call,
    parse_tool_result,
)
from agentic_rl.utils.types import (
    ActionType,
    Role,
    Sample,
    Status,
    Trajectory,
    Turn,
)


def test_all_fixtures_present():
    names = fixture_names()
    assert set(names) == {
        "retail_success",
        "airline_success",
        "airline_fail",
        "telecom_success_dualctrl",
        "telecom_fail_dualctrl",
    }


def test_fixtures_have_gold_reward(all_fixtures):
    """每条 fixture 必须带真实 gold reward_info(离线单测的 ground truth)。"""
    for name, fx in all_fixtures.items():
        assert "gold" in fx, name
        assert "reward_info" in fx["gold"], name
        assert fx["gold"]["reward"] in (0.0, 1.0), name
        # reward 与 reward_info 内部一致
        assert fx["gold"]["reward"] == fx["gold"]["reward_info"]["reward"], name


def test_success_fail_split(all_fixtures):
    assert all_fixtures["retail_success"]["gold"]["reward"] == 1.0
    assert all_fixtures["telecom_success_dualctrl"]["gold"]["reward"] == 1.0
    assert all_fixtures["airline_fail"]["gold"]["reward"] == 0.0
    assert all_fixtures["telecom_fail_dualctrl"]["gold"]["reward"] == 0.0


def test_messages_are_real_multiturn(all_fixtures):
    """轨迹必须是真实多轮：system 开头，含 assistant 和 user 交替。"""
    for name, fx in all_fixtures.items():
        msgs = fx["messages"]
        assert len(msgs) >= 10, name
        assert msgs[0]["role"] == "system", name
        roles = {m["role"] for m in msgs}
        assert "assistant" in roles and "user" in roles, name


def test_assistant_tool_calls_parse(retail_success):
    """assistant 的 <tool_call> 块可解析出 name + arguments。"""
    found = 0
    for m in retail_success["messages"]:
        if m["role"] == "assistant":
            tc = parse_assistant_tool_call(m["content"])
            if tc is not None:
                assert "name" in tc and "arguments" in tc
                found += 1
    assert found >= 1, "至少应有一个可解析的 tool_call"


def test_tool_results_distinguishable_from_user(retail_success):
    """区分'工具结果回填'(伪 user 轮) vs 真实用户发言 —— masking 的关键前提。"""
    tool_results, real_user = 0, 0
    for m in retail_success["messages"]:
        if m["role"] == "user":
            if is_tool_result_message(m["content"]):
                parsed = parse_tool_result(m["content"])
                assert parsed is not None and parsed["tool_name"]
                tool_results += 1
            else:
                real_user += 1
    assert tool_results >= 1, "retail_success 应含工具结果回填"
    assert real_user >= 1, "也应含真实用户发言"


def test_types_carry_trajectory():
    """核心类型能承载一条 mini 轨迹，并正确拍平出 token + loss_mask。"""
    traj = Trajectory(task_id="t", domain="retail", prompt_token_ids=[1, 2, 3])
    # 第 1 轮：agent 生成 4 个 token，环境返回 3 个 token
    traj.turns.append(
        Turn(index=0, action_type=ActionType.TOOL_CALL, action_name="get_user_details",
             assistant_token_ids=[10, 11, 12, 13], obs_token_ids=[20, 21, 22])
    )
    # 第 2 轮：agent 对用户说话(user-involved)，2 个 token；无环境返回
    traj.turns.append(
        Turn(index=1, action_type=ActionType.RESPOND, action_name="respond",
             assistant_token_ids=[30, 31], is_user_involved=True)
    )
    traj.outcome_reward = 1.0

    assert traj.n_turns == 2
    assert traj.all_token_ids() == [1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 30, 31]
    # loss_mask: prompt(3个0) + turn0(assistant 4个1, env 3个0) + turn1(2个1)
    assert traj.full_loss_mask() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1]
    assert traj.response_length == 9
    assert traj.num_user_involved_turns() == 1


def test_sample_from_trajectory():
    traj = Trajectory(task_id="t", domain="retail", prompt_token_ids=[1, 2])
    traj.turns.append(
        Turn(index=0, action_type=ActionType.ANSWER, assistant_token_ids=[5, 6],
             obs_token_ids=[7])
    )
    traj.outcome_reward = 1.0
    traj.status = Status.COMPLETED
    s = Sample.from_trajectory(traj, index=3)
    assert s.reward == 1.0
    assert s.tokens == [1, 2, 5, 6, 7]
    assert s.loss_mask == [1, 1, 0]        # response 部分：assistant 2个1 + env 1个0
    assert s.response_length == 3
    assert s.status == "completed"
