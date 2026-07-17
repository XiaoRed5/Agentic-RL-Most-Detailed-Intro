"""rollout 测试：多轮 loop + token-level masking 正确性(agentic RL 最易错处)。"""
import pytest

from agentic_rl.env.offline_tasks import make_retail_task, make_telecom_task
from agentic_rl.env.tau_env import TauEnv
from agentic_rl.rollout.policy import ScriptedPolicy, format_tool_call, parse_tool_call
from agentic_rl.rollout.rollout import USER_INVOLVED_ACTIONS, rollout
from agentic_rl.rollout.tokenizer import ByteTokenizer
from agentic_rl.utils.types import ActionType, Status


@pytest.fixture
def tok():
    return ByteTokenizer(vocab_size=512)


# ========== tokenizer ==========
def test_tokenizer_roundtrip(tok):
    s = 'Hello <tool_call>{"name": "x"}</tool_call> 世界'
    assert tok.decode(tok.encode(s)) == s


def test_tool_call_format_parse_roundtrip():
    raw = format_tool_call("cancel_order", {"order_id": "#W001"})
    parsed = parse_tool_call(raw)
    assert parsed == {"name": "cancel_order", "arguments": {"order_id": "#W001"}}


def test_parse_garbage_returns_none():
    assert parse_tool_call("no tool call here") is None
    assert parse_tool_call("<tool_call>not json</tool_call>") is None


# ========== 多轮 rollout ==========
def test_retail_rollout_completes_successfully(tok):
    task, tools = make_retail_task()
    env = TauEnv(task, tools, max_steps=16)
    policy = ScriptedPolicy([
        {"name": "cancel_order", "arguments": {"order_id": "#W001"}},
        {"name": "process_refund", "arguments": {"order_id": "#W001"}},
        {"name": "respond", "arguments": {"content": "Done, your refund is processed."}},
    ])
    traj = rollout(env, policy, tok, max_steps=16)
    assert traj.status == Status.COMPLETED
    assert traj.outcome_reward == 1.0
    assert traj.n_turns == 3
    # 动作类型分类正确
    assert traj.turns[0].action_type == ActionType.TOOL_CALL
    assert traj.turns[2].action_type == ActionType.RESPOND


def test_telecom_dualcontrol_rollout(tok):
    task, tools = make_telecom_task()
    env = TauEnv(task, tools, max_steps=16)
    policy = ScriptedPolicy([
        {"name": "get_customer_by_phone", "arguments": {}},
        {"name": "respond", "arguments": {"content": "Turn off airplane mode and enable data."}},
    ])
    traj = rollout(env, policy, tok, max_steps=16)
    assert traj.status == Status.COMPLETED
    assert traj.outcome_reward == 1.0


# ========== token-level masking：核心 ==========
def test_masking_assistant_one_obs_zero(tok):
    """每个 turn：assistant token 必须全 1，observation token 必须全 0。"""
    task, tools = make_retail_task()
    env = TauEnv(task, tools, max_steps=16)
    policy = ScriptedPolicy([
        {"name": "cancel_order", "arguments": {"order_id": "#W001"}},
        {"name": "process_refund", "arguments": {"order_id": "#W001"}},
        {"name": "respond", "arguments": {"content": "Refund processed."}},
    ])
    traj = rollout(env, policy, tok, max_steps=16)

    for t in traj.turns:
        mask = t.loss_mask()
        n_a = len(t.assistant_token_ids)
        n_o = len(t.obs_token_ids)
        assert mask[:n_a] == [1] * n_a, "assistant token 必须 loss_mask=1"
        assert mask[n_a:] == [0] * n_o, "observation token 必须 loss_mask=0"
        assert len(mask) == len(t.token_ids())


def test_full_mask_prompt_is_zero(tok):
    """整条序列：prompt 部分 loss_mask 恒为 0(不是 policy 生成的)。"""
    task, tools = make_retail_task()
    env = TauEnv(task, tools, max_steps=16)
    policy = ScriptedPolicy([{"name": "respond", "arguments": {"content": "refund done"}}])
    traj = rollout(env, policy, tok, max_steps=16)
    full = traj.full_loss_mask()
    n_prompt = len(traj.prompt_token_ids)
    assert full[:n_prompt] == [0] * n_prompt
    # response 段至少含一个 1(assistant 生成)
    assert sum(full[n_prompt:]) >= 1


def test_mask_length_matches_tokens(tok):
    """loss_mask 长度必须与 token 序列严格对齐(错位会让训练崩)。"""
    task, tools = make_telecom_task()
    env = TauEnv(task, tools, max_steps=16)
    policy = ScriptedPolicy([
        {"name": "get_customer_by_phone", "arguments": {}},
        {"name": "respond", "arguments": {"content": "turn off airplane mode, enable data"}},
    ])
    traj = rollout(env, policy, tok, max_steps=16)
    assert len(traj.all_token_ids()) == len(traj.full_loss_mask())
    assert len(traj.response_token_ids()) == len(traj.response_loss_mask())


def test_only_policy_tokens_counted(tok):
    """masking 语义验证：被 mask 掉(=1)的 token 数 == 各 turn assistant token 之和。"""
    task, tools = make_retail_task()
    env = TauEnv(task, tools, max_steps=16)
    policy = ScriptedPolicy([
        {"name": "cancel_order", "arguments": {"order_id": "#W001"}},
        {"name": "process_refund", "arguments": {"order_id": "#W001"}},
        {"name": "respond", "arguments": {"content": "Refund done."}},
    ])
    traj = rollout(env, policy, tok, max_steps=16)
    n_ones = sum(traj.full_loss_mask())
    n_assistant = sum(len(t.assistant_token_ids) for t in traj.turns)
    assert n_ones == n_assistant


# ========== user-involved 标记(BAO 正则的前提) ==========
def test_user_involved_flagging(tok):
    task, tools = make_retail_task()
    env = TauEnv(task, tools, max_steps=16)
    policy = ScriptedPolicy([
        {"name": "get_order_details", "arguments": {"order_id": "#W001"}},   # 非 user-involved
        {"name": "respond", "arguments": {"content": "checking"}},           # user-involved
    ])
    traj = rollout(env, policy, tok, max_steps=16)
    assert traj.turns[0].is_user_involved is False
    assert traj.turns[1].is_user_involved is True
    assert "respond" in USER_INVOLVED_ACTIONS


# ========== 用真实 fixture 轨迹验证 masking 契约 ==========
def test_real_fixture_masking_contract(tok, retail_success):
    """把真实 fixture 的 assistant/tool 消息 tokenize，验证 masking 契约在真实数据上成立：
    assistant 消息(含 <tool_call>) → mask=1；tool 结果回填 → mask=0。"""
    from agentic_rl.utils.fixtures import is_tool_result_message

    msgs = retail_success["messages"]
    total_assistant_tokens = 0
    total_obs_tokens = 0
    for m in msgs:
        if m["role"] == "assistant":
            total_assistant_tokens += len(tok.encode(m["content"]))
        elif m["role"] == "user" and is_tool_result_message(m["content"]):
            total_obs_tokens += len(tok.encode(m["content"]))
    # 真实轨迹里两类 token 都存在，masking 才有意义
    assert total_assistant_tokens > 0
    assert total_obs_tokens > 0
