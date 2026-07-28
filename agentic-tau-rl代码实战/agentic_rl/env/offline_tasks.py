"""具体离线任务定义：retail 退款 + telecom dual-control 修网络。

这两个任务是离线 rollout/训练的真实环境实例，结构对齐 tau2：
- retail：单控，agent 调后台工具改订单状态，reward_basis=[DB, COMMUNICATE]。
- telecom：dual-control，agent 指导用户操作手机(user 侧工具)，reward_basis=[ENV_ASSERTION]。
  agent 不能替用户按手机，只能 respond 指导；用户脚本按指导改 user_db。
"""
from __future__ import annotations

from typing import Any, Optional

from agentic_rl.env.tau_env import (
    STOP_TOKEN,
    TRANSFER_TOKEN,
    EnvState,
    TaskSpec,
    TauEnv,
    ToolSpec,
)


# ===================== RETAIL：取消订单退款 =====================
def _retail_tools() -> dict[str, ToolSpec]:
    def find_user(state: EnvState, args: dict[str, Any]):
        return {"user_id": "chen_smith_8425"}

    def get_order(state: EnvState, args: dict[str, Any]):
        oid = args.get("order_id", "#W001")
        return state.db.get("orders", {}).get(oid, {"error": "not found"})

    def cancel_order(state: EnvState, args: dict[str, Any]):
        oid = args["order_id"]
        order = state.db["orders"][oid]
        order["status"] = "cancelled"
        return {"order_id": oid, "status": "cancelled"}

    def process_refund(state: EnvState, args: dict[str, Any]):
        oid = args["order_id"]
        order = state.db["orders"][oid]
        order["refunded"] = True
        return {"order_id": oid, "refunded": True}

    return {
        "find_user_id_by_name_zip": ToolSpec("find_user_id_by_name_zip", False, find_user),
        "get_order_details": ToolSpec("get_order_details", False, get_order),
        "cancel_order": ToolSpec("cancel_order", True, cancel_order),
        "process_refund": ToolSpec("process_refund", True, process_refund),
    }


def _retail_user_script(env: "TauEnv", agent_utterance: str) -> tuple[str, Optional[str]]:
    """确定性用户：任务(退款)达成即 STOP；被拒绝(agent 说 policy) 会坚持一次后 TRANSFER。"""
    if env._goal_reached():
        return ("Perfect, I see the refund. Thank you so much!", STOP_TOKEN)
    low = agent_utterance.lower()
    if "cannot" in low or "policy" in low or "unable" in low:
        turns_said = len(env.agent_utterances)
        if turns_said >= 2:
            return ("Fine, please transfer me to a human then.", TRANSFER_TOKEN)
        return ("I really need this refund, please check again.", None)
    return ("Okay, my order id is #W001. Please proceed.", None)


def make_retail_task() -> tuple[TaskSpec, dict[str, ToolSpec]]:
    init_db = {"orders": {"#W001": {"status": "pending", "refunded": False, "total": 200}}}
    target_db = {"orders": {"#W001": {"status": "cancelled", "refunded": True, "total": 200}}}
    task = TaskSpec(
        task_id="offline_retail_refund",
        domain="retail",
        system_prompt="You are a retail customer-support agent. Cancel the order and refund.",
        initial_user_msg="Hi, I need to cancel order #W001 and get a refund.",
        init_db=init_db,
        reward_basis=["DB", "COMMUNICATE"],
        target_db=target_db,
        communicate_info=["refund"],   # agent 必须向用户提到 refund
        user_script=_retail_user_script,
    )
    return task, _retail_tools()


# ===================== TELECOM：dual-control 修网络 =====================
def _telecom_tools() -> dict[str, ToolSpec]:
    # agent 侧后台工具
    def get_customer(state: EnvState, args: dict[str, Any]):
        return {"line_status": state.db.get("line_status", "active")}

    def reset_apn(state: EnvState, args: dict[str, Any]):
        state.db["apn"] = "correct"
        return {"apn": "correct"}

    # user 侧工具(dual-control)：由用户脚本在收到 agent 指导后调用
    def toggle_airplane_mode(state: EnvState, args: dict[str, Any]):
        state.user_db["airplane_mode"] = not state.user_db.get("airplane_mode", True)
        return {"airplane_mode": state.user_db["airplane_mode"]}

    def toggle_data(state: EnvState, args: dict[str, Any]):
        state.user_db["mobile_data"] = True
        return {"mobile_data": True}

    return {
        "get_customer_by_phone": ToolSpec("get_customer_by_phone", False, get_customer, side="agent"),
        "reset_apn_settings": ToolSpec("reset_apn_settings", True, reset_apn, side="agent"),
        "toggle_airplane_mode": ToolSpec("toggle_airplane_mode", True, toggle_airplane_mode, side="user"),
        "toggle_data": ToolSpec("toggle_data", True, toggle_data, side="user"),
    }


def _telecom_user_script(env: "TauEnv", agent_utterance: str) -> tuple[str, Optional[str]]:
    """dual-control 用户：按 agent 的指导操作手机(改 user_db)。

    - agent 说"关飞行模式" → 用户 toggle_airplane_mode
    - agent 说"打开移动数据" → 用户 toggle_data
    目标达成(mobile_data on 且 airplane off) → STOP。
    """
    low = agent_utterance.lower()
    tools = env.tools
    if "airplane" in low:
        tools["toggle_airplane_mode"].fn(env.state, {})
    if "data" in low or "mobile data" in low:
        tools["toggle_data"].fn(env.state, {})
    if env._goal_reached():
        return ("It works now, I have signal and data! Thanks!", STOP_TOKEN)
    return ("Okay, I did that. What next?", None)


def make_telecom_task() -> tuple[TaskSpec, dict[str, ToolSpec]]:
    tools = _telecom_tools()
    task = TaskSpec(
        task_id="offline_telecom_dualctrl",
        domain="telecom",
        system_prompt="You are a telecom support agent. Guide the user to fix mobile data.",
        initial_user_msg="My phone has no mobile data. Help me fix it.",
        init_db={"line_status": "active", "apn": "correct"},
        init_user_db={"airplane_mode": True, "mobile_data": False},
        reward_basis=["ENV_ASSERTION"],
        env_assertions=[
            lambda s: s.user_db.get("mobile_data") is True,
            lambda s: s.user_db.get("airplane_mode") is False,
        ],
        user_script=_telecom_user_script,
    )
    return task, tools


OFFLINE_TASKS = {
    "retail": make_retail_task,
    "telecom": make_telecom_task,
}
