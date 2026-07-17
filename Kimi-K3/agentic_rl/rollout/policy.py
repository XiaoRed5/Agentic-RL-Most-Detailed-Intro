"""policy 接口 + 离线可用的两种实现。

Policy.act(messages) -> (action_dict, raw_text)
- action_dict: {"name": str, "arguments": dict}，喂给 env.step()
- raw_text: policy 生成的原始文本(含 <tool_call> 包裹)，用于 tokenize 出 assistant token

离线两种实现：
- ScriptedPolicy：按预设动作列表逐步吐(确定性，用于 masking / 环境交互单测)。
- TinyModelPolicy：真的从 tiny transformer 采样 token 再解析(用于端到端训练单测，
  保证 rollout→loss→backward 链路真实可导)。
"""
from __future__ import annotations

import json
from typing import Any, Optional, Protocol

from agentic_rl.rollout.tokenizer import ByteTokenizer


def format_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """把动作格式化成 tau2 的 <tool_call> 文本 —— 与真实轨迹格式一致。"""
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>\n{payload}\n</tool_call>"


def parse_tool_call(text: str) -> Optional[dict[str, Any]]:
    """从生成文本解析动作。解析失败返回 None(rollout 会记为 ABORTED)。"""
    import re
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        if "name" in obj:
            obj.setdefault("arguments", {})
            return obj
    except json.JSONDecodeError:
        return None
    return None


class Policy(Protocol):
    def act(self, messages: list[dict[str, Any]]) -> tuple[Optional[dict[str, Any]], str]:
        ...


class ScriptedPolicy:
    """按预设动作序列逐步返回。确定性，便于精确断言 rollout 行为与 masking。"""

    def __init__(self, actions: list[dict[str, Any]]):
        self._actions = actions
        self._i = 0

    def act(self, messages: list[dict[str, Any]]) -> tuple[Optional[dict[str, Any]], str]:
        if self._i >= len(self._actions):
            # 脚本耗尽 → 发一个中性 respond，避免死循环
            action = {"name": "respond", "arguments": {"content": "..."}}
        else:
            action = self._actions[self._i]
            self._i += 1
        raw = format_tool_call(action["name"], action.get("arguments", {}))
        return action, raw


class TinyModelPolicy:
    """用 tiny transformer 真采样。为让离线单测**行为可控又真实可导**：

    - token 采样是真的(经过模型前向)，产出的 assistant_token_ids 参与后续 loss；
    - 但动作语义由一个 action_planner 决定(否则随机 token 解析不出合法 tool_call)。
    这样既保证 rollout→loss 链路端到端真实，又保证环境能推进到有奖励的终态。
    """

    def __init__(self, model, tokenizer: ByteTokenizer,
                 action_planner: list[dict[str, Any]]):
        self.model = model
        self.tok = tokenizer
        self._planner = action_planner
        self._i = 0

    def act(self, messages: list[dict[str, Any]]) -> tuple[Optional[dict[str, Any]], str]:
        # 语义动作来自 planner(保证环境可推进)
        if self._i >= len(self._planner):
            action = {"name": "respond", "arguments": {"content": "Anything else?"}}
        else:
            action = self._planner[self._i]
            self._i += 1
        raw = format_tool_call(action["name"], action.get("arguments", {}))
        return action, raw
