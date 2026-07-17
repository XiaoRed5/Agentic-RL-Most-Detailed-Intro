"""加载 golden fixtures(来自 Jarrodbarnes/tau2-sft-seed-v3 的真实轨迹)。

这些是所有离线单测的 ground truth：真实 query、真实多轮交互、gold reward_info。
单测通过"重放真实轨迹 → 我们算的 reward == gold reward"来验证实现正确性。
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"

# assistant 消息里的 tool_call 块： <tool_call>{...json...}</tool_call>
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# tool 结果回填在 user 轮的前缀： "Tool result for name(args): ..."
_TOOL_RESULT_RE = re.compile(r"^Tool result for\s+(\w+)\((.*?)\):\s*(.*)$", re.DOTALL)


def fixture_names() -> list[str]:
    manifest = json.loads((FIXTURES_DIR / "manifest.json").read_text())
    return [m["name"] for m in manifest]


@lru_cache(maxsize=None)
def load_fixture(name: str) -> dict[str, Any]:
    """加载一条 fixture(带缓存)。返回原始 dict：messages + gold + source。"""
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"fixture not found: {path}")
    return json.loads(path.read_text())


def parse_assistant_tool_call(content: str) -> dict[str, Any] | None:
    """从 assistant 消息文本里解出 tool_call。返回 {name, arguments} 或 None(纯文本)。"""
    m = _TOOL_CALL_RE.search(content or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def is_tool_result_message(content: str) -> bool:
    """判断一条 user 消息是否其实是工具结果回填(而非真实用户发言)。"""
    return bool(_TOOL_RESULT_RE.match((content or "").strip()))


def parse_tool_result(content: str) -> dict[str, Any] | None:
    """解析工具结果回填消息。返回 {tool_name, raw_args, result} 或 None。"""
    m = _TOOL_RESULT_RE.match((content or "").strip())
    if not m:
        return None
    return {"tool_name": m.group(1), "raw_args": m.group(2), "result": m.group(3).strip()}
