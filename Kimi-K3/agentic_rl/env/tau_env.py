"""离线 tau2 环境 mock —— 契约兼容 gymnasium 五元组 step()→(obs,reward,term,trunc,info)。

设计要点(对齐 survey ① τ-bench + tau2 源码)：
1. **有状态 DB**：工具调用会真改状态(退款、改签、reset SIM)，reward 靠终态断言。
2. **确定性 user simulator**：真机是 LLM 扮演用户；离线用脚本化确定性用户(可复现)，
   会在任务达成时发 ###STOP###、无法处理时发 ###TRANSFER###。
3. **dual-control**：telecom 域 agent 侧和 user 侧各持工具，两块状态都要改对才算成
   (env_assertion 语义)。
4. **结束权在 user**：agent 无权主动 stop(is_stop 恒 False)，对齐 τ-bench 机制。

这个 mock 不追求覆盖 tau2 全部工具，而是提供一个**结构同构、reward 语义精确**的
最小环境，让 rollout / masking / credit / algo 全链路能在 CPU 上跑通并单测。
真机换成真实 tau2 env 时，rollout 代码调用的 step() 接口一致。
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agentic_rl.env.reward import (
    RewardComponents,
    RewardType,
    compute_reward,
)

# user simulator 结束标记 —— 对齐 τ-bench
STOP_TOKEN = "###STOP###"
TRANSFER_TOKEN = "###TRANSFER###"
OUT_OF_SCOPE_TOKEN = "###OUT-OF-SCOPE###"


@dataclass
class ToolSpec:
    """一个工具的定义：名字、是否改状态、执行函数、归属侧(agent/user)。"""

    name: str
    mutating: bool
    fn: Callable[["EnvState", dict[str, Any]], Any]
    side: str = "agent"  # "agent"(客服后台) | "user"(用户手机端，dual-control)
    description: str = ""


@dataclass
class EnvState:
    """环境的可变状态。DB 是任意可 JSON 序列化的 dict，其 hash 决定 DB reward。"""

    db: dict[str, Any] = field(default_factory=dict)          # agent 侧后台数据库
    user_db: dict[str, Any] = field(default_factory=dict)     # user 侧本地状态(dual-control)
    scratch: dict[str, Any] = field(default_factory=dict)     # 临时(如认证状态)

    def db_hash(self) -> str:
        return _stable_hash(self.db)

    def user_db_hash(self) -> str:
        return _stable_hash(self.user_db)


def _stable_hash(obj: Any) -> str:
    """稳定 hash：JSON 规范化后 sha256。对齐 tau2 用 pydantic 序列化再 hash 的思想。"""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class TaskSpec:
    """一个离线任务：初始状态 + 目标(用于算 reward) + 确定性用户脚本。"""

    task_id: str
    domain: str
    system_prompt: str
    initial_user_msg: str
    init_db: dict[str, Any]
    init_user_db: dict[str, Any] = field(default_factory=dict)
    reward_basis: list[str] = field(default_factory=lambda: ["DB", "COMMUNICATE"])

    # 目标：达成后各 component 的判定所需
    target_db: Optional[dict[str, Any]] = None
    target_user_db: Optional[dict[str, Any]] = None
    env_assertions: list[Callable[[EnvState], bool]] = field(default_factory=list)
    communicate_info: list[str] = field(default_factory=list)

    # 确定性 user simulator：给定到目前为止的 agent 话语 + 环境状态，返回用户回复。
    # 返回 (utterance, end_token or None)。end_token ∈ {STOP,TRANSFER,None}
    user_script: Optional[Callable[["TauEnv", str], tuple[str, Optional[str]]]] = None


@dataclass
class StepResult:
    observation: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class TauEnv:
    """离线 tau2 环境。用法对齐 gym：

        env = TauEnv(task); obs = env.reset()
        while not done:
            result = env.step(agent_action)   # agent_action: {name, arguments}
            obs, reward, term, trunc = result.observation, result.reward, ...
    """

    def __init__(self, task: TaskSpec, tools: dict[str, ToolSpec], max_steps: int = 16):
        self.task = task
        self.tools = tools
        self.max_steps = max_steps
        self.state = EnvState()
        self.step_count = 0
        self.done = False
        self.agent_utterances: list[str] = []   # 累积 agent 对用户说的话(COMMUNICATE 用)
        self.end_reason: Optional[str] = None
        self._authenticated = False

    # ---- lifecycle ----
    def reset(self) -> str:
        self.state = EnvState(
            db=copy.deepcopy(self.task.init_db),
            user_db=copy.deepcopy(self.task.init_user_db),
        )
        self.step_count = 0
        self.done = False
        self.agent_utterances = []
        self.end_reason = None
        return self.task.initial_user_msg

    # ---- 核心 step ----
    def step(self, action: dict[str, Any]) -> StepResult:
        """执行 agent 一个动作。action = {"name": str, "arguments": dict}。

        - name == "respond"：对用户说话 → 触发确定性 user simulator 回复。
          user 可能发 STOP/TRANSFER 结束会话。
        - 其它 name：调用工具(可能改 DB 状态)，返回工具结果字符串。
        """
        if self.done:
            raise RuntimeError("step() called on a finished episode")
        self.step_count += 1
        name = action.get("name", "")
        args = action.get("arguments", {}) or {}

        # 截断：超预算
        truncated = False
        terminated = False
        reward = 0.0
        info: dict[str, Any] = {"action_name": name}

        if name == "respond":
            utterance = str(args.get("content", ""))
            self.agent_utterances.append(utterance)
            obs, end_token = self._run_user(utterance)
            info["user_end_token"] = end_token
            if end_token in (STOP_TOKEN, TRANSFER_TOKEN, OUT_OF_SCOPE_TOKEN):
                terminated = True
                self.done = True
                self.end_reason = end_token
                reward = self._final_reward()
                info["reward_breakdown"] = self._last_breakdown
        else:
            obs = self._run_tool(name, args, info)

        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            self.done = True
            self.end_reason = "max_steps"
            reward = self._final_reward()   # 截断时也结算(通常拿不到满分)
            info["reward_breakdown"] = self._last_breakdown

        return StepResult(obs, reward, terminated, truncated, info)

    # ---- 工具执行 ----
    def _run_tool(self, name: str, args: dict[str, Any], info: dict[str, Any]) -> str:
        spec = self.tools.get(name)
        if spec is None:
            info["error"] = "unknown_tool"
            return f"Error: unknown tool '{name}'"
        try:
            result = spec.fn(self.state, args)
        except Exception as e:  # 工具内部错误 → 返回错误串，交给 agent 处理(容错)
            info["error"] = str(e)
            return f"Tool error in {name}: {e}"
        info["tool_side"] = spec.side
        info["mutating"] = spec.mutating
        return f"Tool result for {name}: {json.dumps(result, ensure_ascii=False)}"

    # ---- 确定性 user simulator ----
    def _run_user(self, agent_utterance: str) -> tuple[str, Optional[str]]:
        if self.task.user_script is not None:
            return self.task.user_script(self, agent_utterance)
        # 默认脚本：一旦目标达成就 STOP，否则给中性回复
        if self._goal_reached():
            return ("Great, that solved it. Thank you!", STOP_TOKEN)
        return ("Okay, please continue.", None)

    # ---- reward ----
    def _goal_reached(self) -> bool:
        comp = self._components()
        res = compute_reward(comp, self.task.reward_basis)
        return res.reward >= 1.0

    def _components(self) -> RewardComponents:
        comp = RewardComponents()
        t = self.task
        if t.target_db is not None:
            comp.db = 1.0 if self.state.db_hash() == _stable_hash(t.target_db) else 0.0
        if t.env_assertions:
            r = 1.0
            for a in t.env_assertions:
                r *= 1.0 if a(self.state) else 0.0
            comp.env_assertion = r
        # COMMUNICATE：required info 都出现在 agent 话语里
        if t.communicate_info:
            joined = "\n".join(self.agent_utterances)
            comp.communicate = 1.0 if all(s in joined for s in t.communicate_info) else 0.0
        else:
            comp.communicate = 1.0
        return comp

    _last_breakdown: dict[str, float] = {}

    def _final_reward(self) -> float:
        comp = self._components()
        res = compute_reward(comp, self.task.reward_basis)
        self._last_breakdown = res.breakdown
        # TRANSFER/OUT_OF_SCOPE 视为未完成任务(除非恰好达成)
        return res.reward
