"""tau2-bench reward 计算 —— 精确复刻官方 evaluator 的乘法门控语义。

参考 tau2-bench 源码 evaluator_env.py：
    最终 reward = ∏ reward_breakdown[c]  for c in reward_basis
每个 component 独立算(DB 终态 hash 匹配 / ENV_ASSERTION 断言 / COMMUNICATE 子串)，
只有列在 reward_basis 里的才乘进最终 reward，其余仅作诊断。

离线单测策略：我们不重跑真实数据库(那需要完整 tau2 env)，而是复刻"给定
各 component 的 pass/fail → 按 basis 乘出 reward"这条聚合逻辑，并用真实 fixture 的
gold reward_info 对拍。真机迁移时，component 的 pass/fail 由真实 tau2 env 提供，
本聚合逻辑一字不改。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class RewardType(str, Enum):
    DB = "DB"
    ENV_ASSERTION = "ENV_ASSERTION"
    NL_ASSERTION = "NL_ASSERTION"
    ACTION = "ACTION"
    COMMUNICATE = "COMMUNICATE"


@dataclass
class RewardComponents:
    """各 reward component 的独立得分(∈[0,1])。None 表示该 component 不适用。"""

    db: Optional[float] = None
    env_assertion: Optional[float] = None
    nl_assertion: Optional[float] = None
    action: Optional[float] = None
    communicate: Optional[float] = None

    def get(self, rtype: RewardType) -> Optional[float]:
        return {
            RewardType.DB: self.db,
            RewardType.ENV_ASSERTION: self.env_assertion,
            RewardType.NL_ASSERTION: self.nl_assertion,
            RewardType.ACTION: self.action,
            RewardType.COMMUNICATE: self.communicate,
        }[rtype]


@dataclass
class RewardResult:
    reward: float
    breakdown: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


def compute_reward(components: RewardComponents,
                   reward_basis: list[RewardType | str]) -> RewardResult:
    """按 tau2 语义聚合最终 reward = ∏ component[c] for c in basis。

    若某个 basis 里的 component 缺失(None)，视为 0(该维度未达成)。
    """
    basis = [RewardType(b) if not isinstance(b, RewardType) else b for b in reward_basis]
    reward = 1.0
    breakdown: dict[str, float] = {}
    for rtype in basis:
        val = components.get(rtype)
        val = 0.0 if val is None else float(val)
        breakdown[rtype.value] = val
        reward *= val
    return RewardResult(reward=reward, breakdown=breakdown,
                        info={"reward_basis": [b.value for b in basis]})


# --- 各 component 的独立计算(离线可验，真机由 tau2 env 提供输入) ---

def db_reward_from_hash(predicted_hash: str, gold_hash: str) -> float:
    """DB 终态：预测数据库 hash 与目标 hash 完全一致才得 1。tau2 用 pydantic hash。"""
    return 1.0 if predicted_hash == gold_hash else 0.0


def env_assertion_reward(assertion_results: list[bool]) -> float:
    """ENV_ASSERTION：所有断言都 met 才得 1(乘积语义)。dual-control 的 telecom 用这个。"""
    r = 1.0
    for ok in assertion_results:
        r *= 1.0 if ok else 0.0
    return r


def communicate_reward(agent_utterances: list[str], required_infos: list[str]) -> float:
    """COMMUNICATE：每条 required_info 都必须作为子串出现在 agent 的某句话里。"""
    if not required_infos:
        return 1.0
    joined = "\n".join(agent_utterances)
    for need in required_infos:
        if need not in joined:
            return 0.0
    return 1.0


def action_reward(action_matches: list[bool]) -> float:
    """ACTION：参考轨迹里每个 action 都被 agent 的某次 tool call 匹配到才得 1。
    仅当 RewardType.ACTION 在 basis 时才 gate reward；否则仅诊断。"""
    if not action_matches:
        return 1.0
    r = 1.0
    for ok in action_matches:
        r *= 1.0 if ok else 0.0
    return r


def reconstruct_from_reward_info(reward_info: dict[str, Any]) -> RewardResult:
    """从 tau2 的 gold reward_info 反推 RewardResult —— 用于单测对拍。

    直接读取 fixture 里 tau2 官方算出的 db_check / env_assertions / reward_basis，
    走我们自己的 compute_reward，验证聚合逻辑与官方一致。
    """
    comp = RewardComponents()
    if reward_info.get("db_check") is not None:
        comp.db = float(reward_info["db_check"].get("db_reward", 0.0))
    env_asserts = reward_info.get("env_assertions") or []
    if env_asserts:
        comp.env_assertion = env_assertion_reward([a.get("met", False) for a in env_asserts])
    comm = reward_info.get("communicate_checks")
    if comm:
        comp.communicate = 1.0 if all(c.get("met", False) for c in comm) else 0.0
    else:
        # basis 含 COMMUNICATE 但无 checks → tau2 视为满足(无要求)
        comp.communicate = 1.0
    action_checks = reward_info.get("action_checks") or []
    if action_checks:
        comp.action = action_reward([a.get("action_match", False) for a in action_checks])

    basis = reward_info.get("reward_basis") or ["DB", "COMMUNICATE"]
    return compute_reward(comp, basis)
