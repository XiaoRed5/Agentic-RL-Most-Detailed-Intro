"""行为塑形正则：BAO(减法压副作用) + PPP(加法正向优化)。

对齐 survey ③ BAO 和 ⑤ PPP。这些在 turn-level / trajectory-level 修正 reward，
治 UserRL 练出来的 agent"话太多、想太多"的病，或正向拉起交互质量。

注意(survey 诚实提醒)：BAO 的 λ_ans / λ_think / w 论文未公开数值，此处用占位默认，
形式与思想严格对齐论文公式。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from agentic_rl.utils.types import Status, Trajectory


# ============ BAO 正则① Information-Seeking ============
def bao_information_seeking_penalty(traj: Trajectory, lambda_ans: float = 0.1) -> list[float]:
    """连续两轮都问用户(a_t, a_{t-1} ∈ 𝒜_u) → 在第 t 轮扣 -λ_ans。

    纯行为判据(论文 Eq.2)：不计算真实信息增益，只看"上一轮问了用户、这一轮又问用户"
    这个可观测条件。逼 agent 每次问完用户先去环境干点实事，别连珠炮追问。
    返回每轮的惩罚(<=0)列表。
    """
    penalties = [0.0] * traj.n_turns
    for t in range(1, traj.n_turns):
        if traj.turns[t].is_user_involved and traj.turns[t - 1].is_user_involved:
            penalties[t] = -lambda_ans
    return penalties


# ============ BAO 正则② Over-Thinking ============
def bao_over_thinking_penalty(
    traj: Trajectory,
    lambda_think: float = 0.1,
    turn_budget: int = 16,
) -> float:
    """失败且提前终止(T' < T) → 罚 -λ_think·(T-T')/T'(非线性放大早失败)。

    T = 固定交互预算(轮数)；T' = 该失败轨迹实际交互轮数。只加在失败且 T'<T 的轨迹上。
    因果链：过度空想 → 提前烧完预算 → 交互不够就失败 → 这个式子狠罚它。
    返回轨迹级惩罚(<=0)。
    """
    failed = traj.outcome_reward < 1.0
    t_prime = traj.n_turns
    if not failed or t_prime >= turn_budget or t_prime <= 0:
        return 0.0
    return -lambda_think * (turn_budget - t_prime) / t_prime


def apply_bao_shaping(
    traj: Trajectory,
    lambda_ans: float = 0.1,
    lambda_think: float = 0.1,
    turn_budget: int = 16,
    info_seeking: bool = True,
    over_thinking: bool = True,
) -> None:
    """把 BAO 两条正则写进每个 Turn.shaped_reward(= immediate + 惩罚)。"""
    turn_pen = [0.0] * traj.n_turns
    if info_seeking:
        for i, p in enumerate(bao_information_seeking_penalty(traj, lambda_ans)):
            turn_pen[i] += p
    if over_thinking:
        # 过早终止惩罚记在最后一轮(轨迹级 → 落到 terminal turn)
        pen = bao_over_thinking_penalty(traj, lambda_think, turn_budget)
        if traj.n_turns > 0:
            turn_pen[-1] += pen
    for turn, p in zip(traj.turns, turn_pen):
        turn.shaped_reward = turn.immediate_reward + p


# ============ PPP 三目标(加法) ============
class Effort(str, Enum):
    LOW = "low"       # 合理澄清，用完整 prompt 已有信息可答
    MEDIUM = "medium" # 用户也答不上(拒答)
    HIGH = "high"     # 甩活给用户，逼其掏敏感信息


@dataclass
class PPPRewards:
    productivity: float   # 活干成没有 ∈[0,1](大头)
    proactivity: float    # 该问才问(小修正)
    personalization: float  # 合偏好(小修正)

    @property
    def total(self) -> float:
        # survey：无权重直接相加
        return self.productivity + self.proactivity + self.personalization


def ppp_proactivity_reward(
    efforts: list[Effort],
    low_bonus: float = 0.05,
    medium_pen: float = -0.1,
    high_pen: float = -0.5,
) -> float:
    """R_Proact = +low_bonus·[所有提问都 low] + Σ(每个 medium 罚 + 每个 high 罚)。

    对齐 survey：合理澄清几乎不扣分，瞎问把活甩回给用户则重罚。
    """
    if not efforts:
        return 0.0
    r = 0.0
    if all(e == Effort.LOW for e in efforts):
        r += low_bonus
    for e in efforts:
        if e == Effort.MEDIUM:
            r += medium_pen
        elif e == Effort.HIGH:
            r += high_pen
    return r


def ppp_personalization_reward(
    satisfied: list[bool],
    violated_penalties: list[float],
    bonus: float = 0.05,
) -> float:
    """R_Pers = Σ(满足一条偏好 +bonus) - Σ(违反的偏好各自的惩罚)。"""
    r = 0.0
    for ok in satisfied:
        if ok:
            r += bonus
    for pen in violated_penalties:
        r -= abs(pen)
    return r


def ppp_total_reward(
    productivity: float,
    efforts: list[Effort],
    pref_satisfied: list[bool],
    pref_violated_pen: list[float],
    low_bonus: float = 0.05,
    medium_pen: float = -0.1,
    high_pen: float = -0.5,
    pref_bonus: float = 0.05,
) -> PPPRewards:
    """组装 PPP 三块。survey：Prod 是 0~1 大头，Proact/Pers 是 ±0.05~-0.5 小修正。"""
    return PPPRewards(
        productivity=productivity,
        proactivity=ppp_proactivity_reward(efforts, low_bonus, medium_pen, high_pen),
        personalization=ppp_personalization_reward(pref_satisfied, pref_violated_pen, pref_bonus),
    )


def ppp_normalized_advantage(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """PPP 优势：先 clip 到 [0,1] 再组内标准化 Â=[clip(R,0,1)-mean]/std。

    survey 关键坑：先 clip 到 0~1，负罚会被压进边界 → 所以惩罚量级必须刻意压小，
    否则在归一化里被大头 Prod 抹平、传不到梯度。此函数如实实现该 clip 语义。
    """
    import torch
    r = torch.tensor(rewards, dtype=torch.float32).clamp(0.0, 1.0)
    if r.numel() <= 1:
        return [0.0 for _ in rewards]
    return ((r - r.mean()) / (r.std(unbiased=False) + eps)).tolist()


# ============ PPP effort 解析：[Cost N] → 三级 ============
import re as _re

_COST_RE = _re.compile(r"\[Cost\s+(\d+)\]")


def parse_effort_from_cost(user_reply: str) -> Optional[Effort]:
    """从模拟用户回复里解析 [Cost N] 标签 → effort 三级(对齐 survey ⑤ PPP 机制)。

    survey 精华：effort 没有单独评判 prompt —— 模拟用户在回复里自己吐 [Cost N]，
    N 越大表示"回答这个问题被迫掏出越敏感的底牌"。SWE 打分档(1~5)：
      Cost1        = 只用完整问题里的信息就能答      → low(合理澄清)
      Cost2-3      = 用户答不上/拒答                  → medium
      Cost4-5      = 用到文件路径/函数名等敏感信息    → high(甩活给用户)
    无标签 → None(视为无提问或无法评估)。
    """
    m = _COST_RE.search(user_reply or "")
    if not m:
        return None
    cost = int(m.group(1))
    if cost <= 1:
        return Effort.LOW
    if cost <= 3:
        return Effort.MEDIUM
    return Effort.HIGH


def efforts_from_user_replies(user_replies: list[str]) -> list[Effort]:
    """批量解析多轮用户回复的 effort(跳过无 [Cost] 标签的)。"""
    out = []
    for r in user_replies:
        e = parse_effort_from_cost(r)
        if e is not None:
            out.append(e)
    return out
