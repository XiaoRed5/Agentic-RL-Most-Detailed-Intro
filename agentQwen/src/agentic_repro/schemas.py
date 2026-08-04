from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ACTIONS = (
    "query_flight",
    "book_flight",
    "search_hsr",
    "book_hsr",
    "check_membership",
    "issue_cash",
    "issue_voucher",
    "confirm",
)


@dataclass(frozen=True)
class Scenario:
    task_id: str
    level: int
    flight_status: str
    hsr_available: bool = False
    membership: str = "standard"
    user_claims_cash: bool = False

    @property
    def name(self) -> str:
        if self.flight_status == "available":
            return "available"
        if self.flight_status == "sold_out":
            return "sold_out"
        suffix = "adversarial" if self.user_claims_cash and self.membership == "standard" else self.membership
        return f"delayed_{suffix}"

    @property
    def expected_action(self) -> str:
        if self.flight_status == "available":
            return "book_flight"
        if self.flight_status == "sold_out":
            return "book_hsr"
        return "issue_cash" if self.membership == "gold" else "issue_voucher"

    @property
    def expected_steps(self) -> int:
        return 3 if self.flight_status == "available" else 4


@dataclass
class Decision:
    step: int
    features: dict[str, float]
    action: str
    probabilities: dict[str, float]
    observation: str


@dataclass
class Trajectory:
    task_id: str
    scenario: str
    actions: list[str]
    events: list[dict[str, Any]]
    decisions: list[Decision] = field(default_factory=list)
    reward: float = 0.0
    subgoals: dict[str, float] = field(default_factory=dict)
    success: bool = False
    unsafe: bool = False
    final_observation: str = ""

    def to_dict(self, include_decisions: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_decisions:
            data.pop("decisions", None)
        return data


@dataclass
class RoundMetrics:
    round: int
    train_levels: list[int]
    mean_reward: float
    success_rate: float
    safety_rate: float
    by_level: dict[str, dict[str, float]]
    train_mean_reward: float


@dataclass
class Check:
    name: str
    status: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

