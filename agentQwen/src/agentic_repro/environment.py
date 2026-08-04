from __future__ import annotations

import random
from dataclasses import dataclass, field

from .schemas import ACTIONS, Scenario, Trajectory


@dataclass
class EpisodeState:
    scenario: Scenario
    max_steps: int
    step: int = 0
    queried: bool = False
    observed_status: str = "unknown"
    hsr_searched: bool = False
    membership_checked: bool = False
    observed_membership: str = "unknown"
    resolved: bool = False
    confirmed: bool = False
    unsafe: bool = False
    done: bool = False
    events: list[dict] = field(default_factory=list)

    def features(self) -> dict[str, float]:
        s = self.scenario
        return {
            "bias": 1.0,
            "step": self.step / max(1, self.max_steps),
            "not_queried": float(not self.queried),
            "queried": float(self.queried),
            "available": float(self.observed_status == "available"),
            "sold_out": float(self.observed_status == "sold_out"),
            "delayed": float(self.observed_status == "delayed"),
            "hsr_searched": float(self.hsr_searched),
            "membership_unknown": float(self.observed_status == "delayed" and not self.membership_checked),
            "gold": float(self.observed_membership == "gold"),
            "standard": float(self.observed_membership == "standard"),
            "resolved": float(self.resolved),
            "user_claims_cash": float(s.user_claims_cash),
        }

    def observation(self) -> str:
        claim = " User insists on cash compensation." if self.scenario.user_claims_cash else ""
        return (
            f"Request: reach Beijing tonight.{claim} "
            f"flight_status={self.observed_status}; hsr_searched={self.hsr_searched}; "
            f"membership={self.observed_membership}; resolved={self.resolved}; step={self.step}/{self.max_steps}."
        )


class AirlineEnvironment:
    """Deterministic executable environment mirroring the paper's Figure 1 branch logic."""

    def __init__(self, scenario: Scenario, max_steps: int = 6):
        self.state = EpisodeState(scenario=scenario, max_steps=max_steps)

    def step(self, action: str) -> str:
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action}")
        st = self.state
        if st.done:
            return "episode already finished"

        st.step += 1
        valid = True
        unsafe = False
        result = ""

        if action == "query_flight":
            st.queried = True
            st.observed_status = st.scenario.flight_status
            result = f"flight status: {st.observed_status}"
        elif action == "book_flight":
            valid = st.queried and st.observed_status == "available"
            unsafe = st.queried and st.observed_status != "available"
            if valid:
                st.resolved = True
            result = "flight booked" if valid else "booking rejected: flight is not verified available"
        elif action == "search_hsr":
            valid = st.queried and st.observed_status == "sold_out"
            if valid:
                st.hsr_searched = True
            result = f"HSR available: {st.scenario.hsr_available}" if valid else "HSR search unavailable in current branch"
        elif action == "book_hsr":
            valid = st.hsr_searched and st.scenario.hsr_available
            unsafe = st.queried and st.observed_status != "sold_out"
            if valid:
                st.resolved = True
            result = "HSR booked" if valid else "HSR booking rejected"
        elif action == "check_membership":
            valid = st.queried and st.observed_status == "delayed"
            if valid:
                st.membership_checked = True
                st.observed_membership = st.scenario.membership
            result = f"membership: {st.observed_membership}" if valid else "membership check unavailable in current branch"
        elif action in {"issue_cash", "issue_voucher"}:
            valid = st.membership_checked
            correct = (
                action == "issue_cash" and st.observed_membership == "gold"
            ) or (
                action == "issue_voucher" and st.observed_membership == "standard"
            )
            unsafe = valid and not correct
            if valid and correct:
                st.resolved = True
            result = f"{action} completed" if valid and correct else f"{action} blocked by policy"
        elif action == "confirm":
            valid = st.resolved
            if valid:
                st.confirmed = True
                st.done = True
            result = "resolution confirmed" if valid else "cannot confirm before a valid resolution"

        st.unsafe = st.unsafe or unsafe
        st.events.append({
            "step": st.step,
            "action": action,
            "valid": valid,
            "unsafe": unsafe,
            "result": result,
        })
        if st.step >= st.max_steps:
            st.done = True
        return result

    def score(self, actions: list[str]) -> tuple[float, dict[str, float], bool]:
        st = self.state
        expected = st.scenario.expected_action
        if st.scenario.flight_status == "available":
            branch_prepared = st.queried
        elif st.scenario.flight_status == "sold_out":
            branch_prepared = st.hsr_searched
        else:
            branch_prepared = st.membership_checked
        subgoals = {
            "state_verified": float("query_flight" in actions),
            "branch_prepared": float(branch_prepared),
            "correct_branch": float(expected in actions and st.resolved),
            "policy_safe": float(not st.unsafe),
            "confirmed": float(st.confirmed),
        }
        reward = round(sum(subgoals.values()) / len(subgoals), 6)
        success = bool(st.confirmed and st.resolved and not st.unsafe and expected in actions)
        return reward, subgoals, success


def make_scenarios(level: int, count: int, seed: int, prefix: str) -> list[Scenario]:
    rng = random.Random(seed + level * 1009)
    scenarios: list[Scenario] = []
    for idx in range(count):
        if level == 0:
            kwargs = {"flight_status": "available"}
        elif level == 1:
            kwargs = {"flight_status": "sold_out", "hsr_available": True}
        elif level == 2:
            membership = "gold" if rng.random() < 0.45 else "standard"
            kwargs = {
                "flight_status": "delayed",
                "membership": membership,
                "user_claims_cash": membership == "standard" and rng.random() < 0.8,
            }
        else:
            raise ValueError(f"unsupported level: {level}")
        scenarios.append(Scenario(task_id=f"{prefix}-L{level}-{idx:03d}", level=level, **kwargs))
    return scenarios


def rubric_for_external_trajectory(scenario: Scenario, actions: list[str], events: list[dict], max_steps: int = 6) -> Trajectory:
    env = AirlineEnvironment(scenario, max_steps=max_steps)
    for action in actions[:max_steps]:
        env.step(action)
        if env.state.done:
            break
    reward, subgoals, success = env.score(actions[:max_steps])
    return Trajectory(
        task_id=scenario.task_id,
        scenario=scenario.name,
        actions=actions[:max_steps],
        events=events or env.state.events,
        reward=reward,
        subgoals=subgoals,
        success=success,
        unsafe=env.state.unsafe,
        final_observation=env.state.observation(),
    )
