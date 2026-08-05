from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .paper_flywheel_schema import AgenticTask, RubricItem, ToolCallSpec, TrajectoryRecord


_TRACE_LOCK = threading.Lock()


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _is_subsequence(expected: tuple[str, ...], observed: tuple[str, ...]) -> bool:
    if not expected:
        return True
    cursor = 0
    for action in observed:
        if action == expected[cursor]:
            cursor += 1
            if cursor == len(expected):
                return True
    return False


@dataclass
class FlightWorldState:
    flight_queried: bool = False
    hsr_searched: bool = False
    nearby_airport_searched: bool = False
    membership_checked: bool = False
    booked_transport: str | None = None
    compensation: str | None = None
    confirmed: bool = False
    user_turns: int = 0
    unsafe_action_count: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)


class PaperFlightEnvironment:
    """Executable micro-environment for AgenticQwen's behavior-tree flywheel.

    The environment exposes fixed public methods so TRL's stateful
    ``environment_factory`` can advertise them as tools.  Task-specific world
    state, user pressure, normal/hack paths, and rubrics arrive through
    ``task_json`` at reset time.
    """

    def __init__(self) -> None:
        self.task: AgenticTask | None = None
        self.state = FlightWorldState()
        self.stage = "unknown"
        self._reward_written = False

    def reset(self, task_json: str, stage: str = "unknown", **_: Any) -> None:
        self.task = AgenticTask.from_dict(json.loads(task_json))
        self.state = FlightWorldState()
        self.stage = str(stage)
        self._reward_written = False
        return None

    def get_reward(self) -> float:
        trajectory = self._trajectory()
        if not self._reward_written:
            path = os.getenv("AGENTICQWEN_PAPER_TRACE_FILE")
            if path:
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                row = {
                    "schema_version": 1,
                    "stage": self.stage,
                    "tree_revision": self._task().tree_revision,
                    "source_branch_id": self._task().source_branch_id,
                    **trajectory.to_dict(),
                }
                serialized = json.dumps(row, ensure_ascii=False, sort_keys=True)
                with _TRACE_LOCK:
                    with target.open("a", encoding="utf-8") as handle:
                        handle.write(serialized + "\n")
            self._reward_written = True
        return trajectory.reward

    def ask_user(self, message: str) -> str:
        """Continue the mock-user conversation and surface adversarial pressure.

        Args:
            message: Short question or confirmation request from the agent.

        Returns:
            The next simulated user response.
        """
        task = self._task()
        self.state.user_turns += 1
        if self.state.user_turns == 1 and task.user_input.adversarial_strategy:
            response = task.user_input.adversarial_strategy
        elif task.user_input.hack_success_followup:
            response = task.user_input.hack_success_followup
        else:
            response = "Please follow the applicable policy and use verified system data."
        return self._record("ask_user", True, {"message": message, "user_response": response})

    def query_flight(self, destination: str) -> str:
        """Query flight inventory for a destination.

        Args:
            destination: City the user wants to reach.

        Returns:
            Inventory status and the destination that was queried.
        """
        task = self._task()
        self.state.flight_queried = True
        return self._record(
            "query_flight",
            True,
            {
                "destination": destination,
                "status": task.environment_state["flight_status"],
            },
        )

    def book_flight(self, destination: str) -> str:
        """Book an available direct flight after inventory was queried.

        Args:
            destination: City to book from the verified inventory result.

        Returns:
            Whether the booking was accepted by the environment.
        """
        task = self._task()
        allowed = self.state.flight_queried and task.environment_state["flight_status"] == "available"
        if allowed:
            self.state.booked_transport = "flight"
        else:
            self.state.unsafe_action_count += 1
        return self._record(
            "book_flight",
            allowed,
            {"destination": destination, "booked": allowed, "error": None if allowed else "FLIGHT_NOT_AVAILABLE"},
        )

    def search_hsr(self, destination: str) -> str:
        """Search high-speed rail when the flight branch is sold out.

        Args:
            destination: City for the alternative rail search.

        Returns:
            Rail availability for the destination.
        """
        task = self._task()
        allowed = self.state.flight_queried and task.environment_state["flight_status"] == "sold_out"
        if allowed:
            self.state.hsr_searched = True
        return self._record(
            "search_hsr",
            allowed,
            {
                "destination": destination,
                "available": bool(task.environment_state.get("hsr_available", False)) if allowed else False,
            },
        )

    def book_hsr(self, destination: str) -> str:
        """Book a rail alternative after a successful rail search.

        Args:
            destination: City to book on the verified rail result.

        Returns:
            Whether the rail booking was accepted.
        """
        task = self._task()
        allowed = self.state.hsr_searched and bool(task.environment_state.get("hsr_available", False))
        if allowed:
            self.state.booked_transport = "hsr"
        else:
            self.state.unsafe_action_count += 1
        return self._record(
            "book_hsr",
            allowed,
            {"destination": destination, "booked": allowed, "error": None if allowed else "HSR_NOT_AVAILABLE"},
        )

    def search_nearby_airport(self, destination: str) -> str:
        """Search nearby-airport alternatives after the primary flight sells out.

        Args:
            destination: City for the nearby-airport search.

        Returns:
            Nearby airport and alternative-flight availability.
        """
        task = self._task()
        allowed = self.state.flight_queried and task.environment_state["flight_status"] == "sold_out"
        if allowed:
            self.state.nearby_airport_searched = True
        return self._record(
            "search_nearby_airport",
            allowed,
            {
                "destination": destination,
                "airport": task.environment_state.get("nearby_airport", "TSN") if allowed else None,
                "available": bool(task.environment_state.get("nearby_airport_available", False)) if allowed else False,
            },
        )

    def book_nearby_flight(self, airport: str) -> str:
        """Book an alternative flight found through a nearby-airport search.

        Args:
            airport: Airport code returned by the nearby-airport search.

        Returns:
            Whether the alternative booking was accepted.
        """
        task = self._task()
        allowed = self.state.nearby_airport_searched and bool(
            task.environment_state.get("nearby_airport_available", False)
        )
        if allowed:
            self.state.booked_transport = "nearby_flight"
        else:
            self.state.unsafe_action_count += 1
        return self._record(
            "book_nearby_flight",
            allowed,
            {"airport": airport, "booked": allowed, "error": None if allowed else "ALT_FLIGHT_NOT_AVAILABLE"},
        )

    def check_membership(self, user_id: str) -> str:
        """Check membership before selecting the delayed-flight remedy.

        Args:
            user_id: User identifier associated with the delayed booking.

        Returns:
            Verified membership tier when the delay branch is active.
        """
        task = self._task()
        allowed = self.state.flight_queried and task.environment_state["flight_status"] == "delayed"
        if allowed:
            self.state.membership_checked = True
        return self._record(
            "check_membership",
            allowed,
            {
                "user_id": user_id,
                "membership": task.environment_state.get("membership", "standard") if allowed else "unknown",
            },
        )

    def issue_cash(self, amount: float) -> str:
        """Issue cash compensation only for an eligible gold member.

        Args:
            amount: Compensation amount requested by the policy branch.

        Returns:
            Whether cash compensation was accepted.
        """
        task = self._task()
        allowed = self.state.membership_checked and task.environment_state.get("membership") == "gold"
        if allowed:
            self.state.compensation = "cash"
        else:
            self.state.unsafe_action_count += 1
        return self._record(
            "issue_cash",
            allowed,
            {"amount": float(amount), "issued": allowed, "error": None if allowed else "CASH_NOT_ALLOWED"},
        )

    def issue_voucher(self, amount: float) -> str:
        """Issue a voucher only for an eligible standard member.

        Args:
            amount: Voucher amount requested by the policy branch.

        Returns:
            Whether the voucher was accepted.
        """
        task = self._task()
        allowed = self.state.membership_checked and task.environment_state.get("membership") == "standard"
        if allowed:
            self.state.compensation = "voucher"
        else:
            self.state.unsafe_action_count += 1
        return self._record(
            "issue_voucher",
            allowed,
            {"amount": float(amount), "issued": allowed, "error": None if allowed else "VOUCHER_NOT_ALLOWED"},
        )

    def confirm(self) -> str:
        """Confirm the valid booking or compensation outcome.

        Returns:
            Whether a valid outcome was ready for confirmation.
        """
        ready = self.state.booked_transport is not None or self.state.compensation is not None
        if ready:
            self.state.confirmed = True
        return self._record("confirm", ready, {"confirmed": ready})

    def _execute(self, call: ToolCallSpec | dict[str, Any]) -> dict[str, Any]:
        spec = call if isinstance(call, ToolCallSpec) else ToolCallSpec.from_dict(call)
        methods: dict[str, Callable[..., str]] = {
            "ask_user": self.ask_user,
            "query_flight": self.query_flight,
            "book_flight": self.book_flight,
            "search_hsr": self.search_hsr,
            "book_hsr": self.book_hsr,
            "search_nearby_airport": self.search_nearby_airport,
            "book_nearby_flight": self.book_nearby_flight,
            "check_membership": self.check_membership,
            "issue_cash": self.issue_cash,
            "issue_voucher": self.issue_voucher,
            "confirm": self.confirm,
        }
        method = methods.get(spec.tool_name)
        if method is None:
            raise ValueError(f"Unknown paper flywheel tool: {spec.tool_name}")
        return json.loads(method(**spec.arguments))

    def _trajectory(self) -> TrajectoryRecord:
        task = self._task()
        observed = tuple(event["tool"] for event in self.state.events if event["tool"] != "ask_user")
        successful = tuple(
            event["tool"]
            for event in self.state.events
            if event["tool"] != "ask_user" and event["ok"]
        )
        # User turns are part of the complete normal trajectory but are not
        # behavior-tree action nodes.  Branch matching therefore compares only
        # environment/tool actions while a separate rubric checks user contact.
        expected_branch_actions = tuple(
            action for action in task.normal_actions if action != "ask_user"
        )
        branch_hit = _is_subsequence(expected_branch_actions, successful)
        scores = {
            item.rubric_id: self._score_rubric(item, observed, successful, branch_hit)
            for item in task.rubric
        }
        total_weight = sum(item.weight for item in task.rubric)
        reward = sum(item.weight * scores[item.rubric_id] for item in task.rubric) / total_weight
        reward = max(0.0, min(1.0, reward))
        success = reward >= task.pass_threshold and branch_hit
        return TrajectoryRecord(
            task_id=task.task_id,
            actions=observed,
            events=tuple(dict(item) for item in self.state.events),
            final_state=self._final_state(),
            reward=round(reward, 6),
            rubric_scores=scores,
            intended_branch_hit=branch_hit,
            success=success,
        )

    def _final_state(self) -> dict[str, Any]:
        return {
            "flight_queried": self.state.flight_queried,
            "hsr_searched": self.state.hsr_searched,
            "nearby_airport_searched": self.state.nearby_airport_searched,
            "membership_checked": self.state.membership_checked,
            "booked_transport": self.state.booked_transport,
            "compensation": self.state.compensation,
            "confirmed": self.state.confirmed,
            "unsafe_action_count": self.state.unsafe_action_count,
            "user_turns": self.state.user_turns,
        }

    def _score_rubric(
        self,
        item: RubricItem,
        observed: tuple[str, ...],
        successful: tuple[str, ...],
        branch_hit: bool,
    ) -> float:
        if item.check == "required_action":
            return float(item.target in successful)
        if item.check == "forbidden_action":
            return float(item.target not in observed)
        if item.check == "state_equals":
            return float(self._final_state().get(item.target) == item.expected)
        if item.check == "branch_hit":
            return float(branch_hit == bool(item.expected))
        if item.check == "no_unsafe_action":
            return float(self.state.unsafe_action_count == 0)
        if item.check == "user_turns_at_least":
            return float(self.state.user_turns >= int(item.expected))
        raise ValueError(f"Unknown rubric check: {item.check}")

    def _task(self) -> AgenticTask:
        if self.task is None:
            raise RuntimeError("Environment must be reset before use")
        return self.task

    def _record(self, tool: str, ok: bool, payload: dict[str, Any]) -> str:
        event = {
            "turn": len(self.state.events) + 1,
            "tool": tool,
            "ok": bool(ok),
            "payload": payload,
        }
        self.state.events.append(event)
        return _json({"ok": bool(ok), **payload})


def execute_path(task: AgenticTask, path: tuple[ToolCallSpec, ...]) -> TrajectoryRecord:
    env = PaperFlightEnvironment()
    env.reset(json.dumps(task.to_dict(), ensure_ascii=False), stage="teacher_validation")
    for call in path:
        env._execute(call)
    return env._trajectory()
