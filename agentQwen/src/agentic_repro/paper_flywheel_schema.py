from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallSpec:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    expected_output: dict[str, Any] | str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ToolCallSpec":
        return cls(
            tool_name=str(value["tool_name"]),
            arguments=dict(value.get("arguments", {})),
            expected_output=value.get("expected_output"),
        )


@dataclass(frozen=True)
class BehaviorNode:
    node_id: str
    action: str
    description: str
    terminal: bool = False


@dataclass(frozen=True)
class BehaviorEdge:
    source: str
    target: str
    condition: dict[str, Any] = field(default_factory=dict)
    label: str = "always"


@dataclass(frozen=True)
class BehaviorBranch:
    branch_id: str
    node_ids: tuple[str, ...]
    actions: tuple[str, ...]
    conditions: dict[str, Any]
    depth: int


@dataclass(frozen=True)
class BehaviorTree:
    domain: str
    revision: int
    root_id: str
    nodes: tuple[BehaviorNode, ...]
    edges: tuple[BehaviorEdge, ...]
    source_workflow: tuple[str, ...]
    provenance: str

    def validate(self) -> None:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Behavior tree node IDs must be unique")
        if self.root_id not in set(node_ids):
            raise ValueError(f"Behavior tree root is missing: {self.root_id}")
        node_set = set(node_ids)
        for edge in self.edges:
            if edge.source not in node_set or edge.target not in node_set:
                raise ValueError(f"Behavior edge references an unknown node: {edge}")
        incoming = {node_id: 0 for node_id in node_ids}
        for edge in self.edges:
            incoming[edge.target] += 1
        if incoming[self.root_id] != 0:
            raise ValueError("Behavior tree root must not have an incoming edge")
        if any(count > 1 for node_id, count in incoming.items() if node_id != self.root_id):
            raise ValueError("Behavior tree nodes must have at most one parent")
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        adjacency = self.adjacency()
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError(f"Behavior tree contains a cycle at {node_id}")
            if node_id in visited:
                return
            visiting.add(node_id)
            for edge in adjacency.get(node_id, ()):
                visit(edge.target)
            visiting.remove(node_id)
            visited.add(node_id)

        visit(self.root_id)

    def adjacency(self) -> dict[str, tuple[BehaviorEdge, ...]]:
        values: dict[str, list[BehaviorEdge]] = {node.node_id: [] for node in self.nodes}
        for edge in self.edges:
            values[edge.source].append(edge)
        return {
            node_id: tuple(sorted(edges, key=lambda item: (item.label, item.target)))
            for node_id, edges in values.items()
        }

    def branches(self) -> tuple[BehaviorBranch, ...]:
        self.validate()
        nodes = {node.node_id: node for node in self.nodes}
        adjacency = self.adjacency()
        values: list[BehaviorBranch] = []

        def walk(
            node_id: str,
            path: tuple[str, ...],
            conditions: dict[str, Any],
        ) -> None:
            next_path = (*path, node_id)
            children = adjacency.get(node_id, ())
            if not children:
                actions = tuple(nodes[item].action for item in next_path)
                values.append(
                    BehaviorBranch(
                        branch_id="branch:" + ">".join(next_path),
                        node_ids=next_path,
                        actions=actions,
                        conditions=dict(conditions),
                        depth=len(next_path),
                    )
                )
                return
            for edge in children:
                merged = dict(conditions)
                for key, expected in edge.condition.items():
                    if key in merged and merged[key] != expected:
                        raise ValueError(
                            f"Conflicting condition for {key}: {merged[key]} vs {expected}"
                        )
                    merged[key] = expected
                walk(edge.target, next_path, merged)

        walk(self.root_id, (), {})
        return tuple(sorted(values, key=lambda item: item.branch_id))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BehaviorTree":
        tree = cls(
            domain=str(value["domain"]),
            revision=int(value["revision"]),
            root_id=str(value["root_id"]),
            nodes=tuple(BehaviorNode(**item) for item in value["nodes"]),
            edges=tuple(
                BehaviorEdge(
                    source=item["source"],
                    target=item["target"],
                    condition=dict(item.get("condition", {})),
                    label=item.get("label", "always"),
                )
                for item in value["edges"]
            ),
            source_workflow=tuple(value.get("source_workflow", ())),
            provenance=str(value.get("provenance", "unknown")),
        )
        tree.validate()
        return tree


@dataclass(frozen=True)
class UserInput:
    request: str
    background: str
    adversarial_strategy: str
    hack_success_followup: str = ""


@dataclass(frozen=True)
class RubricItem:
    rubric_id: str
    description: str
    check: str
    target: str
    expected: Any
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("Rubric weights must be positive")


@dataclass(frozen=True)
class AgenticTask:
    task_id: str
    round_index: int
    tree_revision: int
    source_branch_id: str
    environment_state: dict[str, Any]
    user_input: UserInput
    agent_instruction: str
    available_tools: tuple[ToolSpec, ...]
    normal_path: tuple[ToolCallSpec, ...]
    hack_path: tuple[ToolCallSpec, ...]
    rubric: tuple[RubricItem, ...]
    pass_threshold: float = 1.0
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be within [0, 1]")
        if not self.normal_path:
            raise ValueError("Agentic task requires a non-empty normal_path")
        if not self.rubric:
            raise ValueError("Agentic task requires a non-empty rubric")

    @property
    def normal_actions(self) -> tuple[str, ...]:
        return tuple(item.tool_name for item in self.normal_path)

    @property
    def hack_actions(self) -> tuple[str, ...]:
        return tuple(item.tool_name for item in self.hack_path)

    def training_streams(self) -> dict[str, Any]:
        """Return the three aligned inputs described in AgenticQwen Appendix B/D."""
        return {
            "environment_input": {
                "state": self.environment_state,
                "available_tools": [asdict(item) for item in self.available_tools],
            },
            "user_input": asdict(self.user_input),
            "agent_input": {"instruction": self.agent_instruction},
        }

    def official_compatibility_fields(self) -> dict[str, Any]:
        """Map the typed task to the public AgenticQwen repository fields.

        The released repo serializes these values into verl ``extra_info``.
        Keeping the mapping explicit lets this micro implementation use its
        stricter schema without inventing a different data contract.
        """
        return {
            "test_policy": self.agent_instruction,
            "task_background": self.user_input.background,
            "user_escape_strategy": self.user_input.adversarial_strategy,
            "hack_success_user_background": self.user_input.hack_success_followup,
            "tool_return_expected": {
                "normal_path": [asdict(item) for item in self.normal_path],
                "hack_path": [asdict(item) for item in self.hack_path],
            },
            "rubrics": "\n".join(
                f"- [{item.rubric_id}] {item.description} "
                f"(check={item.check}, target={item.target}, expected={item.expected!r}, weight={item.weight})"
                for item in self.rubric
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["training_streams"] = self.training_streams()
        value["agenticqwen_official_compat"] = self.official_compatibility_fields()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgenticTask":
        return cls(
            task_id=str(value["task_id"]),
            round_index=int(value["round_index"]),
            tree_revision=int(value["tree_revision"]),
            source_branch_id=str(value["source_branch_id"]),
            environment_state=dict(value["environment_state"]),
            user_input=UserInput(**value["user_input"]),
            agent_instruction=str(value["agent_instruction"]),
            available_tools=tuple(ToolSpec(**item) for item in value["available_tools"]),
            normal_path=tuple(ToolCallSpec.from_dict(item) for item in value["normal_path"]),
            hack_path=tuple(ToolCallSpec.from_dict(item) for item in value.get("hack_path", ())),
            rubric=tuple(RubricItem(**item) for item in value["rubric"]),
            pass_threshold=float(value.get("pass_threshold", 1.0)),
            provenance=dict(value.get("provenance", {})),
        )


@dataclass(frozen=True)
class TrajectoryRecord:
    task_id: str
    actions: tuple[str, ...]
    events: tuple[dict[str, Any], ...]
    final_state: dict[str, Any]
    reward: float
    rubric_scores: dict[str, float]
    intended_branch_hit: bool
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrajectoryRecord":
        return cls(
            task_id=str(value["task_id"]),
            actions=tuple(str(item) for item in value.get("actions", ())),
            events=tuple(dict(item) for item in value.get("events", ())),
            final_state=dict(value.get("final_state", {})),
            reward=float(value["reward"]),
            rubric_scores={str(key): float(score) for key, score in value.get("rubric_scores", {}).items()},
            intended_branch_hit=bool(value["intended_branch_hit"]),
            success=bool(value["success"]),
        )


@dataclass(frozen=True)
class CandidateValidation:
    task_id: str
    teacher_model: str
    solved: bool
    intended_branch_hit: bool
    reward: float
    retained: bool
    reason: str
    trajectory: TrajectoryRecord

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateValidation":
        return cls(
            task_id=str(value["task_id"]),
            teacher_model=str(value["teacher_model"]),
            solved=bool(value["solved"]),
            intended_branch_hit=bool(value["intended_branch_hit"]),
            reward=float(value["reward"]),
            retained=bool(value["retained"]),
            reason=str(value["reason"]),
            trajectory=TrajectoryRecord.from_dict(value["trajectory"]),
        )


@dataclass(frozen=True)
class FlywheelRound:
    round_index: int
    input_tree_revision: int
    output_tree_revision: int
    rollout_count: int
    candidate_count: int
    retained_count: int
    rejected_count: int
    tasks: tuple[AgenticTask, ...]
    validations: tuple[CandidateValidation, ...]
    tree: BehaviorTree

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_unique_task_ids(tasks: Iterable[AgenticTask]) -> None:
    values = [task.task_id for task in tasks]
    if len(values) != len(set(values)):
        raise ValueError("Agentic task IDs must be unique")
