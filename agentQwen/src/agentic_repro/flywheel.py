from __future__ import annotations

import math
import random
from collections import defaultdict

from .environment import AirlineEnvironment, make_scenarios
from .policy import LinearSoftmaxPolicy
from .schemas import Decision, RoundMetrics, Scenario, Trajectory


def rollout(policy: LinearSoftmaxPolicy, scenario: Scenario, max_steps: int, sample: bool) -> Trajectory:
    env = AirlineEnvironment(scenario, max_steps=max_steps)
    decisions: list[Decision] = []
    actions: list[str] = []
    while not env.state.done:
        features = env.state.features()
        observation = env.state.observation()
        action, probs = policy.act(features, sample=sample)
        decisions.append(Decision(
            step=env.state.step + 1,
            features=features,
            action=action,
            probabilities=probs,
            observation=observation,
        ))
        actions.append(action)
        env.step(action)
    reward, subgoals, success = env.score(actions)
    return Trajectory(
        task_id=scenario.task_id,
        scenario=scenario.name,
        actions=actions,
        events=env.state.events,
        decisions=decisions,
        reward=reward,
        subgoals=subgoals,
        success=success,
        unsafe=env.state.unsafe,
        final_observation=env.state.observation(),
    )


def evaluate(policy: LinearSoftmaxPolicy, scenarios: list[Scenario], max_steps: int) -> dict:
    trajectories = [rollout(policy, task, max_steps=max_steps, sample=False) for task in scenarios]
    by_level: dict[int, list[Trajectory]] = defaultdict(list)
    for trajectory, scenario in zip(trajectories, scenarios):
        by_level[scenario.level].append(trajectory)

    def stats(items: list[Trajectory]) -> dict[str, float]:
        if not items:
            return {"mean_reward": 0.0, "success_rate": 0.0, "safety_rate": 0.0, "mean_steps": 0.0}
        n = len(items)
        return {
            "mean_reward": round(sum(t.reward for t in items) / n, 6),
            "success_rate": round(sum(t.success for t in items) / n, 6),
            "safety_rate": round(sum(not t.unsafe for t in items) / n, 6),
            "mean_steps": round(sum(len(t.actions) for t in items) / n, 6),
        }

    return {
        "overall": stats(trajectories),
        "by_level": {str(level): stats(items) for level, items in sorted(by_level.items())},
        "trajectories": trajectories,
    }


def train_flywheel(config: dict) -> tuple[LinearSoftmaxPolicy, dict]:
    run = config["run"]
    seed = int(run["seed"])
    rng = random.Random(seed + 31)
    rounds = int(run["rounds"])
    max_steps = int(run["max_steps"])
    group_size = int(run["group_size"])
    updates = int(run["updates_per_round"])
    lr = float(run["learning_rate"])
    train_count = int(run["train_tasks_per_round"])
    eval_count = int(run["eval_tasks_per_level"])

    policy = LinearSoftmaxPolicy(seed)
    eval_scenarios = [
        scenario
        for level in range(rounds)
        for scenario in make_scenarios(level, eval_count, seed + 7000, "eval")
    ]
    baseline = evaluate(policy, eval_scenarios, max_steps)
    round_metrics: list[RoundMetrics] = []

    for round_idx in range(rounds):
        levels = list(range(round_idx + 1))
        per_level = max(1, math.ceil(train_count / len(levels)))
        tasks = [
            scenario
            for level in levels
            for scenario in make_scenarios(level, per_level, seed + round_idx * 10000, f"train-R{round_idx}")
        ]
        train_rewards: list[float] = []
        for _ in range(updates):
            task = tasks[rng.randrange(len(tasks))]
            group = [rollout(policy, task, max_steps=max_steps, sample=True) for _ in range(group_size)]
            rewards = [t.reward for t in group]
            mean = sum(rewards) / len(rewards)
            variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
            std = math.sqrt(variance)
            train_rewards.extend(rewards)
            if std < 1e-8:
                continue
            for trajectory in group:
                advantage = (trajectory.reward - mean) / (std + 1e-8)
                policy.grpo_update(trajectory.decisions, advantage, learning_rate=lr)

        evaluation = evaluate(policy, eval_scenarios, max_steps)
        overall = evaluation["overall"]
        round_metrics.append(RoundMetrics(
            round=round_idx + 1,
            train_levels=levels,
            mean_reward=overall["mean_reward"],
            success_rate=overall["success_rate"],
            safety_rate=overall["safety_rate"],
            by_level=evaluation["by_level"],
            train_mean_reward=round(sum(train_rewards) / max(1, len(train_rewards)), 6),
        ))

    final = evaluate(policy, eval_scenarios, max_steps)
    return policy, {
        "algorithm": "group-relative policy gradient smoke reproduction",
        "claim_level": "structural/algorithmic smoke test; not 8B RL training",
        "seed": seed,
        "baseline": baseline,
        "rounds": [metric.__dict__ for metric in round_metrics],
        "final": final,
        "eval_task_count": len(eval_scenarios),
    }

