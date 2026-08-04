from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .schemas import ACTIONS


class LinearSoftmaxPolicy:
    """Small transparent policy used only for the structural smoke reproduction."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.weights: dict[str, dict[str, float]] = {a: {} for a in ACTIONS}
        for action in ACTIONS:
            self.weights[action]["bias"] = self.rng.uniform(-0.03, 0.03)
        # The paper starts agentic RL from open-source tool-use data.  These two
        # priors reproduce only that generic grammar: inspect before acting and
        # confirm after a valid state change.  Branch selection is still learned.
        self.weights["query_flight"]["bias"] = 0.0
        self.weights["query_flight"]["not_queried"] = 1.2
        self.weights["query_flight"]["queried"] = -1.2
        self.weights["confirm"]["bias"] = -0.2
        self.weights["confirm"]["resolved"] = 1.5

    def logits(self, features: dict[str, float]) -> dict[str, float]:
        return {
            action: sum(self.weights[action].get(name, 0.0) * value for name, value in features.items())
            for action in ACTIONS
        }

    def probabilities(self, features: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
        logits = self.logits(features)
        top = max(logits.values())
        exps = {a: math.exp((v - top) / max(temperature, 1e-6)) for a, v in logits.items()}
        total = sum(exps.values())
        return {a: exps[a] / total for a in ACTIONS}

    def act(self, features: dict[str, float], sample: bool = True) -> tuple[str, dict[str, float]]:
        probs = self.probabilities(features)
        if not sample:
            return max(ACTIONS, key=lambda a: (probs[a], -ACTIONS.index(a))), probs
        value = self.rng.random()
        cumulative = 0.0
        for action in ACTIONS:
            cumulative += probs[action]
            if value <= cumulative:
                return action, probs
        return ACTIONS[-1], probs

    def grpo_update(self, decisions, advantage: float, learning_rate: float, clip: float = 2.5) -> None:
        advantage = max(-clip, min(clip, advantage))
        for decision in decisions:
            for candidate in ACTIONS:
                coeff = (1.0 if candidate == decision.action else 0.0) - decision.probabilities[candidate]
                action_weights = self.weights[candidate]
                for name, value in decision.features.items():
                    action_weights[name] = action_weights.get(name, 0.0) + learning_rate * advantage * coeff * value

    def to_dict(self) -> dict:
        return {"type": "linear-softmax-smoke-policy", "actions": list(ACTIONS), "weights": self.weights}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
