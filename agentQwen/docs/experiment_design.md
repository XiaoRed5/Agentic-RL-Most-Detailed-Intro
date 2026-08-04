# Experiment Design

## Research question

Can a small Qwen3-8B policy learn robust, long-horizon tool use from agentic RL without collapsing under binary group rewards and response-length dilution?

## Hypotheses

| ID | Hypothesis | Metric | Falsification condition |
|---|---|---|---|
| H1 | Vanilla GRPO suffers group saturation | zero-variance group rate | rate remains low across seeds/checkpoints |
| H2 | Turn-Discount protects early planning | median tokens/turn, late tool error | no improvement over Vanilla |
| H3 | PRM-Lite creates local learning signal | zero-variance rate, process-rule coverage | no additional active groups |
| H4 | LATA preserves long-response credit | generalization pass@1, reasoning length | no improvement over Turn-Discount/Vanilla |
| H5 | Joint beats each single component | generalization pass@1 | Joint ≤ max(PRM-Lite, LATA) |

## Fixed matrix

- Experiments: Vanilla, Turn-Discount, PRM-Lite, LATA, Joint.
- Seeds: 42, 43, 44.
- Checkpoints: 50, 100, 150, 200, 250, 300.
- Primary metric: generalization pass@1.
- Secondary metrics: overall pass@1, zero-variance rate, tool-error rate, median reasoning tokens/turn.
- Evaluation: independent process, frozen task split, identical decoding and simulator settings.

## Leakage controls

1. Persist task IDs before training.
2. Separate covered-seen, uncovered-seen, and unseen.
3. Never select checkpoints on unseen performance.
4. Hash prompts, datasets, adapters, and evaluator outputs.
5. Report every seed; do not keep only the best run.

## Current executability boundary

The PRM-Lite and credit-assignment kernels are code-complete and unit-tested. The matrix is intentionally not executed on the one-token action task because `L=1` makes Linear and LATA identical. A valid run requires multi-turn response-token masks.
