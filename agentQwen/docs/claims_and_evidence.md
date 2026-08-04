# Claims and Evidence Ledger

| Claim | Status | Evidence | Missing evidence |
|---|---|---|---|
| Qwen3-8B weights were loaded | Supported | parameter count, model bytes, SHA-256 | none |
| RL changed trainable parameters | Supported | LoRA-B norm and adapter hash changed | none |
| Saved checkpoint reproduces results | Supported | fresh-process 9/9 verifier | none |
| Same-task prompt wording improved | Supported, small N | 25.0% → 41.7% on 12 tasks | more tasks/seeds |
| Fully unseen tasks improved | Not supported | 25.0% → 25.0% on 4 tasks | larger unseen split |
| PRM-Lite breaks saturation | Not supported here | offline active groups 4 → 4 | online multi-turn training |
| BFCL-V4 result reproduced | Not tested | evaluator code and dry-run only | 800-task output/score logs |
| TAU-2 Avg@4 reproduced | Not tested | evaluator code and dry-run only | all tasks × 4 with fixed simulator |
| AgenticQwen-8B paper average 47.4 reproduced | Not tested | paper table only | official benchmark result |
| Round 0–3 data flywheel reproduced | Not tested | 12-task single round only | ≈100K data and 235B pipeline |

Status changes require a new artifact path and hash; editing prose alone is insufficient.
