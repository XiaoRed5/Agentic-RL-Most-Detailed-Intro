# Claims and Evidence Ledger

| Claim | Status | Evidence | Missing evidence |
|---|---|---|---|
| Standard Qwen3-8B weights were loaded on a cloud GPU | Supported | 15-file/16.397 GB SHA-256 manifest; NVIDIA H200 NF4+LoRA smoke | none for this run |
| Stage 1 changed trainable parameters | Supported | before/after parameter hashes differ; 12 steps; loss 0.022327 | none |
| Stage 2 consumed the Stage-1 adapter | Supported | Stage-2 input adapter hash equals Stage-1 weights hash | none |
| Stage 2 changed trainable parameters | Supported | before/after parameter hashes differ; Stage-2 adapter differs from Stage 1; 12 steps; loss -0.030853 | none |
| GRPO received non-zero relative signal | Supported | Stage-1 reward std 0.6144 / 20 unique; Stage-2 0.3297 / 36 unique | group-level saturation histogram for a deeper analysis |
| V1 saturation failure was detected, not hidden | Supported | 48/48 identical 1.3 rewards, zero policy loss, identical adapter hashes, verifier failure | none |
| The curriculum data handoff is auditable | Supported | frozen source trace SHA-256; frontier fallback; 8 hard + 4 replay task manifest | more data/seeds |
| Frozen Stage-1 probe improved | Supported, tiny N | 3/4 → 4/4 | larger probe and multiple seeds |
| Untouched Stage-2 holdout improved | Not supported | 6/6 → 6/6; saturated before Stage 2 | larger holdout and multiple seeds |
| Saved Stage-2 adapter reloads independently | Supported | fresh child process, 6/6 success, mean reward 1.252504 | none for the six-task replay |
| BFCL-V4 multi-turn smoke pipeline | Observed / PASS | official `bfcl-eval==2026.3.23`; 4 selected episodes per base/adapter run; result JSON + score CSV | This proves the official generation/checking path ran; observed `Overall Acc` and `Multi Turn Acc` are both 0.00%, so it is not a quality-improvement claim |
| TAU-2 Avg@4 reproduced | Not tested | evaluator code and dry-run only | all tasks × 4 with fixed simulator |
| AgenticQwen-8B paper average 47.4 reproduced | Not tested | paper table only | official benchmark result |
| Paper-style Round 0–3 data flywheel reproduced | Partial | bounded DS-v4-flash continuation is hard-capped at ≤10 synthesized tasks; Round 0 policy training completed, teacher audit contains 502/read-timeout and branch-hit rejections | stable teacher endpoint, complete Round 1–3, and 235B-equivalent teacher |
| Synthesis budget gate | Supported | config `max_synthetic_trajectories=10`; current cloud profile allows at most one new branch per round; replay is excluded; 45 local tests pass | a completed bounded run with at least one retained new branch |

Status changes require a new artifact path and hash; editing prose alone is insufficient.
