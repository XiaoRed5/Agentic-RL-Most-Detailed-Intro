# Claims and Evidence Ledger

| Claim | Status | Evidence | Missing evidence |
|---|---|---|---|
| Standard Qwen3-8B weights were loaded on a cloud GPU | Supported | 16-file/16.40 GB snapshot manifest; RTX PRO 6000 runtime record | none for this run |
| Stage 1 changed trainable parameters | Supported | before/after parameter hashes differ; 12 steps; loss 0.029398 | none |
| Stage 2 consumed the Stage-1 adapter | Supported | Stage-2 input adapter hash equals Stage-1 weights hash | none |
| Stage 2 changed trainable parameters | Supported | before/after parameter hashes differ; Stage-2 adapter differs from Stage 1; 12 steps; loss -0.029555 | none |
| GRPO received non-zero relative signal | Supported | Stage-1 reward std 0.5268 / 27 unique; Stage-2 0.2673 / 33 unique | group-level saturation histogram for a deeper analysis |
| V1 saturation failure was detected, not hidden | Supported | 48/48 identical 1.3 rewards, zero policy loss, identical adapter hashes, verifier failure | none |
| The curriculum data handoff is auditable | Supported | frozen source trace SHA-256; frontier fallback; 8 hard + 4 replay task manifest | more data/seeds |
| Frozen Stage-1 probe improved | Supported, tiny N | 3/4 → 4/4 | larger probe and multiple seeds |
| Untouched Stage-2 holdout improved | Supported, tiny N | 4/6 → 6/6 | larger holdout and multiple seeds |
| Saved Stage-2 adapter reloads independently | Supported | fresh child process 6913, 6/6 success, mean reward 1.250451 | none for the six-task replay |
| BFCL-V4 result reproduced | Not tested | evaluator code and dry-run only | official result JSON and score CSV |
| TAU-2 Avg@4 reproduced | Not tested | evaluator code and dry-run only | all tasks × 4 with fixed simulator |
| AgenticQwen-8B paper average 47.4 reproduced | Not tested | paper table only | official benchmark result |
| Full Round 0–3 data flywheel reproduced | Not tested | one train→diagnose→synthesize→retrain micro cycle | ≈100K data and 235B pipeline |

Status changes require a new artifact path and hash; editing prose alone is insufficient.
