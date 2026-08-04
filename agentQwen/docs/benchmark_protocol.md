# Benchmark Protocol

## Version pins

- BFCL: `bfcl-eval==2026.3.23`.
- TAU-2: official `sierra-research/tau2-bench` tag `v0.2.0`.
- Policy server: MLX-LM OpenAI-compatible HTTP server.
- Model variants: base Qwen3-8B and the local LoRA adapter in separate result roots.

## BFCL-V4 Multi-Turn

Categories:

1. `multi_turn_base`
2. `multi_turn_miss_func`
3. `multi_turn_miss_param`
4. `multi_turn_long_context`

The smoke profile selects the first five official IDs from each JSONL file and uses `--partial-eval`. The paper profile omits `--run-ids` and evaluates all 800 tasks. Generation logs include the fully transformed input; evaluation uses the official state/execution checker.

## TAU-2

Domains: airline, retail, telecom.

The smoke profile runs five tasks/domain, one trial, maximum 200 steps, and local Qwen3-8B as user simulator. It tests interface correctness only.

The paper profile removes the task cap and sets four trials. It refuses to start unless `TAU2_USER_MODEL` and `TAU2_USER_API_BASE` are supplied. The user simulator choice, prompt, temperature, and model endpoint must be documented with the result because they can materially change Avg@4.

## Result acceptance

A benchmark result is accepted only when:

- the official evaluator exits successfully;
- every requested task ID has a result;
- the score files and inference logs are hashed;
- base and adapter use identical evaluator/simulator settings;
- the result manifest records versions, task count, trials, seed, and scope;
- smoke results are labeled non-comparable to the paper.

Dry-run manifests contain commands and task IDs, not scores.
