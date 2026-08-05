# Industrial AgenticQwen pipeline

`run_industrial_agenticqwen.sh` is the control plane for the official
`haruhi-sudo/data_synth_and_rl` implementation. It does not reimplement the
official task generator or replace verl with a local toy loop. The stages are:

```text
preflight
  → official data_gen (tool set / policy / task)
  → official solve (teacher policy ↔ mock user ↔ mock tools)
  → official rubric judge
  → PASS-only filtering
  → official verl parquet conversion
  → official verl + SGLang multi-turn GRPO
  → independent BFCL/TAU-2 evaluation
```

Every stage writes `artifacts/industrial_agenticqwen/stages/<stage>.json` and
an append-only log. A completed stage is skipped only when its declared output
paths still exist. Commands, upstream commit, provider model, resource
contract, durations, return code, and bounded output hashes are persisted. API
keys are passed through the environment but never written to the manifest or
log.

## Teacher substitution

The paper uses Qwen3-235B-A22B as the strong model for task synthesis,
simulated user/tool behavior, and rubric judging. The configured industrial
profile uses `deepseek-v4-flash-sa-256k` through the same OpenAI-compatible
endpoint because a single H200 cannot host the paper teacher. This is an
explicit substitution, not an equivalence claim. Set `AIGC_APP_ID` (or the
remote credential file) only in the runtime environment.

For bounded cloud debugging, the AgenticQwen controller uses a hard synthesis
budget (`max_synthetic_trajectories <= 10`; current profile: 4 seeds plus at
most one retained branch per round). Replay of an existing task does not spend
the budget. Candidate data is only materialized after the executable teacher
solve + branch-hit gate; an all-rejected round writes a rejection artifact and
stops as `PARTIAL`.

## Resource gates

- One H200 can run a bounded 8B policy profile and official smoke evaluation.
- The official `run_virtual_tool.sh` recipe is written for 8×H100 and requires
  `verl` plus SGLang; the controller reports missing packages instead of
  silently falling back to TRL or a one-token action task.
- Paper-scale claims require approximately 100K retained tasks, the exact
  Qwen3-235B teacher family, all paper prompts, and full benchmark profiles.

Dry-run the complete contract without spending API/GPU time:

```bash
STAGE=all ./run_industrial_agenticqwen.sh --dry-run
```

Run one stage or resume after an interruption:

```bash
STAGE=preflight ./run_industrial_agenticqwen.sh
STAGE=data_gen ./run_industrial_agenticqwen.sh
STAGE=solve ./run_industrial_agenticqwen.sh
```

The resulting manifests are the source of truth for the report. A successful
preflight or dry-run is never counted as a training or benchmark result.
