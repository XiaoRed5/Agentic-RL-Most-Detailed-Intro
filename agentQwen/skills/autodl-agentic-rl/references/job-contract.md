# AutoDL job contract

## State machine

| State | Required evidence |
|---|---|
| `PLANNED` | entry command, model, GPU/VRAM, disk, outputs, budget |
| `PREPARED` | payload + launch/status/collect scripts + local manifest/hash |
| `RENTED` | approved instance running; GPU/price/disk recorded |
| `UPLOADED` | remote file sizes and payload hash match |
| `BOOTSTRAPPING` | venv/model/preflight in progress on data disk |
| `RUNNING` | PID alive, log advances, GPU/process agree |
| `PARTIAL` | some stages/artifacts complete; required verification remains |
| `COMPLETED` | entry process exited successfully and expected outputs exist |
| `COLLECTED` | result archive and sidecar downloaded locally |
| `VERIFIED` | local SHA/open/inventory/required-evidence checks pass |
| `SHUTDOWN` | exact AutoDL instance is powered off; data retention stated |
| `FAILED` | actionable error captured; durable data preserved |

`COMPLETED` is not `VERIFIED`. Never skip `COLLECTED` and local hash checks.

## Environment contract

The generated launcher exports:

- `AUTODL_JOB_DIR`: durable job root;
- `AUTODL_RUN_ID`: run identifier;
- `AUTODL_RUN_ROOT`: unique output directory;
- `MODEL_PATH`: local audited model snapshot;
- `AGENTICQWEN_MODEL_PATH`: compatibility alias for AgenticQwen projects;
- `HF_HOME` and `HF_HUB_CACHE`: data-disk caches;
- `PYTHONPATH=<project>/src` when present.

Entry commands should write all results beneath `$AUTODL_RUN_ROOT` and save checkpoints there.

## Required artifact minimum

- immutable resolved config;
- launch/preflight metadata;
- full run log;
- checkpoints or final adapter/model;
- train/eval metrics;
- representative trajectories for agentic runs;
- data/split manifest and contamination check;
- independent verification/evaluator output;
- artifact inventory with hashes.

## Resume contract

- Reuse the same job name, model directory, and explicit run ID.
- Let hub downloaders resume partial shards.
- Let the trainer discover its latest valid checkpoint.
- Do not treat a summary file alone as stage completion; require the associated weights and hashes.
- Append or preserve old logs/traces unless the program itself provides idempotent regeneration.

## Secret boundary

- Never embed API keys in entry commands, payloads, manifests, screenshots, or result archives.
- Pass secrets only through an interactive session or server environment when required.
- Scan text artifacts before collection; if a secret is found, rotate it and remove it from artifacts before download.

## Cost record

At handoff report:

- GPU and card count;
- displayed hourly price;
- start/end timestamps and approximate billable duration;
- approximate cost (label as estimate);
- instance shutdown status;
- whether the data disk remains retained.
