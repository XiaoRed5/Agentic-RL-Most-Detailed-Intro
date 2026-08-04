---
name: autodl-agentic-rl
description: "Automate the full AutoDL GPU experiment lifecycle for Agentic RL/LLM projects: choose and rent a GPU within an approved budget, package and upload code, download Hugging Face or ModelScope models to the data disk, create environments, preflight CUDA/TRL/bitsandbytes, launch resumable training, monitor logs/GPU/disk, collect and hash artifacts, and shut down billing safely. Use when the user says AutoDL, 租卡, 开实验, 上传代码/模型, 云端训练, 一键部署, 跑 SFT/GRPO/QLoRA, 监控训练, 下载结果, or 训练完关机."
---

# AutoDL Agentic RL

Execute an AutoDL experiment as an evidence-bearing state machine. Keep all durable data under `/root/autodl-tmp`; treat the system disk and browser session as disposable.

## Required resources

- Read `references/autodl-browser.md` before renting, uploading, or shutting down through the UI.
- Read `references/job-contract.md` before preparing a new job or recovering a failed run.
- Use `scripts/prepare_job.py` to generate the upload bundle. Do not hand-build long terminal commands.

## Authority and billing gates

1. Treat an explicit GPU type plus an explicit hourly/total budget or “确认创建” as authorization to rent within that bound.
2. If price, quantity, region, or GPU choice would materially change cost and is not already approved, stop at the final create button and ask.
3. Never enter passwords, payment details, SMS codes, API keys, or Jupyter tokens for the user. Ask them to complete login/recharge when required.
4. Never expose a Jupyter auth token in chat, logs, screenshots, commands, or artifacts.
5. After verified artifact download, shut down the instance when the user requested a complete lifecycle. Preserve the data disk unless deletion was explicitly requested.

## Workflow

### 1. Resolve the job contract

Inspect the project and determine:

- entry command;
- requirements file;
- model ID, source hub, and local model path;
- expected GPU/VRAM and minimum free disk;
- remote output root and completion signal;
- resume behavior;
- artifacts that must return locally;
- billing bound and shutdown policy.

Prefer ModelScope for public Qwen snapshots on mainland AutoDL. Preserve the canonical source model ID in manifests even when loading a local snapshot. Use Hugging Face plus AutoDL academic acceleration only when ModelScope lacks the exact revision.

For QLoRA, distinguish transport from runtime: downloading an official BF16 checkpoint and loading it as 4-bit NF4 is valid; record both the source checkpoint and runtime quantization.

### 2. Build the four-file upload set

Run:

```bash
python3 scripts/prepare_job.py \
  --project-dir <project> \
  --output-dir <local-staging> \
  --name <job-name> \
  --requirements <requirements.txt> \
  --model-id <org/model> \
  --model-source modelscope \
  --model-revision <immutable-revision-when-available> \
  --runtime-quantization '4-bit NF4 + BF16 compute' \
  --min-vram-gib <required> \
  --min-free-gib <required> \
  --require-import torch \
  --entry-command '<command using $AUTODL_RUN_ROOT>'
```

This produces:

- `<job>-payload.tar.gz`
- `<job>-autodl-launch.sh`
- `<job>-autodl-status.sh`
- `<job>-autodl-collect.sh`
- `<job>-autodl-manifest.json` (local audit record; do not upload unless useful)

Read the manifest and verify every local SHA-256 before upload. Large base weights are excluded from the payload by default; download them on the server. Small LoRA adapters may be included only when intentionally placed in the project and explicitly allowed.

Inspect derived directories before packaging and pass `--exclude` for stale runs, reports, slide decks, or checkpoints that are not runtime inputs. Pin `--model-revision` whenever the hub exposes an immutable revision; if it remains unset, label the run as not bitwise reproducible. Record the runtime load mode separately with `--runtime-quantization`.

### 3. Rent or reuse the instance

If no compatible running instance exists, follow `references/autodl-browser.md`. Verify GPU model, VRAM, card count, hourly price, data-disk size, and image before the final create action.

After boot, record non-secret evidence:

- AutoDL instance label/ID;
- GPU name/count/VRAM;
- hourly price shown;
- `/root/autodl-tmp` free space;
- start timestamp.

### 4. Upload and launch

Upload the four generated files to `/root` with the browser file chooser. Use the generated launch script as the only complex remote entry point:

```bash
bash /root/<job>-autodl-launch.sh
```

The launcher must:

1. verify payload SHA-256;
2. extract under `/root/autodl-tmp/jobs/<job>/project`;
3. run a dependency-free disk, GPU, and VRAM preflight before any install or model download;
4. create/reuse a data-disk venv;
5. install requirements;
6. resume the model snapshot and revalidate cached files against their SHA-256 manifest;
7. run the full CUDA/import/model-shard preflight;
8. set `AUTODL_RUN_ROOT`, `MODEL_PATH`, and `AGENTICQWEN_MODEL_PATH`;
9. start the entry command with `nohup`;
10. persist PID, run root, launch metadata, log path, exit code, and terminal `SUCCEEDED`/`FAILED` state.

Do not claim the job launched until `launch.json`, PID, log, and `nvidia-smi` agree.

### 5. Monitor without fragile commands

Run only:

```bash
bash /root/<job>-autodl-status.sh
```

Report meaningful transitions: model download progress, model load, first rollout, first optimizer step, checkpoint, stage completion, evaluation, failure, or completion. Do not narrate unchanged polls.

Keep monitoring until a terminal condition is reached. A stopped PID is not automatically success: inspect `run.log` and required output files.

### 6. Recover in place

On failure:

1. preserve the run root, model cache, log, and checkpoints;
2. identify the first actionable traceback;
3. patch and test locally;
4. regenerate the bundle with the same job name and, when resuming, the same run ID;
5. upload and relaunch;
6. verify that model download and checkpoints resume rather than restart.

Never delete caches or checkpoints to “fix” an error unless corruption is proven and the exact target is validated.

### 7. Collect, verify, then stop billing

After the run's own verifier passes, execute:

```bash
bash /root/<job>-autodl-collect.sh
```

Download both the result archive and its JSON sidecar. Locally verify:

- archive SHA-256 equals the sidecar;
- archive opens;
- artifact inventory exists;
- required summaries/checkpoints/adapters/traces/evaluations exist;
- no secret-like material is present;
- the claimed success status comes from independent evidence, not only process exit code.

Only then shut down the instance. Report the approximate runtime/cost, local artifact path, shutdown state, and whether the data disk remains recoverable.

## Completion standard

Do not say “complete” unless the state has reached `VERIFIED` and, for full-lifecycle requests, `SHUTDOWN`. Use the exact partial state from `references/job-contract.md` otherwise.

## Common failure routing

- Slow/failed Hugging Face Xet CAS: keep partial data, prefer exact ModelScope snapshot, or use `/etc/network_turbo`; do not loop across mirrors blindly.
- Terminal input loses `_`, commas, or quoting: stop typing complex commands; upload/regenerate scripts and execute a hyphen-only filename.
- Disk pressure: inspect exact directories; move durable artifacts to `/root/autodl-tmp`; do not recursively delete broad paths.
- TRL API mismatch: introspect installed signatures before model load, pin tested versions, rerun preflight.
- Browser disconnect: rely on `nohup`, PID, run log, and data-disk state; reconnect and run status script.
- Training succeeded but benchmark failed: preserve training artifacts, fix benchmark environment separately, and keep status `PARTIAL` until evaluation passes.
