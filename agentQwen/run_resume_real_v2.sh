#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hadoop-aipnlp/agenticqwen-paper-debug-20260804"
PYTHON_BIN="/home/hadoop-aipnlp/.conda/envs/agenticqwen-py310/bin/python"
RUN_ROOT="$ROOT/artifacts/agenticqwen_codelab_real_run2"

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export AIGC_APP_ID="$(cat /home/hadoop-aipnlp/.aigc_app_id)"
export AGENTICQWEN_MODEL_PATH="$ROOT/models/Qwen3-8B"
exec "$PYTHON_BIN" -m agentic_repro.paper_grpo_train \
  --config "$ROOT/configs/agenticqwen_codelab_real_v2.json" \
  --output-root "$RUN_ROOT" \
  --mode train
