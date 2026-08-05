#!/usr/bin/env bash
set -euo pipefail

# CodeLab-only sequential runner.  The paper-style behavior-tree flywheel
# remains the primary reproduction; the explicit refund curriculum is run
# afterwards so its failure-mining and fresh-process evidence are auditable
# independently rather than being conflated with the paper run.

ROOT="${AGENTICQWEN_REMOTE_ROOT:-/home/hadoop-aipnlp/agenticqwen-paper-debug-20260804}"
PYTHON="${AGENTICQWEN_PYTHON:-/home/hadoop-aipnlp/.conda/envs/agenticqwen-py310/bin/python}"
MODEL="${AGENTICQWEN_MODEL_PATH:-$ROOT/models/Qwen3-8B}"
PAPER_OUT="${AGENTICQWEN_PAPER_OUT:-$ROOT/artifacts/agenticqwen_codelab_real_run1}"
CURRICULUM_OUT="${AGENTICQWEN_CURRICULUM_OUT:-$ROOT/artifacts/agenticqwen_codelab_curriculum_run1}"
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR" "$PAPER_OUT" "$CURRICULUM_OUT"
export PYTHONPATH="$ROOT/src"
export AGENTICQWEN_MODEL_PATH="$MODEL"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export AGENTICQWEN_ORCHESTRATOR="codelab"

if [[ ! -f "$PAPER_OUT/run_summary.json" ]]; then
  "$PYTHON" -m agentic_repro.paper_grpo_train \
    --config "$ROOT/configs/agenticqwen_codelab_real.json" \
    --output-root "$PAPER_OUT" \
    --mode train 2>&1 | tee "$LOG_DIR/agenticqwen_codelab_real_run1.log"
fi

if [[ ! -f "$CURRICULUM_OUT/run_summary.json" ]]; then
  "$PYTHON" -m agentic_repro.curriculum_train \
    --config "$ROOT/configs/curriculum_qwen3_8b.json" \
    --output-root "$CURRICULUM_OUT" \
    --mode pipeline 2>&1 | tee "$LOG_DIR/agenticqwen_codelab_curriculum_run1.log"
fi

printf '%s\n' "$CURRICULUM_OUT" > "$LOG_DIR/latest_curriculum_run_root.txt"
