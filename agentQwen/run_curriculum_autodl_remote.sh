#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/agenticqwen-reproduction}"
RUN_ID="${RUN_ID:-qwen3-8b-autodl-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/agenticqwen-runs/$RUN_ID}"
VENV_DIR="${VENV_DIR:-/root/autodl-tmp/agenticqwen-train-venv}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/agenticqwen-logs}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen3-8B}"

mkdir -p "$RUN_ROOT" "$LOG_DIR" /root/autodl-tmp/huggingface
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python -m venv --system-site-packages "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements-autodl-train.txt"
if [[ ! -s "$MODEL_DIR/.modelscope_complete.json" ]]; then
  "$VENV_DIR/bin/python" "$PROJECT_DIR/scripts/modelscope_download.py"
fi

export PYTHONPATH="$PROJECT_DIR/src"
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
if [[ -r /etc/network_turbo ]]; then
  set +u
  source /etc/network_turbo
  set -u
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AGENTICQWEN_ORCHESTRATOR=autodl
export AGENTICQWEN_RUN_ID="$RUN_ID"
export AGENTICQWEN_MODEL_PATH="$MODEL_DIR"
export AUTODL_INSTANCE_ID="${AUTODL_INSTANCE_ID:-unknown-autodl-instance}"

cd "$PROJECT_DIR"
"$VENV_DIR/bin/python" -m agentic_repro.curriculum_train \
  --config "$PROJECT_DIR/configs/curriculum_qwen3_8b.json" \
  --output-root "$RUN_ROOT" \
  --mode pipeline 2>&1 | tee "$LOG_DIR/$RUN_ID.log"

printf '%s\n' "$RUN_ROOT" > "$LOG_DIR/latest_run_root.txt"
