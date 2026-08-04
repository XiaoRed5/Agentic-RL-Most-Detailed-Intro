#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/smoke.json}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

ARGS=(all --config "$CONFIG_PATH")
if [[ -n "$QWEN_MODEL_PATH" ]]; then
  ARGS+=(--qwen-model-path "$QWEN_MODEL_PATH")
fi

"$PYTHON_BIN" -m agentic_repro.cli "${ARGS[@]}"

echo "Done: $PROJECT_DIR/agenticqwen_report/index.html"
echo "Ledger: $PROJECT_DIR/artifacts/verification.json"

