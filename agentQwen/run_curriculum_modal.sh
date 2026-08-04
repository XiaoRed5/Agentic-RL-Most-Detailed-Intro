#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_MODAL_BIN="$PROJECT_DIR/../../work/modalenv/bin/modal"
if [[ -n "${MODAL_BIN:-}" ]]; then
  MODAL_BIN="$MODAL_BIN"
elif command -v modal >/dev/null 2>&1; then
  MODAL_BIN="$(command -v modal)"
elif [[ -x "$LOCAL_MODAL_BIN" ]]; then
  MODAL_BIN="$LOCAL_MODAL_BIN"
else
  echo "Modal CLI not found. Install it with: uv venv ../../work/modalenv && uv pip install --python ../../work/modalenv/bin/python 'modal>=1,<2'" >&2
  exit 2
fi
RUN_ID="${RUN_ID:-qwen3-8b-curriculum-$(date -u +%Y%m%d-%H%M%S)}"
MAX_BUDGET_USD="${MAX_BUDGET_USD:-15}"
TRAINING_GPU="${TRAINING_GPU:-A100-80GB}"
ESTIMATED_MINUTES="${ESTIMATED_MINUTES:-180}"
TRAINING_TIMEOUT_MINUTES="${TRAINING_TIMEOUT_MINUTES:-210}"
BENCHMARK_MINUTES="${BENCHMARK_MINUTES:-35}"

cd "$PROJECT_DIR"
RUN_STATUS=0
"$MODAL_BIN" run modal_curriculum.py \
  --run-id "$RUN_ID" \
  --max-budget-usd "$MAX_BUDGET_USD" \
  --training-gpu "$TRAINING_GPU" \
  --estimated-minutes "$ESTIMATED_MINUTES" \
  --training-timeout-minutes "$TRAINING_TIMEOUT_MINUTES" \
  --benchmark-minutes "$BENCHMARK_MINUTES" || RUN_STATUS=$?

mkdir -p "$PROJECT_DIR/artifacts/cloud_curriculum/$RUN_ID"
DOWNLOAD_STATUS=0
"$MODAL_BIN" volume get --force agenticqwen-curriculum-results \
  "/$RUN_ID" \
  "$PROJECT_DIR/artifacts/cloud_curriculum/$RUN_ID" || DOWNLOAD_STATUS=$?

if [[ "$DOWNLOAD_STATUS" -ne 0 ]]; then
  echo "Cloud run status=$RUN_STATUS and artifact download failed status=$DOWNLOAD_STATUS" >&2
  exit "$DOWNLOAD_STATUS"
fi
if [[ "$RUN_STATUS" -ne 0 ]]; then
  echo "Cloud run failed with status=$RUN_STATUS; partial artifacts were preserved locally." >&2
  exit "$RUN_STATUS"
fi

if [[ "${AUTO_FINALIZE:-1}" == "1" ]]; then
  "$PROJECT_DIR/finalize_curriculum_project.sh" \
    "$PROJECT_DIR/artifacts/cloud_curriculum/$RUN_ID"
fi

echo "Downloaded: $PROJECT_DIR/artifacts/cloud_curriculum/$RUN_ID"
