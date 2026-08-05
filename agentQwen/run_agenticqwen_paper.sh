#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/agenticqwen_paper_micro.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/artifacts/agenticqwen_paper_micro}"
MODE="${MODE:-synthesize}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m agentic_repro.paper_grpo_train \
  --config "${CONFIG_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --mode "${MODE}"

if [[ "${MODE}" == "synthesize" ]]; then
  "${PYTHON_BIN}" -m agentic_repro.paper_grpo_train \
    --config "${CONFIG_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --mode verify
fi
