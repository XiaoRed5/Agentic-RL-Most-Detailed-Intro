#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/industrial_agenticqwen.json}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/artifacts/industrial_agenticqwen}"
STAGE="${STAGE:-all}"

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHON_BIN
export AGENTICQWEN_RUN_ROOT="${RUN_ROOT}"
export AGENTICQWEN_UPSTREAM_REPO="${AGENTICQWEN_UPSTREAM_REPO:-${PROJECT_DIR}/../../work/upstream/data_synth_and_rl}"

exec "${PYTHON_BIN}" -m agentic_repro.industrial_pipeline \
  --config "${CONFIG_PATH}" \
  --run-root "${RUN_ROOT}" \
  --stage "${STAGE}" \
  "$@"
