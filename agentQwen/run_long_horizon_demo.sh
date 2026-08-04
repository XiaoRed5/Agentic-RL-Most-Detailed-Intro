#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_DIR/work/mlxenv312/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/long_horizon_demo.json}"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY must be injected through the environment; it is never read from config." >&2
  exit 2
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

"$PYTHON_BIN" -m agentic_repro.trajectory_runner --config "$CONFIG_PATH"

