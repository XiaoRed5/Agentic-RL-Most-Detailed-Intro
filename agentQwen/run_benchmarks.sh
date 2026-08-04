#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_DIR/work/mlxenv312/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/benchmarks.json}"
PROFILE="${PROFILE:-smoke}"
BENCHMARK="${BENCHMARK:-all}"
VARIANTS="${VARIANTS:-base,adapter}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

EXTRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

"$PYTHON_BIN" -m agentic_repro.benchmark_runner \
  --config "$CONFIG_PATH" \
  --profile "$PROFILE" \
  --benchmark "$BENCHMARK" \
  --variants "$VARIANTS" \
  "${EXTRA_ARGS[@]}"

echo "Benchmark artifacts: $PROJECT_DIR/artifacts/benchmarks"
