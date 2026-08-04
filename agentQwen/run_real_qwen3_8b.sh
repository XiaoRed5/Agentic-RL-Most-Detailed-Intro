#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_DIR/work/mlxenv312/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/real_qwen3_8b.json}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

"$PYTHON_BIN" -m agentic_repro.real_grpo --config "$CONFIG_PATH"
"$PYTHON_BIN" -m agentic_repro.verify_real --config "$CONFIG_PATH"
"$PROJECT_DIR/run_ablations.sh"

if [[ "${RUN_TRAJECTORY:-auto}" != "0" ]]; then
  if [[ -n "${DASHSCOPE_API_KEY:-}" ]]; then
    "$PROJECT_DIR/run_long_horizon_demo.sh"
  elif [[ -s "$PROJECT_DIR/artifacts/long_horizon/trajectory_qwen3.7_flash.json" ]]; then
    echo "DASHSCOPE_API_KEY is not set; reusing the verified saved trajectory artifact."
  else
    echo "A trajectory artifact is required. Set DASHSCOPE_API_KEY or run with RUN_TRAJECTORY=0 after supplying an artifact." >&2
    exit 4
  fi
fi

if [[ "${RUN_BENCHMARKS:-0}" == "1" ]]; then
  PROFILE="${BENCHMARK_PROFILE:-smoke}" "$PROJECT_DIR/run_benchmarks.sh"
else
  DRY_RUN=1 PROFILE=smoke "$PROJECT_DIR/run_benchmarks.sh" >/dev/null
fi

"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" -m agentic_repro.blog_report \
  --project "$PROJECT_DIR" \
  --output-dir "$PROJECT_DIR/agenticqwen_report"

if [[ "${BUILD_SLIDES:-1}" == "1" ]]; then
  DEFAULT_NODE="/Users/hongbo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
  NODE_BIN="${NODE_BIN:-$DEFAULT_NODE}"
  if [[ ! -x "$NODE_BIN" ]]; then
    NODE_BIN="$(command -v node || true)"
  fi
  if [[ -z "$NODE_BIN" ]]; then
    echo "Node.js not found; HTML/report completed but PPT generation was skipped." >&2
    exit 3
  fi
  "$NODE_BIN" "$PROJECT_DIR/scripts/build_real_deck.mjs"
fi

echo "Report: $PROJECT_DIR/agenticqwen_report/index.html"
echo "Slides: $PROJECT_DIR/slides/AgenticQwen_LongHorizon_Lab.pptx"
echo "Self-check: $PROJECT_DIR/artifacts/real_qwen3_8b/completion_matrix.json"
