#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_DIR/work/mlxenv312/bin/python}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

"$PYTHON_BIN" -m agentic_repro.ablations \
  --training-log artifacts/real_qwen3_8b/training_log.jsonl \
  --task-manifest artifacts/real_qwen3_8b/task_manifest.json \
  --output artifacts/ablations/offline_reward_diagnostic.json \
  --process-weight 0.3

echo "Diagnostic: $PROJECT_DIR/artifacts/ablations/offline_reward_diagnostic.json"
echo "Matrix: $PROJECT_DIR/configs/ablation_matrix.json"
