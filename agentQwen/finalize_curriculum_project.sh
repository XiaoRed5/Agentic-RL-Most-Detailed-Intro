#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
NODE_BIN="${NODE_BIN:-/Users/hongbo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_DIR/work/mlxenv312/bin/python}"
PRESENTATION_TOOLS="/Users/hongbo/.codex/plugins/cache/openai-primary-runtime/presentations/26.802.11031/skills/presentations/container_tools"
RUN_ROOT="${1:-}"

if [[ -z "$RUN_ROOT" || ! -f "$RUN_ROOT/run_summary.json" ]]; then
  echo "Usage: $0 <downloaded-run-root>" >&2
  exit 2
fi

"$PROJECT_DIR/finalize_curriculum_report.sh" "$RUN_ROOT"
"$NODE_BIN" "$PROJECT_DIR/scripts/build_curriculum_deck.mjs" "$RUN_ROOT"

PPTX="$PROJECT_DIR/slides/AgenticQwen_Curriculum_Cloud.pptx"
"$PYTHON_BIN" "$PRESENTATION_TOOLS/slides_test.py" "$PPTX"

echo "Final HTML: $PROJECT_DIR/agenticqwen_report/index.html"
echo "Final PPTX: $PPTX"
