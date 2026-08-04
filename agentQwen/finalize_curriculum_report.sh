#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_DIR/work/mlxenv312/bin/python}"
RUN_ROOT="${1:-}"

if [[ -z "$RUN_ROOT" ]]; then
  echo "Usage: $0 <downloaded-run-root>" >&2
  exit 2
fi
if [[ ! -f "$RUN_ROOT/run_summary.json" ]]; then
  echo "Missing run_summary.json under $RUN_ROOT" >&2
  exit 2
fi
SOURCE_MD="$PROJECT_DIR/agenticqwen_report/TECHNICAL_REPORT.md"
OUTPUT_HTML="$PROJECT_DIR/agenticqwen_report/index.html"
REVIEW_JSON="$PROJECT_DIR/agenticqwen_report/index.html.review.json"

PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" -m agentic_repro.curriculum_report \
  --run-root "$RUN_ROOT" \
  --output "$SOURCE_MD" \
  --review-output "$REVIEW_JSON" \
  --allow-missing-bfcl

PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" -m agentic_repro.blog_renderer \
  "$SOURCE_MD" \
  --output "$OUTPUT_HTML" \
  --title "AgenticQwen：从失败轨迹到下一轮训练" \
  --brand "AgenticQwen/notes"

echo "Rendered: $OUTPUT_HTML"
echo "Independent review: REVIEW_UNAVAILABLE (recorded in $REVIEW_JSON)"
