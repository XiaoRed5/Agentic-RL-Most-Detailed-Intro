#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
PYTHON_SOURCE="${PYTHON_SOURCE:-$WORKSPACE_DIR/work/mlxenv312/bin/python}"
BENCH_VENV="${BENCH_VENV:-$WORKSPACE_DIR/work/benchvenv312}"
TAU2_REPO="${TAU2_REPO:-$WORKSPACE_DIR/work/upstream/tau2-v0.2.0}"

if [[ ! -x "$PYTHON_SOURCE" ]]; then
  echo "Python 3.12 source interpreter is missing: $PYTHON_SOURCE" >&2
  exit 2
fi

if [[ ! -x "$BENCH_VENV/bin/python" ]]; then
  "$PYTHON_SOURCE" -m venv "$BENCH_VENV"
fi

"$BENCH_VENV/bin/python" -m pip install --upgrade pip
"$BENCH_VENV/bin/python" -m pip install "bfcl-eval==2026.3.23"

if [[ ! -d "$TAU2_REPO/.git" ]]; then
  git clone --branch v0.2.0 --depth 1 \
    https://github.com/sierra-research/tau2-bench.git "$TAU2_REPO"
fi

git -C "$TAU2_REPO" rev-parse HEAD
"$BENCH_VENV/bin/python" -m pip install -e "$TAU2_REPO"
"$BENCH_VENV/bin/bfcl" test-categories >/dev/null
"$BENCH_VENV/bin/tau2" check-data

echo "BFCL: $($BENCH_VENV/bin/python -c 'import importlib.metadata as m; print(m.version("bfcl-eval"))')"
echo "TAU-2: $(git -C "$TAU2_REPO" describe --tags --always)"
