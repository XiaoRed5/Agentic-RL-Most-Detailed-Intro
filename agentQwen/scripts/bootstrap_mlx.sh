#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_DIR/.runtime}"
VENV_DIR="$RUNTIME_DIR/mlx"
MODEL_DIR="${MODEL_DIR:-$RUNTIME_DIR/models/Qwen3-8B-4bit}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

if [[ -z "$UV_BIN" ]]; then
  echo "uv is required. Install it first: https://docs.astral.sh/uv/"
  exit 1
fi

mkdir -p "$RUNTIME_DIR"
"$UV_BIN" venv --python 3.12 "$VENV_DIR"
"$UV_BIN" pip install --python "$VENV_DIR/bin/python" mlx-lm

if [[ "${DOWNLOAD_MODEL:-0}" == "1" ]]; then
  mkdir -p "$(dirname "$MODEL_DIR")"
  HF_HOME="$RUNTIME_DIR/hf-cache" "$VENV_DIR/bin/python" -c \
    "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-8B-4bit', local_dir='$MODEL_DIR')"
else
  echo "MLX backend is ready. Model download was not started."
  echo "When approved: DOWNLOAD_MODEL=1 ./scripts/bootstrap_mlx.sh"
fi

echo "Run after download:"
echo "PYTHON_BIN=$VENV_DIR/bin/python QWEN_MODEL_PATH=$MODEL_DIR ./run.sh"

