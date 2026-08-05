#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${AGENTICQWEN_ROOT:-/home/hadoop-aipnlp/agenticqwen-paper-debug-20260804}"
UPSTREAM_ROOT="${AGENTICQWEN_UPSTREAM_REPO:-$RUN_ROOT/upstream/data_synth_and_rl}"
CACHE_ROOT="$RUN_ROOT/cache/official-verl"
MINIFORGE_ROOT="$RUN_ROOT/venvs/miniforge3-26.3.2-3"
ENV_ROOT="$RUN_ROOT/venvs/verl-sglang-py312"
STATE_ROOT="$RUN_ROOT/artifacts/official_verl_env"
INSTALLER="$CACHE_ROOT/Miniforge3-26.3.2-3-Linux-x86_64.sh"
FLASH_WHEEL="$CACHE_ROOT/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
FLASH_SOURCE_ARCHIVE="$CACHE_ROOT/flash_attn-2.8.1.tar.gz"
FLASH_CUDA122_PATCH="$RUN_ROOT/scripts/patches/flash-attn-2.8.1-cuda12.2.patch"
PIP_INDEX="${AGENTICQWEN_PIP_INDEX:-http://pip.sankuai.com/simple/}"
PIP_HOST="${AGENTICQWEN_PIP_HOST:-pip.sankuai.com}"
PIP_EXTRA_HOST="${AGENTICQWEN_PIP_EXTRA_HOST:-pypi.sankuai.com}"
CONDA_FORGE_CHANNEL="${AGENTICQWEN_CONDA_FORGE_CHANNEL:-http://data-source-conda.sankuai.com/cloud/conda-forge/}"
CONDA_MAIN_CHANNEL="${AGENTICQWEN_CONDA_MAIN_CHANNEL:-http://data-source-conda.sankuai.com/pkgs/main/}"

mkdir -p "$CACHE_ROOT" "$STATE_ROOT" "$RUN_ROOT/venvs"

if [[ ! -s "$INSTALLER" ]]; then
  echo "Missing pinned Miniforge installer: $INSTALLER" >&2
  exit 20
fi
if [[ ! -s "$FLASH_WHEEL" ]]; then
  echo "Missing pinned FlashAttention wheel: $FLASH_WHEEL" >&2
  exit 21
fi

printf '%s  %s\n' \
  '848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c' \
  "$INSTALLER" | sha256sum --check --status
printf '%s  %s\n' \
  '15db5bb6524dcbf292c3c116aa4c2fa823b80abff0eb5bc58454107bca1ba0c2' \
  "$FLASH_WHEEL" | sha256sum --check --status

if [[ ! -x "$MINIFORGE_ROOT/bin/conda" ]]; then
  bash "$INSTALLER" -b -p "$MINIFORGE_ROOT"
fi

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  CONDA_NOTICES=false "$MINIFORGE_ROOT/bin/conda" create -y \
    --override-channels \
    --channel "$CONDA_FORGE_CHANNEL" \
    --channel "$CONDA_MAIN_CHANNEL" \
    -p "$ENV_ROOT" python=3.12 pip
fi

# CodeLab's system GCC 4.8 cannot compile the C++17 FlashAttention sources.
# Keep a pinned compiler inside the same isolated environment.
CONDA_NOTICES=false "$MINIFORGE_ROOT/bin/conda" install -y \
  --override-channels \
  --channel "$CONDA_FORGE_CHANNEL" \
  --channel "$CONDA_MAIN_CHANNEL" \
  -p "$ENV_ROOT" 'gxx_linux-64=11.4.0'

PYTHON_BIN="$ENV_ROOT/bin/python"
export CONDA_PREFIX="$ENV_ROOT"
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC="$ENV_ROOT/bin/x86_64-conda-linux-gnu-cc"
export CXX="$ENV_ROOT/bin/x86_64-conda-linux-gnu-c++"
export PIP_INDEX_URL="$PIP_INDEX"
export PIP_TRUSTED_HOST="$PIP_HOST $PIP_EXTRA_HOST"
PIP=("$PYTHON_BIN" -m pip)

"${PIP[@]}" install --upgrade 'pip<26' setuptools wheel
"${PIP[@]}" install 'sglang[all]==0.5.2' torch-memory-saver
"${PIP[@]}" install \
  'transformers[hf_xet]==4.56.1' accelerate datasets peft hf-transfer \
  'numpy==1.26.4' 'scipy==1.14.1' 'pyarrow>=19.0.0' pandas \
  json5 python-dotenv cachetools \
  'tensordict>=0.8.0,<=0.10.0,!=0.9.0' torchdata \
  'ray[default]>=2.10' codetiming hydra-core pylatexenc qwen-vl-utils \
  wandb dill pybind11 liger-kernel mathruler tensorboard \
  'nvidia-ml-py>=12.560.30' 'fastapi[standard]>=0.115.0' \
  'optree>=0.13.0' 'pydantic>=2.9' 'grpcio>=1.62.1' uvicorn==0.38.0
# The upstream release wheel is ABI-compatible with Python/Torch/CUDA but was
# linked against GLIBC_2.32.  CodeLab currently exposes GLIBC_2.28, so verify
# the import and fall back to an on-host source build when required.
"${PIP[@]}" install --no-deps "$FLASH_WHEEL"
if ! "$PYTHON_BIN" -c 'import flash_attn' >/dev/null 2>&1; then
  "${PIP[@]}" uninstall -y flash-attn
  export MAX_JOBS="${MAX_JOBS:-8}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
  export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}"
  if [[ ! -s "$FLASH_CUDA122_PATCH" ]]; then
    echo "Missing CUDA 12.2 compatibility patch: $FLASH_CUDA122_PATCH" >&2
    exit 22
  fi
  if [[ ! -s "$FLASH_SOURCE_ARCHIVE" ]]; then
    "${PIP[@]}" download --no-build-isolation --no-deps --no-binary=flash-attn \
      'flash-attn==2.8.1' --dest "$CACHE_ROOT"
  fi
  FLASH_BUILD_ROOT="$(mktemp -d "$CACHE_ROOT/flash-attn-2.8.1-build.XXXXXX")"
  tar -xzf "$FLASH_SOURCE_ARCHIVE" -C "$FLASH_BUILD_ROOT" --strip-components=1
  patch --batch --forward -d "$FLASH_BUILD_ROOT" -p1 < "$FLASH_CUDA122_PATCH"
  "${PIP[@]}" install --no-deps --no-build-isolation "$FLASH_BUILD_ROOT"
  rm -rf "$FLASH_BUILD_ROOT"
fi
"${PIP[@]}" install --no-deps --editable "$UPSTREAM_ROOT/RL"

export PYTHONPATH="$UPSTREAM_ROOT/RL${PYTHONPATH:+:$PYTHONPATH}"
export AGENTICQWEN_ENV_STATE="$STATE_ROOT/environment.json"
"$PYTHON_BIN" "$RUN_ROOT/scripts/verify_official_verl_env.py" \
  --upstream "$UPSTREAM_ROOT/RL" \
  --output "$AGENTICQWEN_ENV_STATE"

echo "OFFICIAL_VERL_ENV_READY=$ENV_ROOT"
