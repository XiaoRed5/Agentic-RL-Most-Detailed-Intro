#!/usr/bin/env bash
set -euo pipefail

# ModelScope's large-file parallel Range downloader. Four Range streams is a
# stable setting on the current network; the SDK resumes existing part files.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
target="${1:-${project_root}/work/models/Qwen3-8B-msfast}"
cd "${project_root}"

export MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS="${MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS:-4}"
export MODELSCOPE_DOWNLOAD_PARALLELS="${MODELSCOPE_DOWNLOAD_PARALLELS:-4}"
export MODELSCOPE_DOWNLOAD_INTRA_CLOUD="${MODELSCOPE_DOWNLOAD_INTRA_CLOUD:-true}"
export MODELSCOPE_DOWNLOAD_PART_SIZE_MB="${MODELSCOPE_DOWNLOAD_PART_SIZE_MB:-160}"

python_env="${PYTHON:-work/mlxenv312/bin/python}"
"${python_env}" - <<PY
from modelscope.hub.snapshot_download import snapshot_download
path = snapshot_download(
    "Qwen/Qwen3-8B",
    revision="master",
    local_dir=r"${target}",
    max_workers=8,
)
print("MODEL_COMPLETE", path)
PY
