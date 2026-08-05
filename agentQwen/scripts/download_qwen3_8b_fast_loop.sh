#!/usr/bin/env bash
set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"
target="${1:-${project_root}/work/models/Qwen3-8B-msfast}"
manifest="${project_root}/outputs/agenticqwen-reproduction/artifacts/qwen3_8b_official_manifest.json"
log_path="${project_root}/outputs/agenticqwen-reproduction/artifacts/qwen3_8b_fast_download.log"
python_env="${PYTHON:-${project_root}/work/mlxenv312/bin/python}"

# The SDK keeps completed Range parts. If a transient connection reset makes a
# process exit, restart it against the same directory instead of re-downloading.
export MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS="${MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS:-4}"
export MODELSCOPE_DOWNLOAD_PARALLELS="${MODELSCOPE_DOWNLOAD_PARALLELS:-4}"
export MODELSCOPE_DOWNLOAD_INTRA_CLOUD="${MODELSCOPE_DOWNLOAD_INTRA_CLOUD:-true}"
export MODELSCOPE_DOWNLOAD_PART_SIZE_MB="${MODELSCOPE_DOWNLOAD_PART_SIZE_MB:-160}"

while true; do
    if "${python_env}" outputs/agenticqwen-reproduction/scripts/verify_snapshot.py --model-dir "${target}" --manifest "${manifest}" >/dev/null 2>&1; then
        echo "$(date -Iseconds) snapshot_verified" >> "${log_path}"
        exit 0
    fi
    echo "$(date -Iseconds) resume_snapshot_download target=${target}" >> "${log_path}"
    if "${python_env}" - <<PY >> "${log_path}" 2>&1
from modelscope.hub.snapshot_download import snapshot_download
print(snapshot_download("Qwen/Qwen3-8B", revision="master", local_dir=r"${target}", max_workers=8))
PY
    then
        echo "$(date -Iseconds) sdk_returned_success" >> "${log_path}"
    else
        echo "$(date -Iseconds) sdk_returned_failure; retrying" >> "${log_path}"
    fi
    sleep 5
done
