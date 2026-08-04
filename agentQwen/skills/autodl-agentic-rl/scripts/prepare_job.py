#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import tarfile
from pathlib import Path
from typing import Iterable


SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "wandb",
}
SKIP_SUFFIXES = {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}
SKIP_FILES = {".DS_Store"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError("Job name must contain an ASCII letter or digit")
    return slug[:48]


def excluded(relative: Path, extra: set[str]) -> bool:
    if relative.name in SKIP_FILES:
        return True
    if any(part in SKIP_DIRS or part.startswith(".venv") for part in relative.parts):
        return True
    if relative.suffix.lower() in SKIP_SUFFIXES:
        return True
    text = relative.as_posix()
    return any(relative.match(pattern) or text.startswith(pattern.rstrip("/") + "/") for pattern in extra)


def add_project(archive: tarfile.TarFile, root: Path, extra: set[str]) -> list[str]:
    added: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if excluded(relative, extra) or path.is_symlink() or not path.is_file():
            continue
        arcname = Path("project") / relative
        archive.add(path, arcname=str(arcname), recursive=False)
        added.append(str(relative))
    return added


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def requirements_install(requirements: str | None) -> str:
    if not requirements:
        return ":"
    return f'"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/{requirements}"'


def model_install(source: str) -> str:
    if source == "modelscope":
        return '"$VENV_DIR/bin/python" -m pip install "modelscope>=1.39,<2"'
    return '"$VENV_DIR/bin/python" -m pip install "huggingface-hub>=0.36"'


def require_args(values: Iterable[str]) -> str:
    return " ".join(f"--require {q(value)}" for value in values)


def write(path: Path, content: str, executable: bool = False) -> None:
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a self-contained AutoDL upload bundle")
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--entry-command", required=True)
    parser.add_argument("--requirements", default="requirements.txt")
    parser.add_argument("--model-id")
    parser.add_argument("--model-source", choices=("modelscope", "huggingface"), default="modelscope")
    parser.add_argument("--model-revision")
    parser.add_argument("--runtime-quantization", default="unspecified")
    parser.add_argument("--model-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    parser.add_argument("--min-vram-gib", type=float, default=0.0)
    parser.add_argument("--require-import", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    if "sk-" in args.entry_command.lower():
        raise RuntimeError("Refusing to embed a secret-like API key in the launch command")
    project = args.project_dir.expanduser().resolve()
    if not project.is_dir():
        raise RuntimeError(f"Project directory is missing: {project}")
    if args.requirements and not (project / args.requirements).is_file():
        raise RuntimeError(f"Requirements file is missing: {project / args.requirements}")

    slug = slugify(args.name)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = output / f"{slug}-payload.tar.gz"
    launch = output / f"{slug}-autodl-launch.sh"
    status = output / f"{slug}-autodl-status.sh"
    collect = output / f"{slug}-autodl-collect.sh"
    helper_root = Path(__file__).resolve().parent

    with tarfile.open(payload, "w:gz") as archive:
        project_files = add_project(archive, project, set(args.exclude))
        for helper in ("model_fetch.py", "remote_preflight.py", "package_results.py"):
            archive.add(helper_root / helper, arcname=f"project/.autodl/{helper}", recursive=False)
    payload_hash = sha256(payload)

    remote_job = f"/root/autodl-tmp/jobs/{slug}"
    remote_model = args.model_dir or f"/root/autodl-tmp/models/{slugify((args.model_id or slug).split('/')[-1])}"
    remote_payload = f"/root/{payload.name}"
    remote_result = f"/root/{slug}-results.tar.gz"
    imports = args.require_import or ["torch"]
    run_id_expression = q(args.run_id) if args.run_id else '"$(date -u +%Y%m%d-%H%M%S)"'
    revision_arg = f" --revision {q(args.model_revision)}" if args.model_revision else ""

    if args.model_id:
        model_block = f"""
{model_install(args.model_source)}
"$VENV_DIR/bin/python" "$PROJECT_DIR/.autodl/model_fetch.py" \\
  --model-id {q(args.model_id)} \\
  --source {q(args.model_source)} \\
  --target "$MODEL_DIR"{revision_arg}
""".strip()
        model_preflight = '--model-dir "$MODEL_DIR"'
    else:
        model_block = ":"
        model_preflight = ""

    turbo = """
if [[ -r /etc/network_turbo ]]; then
  set +u
  source /etc/network_turbo
  set -u
fi
""" if args.model_source == "huggingface" else ""

    launch_script = f"""#!/usr/bin/env bash
set -euo pipefail

PAYLOAD={q(remote_payload)}
EXPECTED_SHA={q(payload_hash)}
JOB_DIR={q(remote_job)}
PROJECT_DIR="$JOB_DIR/project"
VENV_DIR="/root/autodl-tmp/venvs/{slug}"
MODEL_DIR={q(remote_model)}
RUN_ID={run_id_expression}
RUN_ROOT="/root/autodl-tmp/runs/{slug}/$RUN_ID"
LOG_DIR="/root/autodl-tmp/logs/{slug}"
ENTRY_COMMAND={q(args.entry_command)}
AUTODL_RUNTIME_QUANTIZATION={q(args.runtime_quantization)}

mkdir -p "$JOB_DIR" "$RUN_ROOT" "$LOG_DIR" "$MODEL_DIR" /root/autodl-tmp/huggingface
ACTUAL_SHA="$(sha256sum "$PAYLOAD" | awk '{{print $1}}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "Payload SHA-256 mismatch: expected=$EXPECTED_SHA actual=$ACTUAL_SHA" >&2
  exit 2
fi
tar -xzf "$PAYLOAD" -C "$JOB_DIR"
BOOTSTRAP_PYTHON=""
for candidate in python3 python /root/miniconda3/bin/python /opt/conda/bin/python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="$(command -v "$candidate")"
    break
  fi
  if [[ -x "$candidate" ]]; then
    BOOTSTRAP_PYTHON="$candidate"
    break
  fi
done
if [[ -z "$BOOTSTRAP_PYTHON" ]]; then
  echo "No usable Python interpreter found in the AutoDL base image" >&2
  exit 127
fi
"$BOOTSTRAP_PYTHON" "$PROJECT_DIR/.autodl/remote_preflight.py" \\
  --basic-only \\
  --data-root /root/autodl-tmp \\
  --min-free-gib {args.min_free_gib} \\
  --min-vram-gib {args.min_vram_gib} \\
  --output "$JOB_DIR/early-preflight.json"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$BOOTSTRAP_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install --upgrade pip
{requirements_install(args.requirements)}

export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=120
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_DIR/src:${{PYTHONPATH:-}}"
export MODEL_PATH="$MODEL_DIR"
export AGENTICQWEN_MODEL_PATH="$MODEL_DIR"
export AUTODL_RUN_ID="$RUN_ID"
export AUTODL_RUN_ROOT="$RUN_ROOT"
export AUTODL_JOB_DIR="$JOB_DIR"
export AUTODL_RUNTIME_QUANTIZATION
{turbo.rstrip()}
{model_block}

"$VENV_DIR/bin/python" "$PROJECT_DIR/.autodl/remote_preflight.py" \\
  --data-root /root/autodl-tmp \\
  --min-free-gib {args.min_free_gib} \\
  --min-vram-gib {args.min_vram_gib} \\
  {model_preflight} {require_args(imports)} \\
  --output "$RUN_ROOT/preflight.json"

printf '%s\\n' "$RUN_ROOT" > "$JOB_DIR/latest-run.txt"
nohup bash -c '
set +e
project_dir="$1"
run_root="$2"
entry_command="$3"
cd "$project_dir"
bash -lc "$entry_command"
code=$?
if [[ "$code" -eq 0 ]]; then state="SUCCEEDED"; else state="FAILED"; fi
finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf "{{\\\"status\\\":\\\"%s\\\",\\\"exit_code\\\":%s,\\\"finished_at\\\":\\\"%s\\\"}}\\n" "$state" "$code" "$finished_at" > "$run_root/process-exit.json"
exit "$code"
' bash "$PROJECT_DIR" "$RUN_ROOT" "$ENTRY_COMMAND" \\
  > "$RUN_ROOT/run.log" 2>&1 < /dev/null &
PID=$!
printf '%s\\n' "$PID" > "$JOB_DIR/latest-pid.txt"
printf '{{"status":"launched","pid":%s,"run_root":"%s","payload_sha256":"%s"}}\\n' \\
  "$PID" "$RUN_ROOT" "$ACTUAL_SHA" > "$RUN_ROOT/launch.json"
echo "LAUNCHED pid=$PID run_root=$RUN_ROOT log=$RUN_ROOT/run.log"
"""

    status_script = f"""#!/usr/bin/env bash
set -euo pipefail
JOB_DIR={q(remote_job)}
VENV_DIR="/root/autodl-tmp/venvs/{slug}"
RUN_ROOT="$(tr -d '\\r\\n' < "$JOB_DIR/latest-run.txt")"
PID="$(tr -d '\\r\\n' < "$JOB_DIR/latest-pid.txt")"
if kill -0 "$PID" 2>/dev/null; then
  echo "STATE=RUNNING PID=$PID RUN_ROOT=$RUN_ROOT"
elif [[ -f "$RUN_ROOT/process-exit.json" ]]; then
  "$VENV_DIR/bin/python" -c 'import json,sys; p=json.load(open(sys.argv[1])); print("STATE=%s EXIT_CODE=%s PID=%s RUN_ROOT=%s" % (p["status"], p["exit_code"], sys.argv[2], sys.argv[3]))' "$RUN_ROOT/process-exit.json" "$PID" "$RUN_ROOT"
else
  echo "STATE=UNKNOWN PID=$PID RUN_ROOT=$RUN_ROOT"
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
df -h /root/autodl-tmp
tail -n 80 "$RUN_ROOT/run.log"
"""

    collect_script = f"""#!/usr/bin/env bash
set -euo pipefail
JOB_DIR={q(remote_job)}
VENV_DIR="/root/autodl-tmp/venvs/{slug}"
RUN_ROOT="$(tr -d '\\r\\n' < "$JOB_DIR/latest-run.txt")"
"$VENV_DIR/bin/python" "$JOB_DIR/project/.autodl/package_results.py" \\
  --run-root "$RUN_ROOT" \\
  --output {q(remote_result)}
echo "DOWNLOAD {remote_result} and {remote_result}.json"
"""

    write(launch, launch_script, executable=True)
    write(status, status_script, executable=True)
    write(collect, collect_script, executable=True)
    manifest = {
        "schema_version": 1,
        "status": "ready",
        "job": slug,
        "project_dir": str(project),
        "project_file_count": len(project_files),
        "entry_command": args.entry_command,
        "model_id": args.model_id,
        "model_source": args.model_source,
        "model_revision": args.model_revision,
        "runtime_quantization": args.runtime_quantization,
        "remote_model_dir": remote_model,
        "remote_run_root": f"/root/autodl-tmp/runs/{slug}/<run-id>",
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (payload, launch, status, collect)
        ],
    }
    manifest_path = output / f"{slug}-autodl-manifest.json"
    write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
