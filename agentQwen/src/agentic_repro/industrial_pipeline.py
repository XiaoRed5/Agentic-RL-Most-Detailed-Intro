"""Resumable orchestration for the official AgenticQwen pipeline.

This module is deliberately a *control plane*, not a second toy trainer.  The
actual task synthesis, simulated-user solving, rubric filtering, parquet
conversion, and verl/SGLang training are executed from the upstream
``haruhi-sudo/data_synth_and_rl`` tree.  Every stage gets a durable manifest,
an append-only log, a redacted command record, and an output inventory so a
remote job can be resumed without silently reusing an incomplete directory.

The teacher model can be substituted (for example DeepSeek-v4-flash) through
the provider environment variables.  The manifest always records the
substitution explicitly; it never calls the substitute Qwen3-235B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
DEFAULT_STAGE_ORDER = (
    "preflight",
    "data_gen",
    "solve",
    "rubric",
    "filter",
    "convert",
    "train",
    "evaluate",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_inventory(path: Path, *, max_files: int = 5000) -> dict[str, Any]:
    """Return a bounded, deterministic inventory suitable for manifests."""
    if not path.exists():
        return {"path": str(path), "exists": False, "files": 0, "bytes": 0}
    if path.is_file():
        return {
            "path": str(path),
            "exists": True,
            "files": 1,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    files = sorted(item for item in path.rglob("*") if item.is_file())
    truncated = len(files) > max_files
    selected = files[:max_files]
    return {
        "path": str(path),
        "exists": True,
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "truncated": truncated,
        "sha256": hashlib.sha256(
            "\n".join(
                f"{item.relative_to(path)}:{item.stat().st_size}:{sha256_file(item)}"
                for item in selected
            ).encode("utf-8")
        ).hexdigest(),
    }


def _expand(value: Any, *, project_root: Path) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value.replace("${PROJECT_ROOT}", str(project_root)))
    if isinstance(value, list):
        return [_expand(item, project_root=project_root) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, project_root=project_root) for key, item in value.items()}
    return value


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("industrial pipeline config must be a JSON object")
    project_root = path.resolve().parent.parent
    os.environ.setdefault(
        "AGENTICQWEN_UPSTREAM_REPO",
        str((project_root / "../../work/upstream/data_synth_and_rl").resolve()),
    )
    value = _expand(value, project_root=project_root)
    value.setdefault("schema_version", SCHEMA_VERSION)
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported industrial pipeline schema: {value['schema_version']}")
    return value


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _official_paths(config: dict[str, Any]) -> dict[str, Path]:
    repo = Path(config["upstream"]["repo"]).expanduser().resolve()
    synthesis = repo / "tool_use_data_synthesis"
    rl = repo / "RL"
    return {
        "repo": repo,
        "synthesis": synthesis,
        "rl": rl,
        "data_gen": synthesis / "run_data_gen.py",
        "solve": synthesis / "run_solve_task.py",
        "rubric": synthesis / "run_rubrics.py",
        "filter": synthesis / "make_filtered_verl_data.py",
        "convert": rl / "my_script/data_process/virtual_tool_use_convert_parquet.py",
    }


def _credential_state(config: dict[str, Any]) -> dict[str, Any]:
    provider = config.get("provider", {})
    names = list(provider.get("api_key_envs", []))
    configured = [name for name in names if os.getenv(name, "").strip()]
    # Secret values are never serialized.  A remote launch can provide a
    # credential file instead; the state still records only its existence.
    files = []
    for raw in provider.get("api_key_files", []):
        path = Path(os.path.expanduser(os.path.expandvars(str(raw))))
        files.append({"path": str(path), "exists": path.is_file()})
    return {
        "api_key_envs": names,
        "configured_envs": configured,
        "configured": bool(configured or any(item["exists"] for item in files)),
        "api_key_files": files,
        "teacher_model": provider.get("teacher_model"),
        "teacher_substitution": provider.get("teacher_substitution"),
        "base_url_env": provider.get("base_url_env"),
    }


def _git_info(repo: Path) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {"is_git": False}
    def run(*args: str) -> str:
        value = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True, check=False)
        return value.stdout.strip()
    return {
        "is_git": True,
        "commit": run("rev-parse", "HEAD"),
        "remote": run("config", "--get", "remote.origin.url"),
        "dirty": bool(run("status", "--porcelain")),
    }


def preflight(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    paths = _official_paths(config)
    required = {name: str(path) for name, path in paths.items() if name not in {"repo", "synthesis", "rl"}}
    missing = {name: value for name, value in required.items() if not Path(value).is_file()}
    repo = paths["repo"]
    expected_remote = str(config.get("upstream", {}).get("expected_remote", ""))
    git = _git_info(repo)
    if not git.get("is_git") and config.get("upstream", {}).get("source_commit"):
        # Remote jobs are often uploaded as a tarball without .git.  Keep the
        # immutable source lock in the experiment config rather than silently
        # treating an unpinned checkout as official code.
        git = {
            "is_git": False,
            "archive_lock": True,
            "commit": config["upstream"]["source_commit"],
            "remote": config["upstream"].get("source_remote", ""),
            "dirty": False,
        }
    remote_ok = not expected_remote or expected_remote in git.get("remote", "")
    credential = _credential_state(config)
    warnings: list[str] = []
    if not credential["configured"]:
        warnings.append("teacher credential is not configured in this process; remote launch may provide it")
    if git.get("dirty"):
        warnings.append("upstream checkout is dirty; pin or archive the commit before a paper claim")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not missing and remote_ok else "FAIL",
        "timestamp": utc_now(),
        "platform": {"python": sys.version, "platform": platform.platform()},
        "upstream": {"paths": {key: str(value) for key, value in paths.items()}, "git": git, "expected_remote": expected_remote, "remote_ok": remote_ok},
        "required_scripts": required,
        "missing": missing,
        "credentials": credential,
        "warnings": warnings,
        "resource_contract": config.get("resource_contract", {}),
    }
    _json(run_root / "preflight.json", result)
    return result


@dataclass(frozen=True)
class StageSpec:
    name: str
    command: tuple[str, ...]
    cwd: Path
    output_paths: tuple[Path, ...]
    env_names: tuple[str, ...]


class IndustrialRunner:
    def __init__(self, config: dict[str, Any], run_root: Path, *, dry_run: bool = False) -> None:
        self.config = config
        self.run_root = run_root.resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.paths = _official_paths(config)

    def _stage(self, name: str) -> StageSpec:
        stages = self.config.get("stages", {})
        if name not in stages:
            raise KeyError(f"stage {name!r} is not configured")
        raw = stages[name]
        cwd = Path(raw.get("cwd", self.paths["repo"])).expanduser().resolve()
        command = tuple(str(item) for item in raw["command"])
        outputs = tuple(Path(str(item)).expanduser().resolve() for item in raw.get("output_paths", []))
        env_names = tuple(str(item) for item in raw.get("env_names", []))
        return StageSpec(name, command, cwd, outputs, env_names)

    def _state(self, name: str) -> Path:
        return self.run_root / "stages" / f"{name}.json"

    def _is_completed(self, name: str) -> bool:
        path = self._state(name)
        if not path.is_file():
            return False
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if value.get("status") != "completed":
            return False
        inventories = value.get("outputs", [])
        return bool(inventories) and all(item.get("exists") for item in inventories)

    def _command_record(self, spec: StageSpec) -> dict[str, Any]:
        env_names = sorted(set(spec.env_names) | set(self.config.get("provider", {}).get("api_key_envs", [])))
        return {
            "argv": list(spec.command),
            "command": shlex.join(spec.command),
            "cwd": str(spec.cwd),
            "env_names": env_names,
            "env_configured": {name: bool(os.getenv(name, "").strip()) for name in env_names},
        }

    def run_stage(self, name: str) -> dict[str, Any]:
        if name != "preflight" and self._is_completed(name):
            return json.loads(self._state(name).read_text(encoding="utf-8"))
        if name == "preflight":
            result = preflight(self.config, self.run_root)
            return {**result, "stage": "preflight"}
        spec = self._stage(name)
        command_record = self._command_record(spec)
        stage_dir = self.run_root / "stages"
        stage_dir.mkdir(parents=True, exist_ok=True)
        log_path = stage_dir / f"{name}.log"
        started = time.perf_counter()
        started_at = utc_now()
        state: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "stage": name,
            "status": "planned" if self.dry_run else "running",
            "started_at": started_at,
            "command": command_record,
            "log": str(log_path),
            "resource_contract": self.config.get("resource_contract", {}),
        }
        _json(self._state(name), state)
        if self.dry_run:
            state.update({"status": "dry_run", "finished_at": utc_now(), "outputs": [tree_inventory(path) for path in spec.output_paths]})
            _json(self._state(name), state)
            return state
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in self.config.get("stage_env", {}).items()})
        for name_key in spec.env_names:
            if name_key in os.environ:
                env[name_key] = os.environ[name_key]
        spec.cwd.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{started_at}] $ {shlex.join(spec.command)}\n")
            process = subprocess.Popen(spec.command, cwd=spec.cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            return_code = process.wait()
        state.update({
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "finished_at": utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "outputs": [tree_inventory(path) for path in spec.output_paths],
        })
        _json(self._state(name), state)
        if return_code != 0:
            raise RuntimeError(f"industrial stage {name} failed with exit code {return_code}; see {log_path}")
        return state

    def run(self, stages: Iterable[str]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for name in stages:
            result = self.run_stage(name)
            results.append(result)
            if result.get("status") == "FAIL" or result.get("status") == "failed":
                break
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed" if results and all(item.get("status") in {"PASS", "completed", "dry_run"} for item in results) else "failed",
            "stages": results,
            "run_root": str(self.run_root),
            "finished_at": utc_now(),
            "paper_scale_claimed": False,
            "teacher_substitution": self.config.get("provider", {}).get("teacher_substitution"),
        }
        _json(self.run_root / "run_summary.json", summary)
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable official AgenticQwen pipeline control plane")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", *DEFAULT_STAGE_ORDER), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("PYTHON_BIN", sys.executable)
    os.environ.setdefault("AGENTICQWEN_RUN_ROOT", str(args.run_root.resolve()))
    config = load_config(args.config.resolve())
    runner = IndustrialRunner(config, args.run_root, dry_run=args.dry_run)
    stages = DEFAULT_STAGE_ORDER if args.stage == "all" else (args.stage,)
    result = runner.run(stages)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
