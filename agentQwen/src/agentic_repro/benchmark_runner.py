from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BFCL_FILE_NAMES = {
    "multi_turn_base": "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
}


def _resolve(project_dir: Path, value: str) -> Path:
    path = Path(value)
    # Keep virtual-environment interpreter symlinks intact. Path.resolve() would
    # collapse ``venv/bin/python`` to the base runtime and lose installed deps.
    candidate = path if path.is_absolute() else project_dir / path
    return Path(os.path.abspath(candidate))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; see {log_path}"
        )


def _bfcl_data_dir(benchmark_python: Path) -> Path:
    command = [
        str(benchmark_python),
        "-c",
        (
            "from pathlib import Path; import bfcl_eval; "
            "print(Path(bfcl_eval.__path__[0]) / 'data')"
        ),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return Path(result.stdout.strip()).resolve()


def select_bfcl_ids(
    data_dir: Path,
    categories: Iterable[str],
    count_per_category: int,
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for category in categories:
        file_name = BFCL_FILE_NAMES[category]
        path = data_dir / file_name
        ids: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                ids.append(str(json.loads(line)["id"]))
                if len(ids) >= count_per_category:
                    break
        if len(ids) != count_per_category:
            raise RuntimeError(
                f"BFCL {category} only provided {len(ids)} IDs; expected {count_per_category}"
            )
        selected[category] = ids
    return selected


def build_bfcl_commands(
    *,
    benchmark_python: Path,
    model_registry_name: str,
    model_path: Path,
    categories: list[str],
    partial_eval: bool,
    num_threads: int,
) -> list[list[str]]:
    bfcl = str(benchmark_python.parent / "bfcl")
    category_arg = ",".join(categories)
    generate = [
        bfcl,
        "generate",
        "--model",
        model_registry_name,
        "--test-category",
        category_arg,
        "--skip-server-setup",
        "--local-model-path",
        str(model_path),
        "--num-threads",
        str(num_threads),
        "--include-input-log",
        "--allow-overwrite",
    ]
    if partial_eval:
        generate.append("--run-ids")
    evaluate = [
        bfcl,
        "evaluate",
        "--model",
        model_registry_name,
        "--test-category",
        category_arg,
    ]
    if partial_eval:
        evaluate.append("--partial-eval")
    return [generate, evaluate]


def build_tau2_commands(
    *,
    tau2_executable: Path,
    domains: Iterable[str],
    agent_model: str,
    user_model: str,
    api_base: str,
    user_api_base: str,
    user_api_key: str,
    num_tasks: int | None,
    num_trials: int,
    max_steps: int,
    max_concurrency: int,
    seed: int,
    variant: str,
) -> list[list[str]]:
    agent_args = json.dumps(
        {"api_base": api_base, "api_key": "EMPTY", "temperature": 0.0},
        separators=(",", ":"),
    )
    user_args = json.dumps(
        {
            "api_base": user_api_base,
            "api_key": user_api_key,
            "temperature": 0.7,
        },
        separators=(",", ":"),
    )
    commands: list[list[str]] = []
    for domain in domains:
        command = [
            str(tau2_executable),
            "run",
            "--domain",
            domain,
            "--agent-llm",
            agent_model,
            "--agent-llm-args",
            agent_args,
            "--user-llm",
            user_model,
            "--user-llm-args",
            user_args,
            "--num-trials",
            str(num_trials),
            "--max-steps",
            str(max_steps),
            "--max-concurrency",
            str(max_concurrency),
            "--seed",
            str(seed),
            "--save-to",
            f"agenticqwen_{variant}_{domain}",
        ]
        if num_tasks is not None:
            command.extend(["--num-tasks", str(num_tasks)])
        commands.append(command)
    return commands


@dataclass
class ManagedMLXServer:
    python: Path
    model_path: Path
    adapter_path: Path | None
    host: str
    port: int
    temperature: float
    max_tokens: int
    enable_thinking: bool
    startup_timeout_seconds: int
    log_path: Path
    process: subprocess.Popen[str] | None = None
    log_handle: Any = None

    @property
    def api_base(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def __enter__(self) -> "ManagedMLXServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        command = [
            str(self.python),
            "-m",
            "mlx_lm",
            "server",
            "--model",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--temp",
            str(self.temperature),
            "--max-tokens",
            str(self.max_tokens),
            "--chat-template-args",
            json.dumps({"enable_thinking": self.enable_thinking}),
        ]
        if self.adapter_path is not None:
            command.extend(["--adapter-path", str(self.adapter_path)])
        self.process = subprocess.Popen(
            command,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.startup_timeout_seconds
        health_url = self.api_base + "/models"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"MLX server exited early; see {self.log_path}")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        return self
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        raise TimeoutError(f"MLX server was not ready after {self.startup_timeout_seconds}s")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        if self.log_handle is not None:
            self.log_handle.close()


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def run(config_path: Path, *, profile: str, benchmark: str, variants: list[str], dry_run: bool) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_dir = config_path.parent.parent.resolve()
    output_root = project_dir / "artifacts" / "benchmarks"
    output_root.mkdir(parents=True, exist_ok=True)

    model_path = _resolve(project_dir, config["model_path"])
    adapter_path = _resolve(project_dir, config["adapter_path"])
    mlx_python = _resolve(project_dir, config["mlx_python"])
    benchmark_python = _resolve(project_dir, config["benchmark_python"])
    tau2_repo = _resolve(project_dir, config["tau2_repo"])
    api_base = f"http://{config['server']['host']}:{config['server']['port']}/v1"

    bfcl_config = config["bfcl"]
    bfcl_count = (
        bfcl_config["smoke_tasks_per_category"]
        if profile == "smoke"
        else bfcl_config["paper_tasks_per_category"]
    )
    bfcl_ids: dict[str, list[str]] = {}
    if benchmark in {"all", "bfcl"}:
        bfcl_ids = select_bfcl_ids(
            _bfcl_data_dir(benchmark_python),
            bfcl_config["categories"],
            bfcl_count,
        )

    tau_config = config["tau2"]
    tau_num_tasks = (
        tau_config["smoke_num_tasks"]
        if profile == "smoke"
        else tau_config["paper_num_tasks"]
    )
    tau_num_trials = (
        tau_config["smoke_num_trials"]
        if profile == "smoke"
        else tau_config["paper_num_trials"]
    )
    user_model = (
        tau_config["local_user_model"]
        if profile == "smoke"
        else os.getenv(tau_config["paper_user_model_env"], "MISSING_REQUIRED_USER_MODEL")
    )
    user_api_base = (
        api_base
        if profile == "smoke"
        else os.getenv(
            tau_config["paper_user_api_base_env"],
            "MISSING_REQUIRED_USER_API_BASE",
        )
    )
    user_api_key = (
        "EMPTY"
        if profile == "smoke"
        else os.getenv(tau_config["paper_user_api_key_env"], "")
    )

    plan_rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_root = output_root / profile / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        adapter = None if variant == "base" else adapter_path
        commands: dict[str, list[list[str]]] = {}
        if benchmark in {"all", "bfcl"}:
            bfcl_root = variant_root / "bfcl"
            bfcl_root.mkdir(parents=True, exist_ok=True)
            (bfcl_root / "test_case_ids_to_generate.json").write_text(
                json.dumps(bfcl_ids, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            commands["bfcl"] = build_bfcl_commands(
                benchmark_python=benchmark_python,
                model_registry_name=bfcl_config["model_registry_name"],
                model_path=model_path,
                categories=bfcl_config["categories"],
                partial_eval=profile == "smoke",
                num_threads=bfcl_config["num_threads"],
            )
        if benchmark in {"all", "tau2"}:
            commands["tau2"] = build_tau2_commands(
                tau2_executable=benchmark_python.parent / "tau2",
                domains=tau_config["domains"],
                agent_model="openai/Qwen3-8B",
                user_model=user_model,
                api_base=api_base,
                user_api_base=user_api_base,
                user_api_key=user_api_key,
                num_tasks=tau_num_tasks,
                num_trials=tau_num_trials,
                max_steps=tau_config["max_steps"],
                max_concurrency=tau_config["max_concurrency"],
                seed=tau_config["seed"],
                variant=variant,
            )
        plan_rows.append(
            {
                "variant": variant,
                "adapter_path": None if adapter is None else str(adapter),
                "commands": commands,
            }
        )

        if dry_run:
            continue
        if profile == "paper" and (
            user_model == "MISSING_REQUIRED_USER_MODEL"
            or user_api_base == "MISSING_REQUIRED_USER_API_BASE"
        ):
            raise RuntimeError(
                "Set "
                f"{tau_config['paper_user_model_env']} and "
                f"{tau_config['paper_user_api_base_env']} before a paper-profile TAU-2 run"
            )
        with ManagedMLXServer(
            python=mlx_python,
            model_path=model_path,
            adapter_path=adapter,
            host=config["server"]["host"],
            port=config["server"]["port"],
            temperature=config["server"]["temperature"],
            max_tokens=config["server"]["max_tokens"],
            enable_thinking=config["server"]["enable_thinking"],
            startup_timeout_seconds=config["server"]["startup_timeout_seconds"],
            log_path=variant_root / "mlx_server.log",
        ):
            env = os.environ.copy()
            env.update(
                {
                    "BFCL_PROJECT_ROOT": str(variant_root / "bfcl"),
                    "REMOTE_OPENAI_BASE_URL": api_base,
                    "REMOTE_OPENAI_API_KEY": "EMPTY",
                    "REMOTE_OPENAI_TOKENIZER_PATH": str(model_path),
                    "TAU2_DATA_DIR": str(tau2_repo / "data" / "tau2"),
                }
            )
            for command in commands.get("bfcl", []):
                _run(
                    command,
                    cwd=variant_root / "bfcl",
                    env=env,
                    log_path=variant_root / "bfcl.log",
                )
            for command in commands.get("tau2", []):
                _run(
                    command,
                    cwd=tau2_repo,
                    env=env,
                    log_path=variant_root / "tau2.log",
                )

    manifest = {
        "schema_version": 1,
        "status": "PLANNED_NOT_RUN" if dry_run else "COMPLETED",
        "profile": profile,
        "benchmark": benchmark,
        "variants": variants,
        "official_versions": {
            "bfcl_eval": bfcl_config["package_version"],
            "tau2_git_tag": tau_config["git_tag"],
        },
        "scope": {
            "bfcl_tasks_per_category": bfcl_count if bfcl_ids else 0,
            "bfcl_total_tasks": sum(len(ids) for ids in bfcl_ids.values()),
            "tau2_num_tasks_per_domain": tau_num_tasks,
            "tau2_num_trials": tau_num_trials,
            "tau2_user_model": user_model,
            "tau2_user_api_base": user_api_base,
        },
        "plan": plan_rows,
        "artifact_inventory": [] if dry_run else _artifact_inventory(output_root / profile),
        "comparability_notes": [
            "BFCL uses the official bfcl-eval generator and exact-match/state evaluator.",
            "TAU-2 smoke uses the local Qwen3-8B as user simulator and is not paper-comparable.",
            "TAU-2 paper profile requires an explicitly configured external user simulator and Avg@4.",
            "The local adapter was trained on action-masked next-tool decisions, not unrestricted BFCL/TAU-2 trajectories.",
        ],
    }
    manifest_path = output_root / (
        "planned_manifest.json" if dry_run else f"run_manifest_{profile}.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run official BFCL-V4 and TAU-2 evaluators against local MLX models"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--benchmark", choices=("all", "bfcl", "tau2"), default="all")
    parser.add_argument("--variants", default="base,adapter")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if not variants or any(item not in {"base", "adapter"} for item in variants):
        raise SystemExit("--variants must be a comma-separated subset of base,adapter")
    manifest = run(
        args.config.resolve(),
        profile=args.profile,
        benchmark=args.benchmark,
        variants=variants,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
