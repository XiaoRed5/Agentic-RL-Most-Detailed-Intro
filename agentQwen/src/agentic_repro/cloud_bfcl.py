from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


MODEL_ID = "Qwen/Qwen3-8B"
BFCL_MODEL = "Qwen/Qwen3-8B-FC"
BFCL_CATEGORIES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)
BFCL_FILES = {
    "multi_turn_base": "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
}


def _model_load_path() -> str:
    return os.getenv("AGENTICQWEN_MODEL_PATH", MODEL_ID)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _bfcl_package_info(bfcl_python: Path) -> dict[str, str]:
    probe = """
import importlib.metadata
import json
from pathlib import Path
import bfcl_eval

print(json.dumps({
    "data_dir": str(Path(bfcl_eval.__path__[0]) / "data"),
    "version": importlib.metadata.version("bfcl-eval"),
}))
"""
    completed = subprocess.run(
        [str(bfcl_python), "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Could not inspect BFCL package with {bfcl_python}: {completed.stderr.strip()}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not inspect BFCL package with {bfcl_python}: {completed.stdout!r}"
        ) from exc
    if not result.get("data_dir") or not result.get("version"):
        raise RuntimeError(f"Incomplete BFCL package metadata: {result}")
    return {"data_dir": str(result["data_dir"]), "version": str(result["version"])}


def _selected_ids(count_per_category: int, *, data_dir: Path) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for category, file_name in BFCL_FILES.items():
        ids: list[str] = []
        with (data_dir / file_name).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                ids.append(str(json.loads(line)["id"]))
                if len(ids) == count_per_category:
                    break
        if len(ids) != count_per_category:
            raise RuntimeError(
                f"BFCL category {category} yielded {len(ids)} IDs, expected {count_per_category}"
            )
        selected[category] = ids
    return selected


def build_vllm_command(
    *,
    adapter_dir: Path | None,
    port: int,
    model_cache: Path,
) -> list[str]:
    # The official BFCL runner only needs an OpenAI-compatible completions
    # endpoint.  On a minimal research image where vLLM is not installed we
    # can launch the auditable Transformers+NF4+PEFT server shipped with this
    # project.  The benchmark/evaluator path remains unchanged.
    server_script = os.getenv("AGENTICQWEN_BFCL_SERVER_SCRIPT")
    if server_script:
        command = [
            sys.executable,
            server_script,
            "--model-path",
            _model_load_path(),
            "--port",
            str(port),
        ]
        if adapter_dir is not None:
            command.extend(["--adapter-path", str(adapter_dir)])
        return command
    command = [
        "vllm",
        "serve",
        _model_load_path(),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--download-dir",
        str(model_cache),
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.88",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "qwen3",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
    ]
    if adapter_dir is None:
        command.extend(["--served-model-name", MODEL_ID])
    else:
        command.extend(
            [
                "--served-model-name",
                "base-user",
                "--enable-lora",
                "--max-lora-rank",
                "16",
                "--lora-modules",
                f"{MODEL_ID}={adapter_dir}",
            ]
        )
    return command


def _wait_for_server(process: subprocess.Popen[str], port: int, timeout: int = 600) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    last_error = "not started"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode}: {last_error}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become healthy within {timeout}s: {last_error}")


@contextmanager
def _server(
    *,
    adapter_dir: Path | None,
    log_path: Path,
    model_cache: Path,
    port: int = 8000,
) -> Iterator[list[str]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_vllm_command(
        adapter_dir=adapter_dir,
        port=port,
        model_cache=model_cache,
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            _wait_for_server(process, port)
            yield command
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


def _run(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
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


def _result_episode_count(result_dir: Path) -> int:
    count = 0
    for path in result_dir.rglob("*_result.json"):
        with path.open("r", encoding="utf-8") as handle:
            count += sum(1 for line in handle if line.strip())
    return count


def _score_rows(score_dir: Path) -> list[dict[str, str]]:
    path = score_dir / "data_overall.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def run_bfcl_smoke(
    *,
    run_root: Path,
    tasks_per_category: int,
    model_cache: Path,
    bfcl_python: Path,
    bfcl_bin: Path,
) -> dict[str, Any]:
    verification = json.loads((run_root / "verification.json").read_text(encoding="utf-8"))
    if verification.get("overall_status") != "PASS":
        raise RuntimeError("Refusing BFCL run because curriculum verification did not PASS")
    adapter_dir = run_root / "stage2" / "adapter"
    if not adapter_dir.is_dir():
        raise RuntimeError(f"Stage-2 adapter is missing: {adapter_dir}")

    benchmark_root = run_root / "benchmarks" / "bfcl_smoke"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    if not bfcl_python.is_file():
        raise RuntimeError(f"BFCL Python is missing: {bfcl_python}")
    if not bfcl_bin.is_file():
        raise RuntimeError(f"BFCL CLI is missing: {bfcl_bin}")
    bfcl_info = _bfcl_package_info(bfcl_python)
    selected = _selected_ids(
        tasks_per_category,
        data_dir=Path(bfcl_info["data_dir"]),
    )
    expected = sum(len(values) for values in selected.values())
    variants: dict[str, Any] = {}

    for variant, adapter in (("base", None), ("stage2_adapter", adapter_dir)):
        variant_root = benchmark_root / variant
        result_dir = variant_root / "result"
        score_dir = variant_root / "score"
        variant_root.mkdir(parents=True, exist_ok=True)
        _write_json(variant_root / "test_case_ids_to_generate.json", selected)
        env = os.environ.copy()
        env.update(
            {
                "BFCL_PROJECT_ROOT": str(variant_root),
                "REMOTE_OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
                "REMOTE_OPENAI_API_KEY": "EMPTY",
                "REMOTE_OPENAI_TOKENIZER_PATH": _model_load_path(),
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        started = time.perf_counter()
        with _server(
            adapter_dir=adapter,
            log_path=variant_root / "vllm.log",
            model_cache=model_cache,
        ) as server_command:
            generate = [
                str(bfcl_bin),
                "generate",
                "--model",
                BFCL_MODEL,
                "--test-category",
                ",".join(BFCL_CATEGORIES),
                "--skip-server-setup",
                "--num-threads",
                "1",
                "--include-input-log",
                "--allow-overwrite",
                "--run-ids",
                "--result-dir",
                str(result_dir),
            ]
            evaluate = [
                str(bfcl_bin),
                "evaluate",
                "--model",
                BFCL_MODEL,
                "--test-category",
                ",".join(BFCL_CATEGORIES),
                "--result-dir",
                str(result_dir),
                "--score-dir",
                str(score_dir),
                "--partial-eval",
            ]
            _run(generate, cwd=variant_root, env=env, log_path=variant_root / "bfcl.log")
            _run(evaluate, cwd=variant_root, env=env, log_path=variant_root / "bfcl.log")
        episode_count = _result_episode_count(result_dir)
        score_rows = _score_rows(score_dir)
        variants[variant] = {
            "status": "PASS" if episode_count >= expected and score_rows else "FAIL",
            "expected_episodes": expected,
            "result_episodes": episode_count,
            "score_rows": score_rows,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "server_command": server_command,
        }

    checks = [
        {
            "name": f"{variant} produced all requested BFCL episodes",
            "status": values["status"],
            "detail": f"expected={values['expected_episodes']}; actual={values['result_episodes']}; score_rows={len(values['score_rows'])}",
        }
        for variant, values in variants.items()
    ]
    overall = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    manifest = {
        "schema_version": 1,
        "overall_status": overall,
        "evidence_class": (
            "official_bfcl_cloud_observed"
            if os.getenv("AGENTICQWEN_ORCHESTRATOR") in {"modal", "autodl"}
            else "official_bfcl_observed"
        ),
        "scope": "official BFCL-V4 multi-turn smoke; not paper-scale",
        "official_version": bfcl_info["version"],
        "runtime_isolation": {
            "vllm_python": os.path.realpath(sys.executable),
            "bfcl_python": str(bfcl_python),
            "bfcl_bin": str(bfcl_bin),
            "reason": "bfcl-eval pins NumPy 1.26 while vLLM requires NumPy 2.x",
        },
        "model_id": MODEL_ID,
        "model_load_path": _model_load_path(),
        "categories": list(BFCL_CATEGORIES),
        "tasks_per_category": tasks_per_category,
        "generation_limits": {
            "max_tokens": int(os.getenv("AGENTICQWEN_BFCL_MAX_TOKENS", "4096")),
            "reason": "Explicit smoke cap; official BFCL generation/evaluation remains unchanged.",
        },
        "selected_ids": selected,
        "variants": variants,
        "checks": checks,
    }
    _write_json(benchmark_root / "verification.json", manifest)
    manifest["artifact_inventory"] = _inventory(benchmark_root)
    _write_json(benchmark_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official BFCL-V4 smoke on base and curriculum adapter")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--tasks-per-category", type=int, default=2)
    parser.add_argument("--model-cache", type=Path, default=Path("/models/huggingface/hub"))
    parser.add_argument("--bfcl-python", type=Path, default=Path("/opt/bfcl-venv/bin/python"))
    parser.add_argument("--bfcl-bin", type=Path, default=Path("/opt/bfcl-venv/bin/bfcl"))
    args = parser.parse_args()
    if args.tasks_per_category < 1:
        raise SystemExit("--tasks-per-category must be positive")
    result = run_bfcl_smoke(
        run_root=args.run_root.resolve(),
        tasks_per_category=args.tasks_per_category,
        model_cache=args.model_cache.resolve(),
        # Do not use Path.resolve() here: venv/bin/python is normally a symlink,
        # and following it would silently escape the isolated BFCL environment.
        bfcl_python=Path(os.path.abspath(args.bfcl_python)),
        bfcl_bin=Path(os.path.abspath(args.bfcl_bin)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
