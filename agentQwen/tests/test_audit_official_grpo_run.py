from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_official_grpo_run.py"
SPEC = importlib.util.spec_from_file_location("audit_official_grpo_run", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_audit_passes_execution_and_marks_mixed_rewards(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "global_step_2" / "actor"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (run_dir / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("2")
    rollout = run_dir / "rollouts" / "2.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        "\n".join(
            json.dumps({"input": "a", "output": "b", "score": score, "step": 2})
            for score in (0.0, 1.0)
        )
        + "\n"
    )
    validation = run_dir / "validation" / "2.jsonl"
    validation.parent.mkdir(parents=True)
    validation.write_text(json.dumps({"input": "v", "output": "o", "score": 1.0}) + "\n")
    dataset = tmp_path / "dataset.json"
    environment = tmp_path / "environment.json"
    output = tmp_path / "audit.json"
    write_json(run_dir / "teacher_api_probe.json", {"status": "PASS"})
    (run_dir / "train.log").write_text(
        "step:1 - actor/grad_norm:0.0 - critic/score/mean:0.0\n"
        "step:2 - actor/grad_norm:1.484375 - critic/score/mean:0.5\n"
    )
    write_json(
        dataset,
        {
            "sources": {"official": {"rows": 37401}},
            "splits": {
                "train": {"official_rows": 512, "synthetic_rows": 0},
                "validation": {"official_rows": 64},
            },
            "policy": {"synthetic_cap": 10},
            "checks": {
                "train_validation_base_id_disjoint": True,
                "synthetic_within_cap": True,
            },
        },
    )
    write_json(environment, {"status": "PASS"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--run-dir",
            str(run_dir),
            "--dataset-manifest",
            str(dataset),
            "--environment-manifest",
            str(environment),
            "--expected-steps",
            "2",
            "--output",
            str(output),
        ],
    )

    assert MODULE.main() == 0
    audit = json.loads(output.read_text())
    assert audit["status"] == "PASS"
    assert audit["learning_signal_status"] == "PASS"
    assert audit["reward"]["unique"] == [0.0, 1.0]
    assert audit["learning_signal"]["positive_grad_norm_observed"] is True
    assert audit["optimizer"]["max_grad_norm"] == 1.484375


def test_reward_variance_without_positive_gradient_is_partial(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "global_step_1" / "actor"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (run_dir / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("1")
    (run_dir / "train.log").write_text("step:1 - actor/grad_norm:0.0\n")
    rollout = run_dir / "rollouts" / "1.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        json.dumps({"score": 0.0}) + "\n" + json.dumps({"score": 1.0}) + "\n"
    )
    validation = run_dir / "validation" / "1.jsonl"
    validation.parent.mkdir(parents=True)
    validation.write_text(json.dumps({"score": 1.0}) + "\n")
    dataset = tmp_path / "dataset.json"
    environment = tmp_path / "environment.json"
    output = tmp_path / "audit.json"
    write_json(run_dir / "teacher_api_probe.json", {"status": "PASS"})
    write_json(
        dataset,
        {
            "sources": {"official": {"rows": 1}},
            "splits": {
                "train": {"official_rows": 1, "synthetic_rows": 0},
                "validation": {"official_rows": 1},
            },
            "policy": {"synthetic_cap": 10},
            "checks": {
                "train_validation_base_id_disjoint": True,
                "synthetic_within_cap": True,
            },
        },
    )
    write_json(environment, {"status": "PASS"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--run-dir",
            str(run_dir),
            "--dataset-manifest",
            str(dataset),
            "--environment-manifest",
            str(environment),
            "--expected-steps",
            "1",
            "--output",
            str(output),
        ],
    )

    assert MODULE.main() == 0
    audit = json.loads(output.read_text())
    assert audit["status"] == "PASS"
    assert audit["learning_signal_status"] == "PARTIAL"
    assert audit["learning_signal"]["positive_grad_norm_observed"] is False
