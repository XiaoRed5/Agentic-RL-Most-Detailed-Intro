from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .flywheel import train_flywheel
from .schemas import Check


def verify(config: dict, project_dir: Path) -> dict:
    artifacts = project_dir / config["paths"]["artifacts"]
    checks: list[Check] = []

    required = ["plan.json", "paper_spec.json", "tasks.jsonl", "metrics.json", "smoke_policy.json", "qwen3_8b_inference.json"]
    missing = [name for name in required if not (artifacts / name).exists()]
    checks.append(Check("artifact_completeness", "PASS" if not missing else "FAIL", f"missing={missing}", {"required": required}))

    metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8")) if not missing else {}
    if metrics:
        values = [metrics["baseline"]["overall"]["mean_reward"], metrics["final"]["overall"]["mean_reward"]]
        values.extend(item["mean_reward"] for item in metrics["rounds"])
        reward_ok = all(0.0 <= value <= 1.0 for value in values)
        checks.append(Check("reward_bounds", "PASS" if reward_ok else "FAIL", f"checked {len(values)} aggregate rewards", {"values": values}))

        baseline = metrics["baseline"]["overall"]["mean_reward"]
        final = metrics["final"]["overall"]["mean_reward"]
        improvement = round(final - baseline, 6)
        min_improvement = float(config["gates"]["min_improvement"])
        checks.append(Check(
            "learning_signal",
            "PASS" if improvement >= min_improvement else "FAIL",
            f"baseline={baseline:.3f}, final={final:.3f}, delta={improvement:.3f}",
            {"minimum_delta": min_improvement},
        ))
        min_final = float(config["gates"]["min_final_reward"])
        checks.append(Check("final_reward_gate", "PASS" if final >= min_final else "FAIL", f"final={final:.3f}, gate={min_final:.3f}"))
        levels = set(metrics["final"]["by_level"])
        checks.append(Check("behavior_tree_coverage", "PASS" if levels == {"0", "1", "2"} else "FAIL", f"levels={sorted(levels)}"))

        if config["gates"].get("require_determinism", False):
            _, rerun = train_flywheel(config)
            rerun_final = rerun["final"]["overall"]
            deterministic = rerun_final == metrics["final"]["overall"]
            checks.append(Check("deterministic_rerun", "PASS" if deterministic else "FAIL", f"rerun={rerun_final}", {"recorded": metrics["final"]["overall"]}))

    qwen = json.loads((artifacts / "qwen3_8b_inference.json").read_text(encoding="utf-8")) if (artifacts / "qwen3_8b_inference.json").exists() else {}
    require_qwen = bool(config["gates"].get("require_qwen_inference", False))
    completed = qwen.get("status") == "completed"
    qwen_status = "PASS" if completed else ("FAIL" if require_qwen else "WARN")
    checks.append(Check(
        "local_qwen3_8b_inference",
        qwen_status,
        f"status={qwen.get('status', 'missing')}, backend={qwen.get('backend', 'n/a')}",
        {"network_fallback": qwen.get("network_fallback"), "model_id": qwen.get("model_id")},
    ))
    if completed:
        provenance_ok = qwen.get("backend") == "MLX" and qwen.get("network_fallback") is False and "Qwen3-8B" in qwen.get("model_id", "")
        checks.append(Check("qwen_provenance", "PASS" if provenance_ok else "FAIL", f"model={qwen.get('model_id')}, quantization={qwen.get('quantization')}"))
        trace_ok = qwen.get("scenario_count", 0) >= 3 and all(item.get("model_turns") for item in qwen.get("results", []))
        checks.append(Check("qwen_trace_retention", "PASS" if trace_ok else "FAIL", f"scenarios={qwen.get('scenario_count', 0)}"))

    failures = [check for check in checks if check.status == "FAIL"]
    result = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS" if not failures else "FAIL",
        "checks": [check.__dict__ for check in checks],
        "summary": {
            "pass": sum(check.status == "PASS" for check in checks),
            "warn": sum(check.status == "WARN" for check in checks),
            "fail": len(failures),
        },
    }
    (artifacts / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

