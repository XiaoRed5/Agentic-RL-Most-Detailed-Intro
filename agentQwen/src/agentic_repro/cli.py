from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .executor import execute
from .planner import write_plan
from .report import build_report
from .verifier import verify


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AgenticQwen plan-execute-verify reproduction")
    parser.add_argument("command", choices=["plan", "execute", "verify", "report", "all"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--qwen-model-path")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    project_dir = config_path.parent.parent
    config = load_config(config_path)

    if args.command in {"plan", "all"}:
        print(f"[planner] {write_plan(config, project_dir)}")
    if args.command in {"execute", "all"}:
        result = execute(config, project_dir, qwen_model_path=args.qwen_model_path)
        print(f"[executor] smoke_final_reward={result['metrics']['final']['overall']['mean_reward']}")
        print(f"[executor] qwen_status={result['qwen']['status']}")
    verification = None
    if args.command in {"verify", "all"}:
        verification = verify(config, project_dir)
        print(f"[verifier] {verification['overall_status']} {verification['summary']}")
    if args.command in {"report", "all"}:
        path = build_report(config, project_dir)
        print(f"[reporter] {path}")

    if verification and verification["overall_status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

