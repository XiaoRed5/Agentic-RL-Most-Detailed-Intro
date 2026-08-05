"""One-request diagnostic for the upstream data-synthesis provider.

It prints parser-level metadata only; the model response is intentionally not
printed because it can contain user data or tool schemas.
"""
from __future__ import annotations

import os
import sys
import traceback

import yaml

sys.path.insert(0, os.environ["AGENTICQWEN_SYNTH_ROOT"])
from configuration import ModelConfiguration
from functions.tool_set_policy_gen import generate_tool_set_policy


cfg_path = sys.argv[1]
config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
step = config["step_models"]["ToolSetGenAgent"]
cfg = ModelConfiguration(
    model_name=step["model_name"],
    temperature=step["temperature"],
    max_tokens=step["max_tokens"],
    api_base=os.environ.get("AIGC_BASE_URL"),
    api_key=os.environ.get("AIGC_APP_ID"),
)
try:
    result = generate_tool_set_policy(
        cfg=cfg,
        background_info="A compliance analyst needs a realistic, state-verifiable multi-tool task.",
        normal_workflow=None,
    )
    print({"status": "ok", "tuple_len": len(result), "none_fields": [item is None for item in result]})
    for idx, item in enumerate(result):
        print(idx, type(item).__name__, len(item) if isinstance(item, str) else None)
except Exception as exc:
    print({"status": "error", "type": type(exc).__name__, "message": str(exc)[:400]})
    traceback.print_exc()
    raise
