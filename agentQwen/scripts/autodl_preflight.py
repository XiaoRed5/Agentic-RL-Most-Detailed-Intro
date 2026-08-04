from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import accelerate
import bitsandbytes
import datasets
import peft
import torch
import transformers
import trl
from trl import GRPOConfig, GRPOTrainer


trainer_parameters = set(inspect.signature(GRPOTrainer.__init__).parameters)
config_parameters = set(inspect.signature(GRPOConfig.__init__).parameters)
required_trainer = {"environment_factory", "processing_class", "peft_config"}
required_config = {
    "max_steps",
    "num_generations",
    "num_generations_eval",
    "max_completion_length",
    "max_tool_calling_iterations",
    "chat_template_kwargs",
    "mask_truncated_completions",
    "loss_type",
}
missing_trainer = sorted(required_trainer - trainer_parameters)
missing_config = sorted(required_config - config_parameters)
if missing_trainer or missing_config:
    raise RuntimeError(
        f"TRL API mismatch: trainer={missing_trainer}; config={missing_config}"
    )
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

gpu = torch.cuda.get_device_properties(0)
disk = shutil.disk_usage(Path("/root/autodl-tmp"))
result = {
    "status": "PASS",
    "versions": {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "datasets": datasets.__version__,
        "bitsandbytes": bitsandbytes.__version__,
    },
    "gpu": {
        "name": gpu.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "memory_bytes": gpu.total_memory,
    },
    "data_disk": {
        "total_bytes": disk.total,
        "used_bytes": disk.used,
        "free_bytes": disk.free,
    },
    "trl_api": {
        "trainer_required": sorted(required_trainer),
        "config_required": sorted(required_config),
    },
}
print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
