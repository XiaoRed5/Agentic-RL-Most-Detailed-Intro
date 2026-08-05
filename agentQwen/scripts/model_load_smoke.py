#!/usr/bin/env python3
"""Load the real Qwen snapshot with the training quantization path.

This is deliberately separate from the trainer: a complete file transfer is
not evidence that Transformers, bitsandbytes, tokenizer, and the LoRA target
modules can all be constructed on the target GPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    import torch
    import transformers
    import bitsandbytes
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = torch.bfloat16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        quantization_config=quant,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        device_map={"": 0},
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer("Use the tools to inspect the environment before acting.", return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=8, do_sample=False)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "model_path": str(args.model_path.resolve()),
        "model_path_sha256": _sha256(args.model_path / "config.json"),
        "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3),
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_lora_parameters": trainable,
        "tokenizer_vocab_size": len(tokenizer),
        "prompt_tokens": int(encoded["input_ids"].shape[-1]),
        "generated_tokens": int(generated.shape[-1] - encoded["input_ids"].shape[-1]),
        "seconds": round(time.perf_counter() - started, 3),
        "platform": platform.platform(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
