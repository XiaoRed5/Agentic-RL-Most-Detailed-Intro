#!/usr/bin/env python3
"""Reload a merged AgenticQwen checkpoint in a fresh process and persist evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model_path = args.model.resolve()
    parquet_path = args.parquet.resolve()
    frame = pd.read_parquet(parquet_path)
    if not 0 <= args.row_index < len(frame):
        raise IndexError(f"row-index {args.row_index} outside dataset of size {len(frame)}")
    prompt = list(frame.iloc[args.row_index]["prompt"])
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    inputs = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    weight_hash_path = model_path / "weights.sha256"
    payload = {
        "schema_version": 1,
        "status": "PASS" if response.strip() else "FAIL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fresh_process": True,
        "model": {
            "path": str(model_path),
            "model_type": config.model_type,
            "architectures": config.architectures,
            "weight_manifest": str(weight_hash_path) if weight_hash_path.is_file() else None,
            "weight_manifest_sha256": sha256_file(weight_hash_path)
            if weight_hash_path.is_file()
            else None,
        },
        "dataset": {
            "path": str(parquet_path),
            "sha256": sha256_file(parquet_path),
            "row_index": args.row_index,
            "task_id": str(frame.iloc[args.row_index]["extra_info"]["index"]),
            "prompt_sha256": hashlib.sha256(
                json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "generated_tokens": int(new_tokens.numel()),
            "response": response,
            "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
        },
        "secret_persisted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
