#!/usr/bin/env python3
"""Small OpenAI-compatible completion server for the official BFCL runner.

This intentionally uses the same Transformers + NF4 + PEFT loading path as the
curriculum trainer.  It is a single-request-at-a-time server for an auditable
smoke benchmark; production serving should use vLLM/SGLang instead.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def load_model(model_path: str, adapter_path: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = torch.bfloat16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant,
        torch_dtype=dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class CompletionHandler(BaseHTTPRequestHandler):
    model: Any = None
    tokenizer: Any = None
    model_name: str = "Qwen/Qwen3-8B"
    lock = threading.Lock()

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[hf-server] " + (fmt % args), flush=True)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            self._json(
                200,
                {"object": "list", "data": [{"id": self.model_name, "object": "model"}]},
            )
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/completions":
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            prompt = body.get("prompt", "")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            # BFCL's OSS handler requests up to 4096 tokens.  A smoke run is
            # intentionally bounded so a reasoning model cannot spend many
            # minutes on one malformed tool call; the cap is explicit in the
            # benchmark manifest and does not affect training.
            max_tokens = min(
                int(body.get("max_tokens", 256)),
                int(os.getenv("AGENTICQWEN_BFCL_MAX_TOKENS", "4096")),
            )
            temperature = float(body.get("temperature", 0.0))
            import torch

            with self.lock, torch.inference_mode():
                encoded = self.tokenizer(str(prompt), return_tensors="pt")
                device = next(self.model.parameters()).device
                encoded = {k: v.to(device) for k, v in encoded.items()}
                input_len = int(encoded["input_ids"].shape[-1])
                kwargs: dict[str, Any] = {
                    **encoded,
                    "max_new_tokens": max_tokens,
                    "do_sample": temperature > 0,
                    "use_cache": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if temperature > 0:
                    kwargs.update({"temperature": temperature, "top_p": float(body.get("top_p", 0.95))})
                output = self.model.generate(**kwargs)
                text = self.tokenizer.decode(output[0][input_len:], skip_special_tokens=False)
                prompt_tokens = input_len
                completion_tokens = int(output.shape[-1] - input_len)
            self._json(
                200,
                {
                    "id": "cmpl-" + uuid.uuid4().hex,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                },
            )
        except Exception as exc:  # keep the official runner's error log useful
            self._json(500, {"error": {"message": repr(exc), "type": type(exc).__name__}})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    args = parser.parse_args()
    print(f"[hf-server] loading base={args.model_path} adapter={args.adapter_path}", flush=True)
    CompletionHandler.model, CompletionHandler.tokenizer = load_model(args.model_path, args.adapter_path)
    CompletionHandler.model_name = args.model_name
    print(f"[hf-server] ready on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), CompletionHandler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
