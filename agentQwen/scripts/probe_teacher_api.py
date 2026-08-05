#!/usr/bin/env python3
"""Probe the teacher endpoint and persist a secret-free reliability record."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--stop-after-success", action="store_true")
    args = parser.parse_args()
    if args.attempts < 1:
        raise ValueError("attempts must be positive")

    api_key = args.key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("teacher key file is empty")
    url = f"{args.api_base.rstrip('/')}/chat/completions"
    attempts = []
    for attempt in range(1, args.attempts + 1):
        started = time.monotonic()
        record = {"attempt": attempt}
        try:
            response = httpx.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": "Reply with exactly OK"}],
                    "max_tokens": 8,
                    "stream": False,
                },
                timeout=90,
            )
            record["status_code"] = response.status_code
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"] or ""
            record.update(
                {
                    "ok": bool(content),
                    "response_chars": len(content),
                    "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            )
        except Exception as exc:  # pragma: no cover - live network diagnostic
            record.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:500]})
        record["latency_seconds"] = round(time.monotonic() - started, 3)
        attempts.append(record)
        if record.get("ok") and args.stop_after_success:
            break

    successes = sum(bool(item.get("ok")) for item in attempts)
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if successes else "FAIL",
        "api_base": args.api_base,
        "model": args.model,
        "secret_persisted": False,
        "successes": successes,
        "attempt_count": len(attempts),
        "max_attempts": args.attempts,
        "stop_after_success": args.stop_after_success,
        "attempts": attempts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "successes": successes, "attempts": len(attempts)}))
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
