from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dashscope_policy import DashScopeFunctionPolicy, initial_messages, text_content
from .long_horizon_env import DuplicateChargeEnvironment, ScriptedSupportUser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def event(
    events: list[dict[str, Any]],
    *,
    role: str,
    event_type: str,
    content: str = "",
    **extra: Any,
) -> None:
    events.append(
        {
            "event_id": len(events) + 1,
            "role": role,
            "event_type": event_type,
            "content": content,
            **extra,
        }
    )


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    return [call for call in calls if isinstance(call, dict)]


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("function", {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    parsed = json.loads(raw or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return parsed


def _usage_total(api_calls: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for call in api_calls:
        for key, value in call.get("usage", {}).items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def run_trajectory(config: dict[str, Any]) -> dict[str, Any]:
    env = DuplicateChargeEnvironment()
    user = ScriptedSupportUser()
    policy_config = config["policy"]
    policy = DashScopeFunctionPolicy(
        model=policy_config["model"],
        base_http_api_url=policy_config["base_http_api_url"],
        seed=int(policy_config.get("seed", 260421590)),
        temperature=float(policy_config.get("temperature", 0.1)),
        max_tokens=int(policy_config.get("max_tokens", 1024)),
    )
    messages = initial_messages(user.initial_message)
    events: list[dict[str, Any]] = []
    api_calls: list[dict[str, Any]] = []
    event(
        events,
        role="user",
        event_type="user_message",
        content=user.initial_message,
        state_after=env.snapshot(),
    )
    env.record_user_message(user.initial_message)
    final_answer = ""
    termination_reason = "max_agent_turns"

    for agent_turn in range(1, int(config["run"].get("max_agent_turns", 16)) + 1):
        response = policy.call(messages, env.tools)
        assistant = response.message
        messages.append(assistant)
        calls = _tool_calls(assistant)
        assistant_text = text_content(assistant.get("content"))
        api_calls.append(
            {
                "agent_turn": agent_turn,
                "request_id": response.request_id,
                "latency_seconds": response.latency_seconds,
                "usage": response.usage,
                "tool_call_count": len(calls),
            }
        )
        if assistant_text:
            event(
                events,
                role="assistant",
                event_type="assistant_message",
                content=assistant_text,
                agent_turn=agent_turn,
                request_id=response.request_id,
                latency_seconds=response.latency_seconds,
                state_after=env.snapshot(),
            )
            final_answer = assistant_text

        if calls:
            for call in calls:
                function = call.get("function", {})
                tool_name = str(function.get("name", ""))
                state_before = env.snapshot()
                ledger_before = len(env.process_rewards)
                try:
                    arguments = _arguments(call)
                    tool_result = env.execute(tool_name, arguments)
                except (json.JSONDecodeError, ValueError) as exc:
                    arguments = {"_raw": function.get("arguments")}
                    tool_result = env.execute("__invalid_json__", {})
                    tool_result["error"]["message"] = str(exc)
                new_rewards = env.process_rewards[ledger_before:]
                state_after = env.snapshot()
                event(
                    events,
                    role="assistant",
                    event_type="tool_call",
                    content=f"{tool_name}({json.dumps(arguments, ensure_ascii=False, sort_keys=True)})",
                    agent_turn=agent_turn,
                    tool_call_id=call.get("id"),
                    tool_name=tool_name,
                    arguments=arguments,
                    state_before=state_before,
                    state_after=state_after,
                    reward_delta=round(sum(item.value for item in new_rewards), 4),
                    reward_events=[item.event for item in new_rewards],
                )
                event(
                    events,
                    role="tool",
                    event_type="tool_result",
                    content=json.dumps(tool_result, ensure_ascii=False, sort_keys=True),
                    agent_turn=agent_turn,
                    tool_call_id=call.get("id"),
                    tool_name=tool_name,
                    result=tool_result,
                    state_after=state_after,
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(tool_result, ensure_ascii=False),
                        "tool_call_id": call.get("id"),
                    }
                )
            continue

        if env.state.refund_id and assistant_text:
            termination_reason = "successful_final_answer"
            break

        user_reply = user.respond(assistant_text, env)
        if not user_reply:
            termination_reason = "user_simulator_stopped"
            break
        env.record_user_message(user_reply)
        messages.append({"role": "user", "content": [{"text": user_reply}]})
        event(
            events,
            role="user",
            event_type="user_message",
            content=user_reply,
            agent_turn=agent_turn,
            state_after=env.snapshot(),
        )

    verification = env.verify(final_answer)
    created_at = datetime.now(timezone.utc).isoformat()
    stable_key = f"{policy.model}|{policy.seed}|{created_at}|{len(events)}"
    trajectory_id = "traj-" + hashlib.sha256(stable_key.encode()).hexdigest()[:12]
    result = {
        "schema_version": 1,
        "trajectory_id": trajectory_id,
        "created_at": created_at,
        "status": "COMPLETE" if verification["success"] else "FAILED_VERIFICATION",
        "paper_comparable": False,
        "task": {
            "task_id": "duplicate-charge-refund-001",
            "domain": "customer_support_payments",
            "goal": "Verify and refund exactly one duplicated CNY 199 charge.",
            "initial_user_message": user.initial_message,
            "hidden_success_condition": "RF-2026-00081 refunds CHG-9002 only after explicit confirmation.",
        },
        "policy": {
            "provider": "DashScope",
            "model": policy.model,
            "base_http_api_url": policy.base_http_api_url,
            "seed": policy.seed,
            "temperature": policy.temperature,
        },
        "runtime": {
            "agent_turns": len(api_calls),
            "events": len(events),
            "tool_calls": sum(1 for item in events if item["event_type"] == "tool_call"),
            "user_messages": sum(1 for item in events if item["role"] == "user"),
            "termination_reason": termination_reason,
            "api_latency_seconds": round(sum(item["latency_seconds"] for item in api_calls), 4),
            "usage": _usage_total(api_calls),
        },
        "events": events,
        "api_calls": api_calls,
        "final_answer": final_answer,
        "verification": verification,
        "claim_boundary": (
            "This is one real API-driven multi-turn tool trajectory in a deterministic mini environment. "
            "It is not a GRPO training result or a TAU-2/BFCL score."
        ),
    }
    serialized = json.dumps(result, ensure_ascii=False)
    if "sk-" in serialized.lower():
        raise RuntimeError("Secret-like text detected in trajectory artifact; refusing to write it.")
    return result


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Long-Horizon Agentic Trajectory",
        "",
        f"- **Trajectory:** `{result['trajectory_id']}`",
        f"- **Model:** `{result['policy']['model']}` via DashScope",
        f"- **Status:** `{result['status']}`",
        f"- **Agent turns / tool calls / events:** {result['runtime']['agent_turns']} / {result['runtime']['tool_calls']} / {result['runtime']['events']}",
        f"- **Outcome / process / combined reward:** {result['verification']['outcome_reward']:.2f} / {result['verification']['process_reward']:.2f} / {result['verification']['combined_reward']:.2f}",
        "",
        "## Task",
        "",
        result["task"]["goal"],
        "",
        "## Full trajectory",
        "",
        "| # | Role | Event | Content | Reward Δ |",
        "|---:|---|---|---|---:|",
    ]
    for item in result["events"]:
        content = item.get("content", "").replace("|", "\\|").replace("\n", "<br>")
        if len(content) > 500:
            content = content[:497] + "..."
        lines.append(
            f"| {item['event_id']} | {item['role']} | {item['event_type']} | {content} | {item.get('reward_delta', '')} |"
        )
    lines.extend(["", "## Verifier", ""])
    for check in result["verification"]["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- **{marker}** `{check['name']}` — {check['detail']}")
    lines.extend(
        [
            "",
            "## Final answer",
            "",
            result["final_answer"],
            "",
            "## Claim boundary",
            "",
            result["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one stateful DashScope tool trajectory.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_json(args.config)
    result = run_trajectory(config)
    output = args.output or Path(config["artifacts"]["trajectory_json"])
    write_json(output, result)
    markdown_path = Path(config["artifacts"]["trajectory_markdown"])
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "trajectory_id": result["trajectory_id"],
                "agent_turns": result["runtime"]["agent_turns"],
                "tool_calls": result["runtime"]["tool_calls"],
                "events": result["runtime"]["events"],
                "outcome_reward": result["verification"]["outcome_reward"],
                "json": str(output.resolve()),
                "markdown": str(markdown_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
