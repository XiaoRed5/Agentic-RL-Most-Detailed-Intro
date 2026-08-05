#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${AGENTICQWEN_ROOT:-/home/hadoop-aipnlp/agenticqwen-paper-debug-20260804}"
UPSTREAM_ROOT="${AGENTICQWEN_UPSTREAM_REPO:-$RUN_ROOT/upstream/data_synth_and_rl}"
PATCH_ROOT="${AGENTICQWEN_PATCH_ROOT:-$RUN_ROOT/scripts/patches}"
PATCH_FILE="$PATCH_ROOT/agenticqwen-qwen3-clarification-fallback.patch"
ROUTING_PATCH_FILE="$PATCH_ROOT/agenticqwen-qwen3-clarification-routing-v2.patch"
STOP_HANDSHAKE_PATCH_FILE="$PATCH_ROOT/agenticqwen-qwen3-stop-handshake-v3.patch"
TARGET="$UPSTREAM_ROOT/RL/verl/experimental/agent_loop/tool_parser.py"
STOP_TERMINATION_PATCH_FILE="$PATCH_ROOT/agenticqwen-mock-user-stop-termination-v4.patch"
LOOP_TARGET="$UPSTREAM_ROOT/RL/verl/experimental/agent_loop/tool_agent_loop.py"
DETERMINISTIC_REPLAY_PATCH_FILE="$PATCH_ROOT/agenticqwen-deterministic-tool-replay-v5.patch"
MOCK_TOOL_TARGET="$UPSTREAM_ROOT/RL/my_script/tools/mock_tool.py"
MIXED_TRANSCRIPT_PATCH_FILE="$PATCH_ROOT/agenticqwen-mixed-transcript-reward-v6.patch"
MESSAGE_PARSER_TARGET="$UPSTREAM_ROOT/RL/my_script/utils/message_parser.py"
ROLE_BOUNDARY_PATCH_FILE="$PATCH_ROOT/agenticqwen-concatenated-role-boundaries-v7.patch"
MANIFEST="${AGENTICQWEN_RUNTIME_PATCH_MANIFEST:-$RUN_ROOT/artifacts/official_runtime_patches.json}"

for required in "$PATCH_FILE" "$ROUTING_PATCH_FILE" "$STOP_HANDSHAKE_PATCH_FILE" "$STOP_TERMINATION_PATCH_FILE" "$DETERMINISTIC_REPLAY_PATCH_FILE" "$MIXED_TRANSCRIPT_PATCH_FILE" "$ROLE_BOUNDARY_PATCH_FILE" "$TARGET" "$LOOP_TARGET" "$MOCK_TOOL_TARGET" "$MESSAGE_PARSER_TARGET"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing runtime patch input: $required" >&2
    exit 40
  fi
done

if grep -q "has learned AgenticQwen's private" "$TARGET"; then
  state="already_applied"
else
  patch --batch --forward -d "$UPSTREAM_ROOT" -p1 < "$PATCH_FILE"
  state="applied"
fi

if ! grep -q "has learned AgenticQwen's private" "$TARGET"; then
  echo "Clarification compatibility patch did not apply cleanly" >&2
  exit 41
fi

if grep -q "routing untagged clarification to mock_user" "$TARGET"; then
  routing_state="already_applied"
else
  patch --batch --forward -d "$UPSTREAM_ROOT" -p1 < "$ROUTING_PATCH_FILE"
  routing_state="applied"
fi

if ! grep -q "treating untagged declarative response as final" "$TARGET"; then
  if ! grep -q "routing untagged response to mock_user for STOP handshake" "$TARGET"; then
    echo "Clarification routing v2 patch did not apply cleanly" >&2
    exit 42
  fi
fi

if grep -q "routing untagged response to mock_user for STOP handshake" "$TARGET"; then
  stop_handshake_state="already_applied"
else
  patch --batch --forward -d "$UPSTREAM_ROOT" -p1 < "$STOP_HANDSHAKE_PATCH_FILE"
  stop_handshake_state="applied"
fi

if ! grep -q "routing untagged response to mock_user for STOP handshake" "$TARGET"; then
  echo "STOP handshake v3 patch did not apply cleanly" >&2
  exit 43
fi

if grep -q "Mock user emitted ###STOP; terminating completed trajectory" "$LOOP_TARGET"; then
  stop_termination_state="already_applied"
else
  patch --batch --forward -d "$UPSTREAM_ROOT" -p1 < "$STOP_TERMINATION_PATCH_FILE"
  stop_termination_state="applied"
fi

if ! grep -q "Mock user emitted ###STOP; terminating completed trajectory" "$LOOP_TARGET"; then
  echo "Mock-user STOP termination v4 patch did not apply cleanly" >&2
  exit 44
fi

if grep -q "Replaying exact official tool fixture" "$MOCK_TOOL_TARGET"; then
  deterministic_replay_state="already_applied"
else
  patch --batch --forward -d "$UPSTREAM_ROOT" -p1 < "$DETERMINISTIC_REPLAY_PATCH_FILE"
  deterministic_replay_state="applied"
fi

if ! grep -q "auditable ###STOP fallback" "$MOCK_TOOL_TARGET"; then
  echo "Deterministic tool replay v5 patch did not apply cleanly" >&2
  exit 45
fi

if grep -q "mixed serialization instead of dropping" "$MESSAGE_PARSER_TARGET"; then
  mixed_transcript_state="already_applied"
else
  patch --batch --forward -d "$UPSTREAM_ROOT" -p1 < "$MIXED_TRANSCRIPT_PATCH_FILE"
  mixed_transcript_state="applied"
fi

if ! grep -q "Agent Response" "$MESSAGE_PARSER_TARGET"; then
  echo "Mixed transcript reward v6 patch did not apply cleanly" >&2
  exit 46
fi

if grep -q "closing XML tag or final sentence" "$MESSAGE_PARSER_TARGET"; then
  role_boundary_state="already_applied"
else
  patch --batch --forward -d "$UPSTREAM_ROOT" -p1 < "$ROLE_BOUNDARY_PATCH_FILE"
  role_boundary_state="applied"
fi

if ! grep -q "pieces = re.split(r'(user|assistant)" "$MESSAGE_PARSER_TARGET"; then
  echo "Concatenated role boundary v7 patch did not apply cleanly" >&2
  exit 47
fi

mkdir -p "$(dirname "$MANIFEST")"
python3 - "$PATCH_FILE" "$ROUTING_PATCH_FILE" "$STOP_HANDSHAKE_PATCH_FILE" "$STOP_TERMINATION_PATCH_FILE" "$DETERMINISTIC_REPLAY_PATCH_FILE" "$MIXED_TRANSCRIPT_PATCH_FILE" "$ROLE_BOUNDARY_PATCH_FILE" "$TARGET" "$LOOP_TARGET" "$MOCK_TOOL_TARGET" "$MESSAGE_PARSER_TARGET" "$MANIFEST" "$state" "$routing_state" "$stop_handshake_state" "$stop_termination_state" "$deterministic_replay_state" "$mixed_transcript_state" "$role_boundary_state" <<'PY'
import hashlib
import json
import pathlib
import sys

patch_path, routing_patch_path, stop_patch_path, stop_termination_patch_path, deterministic_replay_patch_path, mixed_transcript_patch_path, role_boundary_patch_path, target_path, loop_target_path, mock_tool_target_path, message_parser_target_path, manifest_path = map(pathlib.Path, sys.argv[1:13])
state, routing_state, stop_handshake_state, stop_termination_state, deterministic_replay_state, mixed_transcript_state, role_boundary_state = sys.argv[13:20]
payload = {
    "schema_version": 1,
    "status": "PASS",
    "state": {
        "clarification_fallback": state,
        "clarification_routing_v2": routing_state,
        "stop_handshake_v3": stop_handshake_state,
        "mock_user_stop_termination_v4": stop_termination_state,
        "deterministic_tool_replay_v5": deterministic_replay_state,
        "mixed_transcript_reward_v6": mixed_transcript_state,
        "concatenated_role_boundaries_v7": role_boundary_state,
    },
    "reason": "Base Qwen3-8B emits plain-text turns although the released parser requires undocumented <question>/<answer> wrappers, while the official reward requires a mock-user ###STOP handshake.",
    "semantic_scope": "Routes untagged visible assistant text to mock_user, terminates on ###STOP, replays exact published tool fixtures, and uses an audited STOP fallback only when the mock-user API fails after the official normal path completed; generated policy tokens and reward logic are unchanged.",
    "patches": [
        {"path": str(patch_path), "sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest()},
        {"path": str(routing_patch_path), "sha256": hashlib.sha256(routing_patch_path.read_bytes()).hexdigest()},
        {"path": str(stop_patch_path), "sha256": hashlib.sha256(stop_patch_path.read_bytes()).hexdigest()},
        {"path": str(stop_termination_patch_path), "sha256": hashlib.sha256(stop_termination_patch_path.read_bytes()).hexdigest()},
        {"path": str(deterministic_replay_patch_path), "sha256": hashlib.sha256(deterministic_replay_patch_path.read_bytes()).hexdigest()},
        {"path": str(mixed_transcript_patch_path), "sha256": hashlib.sha256(mixed_transcript_patch_path.read_bytes()).hexdigest()},
        {"path": str(role_boundary_patch_path), "sha256": hashlib.sha256(role_boundary_patch_path.read_bytes()).hexdigest()},
    ],
    "target_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
    "target": str(target_path),
    "loop_target_sha256": hashlib.sha256(loop_target_path.read_bytes()).hexdigest(),
    "loop_target": str(loop_target_path),
    "mock_tool_target_sha256": hashlib.sha256(mock_tool_target_path.read_bytes()).hexdigest(),
    "mock_tool_target": str(mock_tool_target_path),
    "message_parser_target_sha256": hashlib.sha256(message_parser_target_path.read_bytes()).hexdigest(),
    "message_parser_target": str(message_parser_target_path),
}
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": payload["status"], "state": payload["state"], "manifest": str(manifest_path)}))
PY
