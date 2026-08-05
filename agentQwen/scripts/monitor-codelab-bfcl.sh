#!/usr/bin/env bash
set -euo pipefail

# Read-only monitor for a long-running CodeLab BFCL smoke.  A connection
# failure is deliberately a distinct exit code so an outer watcher can retry
# instead of treating an empty response as a successful evaluation.
SSH_HOST="${AGENTICQWEN_SSH_HOST:-my-codelab}"
REMOTE_ROOT="${AGENTICQWEN_REMOTE_ROOT:-/home/hadoop-aipnlp/agenticqwen-paper-debug-20260804}"
SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout="${AGENTICQWEN_SSH_TIMEOUT:-8}"
  -o ConnectionAttempts=1
)

if ! /usr/bin/ssh "${SSH_OPTS[@]}" "$SSH_HOST" true 2> >(sed 's/^/[ssh] /' >&2); then
  echo "CONTROL_PLANE_UNAVAILABLE host=$SSH_HOST" >&2
  exit 75
fi

/usr/bin/ssh "${SSH_OPTS[@]}" "$SSH_HOST" "ROOT='$REMOTE_ROOT' bash -s" <<'REMOTE_STATUS'
set -euo pipefail
echo "timestamp=$(date -Is)"
echo "gpu=$(nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo unavailable)"
echo "processes="
ps -eo pid,etime,stat,pcpu,pmem,args | grep -E 'cloud_bfcl|hf_openai_server|bfcl (generate|evaluate)' | grep -v grep || true
BENCH="$ROOT/artifacts/agenticqwen_codelab_curriculum_run1/benchmarks/bfcl_smoke"
for variant in base stage2_adapter; do
  result_count=$(find "$BENCH/$variant/result" -type f -name '*_result.json' 2>/dev/null | wc -l | tr -d ' ')
  score_file=0
  test -f "$BENCH/$variant/score/data_overall.csv" && score_file=1
  echo "variant=$variant result_files=$result_count score_csv=$score_file"
done
echo "base_log_tail="
tail -5 "$BENCH/base/bfcl.log" 2>/dev/null || true
echo "orchestrator_log_tail="
tail -5 "$ROOT/logs/bfcl_smoke.log" 2>/dev/null || true
REMOTE_STATUS
