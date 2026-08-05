#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${AGENTICQWEN_ROOT:-/home/hadoop-aipnlp/agenticqwen-paper-debug-20260804}"
UPSTREAM_ROOT="${AGENTICQWEN_UPSTREAM_REPO:-$RUN_ROOT/upstream/data_synth_and_rl}"
RL_ROOT="$UPSTREAM_ROOT/RL"
ENV_ROOT="${AGENTICQWEN_VERL_ENV:-$RUN_ROOT/venvs/verl-sglang-py312}"
PYTHON_BIN="$ENV_ROOT/bin/python"
MODEL_PATH="${AGENTICQWEN_MODEL_PATH:-$RUN_ROOT/models/Qwen3-8B}"
DATA_ROOT="${AGENTICQWEN_DATA_ROOT:-$RL_ROOT/my_data/tmp/industrial_official_pool_bounded10}"
TRAIN_FILES="${AGENTICQWEN_TRAIN_FILES:-$DATA_ROOT/train.parquet}"
VAL_FILES="${AGENTICQWEN_VAL_FILES:-$DATA_ROOT/val.parquet}"
TOOL_CONFIG_PATH="${AGENTICQWEN_TOOL_CONFIG:-$RUN_ROOT/configs/official_h200_tool_config.yaml}"
OUTPUT_ROOT="${AGENTICQWEN_TRAIN_OUTPUT:-$RUN_ROOT/artifacts/official_grpo_h200}"
APP_ID_FILE="${AIGC_APP_ID_FILE:-/home/hadoop-aipnlp/.aigc_app_id}"
DATASET_MANIFEST="${AGENTICQWEN_DATASET_MANIFEST:-$RL_ROOT/my_data/raw/tool_use/industrial_official_pool_bounded10/dataset_manifest.json}"
ENVIRONMENT_MANIFEST="${AGENTICQWEN_ENVIRONMENT_MANIFEST:-$RUN_ROOT/artifacts/official_verl_env/environment.json}"
AUDIT_SCRIPT="${AGENTICQWEN_AUDIT_SCRIPT:-$RUN_ROOT/scripts/audit_official_grpo_run.py}"
TEACHER_PROBE_SCRIPT="${AGENTICQWEN_TEACHER_PROBE_SCRIPT:-$RUN_ROOT/scripts/probe_teacher_api.py}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-32}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-512}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_TURNS="${MAX_TURNS:-15}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-5}"
MERGE_AFTER_TRAIN="${MERGE_AFTER_TRAIN:-1}"
TEACHER_PROBE_ATTEMPTS="${TEACHER_PROBE_ATTEMPTS:-3}"
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"

for required in "$PYTHON_BIN" "$MODEL_PATH/config.json" "$TRAIN_FILES" "$VAL_FILES" "$TOOL_CONFIG_PATH" "$DATASET_MANIFEST" "$ENVIRONMENT_MANIFEST" "$AUDIT_SCRIPT" "$TEACHER_PROBE_SCRIPT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required training input: $required" >&2
    exit 30
  fi
done
if [[ ! -s "$APP_ID_FILE" ]]; then
  echo "Missing API credential file: $APP_ID_FILE" >&2
  exit 31
fi
if [[ "$TOTAL_TRAINING_STEPS" -lt 1 || "$ROLLOUT_N" -lt 2 ]]; then
  echo "GRPO requires TOTAL_TRAINING_STEPS>=1 and ROLLOUT_N>=2" >&2
  exit 32
fi

AIGC_APP_ID=""
# ``read`` returns non-zero at EOF when a one-line secret file has no trailing
# newline.  Keep the bytes it read and validate the resulting value explicitly.
IFS= read -r AIGC_APP_ID < "$APP_ID_FILE" || true
if [[ -z "$AIGC_APP_ID" ]]; then
  echo "API credential file is empty: $APP_ID_FILE" >&2
  exit 33
fi

export CONDA_PREFIX="$ENV_ROOT"
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC="$ENV_ROOT/bin/x86_64-conda-linux-gnu-cc"
export CXX="$ENV_ROOT/bin/x86_64-conda-linux-gnu-c++"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export PYTHONPATH="$RL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

API_BASE="${AGENTICQWEN_API_BASE:-https://aigc.sankuai.com/v1/openai/native}"
TEACHER_MODEL="${AGENTICQWEN_TEACHER_MODEL:-deepseek-v4-flash-sa-256k}"
export MOCK_USER_API_BASE="$API_BASE"
export MOCK_USER_API_KEY="$AIGC_APP_ID"
export MOCK_USER_MODEL_NAME="$TEACHER_MODEL"
export MOCK_TOOL_API_BASE="$API_BASE"
export MOCK_TOOL_API_KEY="$AIGC_APP_ID"
export MOCK_TOOL_MODEL_NAME="$TEACHER_MODEL"
export REWARD_API_BASE="$API_BASE"
export REWARD_API_KEY="$AIGC_APP_ID"
export REWARD_MODEL_NAME="$TEACHER_MODEL"
unset AIGC_APP_ID

mkdir -p "$OUTPUT_ROOT/checkpoints" "$OUTPUT_ROOT/rollouts" "$OUTPUT_ROOT/validation"
ulimit -n 65535
cd "$RL_ROOT"

"$PYTHON_BIN" "$TEACHER_PROBE_SCRIPT" \
  --api-base "$API_BASE" \
  --model "$TEACHER_MODEL" \
  --key-file "$APP_ID_FILE" \
  --attempts "$TEACHER_PROBE_ATTEMPTS" --stop-after-success \
  --output "$OUTPUT_ROOT/teacher_api_probe.json"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.train_max_samples="$TRAIN_MAX_SAMPLES" \
  data.val_max_samples="$VAL_MAX_SAMPLES" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.32 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.multi_stage_wake_up=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=my_custom_hermes \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_LOOP_WORKERS" \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=agenticqwen_official_h200 \
  trainer.experiment_name=qwen3_8b_official_data_grpo \
  trainer.default_local_dir="$OUTPUT_ROOT/checkpoints" \
  trainer.rollout_data_dir="$OUTPUT_ROOT/rollouts" \
  trainer.validation_data_dir="$OUTPUT_ROOT/validation" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.resume_mode=auto \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.log_val_generations=4 \
  trainer.max_actor_ckpt_to_keep=1 \
  reward_model.reward_manager=dapo \
  custom_reward_function.path=my_script/reward_function.py \
  custom_reward_function.name=compute_score_virtual_tool_completion

if [[ "$MERGE_AFTER_TRAIN" == "1" ]]; then
  LATEST_STEP=""
  if [[ -s "$OUTPUT_ROOT/checkpoints/latest_checkpointed_iteration.txt" ]]; then
    IFS= read -r LATEST_STEP < "$OUTPUT_ROOT/checkpoints/latest_checkpointed_iteration.txt"
  fi
  if [[ -n "$LATEST_STEP" ]]; then
    "$PYTHON_BIN" -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$OUTPUT_ROOT/checkpoints/global_step_${LATEST_STEP}/actor" \
      --target_dir "$OUTPUT_ROOT/merged/global_step_${LATEST_STEP}"
  fi
fi

"$PYTHON_BIN" "$AUDIT_SCRIPT" \
  --run-dir "$OUTPUT_ROOT" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --environment-manifest "$ENVIRONMENT_MANIFEST" \
  --expected-steps "$TOTAL_TRAINING_STEPS" \
  --output "$OUTPUT_ROOT/audit.json"
