#!/usr/bin/env bash
set -euo pipefail
set -x

export VERL_USE_EXTERNAL_MODULES=verl_omni
export VERL_OMNI_QWEN35_ONLY=1
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
FSDP_TURBO_PATH="${FSDP_TURBO_PATH:-$REPO_ROOT/../FSDPTurbo}"
VERL_CORE_PATH="${VERL_CORE_PATH:-/efs_rl/z00886395/codes/verl}"
export PYTHONPATH="$FSDP_TURBO_PATH:$REPO_ROOT:$VERL_CORE_PATH:/vllm:/vllm-ascend:${PYTHONPATH:-}"

: "$MODEL_PATH"
: "$TRAIN_FILE"
VAL_FILE="$TRAIN_FILE"
NNODES=1
N_GPUS_PER_NODE=16

python3 -m verl_omni.trainer.main_omni \
    --config-name=qwen3_5_35b_fsdp_turbo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=[4,8,12,16,24,32,48,56,64] \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    transfer_queue.backend.SimpleStorage.total_storage_size=1024 \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    "$@"
