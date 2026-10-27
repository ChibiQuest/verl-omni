#!/usr/bin/env bash
# DanceGRPO Wan2.2 text-to-video e2e smoke test for Ascend NPU.
set -xeuo pipefail

NUM_NPUS=${NUM_NPUS:-8}
REWARD_NUM_NPUS=${REWARD_NUM_NPUS:-2}
ACTOR_NUM_NPUS=${ACTOR_NUM_NPUS:-$((NUM_NPUS - REWARD_NUM_NPUS))}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/models/tiny-random/Wan2.2-TI2V-5B-Diffusers}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/tokenizer}
REWARD_SNAPSHOT=${REWARD_SNAPSHOT:-${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3}
if [[ -z "${REWARD_MODEL_PATH:-}" ]]; then
    if [[ -d "${REWARD_SNAPSHOT}" ]]; then
        REWARD_MODEL_PATH="${REWARD_SNAPSHOT}"
    else
        REWARD_MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
    fi
fi
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion}
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2}
MICRO_BATCH_SIZE_PER_NPU=${MICRO_BATCH_SIZE_PER_NPU:-1}
DIAG_ROUNDS=${DIAG_ROUNDS:-3}
if (( ACTOR_NUM_NPUS < 1 || REWARD_NUM_NPUS < 1 || ACTOR_NUM_NPUS + REWARD_NUM_NPUS > NUM_NPUS )); then
    echo "Invalid NPU split: total=${NUM_NPUS}, actor=${ACTOR_NUM_NPUS}, reward=${REWARD_NUM_NPUS}" >&2
    exit 2
fi

MINI_BATCH_SIZE=$((MICRO_BATCH_SIZE_PER_NPU * ACTOR_NUM_NPUS))
TRAIN_BATCH_SIZE=$((MINI_BATCH_SIZE * N_RESP_PER_PROMPT))
CUSTOM_CHAT_TEMPLATE='"{% if messages %}{% for message in messages %}{% if message[\"role\"] == \"user\" %}{{ message[\"content\"] }}{% endif %}{% endfor %}{% endif %}</s>"'

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export DEVICE_NAME=npu
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"

set +u
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

diag_python() {
    local label="$1"
    local expr="$2"
    local i

    for ((i = 1; i <= DIAG_ROUNDS; i++)); do
        if ! python3 -c "${expr}"; then
            echo "[DIAG] FAIL: ${label} (round ${i}/${DIAG_ROUNDS})" >&2
            return 1
        fi
    done
    echo "[DIAG] PASS: ${label} (${DIAG_ROUNDS}/${DIAG_ROUNDS} rounds)"
}

DIAG_FAILED=0
set +e
diag_python "A0: TorchNPU + Triton Ascend target detection" \
    "import torch_npu; import triton; print(triton.runtime.driver.active.get_current_target())" \
    || DIAG_FAILED=1
diag_python "A: import vllm (loads the selected platform plugin)" \
    "import vllm; print('import vllm OK')" \
    || DIAG_FAILED=1
diag_python "B: ModelConfig for Qwen2.5-VL reward model (runs model inspection subprocess)" \
    "from vllm.config import ModelConfig; ModelConfig(model='${REWARD_MODEL_PATH}'); print('ModelConfig(inspect) OK')" \
    || DIAG_FAILED=1
diag_python "C: import vllm.model_executor.models.qwen2_5_vl (reward model class)" \
    "import vllm.model_executor.models.qwen2_5_vl; print('import qwen2_5_vl OK')" \
    || DIAG_FAILED=1
diag_python "D: control import vllm.model_executor.models.qwen3_vl" \
    "import vllm.model_executor.models.qwen3_vl; print('import qwen3_vl OK')" \
    || DIAG_FAILED=1
set -e

if [[ "${DIAG_FAILED}" -ne 0 ]]; then
    cat >&2 <<'EOF'
[DIAG] ====================================================================
[DIAG] Wan2.2 reward-path preflight FAILED. The native vLLM / vLLM-Ascend /
[DIAG] torch-npu stack cannot safely initialize or inspect the Qwen2.5-VL
[DIAG] reward model, so starting the training job would fail as well.
[DIAG] ====================================================================
EOF
    exit 1
fi

python3 tests/special_e2e/build_wan22_tiny_random.py \
    --output-dir "${MODEL_PATH}" \
    --verify-load

if [[ ! -f "${TRAIN_FILES}" || ! -f "${VAL_FILES}" ]]; then
    python3 tests/special_e2e/create_dummy_diffusion_data.py \
        --local_save_dir "${DATA_DIR}" \
        --train_size "${TRAIN_BATCH_SIZE}" \
        --val_size 4
fi

python3 -m verl_omni.trainer.main_diffusion \
    trainer.use_v1=false \
    trainer.device=npu \
    algorithm.adv_estimator=dance_grpo \
    actor_rollout_ref.model.algorithm=dance_grpo \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${TOKENIZER_PATH}" \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.custom_chat_template="${CUSTOM_CHAT_TEMPLATE}" \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dance_grpo \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_NPU}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=10000 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers="${ACTOR_NUM_NPUS}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_NPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=5.0 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.pipeline.num_frames=4 \
    +actor_rollout_ref.rollout.pipeline.output_type=np \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=5.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type=dance_sde \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range='[0,4]' \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_NPU}" \
    reward.num_workers=1 \
    reward.reward_model.enable=True \
    reward.reward_model.model_path="${REWARD_MODEL_PATH}" \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.enable_resource_pool=True \
    reward.reward_model.nnodes=1 \
    reward.reward_model.n_gpus_per_node="${REWARD_NUM_NPUS}" \
    reward.reward_model.rollout.tensor_model_parallel_size="${REWARD_NUM_NPUS}" \
    reward.reward_model.rollout.gpu_memory_utilization=0.5 \
    reward.reward_model.rollout.free_cache_engine=False \
    reward.reward_model.rollout.enforce_eager=True \
    reward.reward_model.rollout.do_sample=False \
    reward.reward_model.rollout.temperature=0.0 \
    reward.reward_model.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    reward.reward_model.rollout.response_length=32 \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/genrm_ocr.py \
    reward.custom_reward_function.name=compute_score_ocr \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.seed=42 \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=dancegrpo-wan22-npu-e2e \
    trainer.log_val_generations=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${ACTOR_NUM_NPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    "$@"

echo "DanceGRPO Wan2.2 NPU e2e test passed (training completed successfully)."
