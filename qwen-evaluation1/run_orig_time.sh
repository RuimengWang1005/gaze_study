# save result path
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Prefer local Hugging Face caches (avoid network HEAD requests).
# Set `HF_OFFLINE=0` to re-enable online mode.
HF_OFFLINE="${HF_OFFLINE:-1}"
if [[ "${HF_OFFLINE}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1
fi

ROOT_DIR="${ROOT_DIR:-./resulttime_orig/eval_time}"
mkdir -p "$ROOT_DIR"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
export OPENAI_API_URL=""
export OPENAI_API_KEY=""
set -x

EVAL_TIME="${EVAL_TIME:-true}"
BASE_COMMAND="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 -m accelerate.commands.launch \
    --main_process_port=${MAIN_PROCESS_PORT:-28171} \
    --mixed_precision=bf16 \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --model qwen2_5_vl_with_token_compression \
    --batch_size 1 \
    --log_samples"

# Run the original model path without selector pruning for a fair latency baseline.
METHODS=(
    "orig 7b attn_implementation=flash_attention_2"
)

# Keep `1` here so the folder layout matches the selector baseline layout.
BUDGETS=("${BUDGETS_OVERRIDE:-1}")

# Default to the trained checkpoint backbone without selector pruning.
# To compare against the official pretrained model instead, set:
# MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
MODEL_PATH="${MODEL_PATH:-../output_ckpt2/VisionSelector-Qwen2.5-VL-7B}"
MODEL_NAME="${MODEL_NAME:-VisionSelector-Qwen2.5-VL-7B-orig}"

TASKS=("docvqa_val")

for TASK in "${TASKS[@]}"; do
    for METHOD_CONFIG in "${METHODS[@]}"; do
        METHOD=$(echo "$METHOD_CONFIG" | awk '{print $1}')
        FILENAME=$(echo "$METHOD_CONFIG" | awk '{print $2}')
        ADDITIONAL_ARGS=$(echo "$METHOD_CONFIG" | awk '{$1="";$2=""; print $0}' | xargs)

        for BUDGET in "${BUDGETS[@]}"; do
            OUTPUT_PATH="$ROOT_DIR/${TASK}/${BUDGET}/${MODEL_NAME}_${METHOD}_${BUDGET}_${FILENAME}"
            LOG_FILE="${OUTPUT_PATH}/log_time.log"

            if [ -d "$OUTPUT_PATH" ]; then
                echo "folder exists, skip."
                continue
            fi

            mkdir -p "$OUTPUT_PATH"

            MODEL_ARGS="pretrained=${MODEL_PATH},method=${METHOD},budgets=${BUDGET},${ADDITIONAL_ARGS}"

            COMMAND="EVAL_TIME=$EVAL_TIME ${BASE_COMMAND} --tasks ${TASK} --output_path ${OUTPUT_PATH} --log_samples_suffix ${TASK} --model_args \"${MODEL_ARGS}\""

            {
                eval "${COMMAND}"
                if [[ "${EVAL_TIME,,}" == "true" ]]; then
                    echo "===== Summary ====="
                    echo "extract memory/latency"
                    python3 extract_time.py --path "${LOG_FILE}"
                fi
            } 2>&1 | tee "${LOG_FILE}"
        done
    done
done
