# save result path
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

if [[ "${TASK_SUITE:-lmms}" == "fgvc" ]]; then
    exec bash "$(dirname "$0")/run_selector_fgvc.sh"
fi

# Prefer local Hugging Face caches (avoid network HEAD requests).
# Set `HF_OFFLINE=0` to re-enable online mode.
HF_OFFLINE="${HF_OFFLINE:-1}"
if [[ "${HF_OFFLINE}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1
fi

ROOT_DIR="./resultrw/eval_time"
mkdir -p "$ROOT_DIR"
NUM_PROCESSES=1
export OPENAI_API_URL=""
export OPENAI_API_KEY=""
set -x

EVAL_TIME="${EVAL_TIME:-true}"
BASE_COMMAND="CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --main_process_port=28170 \
    --mixed_precision=bf16 \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --model qwen2_5_vl_with_token_compression \
    --batch_size 1 \
    --log_samples"

# first for method name ; second for FILENAME ; third for additional args
METHODS=(
    # Qwen2-5-VL token compression method for example
    "selector 7b attn_implementation=flash_attention_2"
    #"selector 3b attn_implementation=flash_attention_2"
)

# budgets
BUDGETS=(0.3)
SPATIAL_RATIO=${SPATIAL_RATIO:-${CONTEXTUAL_RATIO:-0.1}}
COSINE_PENALTY_WEIGHT=${COSINE_PENALTY_WEIGHT:-1}
COSINE_PENALTY_THRESHOLD=${COSINE_PENALTY_THRESHOLD:-0.7}
SELECTOR_FOCAL_REDUNDANCY_LIMIT=${SELECTOR_FOCAL_REDUNDANCY_LIMIT:-0}
# Image resolution is controlled by pixel bounds rather than a fixed HxW.
# Keep defaults aligned with lmms-eval's current wrapper behavior.
MAX_PIXELS=${MAX_PIXELS:-1605632}
MIN_PIXELS=${MIN_PIXELS:-200704}
VIDEO_READER_BACKEND=${VIDEO_READER_BACKEND:-decord}
VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-151200}
VIDEO_MIN_PIXELS=${VIDEO_MIN_PIXELS:-100352}

# model path
# MODEL_PATH="../VisionSelector-Qwen2.5-VL-7B"
# MODEL_NAME="VisionSelector-Qwen2.5-VL-7B"
MODEL_PATH="/data/ModelZoo/LVLM/Qwen-Qwen2.5-VL-7B-Instruct"
MODEL_NAME="VisionSelector-Qwen2.5-VL-7B"

TASKS=("docvqa_val")
#TASKS=("ai2d_no_mask" "chartqa" "ocrbench" "scienceqa_img" "mmmu_val" "mme" "pope" "docvqa_val")
#TASKS=("docvqa_val")
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

            MODEL_ARGS="pretrained=${MODEL_PATH},method=${METHOD},budgets=${BUDGET},spatial_ratio=${SPATIAL_RATIO},cosine_penalty_weight=${COSINE_PENALTY_WEIGHT},cosine_penalty_threshold=${COSINE_PENALTY_THRESHOLD},selector_focal_redundancy_limit=${SELECTOR_FOCAL_REDUNDANCY_LIMIT},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},video_reader_backend=${VIDEO_READER_BACKEND},video_max_pixels=${VIDEO_MAX_PIXELS},video_min_pixels=${VIDEO_MIN_PIXELS},${ADDITIONAL_ARGS}"

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
