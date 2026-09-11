# save result path
export HF_HUB_OFFLINE=1
ROOT_DIR="./result211/eval_time"
mkdir -p "$ROOT_DIR"
NUM_PROCESSES=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OPENAI_API_URL=""
export OPENAI_API_KEY=""
set -x

EVAL_TIME=True
BASE_COMMAND="CUDA_VISIBLE_DEVICES=0 python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --model qwen2_5_vl_with_token_compression \
    --batch_size 1 \
    --log_samples"

# first for method name ; second for FILENAME ; third for additional args
METHODS=(
    # Qwen2-5-VL token compression method for example
    #"orig new_preprocess attn_implementation="flash_attention_2""  
    "fastv fastv attn_implementation="flash_attention_2""
    "prumerge+ prumerge+ attn_implementation="flash_attention_2""
    "visionzip_official visionzip_official attn_implementation="flash_attention_2""
    #"dart dart attn_implementation="flash_attention_2""
    "divprune divprune attn_implementation="flash_attention_2""
)


# budgets
BUDGETS=(0.05)

# model path
MODEL_PATH="/data/ModelZoo/LVLM/Qwen-Qwen2.5-VL-7B-Instruct"
MODEL_NAME="qwen25vl_7b"

# MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="qwen25vl_3b"

TASKS=("textvqa_val" "ai2d_no_mask" "chartqa" "ocrbench" "scienceqa_img" "mmmu_val" "mme" "pope" "docvqa_val")


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
