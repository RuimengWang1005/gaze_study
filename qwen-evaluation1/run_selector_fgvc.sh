#!/usr/bin/env bash

set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

HF_OFFLINE="${HF_OFFLINE:-1}"
if [[ "${HF_OFFLINE}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1
fi

ROOT_DIR="${ROOT_DIR:-./resultrw/fgvc_eval}"
OPTIONS_ROOT="${OPTIONS_ROOT:-../../FinePruner_TIP2026/data}"
#MODEL_PATH="${MODEL_PATH:-../../VisionSelector-Qwen2.5-VL-7B}"
MODEL_PATH="${MODEL_PATH:-../output_ckpt666/VisionSelector-Qwen2.5-VL-7B}"
MODEL_NAME="${MODEL_NAME:-VisionSelector-Qwen2.5-VL-7B}"
METHOD="${METHOD:-selector}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

BUDGETS_STR="${BUDGETS:-0.1 0.2}"
DATASETS_STR="${DATASETS:-bird-200 fgvc_aircraft stanford_cars flowers102 oxford_pet}"
SPATIAL_RATIO="${SPATIAL_RATIO:-${CONTEXTUAL_RATIO:-0.1}}"
COSINE_PENALTY_WEIGHT="${COSINE_PENALTY_WEIGHT:-1}"
COSINE_PENALTY_THRESHOLD="${COSINE_PENALTY_THRESHOLD:-0.7}"
MAX_PIXELS="${MAX_PIXELS:-1605632}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-42}"

mkdir -p "${ROOT_DIR}"

declare -A IMAGE_ROOTS
IMAGE_ROOTS["bird-200"]="${BIRD200_ROOT:-/data/DataSet/vlm_dataset/CUB_200_2011}"
IMAGE_ROOTS["fgvc_aircraft"]="${FGVC_AIRCRAFT_ROOT:-/data/DataSet/cls_dataset/fgvc_aircraft}"
IMAGE_ROOTS["stanford_cars"]="${STANFORD_CARS_ROOT:-/data/DataSet/cls_dataset/standford_cars}"
IMAGE_ROOTS["flowers102"]="${FLOWERS102_ROOT:-/data/DataSet/cls_dataset/Flowers102/rawdata/flowers-102}"
IMAGE_ROOTS["oxford_pet"]="${OXFORD_PET_ROOT:-/data/DataSet/cls_dataset/oxford_pets}"

IFS=' ' read -r -a BUDGETS_ARR <<< "${BUDGETS_STR}"
IFS=' ' read -r -a DATASETS_ARR <<< "${DATASETS_STR}"

set -x

ran_any=0

for DATASET in "${DATASETS_ARR[@]}"; do
    IMAGE_ROOT="${IMAGE_ROOTS[${DATASET}]:-}"
    if [[ -z "${IMAGE_ROOT}" ]]; then
        echo "Skip ${DATASET}: dataset root is not set."
        continue
    fi
    if [[ ! -d "${IMAGE_ROOT}" ]]; then
        echo "Skip ${DATASET}: dataset root does not exist: ${IMAGE_ROOT}"
        continue
    fi

    for BUDGET in "${BUDGETS_ARR[@]}"; do
        OUTPUT_PATH="${ROOT_DIR}/${DATASET}/${BUDGET}/${MODEL_NAME}_${METHOD}_${BUDGET}"
        LOG_FILE="${OUTPUT_PATH}/log_fgvc.log"

        if [[ -d "${OUTPUT_PATH}" ]]; then
            echo "folder exists, skip."
            continue
        fi

        mkdir -p "${OUTPUT_PATH}"
        ran_any=1

        CMD=(
            python3 eval_fgvc_selector.py
            --dataset "${DATASET}"
            --image-folder "${IMAGE_ROOT}"
            --model-path "${MODEL_PATH}"
            --options-root "${OPTIONS_ROOT}"
            --results-output-folder "${OUTPUT_PATH}"
            --method "${METHOD}"
            --budgets "${BUDGET}"
            --spatial_ratio "${SPATIAL_RATIO}"
            --cosine_penalty_weight "${COSINE_PENALTY_WEIGHT}"
            --cosine_penalty_threshold "${COSINE_PENALTY_THRESHOLD}"
            --min_pixels "${MIN_PIXELS}"
            --max_pixels "${MAX_PIXELS}"
            --seed "${SEED}"
        )

        if [[ -n "${LIMIT}" ]]; then
            CMD+=(--limit "${LIMIT}")
        fi

        {
            CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}"
        } 2>&1 | tee "${LOG_FILE}"
    done
done

if [[ "${ran_any}" -eq 0 ]]; then
    echo "No FGVC dataset was evaluated."
    echo "Set one or more of: BIRD200_ROOT, FGVC_AIRCRAFT_ROOT, STANFORD_CARS_ROOT, FLOWERS102_ROOT, OXFORD_PET_ROOT"
    exit 1
fi
