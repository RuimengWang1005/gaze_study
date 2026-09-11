set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

PRETRAINED="${PRETRAINED:-../output_ckpt2/VisionSelector-Qwen2.5-VL-7B}"
MODEL_NAME="${MODEL_NAME:-qwen25-vl}"
METHOD="${METHOD:-selector}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-3}}"
IMAGE_PATH="${IMAGE_PATH:-/home/bbhan/gaze/gaicolgemii211/qwen-evaluation1/000000097994.jpg}"
QUESTION="${QUESTION:-What text elements are there in the picture?}"
BUDGET="${BUDGET:-${BUDGETS:-0.1}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-vis_results}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
MIN_PIXELS="${MIN_PIXELS:-803520}"
SPATIAL_RATIO="${SPATIAL_RATIO:-${CONTEXTUAL_RATIO:-0.1}}"
COSINE_PENALTY_WEIGHT="${COSINE_PENALTY_WEIGHT:-1}"
COSINE_PENALTY_THRESHOLD="${COSINE_PENALTY_THRESHOLD:-0.7}"

mkdir -p "${OUTPUT_ROOT}"

echo "Using physical CUDA_VISIBLE_DEVICES=${GPU_ID}; inside PyTorch this appears as logical cuda:0"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 predict.py \
    --pretrained "${PRETRAINED}" \
    --model_name "${MODEL_NAME}" \
    --method "${METHOD}" \
    --budgets "${BUDGET}" \
    --max_pixels "${MAX_PIXELS}" \
    --min_pixels "${MIN_PIXELS}" \
    --spatial_ratio "${SPATIAL_RATIO}" \
    --cosine_penalty_weight "${COSINE_PENALTY_WEIGHT}" \
    --cosine_penalty_threshold "${COSINE_PENALTY_THRESHOLD}" \
    --image_path "${IMAGE_PATH}" \
    --question "${QUESTION}" \
    --output_dir "${OUTPUT_ROOT}/${METHOD}"
