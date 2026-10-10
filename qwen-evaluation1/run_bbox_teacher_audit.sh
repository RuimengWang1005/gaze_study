#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ruimeng/miniconda3/envs/vs_env/bin/python}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_bbox_teacher.py" "$@"
