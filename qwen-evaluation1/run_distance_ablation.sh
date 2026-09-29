#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/home/ruimeng/miniconda3/envs/vs_env/bin/python}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_distance_ablation.py" --python "${PYTHON_BIN}" "$@"
