#!/usr/bin/env bash
set -euo pipefail
AUDIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/home/ruimeng/miniconda3/envs/vs_env/bin/python}"
BASELINE_RESULT="${BASELINE_RESULT:-${AUDIT_DIR}/resultrw/fgvc_eval/bird-200/0.1/VisionSelector-Qwen2.5-VL-7B_selector_0.1/20260902_153802_results.json}"
exec "${PYTHON_BIN}" "${AUDIT_DIR}/audit_fgvc_selection.py" --baseline-result "${BASELINE_RESULT}" "$@"
