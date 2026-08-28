#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FP_PY="${ATEC_FP_PY:-${ATEC_FP_PYTHON:-$HOME/miniforge3/envs/foundationpose/bin/python}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[1/3] NVIDIA GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader

echo "[2/3] FoundationPose 核心、CUDA 扩展和权重"
cd "$ROOT/third_party/FoundationPose"
"$FP_PY" check_env.py

echo "[3/3] 适配提示"
echo "RTX 4060 8GB：推理最长边保持 640，避免 1280x720 首帧注册显存不足。"
