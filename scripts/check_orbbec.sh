#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMBA="${MAMBA:-$HOME/miniforge3/bin/mamba}"
WORKDIR="$(mktemp -d -t atec_orbbec_check.XXXXXX)"
cleanup() { find "$WORKDIR" -depth -delete 2>/dev/null || true; }
trap cleanup EXIT

echo "[1/3] 检查 Orbbec Python 环境"
(
  cd "$WORKDIR"
  "$MAMBA" run -n orbbec python - <<'PY'
import cv2, numpy, pyorbbecsdk
print("numpy:", numpy.__version__)
print("opencv:", cv2.__version__)
print("pyorbbecsdk: OK")
PY
)

echo "[2/3] 检查 USB 设备"
if ! lsusb | grep -i -E '2bc5|orbbec'; then
  echo "未在USB总线上发现Orbbec设备。" >&2
  exit 1
fi

echo "[3/3] 使用ATEC采集工具验证设备与640x480@30硬件D2C配置"
(
  cd "$WORKDIR"
  "$MAMBA" run -n orbbec python "$ROOT/tools/capture_orbbec_rgbd.py" \
    --check-only --width 640 --height 480 --fps 30
)
