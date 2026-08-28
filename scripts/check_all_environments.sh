#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_PY="${ATEC_YOLO_PY:-${ATEC_YOLO_PYTHON:-${YOLO_PY:-$HOME/miniforge3/envs/yolo11/bin/python}}}"

echo "=== 磁盘 ==="
df -h "$ROOT"
echo "=== 相机环境 ==="
"$ROOT/scripts/check_orbbec.sh"
echo "=== FoundationPose环境 ==="
"$ROOT/scripts/check_foundationpose.sh"
echo "=== YOLO11/SAM2环境 ==="
"$YOLO_PY" - "$ROOT" <<'PY'
import sys
from pathlib import Path
import cv2
import torch
import ultralytics

root = Path(sys.argv[1]).resolve()
print('Ultralytics', ultralytics.__version__)
print('PyTorch', torch.__version__)
print('OpenCV', cv2.__version__)
print('CUDA', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU', torch.cuda.get_device_name(0))
for relative in ('models/yolo11s-seg.pt', 'models/yolo11n-seg.pt', 'models/sam2.1_t.pt'):
    model = root / relative
    print(relative, model.exists(), model.stat().st_size if model.exists() else 0)
PY
