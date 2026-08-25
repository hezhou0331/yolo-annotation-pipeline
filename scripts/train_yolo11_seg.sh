#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_PY="${YOLO_PY:-$HOME/miniforge3/envs/yolo11/bin/python}"
if [[ $# -lt 1 ]]; then echo "用法: $0 <dataset.yaml> [epochs] [batch]" >&2; exit 2; fi
DATA="$1"; EPOCHS="${2:-100}"; BATCH="${3:-4}"
exec "$YOLO_PY" "$ROOT/tools/train_yolo11_seg.py" \
  --data "$DATA" --model "$ROOT/models/yolo11s-seg.pt" \
  --epochs "$EPOCHS" --batch "$BATCH" --imgsz 640 --device 0 --workers 4 \
  --require-project-reports --require-source-ids \
  --project "$ROOT/runs/segment" --name A_official_yolo11s_seg
