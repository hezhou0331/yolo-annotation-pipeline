#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FP_PY="${FP_PY:-$HOME/miniforge3/envs/foundationpose/bin/python}"

usage() {
  echo "用法: $0 <project.yaml> [--allow-missing-key-masks] [--dry-run]" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi
MANIFEST="$1"
shift
ALLOW_MISSING=false
DRY_RUN=false
for option in "$@"; do
  case "$option" in
    --allow-missing-key-masks) ALLOW_MISSING=true ;;
    --dry-run) DRY_RUN=true ;;
    *) echo "未知参数: $option" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -f "$MANIFEST" ]]; then echo "manifest不存在: $MANIFEST" >&2; exit 2; fi
if [[ ! -x "$FP_PY" ]]; then echo "FoundationPose Python不存在: $FP_PY" >&2; exit 2; fi

OUTPUT="$($FP_PY - "$MANIFEST" <<'PY'
import sys, yaml
from pathlib import Path
p = Path(sys.argv[1]).expanduser().resolve()
d = yaml.safe_load(p.read_text(encoding="utf-8"))
out = Path(d['project']['output']).expanduser()
out = (p.parent / out).resolve() if not out.is_absolute() else out.resolve()
print(out)
PY
)"

if [[ "$DRY_RUN" == true ]]; then
  REPORT_DIR="$(mktemp -d -t atec_pipeline_dry_run.XXXXXX)"
  trap 'rm -rf "$REPORT_DIR"' EXIT
  SEGMENT_REPORT="$REPORT_DIR/segments.json"
else
  REPORT_DIR="$OUTPUT/project_reports"
  mkdir -p "$REPORT_DIR"
  SEGMENT_REPORT="$REPORT_DIR/segments.json"
fi

SEG_ARGS=(--manifest "$MANIFEST" --output "$SEGMENT_REPORT")
if [[ "$ALLOW_MISSING" != true ]]; then SEG_ARGS+=(--require-ready); fi
ANNOTATE_ARGS=(--manifest "$MANIFEST")
if [[ "$DRY_RUN" == true ]]; then ANNOTATE_ARGS+=(--dry-run); fi

echo "[1/2] 自动分段并检查每段关键mask"
"$FP_PY" "$ROOT/tools/segment_rgbd_sequence.py" "${SEG_ARGS[@]}"
echo "[2/2] FoundationPose/SAM2传播、质量过滤和多实例安全聚合"
"$FP_PY" "$ROOT/tools/annotate_multinstance_project.py" "${ANNOTATE_ARGS[@]}"
if [[ "$DRY_RUN" == true ]]; then
  echo "Dry-run完成：没有运行跟踪器或覆盖YOLO标签。"
  echo "临时分段报告: $SEGMENT_REPORT（脚本退出后删除）"
else
  echo "完成。正式数据集: $OUTPUT"
  echo "请先检查: $OUTPUT/project_reports 以及 $OUTPUT/review"
fi
