#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMBA="${MAMBA:-/home/hezhou/miniforge3/bin/mamba}"
OUTPUT="${1:-$ROOT/projects/atec_real/data/scenes/capture_001}"
if [[ $# -gt 0 ]]; then shift; fi
OUTPUT="$(realpath -m -- "$OUTPUT")"
SYSFS_ROOT="${ATEC_ORBBEC_SYSFS_ROOT:-/sys/bus/usb/devices}"
/usr/bin/python3 "$ROOT/tools/orbbec_usb_diagnostics.py" --sysfs-root "$SYSFS_ROOT"
WORKDIR="$(mktemp -d -t atec_orbbec_capture.XXXXXX)"
cleanup() { find "$WORKDIR" -depth -delete 2>/dev/null || true; }
trap cleanup EXIT
(
  cd "$WORKDIR"
  "$MAMBA" run -n orbbec python "$ROOT/tools/capture_orbbec_rgbd.py" --output "$OUTPUT" "$@"
)
