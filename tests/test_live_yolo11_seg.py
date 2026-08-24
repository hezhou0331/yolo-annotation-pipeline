#!/usr/bin/env python3
"""CPU-only checks for the generic YOLO11-seg live viewer."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.live_yolo11_seg import class_name_summary, parse_source  # noqa: E402


def main() -> int:
    assert parse_source("0") == 0
    assert parse_source("2") == 2
    assert parse_source("video.mp4") == "video.mp4"
    assert class_name_summary({0: "can", 1: "watermelon_rind"}) == "can, watermelon_rind"
    assert class_name_summary(["can", "meal_box"]) == "can, meal_box"
    assert "yellow" not in class_name_summary({0: "can", 1: "watermelon_rind"})
    print("LIVE_YOLO11_SEG_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
