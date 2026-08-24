#!/usr/bin/env python3
"""CPU-only checks for the key-mask editor's visible guidance."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.draw_first_mask import editor_status_lines  # noqa: E402


def main() -> int:
    first, second = editor_status_lines("polygon", False, 14, 25, 1234, False)
    assert "POLYGON" in first and "ADD" in first
    assert "ENTER=apply" in second and "S=save" in second and "Q=exit" in second
    saved_first, saved_second = editor_status_lines("polygon", False, 0, 25, 1234, False)
    assert "SAVED" in saved_first and "Q=exit" in saved_second
    assert all(ord(char) < 128 for char in first + second + saved_first + saved_second)
    print("DRAW_FIRST_MASK_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
