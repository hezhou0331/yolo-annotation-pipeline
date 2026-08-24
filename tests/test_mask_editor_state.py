#!/usr/bin/env python3
"""Regression tests for mask-editor keyboard feedback."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atec_pipeline.mask_editor_state import editor_status, is_save_key  # noqa: E402


def main() -> int:
    # The editor must accept regular S/s and Ctrl+S (ASCII control-S=19).
    assert is_save_key(ord("s"))
    assert is_save_key(ord("S"))
    assert is_save_key(19)
    assert not is_save_key(ord("q"))

    # Save state must be visible in the canvas, not only printed to a terminal.
    assert "SAVED" in editor_status("polygon", False, 0, 25, 7367, False)
    assert "UNSAVED" in editor_status("polygon", False, 3, 25, 7367, True)
    print("MASK_EDITOR_STATE_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
