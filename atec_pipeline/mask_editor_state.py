"""Pure keyboard/status helpers shared by the OpenCV key-mask editor and tests."""
from __future__ import annotations


def is_save_key(key: int) -> bool:
    """Return whether an OpenCV key code represents S, s, or Ctrl+S."""
    return int(key) in (ord("s"), ord("S"), 19)


def editor_status(
    mode: str,
    erase: bool,
    points: int,
    brush_size: int,
    pixels: int,
    dirty: bool,
) -> str:
    """Build ASCII-only status text; pending points also count as unsaved."""
    save_state = "UNSAVED" if bool(dirty) or int(points) > 0 else "SAVED"
    operation = "ERASE" if erase else "ADD"
    return (
        f"{save_state} | {str(mode).upper()} {operation} | "
        f"points={int(points)} brush={int(brush_size)} pixels={int(pixels)}"
    )
