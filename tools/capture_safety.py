"""Small, dependency-free helpers for safe RGB-D capture output handling."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CaptureOutputExistsError(RuntimeError):
    """Raised when a capture would write into a non-empty directory."""


def _has_capture_content(root: Path) -> bool:
    """Treat empty standard subdirectories as an interrupted startup, not data."""
    if not root.exists():
        return False
    standard_dirs = {"rgb", "depth", "masks"}
    for entry in root.iterdir():
        if entry.name in standard_dirs and entry.is_dir() and not any(entry.iterdir()):
            continue
        return True
    return False


def prepare_capture_output(output: Path, *, resume: bool = False) -> dict[str, Path]:
    """Create/validate the standard capture directories.

    A non-empty output is rejected by default to prevent accidental overwrite.
    ``resume=True`` is an explicit opt-in for continuing after Ctrl+C.
    """
    root = output.expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise CaptureOutputExistsError(f"采集输出不是目录：{root}")
    if root.exists() and not resume and _has_capture_content(root):
        raise CaptureOutputExistsError(
            f"采集输出已有数据，为防止覆盖已拒绝写入：{root}；"
            "请换一个目录，或明确使用 --resume 继续采集。"
        )

    directories = {
        "root": root,
        "rgb": root / "rgb",
        "depth": root / "depth",
        "masks": root / "masks",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def write_metadata_atomic(path: Path, metadata: dict[str, Any]) -> None:
    """Write JSON through a sibling temporary file, then atomically replace it."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def finalize_metadata(
    path: Path,
    metadata: dict[str, Any],
    *,
    status: str,
    saved_frames: int,
) -> None:
    """Persist the final status even when the user stopped with Ctrl+C."""
    metadata["status"] = status
    metadata["saved_frames"] = int(saved_frames)
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_metadata_atomic(path, metadata)
