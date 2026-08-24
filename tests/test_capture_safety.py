#!/usr/bin/env python3
"""CPU-only regression tests for safe RGB-D capture finalization."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import atec_pipeline.cli as cli  # noqa: E402
from atec_pipeline.cli import capture_command, parser  # noqa: E402
from capture_safety import (  # noqa: E402
    CaptureOutputExistsError,
    finalize_metadata,
    prepare_capture_output,
    write_metadata_atomic,
)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="atec_capture_test_") as tmp:
        root = Path(tmp) / "scene"

        # A fresh capture gets all expected directories and can be finalized.
        dirs = prepare_capture_output(root)
        assert dirs == {"root": root.resolve(), "rgb": root.resolve() / "rgb", "depth": root.resolve() / "depth", "masks": root.resolve() / "masks"}
        metadata_path = root / "metadata.json"
        metadata = {"frames": [{"id": "000000"}], "saved_frames_this_session": 1}
        write_metadata_atomic(metadata_path, metadata)
        finalize_metadata(metadata_path, metadata, status="stopped_by_ctrl_c", saved_frames=1)
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert saved["status"] == "stopped_by_ctrl_c"
        assert saved["saved_frames"] == 1
        assert saved["frames"] == [{"id": "000000"}]

        # A failed startup may leave only empty standard directories; retrying must remain safe.
        partial = Path(tmp) / "partial"
        prepare_capture_output(partial)
        retried = prepare_capture_output(partial)
        assert retried["rgb"].is_dir()

        # A second run must not silently overwrite a previous capture.
        (root / "rgb" / "000000.png").write_bytes(b"already captured")
        try:
            prepare_capture_output(root)
        except CaptureOutputExistsError as exc:
            assert "--resume" in str(exc)
        else:
            raise AssertionError("non-empty capture output must be protected")

        # Explicit resume is allowed, so Ctrl+C can be used to continue later.
        resumed = prepare_capture_output(root, resume=True)
        assert resumed["rgb"].is_dir() and resumed["depth"].is_dir()

    parsed = parser().parse_args(["capture-yellow-can"])
    assert parsed.interval == 0.1 and parsed.output == ROOT / "projects/atec_real/data/scenes/yellow_can_train_01"
    command = capture_command(
        output=Path("scene"), width=640, height=480, fps=30, warmup=20, interval=0.1,
        max_frames=0, min_depth=200, max_depth=3000, auto=True, no_preview=False, resume=False,
    )
    assert "--auto" in command and "--no-preview" not in command and "--resume" not in command

    captured = []
    original_run = cli.run
    cli.run = lambda cmd, check=True: captured.append(cmd) or 0
    try:
        assert cli.capture_yellow_can(parsed) == 0
    finally:
        cli.run = original_run
    assert captured and "--no-preview" not in captured[-1], ("黄罐快捷采集默认必须显示实时预览", captured[-1])

    print("CAPTURE_SAFETY_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
