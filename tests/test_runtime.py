#!/usr/bin/env python3
"""CPU-only tests for ATEC interpreter resolution and SAM2 precedence."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from atec_pipeline import cli  # noqa: E402
from atec_pipeline.runtime import (  # noqa: E402
    DEFAULT_FP_PY,
    DEFAULT_ORBBEC_PY,
    DEFAULT_YOLO_PY,
    interpreter_path,
    interpreters,
    resolve_sam2_python,
)
from annotate_multinstance_project import run_tracker  # noqa: E402
from rerun_sam2_range import _sam2_command  # noqa: E402


def main() -> int:
    defaults = interpreters(environ={})
    assert defaults == {
        "orbbec": DEFAULT_ORBBEC_PY,
        "foundationpose": DEFAULT_FP_PY,
        "yolo11": DEFAULT_YOLO_PY,
    }

    with tempfile.TemporaryDirectory(prefix="atec_runtime_") as tmp:
        root = Path(tmp)
        canonical = root / "canonical/python"
        legacy = root / "legacy/python"
        manifest_dir = root / "manifests"

        configured = interpreters(environ={
            "ATEC_ORBBEC_PY": str(root / "canonical/orbbec-python"),
            "ATEC_ORBBEC_PYTHON": str(root / "legacy/orbbec-python"),
            "ATEC_FP_PY": str(root / "canonical/fp-python"),
            "ATEC_FP_PYTHON": str(root / "legacy/fp-python"),
            "ATEC_YOLO_PY": str(canonical),
            "ATEC_YOLO_PYTHON": str(legacy),
        })
        assert configured == {
            "orbbec": (root / "canonical/orbbec-python").resolve(),
            "foundationpose": (root / "canonical/fp-python").resolve(),
            "yolo11": canonical.resolve(),
        }

        assert interpreter_path("yolo11", environ={
            "ATEC_YOLO_PY": str(canonical),
            "ATEC_YOLO_PYTHON": str(legacy),
        }) == canonical.resolve()
        assert interpreter_path(
            "yolo11", environ={"ATEC_YOLO_PYTHON": str(legacy)}
        ) == legacy.resolve()
        assert interpreter_path(
            "orbbec", environ={"ATEC_ORBBEC_PYTHON": str(legacy)}
        ) == legacy.resolve()
        assert interpreter_path(
            "foundationpose", environ={"ATEC_FP_PYTHON": str(legacy)}
        ) == legacy.resolve()

        manifest_python = Path("../envs/manifest/bin/python")
        expected_manifest = (manifest_dir / manifest_python).resolve()
        assert resolve_sam2_python(
            manifest_python, manifest_dir=manifest_dir, environ={}
        ) == expected_manifest
        assert resolve_sam2_python(
            manifest_python,
            manifest_dir=manifest_dir,
            environ={"ATEC_YOLO_PYTHON": str(legacy)},
        ) == legacy.resolve()
        assert resolve_sam2_python(
            manifest_python,
            manifest_dir=manifest_dir,
            environ={
                "ATEC_YOLO_PY": str(canonical),
                "ATEC_YOLO_PYTHON": str(legacy),
            },
        ) == canonical.resolve()
        assert resolve_sam2_python(environ={}) == DEFAULT_YOLO_PY

        with patch.dict(os.environ, {"ATEC_YOLO_PY": str(canonical)}, clear=True):
            assert cli.interpreters()["yolo11"] == canonical.resolve()

            common = {
                "scene": "../data/scenes/can_01",
                "sam2_python": str(manifest_python),
            }
            instance = {
                "tracker": "sam2",
                "instance_id": "can_01",
                "class_id": 0,
                "class_name": "can",
                "key_mask_dir": "../data/key_masks/can_01/can_01",
            }
            output = StringIO()
            with redirect_stdout(output):
                run_tracker(
                    instance,
                    common,
                    manifest_dir,
                    root / "stage",
                    root / "tracker.log",
                    dry_run=True,
                )
            first_command = output.getvalue().splitlines()[0]
            assert first_command.startswith(f"运行： {canonical.resolve()} ")

            rerun_command = _sam2_command(
                instance=instance,
                common=common,
                manifest_dir=manifest_dir,
                scene=root / "scene",
                key_mask_dir=root / "key_masks",
                output_mask_dir=root / "output_masks",
                start=0,
                max_frames=1,
            )
            assert rerun_command[0] == str(canonical.resolve())

    try:
        interpreter_path("unknown", environ={})
    except KeyError as exc:
        assert "未知ATEC运行环境" in str(exc)
    else:
        raise AssertionError("unknown runtime names must be rejected")

    print("RUNTIME_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
