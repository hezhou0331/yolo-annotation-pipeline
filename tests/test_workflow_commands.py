#!/usr/bin/env python3
"""CPU-only contract tests for the thin App's command-planning layer."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atec_pipeline.cli import parser as cli_parser  # noqa: E402
from atec_pipeline.workflow_commands import (  # noqa: E402
    build_pipeline_command,
    build_split_command,
    plan_continue_processing,
)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="atec_workflow_commands_") as tmp:
        root = Path(tmp)
        manifest = root / "manifests/can_train.yaml"
        segments = root / "data/scenes/can/can_train/project_reports/segments.json"

        try:
            plan_continue_processing(manifest, segments, masks_complete=False)
        except FileNotFoundError as exc:
            assert "Manifest" in str(exc)
        else:
            raise AssertionError("missing Manifest must stop before any CLI is launched")

        manifest.parent.mkdir(parents=True)
        manifest.write_text("project: {}\n", encoding="utf-8")
        plan = plan_continue_processing(manifest, segments, masks_complete=False)
        assert plan.action == "segment"
        assert plan.task_kind == "segment"
        assert "自动分段" in plan.feedback

        segments.parent.mkdir(parents=True)
        segments.write_text('{"segments": []}\n', encoding="utf-8")
        plan = plan_continue_processing(manifest, segments, masks_complete=False)
        assert plan.action == "mask"
        assert plan.task_kind is None
        assert "关键帧" in plan.feedback

        plan = plan_continue_processing(manifest, segments, masks_complete=True)
        assert plan.action == "run"
        assert plan.task_kind == "run"
        assert "SAM2" in plan.feedback and "YOLO" in plan.feedback

        program, args = build_pipeline_command(ROOT, "run", manifest)
        assert program == str(ROOT / "scripts/atec-pipeline")
        assert args == ["run", str(manifest.resolve())]

        program, args = build_pipeline_command(
            ROOT,
            "segment",
            manifest,
            "--output",
            str(segments),
        )
        assert program == str(ROOT / "scripts/atec-pipeline")
        assert args == ["segment", str(manifest.resolve()), "--output", str(segments)]

        dataset = root / "datasets/atec_yolo11_seg/dataset.yaml"
        program, args = build_split_command(
            ROOT,
            dataset,
            val_scenes=("can_scene_02", "can_scene_01"),
            apply=True,
        )
        assert program == str(ROOT / "scripts/atec-pipeline")
        assert args == [
            "split",
            str(dataset.resolve()),
            "--val-scenes",
            "can_scene_01",
            "can_scene_02",
            "--apply",
        ]

        help_output = StringIO()
        try:
            with redirect_stdout(help_output):
                cli_parser().parse_args(["split", "--help"])
        except SystemExit as exc:
            assert exc.code == 0
        assert "--target-val-ratio" in help_output.getvalue()

    print("WORKFLOW_COMMAND_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
