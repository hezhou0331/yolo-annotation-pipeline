#!/usr/bin/env python3
"""Regression test for single-class capture project manifests."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="atec_single_class_") as tmp:
        project_root = Path(tmp) / "project"
        command = [
            sys.executable,
            str(ROOT / "tools/prepare_atec_project.py"),
            "--project-root", str(project_root),
            "--scene-name", "watermelon_rind_train_01",
            "--only-class", "watermelon_rind",
            "--capture-session-id", "session_watermelon_01",
            "--source-video-id", "session_watermelon_01_clip_01",
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        assert "manifest" in result.stdout
        manifest_path = project_root / "manifests/watermelon_rind_train_01_train.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert set(manifest["classes"].values()) >= {
            "can", "watermelon_rind", "meal_box", "red_paper_bag"
        }
        assert manifest["project"]["capture_session_id"] == "session_watermelon_01"
        assert manifest["project"]["source_video_id"] == "session_watermelon_01_clip_01"
        assert manifest["instances"] == [{
            "instance_id": "watermelon_rind_01",
            "class_id": 1,
            "class_name": "watermelon_rind",
            "tracker": "sam2",
            "start": 0,
            "max_frames": 0,
            "key_mask_dir": str(project_root / "data/key_masks/watermelon_rind_train_01/watermelon_rind_01"),
        }]

    print("SINGLE_CLASS_PROJECT_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
