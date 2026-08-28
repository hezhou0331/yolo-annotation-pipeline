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
        expected_classes = {
            0: "can", 1: "watermelon_rind", 2: "meal_box", 3: "red_paper_bag",
            4: "blue_bin", 5: "green_bin", 6: "red_bin",
            7: "purple_paper_bag", 8: "sand_bottle",
        }
        assert manifest["classes"] == expected_classes
        assert not (project_root / "atec_objects.yaml").exists(), (
            "项目目录不得复制类别配置；唯一来源应保持为configs/atec_objects.yaml"
        )
        assert manifest["project"]["capture_session_id"] == "session_watermelon_01"
        assert manifest["project"]["source_video_id"] == "session_watermelon_01_clip_01"
        assert manifest["project"]["scene"] == "../data/scenes/watermelon_rind_train_01"
        assert manifest["project"]["output"] == "../datasets/atec_yolo11_seg"
        assert not Path(manifest["project"]["sam2_model"]).is_absolute()
        assert (manifest_path.parent / manifest["project"]["sam2_model"]).resolve() == ROOT / "models/sam2.1_t.pt"
        assert "sam2_python" not in manifest["project"]
        assert manifest["instances"] == [{
            "instance_id": "watermelon_rind_01",
            "class_id": 1,
            "class_name": "watermelon_rind",
            "tracker": "sam2",
            "start": 0,
            "max_frames": 0,
            "key_mask_dir": "../data/key_masks/watermelon_rind_train_01/watermelon_rind_01",
        }]

        for class_name, class_id in (("purple_paper_bag", 7), ("sand_bottle", 8)):
            extra_root = Path(tmp) / class_name
            extra_scene = f"{class_name}_train_01"
            subprocess.run([
                sys.executable,
                str(ROOT / "tools/prepare_atec_project.py"),
                "--project-root", str(extra_root),
                "--scene-name", extra_scene,
                "--only-class", class_name,
                "--scene-class", class_name,
                "--capture-session-id", f"session_{class_name}_01",
                "--source-video-id", f"session_{class_name}_01_clip_01",
            ], text=True, capture_output=True, check=True)
            extra_manifest = yaml.safe_load(
                (extra_root / "manifests" / f"{extra_scene}_train.yaml").read_text(encoding="utf-8")
            )
            assert extra_manifest["classes"] == expected_classes
            assert extra_manifest["instances"][0]["class_id"] == class_id
            assert extra_manifest["instances"][0]["class_name"] == class_name
            assert extra_manifest["project"]["scene"] == f"../data/scenes/{class_name}/{extra_scene}"
            assert not (extra_root / "atec_objects.yaml").exists()

    print("SINGLE_CLASS_PROJECT_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
