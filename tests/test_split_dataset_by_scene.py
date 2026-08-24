#!/usr/bin/env python3
"""Regression checks for leakage-safe scene/video dataset splitting."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.split_dataset_by_scene as split_module  # noqa: E402
from tools.split_dataset_by_scene import apply_plan, build_auto_plan, build_plan, scan_groups  # noqa: E402


def write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"image-{value}".encode())


def write_report(path: Path, scene: Path, video: str, stem: str, split: str = "train") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "scene": str(scene),
        "split": split,
        "capture_session_id": f"session_{video}",
        "source_video_id": video,
        "instances": [{"class_id": 0, "class_name": "can"}],
        "frame_status_counts": {"accepted": 1, "review": 0, "rejected": 0},
        "frames": [{"output_id": stem, "status": "accepted"}],
    }), encoding="utf-8")


def main() -> int:
    help_result = subprocess.run(
        [sys.executable, str(ROOT / "tools/split_dataset_by_scene.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "20%" in help_result.stdout

    with tempfile.TemporaryDirectory(prefix="atec_split_") as tmp:
        root = Path(tmp)
        dataset = root / "dataset"
        train_img = dataset / "images/train/scene_a_000001.png"
        train_lbl = dataset / "labels/train/scene_a_000001.txt"
        write_image(train_img, 40)
        train_lbl.parent.mkdir(parents=True, exist_ok=True)
        train_lbl.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
        write_report(dataset / "project_reports/scene_a_train_report.json", root / "scene_a", "video_a", "scene_a_000001")
        (dataset / "dataset.yaml").write_text(yaml.safe_dump({
            "path": str(dataset), "train": "images/train", "val": "images/val", "test": "images/val",
            "names": {0: "can", 1: "watermelon_rind", 2: "meal_box", 3: "red_paper_bag", 4: "blue_bin", 5: "green_bin", 6: "red_bin"},
        }, sort_keys=False), encoding="utf-8")

        groups = scan_groups(dataset)
        assert len(groups) == 1
        assert groups[0].source_video_id == "video_a"

        # 同一个采集会话即使出现不同clip，也必须视为同一隔离组。
        second_report = dataset / "project_reports/scene_a_clip2_train_report.json"
        second_report.write_text(json.dumps({
            "scene": str(root / "scene_a_clip2"), "split": "train",
            "capture_session_id": "session_video_a", "source_video_id": "video_a_clip2",
            "frames": [], "frame_status_counts": {"accepted": 0},
        }), encoding="utf-8")
        grouped = scan_groups(dataset)
        assert len(grouped) == 1
        assert grouped[0].capture_session_id == "session_video_a"
        plan = build_plan(dataset, scenes=["scene_a"], videos=[])
        assert plan["files"] == [(train_img, dataset / "images/val/scene_a_000001.png", train_lbl, dataset / "labels/val/scene_a_000001.txt")]
        apply_plan(plan)
        assert not train_img.exists() and not train_lbl.exists()
        assert (dataset / "images/val/scene_a_000001.png").exists()
        assert (dataset / "labels/val/scene_a_000001.txt").exists()
        report = json.loads((dataset / "project_reports/scene_a_train_report.json").read_text(encoding="utf-8"))
        assert report["split"] == "val"
        data = yaml.safe_load((dataset / "dataset.yaml").read_text(encoding="utf-8"))
        assert data["val"] == "images/val"
        assert "test" not in data, "scene split must remove a stale val-as-test alias"

        try:
            build_plan(dataset, scenes=["missing"], videos=[])
        except ValueError as exc:
            assert "未找到" in str(exc)
        else:
            raise AssertionError("missing scene must be rejected")

    with tempfile.TemporaryDirectory(prefix="atec_auto_split_") as tmp:
        project = Path(tmp) / "project"
        dataset = project / "datasets/atec_yolo11_seg"
        (dataset / "dataset.yaml").parent.mkdir(parents=True, exist_ok=True)
        (dataset / "dataset.yaml").write_text(yaml.safe_dump({
            "path": str(dataset), "train": "images/train", "val": "images/val",
            "names": {0: "can", 1: "watermelon_rind", 2: "meal_box", 3: "red_paper_bag", 4: "blue_bin", 5: "green_bin", 6: "red_bin"},
        }, sort_keys=False), encoding="utf-8")
        scene_specs = [
            ("can_20260824_0800_01", 8),
            ("can_20260824_0900_01", 7),
            ("can_20260824_1000_01", 5),
        ]
        for scene_name, frames in scene_specs:
            scene = project / "data/scenes/can" / scene_name
            scene.mkdir(parents=True)
            (scene / "atec_capture_session.json").write_text(json.dumps({
                "capture_session_id": f"session_{scene_name}",
                "source_video_id": f"video_{scene_name}",
                "split": "train",
            }), encoding="utf-8")
            manifest = project / "manifests" / f"{scene_name}_train.yaml"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(yaml.safe_dump({
                "project": {"scene": str(scene), "split": "train"},
            }, sort_keys=False), encoding="utf-8")
            accepted_frames = []
            for index in range(frames):
                output_id = f"{scene_name}_{index:06d}"
                write_image(dataset / "images/train" / f"{output_id}.png", index)
                label = dataset / "labels/train" / f"{output_id}.txt"
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
                accepted_frames.append({"output_id": output_id, "status": "accepted"})
            report = dataset / "project_reports" / f"{scene_name}_train_report.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({
                "scene": str(scene), "split": "train",
                "capture_session_id": f"session_{scene_name}",
                "source_video_id": f"video_{scene_name}",
                "instances": [{"class_id": 0, "class_name": "can"}],
                "frame_status_counts": {"accepted": frames, "review": 0, "rejected": 0},
                "frames": accepted_frames,
            }), encoding="utf-8")

        plan = build_auto_plan(dataset / "dataset.yaml", target_ratio=0.20)
        selected_scenes = {name for group in plan["groups"] for name in group.scene_names}
        assert selected_scenes == {"can_20260824_1000_01"}, selected_scenes
        assert len(plan["files"]) == 5

        original_atomic_write = split_module._atomic_write_text
        calls = {"count": 0}
        def fail_during_metadata(path: Path, content: str) -> None:
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("forced transaction failure")
            original_atomic_write(path, content)
        split_module._atomic_write_text = fail_during_metadata
        try:
            try:
                apply_plan(plan)
            except RuntimeError as exc:
                assert "forced transaction failure" in str(exc)
            else:
                raise AssertionError("transaction failure must propagate")
        finally:
            split_module._atomic_write_text = original_atomic_write
        assert len(list((dataset / "images/train").glob("*.png"))) == 20
        assert not list((dataset / "images/val").glob("*.png"))
        rollback_scene = "can_20260824_1000_01"
        assert (project / "manifests" / f"{rollback_scene}_train.yaml").exists()
        rollback_report = json.loads((dataset / "project_reports" / f"{rollback_scene}_train_report.json").read_text(encoding="utf-8"))
        assert rollback_report["split"] == "train"

        apply_plan(plan)
        assert len(list((dataset / "images/train").glob("*.png"))) == 15
        assert len(list((dataset / "images/val").glob("*.png"))) == 5
        selected = "can_20260824_1000_01"
        assert not (project / "manifests" / f"{selected}_train.yaml").exists()
        val_manifest = project / "manifests" / f"{selected}_val.yaml"
        assert val_manifest.exists()
        assert yaml.safe_load(val_manifest.read_text(encoding="utf-8"))["project"]["split"] == "val"
        metadata = json.loads((project / "data/scenes/can" / selected / "atec_capture_session.json").read_text(encoding="utf-8"))
        assert metadata["split"] == "val"

    # The newest scene must not win merely because it is newest when its
    # accepted-frame count would badly overshoot the per-class 20% target.
    with tempfile.TemporaryDirectory(prefix="atec_auto_split_ratio_") as tmp:
        root = Path(tmp)
        project = root / "projects/atec_real"
        dataset = project / "datasets/atec_yolo11_seg"
        (dataset / "dataset.yaml").parent.mkdir(parents=True)
        (dataset / "dataset.yaml").write_text(yaml.safe_dump({
            "path": str(dataset), "train": "images/train", "val": "images/val", "names": {0: "can"},
        }), encoding="utf-8")
        specs = [
            ("can_20260824_0800_01", 54),
            ("can_20260824_0900_01", 56),
            ("can_20260824_1000_01", 82),
            ("can_20260824_1100_01", 259),
        ]
        for scene_name, frames in specs:
            scene = project / "data/scenes/can" / scene_name
            scene.mkdir(parents=True)
            accepted = []
            for index in range(frames):
                output_id = f"{scene_name}_{index:06d}"
                write_image(dataset / "images/train" / f"{output_id}.png", index)
                label = dataset / "labels/train" / f"{output_id}.txt"
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
                accepted.append({"output_id": output_id, "status": "accepted"})
            report = dataset / "project_reports" / f"{scene_name}_train_report.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({
                "scene": str(scene), "split": "train",
                "capture_session_id": f"session_{scene_name}",
                "source_video_id": f"video_{scene_name}",
                "instances": [{"class_id": 0, "class_name": "can"}],
                "frame_status_counts": {"accepted": frames, "review": 0, "rejected": 0},
                "frames": accepted,
            }), encoding="utf-8")
        ratio_plan = build_auto_plan(dataset / "dataset.yaml", target_ratio=0.20)
        ratio_scenes = {name for group in ratio_plan["groups"] for name in group.scene_names}
        assert ratio_scenes == {"can_20260824_1000_01"}, ratio_scenes
        assert len(ratio_plan["files"]) == 82

    with tempfile.TemporaryDirectory(prefix="atec_auto_split_insufficient_") as tmp:
        root = Path(tmp)
        dataset = root / "dataset"
        (dataset / "dataset.yaml").parent.mkdir(parents=True)
        (dataset / "dataset.yaml").write_text(yaml.safe_dump({
            "path": str(dataset), "train": "images/train", "val": "images/val", "names": {0: "can"},
        }), encoding="utf-8")
        write_image(dataset / "images/train/only_000000.png", 1)
        label = dataset / "labels/train/only_000000.txt"
        label.parent.mkdir(parents=True)
        label.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
        write_report(dataset / "project_reports/only_train_report.json", root / "only", "only", "only_000000")
        try:
            build_auto_plan(dataset / "dataset.yaml")
        except ValueError as exc:
            assert "至少两个" in str(exc)
        else:
            raise AssertionError("one independent scene must not fabricate validation data")

    print("SPLIT_DATASET_BY_SCENE_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
