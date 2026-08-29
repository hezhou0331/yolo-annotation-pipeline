#!/usr/bin/env python3
"""Regression checks for leakage-safe scene/video dataset splitting."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.split_dataset_by_scene as split_module  # noqa: E402
from atec_pipeline.gui_state import load_scene_human_review_completion  # noqa: E402
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


def write_manifest(path: Path, scene: Path, split: str = "train") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "project": {"scene": str(scene), "split": split},
    }, sort_keys=False), encoding="utf-8")


def stored_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, start=base)).as_posix()


def snapshot_tree(root: Path) -> tuple[tuple[str, ...], dict[str, tuple[bytes, int, int]]]:
    directories = tuple(sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_dir()))
    files = {
        str(path.relative_to(root)): (
            path.read_bytes(), stat.S_IMODE(path.stat().st_mode), path.stat().st_mtime_ns,
        )
        for path in root.rglob("*") if path.is_file()
    }
    return directories, files


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
        project = root / "project"
        dataset = project / "datasets/atec_yolo11_seg"
        scene_a = project / "data/scenes/can/scene_a"
        scene_a.mkdir(parents=True)
        scene_a_manifest = project / "manifests/scene_a_train.yaml"
        write_manifest(scene_a_manifest, scene_a)
        train_img = dataset / "images/train/scene_a_000001.png"
        train_lbl = dataset / "labels/train/scene_a_000001.txt"
        write_image(train_img, 40)
        train_lbl.parent.mkdir(parents=True, exist_ok=True)
        train_lbl.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
        write_report(dataset / "project_reports/scene_a_train_report.json", scene_a, "video_a", "scene_a_000001")
        (dataset / "dataset.yaml").write_text(yaml.safe_dump({
            "path": str(dataset), "train": "images/train", "val": "images/val", "test": "images/val",
            "names": {0: "can", 1: "watermelon_rind", 2: "meal_box", 3: "red_paper_bag", 4: "blue_bin", 5: "green_bin", 6: "red_bin"},
        }, sort_keys=False), encoding="utf-8")

        groups = scan_groups(dataset)
        assert len(groups) == 1
        assert groups[0].source_video_id == "video_a"

        # 同一个采集会话即使出现不同clip，也必须视为同一隔离组。
        scene_a_clip2 = project / "data/scenes/can/scene_a_clip2"
        scene_a_clip2.mkdir(parents=True)
        scene_a_clip2_manifest = project / "manifests/scene_a_clip2_train.yaml"
        write_manifest(scene_a_clip2_manifest, scene_a_clip2)
        second_report = dataset / "project_reports/scene_a_clip2_train_report.json"
        second_report.write_text(json.dumps({
            "scene": str(scene_a_clip2), "split": "train",
            "capture_session_id": "session_video_a", "source_video_id": "video_a_clip2",
            "frames": [], "frame_status_counts": {"accepted": 0},
        }), encoding="utf-8")

        # Every accepted ID has exactly one report/scene owner, even when two
        # reports are joined into the same capture-session isolation group.
        second_payload = json.loads(second_report.read_text(encoding="utf-8"))
        duplicate_payload = dict(second_payload)
        duplicate_payload["frames"] = [{"output_id": "scene_a_000001", "status": "accepted"}]
        duplicate_payload["frame_status_counts"] = {"accepted": 1}
        second_report.write_text(json.dumps(duplicate_payload), encoding="utf-8")
        try:
            build_plan(dataset, scenes=["scene_a"], videos=[])
        except ValueError as exc:
            assert "重复认领" in str(exc)
        else:
            raise AssertionError("different reports must not claim the same accepted output_id")
        second_report.write_text(json.dumps(second_payload), encoding="utf-8")

        # A paired but unreported scene-prefixed formal output is still unsafe:
        # moving only accepted IDs would leave one scene split across train/val.
        extra_image = dataset / "images/train/scene_a_extra.png"
        extra_label = dataset / "labels/train/scene_a_extra.txt"
        write_image(extra_image, 99)
        extra_label.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
        try:
            build_plan(dataset, scenes=["scene_a"], videos=[])
        except ValueError as exc:
            assert "accepted集合不一致" in str(exc) and "scene_a_extra" in str(exc)
        else:
            raise AssertionError("unreported scene-prefixed train output must be rejected")
        extra_image.unlink()
        extra_label.unlink()

        # Global formal train images and labels must be paired before planning.
        orphan_image = dataset / "images/train/unrelated_orphan.png"
        write_image(orphan_image, 100)
        try:
            build_plan(dataset, scenes=["scene_a"], videos=[])
        except ValueError as exc:
            assert "图片/标签不成对" in str(exc) and "unrelated_orphan" in str(exc)
        else:
            raise AssertionError("globally unpaired train output must be rejected")
        orphan_image.unlink()

        # Every selected scene must have an exact, regular train Manifest.
        scene_a_manifest.unlink()
        try:
            build_plan(dataset, scenes=["scene_a"], videos=[])
        except ValueError as exc:
            assert "train Manifest不存在或不是普通文件" in str(exc)
        else:
            raise AssertionError("missing selected-scene train Manifest must be rejected")
        write_manifest(scene_a_manifest, scene_a)

        write_manifest(scene_a_manifest, scene_a, split="val")
        try:
            build_plan(dataset, scenes=["scene_a"], videos=[])
        except ValueError as exc:
            assert "project.split必须严格为train" in str(exc)
        else:
            raise AssertionError("non-train selected-scene Manifest must be rejected")
        write_manifest(scene_a_manifest, scene_a)

        write_manifest(scene_a_manifest, scene_a_clip2)
        try:
            build_plan(dataset, scenes=["scene_a"], videos=[])
        except ValueError as exc:
            assert "project.scene与选中场次不一致" in str(exc)
        else:
            raise AssertionError("Manifest pointing at another scene must be rejected")
        write_manifest(scene_a_manifest, scene_a)

        primary_report = dataset / "project_reports/scene_a_train_report.json"
        primary_payload = json.loads(primary_report.read_text(encoding="utf-8"))
        outside_project_payload = dict(primary_payload)
        outside_project_payload["scene"] = str(root / "outside/scene_a")
        primary_report.write_text(json.dumps(outside_project_payload), encoding="utf-8")
        try:
            build_plan(dataset, scenes=["scene_a"], videos=[])
        except ValueError as exc:
            assert "无法从选中场次解析project_root" in str(exc)
        else:
            raise AssertionError("selected scene without a project root must be rejected")
        primary_report.write_text(json.dumps(primary_payload), encoding="utf-8")

        grouped = scan_groups(dataset)
        assert len(grouped) == 1
        assert grouped[0].capture_session_id == "session_video_a"
        for path in (
            dataset / "dataset.yaml",
            dataset / "project_reports/scene_a_train_report.json",
            second_report,
            scene_a_manifest,
            scene_a_clip2_manifest,
        ):
            path.chmod(0o664)
        plan = build_plan(dataset, scenes=["scene_a"], videos=[])
        assert plan["files"] == [(train_img, dataset / "images/val/scene_a_000001.png", train_lbl, dataset / "labels/val/scene_a_000001.txt")]
        apply_plan(plan)
        assert not train_img.exists() and not train_lbl.exists()
        assert (dataset / "images/val/scene_a_000001.png").exists()
        assert (dataset / "labels/val/scene_a_000001.txt").exists()
        assert not (dataset / "project_reports/scene_a_train_report.json").exists()
        assert not scene_a_manifest.exists()
        assert (project / "manifests/scene_a_val.yaml").is_file()
        report = json.loads((dataset / "project_reports/scene_a_val_report.json").read_text(encoding="utf-8"))
        assert report["split"] == "val"
        assert stat.S_IMODE((dataset / "project_reports/scene_a_val_report.json").stat().st_mode) == 0o664
        assert stat.S_IMODE((dataset / "dataset.yaml").stat().st_mode) == 0o664
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
            segments = scene / "project_reports/segments.json"
            segments.parent.mkdir(parents=True)
            segments.write_text(json.dumps({
                "format_version": 2,
                "scene": "..",
                "manifest": stored_path(manifest, segments.parent),
                "segments": [],
            }), encoding="utf-8")
            accepted_frames = []
            for index in range(frames):
                output_id = f"{scene_name}_{index:06d}"
                write_image(dataset / "images/train" / f"{output_id}.png", index)
                label = dataset / "labels/train" / f"{output_id}.txt"
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
                accepted_frames.append({"output_id": output_id, "status": "accepted"})
            stage = dataset / "_staging" / scene_name / "can_01"
            stage_label = stage / "labels/train" / f"{scene_name}_000000.txt"
            stage_label.parent.mkdir(parents=True)
            stage_label.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
            write_image(stage / "images/train" / f"{scene_name}_000000.png", 0)
            write_image(stage / "rendered_masks/train/class_000_can_01" / f"{scene_name}_000000.png", 1)
            write_image(stage / "visualizations/train/class_000_can_01/accepted" / f"{scene_name}_000000.jpg", 2)
            quality_report = stage / "quality_reports/train/class_000_can_01/quality_report.json"
            quality_report.parent.mkdir(parents=True)
            quality_report.write_text(json.dumps({
                "scene": str(scene), "split": "train", "instance_id": "can_01", "records": [],
            }), encoding="utf-8")
            report = dataset / "project_reports" / f"{scene_name}_train_report.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({
                "format_version": 2,
                "manifest": stored_path(manifest, report.parent),
                "scene": stored_path(scene, report.parent), "split": "train",
                "capture_session_id": f"session_{scene_name}",
                "source_video_id": f"video_{scene_name}",
                "instances": [{
                    "instance_id": "can_01", "class_id": 0, "class_name": "can",
                    "stage": stored_path(stage, report.parent),
                }],
                "frame_status_counts": {"accepted": frames, "review": 0, "rejected": 0},
                "frames": accepted_frames,
            }), encoding="utf-8")
            report.with_suffix(".csv").write_text("output_id,status\n", encoding="utf-8")
            marker = scene / "project_reports/manual_review_complete.json"
            marker.write_text(json.dumps({
                "schema_version": 2,
                "scene": scene_name,
                "class_name": "can",
                "completed_at": "2026-08-29T12:00:00+08:00",
                "export_report": stored_path(report, marker.parent),
                "export_report_mtime_ns": report.stat().st_mtime_ns,
                "frame_status_counts": {"accepted": frames, "review": 0, "rejected": 0},
            }), encoding="utf-8")

        for path in project.rglob("*"):
            if path.is_file():
                path.chmod(0o664)

        plan = build_auto_plan(dataset / "dataset.yaml", target_ratio=0.20)
        selected_scenes = {name for group in plan["groups"] for name in group.scene_names}
        assert selected_scenes == {"can_20260824_1000_01"}, selected_scenes
        assert len(plan["files"]) == 5
        assert len(plan["staging_updates"]) == 5
        assert len(plan["review_marker_updates"]) == 1
        assert len(plan["segments_updates"]) == 1
        before_failure = snapshot_tree(project)

        original_move = split_module.shutil.move
        moved_then_interrupted = False

        def interrupt_after_rename(source: str, destination: str) -> str:
            nonlocal moved_then_interrupted
            if not moved_then_interrupted:
                moved_then_interrupted = True
                Path(source).rename(Path(destination))
                raise KeyboardInterrupt("forced interrupt after rename")
            return original_move(source, destination)

        split_module.shutil.move = interrupt_after_rename
        try:
            try:
                apply_plan(plan)
            except KeyboardInterrupt as exc:
                assert "forced interrupt after rename" in str(exc)
            else:
                raise AssertionError("post-rename KeyboardInterrupt must propagate after rollback")
        finally:
            split_module.shutil.move = original_move
        assert snapshot_tree(project) == before_failure, (
            "post-rename interrupt must restore the complete snapshot"
        )

        interrupted_before_move = False

        def interrupt_without_move(source: str, destination: str) -> str:
            nonlocal interrupted_before_move
            if not interrupted_before_move:
                interrupted_before_move = True
                raise KeyboardInterrupt("forced interrupt before rename")
            return original_move(source, destination)

        split_module.shutil.move = interrupt_without_move
        try:
            try:
                apply_plan(plan)
            except KeyboardInterrupt as exc:
                assert "forced interrupt before rename" in str(exc)
            else:
                raise AssertionError("pre-rename KeyboardInterrupt must propagate after rollback")
        finally:
            split_module.shutil.move = original_move
        assert snapshot_tree(project) == before_failure, (
            "pre-rename interrupt must leave the complete snapshot unchanged"
        )
        rollback_scene = "can_20260824_1000_01"
        assert (project / "manifests" / f"{rollback_scene}_train.yaml").exists()
        rollback_report = json.loads((dataset / "project_reports" / f"{rollback_scene}_train_report.json").read_text(encoding="utf-8"))
        assert rollback_report["split"] == "train"
        rollback_scene_path = project / "data/scenes/can" / rollback_scene
        assert load_scene_human_review_completion(project, rollback_scene_path).valid

        apply_plan(plan)
        assert len(list((dataset / "images/train").glob("*.png"))) == 15
        assert len(list((dataset / "images/val").glob("*.png"))) == 5
        selected = "can_20260824_1000_01"
        assert not (project / "manifests" / f"{selected}_train.yaml").exists()
        val_manifest = project / "manifests" / f"{selected}_val.yaml"
        assert val_manifest.exists()
        assert yaml.safe_load(val_manifest.read_text(encoding="utf-8"))["project"]["split"] == "val"
        selected_scene = project / "data/scenes/can" / selected
        metadata = json.loads((selected_scene / "atec_capture_session.json").read_text(encoding="utf-8"))
        assert metadata["split"] == "val"
        old_report = dataset / "project_reports" / f"{selected}_train_report.json"
        val_report = dataset / "project_reports" / f"{selected}_val_report.json"
        assert not old_report.exists() and val_report.is_file()
        assert not old_report.with_suffix(".csv").exists() and val_report.with_suffix(".csv").is_file()
        report_data = json.loads(val_report.read_text(encoding="utf-8"))
        assert report_data["split"] == "val"
        assert (val_report.parent / report_data["manifest"]).resolve() == val_manifest.resolve()
        segments = selected_scene / "project_reports/segments.json"
        segments_data = json.loads(segments.read_text(encoding="utf-8"))
        assert (segments.parent / segments_data["manifest"]).resolve() == val_manifest.resolve()
        completion = load_scene_human_review_completion(project, selected_scene)
        assert completion.valid and completion.export_report_path == val_report
        stage = dataset / "_staging" / selected / "can_01"
        for parent_name in ("images", "labels", "quality_reports", "rendered_masks", "visualizations"):
            assert not (stage / parent_name / "train").exists()
            assert (stage / parent_name / "val").is_dir()
        moved_quality = stage / "quality_reports/val/class_000_can_01/quality_report.json"
        assert json.loads(moved_quality.read_text(encoding="utf-8"))["split"] == "val"
        for path in (
            dataset / "dataset.yaml",
            val_manifest,
            val_report,
            selected_scene / "atec_capture_session.json",
            segments,
            selected_scene / "project_reports/manual_review_complete.json",
            moved_quality,
        ):
            assert stat.S_IMODE(path.stat().st_mode) == 0o664, path

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
        invalid_marker = project / "data/scenes/can/can_20260824_1000_01/project_reports/manual_review_complete.json"
        for scene_name, frames in specs:
            scene = project / "data/scenes/can" / scene_name
            scene.mkdir(parents=True)
            write_manifest(project / "manifests" / f"{scene_name}_train.yaml", scene)
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
            if scene_name == "can_20260824_1000_01":
                invalid_marker.parent.mkdir(parents=True)
                invalid_marker.write_text(json.dumps({
                    "schema_version": 2, "scene": scene_name, "class_name": "can",
                    "export_report": stored_path(report, invalid_marker.parent),
                    "export_report_mtime_ns": 0,
                }), encoding="utf-8")
        ratio_plan = build_auto_plan(dataset / "dataset.yaml", target_ratio=0.20)
        ratio_scenes = {name for group in ratio_plan["groups"] for name in group.scene_names}
        assert ratio_scenes == {"can_20260824_1000_01"}, ratio_scenes
        assert len(ratio_plan["files"]) == 82
        assert not ratio_plan["review_marker_updates"]
        invalid_marker_before = invalid_marker.read_bytes()
        apply_plan(ratio_plan)
        assert invalid_marker.read_bytes() == invalid_marker_before, "invalid Review marker must not be rewritten"

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
