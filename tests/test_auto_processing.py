#!/usr/bin/env python3
"""CPU-only tests for one-click scene preflight and reporting."""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atec_pipeline.auto_processing import (  # noqa: E402
    AutoBatchRecord,
    build_manifest_init_args,
    plan_auto_scenes,
    preflight_scene,
    scene_is_locked,
    write_batch_report,
)
from atec_pipeline.gui_state import scene_workflow_state  # noqa: E402


def make_scene(root: Path, name: str, *, class_name: str = "can") -> Path:
    scene = root / "data/scenes" / class_name / name
    (scene / "rgb").mkdir(parents=True)
    (scene / "depth").mkdir()
    (scene / "rgb/000000.png").write_bytes(b"rgb")
    (scene / "depth/000000.png").write_bytes(b"depth")
    (scene / "metadata.json").write_text(
        json.dumps({"frames": [{"id": "000000", "color_timestamp_ms": 1000.0, "depth_timestamp_ms": 1000.0}]}), encoding="utf-8"
    )
    return scene


def write_manifest(root: Path, scene: Path, *, split: str = "train") -> Path:
    path = root / "manifests" / f"{scene.name}_{split}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "project:\n"
        f"  scene: ../data/scenes/{scene.parent.name}/{scene.name}\n"
        "  output: ../datasets/atec_yolo11_seg\n"
        f"  split: {split}\n"
        f"  capture_session_id: session_{scene.name}\n"
        f"  source_video_id: {scene.name}_clip_01\n"
        f"  name_prefix: {scene.name}_\n"
        "classes:\n"
        f"  0: {scene.parent.name}\n"
        "instances:\n"
        f"- instance_id: {scene.parent.name}_01\n"
        "  class_id: 0\n"
        f"  class_name: {scene.parent.name}\n"
        "  tracker: sam2\n"
        f"  key_mask_dir: ../data/key_masks/{scene.name}/{scene.parent.name}_01\n",
        encoding="utf-8",
    )
    return path


def write_segments(root: Path, scene: Path, *, mask_exists: bool) -> Path:
    mask = root / "data/key_masks" / scene.name / f"{scene.parent.name}_01/000000.png"
    report = scene / "project_reports/segments.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"scene": str(scene), "segments": [{
            "segment_id": 0,
            "start_id": "000000",
            "end_id": "000000",
            "required_key_mask_paths": {f"{scene.parent.name}_01": str(mask)},
        }]}),
        encoding="utf-8",
    )
    if mask_exists:
        mask.parent.mkdir(parents=True, exist_ok=True)
        mask.write_bytes(b"mask")
    return report


def write_export(root: Path, scene: Path, *, accepted: int, review: int = 0) -> Path:
    report = root / "datasets/atec_yolo11_seg/project_reports" / f"{scene.name}_train_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"frame_status_counts": {
        "accepted": accepted, "review": review, "rejected": 1 if accepted == 0 else 0,
    }}), encoding="utf-8")
    return report


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="atec_auto_processing_") as tmp:
        root = Path(tmp) / "project"

        safe = make_scene(root, "can_safe")
        write_manifest(root, safe)
        write_segments(root, safe, mask_exists=True)
        (safe / "rgb/000001.png").write_bytes(b"interrupted")
        safe_plan = preflight_scene(root, scene_workflow_state(root, safe))
        assert safe_plan.action == "run"
        assert safe_plan.quarantine_path is not None
        assert not (safe / "rgb/000001.png").exists()
        assert (safe_plan.quarantine_path / "rgb/000001.png").is_file()

        unsafe = make_scene(root, "can_unsafe")
        write_manifest(root, unsafe)
        write_segments(root, unsafe, mask_exists=True)
        (unsafe / "depth/000000.png").unlink()
        unsafe_plan = preflight_scene(root, scene_workflow_state(root, unsafe))
        assert unsafe_plan.action == "manual"
        assert "000000:depth" in unsafe_plan.reason

        bad_timestamps = make_scene(root, "can_bad_timestamps")
        write_manifest(root, bad_timestamps)
        write_segments(root, bad_timestamps, mask_exists=True)
        (bad_timestamps / "metadata.json").write_text(
            json.dumps({"frames": [{"id": "000000", "color_timestamp_ms": 1000.0}]}),
            encoding="utf-8",
        )
        timestamp_plan = preflight_scene(root, scene_workflow_state(root, bad_timestamps))
        assert timestamp_plan.action == "manual"
        assert "时间戳" in timestamp_plan.reason

        missing_mask = make_scene(root, "can_missing_mask")
        write_manifest(root, missing_mask)
        write_segments(root, missing_mask, mask_exists=False)
        missing_plan = preflight_scene(root, scene_workflow_state(root, missing_mask))
        assert missing_plan.action == "manual"
        assert "关键帧" in missing_plan.reason

        pending = make_scene(root, "can_pending")
        write_manifest(root, pending)
        write_segments(root, pending, mask_exists=True)
        assert preflight_scene(root, scene_workflow_state(root, pending)).action == "run"

        failed = make_scene(root, "can_failed")
        write_manifest(root, failed)
        write_segments(root, failed, mask_exists=True)
        write_export(root, failed, accepted=0)
        assert preflight_scene(root, scene_workflow_state(root, failed)).action == "run"

        ready = make_scene(root, "can_ready")
        write_manifest(root, ready)
        write_segments(root, ready, mask_exists=True)
        write_export(root, ready, accepted=1)
        ready_plan = preflight_scene(root, scene_workflow_state(root, ready))
        assert ready_plan.action == "skip"

        needs_review = make_scene(root, "can_review")
        write_manifest(root, needs_review)
        write_segments(root, needs_review, mask_exists=True)
        write_export(root, needs_review, accepted=1, review=2)
        review_plan = preflight_scene(root, scene_workflow_state(root, needs_review))
        assert review_plan.action == "manual"
        assert "Review" in review_plan.reason

        no_segments = make_scene(root, "can_segment")
        write_manifest(root, no_segments)
        assert preflight_scene(root, scene_workflow_state(root, no_segments)).action == "segment"

        malformed = make_scene(root, "can_bad_manifest")
        malformed_manifest = root / "manifests/can_bad_manifest_train.yaml"
        malformed_manifest.parent.mkdir(parents=True, exist_ok=True)
        malformed_manifest.write_text("project: [not, a, mapping]\n", encoding="utf-8")
        malformed_plan = preflight_scene(root, scene_workflow_state(root, malformed))
        assert malformed_plan.action == "manual"
        assert "Manifest" in malformed_plan.reason

        repairable = make_scene(root, "can_repair_manifest")
        repairable_manifest = write_manifest(root, repairable)
        repairable_data = yaml.safe_load(repairable_manifest.read_text(encoding="utf-8"))
        repairable_data["project"]["scene"] = "../data/scenes/can/wrong_scene"
        repairable_data["project"].pop("output")
        repairable_data["instances"][0].pop("class_id")
        repairable_data["instances"][0].pop("class_name")
        repairable_manifest.write_text(
            yaml.safe_dump(repairable_data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        (repairable / "atec_capture_session.json").write_text(json.dumps({
            "scene_name": repairable.name,
            "class_name": "can",
            "split": "train",
            "capture_session_id": f"session_{repairable.name}",
            "source_video_id": f"{repairable.name}_clip_01",
        }), encoding="utf-8")
        repairable_plan = preflight_scene(root, scene_workflow_state(root, repairable))
        assert repairable_plan.action == "segment"
        repaired_data = yaml.safe_load(repairable_manifest.read_text(encoding="utf-8"))
        assert (repairable_manifest.parent / repaired_data["project"]["scene"]).resolve() == repairable.resolve()
        assert repaired_data["project"]["output"] == "../datasets/atec_yolo11_seg"
        assert repaired_data["instances"][0]["class_id"] == 0
        assert repaired_data["instances"][0]["class_name"] == "can"
        assert list(repairable_manifest.parent.glob(repairable_manifest.name + ".auto_processing_backup_*"))

        mixed = make_scene(root, "can_mixed_manifest")
        mixed_manifest = write_manifest(root, mixed)
        mixed_data = yaml.safe_load(mixed_manifest.read_text(encoding="utf-8"))
        mixed_data["classes"][1] = "watermelon_rind"
        mixed_data["instances"].append({
            "instance_id": "watermelon_rind_01",
            "class_id": 1,
            "class_name": "watermelon_rind",
            "tracker": "mask_sequence",
            "mask_dir": "../data/tracked_masks/can_mixed_manifest/watermelon_rind_01",
        })
        mixed_manifest.write_text(
            yaml.safe_dump(mixed_data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        mixed_plan = preflight_scene(root, scene_workflow_state(root, mixed))
        assert mixed_plan.action == "segment", "合法的混合类别 Manifest 不能被场景目录类别误伤"

        no_manifest = make_scene(root, "can_init")
        (no_manifest / "atec_capture_session.json").write_text(json.dumps({
            "scene_name": no_manifest.name,
            "class_name": "can",
            "split": "train",
            "capture_session_id": "session_can_init",
            "source_video_id": "can_init_clip_01",
        }), encoding="utf-8")
        init_plan = preflight_scene(root, scene_workflow_state(root, no_manifest))
        assert init_plan.action == "init"
        init_args = build_manifest_init_args(root, init_plan)
        assert init_args[:2] == ["init", str(root)]
        assert "--only-class" in init_args and "can" in init_args
        assert "--capture-session-id" in init_args and "session_can_init" in init_args

        ambiguous = make_scene(root, "can_ambiguous")
        (ambiguous / "atec_capture_session.json").write_text(json.dumps({
            "scene_name": ambiguous.name, "class_name": "red_bin", "split": "train",
        }), encoding="utf-8")
        ambiguous_plan = preflight_scene(root, scene_workflow_state(root, ambiguous))
        assert ambiguous_plan.action == "manual"
        assert "不一致" in ambiguous_plan.reason or "缺少" in ambiguous_plan.reason

        plans = plan_auto_scenes(root, [
            scene_workflow_state(root, ready), scene_workflow_state(root, pending),
            scene_workflow_state(root, missing_mask),
        ])
        assert [plan.action for plan in plans] == ["skip", "run", "manual"]

        assert scene_is_locked(True, "can_pending", "can_pending")
        assert not scene_is_locked(True, "can_pending", "can_ready")
        assert not scene_is_locked(False, "can_pending", "can_pending")

        records = [
            AutoBatchRecord("can_ready", "skip", "completed", "已有有效导出"),
            AutoBatchRecord("can_failed", "run", "failed", "exit=1", exit_code=1),
            AutoBatchRecord("can_missing_mask", "preflight", "manual", "缺少关键帧 Mask"),
            AutoBatchRecord("can_pending", "run", "success", "accepted=10"),
        ]
        started = datetime(2026, 8, 26, 10, 0, tzinfo=timezone.utc)
        finished = datetime(2026, 8, 26, 10, 5, tzinfo=timezone.utc)
        report_path = write_batch_report(root, records, started_at=started, finished_at=finished, cancelled=True)
        latest = root / "project_reports/auto_processing/latest.json"
        assert report_path.is_file() and latest.is_file()
        payload = json.loads(latest.read_text(encoding="utf-8"))
        assert payload["cancelled"] is True
        assert payload["summary"] == {"success": 1, "skipped": 1, "failed": 1, "manual": 1, "cancelled": 0}
        assert len(payload["scenes"]) == 4

    print("AUTO_PROCESSING_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
