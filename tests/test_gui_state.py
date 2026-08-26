#!/usr/bin/env python3
"""CPU-only tests for the desktop App's path and staging state layer."""
from __future__ import annotations

import tempfile
from datetime import datetime
import json
from pathlib import Path
import shutil
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from atec_pipeline.gui_state import (  # noqa: E402
    SceneWorkflowState,
    discard_failed_empty_capture,
    find_best_weights,
    has_paired_rgbd_frames,
    load_export_summary,
    load_mask_progress,
    load_object_classes,
    load_training_summary,
    load_scene_human_review_completion,
    mark_scene_human_review_complete,
    clear_scene_human_review_complete,
    make_capture_session,
    paired_rgbd_frame_count,
    inspect_scene_integrity,
    repair_scene_integrity,
    scene_workflow_state,
    scene_for_number,
    summarize_scene_states,
)


def main() -> int:
    classes = load_object_classes(ROOT / "configs/atec_objects.yaml")
    assert len(classes) == 7
    assert classes[0].name == "can" and classes[0].chinese_name == "易拉罐"
    assert classes[6].name == "red_bin" and classes[6].class_id == 6
    assert scene_for_number(classes, 1).name == "can"
    assert scene_for_number(classes, 7).name == "red_bin"

    summary_states = [
        SceneWorkflowState(
            scene_name="can_waiting",
            class_name="can",
            scene_dir=Path("/tmp/can_waiting"),
            split="train",
            manifest_path=None,
            segments_path=Path("/tmp/can_waiting/segments.json"),
            export_report_path=None,
            rgb_frames=77,
            depth_frames=77,
            paired_frames=77,
            mask_completed=1,
            mask_total=1,
            group="keyframes_complete",
            code="pending_export",
            color="blue",
            detail="待 SAM2 传播",
        ),
        SceneWorkflowState(
            scene_name="can_ready",
            class_name="can",
            scene_dir=Path("/tmp/can_ready"),
            split="train",
            manifest_path=None,
            segments_path=Path("/tmp/can_ready/segments.json"),
            export_report_path=Path("/tmp/can_ready/report.json"),
            rgb_frames=90,
            depth_frames=90,
            paired_frames=90,
            mask_completed=1,
            mask_total=1,
            group="keyframes_complete",
            code="export_needs_review",
            color="yellow",
            detail="部分帧待检查",
            accepted=53,
            review=7,
            rejected=30,
        ),
        SceneWorkflowState(
            scene_name="can_manual",
            class_name="can",
            scene_dir=Path("/tmp/can_manual"),
            split="train",
            manifest_path=None,
            segments_path=Path("/tmp/can_manual/segments.json"),
            export_report_path=None,
            rgb_frames=20,
            depth_frames=20,
            paired_frames=20,
            mask_completed=0,
            mask_total=1,
            group="needs_manual",
            code="masks_missing",
            color="gray",
            detail="缺少关键帧",
        ),
    ]
    class_summary = summarize_scene_states(summary_states)
    assert class_summary.scene_count == 3
    assert class_summary.paired_frames == 187
    assert class_summary.processed_frames == 90
    assert (class_summary.accepted, class_summary.review, class_summary.rejected) == (53, 7, 30)
    assert class_summary.pending_propagation_scenes == 1
    assert class_summary.needs_manual_scenes == 1

    with tempfile.TemporaryDirectory(prefix="atec_gui_state_") as tmp:
        project_root = Path(tmp) / "project"
        fixed_now = datetime(2026, 8, 23, 14, 30)
        first = make_capture_session(project_root, classes[0], "train", now=fixed_now)
        assert first.scene_name == "can_20260823_1430_01"
        assert first.staging_dir == project_root / "data/.staging/can_20260823_1430_01"
        first.staging_dir.mkdir(parents=True)
        (first.staging_dir / "rgb").mkdir()
        (first.staging_dir / "depth").mkdir()
        (first.staging_dir / "rgb/000000.png").write_bytes(b"rgb")
        (first.staging_dir / "rgb/000001.png").write_bytes(b"rgb-only")
        (first.staging_dir / "depth/000000.png").write_bytes(b"depth")
        assert paired_rgbd_frame_count(first.staging_dir) == 1
        assert has_paired_rgbd_frames(first.staging_dir)

        second = make_capture_session(project_root, classes[0], "train", now=fixed_now)
        assert second.scene_name == "can_20260823_1430_02"

        saved = first.save()
        assert saved == first.scene_dir
        assert saved == project_root / "data/scenes/can/can_20260823_1430_01"
        assert (saved / "rgb/000000.png").read_bytes() == b"rgb"
        assert not first.staging_dir.exists()

        discarded = make_capture_session(project_root, classes[1], "train", now=fixed_now)
        discarded.staging_dir.mkdir(parents=True)
        (discarded.staging_dir / "metadata.json").write_text("{}", encoding="utf-8")
        discarded.discard()
        assert not discarded.staging_dir.exists()

        failed_empty = make_capture_session(project_root, classes[2], "train", now=fixed_now)
        (failed_empty.staging_dir / "rgb").mkdir(parents=True)
        (failed_empty.staging_dir / "depth").mkdir()
        assert discard_failed_empty_capture(failed_empty)
        assert not failed_empty.staging_dir.exists()

        failed_partial = make_capture_session(project_root, classes[3], "train", now=fixed_now)
        (failed_partial.staging_dir / "rgb").mkdir(parents=True)
        (failed_partial.staging_dir / "depth").mkdir()
        (failed_partial.staging_dir / "rgb/000000.png").write_bytes(b"keep-for-debug")
        assert not discard_failed_empty_capture(failed_partial)
        assert failed_partial.staging_dir.exists(), "partial data must be preserved for diagnosis"
        failed_partial.discard()

        # Integrity checks distinguish an unrecorded interrupted-write orphan
        # from a metadata-recorded frame whose RGB/depth payload is missing.
        integrity_scene = project_root / "data/scenes/can/integrity_scene"
        (integrity_scene / "rgb").mkdir(parents=True)
        (integrity_scene / "depth").mkdir()
        (integrity_scene / "rgb/000000.png").write_bytes(b"rgb-0")
        (integrity_scene / "depth/000000.png").write_bytes(b"depth-0")
        (integrity_scene / "metadata.json").write_text(
            json.dumps({"frames": [{"id": "000000"}]}), encoding="utf-8"
        )
        complete = inspect_scene_integrity(integrity_scene)
        assert complete.is_complete
        assert not complete.safe_orphans
        assert not complete.unsafe_recorded_missing

        (integrity_scene / "rgb/000001.png").write_bytes(b"interrupted-rgb")
        orphan = inspect_scene_integrity(integrity_scene)
        assert not orphan.is_complete
        assert orphan.can_auto_repair
        assert orphan.orphan_rgb == (integrity_scene / "rgb/000001.png",)
        assert orphan.safe_orphans == orphan.orphan_rgb
        quarantine = integrity_scene / ".integrity_quarantine/test-repair"
        repaired_to = repair_scene_integrity(orphan, quarantine_root=quarantine)
        assert repaired_to == quarantine
        assert not (integrity_scene / "rgb/000001.png").exists()
        assert (quarantine / "rgb/000001.png").read_bytes() == b"interrupted-rgb"
        assert inspect_scene_integrity(integrity_scene).is_complete

        # A metadata-recorded frame is not safe to discard when one stream is
        # missing; automatic repair must refuse it.
        unsafe_scene = project_root / "data/scenes/can/unsafe_integrity_scene"
        (unsafe_scene / "rgb").mkdir(parents=True)
        (unsafe_scene / "depth").mkdir()
        (unsafe_scene / "rgb/000000.png").write_bytes(b"rgb")
        (unsafe_scene / "metadata.json").write_text(
            json.dumps({"frames": [{"id": "000000"}]}), encoding="utf-8"
        )
        unsafe = inspect_scene_integrity(unsafe_scene)
        assert not unsafe.can_auto_repair
        assert unsafe.unsafe_recorded_missing == ("000000:depth",)
        try:
            repair_scene_integrity(unsafe)
        except ValueError as exc:
            assert "不能自动修复" in str(exc)
        else:
            raise AssertionError("metadata-recorded missing data must not be auto-repaired")

        # If a multi-file repair fails, already moved files are restored.
        rollback_scene = project_root / "data/scenes/can/rollback_integrity_scene"
        (rollback_scene / "rgb").mkdir(parents=True)
        (rollback_scene / "depth").mkdir()
        (rollback_scene / "rgb/000000.png").write_bytes(b"rgb-0")
        (rollback_scene / "depth/000000.png").write_bytes(b"depth-0")
        (rollback_scene / "rgb/000001.png").write_bytes(b"rgb-orphan")
        (rollback_scene / "depth/000002.png").write_bytes(b"depth-orphan")
        (rollback_scene / "metadata.json").write_text(
            json.dumps({"frames": [{"id": "000000"}]}), encoding="utf-8"
        )
        rollback_report = inspect_scene_integrity(rollback_scene)
        real_move = shutil.move
        forward_moves = 0

        def fail_second_forward(src, dst):
            nonlocal forward_moves
            if ".integrity_quarantine" not in str(src):
                forward_moves += 1
                if forward_moves == 2:
                    raise OSError("simulated move failure")
            return real_move(src, dst)

        with patch("atec_pipeline.gui_state.shutil.move", side_effect=fail_second_forward):
            try:
                repair_scene_integrity(
                    rollback_report,
                    quarantine_root=rollback_scene / ".integrity_quarantine/rollback",
                )
            except RuntimeError as exc:
                assert "已回滚" in str(exc)
            else:
                raise AssertionError("repair must surface and roll back a partial move failure")
        assert (rollback_scene / "rgb/000001.png").read_bytes() == b"rgb-orphan"
        assert (rollback_scene / "depth/000002.png").read_bytes() == b"depth-orphan"

        conflict = make_capture_session(project_root, classes[0], "train", now=fixed_now, scene_name=first.scene_name)
        conflict.staging_dir.mkdir(parents=True)
        try:
            conflict.save()
        except FileExistsError:
            pass
        else:
            raise AssertionError("formalizing a duplicate scene must be rejected")
        conflict.discard()

        # Key-mask progress must be derived from actual required files, not stale JSON flags.
        report = project_root / "scene/project_reports/segments.json"
        mask_a = project_root / "key_masks/can_01/000000.png"
        mask_b = project_root / "key_masks/can_01/000050.png"
        report.parent.mkdir(parents=True)
        report.write_text(
            "{\n"
            '  "segments": [\n'
            '    {"segment_id": 0, "start_id": "000000", "end_id": "000049", '
            f'"required_key_mask_paths": {{"can_01": "{mask_a}"}}, "missing_key_masks": ["can_01"]}},\n'
            '    {"segment_id": 1, "start_id": "000050", "end_id": "000099", '
            f'"required_key_mask_paths": {{"can_01": "{mask_b}"}}, "missing_key_masks": ["can_01"]}}\n'
            "  ]\n}",
            encoding="utf-8",
        )
        mask_a.parent.mkdir(parents=True)
        mask_a.write_bytes(b"mask")
        progress = load_mask_progress(report)
        assert progress.completed_required == 1 and progress.total_required == 2
        assert progress.segments[0].complete
        assert not progress.segments[1].complete
        mask_b.write_bytes(b"mask")
        assert load_mask_progress(report).complete

        # Shared snapshots may contain absolute paths from the producer
        # machine.  Re-anchor those paths at the local projects/atec_real
        # root instead of reporting valid masks as missing.
        portable_scene = project_root / "data/scenes/can/portable_scene"
        portable_report = portable_scene / "project_reports/segments.json"
        portable_mask = project_root / "data/key_masks/portable_scene/can_01/000000.png"
        portable_report.parent.mkdir(parents=True)
        portable_mask.parent.mkdir(parents=True)
        portable_mask.write_bytes(b"mask")
        portable_report.write_text(
            json.dumps({"segments": [{
                "segment_id": 0, "start_id": "000000", "end_id": "000000",
                "required_key_mask_paths": {
                    "can_01": "/producer-machine/projects/atec_real/data/key_masks/portable_scene/can_01/000000.png"
                },
            }]}),
            encoding="utf-8",
        )
        portable_progress = load_mask_progress(portable_report)
        assert portable_progress.complete

        # Training effect summary uses validation mask metrics and locates best.pt.
        run_dir = project_root / "runs/segment/demo"
        (run_dir / "weights").mkdir(parents=True)
        (run_dir / "weights/best.pt").write_bytes(b"weights")
        (run_dir / "results.csv").write_text(
            "epoch,metrics/precision(M),metrics/recall(M),metrics/mAP50(M),metrics/mAP50-95(M)\n"
            "0,0.50,0.40,0.45,0.30\n"
            "1,0.70,0.60,0.65,0.55\n",
            encoding="utf-8",
        )
        summary = load_training_summary(project_root / "runs/segment", "demo")
        assert summary is not None
        assert summary.best_epoch == 1
        assert summary.mask_map50_95 == 0.55
        assert summary.best_weights == run_dir / "weights/best.pt"
        assert find_best_weights(project_root / "runs/segment", "demo") == run_dir / "weights/best.pt"
        assert find_best_weights(project_root / "runs/segment", "missing") == run_dir / "weights/best.pt"
        assert find_best_weights(project_root / "models", "") is None, "official/base models must not be used as training results"

        export_report = project_root / "datasets/out/project_reports/scene_train_report.json"
        export_report.parent.mkdir(parents=True)
        export_report.write_text(
            '{"frame_status_counts": {"accepted": 80, "review": 15, "rejected": 5}, "frames": []}',
            encoding="utf-8",
        )
        export = load_export_summary(export_report)
        assert export is not None
        assert export.total == 100 and export.accepted == 80
        assert export.needs_review == 20
        assert not export.all_accepted

        # Scene state is computed centrally and distinguishes manual work from
        # completed keyframes plus downstream export quality.
        scene = project_root / "data/scenes/can/state_scene"
        (scene / "rgb").mkdir(parents=True)
        (scene / "depth").mkdir()
        (scene / "rgb/000000.png").write_bytes(b"rgb")
        (scene / "depth/000000.png").write_bytes(b"depth")
        state = scene_workflow_state(project_root, scene)
        assert state.group == "needs_manual" and state.code == "manifest_missing"

        manifest = project_root / "manifests/state_scene_train.yaml"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            "project:\n  scene: data/scenes/can/state_scene\n  split: train\n",
            encoding="utf-8",
        )
        state = scene_workflow_state(project_root, scene)
        assert state.code == "segments_missing"

        state_segments = scene / "project_reports/segments.json"
        state_mask = project_root / "data/key_masks/state_scene/can_01/000000.png"
        state_segments.parent.mkdir(parents=True)
        state_segments.write_text(
            json.dumps({"segments": [{
                "segment_id": 0, "start_id": "000000", "end_id": "000000",
                "required_key_mask_paths": {"can_01": str(state_mask)},
            }]}),
            encoding="utf-8",
        )
        state = scene_workflow_state(project_root, scene)
        assert state.code == "masks_missing" and state.mask_completed == 0 and state.mask_total == 1

        state_mask.parent.mkdir(parents=True)
        state_mask.write_bytes(b"mask")
        state = scene_workflow_state(project_root, scene)
        assert state.group == "keyframes_complete" and state.code == "pending_export" and state.color == "blue"

        report_dir = project_root / "datasets/atec_yolo11_seg/project_reports"
        report_dir.mkdir(parents=True)
        state_report = report_dir / "state_scene_train_report.json"
        state_report.write_text(
            json.dumps({"frame_status_counts": {"accepted": 0, "review": 0, "rejected": 1}}),
            encoding="utf-8",
        )
        state = scene_workflow_state(project_root, scene)
        assert state.code == "export_failed" and state.color == "red" and not state.training_eligible

        state_report.write_text(
            json.dumps({"frame_status_counts": {"accepted": 1, "review": 1, "rejected": 0}}),
            encoding="utf-8",
        )
        state = scene_workflow_state(project_root, scene)
        assert state.code == "export_needs_review" and state.color == "yellow" and state.training_eligible

        state_report.write_text(
            json.dumps({"frame_status_counts": {"accepted": 1, "review": 0, "rejected": 0}}),
            encoding="utf-8",
        )
        state = scene_workflow_state(project_root, scene)
        assert state.code == "dataset_ready" and state.color == "green" and state.training_eligible

        # Human Review completion is explicit, persists beside scene reports,
        # and is bound to the exact export report version that was checked.
        completion = load_scene_human_review_completion(project_root, scene)
        assert not completion.valid and completion.reason == "marker_missing"
        marker = mark_scene_human_review_complete(project_root, scene)
        assert marker == scene / "project_reports/manual_review_complete.json"
        assert marker.is_file()
        marker_data = json.loads(marker.read_text(encoding="utf-8"))
        assert marker_data["scene"] == "state_scene"
        assert marker_data["class_name"] == "can"
        assert marker_data["frame_status_counts"] == {"accepted": 1, "review": 0, "rejected": 0}

        completion = load_scene_human_review_completion(project_root, scene)
        assert completion.valid and completion.marker_path == marker
        state = scene_workflow_state(project_root, scene)
        assert state.group == "human_reviewed"
        assert state.human_review_complete
        assert state.code == "dataset_ready"

        # Any re-export changes the report mtime and invalidates the old review marker.
        report_payload = json.loads(state_report.read_text(encoding="utf-8"))
        report_payload["review_revision"] = 2
        state_report.write_text(json.dumps(report_payload), encoding="utf-8")
        completion = load_scene_human_review_completion(project_root, scene)
        assert not completion.valid and completion.reason == "export_report_changed"
        state = scene_workflow_state(project_root, scene)
        assert state.group == "keyframes_complete"
        assert not state.human_review_complete

        # Re-marking binds to the new report; explicit cancellation removes it.
        mark_scene_human_review_complete(project_root, scene)
        assert load_scene_human_review_completion(project_root, scene).valid
        assert clear_scene_human_review_complete(scene)
        assert not marker.exists()
        assert not clear_scene_human_review_complete(scene)

        # A scene without an export report cannot be falsely marked reviewed.
        no_export_scene = project_root / "data/scenes/can/no_export_scene"
        (no_export_scene / "rgb").mkdir(parents=True)
        (no_export_scene / "depth").mkdir()
        try:
            mark_scene_human_review_complete(project_root, no_export_scene)
        except FileNotFoundError as exc:
            assert "导出报告" in str(exc)
        else:
            raise AssertionError("scene without export report must not be marked reviewed")

    print("GUI_STATE_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
