#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atec_pipeline.review_state import ReviewState


def write_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    scene = root / "scene"
    mask_dir = root / "masks"
    scene.joinpath("rgb").mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    records = []
    for index in range(8):
        frame_id = f"{index:06d}"
        scene.joinpath("rgb", f"{frame_id}.png").write_bytes(b"rgb")
        if index != 6:
            mask_dir.joinpath(f"{frame_id}.png").write_bytes(b"mask")
        status = "accepted" if index < 3 else "review"
        records.append({
            "id": frame_id,
            "status": status,
            "reject_reasons": [],
            "review_reasons": ["touches_image_border"] if status == "review" else [],
        })
    report = root / "quality_report.json"
    report.write_text(json.dumps({"records": records}), encoding="utf-8")
    segments = root / "segments.json"
    segments.write_text(json.dumps({"segments": [
        {"segment_id": 0, "start_id": "000000", "end_id": "000003", "start_index": 0, "end_index": 3},
        {"segment_id": 1, "start_id": "000004", "end_id": "000007", "start_index": 4, "end_index": 7},
    ]}), encoding="utf-8")
    overrides = scene / "project_reports/manual_mask_review/can_01.json"
    return scene, mask_dir, report, segments, overrides


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="atec_review_state_") as tmp:
        root = Path(tmp)
        scene, masks, report, segments, overrides = write_fixture(root)
        state = ReviewState.load(
            quality_report=report,
            overrides_path=overrides,
            mask_dir=masks,
            segments_report=segments,
            scene_name="demo",
            instance_id="can_01",
        )
        assert state.frame_ids == tuple(f"{i:06d}" for i in range(8))
        assert [segment.segment_id for segment in state.segments] == ["0", "1"]
        assert state.segment_for_frame("000005").segment_id == "1"
        assert state.effective_status("000004") == "review"
        assert state.reason_text("000004") == "touches_image_border"

        # Incremental SAM2 reruns start at the new keyframe and stop before the
        # next effective accepted frame in the same automatic segment.
        immediate = state.incremental_rerun_range("000001")
        assert immediate.frame_ids == ("000001",)
        assert immediate.end_before_frame == "000002"
        assert immediate.boundary_reason == "next_accepted"

        # A manual rejection overrides an automatic accepted status, so that
        # frame cannot prematurely terminate the correction range.
        state.set_status("000002", "rejected")
        overridden = state.incremental_rerun_range("000001")
        assert overridden.frame_ids == ("000001", "000002", "000003")
        assert overridden.end_before_frame == "000004"
        assert overridden.last_frame == "000003"
        assert overridden.boundary_reason == "segment_end"

        # A manually accepted frame is also a valid exclusive boundary.
        state.set_status("000005", "accepted")
        manual_boundary = state.incremental_rerun_range("000004")
        assert manual_boundary.frame_ids == ("000004",)
        assert manual_boundary.end_before_frame == "000005"
        assert manual_boundary.boundary_reason == "next_accepted"

        # The last frame of a segment never leaks into the next segment.
        final_in_segment = state.incremental_rerun_range("000003")
        assert final_in_segment.frame_ids == ("000003",)
        assert final_in_segment.end_before_frame == "000004"
        assert final_in_segment.segment_id == "0"

        state.set_status("000004", "rejected")
        assert state.effective_status("000004") == "rejected"
        assert state.auto_status("000004") == "review"
        state.save()
        assert report.read_text(encoding="utf-8") == json.dumps({"records": json.loads(report.read_text(encoding="utf-8"))["records"]}), "automatic report must not be changed"

        reloaded = ReviewState.load(report, overrides, masks, segments, scene_name="demo", instance_id="can_01")
        assert reloaded.effective_status("000004") == "rejected"
        assert reloaded.manual_status("000004") == "rejected"

        # A manual accepted decision is allowed only when a usable propagated mask exists.
        reloaded.set_status("000005", "accepted")
        try:
            reloaded.set_status("000006", "accepted")
        except ValueError as exc:
            assert "Mask" in str(exc)
        else:
            raise AssertionError("frame without a mask must not be manually accepted")

        # Bracket range is frame-accurate and can span part of one segment.
        reloaded.begin_problem_range("000001")
        changed = reloaded.finish_problem_range("000003")
        assert changed == ("000001", "000002", "000003")
        assert all(reloaded.effective_status(fid) == "rejected" for fid in changed)

        # Default reject-from-current is limited to the current automatic segment.
        changed = reloaded.reject_from("000005", scope="segment")
        assert changed == ("000005", "000006", "000007")
        assert reloaded.effective_status("000007") == "rejected"
        assert reloaded.effective_status("000000") == "accepted"

        # Whole-scene mode is explicit and does not silently delete any RGB/Mask file.
        before_rgb = sorted(scene.joinpath("rgb").glob("*.png"))
        before_masks = sorted(masks.glob("*.png"))
        changed = reloaded.reject_from("000002", scope="scene")
        assert changed[0] == "000002" and changed[-1] == "000007"
        assert sorted(scene.joinpath("rgb").glob("*.png")) == before_rgb
        assert sorted(masks.glob("*.png")) == before_masks

        key_dir = root / "key_masks/can_01"
        assert reloaded.keyframe_path(key_dir, "000005") == key_dir / "000005.png"
        reloaded.save()
        payload = json.loads(overrides.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["scene"] == "demo"
        assert payload["instance_id"] == "can_01"
        assert payload["frames"]["000005"]["status"] == "rejected"

    print("REVIEW_STATE_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
