#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from atec_pipeline.review_state import ReviewState
from review_mask_sequence import KEY_ACTIONS, PlayerModel, render_review_frame


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="atec_review_player_") as tmp:
        root = Path(tmp)
        scene = root / "scene"
        rgb_dir = scene / "rgb"
        masks = root / "masks"
        rgb_dir.mkdir(parents=True)
        masks.mkdir()
        records = []
        for index in range(6):
            frame_id = f"{index:06d}"
            image = np.full((90, 140, 3), 35 + index * 5, np.uint8)
            cv2.imwrite(str(rgb_dir / f"{frame_id}.png"), image)
            mask = np.zeros((90, 140), np.uint8)
            cv2.rectangle(mask, (35 + index, 20), (75 + index, 65), 255, -1)
            cv2.imwrite(str(masks / f"{frame_id}.png"), mask)
            records.append({
                "id": frame_id,
                "status": "review" if index == 4 else "accepted",
                "review_reasons": ["touches_image_border"] if index == 4 else [],
                "reject_reasons": [],
            })
        report = root / "quality_report.json"
        report.write_text(json.dumps({"scene": str(scene), "instance_id": "can_01", "records": records}), encoding="utf-8")
        segments = root / "segments.json"
        segments.write_text(json.dumps({"segments": [
            {"segment_id": 0, "start_id": "000000", "end_id": "000002"},
            {"segment_id": 1, "start_id": "000003", "end_id": "000005"},
        ]}), encoding="utf-8")
        overrides = scene / "project_reports/manual_mask_review/can_01.json"
        state = ReviewState.load(report, overrides, masks, segments, scene_name="scene", instance_id="can_01")
        player = PlayerModel(state, selected_segment_ids=("1",))
        assert player.playable_frame_ids == ("000003", "000004", "000005")
        assert player.current_frame_id == "000003"
        assert player.move(1) == "000004"
        assert player.move(-1) == "000003"
        assert not player.changed
        player.set_current_status("rejected")
        assert player.changed
        assert state.effective_status("000003") == "rejected"
        player.playing = True
        assert player.advance_for_playback() == "000004"
        assert player.advance_for_playback() == "000005"
        assert player.advance_for_playback() == "000005"
        assert not player.playing, "playback pauses at the selected segment boundary"

        rendered, buttons = render_review_frame(
            cv2.imread(str(rgb_dir / "000004.png")),
            cv2.imread(str(masks / "000004.png"), cv2.IMREAD_GRAYSCALE),
            frame_id="000004",
            frame_position=2,
            frame_total=3,
            segment_label="分段 1 (2/2)",
            status="review",
            auto_status="review",
            reason="touches_image_border",
            playing=False,
            range_start=None,
        )
        assert rendered.shape[0] > 90, "render includes an information/button panel"
        assert rendered.shape[1] >= 640
        assert {button.action for button in buttons} == {"reject_scene_tail", "add_keyframe"}
        assert int(rendered.sum()) > 0

        assert KEY_ACTIONS[ord("a")] == "accepted"
        assert KEY_ACTIONS[ord("r")] == "review"
        assert KEY_ACTIONS[ord("x")] == "rejected"
        assert KEY_ACTIONS[ord("[")] == "range_start"
        assert KEY_ACTIONS[ord("]")] == "range_end"
        assert KEY_ACTIONS[ord("k")] == "add_keyframe"

    print("REVIEW_MASK_SEQUENCE_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
