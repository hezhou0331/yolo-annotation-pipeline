#!/usr/bin/env python3
"""Interactive, frame-accurate review of propagated RGB mask sequences."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from atec_pipeline.review_state import ReviewState, ReviewSegment

WINDOW_NAME = "ATEC Mask Review"
RERUN_EXIT_CODE = 20
KEY_ACTIONS = {
    ord("a"): "accepted", ord("A"): "accepted",
    ord("r"): "review", ord("R"): "review",
    ord("x"): "rejected", ord("X"): "rejected",
    ord("["): "range_start", ord("]"): "range_end",
    ord("k"): "add_keyframe", ord("K"): "add_keyframe",
    ord("q"): "quit", ord("Q"): "quit",
    27: "quit", 32: "toggle_play",
}
LEFT_KEYS = {81, 2424832, 65361}
RIGHT_KEYS = {83, 2555904, 65363}
UP_KEYS = {82, 2490368, 65362}
DOWN_KEYS = {84, 2621440, 65364}


@dataclass(frozen=True)
class ClickButton:
    action: str
    label: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class PlayerModel:
    def __init__(self, state: ReviewState, selected_segment_ids: Iterable[str] = ()) -> None:
        self.state = state
        selected = {str(value) for value in selected_segment_ids}
        self.selected_segments = tuple(
            segment for segment in state.segments if not selected or segment.segment_id in selected
        )
        if not self.selected_segments:
            raise ValueError("所选分段不存在")
        self.playable_frame_ids = tuple(
            frame_id for segment in self.selected_segments for frame_id in segment.frame_ids
        )
        self.index = 0
        self.playing = False
        self.changed = False
        self._boundary_waiting = False

    @property
    def current_frame_id(self) -> str:
        return self.playable_frame_ids[self.index]

    @property
    def current_segment(self) -> ReviewSegment:
        return self.state.segment_for_frame(self.current_frame_id)

    def move(self, delta: int) -> str:
        self.index = max(0, min(len(self.playable_frame_ids) - 1, self.index + delta))
        self._boundary_waiting = False
        return self.current_frame_id

    def move_segment(self, delta: int) -> str:
        current = self.current_segment
        current_index = self.selected_segments.index(current)
        target = max(0, min(len(self.selected_segments) - 1, current_index + delta))
        frame_id = self.selected_segments[target].frame_ids[0]
        self.index = self.playable_frame_ids.index(frame_id)
        self.playing = False
        self._boundary_waiting = False
        return frame_id

    def advance_for_playback(self) -> str:
        if self.index >= len(self.playable_frame_ids) - 1:
            self.playing = False
            return self.current_frame_id
        current_segment = self.current_segment.segment_id
        next_frame = self.playable_frame_ids[self.index + 1]
        next_segment = self.state.segment_for_frame(next_frame).segment_id
        if current_segment != next_segment and not self._boundary_waiting:
            self.playing = False
            self._boundary_waiting = True
            return self.current_frame_id
        self.index += 1
        self._boundary_waiting = False
        return self.current_frame_id

    def set_current_status(self, status: str) -> None:
        self.state.set_status(self.current_frame_id, status)
        self.state.save()
        self.changed = True

    def finish_problem_range(self) -> tuple[str, ...]:
        changed = self.state.finish_problem_range(self.current_frame_id)
        self.state.save()
        self.changed = bool(changed) or self.changed
        return changed

    def reject_scene_tail(self) -> tuple[str, ...]:
        changed = self.state.reject_from(self.current_frame_id, scope="scene")
        self.state.save()
        self.changed = bool(changed) or self.changed
        return changed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="连续检查 SAM2 Mask 传播效果")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--segments-report", type=Path)
    parser.add_argument("--review-overrides", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--key-mask-dir", type=Path, required=True)
    parser.add_argument("--segments", default="", help="逗号分隔的segment_id；留空检查全部")
    parser.add_argument("--action-file", type=Path)
    parser.add_argument("--editor-python", default=sys.executable)
    parser.add_argument("--editor-script", type=Path, default=WORKSPACE / "tools/draw_first_mask.py")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)


def _put_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale: float = 0.58) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_review_frame(
    bgr: np.ndarray,
    mask: np.ndarray | None,
    *,
    frame_id: str,
    frame_position: int,
    frame_total: int,
    segment_label: str,
    status: str,
    auto_status: str,
    reason: str,
    playing: bool,
    range_start: str | None,
) -> tuple[np.ndarray, tuple[ClickButton, ...]]:
    if bgr is None:
        raise ValueError("RGB frame is missing")
    canvas = bgr.copy()
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.max(axis=2)
        if mask.shape != canvas.shape[:2]:
            mask = cv2.resize(mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
        binary = mask > 0
        if binary.any():
            color = {"accepted": (0, 190, 0), "review": (0, 190, 255), "rejected": (0, 0, 255)}.get(status, (255, 0, 255))
            overlay = canvas.copy()
            overlay[binary] = color
            canvas = cv2.addWeighted(canvas, 0.68, overlay, 0.32, 0)
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)

    target_width = max(760, canvas.shape[1])
    if canvas.shape[1] < target_width:
        canvas = cv2.copyMakeBorder(canvas, 0, 0, 0, target_width - canvas.shape[1], cv2.BORDER_CONSTANT, value=(18, 18, 18))
    panel_height = 154
    output = cv2.copyMakeBorder(canvas, 0, panel_height, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    status_color = {"accepted": (70, 220, 70), "review": (0, 210, 255), "rejected": (60, 60, 255)}.get(status, (230, 230, 230))
    _put_text(output, f"Frame: {frame_id}  ({frame_position}/{frame_total})    {segment_label}", (14, canvas.shape[0] + 25), (235, 235, 235))
    _put_text(output, f"Status: {status}    auto: {auto_status}    {'PLAY' if playing else 'PAUSE'}", (14, canvas.shape[0] + 50), status_color)
    _put_text(output, f"Reason: {reason[:110]}", (14, canvas.shape[0] + 75), (210, 210, 210), 0.50)
    range_text = f"Problem range starts at {range_start}" if range_start else "[ start range, ] reject range; A/R/X set frame; K add keyframe; arrows step; Space play"
    _put_text(output, range_text, (14, canvas.shape[0] + 98), (180, 180, 180), 0.45)

    y1 = canvas.shape[0] + 112
    y2 = canvas.shape[0] + 145
    first = ClickButton("reject_scene_tail", "Reject all remaining frames", 14, y1, 330, y2)
    second = ClickButton("add_keyframe", "Add keyframe and rerun", 348, y1, 650, y2)
    for button in (first, second):
        cv2.rectangle(output, (button.x1, button.y1), (button.x2, button.y2), (75, 75, 75), -1)
        cv2.rectangle(output, (button.x1, button.y1), (button.x2, button.y2), (180, 180, 180), 1)
        _put_text(output, button.label, (button.x1 + 10, button.y1 + 22), (245, 245, 245), 0.48)
    return output, (first, second)


def _write_action(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _add_keyframe(args: argparse.Namespace, state: ReviewState, frame_id: str) -> bool:
    image = args.scene.expanduser().resolve() / "rgb" / f"{frame_id}.png"
    propagated = state.mask_path(frame_id)
    output = state.keyframe_path(args.key_mask_dir, frame_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.editor_python), str(args.editor_script.expanduser().resolve()),
        "--image", str(image), "--output", str(output),
    ]
    if propagated.is_file():
        command.extend(["--mask-input", str(propagated)])
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0 or not output.is_file():
        return False
    _write_action(args.action_file, {"action": "rerun", "frame_id": frame_id, "key_mask": str(output)})
    return True


def run_player(args: argparse.Namespace) -> int:
    selected = tuple(part.strip() for part in args.segments.split(",") if part.strip())
    state = ReviewState.load(
        args.quality_report, args.review_overrides, args.mask_dir, args.segments_report,
        scene_name=args.scene.name, instance_id=args.instance_id,
    )
    player = PlayerModel(state, selected)
    if args.headless:
        return 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    click_action: list[str] = []
    buttons: tuple[ClickButton, ...] = ()

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        for button in buttons:
            if button.contains(x, y):
                click_action.append(button.action)
                break

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    delay = max(1, int(1000.0 / max(0.5, args.fps)))
    try:
        while True:
            frame_id = player.current_frame_id
            rgb = cv2.imread(str(args.scene / "rgb" / f"{frame_id}.png"), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(state.mask_path(frame_id)), cv2.IMREAD_GRAYSCALE)
            segment = player.current_segment
            segment_number = player.selected_segments.index(segment) + 1
            rendered, buttons = render_review_frame(
                rgb, mask, frame_id=frame_id, frame_position=player.index + 1,
                frame_total=len(player.playable_frame_ids),
                segment_label=f"Segment {segment.segment_id} ({segment_number}/{len(player.selected_segments)})",
                status=state.effective_status(frame_id), auto_status=state.auto_status(frame_id),
                reason=state.reason_text(frame_id), playing=player.playing,
                range_start=state.problem_range_start,
            )
            cv2.imshow(WINDOW_NAME, rendered)
            key = cv2.waitKeyEx(delay if player.playing else 30)
            action = click_action.pop(0) if click_action else KEY_ACTIONS.get(key)
            if action == "quit":
                break
            if key in LEFT_KEYS:
                player.playing = False; player.move(-1)
            elif key in RIGHT_KEYS:
                player.playing = False; player.move(1)
            elif key in UP_KEYS:
                player.move_segment(-1)
            elif key in DOWN_KEYS:
                player.move_segment(1)
            elif action == "toggle_play":
                player.playing = not player.playing
            elif action in {"accepted", "review", "rejected"}:
                try:
                    player.set_current_status(action)
                except ValueError as exc:
                    print(exc, file=sys.stderr)
            elif action == "range_start":
                state.begin_problem_range(frame_id)
            elif action == "range_end":
                try:
                    player.finish_problem_range()
                except RuntimeError as exc:
                    print(exc, file=sys.stderr)
            elif action == "reject_scene_tail":
                player.reject_scene_tail()
            elif action == "add_keyframe":
                player.playing = False
                if _add_keyframe(args, state, frame_id):
                    return RERUN_EXIT_CODE
            elif player.playing:
                player.advance_for_playback()
    finally:
        state.save()
        if player.changed and (args.action_file is None or not args.action_file.expanduser().is_file()):
            _write_action(args.action_file, {"action": "review_changed"})
        cv2.destroyWindow(WINDOW_NAME)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_player(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
