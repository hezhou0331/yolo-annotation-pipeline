#!/usr/bin/env python3
"""Interactive polygon/brush editor for per-instance key-frame masks.

The editor can resume an existing mask, add multiple disconnected polygons and
erase holes or boundary mistakes. Each instance is saved as its own binary PNG.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from atec_pipeline.mask_editor_state import editor_status, is_save_key


def parse_args():
    parser = argparse.ArgumentParser(description="用多边形和画笔为关键帧绘制目标掩膜")
    parser.add_argument("--image", type=Path, required=True, help="关键帧RGB图片")
    parser.add_argument("--output", type=Path, required=True, help="输出二值mask PNG")
    parser.add_argument("--mask-input", type=Path, default=None, help="可选：以已有mask为起点；默认自动读取output")
    parser.add_argument("--brush-size", type=int, default=18, help="初始画笔直径，像素")
    parser.add_argument("--no-resume", action="store_true", help="即使output已存在也从空mask开始")
    return parser.parse_args()


def load_initial_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros(shape, dtype=np.uint8)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise SystemExit(f"无法读取已有mask：{path}")
    if raw.ndim == 3:
        raw = raw.max(axis=2)
    if raw.shape != shape:
        raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(raw > 0, 255, 0).astype(np.uint8)


def apply_polygon(mask: np.ndarray, points: list[tuple[int, int]], erase: bool) -> None:
    if len(points) < 3:
        raise ValueError("多边形至少需要3个点")
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 0 if erase else 255)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), binary):
        raise RuntimeError(f"保存失败：{path}")


def editor_status_lines(
    mode: str, erase: bool, points: int, brush_size: int, pixels: int, dirty: bool
) -> tuple[str, str]:
    """ASCII-only status text so OpenCV can render it on every machine."""
    first = editor_status(mode, erase, points, brush_size, pixels, dirty)
    second = "ENTER=apply | P=polygon B=brush E=erase | S/Ctrl+S=save Q=exit"
    return first, second


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"无法读取图片：{args.image}")
    if args.brush_size < 1:
        raise SystemExit("--brush-size必须大于0")

    resume_path = None if args.no_resume else (args.mask_input or args.output)
    mask = load_initial_mask(resume_path, image.shape[:2])
    points: list[tuple[int, int]] = []
    mode = "polygon"
    erase = False
    brush_size = args.brush_size
    show_overlay = True
    dirty = False
    drawing = False
    previous_xy: tuple[int, int] | None = None
    window = "ATEC key-mask editor"

    def paint_line(start: tuple[int, int], end: tuple[int, int], erase_now: bool) -> None:
        nonlocal dirty
        cv2.line(mask, start, end, 0 if erase_now else 255, brush_size, cv2.LINE_AA)
        dirty = True

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, previous_xy
        if mode == "polygon":
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and points:
                points.pop()
            return

        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            drawing = True
            previous_xy = (x, y)
            paint_line(previous_xy, previous_xy, erase if event == cv2.EVENT_LBUTTONDOWN else True)
        elif event == cv2.EVENT_MOUSEMOVE and drawing and previous_xy is not None:
            current = (x, y)
            right_down = bool(flags & cv2.EVENT_FLAG_RBUTTON)
            paint_line(previous_xy, current, True if right_down else erase)
            previous_xy = current
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            drawing = False
            previous_xy = None

    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window, min(1400, image.shape[1]), min(900, image.shape[0]))
    cv2.setMouseCallback(window, on_mouse)

    print("P：多边形；B：画笔；E：切换添加/擦除；Enter：应用当前多边形。")
    print("左键：加点/绘制；右键：撤点/临时擦除；U：撤销点；R：清当前点；C：清空全部mask。")
    print("[ / ]：缩小/放大画笔；M：显示/隐藏覆盖；S或Ctrl+S：保存；Q/ESC：退出。")
    while True:
        canvas = image.copy()
        if show_overlay and mask.any():
            color = np.zeros_like(canvas)
            color[:, :, 1] = mask
            canvas = cv2.addWeighted(canvas, 0.72, color, 0.28, 0)
        if points:
            pts = np.asarray(points, dtype=np.int32)
            point_color = (0, 80, 255) if erase else (0, 255, 255)
            for point in points:
                cv2.circle(canvas, point, 4, point_color, -1, cv2.LINE_AA)
            if len(points) >= 2:
                cv2.polylines(canvas, [pts], len(points) >= 3, point_color, 2, cv2.LINE_AA)
            if len(points) >= 3:
                preview = canvas.copy()
                cv2.fillPoly(preview, [pts], (0, 0, 255) if erase else (0, 255, 0))
                canvas = cv2.addWeighted(canvas, 0.78, preview, 0.22, 0)

        status, shortcuts = editor_status_lines(
            mode, erase, len(points), brush_size, int((mask > 0).sum()), dirty
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 66), (16, 24, 40), -1)
        cv2.putText(canvas, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, shortcuts, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 235, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            unsaved = dirty or bool(points)
            if unsaved:
                print("有未保存修改；未写入磁盘。按S或Ctrl+S保存后再退出。")
            cv2.destroyAllWindows()
            return 1 if unsaved else 0
        if key in (ord("p"), ord("P")):
            mode = "polygon"
        elif key in (ord("b"), ord("B")):
            mode = "brush"
            points.clear()
        elif key in (ord("e"), ord("E")):
            erase = not erase
        elif key in (ord("u"), ord("U"), 8) and points:
            points.pop()
        elif key in (ord("r"), ord("R")):
            points.clear()
        elif key in (ord("c"), ord("C")):
            mask[:] = 0
            points.clear()
            dirty = True
        elif key in (ord("m"), ord("M")):
            show_overlay = not show_overlay
        elif key in (ord("["), ord("-")):
            brush_size = max(1, brush_size - 2)
        elif key in (ord("]"), ord("+"), ord("=")):
            brush_size = min(300, brush_size + 2)
        elif key in (13, 10):
            if mode != "polygon" or len(points) < 3:
                print("当前不是多边形模式，或点数不足3。")
                continue
            apply_polygon(mask, points, erase)
            points.clear()
            dirty = True
        elif is_save_key(key):
            if points:
                print("还有未应用的多边形点；按Enter应用，或R清除后再保存。")
                continue
            if not mask.any():
                print("mask为空，不保存。")
                continue
            save_mask(args.output, mask)
            dirty = False
            print(f"已保存：{args.output}，目标像素：{int((mask > 0).sum())}")


if __name__ == "__main__":
    raise SystemExit(main())
