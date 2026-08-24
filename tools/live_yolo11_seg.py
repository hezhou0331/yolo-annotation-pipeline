#!/usr/bin/env python3
"""Generic live YOLO11-seg viewer for any classes contained in a trained model."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import cv2


def parse_source(value: str) -> int | str:
    value = str(value).strip()
    return int(value) if value.isdigit() else value


def class_name_summary(names: Any) -> str:
    if isinstance(names, dict):
        values = [str(names[key]) for key in sorted(names, key=lambda item: int(item) if str(item).isdigit() else str(item))]
    elif isinstance(names, (list, tuple)):
        values = [str(value) for value in names]
    else:
        values = [str(names)] if names else []
    return ", ".join(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用真实训练best.pt启动通用YOLO11-seg实时识别")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--window", default="ATEC YOLO11-seg Live")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"训练权重不存在: {model_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf必须在0到1之间")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    source = parse_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise SystemExit(f"无法打开摄像头/视频源: {source}")

    classes = class_name_summary(getattr(model, "names", None))
    print(f"实时识别已启动：model={model_path} source={source} classes={classes or '未知'}")
    print("按 Q 或 Esc 退出；也可在 App 点击停止实时识别。")
    previous = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"视频源读取失败: {source}")
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            annotated = results[0].plot() if results else frame.copy()
            now = time.perf_counter()
            fps = 1.0 / max(now - previous, 1e-6)
            previous = now
            cv2.putText(
                annotated, f"FPS {fps:.1f} | {model_path.name}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 255, 30), 2, cv2.LINE_AA,
            )
            if classes:
                cv2.putText(
                    annotated, f"Classes: {classes}", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 255, 255), 1, cv2.LINE_AA,
                )
            cv2.imshow(args.window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
