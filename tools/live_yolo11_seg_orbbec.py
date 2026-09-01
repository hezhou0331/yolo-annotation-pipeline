#!/usr/bin/env python3
"""YOLO11-seg live viewer with aligned Orbbec RGB-D instance distances."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from atec_pipeline.runtime import interpreter_path


try:
    from .live_yolo11_seg import class_name_summary
    from .orbbec_stream_protocol import RgbdPacket, recv_rgbd_packet
except ImportError:  # Direct script execution.
    from live_yolo11_seg import class_name_summary
    from orbbec_stream_protocol import RgbdPacket, recv_rgbd_packet

DEFAULT_ORBBEC_PYTHON = interpreter_path("orbbec")


@dataclass(frozen=True)
class InstanceDepth:
    label: str
    confidence: float
    depth_m: float | None
    xyz_m: tuple[float, float, float] | None
    box_xyxy: tuple[int, int, int, int]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def decode_rgbd_images(packet: RgbdPacket) -> tuple[np.ndarray, np.ndarray]:
    color = cv2.imdecode(np.frombuffer(packet.rgb_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.frombuffer(packet.depth_png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if color is None:
        raise ValueError("无法解码 Orbbec RGB帧")
    if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("无法解码 Orbbec uint16毫米深度帧")
    if color.shape[:2] != depth.shape or color.shape[1] != packet.width or color.shape[0] != packet.height:
        raise ValueError(
            f"RGB-D帧尺寸不一致: RGB={color.shape[:2]}, Depth={depth.shape}, "
            f"packet=({packet.height}, {packet.width})"
        )
    return color, depth


def measure_instance_depths(
    result: Any,
    depth_mm: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    *,
    min_depth_mm: int = 100,
    max_depth_mm: int = 5000,
    min_valid_pixels: int = 20,
) -> list[InstanceDepth]:
    """Measure robust per-instance Z and XYZ from aligned depth inside each segmentation mask."""
    if depth_mm.ndim != 2:
        raise ValueError("深度图必须是二维数组")
    if min_depth_mm < 0 or max_depth_mm <= min_depth_mm or min_valid_pixels <= 0:
        raise ValueError("深度范围或最小有效像素数无效")
    masks = getattr(getattr(result, "masks", None), "data", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None:
        return []

    mask_array = _as_numpy(masks)
    if mask_array.ndim == 2:
        mask_array = mask_array[None, ...]
    classes = _as_numpy(boxes.cls).reshape(-1)
    confidences = _as_numpy(boxes.conf).reshape(-1)
    box_array = _as_numpy(boxes.xyxy).reshape(-1, 4)
    count = min(len(mask_array), len(classes), len(confidences), len(box_array))
    fx, fy, cx, cy = intrinsics
    if fx <= 0 or fy <= 0:
        raise ValueError("相机焦距内参必须大于0")

    measurements: list[InstanceDepth] = []
    valid_depth = (depth_mm >= min_depth_mm) & (depth_mm <= max_depth_mm)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for index in range(count):
        resized = cv2.resize(
            mask_array[index].astype(np.float32),
            (depth_mm.shape[1], depth_mm.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = resized > 0.5
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        eroded_valid = eroded & valid_depth
        selected = eroded_valid if int(eroded_valid.sum()) >= min_valid_pixels else mask & valid_depth
        depth_m: float | None = None
        xyz_m: tuple[float, float, float] | None = None
        if int(selected.sum()) >= min_valid_pixels:
            rows, columns = np.nonzero(selected)
            z_values = depth_mm[selected].astype(np.float64) * 0.001
            z_value = float(np.median(z_values))
            x_value = float(np.median((columns.astype(np.float64) - cx) * z_values / fx))
            y_value = float(np.median((rows.astype(np.float64) - cy) * z_values / fy))
            depth_m = z_value
            xyz_m = (x_value, y_value, z_value)

        class_id = int(classes[index])
        box = tuple(int(round(float(value))) for value in box_array[index])
        measurements.append(
            InstanceDepth(
                label=_class_name(getattr(result, "names", None), class_id),
                confidence=float(confidences[index]),
                depth_m=depth_m,
                xyz_m=xyz_m,
                box_xyxy=box,
            )
        )
    return measurements


def draw_instance_depths(
    image: np.ndarray,
    measurements: list[InstanceDepth],
    *,
    show_xyz: bool = False,
) -> np.ndarray:
    """Draw class, confidence and robust depth next to each YOLO instance."""
    for item in measurements:
        x1, y1, _x2, _y2 = item.box_xyxy
        if show_xyz and item.xyz_m is not None:
            x_m, y_m, z_m = item.xyz_m
            text = f"{item.label} {item.confidence:.2f} X={x_m:.2f} Y={y_m:.2f} Z={z_m:.2f}m"
        elif item.depth_m is not None:
            text = f"{item.label} {item.confidence:.2f} Z={item.depth_m:.2f}m"
        else:
            text = f"{item.label} {item.confidence:.2f} Z=--"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        left = max(0, min(x1, image.shape[1] - text_width - 4))
        bottom = max(text_height + baseline + 4, min(y1 - 4, image.shape[0] - 2))
        cv2.rectangle(
            image,
            (left, bottom - text_height - baseline - 4),
            (left + text_width + 4, bottom + 2),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            image,
            text,
            (left + 2, bottom - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def build_stream_command(
    *,
    orbbec_python: Path,
    streamer: Path,
    socket_path: Path,
    width: int,
    height: int,
    fps: int,
    warmup: int,
    jpeg_quality: int,
) -> list[str]:
    return [
        str(orbbec_python),
        str(streamer),
        "--socket", str(socket_path),
        "--width", str(width),
        "--height", str(height),
        "--fps", str(fps),
        "--warmup", str(warmup),
        "--jpeg-quality", str(jpeg_quality),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Orbbec 对齐RGB-D启动YOLO11-seg实例距离识别")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--window", default="ATEC YOLO11-seg Live - Orbbec")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--min-depth-mm", type=int, default=100)
    parser.add_argument("--max-depth-mm", type=int, default=5000)
    parser.add_argument("--min-depth-pixels", type=int, default=20)
    parser.add_argument("--show-xyz", action="store_true", help="显示相机坐标X/Y/Z，而不只显示距离Z")
    parser.add_argument("--orbbec-python", type=Path, default=DEFAULT_ORBBEC_PYTHON)
    parser.add_argument(
        "--streamer",
        type=Path,
        default=Path(__file__).with_name("orbbec_rgb_stream.py"),
    )
    return parser.parse_args()


def _wait_for_connection(server: socket.socket, child: subprocess.Popen[Any]) -> socket.socket:
    server.settimeout(0.25)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            connection, _ = server.accept()
            connection.settimeout(0.5)
            return connection
        except socket.timeout:
            return_code = child.poll()
            if return_code is not None:
                raise RuntimeError(f"Orbbec RGB流进程提前退出，exit={return_code}")
    raise TimeoutError("等待 Orbbec RGB流连接超时；请检查相机是否被其他程序占用")


def _terminate_child(child: subprocess.Popen[Any] | None) -> None:
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=2.0)


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    orbbec_python = args.orbbec_python.expanduser().resolve()
    streamer = args.streamer.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"训练权重不存在: {model_path}")
    if not orbbec_python.is_file():
        raise SystemExit(f"Orbbec Python环境不存在: {orbbec_python}")
    if args.min_depth_mm < 0 or args.max_depth_mm <= args.min_depth_mm:
        raise SystemExit("深度范围无效：必须满足0 <= min-depth-mm < max-depth-mm")
    if args.min_depth_pixels <= 0:
        raise SystemExit("--min-depth-pixels必须大于0")
    if not streamer.is_file():
        raise SystemExit(f"Orbbec RGB流工具不存在: {streamer}")
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf必须在0到1之间")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    classes = class_name_summary(getattr(model, "names", None))
    child: subprocess.Popen[Any] | None = None
    connection: socket.socket | None = None
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    try:
        with tempfile.TemporaryDirectory(prefix="atec_orbbec_live_") as tmp:
            socket_path = Path(tmp) / "rgbd.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                server.bind(str(socket_path))
                server.listen(1)
                command = build_stream_command(
                    orbbec_python=orbbec_python,
                    streamer=streamer,
                    socket_path=socket_path,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    warmup=args.warmup,
                    jpeg_quality=args.jpeg_quality,
                )
                print("启动 Orbbec SDK 对齐RGB-D流：" + " ".join(command), flush=True)
                child = subprocess.Popen(command)
                connection = _wait_for_connection(server, child)
            finally:
                server.close()

            print(
                f"Orbbec RGB-D实时识别已启动：model={model_path} conf={args.conf:.2f} "
                f"classes={classes or '未知'}",
                flush=True,
            )
            print("每个实例显示Mask内有效深度中位数Z；使用--show-xyz可显示相机坐标。", flush=True)
            print("按 Q 或 Esc 退出；也可在 App 点击停止实时识别。", flush=True)
            previous = time.perf_counter()
            while not stop_requested:
                try:
                    packet = recv_rgbd_packet(connection)
                except socket.timeout:
                    if child.poll() is not None:
                        raise RuntimeError(f"Orbbec RGB-D流意外退出，exit={child.returncode}")
                    continue
                frame, depth_mm = decode_rgbd_images(packet)
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )
                if results:
                    result = results[0]
                    measurements = measure_instance_depths(
                        result,
                        depth_mm,
                        packet.intrinsics,
                        min_depth_mm=args.min_depth_mm,
                        max_depth_mm=args.max_depth_mm,
                        min_valid_pixels=args.min_depth_pixels,
                    )
                    annotated = result.plot(labels=not measurements)
                    if measurements:
                        draw_instance_depths(
                            annotated, measurements, show_xyz=args.show_xyz
                        )
                else:
                    annotated = frame.copy()
                now = time.perf_counter()
                display_fps = 1.0 / max(now - previous, 1e-6)
                previous = now
                cv2.putText(
                    annotated,
                    f"Orbbec | FPS {display_fps:.1f} | conf {args.conf:.2f} | {model_path.name}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (30, 255, 30),
                    2,
                    cv2.LINE_AA,
                )
                if classes:
                    cv2.putText(
                        annotated,
                        f"Classes: {classes}",
                        (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (30, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imshow(args.window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
        if connection is not None:
            connection.close()
        _terminate_child(child)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
