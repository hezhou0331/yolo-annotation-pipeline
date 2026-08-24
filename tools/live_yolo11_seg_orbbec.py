#!/usr/bin/env python3
"""Generic YOLO11-seg live viewer backed by Orbbec SDK RGB frames."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any

import cv2
import numpy as np

try:
    from .live_yolo11_seg import class_name_summary
    from .orbbec_stream_protocol import recv_packet
except ImportError:  # Direct script execution.
    from live_yolo11_seg import class_name_summary
    from orbbec_stream_protocol import recv_packet

DEFAULT_ORBBEC_PYTHON = Path("/home/hezhou/miniforge3/envs/orbbec/bin/python")


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
    parser = argparse.ArgumentParser(description="使用 Orbbec SDK RGB 启动通用YOLO11-seg实时识别")
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
            socket_path = Path(tmp) / "rgb.sock"
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
                print("启动 Orbbec SDK RGB流：" + " ".join(command), flush=True)
                child = subprocess.Popen(command)
                connection = _wait_for_connection(server, child)
            finally:
                server.close()

            print(
                f"Orbbec实时识别已启动：model={model_path} conf={args.conf:.2f} "
                f"classes={classes or '未知'}",
                flush=True,
            )
            print("按 Q 或 Esc 退出；也可在 App 点击停止实时识别。", flush=True)
            previous = time.perf_counter()
            while not stop_requested:
                try:
                    payload = recv_packet(connection)
                except socket.timeout:
                    if child.poll() is not None:
                        raise RuntimeError(f"Orbbec RGB流意外退出，exit={child.returncode}")
                    continue
                frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("无法解码 Orbbec RGB帧")
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )
                annotated = results[0].plot() if results else frame.copy()
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
