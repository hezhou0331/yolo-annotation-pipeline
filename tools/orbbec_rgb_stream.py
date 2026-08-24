#!/usr/bin/env python3
"""Read Orbbec SDK RGB frames and stream JPEG packets over a local Unix socket."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import socket
import sys

import cv2

from capture_orbbec_rgbd import find_profile, profile_text, rgb_frame_to_bgr
from orbbec_stream_protocol import send_packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orbbec SDK RGB 本地图像流")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    def stop_cleanly(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_cleanly)
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        print("宽度、高度和帧率必须大于0。", file=sys.stderr, flush=True)
        return 2
    if args.warmup < 0:
        print("--warmup不能小于0。", file=sys.stderr, flush=True)
        return 2
    if not 50 <= args.jpeg_quality <= 100:
        print("--jpeg-quality必须在50到100之间。", file=sys.stderr, flush=True)
        return 2

    from pyorbbecsdk import Config, Context, OBFormat, OBSensorType, Pipeline

    context = Context()
    devices = context.query_devices()
    if devices.get_count() == 0:
        print("未发现 Orbbec 相机。请检查 USB 连接与 udev 权限。", file=sys.stderr, flush=True)
        return 1

    pipeline = Pipeline()
    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = find_profile(profiles, args.width, args.height, args.fps, OBFormat.RGB)
    if color_profile is None:
        print(
            f"找不到 Orbbec RGB {args.width}x{args.height}@{args.fps} 配置。",
            file=sys.stderr,
            flush=True,
        )
        return 1

    config = Config()
    config.enable_stream(color_profile)
    connection: socket.socket | None = None
    started = False
    try:
        pipeline.start(config)
        started = True
        valid_frames = 0
        while valid_frames < args.warmup:
            frames = pipeline.wait_for_frames(1000)
            if frames and frames.get_color_frame():
                valid_frames += 1

        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.connect(str(args.socket))
        print(
            f"Orbbec RGB流已连接：{profile_text(color_profile)}，预热 {valid_frames} 帧，"
            f"JPEG质量 {args.jpeg_quality}",
            flush=True,
        )

        first_frame = True
        while True:
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_bgr = rgb_frame_to_bgr(color_frame)
            if first_frame:
                means = color_bgr.mean(axis=(0, 1))
                print(
                    f"Orbbec首帧：{color_bgr.shape[1]}x{color_bgr.shape[0]} "
                    f"BGR均值=({means[0]:.1f}, {means[1]:.1f}, {means[2]:.1f})",
                    flush=True,
                )
                first_frame = False
            ok, encoded = cv2.imencode(
                ".jpg",
                color_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            if not ok:
                raise RuntimeError("Orbbec RGB帧JPEG编码失败")
            send_packet(connection, encoded.tobytes())
    except (BrokenPipeError, ConnectionResetError, EOFError):
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if connection is not None:
            connection.close()
        if started:
            try:
                pipeline.stop()
            except Exception as exc:
                print(f"停止 Orbbec Pipeline 时出现提示：{exc}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
