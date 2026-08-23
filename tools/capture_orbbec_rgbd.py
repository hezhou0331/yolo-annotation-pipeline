#!/usr/bin/env python3
"""Capture aligned RGB-D frames from an Orbbec camera for FoundationPose.

The default 640x480@30 profile is intentional: Gemini 336L supports hardware
Depth-to-Color alignment at this resolution. Depth PNG values are millimetres,
which matches FoundationPose's YcbineoatReader convention (PNG / 1000 = metres).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from pyorbbecsdk import (
    Config,
    Context,
    OBAlignMode,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBSensorType,
    Pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集 Orbbec 对齐 RGB-D 数据")
    parser.add_argument("--output", type=Path, default=Path("dataset/capture_001"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=20, help="保存前丢弃的预热帧数")
    parser.add_argument("--auto", action="store_true", help="按时间间隔自动保存；否则按空格保存")
    parser.add_argument("--interval", type=float, default=0.5, help="自动保存间隔，单位秒")
    parser.add_argument("--max-frames", type=int, default=0, help="保存数量；0 表示不限制")
    parser.add_argument("--no-preview", action="store_true", help="不显示预览窗口（必须配合 --auto）")
    parser.add_argument("--min-depth", type=int, default=100, help="预览/统计的最小深度 mm")
    parser.add_argument("--max-depth", type=int, default=5000, help="预览/统计的最大深度 mm")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只枚举设备并验证默认RGB/深度配置，不启动采集、不写文件",
    )
    return parser.parse_args()


def find_profile(profile_list, width: int, height: int, fps: int, fmt):
    for i in range(len(profile_list)):
        p = profile_list[i]
        if (
            p.get_width() == width
            and p.get_height() == height
            and p.get_fps() == fps
            and p.get_format() == fmt
        ):
            return p
    return None


def profile_text(profile) -> str:
    return (
        f"{profile.get_width()}x{profile.get_height()}@{profile.get_fps()} "
        f"{profile.get_format()}"
    )


def rgb_frame_to_bgr(frame) -> np.ndarray:
    if frame.get_format() != OBFormat.RGB:
        raise RuntimeError(f"预期 RGB 帧，实际为 {frame.get_format()}")
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    expected = frame.get_width() * frame.get_height() * 3
    if data.size != expected:
        raise RuntimeError(f"彩色帧大小异常：{data.size}，预期 {expected}")
    rgb = data.reshape(frame.get_height(), frame.get_width(), 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def depth_frame_to_mm(frame) -> tuple[np.ndarray, float]:
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16)
    expected = frame.get_width() * frame.get_height()
    if raw.size != expected:
        raise RuntimeError(f"深度帧大小异常：{raw.size}，预期 {expected}")
    raw = raw.reshape(frame.get_height(), frame.get_width())
    scale = float(frame.get_depth_scale())
    depth_mm = np.rint(raw.astype(np.float32) * scale)
    depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return depth_mm, scale


def depth_preview(depth_mm: np.ndarray, min_depth: int, max_depth: int) -> np.ndarray:
    valid = (depth_mm >= min_depth) & (depth_mm <= max_depth)
    normalized = np.zeros(depth_mm.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth_mm.astype(np.float32), min_depth, max_depth)
        normalized[valid] = ((clipped[valid] - min_depth) * 255.0 / (max_depth - min_depth)).astype(np.uint8)
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def next_index(rgb_dir: Path) -> int:
    indices = []
    for path in rgb_dir.glob("*.png"):
        try:
            indices.append(int(path.stem))
        except ValueError:
            pass
    return max(indices, default=-1) + 1


def device_metadata(device_info) -> dict:
    def safe(method: str, default=None):
        try:
            return getattr(device_info, method)()
        except Exception:
            return default

    return {
        "name": safe("get_name"),
        "serial_number": safe("get_serial_number"),
        "firmware_version": safe("get_firmware_version"),
        "hardware_version": safe("get_hardware_version"),
        "connection_type": safe("get_connection_type"),
        "vid": safe("get_vid"),
        "pid": safe("get_pid"),
    }


def main() -> int:
    args = parse_args()
    if args.no_preview and not args.auto:
        print("错误：--no-preview 模式无法按空格保存，请同时加 --auto。", file=sys.stderr)
        return 2
    if args.interval < 0:
        print("错误：--interval 不能小于 0。", file=sys.stderr)
        return 2

    context = Context()
    devices = context.query_devices()
    if devices.get_count() == 0:
        print("未发现 Orbbec 相机。请检查 USB 连接与 udev 权限。", file=sys.stderr)
        return 1

    pipeline = Pipeline()
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = find_profile(color_profiles, args.width, args.height, args.fps, OBFormat.RGB)
    if color_profile is None:
        print(f"找不到 RGB {args.width}x{args.height}@{args.fps} 配置。", file=sys.stderr)
        print("建议 Gemini 336L 使用 640x480@30。", file=sys.stderr)
        return 1

    aligned_depth_profiles = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
    depth_profile = find_profile(
        aligned_depth_profiles, args.width, args.height, args.fps, OBFormat.Y16
    )
    if depth_profile is None:
        print("该彩色配置没有同分辨率/帧率的硬件 D2C 深度配置。", file=sys.stderr)
        print("请改用 --width 640 --height 480 --fps 30。", file=sys.stderr)
        return 1

    if args.check_only:
        info = device_metadata(devices.get_device_by_index(0).get_device_info())
        print(f"设备：{info['name']}")
        print(f"序列号：{info['serial_number']}")
        print(f"固件：{info['firmware_version']}")
        print(f"连接：{info['connection_type']}")
        print(f"彩色配置：{profile_text(color_profile)}")
        print(f"硬件D2C深度配置：{profile_text(depth_profile)}")
        print("检查完成：未启动采集，未写入任何数据。")
        return 0

    config = Config()
    config.enable_stream(depth_profile)
    config.enable_stream(color_profile)
    config.set_align_mode(OBAlignMode.HW_MODE)
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    try:
        pipeline.enable_frame_sync()
    except Exception as exc:
        print(f"提示：硬件帧同步未启用：{exc}")

    output = args.output.expanduser().resolve()
    rgb_dir = output / "rgb"
    depth_dir = output / "depth"
    masks_dir = output / "masks"
    for directory in (rgb_dir, depth_dir, masks_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"设备：{device_metadata(devices.get_device_by_index(0).get_device_info())['name']}")
    print(f"彩色配置：{profile_text(color_profile)}")
    print(f"原始深度配置：{profile_text(depth_profile)}")
    print("对齐：硬件 Depth -> Color")
    print(f"输出：{output}")

    pipeline.start(config)
    saved = 0
    index = next_index(rgb_dir)
    records: list[dict] = []
    last_save_time = -1e9
    started_monotonic = time.monotonic()

    try:
        valid_frames = 0
        while valid_frames < args.warmup:
            frames = pipeline.wait_for_frames(1000)
            if frames and frames.get_color_frame() and frames.get_depth_frame():
                valid_frames += 1
        print(f"预热完成：{valid_frames} 帧")

        camera_param = pipeline.get_camera_param()
        intr = camera_param.rgb_intrinsic
        K = np.array(
            [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        np.savetxt(output / "cam_K.txt", K, fmt="%.10f")
        np.savetxt(output / "K.txt", K, fmt="%.10f")

        metadata = {
            "format_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "device": device_metadata(pipeline.get_device().get_device_info()),
            "alignment": "hardware_depth_to_color",
            "color_profile": {
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "format": "RGB888",
            },
            "depth_profile_source": {
                "width": depth_profile.get_width(),
                "height": depth_profile.get_height(),
                "fps": depth_profile.get_fps(),
                "format": "Y16",
            },
            "saved_depth_unit": "millimetre",
            "depth_to_metre": 0.001,
            "camera_matrix": K.tolist(),
            "rgb_distortion": {
                "k1": float(camera_param.rgb_distortion.k1),
                "k2": float(camera_param.rgb_distortion.k2),
                "k3": float(camera_param.rgb_distortion.k3),
                "p1": float(camera_param.rgb_distortion.p1),
                "p2": float(camera_param.rgb_distortion.p2),
            },
            "frames": records,
        }

        if not args.no_preview:
            cv2.namedWindow("Orbbec RGB-D Capture", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Orbbec RGB-D Capture", 1280, 480)
            print("操作：空格保存，Q/ESC 退出。自动模式会按设置的间隔保存。")

        while True:
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_bgr = rgb_frame_to_bgr(color_frame)
            depth_mm, sensor_scale = depth_frame_to_mm(depth_frame)
            if color_bgr.shape[:2] != depth_mm.shape:
                raise RuntimeError(
                    f"对齐失败：RGB={color_bgr.shape[:2]}，Depth={depth_mm.shape}"
                )

            now = time.monotonic()
            should_save = args.auto and (now - last_save_time >= args.interval)
            key = -1
            if not args.no_preview:
                preview_depth = depth_preview(depth_mm, args.min_depth, args.max_depth)
                preview = np.hstack([color_bgr, preview_depth])
                mode = "AUTO" if args.auto else "SPACE TO SAVE"
                cv2.putText(
                    preview,
                    f"{mode}  saved={saved}  next={index:06d}",
                    (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Orbbec RGB-D Capture", preview)
                key = cv2.waitKey(1) & 0xFF
                should_save = should_save or key == ord(" ")

            if should_save:
                stem = f"{index:06d}"
                rgb_path = rgb_dir / f"{stem}.png"
                depth_path = depth_dir / f"{stem}.png"
                if not cv2.imwrite(str(rgb_path), color_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise RuntimeError(f"RGB 保存失败：{rgb_path}")
                if not cv2.imwrite(str(depth_path), depth_mm, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise RuntimeError(f"深度保存失败：{depth_path}")

                valid = (depth_mm >= args.min_depth) & (depth_mm <= args.max_depth)
                records.append(
                    {
                        "id": stem,
                        "color_timestamp_ms": float(color_frame.get_timestamp()),
                        "depth_timestamp_ms": float(depth_frame.get_timestamp()),
                        "sensor_depth_scale": sensor_scale,
                        "valid_depth_ratio": float(valid.mean()),
                    }
                )
                metadata["frames"] = records
                metadata["saved_frames_this_session"] = saved + 1
                metadata["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
                (output / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    f"已保存 {stem} | RGB {color_bgr.shape[1]}x{color_bgr.shape[0]} | "
                    f"有效深度 {valid.mean():.1%}"
                )
                index += 1
                saved += 1
                last_save_time = now

                if args.max_frames > 0 and saved >= args.max_frames:
                    break

            if key in (ord("q"), ord("Q"), 27):
                break

    except KeyboardInterrupt:
        print("收到中断，正在安全退出。")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - started_monotonic
    print(f"采集结束：本次保存 {saved} 帧，用时 {elapsed:.1f} 秒。")
    print(f"数据目录：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
