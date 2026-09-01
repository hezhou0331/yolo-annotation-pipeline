#!/usr/bin/env python3
"""CPU-only checks for the Orbbec SDK live YOLO bridge; no camera is started."""
from __future__ import annotations

from types import SimpleNamespace
import socket
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.live_yolo11_seg_orbbec import (  # noqa: E402
    build_stream_command,
    decode_rgbd_images,
    draw_instance_depths,
    measure_instance_depths,
)
from tools.orbbec_stream_protocol import (  # noqa: E402
    RgbdPacket, recv_packet, recv_rgbd_packet, send_packet, send_rgbd_packet,
)


def main() -> int:
    left, right = socket.socketpair()
    try:
        payload = b"\xff\xd8ATEC-JPEG\xff\xd9"
        send_packet(left, payload)
        assert recv_packet(right) == payload
    finally:
        left.close()
        right.close()


    color = np.zeros((8, 10, 3), dtype=np.uint8)
    depth = np.zeros((8, 10), dtype=np.uint16)
    mask = np.zeros((8, 10), dtype=np.float32)
    mask[1:7, 1:9] = 1.0
    depth[mask > 0.5] = 1000
    depth[3, 4] = 4000
    rgb_ok, rgb_jpeg = cv2.imencode(".jpg", color)
    depth_ok, depth_png = cv2.imencode(".png", depth)
    assert rgb_ok and depth_ok
    source_packet = RgbdPacket(
        rgb_jpeg=rgb_jpeg.tobytes(),
        depth_png=depth_png.tobytes(),
        width=10,
        height=8,
        intrinsics=(100.0, 100.0, 5.0, 4.0),
    )

    left, right = socket.socketpair()
    try:
        send_rgbd_packet(left, source_packet)
        received_packet = recv_rgbd_packet(right)
    finally:
        left.close()
        right.close()
    decoded_color, decoded_depth = decode_rgbd_images(received_packet)
    assert decoded_color.shape == color.shape
    assert decoded_depth.dtype == np.uint16
    assert np.array_equal(decoded_depth, depth)

    result = SimpleNamespace(
        masks=SimpleNamespace(data=mask[None, ...]),
        boxes=SimpleNamespace(
            cls=np.asarray([8.0]),
            conf=np.asarray([0.91]),
            xyxy=np.asarray([[1.0, 1.0, 8.0, 6.0]]),
        ),
        names={8: "sand_bottle"},
    )
    measurements = measure_instance_depths(
        result,
        decoded_depth,
        received_packet.intrinsics,
        min_valid_pixels=20,
    )
    assert len(measurements) == 1
    item = measurements[0]
    assert item.label == "sand_bottle"
    assert abs(item.confidence - 0.91) < 1e-6
    assert item.depth_m == 1.0, "median must reject the single 4m outlier"
    assert item.xyz_m is not None and item.xyz_m[2] == 1.0
    annotated = draw_instance_depths(decoded_color.copy(), measurements)
    assert np.any(annotated != decoded_color)

    command = build_stream_command(
        orbbec_python=Path("/opt/orbbec/bin/python"),
        streamer=ROOT / "tools/orbbec_rgb_stream.py",
        socket_path=Path("/tmp/atec-orbbec.sock"),
        width=640,
        height=480,
        fps=30,
        warmup=30,
        jpeg_quality=90,
    )
    assert command[0] == "/opt/orbbec/bin/python"
    assert command[1].endswith("tools/orbbec_rgb_stream.py")
    assert command[command.index("--socket") + 1] == "/tmp/atec-orbbec.sock"
    assert command[command.index("--width") + 1] == "640"
    assert command[command.index("--height") + 1] == "480"
    assert command[command.index("--fps") + 1] == "30"
    assert command[command.index("--warmup") + 1] == "30"
    assert command[command.index("--jpeg-quality") + 1] == "90"
    print("LIVE_YOLO11_SEG_ORBBEC_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
