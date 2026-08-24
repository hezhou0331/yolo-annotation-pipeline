#!/usr/bin/env python3
"""CPU-only checks for the Orbbec SDK live YOLO bridge; no camera is started."""
from __future__ import annotations

import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.live_yolo11_seg_orbbec import build_stream_command  # noqa: E402
from tools.orbbec_stream_protocol import recv_packet, send_packet  # noqa: E402


def main() -> int:
    left, right = socket.socketpair()
    try:
        payload = b"\xff\xd8ATEC-JPEG\xff\xd9"
        send_packet(left, payload)
        assert recv_packet(right) == payload
    finally:
        left.close()
        right.close()

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
