#!/usr/bin/env python3
"""Small length-prefixed protocol used between Orbbec and YOLO environments."""
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

_HEADER = struct.Struct("!I")
_RGBD_HEADER = struct.Struct("!8sIIII4d")
_RGBD_MAGIC = b"ATECRGBD"
MAX_PACKET_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class RgbdPacket:
    rgb_jpeg: bytes
    depth_png: bytes
    width: int
    height: int
    intrinsics: tuple[float, float, float, float]


def send_packet(connection: socket.socket, payload: bytes) -> None:
    """Send one non-empty binary payload as a length-prefixed packet."""
    if not payload:
        raise ValueError("不能发送空帧")
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError(f"帧数据过大: {len(payload)} bytes")
    connection.sendall(_HEADER.pack(len(payload)))
    connection.sendall(payload)


def recv_exact(connection: socket.socket, size: int) -> bytes:
    """Receive exactly *size* bytes or raise EOFError when the peer closes."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("Orbbec 图像流已关闭")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_packet(connection: socket.socket) -> bytes:
    """Receive and validate one frame packet."""
    (size,) = _HEADER.unpack(recv_exact(connection, _HEADER.size))
    if size <= 0 or size > MAX_PACKET_BYTES:
        raise ValueError(f"无效的 Orbbec 帧包大小: {size}")
    return recv_exact(connection, size)


def pack_rgbd_packet(
    rgb_jpeg: bytes,
    depth_png: bytes,
    *,
    width: int,
    height: int,
    intrinsics: tuple[float, float, float, float],
) -> bytes:
    """Pack one aligned RGB-D frame and RGB intrinsics into one atomic payload."""
    if not rgb_jpeg or not depth_png:
        raise ValueError("RGB-D帧不能包含空图像")
    if width <= 0 or height <= 0:
        raise ValueError("RGB-D帧尺寸必须大于0")
    if len(intrinsics) != 4 or not all(float(value) > 0 for value in intrinsics[:2]):
        raise ValueError("相机内参必须为有效的(fx, fy, cx, cy)")
    header = _RGBD_HEADER.pack(
        _RGBD_MAGIC,
        width,
        height,
        len(rgb_jpeg),
        len(depth_png),
        *[float(value) for value in intrinsics],
    )
    return header + rgb_jpeg + depth_png


def unpack_rgbd_packet(payload: bytes) -> RgbdPacket:
    """Validate and unpack one aligned RGB-D payload."""
    if len(payload) < _RGBD_HEADER.size:
        raise ValueError("RGB-D帧包头不完整")
    magic, width, height, rgb_size, depth_size, fx, fy, cx, cy = _RGBD_HEADER.unpack_from(payload)
    if magic != _RGBD_MAGIC:
        raise ValueError("RGB-D帧包标识无效")
    expected = _RGBD_HEADER.size + rgb_size + depth_size
    if rgb_size <= 0 or depth_size <= 0 or len(payload) != expected:
        raise ValueError(
            f"RGB-D帧包长度无效: actual={len(payload)}, expected={expected}"
        )
    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("RGB-D帧尺寸或相机内参无效")
    rgb_start = _RGBD_HEADER.size
    depth_start = rgb_start + rgb_size
    return RgbdPacket(
        rgb_jpeg=payload[rgb_start:depth_start],
        depth_png=payload[depth_start:],
        width=width,
        height=height,
        intrinsics=(fx, fy, cx, cy),
    )


def send_rgbd_packet(connection: socket.socket, packet: RgbdPacket) -> None:
    send_packet(
        connection,
        pack_rgbd_packet(
            packet.rgb_jpeg,
            packet.depth_png,
            width=packet.width,
            height=packet.height,
            intrinsics=packet.intrinsics,
        ),
    )


def recv_rgbd_packet(connection: socket.socket) -> RgbdPacket:
    return unpack_rgbd_packet(recv_packet(connection))
