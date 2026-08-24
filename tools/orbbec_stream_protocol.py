#!/usr/bin/env python3
"""Small length-prefixed protocol used between Orbbec and YOLO environments."""
from __future__ import annotations

import socket
import struct

_HEADER = struct.Struct("!I")
MAX_PACKET_BYTES = 16 * 1024 * 1024


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
