#!/usr/bin/env python3
"""Fast Linux sysfs preflight for Orbbec RGB-D capture.

This check deliberately runs before pyorbbecsdk.  It distinguishes a camera
that Linux cannot enumerate from an SDK/profile error and rejects USB 2 links
that are not suitable for the Gemini 336L RGB-D capture profile.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ORBBEC_VENDOR_ID = "2bc5"
USB3_MIN_SPEED_MBPS = 5000.0


@dataclass(frozen=True)
class OrbbecUsbDevice:
    sysfs_name: str
    product_id: str
    product_name: str
    speed_mbps: float
    bus_number: str
    device_number: str


def _read(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return default


def inspect_orbbec_usb(sysfs_root: Path = Path("/sys/bus/usb/devices")) -> list[OrbbecUsbDevice]:
    devices: list[OrbbecUsbDevice] = []
    if not sysfs_root.is_dir():
        return devices
    for path in sorted(sysfs_root.iterdir(), key=lambda item: item.name):
        if _read(path / "idVendor").lower() != ORBBEC_VENDOR_ID:
            continue
        speed_text = _read(path / "speed", "0")
        try:
            speed = float(speed_text)
        except ValueError:
            speed = 0.0
        devices.append(
            OrbbecUsbDevice(
                sysfs_name=path.name,
                product_id=_read(path / "idProduct", "unknown"),
                product_name=_read(path / "product", "Orbbec device"),
                speed_mbps=speed,
                bus_number=_read(path / "busnum", "?"),
                device_number=_read(path / "devnum", "?"),
            )
        )
    return devices


def readiness_message(devices: Iterable[OrbbecUsbDevice]) -> tuple[bool, str]:
    found = list(devices)
    if not found:
        return False, (
            "未在 USB 总线上发现 Orbbec 相机（厂商ID 2bc5）。\n"
            "采集尚未进入 SDK：请重新插紧相机两端、换一根支持 USB 3 的数据线，"
            "并换到电脑的另一个 USB 3/USB-C 接口。\n"
            "检查命令：lsusb | grep -i 2bc5 && lsusb -t"
        )

    usb3 = [device for device in found if device.speed_mbps >= USB3_MIN_SPEED_MBPS]
    if not usb3:
        details = ", ".join(
            f"{device.product_name} {device.speed_mbps:g}M ({device.sysfs_name})"
            for device in found
        )
        return False, (
            f"发现 Orbbec 相机，但当前是 USB 2/低速连接：{details}。\n"
            "Gemini 336L RGB-D 采集要求稳定的 USB 3 链路；本次不会启动 SDK，"
            "请换 USB 3 接口或数据线。"
        )

    details = ", ".join(
        f"{device.product_name} {device.speed_mbps:g}M ({device.sysfs_name})"
        for device in usb3
    )
    return True, f"Orbbec USB 预检通过：USB 3，{details}"


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 Orbbec 相机的 Linux USB 枚举与链路速度")
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys/bus/usb/devices"))
    args = parser.parse_args()
    ok, message = readiness_message(inspect_orbbec_usb(args.sysfs_root))
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
