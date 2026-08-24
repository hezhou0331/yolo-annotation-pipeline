#!/usr/bin/env python3
"""CPU-only tests for Orbbec USB readiness diagnostics."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from orbbec_usb_diagnostics import inspect_orbbec_usb, readiness_message  # noqa: E402


def write_device(root: Path, name: str, *, vendor: str, product: str, speed: str) -> None:
    device = root / name
    device.mkdir(parents=True)
    (device / "idVendor").write_text(vendor, encoding="ascii")
    (device / "idProduct").write_text(product, encoding="ascii")
    (device / "product").write_text("Orbbec Gemini 336L", encoding="utf-8")
    (device / "speed").write_text(speed, encoding="ascii")
    (device / "busnum").write_text("8", encoding="ascii")
    (device / "devnum").write_text("3", encoding="ascii")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="atec_usb_diag_") as tmp:
        sysfs = Path(tmp)
        devices = inspect_orbbec_usb(sysfs)
        ok, message = readiness_message(devices)
        assert not ok
        assert "未在 USB 总线上发现" in message
        assert "2bc5" in message

        write_device(sysfs, "7-1.1", vendor="2bc5", product="0807", speed="480")
        devices = inspect_orbbec_usb(sysfs)
        ok, message = readiness_message(devices)
        assert not ok
        assert "USB 2" in message
        assert "480" in message

        (sysfs / "7-1.1" / "speed").write_text("5000", encoding="ascii")
        devices = inspect_orbbec_usb(sysfs)
        ok, message = readiness_message(devices)
        assert ok
        assert "USB 3" in message
        assert "5000" in message

        write_device(sysfs, "1-2", vendor="1234", product="5678", speed="5000")
        assert len(inspect_orbbec_usb(sysfs)) == 1

    print("ORBBEC_USB_DIAGNOSTICS_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
