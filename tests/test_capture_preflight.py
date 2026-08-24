#!/usr/bin/env python3
"""Regression test: USB readiness must be checked before starting mamba/SDK."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="atec_capture_preflight_") as tmp:
        temp = Path(tmp)
        empty_sysfs = temp / "usb-devices"
        empty_sysfs.mkdir()
        marker = temp / "mamba-was-run"
        fake_mamba = temp / "mamba"
        fake_mamba.write_text(
            f"#!/bin/sh\ntouch '{marker}'\nexit 0\n",
            encoding="utf-8",
        )
        fake_mamba.chmod(0o755)

        env = os.environ.copy()
        env["MAMBA"] = str(fake_mamba)
        env["ATEC_ORBBEC_SYSFS_ROOT"] = str(empty_sysfs)
        result = subprocess.run(
            [str(ROOT / "scripts/capture_orbbec.sh"), str(temp / "output"), "--check-only"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        assert result.returncode == 1, result.stdout
        assert not marker.exists(), "mamba/SDK must not start when USB preflight fails"
        assert "未在 USB 总线上发现" in result.stdout

    print("CAPTURE_PREFLIGHT_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
