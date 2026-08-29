#!/usr/bin/env python3
"""Regression test for the external-webcam YOLO launcher."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "atec-live-yolo"


def main() -> int:
    assert LAUNCHER.is_file(), "missing scripts/atec-live-yolo"
    assert os.access(LAUNCHER, os.X_OK), "scripts/atec-live-yolo must be executable"
    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    expected_model_default = (
        'MODEL="${ATEC_YOLO_MODEL:-$ROOT/runs/segment/'
        'atec_9class_reviewed_20260829/weights/best.pt}"'
    )
    assert launcher_source.count(expected_model_default) == 1, (
        "launcher must default exactly once to the reviewed nine-class best.pt"
    )

    with tempfile.TemporaryDirectory(prefix="atec_live_yolo_") as tmp_dir:
        tmp = Path(tmp_dir)
        fake_python = tmp / "python"
        fake_python.write_text(
            "#!/bin/sh\nprintf '%s\\n' \"$@\"\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        fake_model = tmp / "best.pt"
        fake_model.write_bytes(b"test")

        env = os.environ.copy()
        env.update(
            {
                "ATEC_YOLO_PY": str(fake_python),
                "ATEC_YOLO_MODEL": str(fake_model),
            }
        )
        result = subprocess.run(
            [str(LAUNCHER), "--conf", "0.30"],
            cwd=ROOT,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )

    args = result.stdout.splitlines()
    assert args[0] == str(ROOT / "tools" / "live_yolo11_seg.py")
    assert args[args.index("--model") + 1] == str(fake_model)
    assert args[args.index("--source") + 1] == "0"
    assert args[args.index("--device") + 1] == "0"
    # User options are appended after defaults so argparse uses the last value.
    assert args[-2:] == ["--conf", "0.30"]
    print("LIVE_YOLO_LAUNCHER_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
