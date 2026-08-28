#!/usr/bin/env python3
"""Regression test for launching the App from a Snap-packaged VS Code terminal."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/atec-app"


def main() -> int:
    source = LAUNCHER.read_text(encoding="utf-8")
    python_exec = 'exec "$APP_PY" -m atec_pipeline.gui_app "$@"'
    assert python_exec in source, "launcher entry point changed; update this regression test"

    # Execute the launcher's real environment-cleanup statements, but print the
    # resulting environment instead of opening a GUI.
    probe_source = source.replace(python_exec, "exec /usr/bin/env -0")
    with tempfile.TemporaryDirectory(prefix="atec_launcher_") as tmp:
        probe = Path(tmp) / "atec-app"
        probe.write_text(probe_source, encoding="utf-8")
        probe.chmod(0o755)

        env = os.environ.copy()
        env.update(
            {
                "GTK_PATH": "/snap/code/current/usr/lib/x86_64-linux-gnu/gtk-3.0",
                "GTK_EXE_PREFIX": "/snap/code/current/usr",
                "GTK_MODULES": "gail:atk-bridge",
            }
        )
        result = subprocess.run(
            [str(probe)],
            env=env,
            check=True,
            stdout=subprocess.PIPE,
        )

    sanitized = {}
    for item in result.stdout.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            sanitized[key.decode()] = value.decode(errors="surrogateescape")

    assert "GTK_PATH" not in sanitized, "Snap GTK_PATH must not reach system PyQt5"
    assert "GTK_EXE_PREFIX" not in sanitized, "Snap GTK_EXE_PREFIX must not reach system PyQt5"
    print("ATEC_APP_LAUNCHER_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
