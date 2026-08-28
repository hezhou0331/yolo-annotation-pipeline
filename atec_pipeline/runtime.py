"""Resolve the Python interpreters used by the ATEC workflow.

The short ``ATEC_*_PY`` names are canonical.  The older
``ATEC_*_PYTHON`` names remain supported so existing operator shells keep
working while the repository moves between machines.
"""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any


DEFAULT_ORBBEC_PY = Path.home() / "miniforge3/envs/orbbec/bin/python"
DEFAULT_FP_PY = Path.home() / "miniforge3/envs/foundationpose/bin/python"
DEFAULT_YOLO_PY = Path.home() / "miniforge3/envs/yolo11/bin/python"

_RUNTIME_DEFAULTS = {
    "orbbec": DEFAULT_ORBBEC_PY,
    "foundationpose": DEFAULT_FP_PY,
    "yolo11": DEFAULT_YOLO_PY,
}
_RUNTIME_ENV_NAMES = {
    "orbbec": ("ATEC_ORBBEC_PY", "ATEC_ORBBEC_PYTHON"),
    "foundationpose": ("ATEC_FP_PY", "ATEC_FP_PYTHON"),
    "yolo11": ("ATEC_YOLO_PY", "ATEC_YOLO_PYTHON"),
}


def _environment_value(
    names: tuple[str, ...],
    environ: Mapping[str, str] | None = None,
) -> str | None:
    values = os.environ if environ is None else environ
    for name in names:
        value = values.get(name)
        if value is not None and value.strip():
            return value
    return None


def _resolved_path(value: Any, *, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir).expanduser() / path
    return Path(os.path.abspath(path))


def interpreter_path(
    runtime: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve one runtime, preferring its canonical variable over its alias."""

    if runtime not in _RUNTIME_DEFAULTS:
        raise KeyError(f"未知ATEC运行环境：{runtime}")
    value = _environment_value(_RUNTIME_ENV_NAMES[runtime], environ)
    return _resolved_path(value if value is not None else _RUNTIME_DEFAULTS[runtime])


def interpreters(
    *, environ: Mapping[str, str] | None = None
) -> dict[str, Path]:
    """Return all configured workflow interpreters."""

    return {
        runtime: interpreter_path(runtime, environ=environ)
        for runtime in _RUNTIME_DEFAULTS
    }


def resolve_sam2_python(
    manifest_value: str | Path | None = None,
    *,
    manifest_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve SAM2 Python with environment > Manifest > default priority."""

    environment_value = _environment_value(_RUNTIME_ENV_NAMES["yolo11"], environ)
    if environment_value is not None:
        return _resolved_path(environment_value)
    if manifest_value not in (None, ""):
        return _resolved_path(manifest_value, base_dir=manifest_dir)
    return _resolved_path(DEFAULT_YOLO_PY)
