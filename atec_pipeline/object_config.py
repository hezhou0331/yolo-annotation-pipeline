"""Shared ATEC class-map loading and compatibility checks.

The YAML file is the single source of truth.  GUI and CLI tools may expose the
classes differently, but they must not carry independent hard-coded ID maps.
Historical datasets remain valid when their names form a contiguous prefix of
the current class map (for example the original seven-class project).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "atec_objects.yaml"


def normalize_class_map(raw: Any) -> dict[int, str]:
    """Normalize a list/dict class declaration and enforce YOLO ID safety."""

    if isinstance(raw, list):
        classes = {index: str(name) for index, name in enumerate(raw)}
    elif isinstance(raw, dict):
        classes = {int(class_id): str(name) for class_id, name in raw.items()}
    else:
        raise ValueError("classes必须是列表或ID到名称的映射")
    classes = dict(sorted(classes.items()))
    if not classes:
        raise ValueError("classes不能为空")
    if list(classes) != list(range(len(classes))):
        raise ValueError(f"类别ID必须从0连续编号，实际为{list(classes)}")
    names = list(classes.values())
    if any(not name.strip() for name in names):
        raise ValueError("类别名称不能为空")
    if len(set(names)) != len(names):
        raise ValueError("类别名称不能重复")
    return classes


def load_class_map(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[int, str]:
    """Load the official ordered class map from ``atec_objects.yaml``."""

    path = Path(config_path).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置顶层必须是映射：{path}")
    return normalize_class_map(data.get("classes"))


def is_supported_class_prefix(candidate: Any, official: Any) -> bool:
    """Whether candidate is an exact contiguous prefix of official classes."""

    try:
        candidate_map = normalize_class_map(candidate)
        official_map = normalize_class_map(official)
    except (TypeError, ValueError):
        return False
    if len(candidate_map) > len(official_map):
        return False
    return all(official_map[class_id] == name for class_id, name in candidate_map.items())


def require_supported_class_prefix(
    candidate: Any,
    official: Any,
    *,
    context: str = "类别配置",
) -> dict[int, str]:
    """Normalize candidate or raise an actionable incompatibility error."""

    candidate_map = normalize_class_map(candidate)
    official_map = normalize_class_map(official)
    if not is_supported_class_prefix(candidate_map, official_map):
        raise ValueError(
            f"{context}必须是当前官方类别的连续前缀；实际={candidate_map}，官方={official_map}"
        )
    return candidate_map


def merge_compatible_class_maps(*class_maps: Any) -> dict[int, str]:
    """Return the longest of compatible prefix maps, rejecting ID conflicts."""

    normalized = [normalize_class_map(value) for value in class_maps if value]
    if not normalized:
        raise ValueError("没有可合并的类别配置")
    longest = max(normalized, key=len)
    for candidate in normalized:
        if not is_supported_class_prefix(candidate, longest):
            raise ValueError(f"类别配置冲突：{candidate} 与 {longest}")
    return dict(longest)


def class_names(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[str, ...]:
    """Return class names in numeric order for argparse choices."""

    return tuple(load_class_map(config_path).values())
