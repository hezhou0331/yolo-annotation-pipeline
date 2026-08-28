#!/usr/bin/env python3
"""Contract tests for the repository-wide ATEC class configuration."""
from __future__ import annotations

from pathlib import Path

import yaml
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atec_pipeline.object_config import (  # noqa: E402
    is_supported_class_prefix,
    load_class_map,
    merge_compatible_class_maps,
)

EXPECTED_NINE_CLASSES = {
    0: "can",
    1: "watermelon_rind",
    2: "meal_box",
    3: "red_paper_bag",
    4: "blue_bin",
    5: "green_bin",
    6: "red_bin",
    7: "purple_paper_bag",
    8: "sand_bottle",
}


def main() -> int:
    canonical_path = ROOT / "configs/atec_objects.yaml"
    config = yaml.safe_load(canonical_path.read_text(encoding="utf-8"))
    official = load_class_map(canonical_path)
    assert official == EXPECTED_NINE_CLASSES
    assert not (ROOT / "projects/atec_real/atec_objects.yaml").exists()

    teacher = yaml.safe_load((ROOT / "configs/xcx_teacher.yaml").read_text(encoding="utf-8"))
    assert "official_classes" not in teacher
    assert set(teacher["source_to_official"].values()) <= set(official.values())

    project_example = yaml.safe_load((ROOT / "configs/project_example.yaml").read_text(encoding="utf-8"))
    assert project_example["classes"] == official

    bonus = {int(class_id): str(name) for class_id, name in config["bonus_classes"]["proposed_if_enabled"].items()}
    assert not (set(official) & set(bonus)), (official, bonus)
    assert min(bonus) == len(official), (official, bonus)

    historical_seven = {key: EXPECTED_NINE_CLASSES[key] for key in range(7)}
    assert is_supported_class_prefix(historical_seven, official)
    assert is_supported_class_prefix(official, official)
    assert not is_supported_class_prefix({0: "wrong_can"}, official)
    assert not is_supported_class_prefix({1: "watermelon_rind"}, official)
    assert merge_compatible_class_maps(historical_seven, official) == official
    assert merge_compatible_class_maps(official, historical_seven) == official

    try:
        merge_compatible_class_maps({0: "can"}, {0: "bottle"})
    except ValueError as exc:
        assert "冲突" in str(exc)
    else:
        raise AssertionError("conflicting class maps must be rejected")

    print("OBJECT_CONFIG_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
