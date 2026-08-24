#!/usr/bin/env python3
"""Regression checks for stage training with only the classes that have accepted labels."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_yolo11_seg import OFFICIAL_CLASSES, resolve_active_training_names, write_training_dataset_yaml


def main() -> int:
    split_reports = {
        "train": {"instances": {"0": 10, "1": 8, "2": 7, "3": 9}},
        "val": {"instances": {"0": 3, "1": 2, "2": 2, "3": 3}},
    }
    active = resolve_active_training_names(OFFICIAL_CLASSES, split_reports)
    assert active == {
        0: "can",
        1: "watermelon_rind",
        2: "meal_box",
        3: "red_paper_bag",
    }

    with tempfile.TemporaryDirectory(prefix="atec_subset_") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "dataset.yaml"
        source.write_text(yaml.safe_dump({
            "path": str(tmp_path / "dataset"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/val",
            "names": OFFICIAL_CLASSES,
        }, sort_keys=False), encoding="utf-8")
        generated = write_training_dataset_yaml(
            source, yaml.safe_load(source.read_text(encoding="utf-8")), active, tmp_path / "run"
        )
        generated_data = yaml.safe_load(generated.read_text(encoding="utf-8"))
        assert generated_data["names"] == active
        assert "test" not in generated_data, "val must not be advertised as an independent test split"
        assert generated.name == "dataset_training.yaml"

    try:
        resolve_active_training_names(OFFICIAL_CLASSES, {
            "train": {"instances": {"0": 3, "2": 1}},
            "val": {"instances": {"0": 1, "2": 1}},
        })
    except ValueError as exc:
        assert "连续" in str(exc)
    else:
        raise AssertionError("non-contiguous active class IDs must be rejected")

    try:
        resolve_active_training_names(OFFICIAL_CLASSES, {
            "train": {"instances": {"0": 3, "1": 2}},
            "val": {"instances": {"0": 1}},
        })
    except ValueError as exc:
        assert "val" in str(exc) and "watermelon_rind" in str(exc)
    else:
        raise AssertionError("each active class must have independent validation instances")

    print("TRAINING_CLASS_SUBSET_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
