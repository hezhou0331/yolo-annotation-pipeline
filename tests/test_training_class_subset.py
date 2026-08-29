#!/usr/bin/env python3
"""Regression checks for stage training with only the classes that have accepted labels."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_yolo11_seg import (
    OFFICIAL_CLASSES,
    claim_new_run_directory,
    resolve_active_training_names,
    resume_training,
    validate_run_target,
    write_training_dataset_yaml,
)


def expect_runtime_error(callable_, expected: str) -> None:
    try:
        callable_()
    except RuntimeError as exc:
        assert expected in str(exc), exc
    else:
        raise AssertionError(f"expected RuntimeError containing {expected!r}")


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
        expect_runtime_error(
            lambda: validate_run_target(tmp_path / "run", resume=False),
            "拒绝覆盖",
        )
        try:
            write_training_dataset_yaml(
                source,
                yaml.safe_load(source.read_text(encoding="utf-8")),
                active,
                tmp_path / "run",
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("run-local dataset YAML must never be overwritten")

        claimed = tmp_path / "claimed-run"
        validate_run_target(claimed, resume=False)
        claim_new_run_directory(claimed)
        assert claimed.is_dir()
        expect_runtime_error(
            lambda: claim_new_run_directory(claimed),
            "拒绝覆盖",
        )

        resume_run = tmp_path / "resume-run"
        resume_run.mkdir()
        expect_runtime_error(
            lambda: validate_run_target(resume_run, resume=True),
            "dataset YAML",
        )
        resume_data = resume_run / "dataset_training.yaml"
        resume_data.write_text(
            yaml.safe_dump({"names": active}, sort_keys=False), encoding="utf-8"
        )
        expect_runtime_error(
            lambda: validate_run_target(resume_run, resume=True),
            "weights/last.pt",
        )
        last_weights = resume_run / "weights/last.pt"
        last_weights.parent.mkdir()
        last_weights.write_bytes(b"checkpoint")
        original_yaml = resume_data.read_bytes()
        original_mtime = resume_data.stat().st_mtime_ns

        class FakeYOLO:
            loaded_path = None
            train_kwargs = None
            train_calls = 0
            checkpoint = {
                "epoch": 12,
                "optimizer": {"state": {}},
                "train_args": {
                    "project": str(resume_run.parent),
                    "name": resume_run.name,
                    "data": str(resume_data),
                },
            }
            task = "segment"
            names = active

            def __init__(self, path):
                FakeYOLO.loaded_path = path
                self.ckpt = FakeYOLO.checkpoint

            def train(self, **kwargs):
                FakeYOLO.train_calls += 1
                FakeYOLO.train_kwargs = kwargs
                return "resumed"

        assert resume_training(FakeYOLO, resume_run, active) == "resumed"
        assert Path(FakeYOLO.loaded_path) == last_weights.resolve()
        assert FakeYOLO.train_kwargs == {"resume": True}
        assert FakeYOLO.train_calls == 1
        assert resume_data.read_bytes() == original_yaml
        assert resume_data.stat().st_mtime_ns == original_mtime

        def assert_resume_rejected(checkpoint, expected):
            FakeYOLO.checkpoint = checkpoint
            FakeYOLO.train_kwargs = None
            FakeYOLO.train_calls = 0
            expect_runtime_error(
                lambda: resume_training(FakeYOLO, resume_run, active), expected
            )
            assert FakeYOLO.train_calls == 0 and FakeYOLO.train_kwargs is None

        valid_checkpoint = FakeYOLO.checkpoint
        assert_resume_rejected(
            {**valid_checkpoint, "epoch": -1, "optimizer": None},
            "不能resume",
        )
        assert_resume_rejected(
            {
                **valid_checkpoint,
                "train_args": {
                    **valid_checkpoint["train_args"],
                    "project": str(tmp_path / "historical-project"),
                },
            },
            "project来源不符",
        )
        assert_resume_rejected(
            {
                **valid_checkpoint,
                "train_args": {
                    **valid_checkpoint["train_args"],
                    "name": "historical-run",
                },
            },
            "name来源不符",
        )
        assert_resume_rejected(
            {
                **valid_checkpoint,
                "train_args": {
                    **valid_checkpoint["train_args"],
                    "data": str(tmp_path / "historical-dataset.yaml"),
                },
            },
            "data来源不符",
        )

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
