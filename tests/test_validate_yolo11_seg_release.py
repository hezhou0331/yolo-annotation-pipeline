#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "tools/validate_yolo11_seg_release.py"
    spec = importlib.util.spec_from_file_location("validate_yolo_release", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def expect_runtime_error(callable_, expected: str) -> None:
    try:
        callable_()
    except RuntimeError as exc:
        assert expected in str(exc), exc
    else:
        raise AssertionError(f"expected RuntimeError containing {expected!r}")


def main() -> None:
    release = load_module()
    with tempfile.TemporaryDirectory(prefix="atec_release_validation_test_") as value:
        temp = Path(value)
        results_csv = temp / "results.csv"
        results_csv.write_text(
            "epoch,metrics/precision(M),metrics/recall(M),metrics/mAP50(M),"
            "metrics/mAP50-95(M),metrics/mAP50-95(B)\n"
            "1,0.80,0.81,0.92,0.90,0.40\n"
            "2,0.90,0.91,0.92,0.85,0.60\n",
            encoding="utf-8",
        )
        summary = release.inspect_results_csv(results_csv)
        assert summary["epochs_recorded"] == 2
        assert summary["best_epoch"] == 2
        assert summary["best_metrics"]["metrics/mAP50-95(M)"] == 0.85
        assert summary["selection_fitness"] == 1.45
        assert summary["mask_peak_epoch"] == 1
        assert summary["mask_peak_map50_95"] == 0.90

        invalid_csv = temp / "invalid.csv"
        invalid_csv.write_text(
            "epoch,metrics/precision(M),metrics/recall(M),metrics/mAP50(M),"
            "metrics/mAP50-95(M),metrics/mAP50-95(B)\n"
            "1,0.80,nan,0.82,0.70,0.75\n",
            encoding="utf-8",
        )
        expect_runtime_error(
            lambda: release.inspect_results_csv(invalid_csv), "不是有限数值"
        )

        finalized_checkpoint = {
            "model": object(),
            "epoch": -1,
            "train_args": {
                "task": "segment",
                "name": "release-run",
                "data": "dataset_training.yaml",
            },
        }
        finalized = release.validate_finalized_checkpoint(
            finalized_checkpoint,
            label="best.pt",
            expected_run_name="release-run",
        )
        assert finalized["epoch"] == -1 and finalized["stripped"] is True

        active_epoch = dict(finalized_checkpoint, epoch=7)
        expect_runtime_error(
            lambda: release.validate_finalized_checkpoint(
                active_epoch,
                label="last.pt",
                expected_run_name="release-run",
            ),
            "训练未正常完成，禁止晋升",
        )
        active_optimizer = dict(finalized_checkpoint, optimizer={"state": {}})
        expect_runtime_error(
            lambda: release.validate_finalized_checkpoint(
                active_optimizer,
                label="best.pt",
                expected_run_name="release-run",
            ),
            "训练未正常完成，禁止晋升",
        )

        dataset = temp / "dataset"
        image_dir = dataset / "images/val"
        label_dir = dataset / "labels/val"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for class_id in release.EXPECTED_CLASSES:
            stem = f"sample_{class_id}"
            (image_dir / f"{stem}.png").write_bytes(b"not-decoded-by-this-test")
            (label_dir / f"{stem}.txt").write_text(
                f"{class_id} 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8"
            )
        data_path = dataset / "dataset.yaml"
        data_path.write_text(
            yaml.safe_dump(
                {
                    "path": ".",
                    "train": "images/train",
                    "val": "images/val",
                    "test": ["images/test-a", "images/test-b"],
                    "names": release.EXPECTED_CLASSES,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        source_data_yaml = data_path.read_bytes()
        samples = release.select_val_samples(data_path)
        assert set(samples) == set(range(9))
        assert all(path.is_file() for path in samples.values())

        class FakeSegmentMetrics:
            ap_class_index = list(range(9))
            map = 0.75
            map50 = 0.90
            map75 = 0.80
            mp = 0.85
            mr = 0.86

            def class_result(self, index):
                return 0.80 + index / 100, 0.81, 0.90, 0.75

        class FakeMetrics:
            seg = FakeSegmentMetrics()
            results_dict = {"metrics/mAP50-95(M)": 0.75}

        class FakeModel:
            task = "segment"
            names = release.EXPECTED_CLASSES
            validated_data_path = None

            def val(self, **kwargs):
                assert kwargs["split"] == "val" and kwargs["plots"] is False
                resolved_path = Path(kwargs["data"])
                assert resolved_path != data_path
                assert resolved_path.is_file()
                resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
                assert Path(resolved["path"]).is_absolute()
                assert Path(resolved["path"]) == dataset.resolve()
                assert resolved["train"] == "images/train"
                assert resolved["val"] == "images/val"
                assert resolved["test"] == ["images/test-a", "images/test-b"]
                assert release.normalize_names(resolved["names"]) == release.EXPECTED_CLASSES
                self.validated_data_path = resolved_path
                return FakeMetrics()

            def predict(self, **kwargs):
                assert kwargs["save"] is False
                return [object()]

        model = FakeModel()
        assert release.validate_model_identity(model) == release.EXPECTED_CLASSES
        metrics = release.validate_metrics(
            model,
            data_path=data_path,
            device="cpu",
            imgsz=640,
            project=temp / "validation-output",
        )
        assert metrics["mask_map50_95"] == 0.75
        assert set(metrics["per_class"]) == set(release.EXPECTED_CLASSES.values())
        assert model.validated_data_path is not None
        assert not model.validated_data_path.exists()
        assert data_path.read_bytes() == source_data_yaml
        completed = release.smoke_predict(model, samples, device="cpu", imgsz=640)
        assert set(completed) == set(release.EXPECTED_CLASSES.values())

        bad_task = FakeModel()
        bad_task.task = "detect"
        expect_runtime_error(
            lambda: release.validate_model_identity(bad_task), "必须为segment"
        )
        bad_names = FakeModel()
        bad_names.names = dict(release.EXPECTED_CLASSES)
        bad_names.names.pop(8)
        expect_runtime_error(
            lambda: release.validate_model_identity(bad_names), "严格等于0-8九类"
        )

        report = {
            "names": release.normalize_names(list(release.EXPECTED_CLASSES.values())),
            "samples": {str(key): str(path) for key, path in samples.items()},
        }
        assert json.loads(json.dumps(report))["names"]["8"] == "sand_bottle"

    print("test_validate_yolo11_seg_release: PASS")


if __name__ == "__main__":
    main()
