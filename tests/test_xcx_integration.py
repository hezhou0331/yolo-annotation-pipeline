#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def assert_relative_path(value: str, *, base: Path, expected: Path) -> None:
    assert not Path(value).is_absolute(), value
    assert (base / value).resolve() == expected.resolve(), (value, base, expected)


def check_portable_proposal_relocation(propose, temp: Path) -> None:
    source_project = temp / "proposal_project_old"
    scene = source_project / "data/scenes/can/portable_scene"
    rgb_dir = scene / "rgb"
    rgb_dir.mkdir(parents=True)
    image_path = rgb_dir / "000001.png"
    image = np.full((24, 32, 3), 90, np.uint8)
    assert cv2.imwrite(str(image_path), image)

    manifest_path = source_project / "manifests/portable.yaml"
    manifest_path.parent.mkdir(parents=True)
    key_dir = source_project / "data/key_masks/portable_scene/can_01"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "classes": {0: "can"},
                "project": {
                    "scene": "../data/scenes/can/portable_scene",
                    "split": "train",
                },
                "instances": [
                    {
                        "instance_id": "can_01",
                        "class_id": 0,
                        "class_name": "can",
                        "key_mask_dir": "../data/key_masks/portable_scene/can_01",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    segments_path = scene / "project_reports/segments.json"
    segments_path.parent.mkdir(parents=True)
    segments_path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "segments": [
                    {
                        "segment_id": 0,
                        "start_id": "000001",
                        "missing_key_masks": ["can_01"],
                        "required_key_mask_paths": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assets = source_project / "assets"
    assets.mkdir(parents=True)
    detector_weights = assets / "detector.pt"
    sam_weights = assets / "sam2.pt"
    detector_weights.write_bytes(b"fake-detector")
    sam_weights.write_bytes(b"fake-sam")
    teacher_path = source_project / "reports/teacher.yaml"
    teacher_path.parent.mkdir(parents=True)
    teacher_path.write_text(
        yaml.safe_dump(
            {
                "source_to_official": {"yellow_can": "can"},
                "hard_negative_classes": [],
                "detectors": [
                    {
                        "name": "portable_detector",
                        "weights": "../assets/detector.pt",
                        "proposal_source": True,
                        "allowed_official_classes": ["can"],
                    }
                ],
                "sam2": {"model": "../assets/sam2.pt", "device": "cpu", "imgsz": 640},
                "inference": {"device": "cpu", "imgsz": 640},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class FakeYOLO:
        def __init__(self, weights):
            assert Path(weights).is_file()

        def predict(self, **_kwargs):
            return [object()]

    class FakeSAM:
        def __init__(self, weights):
            assert Path(weights).is_file()

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    fake_ultralytics.SAM = FakeSAM
    detection = {
        "detector": "portable_detector",
        "source_class_id": 0,
        "source_class": "yellow_can",
        "confidence": 0.95,
        "bbox_xyxy": [4.0, 3.0, 24.0, 20.0],
    }
    generated_mask = np.zeros((24, 32), np.uint8)
    generated_mask[3:20, 4:24] = 255
    output = source_project / "data/candidate_masks/portable_scene"
    with (
        patch.dict(sys.modules, {"ultralytics": fake_ultralytics}),
        patch.object(propose, "extract_detections", return_value=[detection]),
        patch.object(propose, "sam_mask_from_box", return_value=generated_mask),
    ):
        result = propose.propose(
            argparse.Namespace(
                manifest=manifest_path,
                segments=segments_path,
                teacher_config=teacher_path,
                output=output,
                segment_id=None,
                check_only=False,
            )
        )
    assert result == 0

    proposal_files = sorted((output / "000001").glob("*.json"))
    assert len(proposal_files) == 1
    proposal_path = proposal_files[0]
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    assert proposal["format_version"] == 2 and proposal["path_base"] == "."
    assert_relative_path(
        proposal["manifest"], base=proposal_path.parent, expected=manifest_path
    )
    assert_relative_path(
        proposal["scene"], base=proposal_path.parent, expected=scene
    )
    assert_relative_path(
        proposal["image"], base=proposal_path.parent, expected=image_path
    )
    assert_relative_path(
        proposal["candidate_mask"],
        base=proposal_path.parent,
        expected=proposal_path.with_suffix(".png"),
    )
    assert_relative_path(
        proposal["overlay"],
        base=proposal_path.parent,
        expected=proposal_path.with_name(f"{proposal_path.stem}_overlay.jpg"),
    )
    assert str(propose.ROOT) not in proposal["review_command_template"]
    assert proposal["review_command_template"].startswith("./scripts/atec-pipeline ")

    report_path = output / "proposal_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["format_version"] == 2 and report["path_base"] == "."
    for field, expected in (
        ("manifest", manifest_path),
        ("segments", segments_path),
        ("teacher_config", teacher_path),
        ("scene", scene),
        ("proposal_root", output),
    ):
        assert_relative_path(report[field], base=report_path.parent, expected=expected)
    embedded = report["proposals"][0]
    assert_relative_path(
        embedded["proposal_json"], base=report_path.parent, expected=proposal_path
    )
    assert_relative_path(
        embedded["candidate_mask"],
        base=report_path.parent,
        expected=proposal_path.with_suffix(".png"),
    )

    moved_project = temp / "proposal_project_moved"
    shutil.move(str(source_project), str(moved_project))
    moved_proposal = moved_project / proposal_path.relative_to(source_project)
    propose.promote(
        argparse.Namespace(proposal_json=moved_proposal, instance_id="can_01")
    )
    moved_key_dir = moved_project / key_dir.relative_to(source_project)
    accepted = moved_key_dir / "000001.png"
    assert accepted.is_file()
    receipt_path = accepted.with_suffix(".promotion.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["format_version"] == 2 and receipt["path_base"] == "."
    assert_relative_path(
        receipt["proposal_json"], base=receipt_path.parent, expected=moved_proposal
    )
    assert_relative_path(
        receipt["destination"], base=receipt_path.parent, expected=accepted
    )


def main() -> None:
    propose = load_module("propose_key_masks", ROOT / "tools/propose_key_masks_yolo_sam2.py")
    train = load_module("train_seg", ROOT / "tools/train_yolo11_seg.py")
    cli = load_module("atec_cli", ROOT / "atec_pipeline/cli.py")
    evaluate = load_module("evaluate_candidates", ROOT / "tools/evaluate_yolo11_seg_candidates.py")

    config = yaml.safe_load((ROOT / "configs/xcx_teacher.yaml").read_text(encoding="utf-8"))
    propose.validate_teacher_config(config)
    assert "official_classes" not in config
    assert propose.normalized_classes(list(propose.EXPECTED_CLASSES.values())) == propose.EXPECTED_CLASSES
    assert propose.EXPECTED_CLASSES[7] == "purple_paper_bag"
    assert propose.EXPECTED_CLASSES[8] == "sand_bottle"
    assert config["source_to_official"]["purple_paper_bag"] == "purple_paper_bag"
    assert "purple_paper_bag" not in config["hard_negative_classes"]
    invalid_config = dict(config)
    invalid_config["source_to_official"] = {**config["source_to_official"], "mystery": "not_official"}
    try:
        propose.validate_teacher_config(invalid_config)
    except ValueError as exc:
        assert "非正式类别" in str(exc)
    else:
        raise AssertionError("xcx映射目标必须来自canonical类别配置")

    mapping = config["source_to_official"]
    negatives = set(config["hard_negative_classes"])
    detector = config["detectors"][0]
    missing = {"can": ["can_01", "can_02"]}
    hard = propose.classify_detection("plastic_bottle", detector, mapping, negatives, missing)
    assert hard == {"kind": "hard_negative"}
    mapped = propose.classify_detection("yellow_can", detector, mapping, negatives, missing)
    assert mapped["kind"] == "proposal" and mapped["candidate_instance_ids"] == ["can_01", "can_02"]
    assert len(mapped["candidate_instance_ids"]) == 2  # 同类多实例必须保留给人工选择
    cross = propose.classify_detection("yellow_can", config["detectors"][2], mapping, negatives, missing)
    assert cross["kind"] == "crosscheck" and cross["classification"] == "crosscheck_only"

    with tempfile.TemporaryDirectory(prefix="atec_xcx_test_") as value:
        temp = Path(value)
        image = temp / "frame.png"
        cv2.imwrite(str(image), np.full((24, 32, 3), 90, np.uint8))

        class FakeTensor:
            def __init__(self, array): self.array = np.asarray(array)
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.array

        class FakeMasks:
            data = [FakeTensor(np.pad(np.ones((6, 8), np.float32), ((2, 2), (1, 1))))]

        class FakeResult:
            masks = FakeMasks()

        class FakeSAM:
            def predict(self, **kwargs):
                assert kwargs["bboxes"] == [[1.0, 2.0, 20.0, 21.0]]
                return [FakeResult()]

        mask = propose.sam_mask_from_box(FakeSAM(), image, [1.0, 2.0, 20.0, 21.0], "cpu", 640)
        assert mask.shape == (24, 32) and mask.dtype == np.uint8 and np.any(mask)

        manifest_path = temp / "manifest.yaml"
        key_dir = temp / "accepted" / "can_01"
        manifest_path.write_text(yaml.safe_dump({
            "classes": propose.EXPECTED_CLASSES,
            "project": {"scene": str(temp), "split": "train"},
            "instances": [{
                "instance_id": "can_01", "class_id": 0, "class_name": "can", "key_mask_dir": str(key_dir),
            }],
        }, sort_keys=False), encoding="utf-8")
        val_manifest = temp / "val_manifest.yaml"
        val_manifest.write_text(yaml.safe_dump({
            "classes": propose.EXPECTED_CLASSES,
            "project": {"scene": str(temp), "split": "val"},
            "instances": [{"instance_id": "can_01", "class_id": 0, "class_name": "can", "key_mask_dir": str(key_dir)}],
        }, sort_keys=False), encoding="utf-8")
        empty_segments = temp / "segments.json"
        empty_segments.write_text(json.dumps({"segments": []}), encoding="utf-8")

        historical_seven = {class_id: propose.EXPECTED_CLASSES[class_id] for class_id in range(7)}
        historical_manifest = temp / "historical_seven.yaml"
        historical_manifest.write_text(yaml.safe_dump({
            "classes": historical_seven,
            "project": {"scene": str(temp), "split": "train"},
            "instances": [{
                "instance_id": "can_01", "class_id": 0, "class_name": "can", "key_mask_dir": str(key_dir),
            }],
        }, sort_keys=False), encoding="utf-8")
        historical_context = propose.load_context(
            historical_manifest, empty_segments, ROOT / "configs/xcx_teacher.yaml"
        )
        assert historical_context["manifest"]["classes"] == historical_seven

        incompatible_manifest = temp / "incompatible.yaml"
        incompatible_manifest.write_text(yaml.safe_dump({
            "classes": {**historical_seven, 6: "wrong_red_bin"},
            "project": {"scene": str(temp), "split": "train"},
            "instances": [{
                "instance_id": "can_01", "class_id": 0, "class_name": "can", "key_mask_dir": str(key_dir),
            }],
        }, sort_keys=False), encoding="utf-8")
        try:
            propose.load_context(incompatible_manifest, empty_segments, ROOT / "configs/xcx_teacher.yaml")
        except ValueError as exc:
            assert "连续前缀" in str(exc)
        else:
            raise AssertionError("xcx候选必须拒绝名称冲突的历史Manifest")

        class HistoricalModel:
            names = historical_seven

        class CurrentModel:
            names = propose.EXPECTED_CLASSES

        class IncompatibleModel:
            names = {**historical_seven, 6: "wrong_red_bin"}

        assert evaluate.validated_model_names(HistoricalModel()) == historical_seven
        assert evaluate.validated_model_names(CurrentModel()) == propose.EXPECTED_CLASSES
        try:
            evaluate.validated_model_names(IncompatibleModel())
        except RuntimeError as exc:
            assert "连续前缀" in str(exc)
        else:
            raise AssertionError("候选评估必须拒绝类别名称冲突的模型")

        assert train.is_supported_class_prefix(historical_seven, train.OFFICIAL_CLASSES)
        assert not train.is_supported_class_prefix(IncompatibleModel.names, train.OFFICIAL_CLASSES)
        try:
            propose.propose(argparse.Namespace(
                manifest=val_manifest, segments=empty_segments,
                teacher_config=ROOT / "configs/xcx_teacher.yaml", output=None,
                segment_id=None, check_only=True,
            ))
        except SystemExit as exc:
            assert "拒绝在val" in str(exc)
        else:
            raise AssertionError("val manifest必须拒绝xcx候选")

        candidate = temp / "candidate.png"
        cv2.imwrite(str(candidate), mask)
        proposal_json = temp / "proposal.json"
        proposal_json.write_text(json.dumps({
            "manifest": str(manifest_path), "official_class": "can", "frame_id": "000001",
            "candidate_instance_ids": ["can_01"], "candidate_mask": str(candidate),
        }), encoding="utf-8")
        propose.promote(argparse.Namespace(proposal_json=proposal_json, instance_id="can_01"))
        accepted = key_dir / "000001.png"
        assert accepted.is_file()
        try:
            propose.promote(argparse.Namespace(proposal_json=proposal_json, instance_id="can_01"))
        except SystemExit as exc:
            assert "拒绝覆盖" in str(exc)
        else:
            raise AssertionError("promote必须拒绝覆盖人工已接受Mask")

        check_portable_proposal_relocation(propose, temp)

        dataset = temp / "dataset"
        image_dir = dataset / "images/train"
        label_dir = dataset / "labels/train"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        cv2.imwrite(str(image_dir / "negative_001.png"), np.zeros((16, 16, 3), np.uint8))
        (label_dir / "negative_001.txt").write_text("", encoding="utf-8")
        allowed = train.validate_split(dataset, "images/train", train.OFFICIAL_CLASSES, {"negative_001"})
        assert not allowed["errors"] and allowed["reviewed_negative_count"] == 1
        rejected = train.validate_split(dataset, "images/train", train.OFFICIAL_CLASSES, set())
        assert any("未人工确认的空标签" in item for item in rejected["errors"])

        reports = dataset / "project_reports"
        reports.mkdir(parents=True)
        for split, video in (("train", "video_train"), ("val", "video_val")):
            (reports / f"{split}_report.json").write_text(json.dumps({
                "scene": str(temp / split), "split": split,
                "capture_session_id": "session_shared", "source_video_id": video,
                "frame_status_counts": {"accepted": 1},
            }), encoding="utf-8")
        leakage = train.inspect_scene_reports(dataset)
        assert leakage["reused_train_val_capture_sessions"]
        assert not leakage["reused_train_val_source_videos"]

    class FakeParam:
        def __init__(self, shape, count): self.shape, self._count = shape, count
        def numel(self): return self._count

    target = {
        "model.0.weight": FakeParam((2, 2), 4),
        "model.1.weight": FakeParam((3, 3), 9),
        "model.2.head": FakeParam((4, 4), 16),
    }
    source = {
        "model.0.weight": FakeParam((2, 2), 4),
        "model.1.weight": FakeParam((3, 3), 9),
        "model.2.head": FakeParam((8, 8), 64),
    }
    selected, report = train.select_compatible_transfer_state(target, source, ("model.2.",))
    assert set(selected) == {"model.0.weight", "model.1.weight"}
    assert report["loaded_parameter_fraction"] == 1.0

    parsed = cli.parser().parse_args(["propose", "manifest.yaml", "segments.json", "--check-only"])
    assert parsed.command == "propose" and parsed.check_only
    parsed_train = cli.parser().parse_args(["train", "dataset.yaml", "--init-mode", "xcx-transfer"])
    assert parsed_train.init_mode == "xcx-transfer" and parsed_train.seed == 0

    print("XCX_INTEGRATION_ASSERTIONS_PASSED")


if __name__ == "__main__":
    main()
