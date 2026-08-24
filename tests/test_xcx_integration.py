#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

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


def main() -> None:
    propose = load_module("propose_key_masks", ROOT / "tools/propose_key_masks_yolo_sam2.py")
    train = load_module("train_seg", ROOT / "tools/train_yolo11_seg.py")
    cli = load_module("atec_cli", ROOT / "atec_pipeline/cli.py")

    config = yaml.safe_load((ROOT / "configs/xcx_teacher.yaml").read_text(encoding="utf-8"))
    propose.validate_teacher_config(config)
    assert propose.normalized_classes(config["official_classes"]) == propose.EXPECTED_CLASSES

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
