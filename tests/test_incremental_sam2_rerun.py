#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from atec_pipeline import cli
import rerun_sam2_range as rerun_module
from rerun_sam2_range import (
    apply_partial_masks,
    merge_propagation_reports,
    normalize_frame_ranges,
    parse_range_spec,
    select_frame_range,
)


def main() -> int:
    assert select_frame_range(
        ("000000", "000001", "000002", "000003"), "000001", "000003"
    ) == ("000001", "000002")
    assert select_frame_range(
        ("000000", "000001", "000002"), "000002", None
    ) == ("000002",)
    for start, end, expected in [
        ("999999", None, "起始帧"),
        ("000002", "000001", "结束边界"),
    ]:
        try:
            select_frame_range(("000000", "000001", "000002"), start, end)
        except ValueError as exc:
            assert expected in str(exc), str(exc)
        else:
            raise AssertionError("invalid incremental range must fail")

    assert parse_range_spec("000057:000064") == ("000057", "000064")
    assert parse_range_spec("000090:") == ("000090", None)

    assert rerun_module.format_frame_range(
        ("000000", "000001"), "000000", None
    ) == "000000–000001（包含末帧）"
    assert rerun_module.format_frame_range(
        ("000000", "000001"), "000000", "000001"
    ) == "000000–000001（结束边界不包含）"

    try:
        rerun_module.validate_complete_report(
            {"records": [{"id": "000000", "status": "accepted"}]},
            ("000000", "000001"),
        )
    except ValueError as exc:
        assert "旧SAM2报告不完整" in str(exc)
        assert "000001" in str(exc)
    else:
        raise AssertionError("incremental rerun must reject an incomplete old report")
    assert normalize_frame_ranges(
        ("000000", "000001", "000002", "000003", "000004", "000005"),
        (("000001", "000003"), ("000003", "000005")),
    ) == (("000001", "000005"),), "adjacent K ranges should share one SAM2 subprocess"
    assert normalize_frame_ranges(
        ("000000", "000001", "000002", "000003", "000004", "000005"),
        (("000001", "000002"), ("000004", None)),
    ) == (("000001", "000002"), ("000004", None))

    assert rerun_module._validated_ranges(
        ("000000", "000001", "000002", "000003"),
        ("000000", "000001", "000002", "000003"),
        {"0": ("000000", "000001"), "1": ("000002", "000003")},
        (
            {"segment_id": "0", "start_frame": "000001", "end_before_frame": "000002"},
            {"segment_id": "1", "start_frame": "000002", "end_before_frame": None},
        ),
    ) == (("000001", "000002"), ("000002", None)), (
        "adjacent corrections from different automatic segments must remain separate"
    )

    old_report = {
        "scene": "/scene",
        "status_counts": {"accepted": 3, "rejected": 1},
        "reregistration_requests": [
            {"after_frame": "000001", "reason": "old-inside"},
            {"after_frame": "000003", "reason": "old-outside"},
        ],
        "records": [
            {"id": "000000", "status": "accepted", "mode": "old"},
            {"id": "000001", "status": "accepted", "mode": "old"},
            {"id": "000002", "status": "rejected", "mode": "old"},
            {"id": "000003", "status": "accepted", "mode": "old"},
        ],
    }
    partial_report = {
        "records": [
            {"id": "000001", "status": "accepted", "mode": "key_mask", "recovery": {}},
            {"id": "000002", "status": "accepted", "mode": "track", "recovery": {}},
        ],
        "reregistration_requests": [{"after_frame": "000002", "reason": "new-inside"}],
    }
    merged = merge_propagation_reports(old_report, partial_report, ("000001", "000002"))
    assert [row["mode"] for row in merged["records"]] == ["old", "key_mask", "track", "old"]
    assert merged["status_counts"] == {"accepted": 4, "rejected": 0}
    assert [row["after_frame"] for row in merged["reregistration_requests"]] == ["000003", "000002"]
    assert merged["incremental_updates"][-1]["frames"] == ["000001", "000002"]

    with tempfile.TemporaryDirectory(prefix="atec_incremental_masks_") as tmp:
        root = Path(tmp)
        existing = root / "existing"
        partial = root / "partial"
        candidate = root / "candidate"
        existing.mkdir(); partial.mkdir()
        for frame_id, payload in {
            "000000": b"outside-before",
            "000001": b"old-inside-a",
            "000002": b"old-inside-b",
            "000003": b"outside-after",
        }.items():
            (existing / f"{frame_id}.png").write_bytes(payload)
        outside_inode = os.stat(existing / "000003.png").st_ino
        (partial / "000001.png").write_bytes(b"new-inside-a")
        # 000002 is absent from the successful partial output and must remove
        # the stale old mask only inside the selected range.
        apply_partial_masks(existing, partial, candidate, ("000001", "000002"))
        assert (candidate / "000000.png").read_bytes() == b"outside-before"
        assert (candidate / "000001.png").read_bytes() == b"new-inside-a"
        assert not (candidate / "000002.png").exists()
        assert (candidate / "000003.png").read_bytes() == b"outside-after"
        assert os.stat(candidate / "000003.png").st_ino == outside_inode
        assert (existing / "000001.png").read_bytes() == b"old-inside-a"
        assert (existing / "000002.png").read_bytes() == b"old-inside-b"

    parsed = cli.parser().parse_args([
        "rerun-range", "/tmp/demo.yaml", "--instance-id", "can_01",
        "--start-frame", "000057", "--end-before-frame", "000064",
    ])
    captured: list[list[str]] = []
    original_run = cli.run
    cli.run = lambda command, **_kwargs: captured.append([str(x) for x in command]) or 0  # type: ignore[assignment]
    try:
        assert parsed.func(parsed) == 0
    finally:
        cli.run = original_run  # type: ignore[assignment]
    command = captured[0]
    assert command[1].endswith("tools/rerun_sam2_range.py")
    assert command[command.index("--instance-id") + 1] == "can_01"
    assert command[command.index("--start-frame") + 1] == "000057"
    assert command[command.index("--end-before-frame") + 1] == "000064"

    with tempfile.TemporaryDirectory(prefix="atec_incremental_action_") as tmp:
        action_file = Path(tmp) / "player_action.json"
        action_file.write_text(json.dumps({
            "action": "rerun_ranges",
            "instance_id": "can_01",
            "ranges": [
                {"segment_id": "0", "start_frame": "000057", "end_before_frame": "000064"},
                {"segment_id": "0", "start_frame": "000070", "end_before_frame": None},
            ],
        }), encoding="utf-8")
        parsed = cli.parser().parse_args([
            "rerun-range", "/tmp/demo.yaml", "--instance-id", "can_01",
            "--ranges-file", str(action_file),
        ])
        captured.clear()
        cli.run = lambda command, **_kwargs: captured.append([str(x) for x in command]) or 0  # type: ignore[assignment]
        try:
            assert parsed.func(parsed) == 0
        finally:
            cli.run = original_run  # type: ignore[assignment]
        command = captured[0]
        assert command[command.index("--ranges-file") + 1] == str(action_file)

    # End-to-end transaction with a fake SAM2 subprocess: the formal mask
    # directory is swapped only after all partial ranges succeed, outside
    # inodes remain untouched, the full report survives, and aggregation runs
    # exactly once for the whole Review session.
    with tempfile.TemporaryDirectory(prefix="atec_incremental_transaction_") as tmp:
        root = Path(tmp)
        scene = root / "scene"
        rgb = scene / "rgb"
        depth = scene / "depth"
        key_masks = root / "key_masks/can_01"
        output = root / "dataset"
        stage_masks = output / "_staging/scene/can_01/_sam2_masks"
        for directory in (rgb, depth, key_masks, stage_masks):
            directory.mkdir(parents=True, exist_ok=True)
        frame_ids = tuple(f"{index:06d}" for index in range(4))
        for index, frame_id in enumerate(frame_ids):
            image = np.full((12, 16, 3), 20 + index, np.uint8)
            cv2.imwrite(str(rgb / f"{frame_id}.png"), image)
            cv2.imwrite(str(depth / f"{frame_id}.png"), np.ones((12, 16), np.uint16))
            (stage_masks / f"{frame_id}.png").write_bytes(f"old-{frame_id}".encode())
        key = np.zeros((12, 16), np.uint8)
        key[2:8, 3:10] = 255
        for key_frame in ("000000", "000002"):
            cv2.imwrite(str(key_masks / f"{key_frame}.png"), key)
        (scene / "project_reports").mkdir(parents=True)
        (scene / "project_reports/segments.json").write_text(json.dumps({
            "segments": [{"segment_id": 0, "start_id": "000000", "end_id": "000003"}]
        }), encoding="utf-8")
        old_report = {
            "scene": str(scene),
            "output_mask_dir": str(stage_masks),
            "parameters": {"output_mask_dir": str(stage_masks)},
            "status_counts": {"accepted": 4, "rejected": 0},
            "auto_reregistration": {"attempted": 0, "succeeded": 0},
            "reregistration_requests": [],
            "records": [
                {"id": frame_id, "status": "accepted", "mode": "old", "recovery": {}}
                for frame_id in frame_ids
            ],
        }
        (stage_masks / "sam2_propagation_report.json").write_text(
            json.dumps(old_report), encoding="utf-8"
        )
        model = root / "sam2.pt"
        model.write_bytes(b"model")
        manifest = root / "manifest.yaml"
        manifest.write_text(
            "project:\n"
            f"  scene: {scene}\n"
            f"  output: {output}\n"
            f"  sam2_python: {sys.executable}\n"
            f"  sam2_model: {model}\n"
            "  split: train\n"
            "instances:\n"
            "- instance_id: can_01\n"
            "  class_id: 0\n"
            "  class_name: can\n"
            "  tracker: sam2\n"
            f"  key_mask_dir: {key_masks}\n",
            encoding="utf-8",
        )
        action_file = root / "player_action.json"
        action_file.write_text(json.dumps({
            "action": "rerun_ranges",
            "instance_id": "can_01",
            "ranges": [
                {"segment_id": "0", "start_frame": "000000", "end_before_frame": "000001"},
                {"segment_id": "0", "start_frame": "000002", "end_before_frame": "000003"},
            ],
        }), encoding="utf-8")
        before_outside_inode = os.stat(stage_masks / "000003.png").st_ino
        aggregate_calls: list[list[str]] = []
        sam2_calls: dict[int, int] = {}
        attempt_dirs: dict[int, list[Path]] = {}
        original_tool_run = rerun_module._run

        def fake_run(command: list[str], *, dry_run: bool, env: dict[str, str]) -> None:
            del dry_run, env
            if any("propagate_masks_sam2.py" in part for part in command):
                partial_dir = Path(command[command.index("--output-mask-dir") + 1])
                start = int(command[command.index("--start") + 1])
                count = int(command[command.index("--max-frames") + 1])
                sam2_calls[start] = sam2_calls.get(start, 0) + 1
                attempt_dirs.setdefault(start, []).append(partial_dir)
                partial_dir.mkdir(parents=True, exist_ok=True)
                if start == 2 and sam2_calls[start] == 1:
                    (partial_dir / "000002.png").write_bytes(b"stale-partial")
                    raise subprocess.CalledProcessError(1, command)
                selected = frame_ids[start:start + count]
                for frame_id in selected:
                    (partial_dir / f"{frame_id}.png").write_bytes(f"new-{frame_id}".encode())
                partial_report = {
                    "records": [
                        {"id": frame_id, "status": "accepted", "mode": "local", "recovery": {}}
                        for frame_id in selected
                    ],
                    "reregistration_requests": [],
                }
                (partial_dir / "sam2_propagation_report.json").write_text(
                    json.dumps(partial_report), encoding="utf-8"
                )
            else:
                aggregate_calls.append(command)

        rerun_module._run = fake_run
        try:
            assert rerun_module.main([
                "--manifest", str(manifest),
                "--instance-id", "can_01",
                "--ranges-file", str(action_file),
            ]) == 0
        finally:
            rerun_module._run = original_tool_run
        assert (stage_masks / "000000.png").read_bytes() == b"new-000000"
        assert (stage_masks / "000001.png").read_bytes() == b"old-000001"
        assert (stage_masks / "000002.png").read_bytes() == b"new-000002"
        assert (stage_masks / "000003.png").read_bytes() == b"old-000003"
        assert os.stat(stage_masks / "000003.png").st_ino == before_outside_inode
        merged_report = json.loads(
            (stage_masks / "sam2_propagation_report.json").read_text(encoding="utf-8")
        )
        assert len(merged_report["records"]) == 4
        assert merged_report["output_mask_dir"] == str(stage_masks)
        assert merged_report["parameters"]["output_mask_dir"] == str(stage_masks)
        assert [row["frames"] for row in merged_report["incremental_updates"]] == [
            ["000000"], ["000002"],
        ]
        assert sam2_calls == {0: 1, 2: 2}
        assert attempt_dirs[2][0] != attempt_dirs[2][1]
        assert len(aggregate_calls) == 1
        assert "--skip-tracking" in aggregate_calls[0]
        assert not list(stage_masks.parent.glob(".incremental-*"))

    with tempfile.TemporaryDirectory(prefix="atec_incremental_failure_") as tmp:
        root = Path(tmp)
        scene = root / "scene"
        rgb = scene / "rgb"
        key_masks = root / "key_masks/can_01"
        output = root / "dataset"
        stage_masks = output / "_staging/scene/can_01/_sam2_masks"
        for directory in (rgb, key_masks, stage_masks):
            directory.mkdir(parents=True, exist_ok=True)
        image = np.full((10, 10, 3), 40, np.uint8)
        key = np.zeros((10, 10), np.uint8); key[2:7, 2:7] = 255
        for frame_id in ("000000", "000001"):
            cv2.imwrite(str(rgb / f"{frame_id}.png"), image)
            (stage_masks / f"{frame_id}.png").write_bytes(f"old-{frame_id}".encode())
        cv2.imwrite(str(key_masks / "000000.png"), key)
        old_report = {
            "status_counts": {"accepted": 2, "rejected": 0},
            "records": [
                {"id": frame_id, "status": "accepted", "mode": "old", "recovery": {}}
                for frame_id in ("000000", "000001")
            ],
            "reregistration_requests": [],
        }
        (stage_masks / "sam2_propagation_report.json").write_text(json.dumps(old_report), encoding="utf-8")
        model = root / "sam2.pt"; model.write_bytes(b"model")
        manifest = root / "manifest.yaml"
        manifest.write_text(
            "project:\n"
            f"  scene: {scene}\n"
            f"  output: {output}\n"
            f"  sam2_python: {sys.executable}\n"
            f"  sam2_model: {model}\n"
            "instances:\n"
            "- instance_id: can_01\n"
            "  class_id: 0\n"
            "  class_name: can\n"
            "  tracker: sam2\n"
            f"  key_mask_dir: {key_masks}\n",
            encoding="utf-8",
        )
        before = {path.name: path.read_bytes() for path in stage_masks.iterdir()}
        failed_sam2_calls = 0

        def fail_sam2(*_args: object, **_kwargs: object) -> None:
            nonlocal failed_sam2_calls
            failed_sam2_calls += 1
            raise subprocess.CalledProcessError(1, "fake-sam2")

        rerun_module._run = fail_sam2
        try:
            try:
                rerun_module.main([
                    "--manifest", str(manifest),
                    "--instance-id", "can_01",
                    "--start-frame", "000000",
                ])
            except subprocess.CalledProcessError:
                pass
            else:
                raise AssertionError("SAM2 failure must be reported")
        finally:
            rerun_module._run = original_tool_run
        after = {path.name: path.read_bytes() for path in stage_masks.iterdir()}
        assert after == before, "SAM2 failure must leave the formal mask tree/report untouched"
        assert failed_sam2_calls == 2
        assert not list(stage_masks.parent.glob(".incremental-*"))

        signal_calls = 0

        def fail_from_signal(*_args: object, **_kwargs: object) -> None:
            nonlocal signal_calls
            signal_calls += 1
            raise subprocess.CalledProcessError(-15, "fake-sam2")

        rerun_module._run = fail_from_signal
        try:
            try:
                rerun_module.main([
                    "--manifest", str(manifest),
                    "--instance-id", "can_01",
                    "--start-frame", "000000",
                ])
            except subprocess.CalledProcessError as error:
                assert error.returncode == -15
            else:
                raise AssertionError("signal termination must be reported without retry")
        finally:
            rerun_module._run = original_tool_run
        assert signal_calls == 1
        assert {path.name: path.read_bytes() for path in stage_masks.iterdir()} == before
        assert not list(stage_masks.parent.glob(".incremental-*"))

        dry_run_calls = 0

        def record_dry_run(
            command: list[str], *, dry_run: bool, env: dict[str, str]
        ) -> None:
            nonlocal dry_run_calls
            del command, env
            assert dry_run
            dry_run_calls += 1

        rerun_module._run = record_dry_run
        try:
            assert rerun_module.main([
                "--manifest", str(manifest),
                "--instance-id", "can_01",
                "--start-frame", "000000",
                "--dry-run",
            ]) == 0
        finally:
            rerun_module._run = original_tool_run
        assert dry_run_calls == 1
        assert {path.name: path.read_bytes() for path in stage_masks.iterdir()} == before

    with tempfile.TemporaryDirectory(prefix="atec_incremental_aggregate_failure_") as tmp:
        root = Path(tmp)
        scene = root / "scene"
        rgb = scene / "rgb"
        key_masks = root / "key_masks/can_01"
        output = root / "dataset"
        stage_masks = output / "_staging/scene/can_01/_sam2_masks"
        for directory in (rgb, key_masks, stage_masks):
            directory.mkdir(parents=True, exist_ok=True)
        frame_ids = ("000000", "000001")
        image = np.full((10, 10, 3), 50, np.uint8)
        key = np.zeros((10, 10), np.uint8)
        key[2:7, 2:7] = 255
        for frame_id in frame_ids:
            cv2.imwrite(str(rgb / f"{frame_id}.png"), image)
            (stage_masks / f"{frame_id}.png").write_bytes(f"old-{frame_id}".encode())
        cv2.imwrite(str(key_masks / "000000.png"), key)
        old_report = {
            "status_counts": {"accepted": 2, "rejected": 0},
            "records": [
                {"id": frame_id, "status": "accepted", "mode": "old", "recovery": {}}
                for frame_id in frame_ids
            ],
            "reregistration_requests": [],
        }
        (stage_masks / "sam2_propagation_report.json").write_text(
            json.dumps(old_report), encoding="utf-8"
        )
        model = root / "sam2.pt"
        model.write_bytes(b"model")
        manifest = root / "manifest.yaml"
        manifest.write_text(
            "project:\n"
            f"  scene: {scene}\n"
            f"  output: {output}\n"
            f"  sam2_python: {sys.executable}\n"
            f"  sam2_model: {model}\n"
            "instances:\n"
            "- instance_id: can_01\n"
            "  class_id: 0\n"
            "  class_name: can\n"
            "  tracker: sam2\n"
            f"  key_mask_dir: {key_masks}\n",
            encoding="utf-8",
        )
        before = {path.name: path.read_bytes() for path in stage_masks.iterdir()}
        aggregate_calls = 0

        def fail_first_aggregate(
            command: list[str], *, dry_run: bool, env: dict[str, str]
        ) -> None:
            nonlocal aggregate_calls
            del dry_run, env
            if any("propagate_masks_sam2.py" in part for part in command):
                partial_dir = Path(command[command.index("--output-mask-dir") + 1])
                partial_dir.mkdir(parents=True, exist_ok=True)
                for frame_id in frame_ids:
                    (partial_dir / f"{frame_id}.png").write_bytes(f"new-{frame_id}".encode())
                (partial_dir / "sam2_propagation_report.json").write_text(
                    json.dumps({
                        "records": [
                            {
                                "id": frame_id,
                                "status": "accepted",
                                "mode": "local",
                                "recovery": {},
                            }
                            for frame_id in frame_ids
                        ],
                        "reregistration_requests": [],
                    }),
                    encoding="utf-8",
                )
                return
            aggregate_calls += 1
            if aggregate_calls == 1:
                raise subprocess.CalledProcessError(1, "fake-aggregate")

        rerun_module._run = fail_first_aggregate
        try:
            try:
                rerun_module.main([
                    "--manifest", str(manifest),
                    "--instance-id", "can_01",
                    "--start-frame", "000000",
                ])
            except RuntimeError as exc:
                assert "已恢复旧Mask和标签" in str(exc)
            else:
                raise AssertionError("aggregate failure must be reported after rollback")
        finally:
            rerun_module._run = original_tool_run
        after = {path.name: path.read_bytes() for path in stage_masks.iterdir()}
        assert after == before, "aggregate failure must restore the old formal mask tree/report"
        assert aggregate_calls == 2, "rollback must rebuild labels once from the restored old masks"

    print("INCREMENTAL_SAM2_RERUN_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
