#!/usr/bin/env python3
"""Atomically replace only selected SAM2 ranges, then rebuild quality/YOLO output."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable
import uuid

import cv2
import yaml

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from annotate_multinstance_project import add_optional, resolve_path, safe_name
from atec_pipeline.runtime import resolve_sam2_python

FrameRange = tuple[str, str | None]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="局部重跑一个或多个SAM2关键帧区间")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--start-frame")
    source.add_argument("--ranges-file", type=Path)
    parser.add_argument("--end-before-frame")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def parse_range_spec(value: str) -> FrameRange:
    start, separator, end = str(value).strip().partition(":")
    if not separator or not start:
        raise ValueError(f"局部范围必须是 START:END 格式：{value}")
    return start, end or None


def select_frame_range(
    frame_ids: Iterable[str],
    start_frame: str,
    end_before_frame: str | None,
) -> tuple[str, ...]:
    ordered = tuple(str(value) for value in frame_ids)
    if start_frame not in ordered:
        raise ValueError(f"起始帧不存在：{start_frame}")
    start_index = ordered.index(start_frame)
    if end_before_frame is None:
        end_index = len(ordered)
    else:
        if end_before_frame not in ordered:
            raise ValueError(f"结束边界帧不存在：{end_before_frame}")
        end_index = ordered.index(end_before_frame)
        if end_index <= start_index:
            raise ValueError(
                f"结束边界必须位于起始帧之后：{start_frame} -> {end_before_frame}"
            )
    selected = ordered[start_index:end_index]
    if not selected:
        raise ValueError(f"局部传播范围为空：{start_frame} -> {end_before_frame}")
    return selected


def format_frame_range(
    frame_ids: Iterable[str],
    start_frame: str,
    end_before_frame: str | None,
) -> str:
    ordered = tuple(str(value) for value in frame_ids)
    if not ordered:
        raise ValueError("没有可显示的帧")
    if end_before_frame is None:
        return f"{start_frame}–{ordered[-1]}（包含末帧）"
    return f"{start_frame}–{end_before_frame}（结束边界不包含）"


def normalize_frame_ranges(
    frame_ids: Iterable[str], ranges: Iterable[FrameRange]
) -> tuple[FrameRange, ...]:
    """Merge overlapping/adjacent ranges while retaining exclusive boundaries."""
    ordered = tuple(str(value) for value in frame_ids)
    positions = {frame_id: index for index, frame_id in enumerate(ordered)}
    intervals: list[tuple[int, int]] = []
    for start, end in ranges:
        selected = select_frame_range(ordered, start, end)
        start_index = positions[selected[0]]
        end_index = positions[selected[-1]] + 1
        intervals.append((start_index, end_index))
    if not intervals:
        raise ValueError("没有局部传播范围")
    intervals.sort()
    merged: list[list[int]] = []
    for start_index, end_index in intervals:
        if not merged or start_index > merged[-1][1]:
            merged.append([start_index, end_index])
        else:
            merged[-1][1] = max(merged[-1][1], end_index)
    return tuple(
        (ordered[start], ordered[end] if end < len(ordered) else None)
        for start, end in merged
    )


def validate_complete_report(
    report: dict[str, Any],
    active_frame_ids: Iterable[str],
) -> None:
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("旧SAM2报告不完整：缺少records列表")
    reported = {
        str(record.get("id"))
        for record in records
        if isinstance(record, dict) and record.get("id") is not None
    }
    missing = [str(frame_id) for frame_id in active_frame_ids if str(frame_id) not in reported]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(
            f"旧SAM2报告不完整，缺少 {len(missing)} 个活动帧记录：{preview}{suffix}"
        )


def merge_propagation_reports(
    old_report: dict[str, Any],
    partial_report: dict[str, Any],
    replaced_frame_ids: Iterable[str],
) -> dict[str, Any]:
    replaced = {str(frame_id) for frame_id in replaced_frame_ids}
    merged = copy.deepcopy(old_report)
    old_records = [dict(row) for row in old_report.get("records", [])]
    partial_records = {
        str(row.get("id")): dict(row)
        for row in partial_report.get("records", [])
        if str(row.get("id")) in replaced
    }
    missing = replaced.difference(partial_records)
    if missing:
        raise ValueError(f"局部SAM2报告缺少帧记录：{', '.join(sorted(missing))}")
    merged_records = [partial_records.get(str(row.get("id")), row) for row in old_records]
    known = {str(row.get("id")) for row in old_records}
    merged_records.extend(
        partial_records[frame_id] for frame_id in sorted(partial_records) if frame_id not in known
    )
    merged["records"] = merged_records

    count_keys = set((old_report.get("status_counts") or {}).keys()) | {"accepted", "rejected"}
    count_keys.update(str(row.get("status")) for row in merged_records if row.get("status"))
    merged["status_counts"] = {
        status: sum(str(row.get("status")) == status for row in merged_records)
        for status in sorted(count_keys)
    }
    merged["auto_reregistration"] = {
        "attempted": sum(
            bool((row.get("recovery") or {}).get("attempted")) for row in merged_records
        ),
        "succeeded": sum(
            bool((row.get("recovery") or {}).get("succeeded")) for row in merged_records
        ),
    }
    old_requests = [
        dict(row) for row in old_report.get("reregistration_requests", [])
        if str(row.get("after_frame")) not in replaced
    ]
    new_requests = [
        dict(row) for row in partial_report.get("reregistration_requests", [])
        if str(row.get("after_frame")) in replaced
    ]
    merged["reregistration_requests"] = old_requests + new_requests
    updates = list(old_report.get("incremental_updates") or [])
    updates.append({
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "frames": sorted(replaced),
    })
    merged["incremental_updates"] = updates
    return merged


def _hardlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def apply_partial_masks(
    existing_mask_dir: Path,
    partial_mask_dir: Path,
    candidate_mask_dir: Path,
    replaced_frame_ids: Iterable[str],
) -> None:
    """Build a complete candidate tree without mutating the existing masks."""
    existing = Path(existing_mask_dir).expanduser().resolve()
    partial = Path(partial_mask_dir).expanduser().resolve()
    candidate = Path(candidate_mask_dir).expanduser().resolve()
    replaced = {str(frame_id) for frame_id in replaced_frame_ids}
    if candidate.exists():
        raise FileExistsError(f"候选Mask目录已存在：{candidate}")
    candidate.mkdir(parents=True)
    for source in sorted(existing.glob("*.png")):
        if source.stem not in replaced:
            _hardlink_or_copy(source, candidate / source.name)
    for source in sorted(partial.glob("*.png")):
        if source.stem in replaced:
            _hardlink_or_copy(source, candidate / source.name)


def _write_propagation_report(
    mask_dir: Path,
    report: dict[str, Any],
    *,
    logical_output_dir: Path | None = None,
) -> None:
    output = Path(mask_dir)
    logical_output = Path(logical_output_dir) if logical_output_dir is not None else output
    report["output_mask_dir"] = str(logical_output)
    parameters = report.get("parameters")
    if isinstance(parameters, dict):
        parameters["output_mask_dir"] = str(logical_output)
    (output / "sam2_propagation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    with (output / "sam2_propagation_report.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = ["id", "status", "mode", "score", "reasons", "key_mask"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in report.get("records", []):
            writer.writerow({
                **{key: record.get(key) for key in fields},
                "reasons": "|".join(record.get("reasons", [])),
            })


def _load_segments(scene: Path, frame_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    path = scene / "project_reports" / "segments.json"
    if not path.is_file():
        fallback = scene / "segments.json"
        path = fallback if fallback.is_file() else path
    if not path.is_file():
        return {"0": frame_ids}
    payload = json.loads(path.read_text(encoding="utf-8"))
    positions = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    result: dict[str, tuple[str, ...]] = {}
    for index, raw in enumerate(payload.get("segments") or []):
        start = str(raw.get("start_id", ""))
        end = str(raw.get("end_id", ""))
        if start not in positions or end not in positions:
            continue
        left, right = positions[start], positions[end]
        if right < left:
            left, right = right, left
        result[str(raw.get("segment_id", index))] = frame_ids[left:right + 1]
    return result or {"0": frame_ids}


def _requested_ranges(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.ranges_file:
        path = args.ranges_file.expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("action") not in {"rerun_ranges", "rerun_range"}:
            raise ValueError(f"动作文件不是局部传播请求：{path}")
        if safe_name(str(payload.get("instance_id", ""))) != safe_name(args.instance_id):
            raise ValueError("动作文件 instance_id 与命令不一致")
        rows = payload.get("ranges")
        if rows is None and payload.get("start_frame"):
            rows = [payload]
        if not isinstance(rows, list) or not rows:
            raise ValueError("动作文件没有 ranges")
        return [dict(row) for row in rows]
    return [{
        "segment_id": None,
        "start_frame": args.start_frame,
        "end_before_frame": args.end_before_frame,
    }]


def _validated_ranges(
    frame_ids: tuple[str, ...],
    active_frame_ids: tuple[str, ...],
    segments: dict[str, tuple[str, ...]],
    requests: Iterable[dict[str, Any]],
) -> tuple[FrameRange, ...]:
    positions = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    active = set(active_frame_ids)
    segment_items = list(segments.items())
    verified: list[tuple[str, FrameRange]] = []
    for request in requests:
        start = str(request.get("start_frame", ""))
        declared_segment = request.get("segment_id")
        containing = [(segment_id, members) for segment_id, members in segment_items if start in members]
        if not containing:
            raise ValueError(f"起始帧不属于自动分段：{start}")
        segment_id, members = containing[0]
        if declared_segment is not None and str(declared_segment) != segment_id:
            raise ValueError(
                f"动作文件分段不匹配：帧 {start} 实际属于 {segment_id}，收到 {declared_segment}"
            )
        end = request.get("end_before_frame")
        end = str(end) if end not in (None, "") else None
        if end is None:
            segment_last_position = positions[members[-1]]
            end = frame_ids[segment_last_position + 1] if segment_last_position + 1 < len(frame_ids) else None
        selected = select_frame_range(frame_ids, start, end)
        if any(frame_id not in members for frame_id in selected):
            raise ValueError(f"局部传播禁止跨自动分段：{start} -> {end}")
        if any(frame_id not in active for frame_id in selected):
            raise ValueError(f"局部传播超出实例有效帧范围：{start} -> {end}")
        verified.append((segment_id, (start, end)))

    normalized: list[FrameRange] = []
    for segment_id, members in segment_items:
        group = [frame_range for current_id, frame_range in verified if current_id == segment_id]
        if group:
            normalized.extend(normalize_frame_ranges(frame_ids, group))
    normalized.sort(key=lambda item: positions[item[0]])
    return tuple(normalized)


def _nonempty_key_mask(key_mask_dir: Path, frame_id: str) -> Path:
    path = key_mask_dir / f"{frame_id}.png"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0 or not bool((image > 0).any()):
        raise ValueError(f"起始帧缺少非空关键Mask：{path}")
    return path


def _sam2_command(
    *,
    instance: dict[str, Any],
    common: dict[str, Any],
    manifest_dir: Path,
    scene: Path,
    key_mask_dir: Path,
    output_mask_dir: Path,
    start: int,
    max_frames: int,
) -> list[str]:
    sam2_python = resolve_sam2_python(
        common.get("sam2_python"), manifest_dir=manifest_dir
    )
    sam2_model = resolve_path(
        common.get("sam2_model", WORKSPACE / "models" / "sam2.1_t.pt"),
        manifest_dir,
        required=True,
    )
    command = [
        str(sam2_python), str(WORKSPACE / "tools" / "propagate_masks_sam2.py"),
        "--scene", str(scene),
        "--key-mask-dir", str(key_mask_dir),
        "--output-mask-dir", str(output_mask_dir),
        "--model", str(sam2_model),
        "--device", str(common.get("sam2_device", 0)),
        "--imgsz", str(common.get("sam2_imgsz", 640)),
        "--start", str(start),
        "--max-frames", str(max_frames),
        "--memory-update-interval", str(
            instance.get("memory_update_interval", common.get("sam2_memory_update_interval", 5))
        ),
        "--max-consecutive-failures", str(common.get("max_consecutive_rejects", 3)),
        "--auto-reregister-after-failures", str(
            instance.get(
                "auto_reregister_after_failures",
                common.get("sam2_auto_reregister_after_failures", 1),
            )
        ),
        "--min-recovery-seed-iou", str(
            instance.get(
                "min_recovery_seed_iou", common.get("sam2_min_recovery_seed_iou", 0.35)
            )
        ),
        "--max-flow-shift-norm", str(
            instance.get("max_flow_shift_norm", common.get("sam2_max_flow_shift_norm", 0.35))
        ),
    ]
    command.append(
        "--auto-reregister"
        if instance.get("auto_reregister", common.get("sam2_auto_reregister", True))
        else "--no-auto-reregister"
    )
    quality = (common.get("mask_quality") or {}) | (instance.get("quality") or {})
    for key, flag in {
        "min_mask_area": "--min-mask-area",
        "min_area_ratio": "--min-area-ratio",
        "max_area_ratio": "--max-area-ratio",
        "min_depth_coverage": "--min-depth-coverage",
    }.items():
        if quality.get(key) is not None:
            add_optional(command, flag, quality[key])
    return command


def _load_manifest(path: Path, instance_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest必须是YAML对象")
    common = payload.get("project") or {}
    target = safe_name(instance_id)
    matches = [
        dict(instance) for instance in (payload.get("instances") or [])
        if safe_name(str(instance.get("instance_id", ""))) == target
    ]
    if len(matches) != 1:
        raise ValueError(f"Manifest中无法唯一找到实例：{instance_id}")
    instance = matches[0]
    if str(instance.get("tracker", "foundationpose")) != "sam2":
        raise ValueError(f"局部传播只支持SAM2实例：{instance_id}")
    return payload, common, instance


def _run(command: list[str], *, dry_run: bool, env: dict[str, str]) -> None:
    print("运行：", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest_dir = manifest_path.parent
    _manifest, common, instance = _load_manifest(manifest_path, args.instance_id)
    scene = resolve_path(common.get("scene"), manifest_dir, required=True)
    output = resolve_path(common.get("output"), manifest_dir, required=True)
    key_mask_dir = resolve_path(
        instance.get("key_mask_dir") or instance.get("registration_mask_dir"),
        manifest_dir,
        required=True,
    )
    if scene is None or output is None or key_mask_dir is None:
        raise ValueError("Manifest缺少局部传播所需路径")
    rgb_files = sorted((scene / "rgb").glob("*.png"))
    frame_ids = tuple(path.stem for path in rgb_files)
    if not frame_ids:
        raise ValueError(f"场次没有RGB帧：{scene}")
    instance_start = max(0, int(instance.get("start", 0)))
    instance_max = int(instance.get("max_frames", 0))
    instance_end = len(frame_ids) if instance_max <= 0 else min(len(frame_ids), instance_start + instance_max)
    active_frame_ids = frame_ids[instance_start:instance_end]
    segments = _load_segments(scene, frame_ids)
    requests = _requested_ranges(args)
    for request in requests:
        _nonempty_key_mask(key_mask_dir, str(request.get("start_frame", "")))
    ranges = _validated_ranges(frame_ids, active_frame_ids, segments, requests)

    stage = output / "_staging" / safe_name(scene.name) / safe_name(args.instance_id)
    existing = stage / "_sam2_masks"
    old_report_path = existing / "sam2_propagation_report.json"
    if not existing.is_dir() or not old_report_path.is_file():
        raise FileNotFoundError(f"缺少可局部更新的完整SAM2结果：{existing}")
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    validate_complete_report(old_report, active_frame_ids)

    print(
        "局部传播批次：" + ", ".join(
            format_frame_range(frame_ids, start, end) for start, end in ranges
        ),
        flush=True,
    )
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.dry_run:
        for index, (start, end) in enumerate(ranges):
            selected = select_frame_range(frame_ids, start, end)
            command = _sam2_command(
                instance=instance,
                common=common,
                manifest_dir=manifest_dir,
                scene=scene,
                key_mask_dir=key_mask_dir,
                output_mask_dir=stage / f"_dry_run_partial_{index:02d}",
                start=frame_ids.index(start),
                max_frames=len(selected),
            )
            _run(command, dry_run=True, env=env)
        print("dry-run：未修改任何Mask、报告或YOLO数据。")
        return 0

    work = stage / f".incremental-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    partial_combined = work / "partial_combined"
    candidate = work / "candidate_masks"
    partial_combined.mkdir(parents=True)
    merged_report = old_report
    replaced_frames: list[str] = []
    try:
        for index, (start, end) in enumerate(ranges):
            selected = select_frame_range(frame_ids, start, end)
            partial_dir = work / f"partial_{index:02d}"
            command = _sam2_command(
                instance=instance,
                common=common,
                manifest_dir=manifest_dir,
                scene=scene,
                key_mask_dir=key_mask_dir,
                output_mask_dir=partial_dir,
                start=frame_ids.index(start),
                max_frames=len(selected),
            )
            _run(command, dry_run=False, env=env)
            partial_report_path = partial_dir / "sam2_propagation_report.json"
            if not partial_report_path.is_file():
                raise RuntimeError(f"局部SAM2未生成报告：{partial_report_path}")
            partial_report = json.loads(partial_report_path.read_text(encoding="utf-8"))
            merged_report = merge_propagation_reports(merged_report, partial_report, selected)
            replaced_frames.extend(selected)
            for mask in partial_dir.glob("*.png"):
                _hardlink_or_copy(mask, partial_combined / mask.name)

        unique_replaced = tuple(dict.fromkeys(replaced_frames))
        apply_partial_masks(existing, partial_combined, candidate, unique_replaced)
        _write_propagation_report(
            candidate,
            merged_report,
            logical_output_dir=existing,
        )

        backup = work / "previous_masks"
        os.replace(existing, backup)
        try:
            os.replace(candidate, existing)
        except BaseException:
            os.replace(backup, existing)
            raise

        aggregate_command = [
            sys.executable,
            str(WORKSPACE / "tools" / "annotate_multinstance_project.py"),
            "--manifest", str(manifest_path),
            "--skip-tracking",
        ]
        try:
            _run(aggregate_command, dry_run=False, env=env)
        except BaseException as aggregate_error:
            failed_candidate = work / "failed_candidate"
            if existing.exists():
                os.replace(existing, failed_candidate)
            os.replace(backup, existing)
            try:
                _run(aggregate_command, dry_run=False, env=env)
            except BaseException as restore_error:
                raise RuntimeError(
                    f"局部聚合失败且恢复旧标签也失败：{aggregate_error}; restore={restore_error}"
                ) from aggregate_error
            raise RuntimeError(f"局部聚合失败，已恢复旧Mask和标签：{aggregate_error}") from aggregate_error
        shutil.rmtree(backup)
    except BaseException:
        # SAM2或候选构建失败发生在正式目录替换前；旧Mask/标签保持不变。
        raise
    finally:
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)

    print(
        f"局部传播完成：更新 {len(set(replaced_frames))} 帧；范围外SAM2 Mask保持不变；"
        "质量检查和YOLO聚合已刷新。",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
