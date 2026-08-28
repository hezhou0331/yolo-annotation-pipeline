#!/usr/bin/env python3
"""Analyze an aligned RGB-D sequence and propose safe annotation segments.

This tool does not pretend that FoundationPose can recover after loss without a
mask.  It detects hard boundaries and writes the exact key-frame masks that are
required before automatic registration/tracking can continue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from atec_pipeline.path_compat import infer_project_root, portable_path, resolve_compatible_path


def parse_args():
    p = argparse.ArgumentParser(description="RGB-D自动分段与关键mask需求分析")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene", type=Path)
    source.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--registration-root", type=Path, default=None, help="其下每个子目录代表一个instance_id")
    p.add_argument("--instances", nargs="*", default=[], help="需要首帧mask的实例ID")
    p.add_argument("--max-segment-frames", type=int, default=300)
    p.add_argument("--scene-cut-threshold", type=float, default=0.20, help="缩略图平均绝对差，0到1")
    p.add_argument("--timestamp-gap-ms", type=float, default=0.0, help="0表示自动用中位间隔的3倍")
    p.add_argument("--min-depth-valid-ratio", type=float, default=0.20)
    p.add_argument("--require-ready", action="store_true", help="任一分段缺关键mask时返回非零")
    return p.parse_args()


def thumbnail(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def load_metadata(scene: Path):
    path = scene / "metadata.json"
    if not path.exists():
        return {}, None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {str(x.get("id")): x for x in data.get("frames", [])}
    return rows, data


def mask_exists(root: Path | None, instance: str, stem: str, key_dirs: dict[str, Path] | None = None):
    candidates = []
    if key_dirs and instance in key_dirs:
        candidates.append(key_dirs[instance] / f"{stem}.png")
    if root is not None:
        candidates.extend((root / instance / f"{stem}.png", root / f"{stem}.png"))
    for candidate in candidates:
        if candidate.exists():
            mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if mask is not None and np.any(mask > 0):
                return candidate.resolve()
    return None


def main():
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else None
    key_dirs: dict[str, Path] = {}
    if manifest_path:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        base = manifest_path.parent
        project_root = infer_project_root(manifest_path, repository_root=WORKSPACE)
        scene = resolve_compatible_path(
            manifest["project"]["scene"],
            base=base,
            repository_root=WORKSPACE,
            project_root=project_root,
        )
        if not args.instances:
            args.instances = [str(x["instance_id"]) for x in manifest.get("instances", [])]
        for instance in manifest.get("instances", []):
            value = instance.get("key_mask_dir") or instance.get("registration_mask_dir")
            if value:
                key_dirs[str(instance["instance_id"])] = resolve_compatible_path(
                    value,
                    base=base,
                    repository_root=WORKSPACE,
                    project_root=project_root,
                )
    else:
        scene = args.scene.expanduser().resolve()
        project_root = infer_project_root(scene, repository_root=WORKSPACE)
    output = resolve_compatible_path(
        args.output or scene / "segments.json",
        base=Path.cwd(),
        repository_root=WORKSPACE,
        project_root=project_root,
    )
    root = (
        resolve_compatible_path(
            args.registration_root,
            base=Path.cwd(),
            repository_root=WORKSPACE,
            project_root=project_root,
        )
        if args.registration_root
        else None
    )
    rgb_files = sorted((scene / "rgb").glob("*.png"))
    if not rgb_files:
        raise SystemExit(f"没有RGB帧：{scene / 'rgb'}")
    metadata_rows, metadata = load_metadata(scene)

    timestamps = []
    for p in rgb_files:
        row = metadata_rows.get(p.stem, {})
        ts = row.get("color_timestamp_ms")
        if ts is not None:
            timestamps.append(float(ts))
    positive_gaps = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    median_gap = float(np.median(positive_gaps)) if positive_gaps else None
    gap_limit = args.timestamp_gap_ms if args.timestamp_gap_ms > 0 else ((median_gap * 3.0) if median_gap else None)

    frame_info = []
    boundaries = {0: ["sequence_start"]}
    previous_thumb = None
    previous_ts = None
    last_boundary = 0
    for idx, rgb_path in enumerate(rgb_files):
        stem = rgb_path.stem
        reasons = []
        depth_path = scene / "depth" / f"{stem}.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path.exists() else None
        if depth is None:
            depth_valid_ratio = 0.0
            reasons.append("depth_missing")
        else:
            depth_valid_ratio = float(np.mean(depth > 0))
            if depth_valid_ratio < args.min_depth_valid_ratio:
                reasons.append("depth_valid_ratio_low")

        thumb = thumbnail(rgb_path)
        diff = None
        if thumb is None:
            reasons.append("rgb_unreadable")
        elif previous_thumb is not None:
            diff = float(np.mean(np.abs(thumb - previous_thumb)))
            if diff >= args.scene_cut_threshold:
                boundaries.setdefault(idx, []).append("visual_scene_cut")
        if thumb is not None:
            previous_thumb = thumb

        row = metadata_rows.get(stem, {})
        ts = row.get("color_timestamp_ms")
        ts = float(ts) if ts is not None else None
        if ts is not None and previous_ts is not None and gap_limit is not None and ts - previous_ts > gap_limit:
            boundaries.setdefault(idx, []).append("timestamp_gap")
        if ts is not None:
            previous_ts = ts

        if idx - last_boundary >= args.max_segment_frames:
            boundaries.setdefault(idx, []).append("max_segment_length")
        if idx in boundaries:
            last_boundary = idx
        frame_info.append({
            "index": idx, "id": stem, "rgb": rgb_path, "depth": depth_path,
            "timestamp_ms": ts, "visual_diff": diff,
            "depth_valid_ratio": depth_valid_ratio, "warnings": reasons,
        })

    starts = sorted(boundaries)
    segments = []
    for seg_idx, start in enumerate(starts):
        end = (starts[seg_idx + 1] - 1) if seg_idx + 1 < len(starts) else len(rgb_files) - 1
        start_stem = rgb_files[start].stem
        key_masks = {instance: mask_exists(root, instance, start_stem, key_dirs) for instance in args.instances}
        required_key_mask_paths = {}
        for instance in args.instances:
            target_dir = key_dirs.get(instance) if key_dirs else None
            if target_dir is None and root is not None:
                target_dir = root / instance
            required_key_mask_paths[instance] = (target_dir / f"{start_stem}.png").resolve() if target_dir else None
        missing = [instance for instance, path in key_masks.items() if path is None]
        bad_frames = [f["id"] for f in frame_info[start:end + 1] if f["warnings"]]
        segments.append({
            "segment_id": seg_idx, "start_index": start, "end_index": end,
            "start_id": start_stem, "end_id": rgb_files[end].stem,
            "frame_count": end - start + 1, "boundary_reasons": boundaries[start],
            "key_masks": key_masks, "required_key_mask_paths": required_key_mask_paths, "missing_key_masks": missing,
            "ready_for_automatic_tracking": not missing,
            "frames_with_capture_warnings": bad_frames,
        })

    report_base = output.parent

    def stored(path: Path | None) -> str | None:
        if path is None:
            return None
        return portable_path(
            path,
            relative_to=report_base,
            repository_root=WORKSPACE,
            project_root=project_root,
        )

    for frame in frame_info:
        frame["rgb"] = stored(frame["rgb"])
        frame["depth"] = stored(frame["depth"])
    for segment in segments:
        segment["key_masks"] = {instance: stored(path) for instance, path in segment["key_masks"].items()}
        segment["required_key_mask_paths"] = {
            instance: stored(path) for instance, path in segment["required_key_mask_paths"].items()
        }

    report = {
        "format_version": 2, "scene": stored(scene), "manifest": stored(manifest_path), "frame_count": len(rgb_files),
        "instances": args.instances, "registration_root": stored(root),
        "parameters": {
            "max_segment_frames": args.max_segment_frames,
            "scene_cut_threshold": args.scene_cut_threshold,
            "timestamp_gap_ms": gap_limit, "timestamp_gap_source": "manual" if args.timestamp_gap_ms > 0 else "auto_3x_median",
            "median_timestamp_gap_ms": median_gap,
            "min_depth_valid_ratio": args.min_depth_valid_ratio,
        },
        "segments": segments, "frames": frame_info,
        "truth_boundary": (
            "自动分段只能提出候选边界；没有人工/可靠关键帧mask时，"
            "FoundationPose或SAM2都不能被视为已可靠自动重注册。"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ready = sum(s["ready_for_automatic_tracking"] for s in segments)
    print(f"分段完成：{len(rgb_files)}帧 -> {len(segments)}段；已具备全部关键mask：{ready}/{len(segments)}")
    print(f"报告：{output}")
    missing_total = sum(len(s["missing_key_masks"]) for s in segments)
    if args.require_ready and missing_total:
        print(f"尚缺{missing_total}个分段关键mask；已停止自动标注，避免生成不完整标签。")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
