#!/usr/bin/env python3
"""Convert a propagated per-frame mask sequence into quality-gated YOLO labels.

This is the non-rigid-object branch of the hybrid pipeline.  A video mask
tracker can write one PNG per RGB frame; this script applies the same
accepted/review/rejected policy used by the FoundationPose branch.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from annotate_foundationpose_yolo import detection_line, find_depth_scale, mask_bbox, place_image, segmentation_line, write_label
from quality_metrics import QualityThresholds, classify_quality, compute_quality_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mask序列质量过滤并导出YOLO")
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--class-id", type=int, required=True)
    p.add_argument("--class-name", required=True)
    p.add_argument("--instance-id", required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--task", choices=("detection", "segmentation"), default="segmentation")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--name-prefix", default=None)
    p.add_argument("--depth-to-metre", type=float, default=None)
    p.add_argument("--image-mode", choices=("hardlink", "copy", "none"), default="none")
    p.add_argument("--include-review", action="store_true")
    p.add_argument("--review-overrides", type=Path, default=None)
    p.add_argument("--polygon-epsilon", type=float, default=0.002)
    p.add_argument("--min-mask-area", type=int, default=80)
    p.add_argument("--min-depth-coverage", type=float, default=0.25)
    p.add_argument("--min-area-ratio", type=float, default=0.35)
    p.add_argument("--max-area-ratio", type=float, default=2.8)
    p.add_argument("--max-center-shift-norm", type=float, default=0.40)
    p.add_argument("--min-dominant-component-ratio", type=float, default=0.70)
    return p.parse_args()


def load_binary(path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3:
        raw = raw.max(axis=2)
    if raw.shape != shape:
        raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return raw > 0


@dataclass(frozen=True)
class ManualReviewDecision:
    status: str
    auto_status: str
    manual_status: str | None
    manual_reason: str | None
    reject_reasons: tuple[str, ...]
    review_reasons: tuple[str, ...]


def load_manual_review(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    result = {}
    for frame_id, row in (payload.get("frames") or {}).items():
        status = str((row or {}).get("status", ""))
        if status in {"accepted", "review", "rejected"}:
            result[str(frame_id)] = dict(row)
    return result


def apply_manual_review(
    frame_id: str,
    auto_status: str,
    reject_reasons: list[str],
    review_reasons: list[str],
    line: str | None,
    overrides: dict[str, dict],
) -> ManualReviewDecision:
    override = overrides.get(frame_id)
    if not override:
        return ManualReviewDecision(
            auto_status, auto_status, None, None,
            tuple(reject_reasons), tuple(review_reasons),
        )
    manual_status = str(override["status"])
    manual_reason = str(override.get("reason") or f"manual_{manual_status}")
    if manual_status == "accepted":
        if line is None:
            return ManualReviewDecision(
                "rejected", auto_status, manual_status, manual_reason,
                ("manual_accept_blocked_invalid_mask",), (),
            )
        return ManualReviewDecision("accepted", auto_status, manual_status, manual_reason, (), ())
    if manual_status == "review":
        return ManualReviewDecision("review", auto_status, manual_status, manual_reason, (), (manual_reason,))
    return ManualReviewDecision("rejected", auto_status, manual_status, manual_reason, (manual_reason,), ())


def main() -> int:
    args = parse_args()
    scene = args.scene.expanduser().resolve()
    mask_dir = args.mask_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    rgb_files = sorted((scene / "rgb").glob("*.png"))
    if not rgb_files:
        raise SystemExit(f"没有RGB帧：{scene / 'rgb'}")
    end = len(rgb_files) if args.max_frames <= 0 else min(len(rgb_files), args.start + args.max_frames)
    selected = rgb_files[args.start:end]
    prefix = f"{scene.name}_" if args.name_prefix is None else args.name_prefix
    depth_scale = find_depth_scale(scene, args.depth_to_metre)
    instance_dir = f"class_{args.class_id:03d}_{args.instance_id}"
    labels_dir = output / "labels" / args.split
    images_dir = output / "images" / args.split
    masks_out = output / "rendered_masks" / args.split / instance_dir
    vis_root = output / "visualizations" / args.split / instance_dir
    report_dir = output / "quality_reports" / args.split / instance_dir
    for d in (labels_dir, images_dir, masks_out, report_dir):
        d.mkdir(parents=True, exist_ok=True)
    for status in ("accepted", "review", "rejected"):
        (vis_root / status).mkdir(parents=True, exist_ok=True)

    thresholds = QualityThresholds(
        min_mask_area=args.min_mask_area,
        min_depth_coverage=args.min_depth_coverage,
        max_depth_median_abs_m=1e9,
        max_depth_rmse_m=1e9,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        max_center_shift_norm=args.max_center_shift_norm,
        max_translation_jump_m=1e9,
        max_rotation_jump_deg=1e9,
        min_dominant_component_ratio=args.min_dominant_component_ratio,
    )
    manual_overrides = load_manual_review(args.review_overrides)
    previous_good_mask = None
    records = []
    for index, rgb_path in enumerate(selected):
        stem = rgb_path.stem
        output_id = f"{prefix}{stem}"
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(scene / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or depth_raw is None:
            raise RuntimeError(f"RGB或深度缺失：{stem}")
        depth_m = depth_raw.astype(np.float32) * depth_scale
        mask_path = mask_dir / f"{stem}.png"
        mask = load_binary(mask_path, bgr.shape[:2])

        reject_reasons = []
        review_reasons = []
        metrics = {}
        line = None
        if mask is None:
            status = "rejected"
            reject_reasons = ["mask_file_missing"]
            mask = np.zeros(bgr.shape[:2], dtype=bool)
        else:
            # Reuse observed depth as rendered depth: only coverage and mask
            # continuity are meaningful for a model-free mask sequence.
            metrics = compute_quality_metrics(
                mask, depth_m, depth_m, np.eye(4, dtype=np.float64),
                previous_mask=previous_good_mask,
                previous_pose=None,
                border_margin_px=thresholds.border_margin_px,
            )
            status, reject_reasons, review_reasons = classify_quality(metrics, thresholds, registration_frame=index == 0)
            bbox = mask_bbox(mask) if int(mask.sum()) >= thresholds.min_mask_area else None
            if bbox is not None:
                line = detection_line(args.class_id, bbox, bgr.shape[1], bgr.shape[0]) if args.task == "detection" else segmentation_line(args.class_id, mask, args.polygon_epsilon)

        auto_status = status
        auto_reject_reasons = list(reject_reasons)
        auto_review_reasons = list(review_reasons)
        decision = apply_manual_review(stem, auto_status, reject_reasons, review_reasons, line, manual_overrides)
        status = decision.status
        reject_reasons = list(decision.reject_reasons)
        review_reasons = list(decision.review_reasons)
        eligible = line is not None and (status == "accepted" or (status == "review" and args.include_review))
        label_path = labels_dir / f"{output_id}.txt"
        if eligible:
            write_label(label_path, line, False)
            place_image(rgb_path, images_dir / f"{output_id}{rgb_path.suffix.lower()}", args.image_mode)
        elif label_path.exists():
            label_path.unlink()
        if auto_status != "rejected":
            previous_good_mask = mask.copy()

        cv2.imwrite(str(masks_out / f"{output_id}.png"), mask.astype(np.uint8) * 255)
        vis = bgr.copy()
        color = {"accepted": (0, 180, 0), "review": (0, 180, 255), "rejected": (0, 0, 255)}[status]
        overlay = vis.copy(); overlay[mask] = color
        vis = cv2.addWeighted(vis, 0.72, overlay, 0.28, 0)
        cv2.putText(vis, f"{stem} mask_sequence {status}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for old_status in ("accepted", "review", "rejected"):
            old_path = vis_root / old_status / f"{output_id}.jpg"
            if old_status != status and old_path.exists():
                old_path.unlink()
        cv2.imwrite(str(vis_root / status / f"{output_id}.jpg"), vis)
        records.append({
            "id": stem,
            "output_id": output_id,
            "status": status,
            "auto_status": auto_status,
            "manual_status": decision.manual_status,
            "manual_reason": decision.manual_reason,
            "auto_reject_reasons": auto_reject_reasons,
            "auto_review_reasons": auto_review_reasons,
            "mode": "mask_sequence",
            "segment_id": 0,
            "has_label": bool(eligible),
            "registration_mask": str(mask_path) if index == 0 else None,
            "reject_reasons": reject_reasons,
            "review_reasons": review_reasons,
            "metrics": metrics,
        })

    counts = {s: sum(r["status"] == s for r in records) for s in ("accepted", "review", "rejected")}
    report = {
        "scene": str(scene), "mask_dir": str(mask_dir), "class_id": args.class_id,
        "class_name": args.class_name, "instance_id": args.instance_id,
        "split": args.split, "tracker": "mask_sequence", "thresholds": thresholds.to_dict(),
        "frames_processed": len(records), "frames_labeled": sum(r["has_label"] for r in records),
        "status_counts": counts, "records": records,
    }
    (report_dir / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["id", "output_id", "status", "auto_status", "manual_status", "manual_reason", "has_label", "reject_reasons", "review_reasons"]
    with (report_dir / "quality_report.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in records:
            writer.writerow({"id": r["id"], "output_id": r["output_id"], "status": r["status"], "auto_status": r["auto_status"], "manual_status": r["manual_status"], "manual_reason": r["manual_reason"], "has_label": r["has_label"], "reject_reasons": "|".join(r["reject_reasons"]), "review_reasons": "|".join(r["review_reasons"])})
    print(f"完成：accepted={counts['accepted']} review={counts['review']} rejected={counts['rejected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
