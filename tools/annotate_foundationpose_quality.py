#!/usr/bin/env python3
"""FoundationPose single-instance annotation with automatic quality gates.

Unlike the legacy exporter, frames marked review/rejected are not inserted into
the formal YOLO dataset.  Any non-empty mask found in the registration-mask
directory can be used as a keyframe for automatic re-registration.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from annotate_foundationpose_yolo import (
    DEFAULT_FP_DIR,
    detection_line,
    find_depth_scale,
    load_frame,
    load_mesh,
    mask_bbox,
    place_image,
    render_mask,
    resize_for_inference,
    segmentation_line,
    write_label,
)
from quality_metrics import QualityThresholds, classify_quality, compute_quality_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FoundationPose质量门控标注")
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--foundationpose-dir", type=Path, default=DEFAULT_FP_DIR)
    p.add_argument("--class-id", type=int, required=True)
    p.add_argument("--class-name", required=True)
    p.add_argument("--instance-id", default=None, help="同类多实例的唯一名称")
    p.add_argument("--first-mask", type=Path, default=None)
    p.add_argument("--registration-mask-dir", type=Path, default=None)
    p.add_argument("--reregister-on-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--name-prefix", default=None)
    p.add_argument("--task", choices=("detection", "segmentation"), default="segmentation")
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--mesh-unit", choices=("m", "cm", "mm"), default="m")
    p.add_argument("--depth-to-metre", type=float, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--est-refine-iter", type=int, default=3)
    p.add_argument("--track-refine-iter", type=int, default=2)
    p.add_argument("--inference-max-side", type=int, default=640)
    p.add_argument("--mask-mode", choices=("visible", "full"), default="visible")
    p.add_argument("--occlusion-tolerance", type=float, default=0.03)
    p.add_argument("--polygon-epsilon", type=float, default=0.002)
    p.add_argument("--image-mode", choices=("hardlink", "copy", "none"), default="hardlink")
    p.add_argument("--append-labels", action="store_true")
    p.add_argument("--include-review", action="store_true")
    p.add_argument("--no-visualization", action="store_true")
    p.add_argument("--max-consecutive-rejects", type=int, default=3)
    p.add_argument("--stop-after-lost", action="store_true")
    p.add_argument("--min-registration-iou", type=float, default=0.45)

    p.add_argument("--min-mask-area", type=int, default=80)
    p.add_argument("--min-area-fraction", type=float, default=0.00015)
    p.add_argument("--max-area-fraction", type=float, default=0.65)
    p.add_argument("--min-depth-coverage", type=float, default=0.45)
    p.add_argument("--max-depth-median-abs-m", type=float, default=0.05)
    p.add_argument("--max-depth-rmse-m", type=float, default=0.08)
    p.add_argument("--min-area-ratio", type=float, default=0.45)
    p.add_argument("--max-area-ratio", type=float, default=2.20)
    p.add_argument("--max-center-shift-norm", type=float, default=0.35)
    p.add_argument("--max-translation-jump-m", type=float, default=0.20)
    p.add_argument("--max-rotation-jump-deg", type=float, default=55.0)
    p.add_argument("--min-dominant-component-ratio", type=float, default=0.75)
    p.add_argument("--border-margin-px", type=int, default=2)
    return p.parse_args()


def load_mask(path: Path, expected_shape: tuple[int, int] | None = None) -> np.ndarray | None:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3:
        raw = raw.max(axis=2)
    mask = raw > 0
    if expected_shape is not None and mask.shape != expected_shape:
        mask = cv2.resize(mask.astype(np.uint8), (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def safe_name(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in value.strip())
    return cleaned or "instance"


def make_thresholds(args: argparse.Namespace) -> QualityThresholds:
    return QualityThresholds(
        min_mask_area=args.min_mask_area,
        min_area_fraction=args.min_area_fraction,
        max_area_fraction=args.max_area_fraction,
        min_depth_coverage=args.min_depth_coverage,
        max_depth_median_abs_m=args.max_depth_median_abs_m,
        max_depth_rmse_m=args.max_depth_rmse_m,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        max_center_shift_norm=args.max_center_shift_norm,
        max_translation_jump_m=args.max_translation_jump_m,
        max_rotation_jump_deg=args.max_rotation_jump_deg,
        min_dominant_component_ratio=args.min_dominant_component_ratio,
        border_margin_px=args.border_margin_px,
    )


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted({k for r in records for k in r.get("metrics", {}).keys()})
    fields = [
        "id", "output_id", "status", "mode", "has_label", "seconds",
        "reject_reasons", "review_reasons", "registration_mask", "segment_id",
    ] + metric_keys
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {k: record.get(k) for k in fields}
            row["reject_reasons"] = "|".join(record.get("reject_reasons", []))
            row["review_reasons"] = "|".join(record.get("review_reasons", []))
            row.update(record.get("metrics", {}))
            for key, value in list(row.items()):
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    scene = args.scene.expanduser().resolve()
    mesh_path = args.mesh.expanduser().resolve()
    output = args.output.expanduser().resolve()
    fp_dir = args.foundationpose_dir.expanduser().resolve()
    instance_id = safe_name(args.instance_id or f"{args.class_name}_01")
    registration_dir = (args.registration_mask_dir or (scene / "masks")).expanduser().resolve()
    thresholds = make_thresholds(args)

    if not fp_dir.exists():
        raise SystemExit(f"FoundationPose目录不存在：{fp_dir}")
    sys.path.insert(0, str(fp_dir))
    os.chdir(fp_dir)

    import torch
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    from Utils import make_mesh_tensors, set_logging_format, set_seed

    set_logging_format()
    set_seed(0)

    rgb_files = sorted((scene / "rgb").glob("*.png"))
    if not rgb_files:
        raise SystemExit(f"没有找到RGB图片：{scene / 'rgb'}")
    if args.start < 0 or args.start >= len(rgb_files):
        raise SystemExit(f"--start超出范围：共有{len(rgb_files)}帧")
    end = len(rgb_files) if args.max_frames <= 0 else min(len(rgb_files), args.start + args.max_frames)
    selected = rgb_files[args.start:end]

    k_path = scene / "cam_K.txt"
    if not k_path.exists():
        k_path = scene / "K.txt"
    if not k_path.exists():
        raise SystemExit("缺少cam_K.txt或K.txt")
    K = np.loadtxt(k_path).reshape(3, 3).astype(np.float64)
    depth_scale = find_depth_scale(scene, args.depth_to_metre)
    mesh = load_mesh(mesh_path, args.mesh_unit)

    first_stem = selected[0].stem
    first_mask_path = args.first_mask.expanduser().resolve() if args.first_mask else registration_dir / f"{first_stem}.png"
    if load_mask(first_mask_path) is None:
        raise SystemExit(f"缺少首帧mask：{first_mask_path}")

    output.mkdir(parents=True, exist_ok=True)
    name_prefix = f"{scene.name}_" if args.name_prefix is None else args.name_prefix
    instance_dir = f"class_{args.class_id:03d}_{instance_id}"
    poses_dir = output / "poses" / args.split / instance_dir
    masks_dir = output / "rendered_masks" / args.split / instance_dir
    vis_root = output / "visualizations" / args.split / instance_dir
    images_dir = output / "images" / args.split
    labels_dir = output / "labels" / args.split
    for directory in (poses_dir, masks_dir, images_dir, labels_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if not args.no_visualization:
        for status in ("accepted", "review", "rejected"):
            (vis_root / status).mkdir(parents=True, exist_ok=True)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(output / "foundationpose_debug" / args.split / instance_dir),
        debug=0,
        glctx=glctx,
    )
    mesh_tensors = make_mesh_tensors(mesh)

    records: list[dict] = []
    requests: list[dict] = []
    previous_good_mask: np.ndarray | None = None
    previous_good_pose: np.ndarray | None = None
    consecutive_rejects = 0
    segment_id = 0
    started = time.time()

    for local_i, rgb_path in enumerate(selected):
        stem = rgb_path.stem
        output_stem = f"{name_prefix}{stem}"
        depth_path = scene / "depth" / f"{stem}.png"
        bgr, rgb, depth_m = load_frame(rgb_path, depth_path, depth_scale)

        candidate_mask_path = first_mask_path if local_i == 0 else registration_dir / f"{stem}.png"
        manual_mask = None
        if local_i == 0 or (args.reregister_on_mask and candidate_mask_path.exists()):
            manual_mask = load_mask(candidate_mask_path, depth_m.shape)
            if manual_mask is not None and int(manual_mask.sum()) < 4:
                manual_mask = None
        should_register = local_i == 0 or manual_mask is not None
        if local_i == 0 and manual_mask is None:
            raise RuntimeError("首帧mask有效像素过少")
        if should_register and local_i > 0:
            segment_id += 1
            previous_good_mask = None
            previous_good_pose = None
            consecutive_rejects = 0

        rgb_inf, depth_inf, mask_inf, K_inf, scale = resize_for_inference(
            rgb, depth_m, manual_mask, K, args.inference_max_side
        )
        frame_started = time.time()
        if should_register:
            pose = estimator.register(
                K=K_inf,
                rgb=rgb_inf,
                depth=depth_inf,
                ob_mask=mask_inf.astype(bool),
                iteration=args.est_refine_iter,
            )
            mode = "register"
        else:
            pose = estimator.track_one(
                rgb=rgb_inf,
                depth=depth_inf,
                K=K_inf,
                iteration=args.track_refine_iter,
            )
            mode = "track"

        pose = np.asarray(pose).reshape(4, 4)
        np.savetxt(poses_dir / f"{output_stem}.txt", pose, fmt="%.10f")
        mask, rendered_depth = render_mask(
            mesh_tensors, pose, K, bgr.shape[0], bgr.shape[1], glctx,
            depth_m, args.mask_mode, args.occlusion_tolerance,
        )
        cv2.imwrite(str(masks_dir / f"{output_stem}.png"), mask.astype(np.uint8) * 255, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        metrics = compute_quality_metrics(
            mask, rendered_depth, depth_m, pose,
            previous_mask=previous_good_mask,
            previous_pose=previous_good_pose,
            border_margin_px=thresholds.border_margin_px,
        )
        if manual_mask is not None:
            union = np.logical_or(mask, manual_mask).sum()
            intersection = np.logical_and(mask, manual_mask).sum()
            metrics["registration_mask_iou"] = float(intersection / max(1, union))

        status, reject_reasons, review_reasons = classify_quality(metrics, thresholds, registration_frame=should_register)
        reg_iou = metrics.get("registration_mask_iou")
        if reg_iou is not None and reg_iou < args.min_registration_iou:
            status = "rejected"
            reject_reasons.append("registration_mask_iou_too_low")

        bbox = mask_bbox(mask) if int(mask.sum()) >= thresholds.min_mask_area else None
        line = None
        if bbox is not None:
            if args.task == "detection":
                line = detection_line(args.class_id, bbox, bgr.shape[1], bgr.shape[0])
            else:
                line = segmentation_line(args.class_id, mask, args.polygon_epsilon)
        eligible = line is not None and (status == "accepted" or (status == "review" and args.include_review))

        label_path = labels_dir / f"{output_stem}.txt"
        if eligible:
            write_label(label_path, line, args.append_labels)
            place_image(rgb_path, images_dir / f"{output_stem}{rgb_path.suffix.lower()}", args.image_mode)
        elif not args.append_labels and label_path.exists():
            label_path.unlink()

        if status == "rejected":
            consecutive_rejects += 1
        else:
            consecutive_rejects = 0
            previous_good_mask = mask.copy()
            previous_good_pose = pose.copy()

        if consecutive_rejects == args.max_consecutive_rejects:
            requests.append({
                "frame": stem,
                "output_id": output_stem,
                "instance_id": instance_id,
                "class_id": args.class_id,
                "reason": "consecutive_rejected_frames",
                "required_action": f"请在{registration_dir / (stem + '.png')}补画该实例mask后重跑；脚本会在此帧自动register。",
            })

        if not args.no_visualization:
            vis = bgr.copy()
            overlay = vis.copy()
            color = {"accepted": (0, 180, 0), "review": (0, 180, 255), "rejected": (0, 0, 255)}[status]
            overlay[mask] = color
            vis = cv2.addWeighted(vis, 0.72, overlay, 0.28, 0)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), color, 2)
            reason_text = ",".join((reject_reasons or review_reasons)[:2])
            cv2.putText(vis, f"{stem} {mode} {status}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            if reason_text:
                cv2.putText(vis, reason_text, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.imwrite(str(vis_root / status / f"{output_stem}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

        elapsed = time.time() - frame_started
        record = {
            "id": stem,
            "output_id": output_stem,
            "status": status,
            "mode": mode,
            "segment_id": segment_id,
            "seconds": elapsed,
            "inference_scale": scale,
            "has_label": bool(eligible),
            "registration_mask": str(candidate_mask_path) if manual_mask is not None else None,
            "reject_reasons": reject_reasons,
            "review_reasons": review_reasons,
            "metrics": metrics,
        }
        records.append(record)
        print(f"[{local_i + 1}/{len(selected)}] {stem}: {mode} {status}, label={'yes' if eligible else 'no'}, {elapsed:.2f}s")

        torch.cuda.empty_cache()
        gc.collect()
        if args.stop_after_lost and consecutive_rejects >= args.max_consecutive_rejects:
            break

    classes_path = output / "classes.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8")) if classes_path.exists() else {}
    classes[str(args.class_id)] = args.class_name
    classes = dict(sorted(classes.items(), key=lambda item: int(item[0])))
    classes_path.write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")

    val_has_images = any((output / "images" / "val").glob("*"))
    test_has_images = any((output / "images" / "test").glob("*"))
    val_rel = "images/val" if val_has_images else "images/train"
    test_rel = "images/test" if test_has_images else val_rel
    names_yaml = "".join(f"  {class_id}: {name}\n" for class_id, name in classes.items())
    (output / "dataset.yaml").write_text(
        f"path: {output}\ntrain: images/train\nval: {val_rel}\ntest: {test_rel}\nnames:\n{names_yaml}",
        encoding="utf-8",
    )

    counts = {status: sum(r["status"] == status for r in records) for status in ("accepted", "review", "rejected")}
    report = {
        "scene": str(scene),
        "mesh": str(mesh_path),
        "mesh_unit": args.mesh_unit,
        "class_id": args.class_id,
        "class_name": args.class_name,
        "instance_id": instance_id,
        "split": args.split,
        "depth_to_metre": depth_scale,
        "thresholds": thresholds.to_dict(),
        "min_registration_iou": args.min_registration_iou,
        "frames_processed": len(records),
        "frames_labeled": sum(bool(r["has_label"]) for r in records),
        "status_counts": counts,
        "segments": (max((r["segment_id"] for r in records), default=-1) + 1),
        "reregistration_requests": requests,
        "total_seconds": time.time() - started,
        "records": records,
    }
    report_dir = output / "quality_reports" / args.split / instance_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(report_dir / "quality_report.csv", records)
    (report_dir / "reregistration_requests.json").write_text(json.dumps(requests, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：accepted={counts['accepted']} review={counts['review']} rejected={counts['rejected']}，标签={report['frames_labeled']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
