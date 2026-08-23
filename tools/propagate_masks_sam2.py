#!/usr/bin/env python3
"""Propagate one instance mask through RGB frames with Ultralytics SAM2.

Any non-empty mask in --key-mask-dir whose filename matches an RGB frame is a
trusted keyframe. At each such frame the SAM2 memory is rebuilt, providing a
deterministic re-registration point. If ordinary tracking fails, an optional
optical-flow seed is transferred from the last accepted frame and used to
rebuild SAM2 automatically. Failed predictions are omitted so the downstream
quality gate cannot turn a lost object into a false-negative training sample.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM
from ultralytics.models.sam import SAM2DynamicInteractivePredictor

from sam2_recovery import mask_iou, warp_mask_with_flow


def parse_args():
    p = argparse.ArgumentParser(description="首帧/关键帧mask驱动的SAM2视频传播")
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--key-mask-dir", type=Path, required=True)
    p.add_argument("--output-mask-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("models/sam2.1_t.pt"))
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--memory-update-interval", type=int, default=5, help="0表示只使用人工关键帧记忆")
    p.add_argument("--min-mask-area", type=int, default=80)
    p.add_argument("--min-area-ratio", type=float, default=0.35)
    p.add_argument("--max-area-ratio", type=float, default=2.8)
    p.add_argument("--min-depth-coverage", type=float, default=0.20)
    p.add_argument("--max-consecutive-failures", type=int, default=3)
    p.add_argument("--depth-to-metre", type=float, default=0.001)
    p.add_argument(
        "--auto-reregister", action=argparse.BooleanOptionalAction, default=True,
        help="跟踪失败后用最后可信Mask的光流投影重建SAM2记忆；默认开启",
    )
    p.add_argument(
        "--auto-reregister-after-failures", type=int, default=1,
        help="连续失败达到该次数后尝试自动重注册",
    )
    p.add_argument(
        "--min-recovery-seed-iou", type=float, default=0.35,
        help="SAM2恢复结果与光流种子Mask的最小IoU",
    )
    p.add_argument(
        "--max-flow-shift-norm", type=float, default=0.35,
        help="允许的光流P95位移/图像对角线；过大时禁止自动重注册",
    )
    return p.parse_args()


def read_mask(path: Path, shape):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3:
        raw = raw.max(axis=2)
    if raw.shape != shape:
        raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = raw > 0
    return mask if mask.any() else None


def build_predictor(model: SAM, args):
    overrides = {
        "conf": 0.0, "task": "segment", "mode": "predict",
        "imgsz": args.imgsz, "device": args.device,
        "retina_masks": True, "verbose": False, "save": False, "save_txt": False,
    }
    predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=2)
    predictor.setup_model(model=model.model, verbose=False)
    return predictor


def result_mask(results, shape):
    if not results:
        return None, None
    result = results[0]
    if result.masks is None or len(result.masks.data) == 0:
        return None, None
    data = result.masks.data.detach().float().cpu().numpy()
    areas = data.reshape(len(data), -1).sum(axis=1)
    idx = int(np.argmax(areas))
    mask = data[idx] > 0.0
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    score = None
    if result.boxes is not None and getattr(result.boxes, "conf", None) is not None and len(result.boxes.conf) > idx:
        score = float(result.boxes.conf[idx].detach().cpu())
    return mask, score



def quick_quality(mask, depth, previous_mask, args):
    area = int(mask.sum()) if mask is not None else 0
    coverage = 0.0
    if area and depth is not None:
        coverage = float(np.logical_and(mask, depth > 0).sum() / area)
    ratio = None if previous_mask is None or not previous_mask.any() else float(area / max(1, int(previous_mask.sum())))
    reasons = []
    if area < args.min_mask_area:
        reasons.append("mask_area_too_small")
    if ratio is not None and not (args.min_area_ratio <= ratio <= args.max_area_ratio):
        reasons.append("mask_area_jump")
    if coverage < args.min_depth_coverage:
        reasons.append("depth_coverage_too_low")
    return not reasons, {"mask_area": area, "area_ratio": ratio, "depth_coverage": coverage}, reasons


def main():
    args = parse_args()
    if args.auto_reregister_after_failures < 1:
        raise SystemExit("--auto-reregister-after-failures 必须>=1")
    if not 0.0 <= args.min_recovery_seed_iou <= 1.0:
        raise SystemExit("--min-recovery-seed-iou 必须在0到1之间")
    scene = args.scene.expanduser().resolve()
    key_dir = args.key_mask_dir.expanduser().resolve()
    output = args.output_mask_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rgb_files = sorted((scene / "rgb").glob("*.png"))
    end = len(rgb_files) if args.max_frames <= 0 else min(len(rgb_files), args.start + args.max_frames)
    selected = rgb_files[args.start:end]
    if not selected:
        raise SystemExit("没有选择到RGB帧")
    selected_ids = {x.stem for x in selected}
    key_ids = [p.stem for p in key_dir.glob("*.png") if p.stem in selected_ids]
    if not key_ids:
        raise SystemExit(f"所选帧中没有关键mask：{key_dir}")

    model_path = args.model.expanduser().resolve()
    model = SAM(str(model_path))
    predictor = None
    previous_good = None
    previous_good_bgr = None
    failures = 0
    records = []
    requests = []
    frames_since_key = 0

    for rgb_path in selected:
        stem = rgb_path.stem
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(scene / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            records.append({"id": stem, "status": "rejected", "mode": "rgb_missing"})
            continue
        key_path = key_dir / f"{stem}.png"
        key_mask = read_mask(key_path, bgr.shape[:2]) if key_path.exists() else None
        mode = "track"
        score = None
        recovery = None
        try:
            if key_mask is not None:
                # A trusted keyframe starts a fresh memory bank: deterministic re-registration.
                predictor = build_predictor(model, args)
                results = predictor(source=bgr, masks=key_mask[None].astype(np.uint8), obj_ids=[0], update_memory=True)
                mask, score = result_mask(results, bgr.shape[:2])
                mode = "keyframe_register"
                frames_since_key = 0
                if mask is None:
                    mask = key_mask
            elif predictor is None:
                records.append({
                    "id": stem, "status": "rejected", "mode": "waiting_for_keyframe",
                    "reasons": ["no_prior_keyframe"], "recovery": None,
                })
                continue
            else:
                results = predictor(source=bgr)
                mask, score = result_mask(results, bgr.shape[:2])
                frames_since_key += 1
        except Exception as exc:
            mask = None
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = None

        good, metrics, reasons = quick_quality(mask, depth, previous_good, args)
        if key_mask is not None:
            # The human mask is ground truth at a keyframe and must not be replaced by a worse prediction.
            mask = key_mask
            good, metrics, reasons = quick_quality(mask, depth, previous_good, args)
            if good:
                mode = "keyframe_register"
        if error:
            reasons.append(error)
            good = False

        original_failure_reasons = list(reasons)
        should_recover = (
            not good and key_mask is None and args.auto_reregister
            and previous_good is not None and previous_good_bgr is not None
            and failures + 1 >= args.auto_reregister_after_failures
        )
        if should_recover:
            seed, flow_metrics, flow_reasons = warp_mask_with_flow(
                previous_good_bgr, bgr, previous_good,
                max_flow_shift_norm=args.max_flow_shift_norm,
            )
            recovery = {
                "attempted": True,
                "seed_metrics": flow_metrics,
                "seed_reasons": flow_reasons,
                "trigger_reasons": original_failure_reasons,
            }
            if seed is not None:
                try:
                    recovered_predictor = build_predictor(model, args)
                    recovered_results = recovered_predictor(
                        source=bgr, masks=seed[None].astype(np.uint8), obj_ids=[0], update_memory=True,
                    )
                    recovered_mask, recovered_score = result_mask(recovered_results, bgr.shape[:2])
                except Exception as exc:
                    recovery["error"] = f"{type(exc).__name__}: {exc}"
                else:
                    seed_iou = mask_iou(recovered_mask, seed)
                    recovery["seed_iou"] = seed_iou
                    recovered_good, recovered_metrics, recovered_reasons = quick_quality(
                        recovered_mask, depth, previous_good, args,
                    )
                    if seed_iou < args.min_recovery_seed_iou:
                        recovered_reasons.append("recovery_seed_iou_too_low")
                        recovered_good = False
                    recovery["quality_metrics"] = recovered_metrics
                    recovery["quality_reasons"] = recovered_reasons
                    if recovered_good:
                        predictor = recovered_predictor
                        mask = recovered_mask
                        score = recovered_score
                        good = True
                        metrics = recovered_metrics | {"recovery_seed_iou": seed_iou} | flow_metrics
                        reasons = []
                        mode = "auto_reregister_flow"
                        frames_since_key = 0
                        recovery["succeeded"] = True
            if not good:
                recovery["succeeded"] = False
                reasons = original_failure_reasons + flow_reasons
                reasons.append("auto_reregister_failed")

        if good:
            cv2.imwrite(str(output / f"{stem}.png"), mask.astype(np.uint8) * 255)
            previous_good = mask.copy()
            previous_good_bgr = bgr.copy()
            failures = 0
            if key_mask is None and mode != "auto_reregister_flow" and args.memory_update_interval > 0 and frames_since_key % args.memory_update_interval == 0:
                try:
                    predictor(source=bgr, masks=mask[None].astype(np.uint8), obj_ids=[0], update_memory=True)
                    mode = "track_memory_update"
                except Exception as exc:
                    reasons.append(f"memory_update_failed:{type(exc).__name__}")
            status = "accepted"
        else:
            failures += 1
            status = "rejected"
            stale = output / f"{stem}.png"
            if stale.exists():
                stale.unlink()
            if failures == args.max_consecutive_failures:
                requests.append({
                    "after_frame": stem, "suggested_keyframe": stem,
                    "reason": f"{failures}_consecutive_sam2_failures_after_auto_recovery",
                    "required_mask": str(key_dir / f"{stem}.png"),
                })
        records.append({
            "id": stem, "status": status, "mode": mode, "score": score,
            "metrics": metrics, "reasons": reasons, "recovery": recovery,
            "key_mask": str(key_dir / f"{stem}.png") if key_mask is not None else None,
        })

    report = {
        "scene": str(scene), "key_mask_dir": str(key_dir), "output_mask_dir": str(output),
        "model": str(model_path), "parameters": vars(args) | {
            "scene": str(scene), "key_mask_dir": str(key_dir),
            "output_mask_dir": str(output), "model": str(model_path),
        },
        "status_counts": {s: sum(r["status"] == s for r in records) for s in ("accepted", "rejected")},
        "auto_reregistration": {
            "attempted": sum(bool(r.get("recovery", {}).get("attempted")) for r in records if r.get("recovery")),
            "succeeded": sum(bool(r.get("recovery", {}).get("succeeded")) for r in records if r.get("recovery")),
        },
        "reregistration_requests": requests, "records": records,
    }
    report_path = output / "sam2_propagation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with (output / "sam2_propagation_report.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "status", "mode", "score", "reasons", "key_mask"])
        writer.writeheader()
        for r in records:
            writer.writerow({**{k: r.get(k) for k in writer.fieldnames}, "reasons": "|".join(r.get("reasons", []))})
    print(
        f"SAM2传播完成：accepted={report['status_counts']['accepted']} "
        f"rejected={report['status_counts']['rejected']} "
        f"auto_reregister={report['auto_reregistration']['succeeded']}/{report['auto_reregistration']['attempted']}"
    )
    print(f"输出：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
