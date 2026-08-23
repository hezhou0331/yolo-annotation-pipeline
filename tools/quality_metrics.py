#!/usr/bin/env python3
"""Quality metrics shared by the RGB-D auto-annotation pipeline.

The module has no FoundationPose dependency, so it can be unit-tested in the
lightweight YOLO environment.  All distances are metres.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class QualityThresholds:
    min_mask_area: int = 80
    min_area_fraction: float = 0.00015
    max_area_fraction: float = 0.65
    min_depth_coverage: float = 0.45
    max_depth_median_abs_m: float = 0.05
    max_depth_rmse_m: float = 0.08
    min_area_ratio: float = 0.45
    max_area_ratio: float = 2.20
    max_center_shift_norm: float = 0.35
    max_translation_jump_m: float = 0.20
    max_rotation_jump_deg: float = 55.0
    min_dominant_component_ratio: float = 0.75
    border_margin_px: int = 2
    review_depth_coverage: float = 0.60
    review_depth_median_abs_m: float = 0.035
    review_center_shift_norm: float = 0.22
    review_translation_jump_m: float = 0.12
    review_rotation_jump_deg: float = 35.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def rotation_jump_deg(previous_pose: np.ndarray | None, pose: np.ndarray) -> float | None:
    if previous_pose is None:
        return None
    relative = previous_pose[:3, :3].T @ pose[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def translation_jump_m(previous_pose: np.ndarray | None, pose: np.ndarray) -> float | None:
    if previous_pose is None:
        return None
    return float(np.linalg.norm(pose[:3, 3] - previous_pose[:3, 3]))


def compute_quality_metrics(
    mask: np.ndarray,
    rendered_depth_m: np.ndarray,
    observed_depth_m: np.ndarray,
    pose: np.ndarray,
    previous_mask: np.ndarray | None = None,
    previous_pose: np.ndarray | None = None,
    border_margin_px: int = 2,
) -> dict[str, Any]:
    mask = mask.astype(bool)
    h, w = mask.shape
    area = int(mask.sum())
    image_area = max(1, h * w)
    bbox = bbox_from_mask(mask)

    metrics: dict[str, Any] = {
        "mask_area": area,
        "area_fraction": float(area / image_area),
        "bbox_xyxy": list(bbox) if bbox else None,
        "touches_border": False,
        "center_xy": None,
        "center_shift_norm": None,
        "area_ratio": None,
        "connected_components": 0,
        "dominant_component_ratio": 0.0,
        "depth_coverage": 0.0,
        "depth_valid_pixels": 0,
        "depth_median_abs_m": None,
        "depth_rmse_m": None,
        "translation_jump_m": translation_jump_m(previous_pose, pose),
        "rotation_jump_deg": rotation_jump_deg(previous_pose, pose),
    }

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        margin = max(0, int(border_margin_px))
        metrics["touches_border"] = bool(
            x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin
        )
        metrics["center_xy"] = [(x1 + x2) * 0.5, (y1 + y2) * 0.5]

    if area > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        component_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty((0,), dtype=np.int32)
        metrics["connected_components"] = int(component_areas.size)
        if component_areas.size:
            metrics["dominant_component_ratio"] = float(component_areas.max() / max(1, area))

        observed_valid = np.isfinite(observed_depth_m) & (observed_depth_m > 0)
        rendered_valid = np.isfinite(rendered_depth_m) & (rendered_depth_m > 0)
        comparable = mask & observed_valid & rendered_valid
        metrics["depth_valid_pixels"] = int(comparable.sum())
        metrics["depth_coverage"] = float(comparable.sum() / max(1, area))
        if comparable.any():
            residual = rendered_depth_m[comparable] - observed_depth_m[comparable]
            abs_residual = np.abs(residual)
            metrics["depth_median_abs_m"] = float(np.median(abs_residual))
            metrics["depth_rmse_m"] = float(np.sqrt(np.mean(residual * residual)))

    if previous_mask is not None and previous_mask.any() and area > 0:
        previous_area = int(previous_mask.astype(bool).sum())
        metrics["area_ratio"] = float(area / max(1, previous_area))
        prev_bbox = bbox_from_mask(previous_mask.astype(bool))
        if bbox is not None and prev_bbox is not None:
            x1, y1, x2, y2 = bbox
            px1, py1, px2, py2 = prev_bbox
            center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
            prev_center = np.asarray([(px1 + px2) * 0.5, (py1 + py2) * 0.5], dtype=np.float64)
            metrics["center_shift_norm"] = float(np.linalg.norm(center - prev_center) / np.hypot(w, h))

    return metrics


def classify_quality(
    metrics: dict[str, Any],
    thresholds: QualityThresholds,
    registration_frame: bool = False,
) -> tuple[str, list[str], list[str]]:
    reject: list[str] = []
    review: list[str] = []

    def too_low(key: str, limit: float, reason: str) -> None:
        value = metrics.get(key)
        if value is None:
            return
        if float(value) < limit:
            reject.append(reason)

    def too_high(key: str, limit: float, reason: str) -> None:
        value = metrics.get(key)
        if value is None:
            return
        if float(value) > limit:
            reject.append(reason)

    too_low("mask_area", thresholds.min_mask_area, "mask_area_too_small")
    too_low("area_fraction", thresholds.min_area_fraction, "object_too_small_in_image")
    too_high("area_fraction", thresholds.max_area_fraction, "object_too_large_in_image")
    too_low("depth_coverage", thresholds.min_depth_coverage, "depth_coverage_too_low")
    too_high("depth_median_abs_m", thresholds.max_depth_median_abs_m, "depth_alignment_bad")
    too_high("depth_rmse_m", thresholds.max_depth_rmse_m, "depth_rmse_bad")
    too_low("dominant_component_ratio", thresholds.min_dominant_component_ratio, "mask_fragmented")

    if not registration_frame:
        ratio = metrics.get("area_ratio")
        if ratio is not None and not (thresholds.min_area_ratio <= float(ratio) <= thresholds.max_area_ratio):
            reject.append("mask_area_jump")
        too_high("center_shift_norm", thresholds.max_center_shift_norm, "mask_center_jump")
        too_high("translation_jump_m", thresholds.max_translation_jump_m, "translation_jump")
        too_high("rotation_jump_deg", thresholds.max_rotation_jump_deg, "rotation_jump")

    if not reject:
        if metrics.get("touches_border"):
            review.append("touches_image_border")
        coverage = metrics.get("depth_coverage")
        if coverage is not None and float(coverage) < thresholds.review_depth_coverage:
            review.append("depth_coverage_marginal")
        median = metrics.get("depth_median_abs_m")
        if median is not None and float(median) > thresholds.review_depth_median_abs_m:
            review.append("depth_alignment_marginal")
        if not registration_frame:
            for key, limit, reason in (
                ("center_shift_norm", thresholds.review_center_shift_norm, "large_center_motion"),
                ("translation_jump_m", thresholds.review_translation_jump_m, "large_translation_motion"),
                ("rotation_jump_deg", thresholds.review_rotation_jump_deg, "large_rotation_motion"),
            ):
                value = metrics.get(key)
                if value is not None and float(value) > limit:
                    review.append(reason)

    status = "rejected" if reject else ("review" if review else "accepted")
    return status, reject, review
