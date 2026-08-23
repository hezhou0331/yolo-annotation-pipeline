#!/usr/bin/env python3
"""Conservative optical-flow recovery helpers for SAM2 re-registration.

This module has no Ultralytics dependency so its geometry can be unit-tested on
CPU. A recovery seed is only a proposal; the caller must still run SAM2 and the
RGB-D quality gates before accepting a label.
"""
from __future__ import annotations

import cv2
import numpy as np


def mask_iou(mask_a, mask_b) -> float:
    if mask_a is None or mask_b is None:
        return 0.0
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    return 0.0 if union <= 0 else float(np.logical_and(a, b).sum() / union)


def remap_mask_with_backward_flow(mask, backward_flow):
    """Warp a previous-frame mask into the current frame.

    backward_flow[y, x] points from a current-frame pixel to its source
    position in the previous frame. This inverse-map convention matches
    cv2.remap and avoids holes from forward splatting.
    """
    source = np.asarray(mask, dtype=np.uint8)
    flow = np.asarray(backward_flow, dtype=np.float32)
    if source.ndim != 2 or flow.shape[:2] != source.shape or flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError("mask与backward_flow尺寸不匹配")
    h, w = source.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = cv2.remap(
        source,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    return warped > 0


def warp_mask_with_flow(previous_bgr, current_bgr, previous_mask, max_flow_shift_norm=0.35):
    """Create a conservative current-frame seed from the last accepted frame.

    Returns (seed_or_none, metrics, rejection_reasons). Excessive motion,
    shape mismatch, or an empty projection rejects the seed before SAM2 is
    allowed to rebuild its memory.
    """
    if previous_bgr is None or current_bgr is None or previous_mask is None:
        return None, {}, ["recovery_history_missing"]
    if previous_bgr.shape[:2] != current_bgr.shape[:2] or previous_mask.shape != current_bgr.shape[:2]:
        return None, {}, ["recovery_shape_mismatch"]
    prev_gray = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    # Farneback(prev, next) returns prev->next. Reversing image order gives
    # current->previous, the inverse map required by remap_mask_with_backward_flow.
    backward_flow = cv2.calcOpticalFlowFarneback(
        curr_gray,
        prev_gray,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=25,
        iterations=4,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    seed = remap_mask_with_backward_flow(previous_mask, backward_flow)
    area = int(seed.sum())
    if area <= 0:
        return None, {"seed_area": 0}, ["recovery_seed_empty"]
    magnitude = np.linalg.norm(backward_flow, axis=2)
    values = magnitude[seed]
    diagonal = float(np.hypot(seed.shape[0], seed.shape[1]))
    metrics = {
        "seed_area": area,
        "flow_median_px": float(np.median(values)) if values.size else 0.0,
        "flow_p95_px": float(np.percentile(values, 95)) if values.size else 0.0,
    }
    metrics["flow_p95_shift_norm"] = metrics["flow_p95_px"] / max(diagonal, 1.0)
    if metrics["flow_p95_shift_norm"] > max_flow_shift_norm:
        return None, metrics, ["recovery_flow_shift_too_large"]
    return seed, metrics, []
