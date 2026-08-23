#!/usr/bin/env python3
"""CPU-only tests for conservative SAM2 optical-flow recovery."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from sam2_recovery import mask_iou, remap_mask_with_backward_flow, warp_mask_with_flow


def rectangle(shape=(48, 64), x0=10, y0=12, x1=22, y1=28):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def main():
    a = rectangle()
    b = rectangle(x0=14, x1=26)
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, None) == 0.0
    expected_iou = (8 * 16) / ((12 + 12 - 8) * 16)
    assert abs(mask_iou(a, b) - expected_iou) < 1e-9

    # cv2.remap uses an inverse map. A backward x-flow of -5 moves the mask +5.
    flow = np.zeros((*a.shape, 2), dtype=np.float32)
    flow[..., 0] = -5.0
    shifted = remap_mask_with_backward_flow(a, flow)
    expected = rectangle(x0=15, x1=27)
    assert np.array_equal(shifted, expected), "backward-flow direction is wrong"

    try:
        remap_mask_with_backward_flow(a, np.zeros((10, 10, 2), np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must be rejected")

    previous = np.zeros((*a.shape, 3), dtype=np.uint8)
    current = previous.copy()
    with patch("sam2_recovery.cv2.calcOpticalFlowFarneback", return_value=flow):
        seed, metrics, reasons = warp_mask_with_flow(previous, current, a, max_flow_shift_norm=0.20)
    assert not reasons and seed is not None
    assert np.array_equal(seed, expected)
    assert metrics["flow_p95_px"] == 5.0

    huge = np.zeros_like(flow)
    huge[..., 0] = -20.0
    with patch("sam2_recovery.cv2.calcOpticalFlowFarneback", return_value=huge):
        seed, metrics, reasons = warp_mask_with_flow(previous, current, a, max_flow_shift_norm=0.20)
    assert seed is None
    assert "recovery_flow_shift_too_large" in reasons
    assert metrics["flow_p95_shift_norm"] > 0.20

    bad_current = np.zeros((40, 64, 3), dtype=np.uint8)
    seed, metrics, reasons = warp_mask_with_flow(previous, bad_current, a)
    assert seed is None and reasons == ["recovery_shape_mismatch"]

    print("SAM2_RECOVERY_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
