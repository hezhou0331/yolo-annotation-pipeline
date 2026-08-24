#!/usr/bin/env python3
"""Regression tests for SAM2 binary-mask prompt compatibility."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from propagate_masks_sam2 import prepare_mask_prompt, predict_with_mask_prompt  # noqa: E402


class RecordingPredictor:
    def __init__(self, result=None):
        self.calls = []
        self.result = [] if result is None else result

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def main() -> int:
    mask = np.zeros((12, 16), dtype=bool)
    mask[2:8, 3:11] = True
    prompt = prepare_mask_prompt(mask)
    assert prompt.shape == (1, 12, 16, 1)
    assert prompt.dtype == np.uint8
    assert int(prompt.sum()) == 48

    already_batched = prepare_mask_prompt(prompt)
    assert already_batched.shape == prompt.shape
    assert np.array_equal(already_batched, prompt)

    for invalid, expected in [
        (np.zeros((12, 16), dtype=np.uint8), "不能为空"),
        (np.zeros((2, 12, 16), dtype=np.uint8), "二维"),
        (np.zeros((1, 12, 16, 2), dtype=np.uint8), "单通道"),
    ]:
        try:
            prepare_mask_prompt(invalid)
        except ValueError as exc:
            assert expected in str(exc), str(exc)
        else:
            raise AssertionError(f"invalid prompt must be rejected: {invalid.shape}")

    predictor = RecordingPredictor(result=["ok"])
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    result = predict_with_mask_prompt(predictor, image, mask, update_memory=True)
    assert result == ["ok"]
    assert len(predictor.calls) == 1
    call = predictor.calls[0]
    assert call["masks"].shape == (1, 12, 16, 1)
    assert call["obj_ids"] == [0]
    assert call["update_memory"] is True

    print("SAM2_PROPAGATION_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
