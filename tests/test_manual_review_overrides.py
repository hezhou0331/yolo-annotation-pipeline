#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from annotate_mask_sequence_yolo import apply_manual_review, load_manual_review
from annotate_multinstance_project import add_review_override_arg


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="atec_manual_gate_") as tmp:
        path = Path(tmp) / "review.json"
        path.write_text(json.dumps({"frames": {
            "000001": {"status": "rejected", "reason": "manual_bad_mask"},
            "000002": {"status": "review", "reason": "manual_check"},
            "000003": {"status": "accepted", "reason": "manual_confirmed"},
            "000004": {"status": "accepted", "reason": "manual_confirmed"},
        }}), encoding="utf-8")
        overrides = load_manual_review(path)
        assert set(overrides) == {"000001", "000002", "000003", "000004"}

        decision = apply_manual_review("000001", "accepted", [], [], "0 0.1 0.1 0.2 0.2", overrides)
        assert decision.status == "rejected" and decision.manual_status == "rejected"
        assert decision.reject_reasons == ("manual_bad_mask",)

        decision = apply_manual_review("000002", "accepted", [], [], "line", overrides)
        assert decision.status == "review" and decision.manual_status == "review"
        assert decision.review_reasons == ("manual_check",)

        decision = apply_manual_review("000003", "review", [], ["touches_image_border"], "line", overrides)
        assert decision.status == "accepted" and decision.manual_status == "accepted"
        assert not decision.reject_reasons and not decision.review_reasons

        decision = apply_manual_review("000004", "rejected", ["mask_file_missing"], [], None, overrides)
        assert decision.status == "rejected", "manual accepted must not invent a YOLO polygon"
        assert decision.manual_status == "accepted"
        assert "manual_accept_blocked_invalid_mask" in decision.reject_reasons

        decision = apply_manual_review("000999", "review", [], ["auto_reason"], "line", overrides)
        assert decision.status == "review" and decision.manual_status is None
        assert decision.review_reasons == ("auto_reason",)

        command = ["python", "annotate_mask_sequence_yolo.py"]
        expected = Path(tmp) / "scene/project_reports/manual_mask_review/can_01.json"
        add_review_override_arg(command, Path(tmp) / "scene", "can_01")
        assert command[-2:] == ["--review-overrides", str(expected)]

    print("MANUAL_REVIEW_OVERRIDE_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
