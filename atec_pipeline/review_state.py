"""Persistent manual review decisions for propagated mask sequences.

The automatic quality report remains immutable.  Human decisions are stored in
one small sidecar JSON file under the source scene so rerunning SAM2 or moving a
YOLO split cannot silently lose them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

VALID_STATUSES = {"accepted", "review", "rejected"}


@dataclass(frozen=True)
class ReviewSegment:
    segment_id: str
    start_id: str
    end_id: str
    frame_ids: tuple[str, ...]


class ReviewState:
    def __init__(
        self,
        *,
        records: dict[str, dict[str, Any]],
        frame_ids: tuple[str, ...],
        segments: tuple[ReviewSegment, ...],
        overrides_path: Path,
        mask_dir: Path,
        scene_name: str,
        instance_id: str,
        overrides: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.records = records
        self.frame_ids = frame_ids
        self.segments = segments
        self.overrides_path = Path(overrides_path).expanduser().resolve()
        self.mask_dir = Path(mask_dir).expanduser().resolve()
        self.scene_name = scene_name
        self.instance_id = instance_id
        self.overrides: dict[str, dict[str, Any]] = dict(overrides or {})
        self.problem_range_start: str | None = None

    @classmethod
    def load(
        cls,
        quality_report: Path,
        overrides_path: Path,
        mask_dir: Path,
        segments_report: Path | None = None,
        *,
        scene_name: str = "",
        instance_id: str = "",
    ) -> "ReviewState":
        report_path = Path(quality_report).expanduser().resolve()
        data = json.loads(report_path.read_text(encoding="utf-8"))
        raw_records = [row for row in data.get("records", []) if row.get("id") is not None]
        records = {str(row["id"]): dict(row) for row in raw_records}
        frame_ids = tuple(sorted(records))
        if not frame_ids:
            raise ValueError(f"质量报告没有逐帧记录：{report_path}")

        segments = _load_segments(segments_report, frame_ids)
        override_path = Path(overrides_path).expanduser().resolve()
        overrides: dict[str, dict[str, Any]] = {}
        if override_path.is_file():
            payload = json.loads(override_path.read_text(encoding="utf-8"))
            for frame_id, row in (payload.get("frames") or {}).items():
                status = str((row or {}).get("status", ""))
                if frame_id in records and status in VALID_STATUSES:
                    overrides[str(frame_id)] = dict(row)
        return cls(
            records=records,
            frame_ids=frame_ids,
            segments=segments,
            overrides_path=override_path,
            mask_dir=mask_dir,
            scene_name=scene_name or Path(str(data.get("scene", "scene"))).name,
            instance_id=instance_id or str(data.get("instance_id", "instance")),
            overrides=overrides,
        )

    def auto_status(self, frame_id: str) -> str:
        return str(self.records[frame_id].get("auto_status") or self.records[frame_id].get("status") or "rejected")

    def manual_status(self, frame_id: str) -> str | None:
        row = self.overrides.get(frame_id)
        return str(row["status"]) if row else None

    def effective_status(self, frame_id: str) -> str:
        return self.manual_status(frame_id) or self.auto_status(frame_id)

    def reason_text(self, frame_id: str) -> str:
        manual = self.overrides.get(frame_id)
        if manual:
            return str(manual.get("reason") or f"manual_{manual['status']}")
        record = self.records[frame_id]
        reasons = list(record.get("reject_reasons") or []) + list(record.get("review_reasons") or [])
        return ", ".join(str(reason) for reason in reasons) or "-"

    def mask_path(self, frame_id: str) -> Path:
        return self.mask_dir / f"{frame_id}.png"

    def has_mask(self, frame_id: str) -> bool:
        path = self.mask_path(frame_id)
        return path.is_file() and path.stat().st_size > 0

    def set_status(self, frame_id: str, status: str, *, reason: str | None = None) -> None:
        if frame_id not in self.records:
            raise KeyError(f"未知帧：{frame_id}")
        if status not in VALID_STATUSES:
            raise ValueError(f"不支持的人工状态：{status}")
        if status == "accepted" and not self.has_mask(frame_id):
            raise ValueError(f"当前帧没有有效 Mask，不能设为 accepted：{frame_id}")
        self.overrides[frame_id] = {
            "status": status,
            "reason": reason or f"manual_{status}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def clear_status(self, frame_id: str) -> None:
        self.overrides.pop(frame_id, None)

    def segment_for_frame(self, frame_id: str) -> ReviewSegment:
        for segment in self.segments:
            if frame_id in segment.frame_ids:
                return segment
        raise KeyError(f"帧不属于任何分段：{frame_id}")

    def begin_problem_range(self, frame_id: str) -> None:
        if frame_id not in self.records:
            raise KeyError(frame_id)
        self.problem_range_start = frame_id

    def finish_problem_range(self, frame_id: str) -> tuple[str, ...]:
        if self.problem_range_start is None:
            raise RuntimeError("尚未设置问题区间起点")
        start = self.frame_ids.index(self.problem_range_start)
        end = self.frame_ids.index(frame_id)
        if end < start:
            start, end = end, start
        changed = self.frame_ids[start:end + 1]
        for item in changed:
            self.set_status(item, "rejected", reason="manual_problem_range")
        self.problem_range_start = None
        return changed

    def reject_from(self, frame_id: str, *, scope: str = "segment") -> tuple[str, ...]:
        if scope not in {"segment", "scene"}:
            raise ValueError("scope 必须是 segment 或 scene")
        start = self.frame_ids.index(frame_id)
        if scope == "scene":
            changed = self.frame_ids[start:]
            reason = "manual_rejected_scene_tail"
        else:
            segment = self.segment_for_frame(frame_id)
            segment_ids = segment.frame_ids
            segment_start = segment_ids.index(frame_id)
            changed = segment_ids[segment_start:]
            reason = "manual_rejected_segment_tail"
        for item in changed:
            self.set_status(item, "rejected", reason=reason)
        return tuple(changed)

    def keyframe_path(self, key_mask_dir: Path, frame_id: str) -> Path:
        if frame_id not in self.records:
            raise KeyError(frame_id)
        return Path(key_mask_dir).expanduser().resolve() / f"{frame_id}.png"

    def save(self) -> None:
        payload = {
            "schema_version": 1,
            "scene": self.scene_name,
            "instance_id": self.instance_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "frames": {key: self.overrides[key] for key in sorted(self.overrides)},
        }
        self.overrides_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.overrides_path.with_name(f".{self.overrides_path.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.overrides_path)


def _load_segments(path: Path | None, frame_ids: tuple[str, ...]) -> tuple[ReviewSegment, ...]:
    if path is None or not Path(path).expanduser().is_file():
        return (ReviewSegment("0", frame_ids[0], frame_ids[-1], frame_ids),)
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    position = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    segments: list[ReviewSegment] = []
    for index, raw in enumerate(payload.get("segments") or []):
        start_id = str(raw.get("start_id", frame_ids[0]))
        end_id = str(raw.get("end_id", frame_ids[-1]))
        if start_id not in position or end_id not in position:
            continue
        start = position[start_id]
        end = position[end_id]
        if end < start:
            start, end = end, start
        members = frame_ids[start:end + 1]
        if members:
            segments.append(ReviewSegment(str(raw.get("segment_id", index)), members[0], members[-1], members))
    return tuple(segments) or (ReviewSegment("0", frame_ids[0], frame_ids[-1], frame_ids),)
