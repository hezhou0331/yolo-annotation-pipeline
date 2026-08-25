#!/usr/bin/env python3
"""Pure path/state helpers for the ATEC desktop application.

This module deliberately contains no Qt imports so it can be tested on a
headless machine and reused by CLI tooling.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from typing import Iterable, Sequence

import yaml


@dataclass(frozen=True)
class ObjectClass:
    class_id: int
    name: str
    chinese_name: str


@dataclass(frozen=True)
class MaskSegmentProgress:
    segment_id: str
    start_id: str
    end_id: str
    completed_required: int
    total_required: int
    missing_instances: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.total_required > 0 and self.completed_required == self.total_required


@dataclass(frozen=True)
class MaskProgress:
    segments: tuple[MaskSegmentProgress, ...]
    completed_required: int
    total_required: int

    @property
    def complete(self) -> bool:
        return bool(self.segments) and self.total_required > 0 and self.completed_required == self.total_required


@dataclass(frozen=True)
class TrainingSummary:
    run_dir: Path
    results_csv: Path
    best_weights: Path | None
    best_epoch: int
    mask_precision: float | None
    mask_recall: float | None
    mask_map50: float | None
    mask_map50_95: float | None


@dataclass(frozen=True)
class ExportSummary:
    report_path: Path
    accepted: int
    review: int
    rejected: int

    @property
    def total(self) -> int:
        return self.accepted + self.review + self.rejected

    @property
    def needs_review(self) -> int:
        return self.review + self.rejected

    @property
    def all_accepted(self) -> bool:
        return self.total > 0 and self.needs_review == 0


@dataclass(frozen=True)
class SceneWorkflowState:
    """One scene's filesystem-derived workflow state for CLI and GUI use."""

    scene_name: str
    class_name: str
    scene_dir: Path
    split: str
    manifest_path: Path | None
    segments_path: Path
    export_report_path: Path | None
    rgb_frames: int
    depth_frames: int
    paired_frames: int
    mask_completed: int
    mask_total: int
    group: str
    code: str
    color: str
    detail: str
    accepted: int = 0
    review: int = 0
    rejected: int = 0

    @property
    def training_eligible(self) -> bool:
        return self.accepted > 0

    @property
    def masks_complete(self) -> bool:
        return self.mask_total > 0 and self.mask_completed == self.mask_total


@dataclass(frozen=True)
class ClassWorkflowSummary:
    """Aggregated frame and workflow counts for the currently selected class."""

    scene_count: int
    paired_frames: int
    processed_frames: int
    accepted: int
    review: int
    rejected: int
    pending_propagation_scenes: int
    needs_manual_scenes: int


def summarize_scene_states(states: Iterable[SceneWorkflowState]) -> ClassWorkflowSummary:
    """Summarize scene-derived counts without reading the filesystem again."""
    scene_states = tuple(states)
    accepted = sum(state.accepted for state in scene_states)
    review = sum(state.review for state in scene_states)
    rejected = sum(state.rejected for state in scene_states)
    return ClassWorkflowSummary(
        scene_count=len(scene_states),
        paired_frames=sum(state.paired_frames for state in scene_states),
        processed_frames=accepted + review + rejected,
        accepted=accepted,
        review=review,
        rejected=rejected,
        pending_propagation_scenes=sum(state.code == "pending_export" for state in scene_states),
        needs_manual_scenes=sum(state.group == "needs_manual" for state in scene_states),
    )


@dataclass(frozen=True)
class SceneIntegrityReport:
    """Filesystem/metadata consistency for one RGB-D scene.

    ``safe_orphans`` are one-sided PNG files whose frame id was never
    committed to ``metadata.json``. They are typical of Ctrl+C arriving
    between the RGB and depth writes and can be quarantined without deleting
    a recorded frame. ``unsafe_recorded_missing`` always requires manual
    inspection.
    """

    scene_dir: Path
    rgb_count: int
    depth_count: int
    paired_count: int
    orphan_rgb: tuple[Path, ...]
    orphan_depth: tuple[Path, ...]
    metadata_frame_ids: frozenset[str]
    safe_orphans: tuple[Path, ...]
    unsafe_recorded_missing: tuple[str, ...]
    metadata_error: str | None = None

    @property
    def is_complete(self) -> bool:
        return not self.orphan_rgb and not self.orphan_depth and not self.unsafe_recorded_missing

    @property
    def can_auto_repair(self) -> bool:
        all_orphans = len(self.orphan_rgb) + len(self.orphan_depth)
        return (
            all_orphans > 0
            and len(self.safe_orphans) == all_orphans
            and not self.unsafe_recorded_missing
            and self.metadata_error is None
        )

    @property
    def summary(self) -> str:
        if self.is_complete:
            return f"数据完整：RGB/Depth 各 {self.paired_count} 帧，全部配对"
        if self.can_auto_repair:
            return (
                f"发现 {len(self.safe_orphans)} 个未配对且未写入 metadata 的孤立文件；"
                f"可安全隔离，保留 {self.paired_count} 对 RGB-D"
            )
        details = "、".join(self.unsafe_recorded_missing) or self.metadata_error or "存在未知不一致"
        return f"不能自动修复：{details}"


def load_export_summary(report_path: Path) -> ExportSummary | None:
    """Read accepted/review/rejected counts from one scene export report."""
    report = Path(report_path).expanduser().resolve()
    if not report.is_file():
        return None
    data = json.loads(report.read_text(encoding="utf-8"))
    counts = data.get("frame_status_counts") or {}
    return ExportSummary(
        report_path=report,
        accepted=int(counts.get("accepted", 0)),
        review=int(counts.get("review", 0)),
        rejected=int(counts.get("rejected", 0)),
    )


def find_scene_manifest(project_root: Path, scene_name: str) -> tuple[Path | None, str]:
    """Find one scene manifest without assuming the currently selected split."""
    root = Path(project_root).expanduser().resolve()
    for split in ("train", "val", "test"):
        candidate = root / "manifests" / f"{scene_name}_{split}.yaml"
        if candidate.is_file():
            try:
                data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                manifest_split = str((data.get("project") or {}).get("split") or split)
            except (OSError, TypeError, yaml.YAMLError):
                manifest_split = split
            return candidate, manifest_split
    return None, "train"


def _scene_export_report(project_root: Path, scene_name: str, split: str) -> Path | None:
    report_root = Path(project_root).expanduser().resolve() / "datasets" / "atec_yolo11_seg" / "project_reports"
    preferred = report_root / f"{scene_name}_{split}_report.json"
    if preferred.is_file():
        return preferred
    for candidate_split in ("train", "val", "test"):
        candidate = report_root / f"{scene_name}_{candidate_split}_report.json"
        if candidate.is_file():
            return candidate
    return None


def scene_workflow_state(project_root: Path, scene_dir: Path) -> SceneWorkflowState:
    """Calculate a scene state solely from files, never from stale GUI flags."""
    root = Path(project_root).expanduser().resolve()
    scene = Path(scene_dir).expanduser().resolve()
    scene_name = scene.name
    class_name = scene.parent.name
    integrity = inspect_scene_integrity(scene)
    rgb_frames = integrity.rgb_count
    depth_frames = integrity.depth_count
    paired_frames = integrity.paired_count
    manifest, split = find_scene_manifest(root, scene_name)
    segments = scene / "project_reports" / "segments.json"
    progress = load_mask_progress(segments) if segments.is_file() else MaskProgress((), 0, 0)
    report_path = _scene_export_report(root, scene_name, split)
    export = load_export_summary(report_path) if report_path else None

    common = dict(
        scene_name=scene_name,
        class_name=class_name,
        scene_dir=scene,
        split=split,
        manifest_path=manifest,
        segments_path=segments,
        export_report_path=report_path,
        rgb_frames=rgb_frames,
        depth_frames=depth_frames,
        paired_frames=paired_frames,
        mask_completed=progress.completed_required,
        mask_total=progress.total_required,
    )
    if paired_frames <= 0 or not integrity.is_complete:
        repair_hint = "；可在 App 中安全修复" if integrity.can_auto_repair else "；需要人工检查"
        detail = (
            f"RGB {rgb_frames} / Depth {depth_frames}，配对 {paired_frames}；"
            f"需要检查数据完整性{repair_hint}"
        )
        return SceneWorkflowState(**common, group="needs_manual", code="rgbd_incomplete", color="gray", detail=detail)
    if manifest is None:
        return SceneWorkflowState(
            **common, group="needs_manual", code="manifest_missing", color="gray",
            detail=f"{paired_frames} 对 RGB-D；尚未创建 Manifest",
        )
    if not segments.is_file():
        return SceneWorkflowState(
            **common, group="needs_manual", code="segments_missing", color="gray",
            detail=f"{paired_frames} 对 RGB-D；尚未自动分段",
        )
    if not progress.complete:
        missing = max(0, progress.total_required - progress.completed_required)
        return SceneWorkflowState(
            **common, group="needs_manual", code="masks_missing", color="gray",
            detail=(f"{paired_frames} 对 RGB-D；关键帧 {progress.completed_required}/{progress.total_required}；"
                    f"还缺 {missing} 个 Mask"),
        )
    if export is None:
        return SceneWorkflowState(
            **common, group="keyframes_complete", code="pending_export", color="blue",
            detail=f"{paired_frames} 对 RGB-D；关键帧 {progress.completed_required}/{progress.total_required}；待 SAM2 传播",
        )
    counts = dict(accepted=export.accepted, review=export.review, rejected=export.rejected)
    if export.accepted <= 0:
        return SceneWorkflowState(
            **common, **counts, group="keyframes_complete", code="export_failed", color="red",
            detail=(f"{paired_frames} 对 RGB-D；关键帧 {progress.completed_required}/{progress.total_required}；"
                    f"自动处理失败：accepted 0，rejected {export.rejected}"),
        )
    if export.needs_review:
        return SceneWorkflowState(
            **common, **counts, group="keyframes_complete", code="export_needs_review", color="yellow",
            detail=(f"accepted {export.accepted} / review {export.review} / rejected {export.rejected}；"
                    "仅 accepted 可进入数据集"),
        )
    return SceneWorkflowState(
        **common, **counts, group="keyframes_complete", code="dataset_ready", color="green",
        detail=f"accepted {export.accepted}；可用于 {split} 数据集",
    )


def _resolved_report_path(raw_path: str, report: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (report.parent / path).resolve()


def load_mask_progress(report_path: Path) -> MaskProgress:
    """Read per-segment key-mask completion from the actual required PNG files."""
    report = Path(report_path).expanduser().resolve()
    if not report.is_file():
        return MaskProgress((), 0, 0)
    data = json.loads(report.read_text(encoding="utf-8"))
    items: list[MaskSegmentProgress] = []
    for index, segment in enumerate(data.get("segments", [])):
        required = segment.get("required_key_mask_paths") or {}
        if required:
            missing = tuple(
                str(instance)
                for instance, raw_path in required.items()
                if not _resolved_report_path(str(raw_path), report).is_file()
            )
            total = len(required)
        else:
            missing = tuple(str(item) for item in (segment.get("missing_key_masks") or []))
            key_masks = segment.get("key_masks") or {}
            total = len(key_masks) or len(missing)
        items.append(MaskSegmentProgress(
            segment_id=str(segment.get("segment_id", index)),
            start_id=str(segment.get("start_id", "?")),
            end_id=str(segment.get("end_id", "?")),
            completed_required=max(0, total - len(missing)),
            total_required=total,
            missing_instances=missing,
        ))
    total_required = sum(item.total_required for item in items)
    completed_required = sum(item.completed_required for item in items)
    return MaskProgress(tuple(items), completed_required, total_required)


def _metric(row: dict[str, str], name: str) -> float | None:
    value = row.get(name)
    if value is None:
        return None
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


def load_training_summary(runs_root: Path, experiment_name: str = "") -> TrainingSummary | None:
    """Return the best validation-mask row from one experiment or the latest run."""
    root = Path(runs_root).expanduser().resolve()
    preferred = root / experiment_name / "results.csv" if experiment_name.strip() else None
    if preferred and preferred.is_file():
        results_csv = preferred
    else:
        candidates = [path for path in root.glob("*/results.csv") if path.is_file()] if root.is_dir() else []
        if not candidates:
            return None
        results_csv = max(candidates, key=lambda path: path.stat().st_mtime)
    with results_csv.open(encoding="utf-8", newline="") as handle:
        rows = [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        return None
    primary = "metrics/mAP50-95(M)"
    best = max(rows, key=lambda row: _metric(row, primary) if _metric(row, primary) is not None else float("-inf"))
    epoch_value = _metric(best, "epoch")
    run_dir = results_csv.parent
    best_weights = run_dir / "weights" / "best.pt"
    return TrainingSummary(
        run_dir=run_dir,
        results_csv=results_csv,
        best_weights=best_weights if best_weights.is_file() else None,
        best_epoch=int(epoch_value) if epoch_value is not None else -1,
        mask_precision=_metric(best, "metrics/precision(M)"),
        mask_recall=_metric(best, "metrics/recall(M)"),
        mask_map50=_metric(best, "metrics/mAP50(M)"),
        mask_map50_95=_metric(best, primary),
    )


def find_best_weights(runs_root: Path, experiment_name: str = "") -> Path | None:
    """Find a real training ``best.pt``; never search the base-model directory."""
    root = Path(runs_root).expanduser().resolve()
    preferred = root / experiment_name / "weights" / "best.pt" if experiment_name.strip() else None
    if preferred and preferred.is_file():
        return preferred
    candidates = [path for path in root.glob("*/weights/best.pt") if path.is_file()] if root.is_dir() else []
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


@dataclass
class CaptureSession:
    project_root: Path
    object_class: ObjectClass
    split: str
    scene_name: str
    staging_dir: Path
    scene_dir: Path
    capture_session_id: str
    source_video_id: str
    remark: str = ""

    def save(self) -> Path:
        """Atomically formalize the staging directory without overwriting data."""
        staging = self.staging_dir.resolve()
        staging_root = (self.project_root / "data" / ".staging").resolve()
        if staging.parent != staging_root:
            raise ValueError(f"拒绝保存不在staging目录下的路径: {staging}")
        if not staging.exists():
            raise FileNotFoundError(f"采集暂存目录不存在: {staging}")
        target = self.scene_dir.resolve()
        if target.exists():
            raise FileExistsError(f"正式场次已存在，不覆盖: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(target))
        return target

    def discard(self) -> None:
        """Delete only this session's staging directory."""
        staging = self.staging_dir.resolve()
        staging_root = (self.project_root / "data" / ".staging").resolve()
        if staging.parent != staging_root:
            raise ValueError(f"拒绝删除不在staging目录下的路径: {staging}")
        if staging.exists():
            shutil.rmtree(staging)

    def metadata(self) -> dict[str, str]:
        return {
            "capture_session_id": self.capture_session_id,
            "source_video_id": self.source_video_id,
            "class_name": self.object_class.name,
            "class_id": str(self.object_class.class_id),
            "split": self.split,
            "scene_name": self.scene_name,
            "remark": self.remark,
        }


def _png_stems(directory: Path) -> set[str]:
    """Return frame ids for PNG files in one RGB-D stream directory."""
    path = Path(directory)
    if not path.is_dir():
        return set()
    return {item.stem for item in path.glob("*.png") if item.is_file()}


def inspect_scene_integrity(scene_dir: Path) -> SceneIntegrityReport:
    """Inspect RGB/depth pairing and metadata without modifying the scene."""
    scene = Path(scene_dir).expanduser().resolve()
    rgb_stems = _png_stems(scene / "rgb")
    depth_stems = _png_stems(scene / "depth")
    orphan_rgb = tuple(scene / "rgb" / f"{stem}.png" for stem in sorted(rgb_stems - depth_stems))
    orphan_depth = tuple(scene / "depth" / f"{stem}.png" for stem in sorted(depth_stems - rgb_stems))

    metadata_ids: set[str] = set()
    metadata_error: str | None = None
    metadata_path = scene / "metadata.json"
    if metadata_path.is_file():
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            frames = data.get("frames") or []
            if not isinstance(frames, list):
                raise ValueError("metadata.frames 不是列表")
            for row in frames:
                if isinstance(row, dict) and row.get("id") is not None:
                    metadata_ids.add(str(row["id"]))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            metadata_error = f"metadata.json 无法读取：{exc}"
    elif orphan_rgb or orphan_depth:
        metadata_error = "metadata.json 不存在，无法证明孤立文件未被记录"

    unsafe: list[str] = []
    for frame_id in sorted(metadata_ids):
        if frame_id not in rgb_stems:
            unsafe.append(f"{frame_id}:rgb")
        if frame_id not in depth_stems:
            unsafe.append(f"{frame_id}:depth")

    safe_orphans: list[Path] = []
    if metadata_error is None:
        safe_orphans.extend(path for path in orphan_rgb if path.stem not in metadata_ids)
        safe_orphans.extend(path for path in orphan_depth if path.stem not in metadata_ids)

    return SceneIntegrityReport(
        scene_dir=scene,
        rgb_count=len(rgb_stems),
        depth_count=len(depth_stems),
        paired_count=len(rgb_stems & depth_stems),
        orphan_rgb=orphan_rgb,
        orphan_depth=orphan_depth,
        metadata_frame_ids=frozenset(metadata_ids),
        safe_orphans=tuple(sorted(safe_orphans, key=lambda path: str(path))),
        unsafe_recorded_missing=tuple(unsafe),
        metadata_error=metadata_error,
    )


def repair_scene_integrity(
    report: SceneIntegrityReport,
    *,
    quarantine_root: Path | None = None,
) -> Path:
    """Move proven-safe orphan files to a recoverable scene-local quarantine.

    The operation is transactional for file moves: if any move fails, files
    already moved during this call are returned to their original locations.
    No RGB-D or metadata file is permanently deleted.
    """
    scene = report.scene_dir.resolve()
    current = inspect_scene_integrity(scene)
    if not current.can_auto_repair:
        raise ValueError(f"该场次不能自动修复：{current.summary}")
    if current.safe_orphans != report.safe_orphans:
        raise RuntimeError("场次文件在检查后发生变化，请重新检查再修复")

    quarantine_base = (scene / ".integrity_quarantine").resolve()
    if quarantine_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target = quarantine_base / stamp
    else:
        target = Path(quarantine_root).expanduser().resolve()
    if not target.is_relative_to(quarantine_base):
        raise ValueError(f"隔离目录必须位于场次内部：{quarantine_base}")
    if target.exists():
        raise FileExistsError(f"隔离目录已存在，不覆盖：{target}")

    moves: list[tuple[Path, Path]] = []
    for source in current.safe_orphans:
        stream = source.parent.name
        destination = target / stream / source.name
        if destination.exists():
            raise FileExistsError(f"隔离目标已存在，不覆盖：{destination}")
        moves.append((source, destination))

    completed: list[tuple[Path, Path]] = []
    try:
        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            completed.append((source, destination))
    except Exception as exc:
        rollback_errors: list[str] = []
        for source, destination in reversed(completed):
            try:
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(source))
            except Exception as rollback_exc:  # pragma: no cover - catastrophic filesystem failure
                rollback_errors.append(f"{destination} -> {source}: {rollback_exc}")
        if target.exists():
            try:
                shutil.rmtree(target)
            except OSError:
                pass
        if rollback_errors:
            raise RuntimeError(
                f"完整性修复失败，且回滚不完整：{exc}；{'；'.join(rollback_errors)}"
            ) from exc
        raise RuntimeError(f"完整性修复失败，已回滚：{exc}") from exc

    repaired = inspect_scene_integrity(scene)
    if not repaired.is_complete:
        # This should not occur after a stable preflight; restore rather than
        # leaving a silently inconsistent formal scene.
        rollback_errors: list[str] = []
        for source, destination in reversed(completed):
            try:
                shutil.move(str(destination), str(source))
            except Exception as rollback_exc:  # pragma: no cover
                rollback_errors.append(str(rollback_exc))
        if not rollback_errors and target.exists():
            shutil.rmtree(target)
        suffix = f"；回滚错误：{'；'.join(rollback_errors)}" if rollback_errors else "，已回滚"
        raise RuntimeError(f"修复后复检仍不完整{suffix}")
    return target


def paired_rgbd_frame_count(scene_dir: Path) -> int:
    """Count frame ids that have both an RGB PNG and a depth PNG."""
    root = Path(scene_dir)
    return len(_png_stems(root / "rgb") & _png_stems(root / "depth"))


def has_paired_rgbd_frames(scene_dir: Path) -> bool:
    """Return whether a capture contains at least one complete RGB/depth pair."""
    return paired_rgbd_frame_count(scene_dir) > 0


def discard_failed_empty_capture(session: CaptureSession) -> bool:
    """Remove failed staging output only when neither stream contains a PNG.

    A one-sided or otherwise partial capture is deliberately retained for
    diagnosis; only the common empty rgb/depth directory skeleton is removed.
    """
    staging = session.staging_dir
    if not staging.exists():
        return False
    rgb_frames = _png_stems(staging / "rgb")
    depth_frames = _png_stems(staging / "depth")
    if rgb_frames or depth_frames:
        return False
    session.discard()
    return True


def load_object_classes(config_path: Path) -> tuple[ObjectClass, ...]:
    """Load the numeric class order and Chinese display names from YAML."""
    data = yaml.safe_load(Path(config_path).expanduser().read_text(encoding="utf-8")) or {}
    raw_classes = data.get("classes", {})
    objects = data.get("objects", {})
    result: list[ObjectClass] = []
    for raw_id, name in raw_classes.items():
        class_id = int(raw_id)
        object_info = objects.get(name, {}) or {}
        chinese_name = str(object_info.get("chinese_name", name))
        result.append(ObjectClass(class_id, str(name), chinese_name))
    result.sort(key=lambda item: item.class_id)
    if not result:
        raise ValueError(f"配置中没有classes: {config_path}")
    return tuple(result)


def scene_for_number(classes: Sequence[ObjectClass], number: int) -> ObjectClass:
    if number < 1 or number > len(classes):
        raise ValueError(f"类别按键必须在1到{len(classes)}之间: {number}")
    return classes[number - 1]


def _next_scene_number(project_root: Path, class_name: str, timestamp: str) -> int:
    pattern = re.compile(rf"^{re.escape(class_name)}_{re.escape(timestamp)}_(\d+)$")
    candidates = []
    for parent in (project_root / "data" / "scenes" / class_name, project_root / "data" / ".staging"):
        if not parent.exists():
            continue
        for path in parent.iterdir():
            match = pattern.match(path.name)
            if match:
                candidates.append(int(match.group(1)))
    return max(candidates, default=0) + 1


def make_capture_session(
    project_root: Path,
    object_class: ObjectClass,
    split: str,
    *,
    now: datetime | None = None,
    scene_name: str | None = None,
    remark: str = "",
) -> CaptureSession:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split必须是train/val/test: {split}")
    root = Path(project_root).expanduser().resolve()
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M")
    if scene_name is None:
        sequence = _next_scene_number(root, object_class.name, timestamp)
        scene_name = f"{object_class.name}_{timestamp}_{sequence:02d}"
    staging = root / "data" / ".staging" / scene_name
    scene = root / "data" / "scenes" / object_class.name / scene_name
    return CaptureSession(
        project_root=root,
        object_class=object_class,
        split=split,
        scene_name=scene_name,
        staging_dir=staging,
        scene_dir=scene,
        capture_session_id=f"session_{scene_name}",
        source_video_id=f"{scene_name}_clip_01",
        remark=remark,
    )


def write_session_metadata(session: CaptureSession, scene_dir: Path | None = None) -> Path:
    """Write a small app-owned metadata file without replacing sensor metadata."""
    target_dir = (scene_dir or session.scene_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "atec_capture_session.json"
    path.write_text(json.dumps(session.metadata(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def manifest_for_session(session: CaptureSession) -> Path:
    return session.project_root / "manifests" / f"{session.scene_name}_{session.split}.yaml"
