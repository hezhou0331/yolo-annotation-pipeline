"""Safe planning and reporting for the App's one-click scene processor."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import copy
import json
import math
import os
from pathlib import Path
import shutil
from typing import Iterable
import yaml

from .gui_state import (
    SceneWorkflowState,
    inspect_scene_integrity,
    repair_scene_integrity,
    scene_workflow_state,
)

_ALLOWED_CLASSES = {
    "can", "watermelon_rind", "meal_box", "red_paper_bag",
    "blue_bin", "green_bin", "red_bin",
}
_ALLOWED_SPLITS = {"train", "val", "test"}
_CLASS_IDS = {
    "can": 0, "watermelon_rind": 1, "meal_box": 2, "red_paper_bag": 3,
    "blue_bin": 4, "green_bin": 5, "red_bin": 6,
}


@dataclass(frozen=True)
class AutoScenePlan:
    """The next safe automatic action for one scene."""

    scene_name: str
    scene_dir: Path
    class_name: str
    split: str
    manifest_path: Path | None
    action: str
    stage: str
    reason: str
    quarantine_path: Path | None = None
    capture_session_id: str | None = None
    source_video_id: str | None = None
    manifest_backup_path: Path | None = None


@dataclass(frozen=True)
class AutoBatchRecord:
    """One terminal or checkpoint result included in a batch report."""

    scene_name: str
    stage: str
    status: str
    reason: str
    exit_code: int | None = None
    quarantine_path: str | None = None
    manifest_path: str | None = None
    manifest_backup_path: str | None = None


def _manual_plan(
    state: SceneWorkflowState,
    reason: str,
    *,
    quarantine_path: Path | None = None,
) -> AutoScenePlan:
    return AutoScenePlan(
        scene_name=state.scene_name,
        scene_dir=state.scene_dir,
        class_name=state.class_name,
        split=state.split,
        manifest_path=state.manifest_path,
        action="manual",
        stage="preflight",
        reason=reason,
        quarantine_path=quarantine_path,
    )


def _portable(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), base.resolve())


def _resolved_manifest_path(manifest: Path, value: object) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def _validate_or_repair_manifest(
    project_root: Path, state: SceneWorkflowState,
) -> tuple[Path | None, tuple[str, ...], str | None]:
    """Validate an existing Manifest and repair only metadata-proven fields."""
    manifest = state.manifest_path
    if manifest is None:
        return None, (), None
    try:
        loaded = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except (OSError, TypeError, yaml.YAMLError) as exc:
        return None, (), f"Manifest 无法读取：{exc}"
    if not isinstance(loaded, dict):
        return None, (), "Manifest 顶层必须是映射"
    data = copy.deepcopy(loaded)
    metadata, _metadata_error = _load_init_metadata(state)
    can_repair = metadata is not None
    repairs: list[str] = []

    project = data.get("project")
    if not isinstance(project, dict):
        return None, (), "Manifest.project 必须是映射"
    expected_scene = state.scene_dir.resolve()
    expected_output = (Path(project_root).resolve() / "datasets/atec_yolo11_seg").resolve()
    path_fields = {"scene": expected_scene, "output": expected_output}
    for key, expected in path_fields.items():
        if _resolved_manifest_path(manifest, project.get(key)) == expected:
            continue
        if not can_repair:
            return None, (), f"Manifest.project.{key} 缺失或路径错误，且采集元数据不足以安全修复"
        project[key] = _portable(manifest.parent, expected)
        repairs.append(f"project.{key}")

    expected_scalars = {
        "split": metadata["split"] if metadata else state.split,
        "capture_session_id": metadata["capture_session_id"] if metadata else str(project.get("capture_session_id") or ""),
        "source_video_id": metadata["source_video_id"] if metadata else str(project.get("source_video_id") or ""),
        "name_prefix": f"{state.scene_name}_",
    }
    for key, expected in expected_scalars.items():
        current = str(project.get(key) or "").strip()
        if current and current == expected:
            continue
        if not expected or not can_repair:
            return None, (), f"Manifest.project.{key} 缺失或与采集元数据不一致"
        project[key] = expected
        repairs.append(f"project.{key}")

    classes = data.get("classes")
    if not isinstance(classes, dict):
        if not can_repair:
            return None, (), "Manifest.classes 缺失或格式错误"
        classes = {}
        data["classes"] = classes
        repairs.append("classes")
    expected_class_id = _CLASS_IDS[state.class_name]
    class_value = classes.get(expected_class_id, classes.get(str(expected_class_id)))
    if class_value != state.class_name:
        if not can_repair:
            return None, (), f"Manifest.classes 缺少当前类别 {state.class_name}"
        classes[expected_class_id] = state.class_name
        classes.pop(str(expected_class_id), None)
        repairs.append(f"classes.{expected_class_id}")

    instances = data.get("instances")
    if not isinstance(instances, list) or not instances:
        return None, (), "Manifest.instances 必须包含至少一个实例"
    seen_ids: set[str] = set()
    class_names_by_id = {class_id: name for name, class_id in _CLASS_IDS.items()}
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            return None, (), f"Manifest.instances[{index}] 不是映射"
        instance_id = str(instance.get("instance_id") or "").strip()
        if not instance_id or instance_id in seen_ids:
            return None, (), "Manifest 实例编号缺失或重复，不能自动猜测"
        seen_ids.add(instance_id)

        raw_name = str(instance.get("class_name") or "").strip()
        raw_id = instance.get("class_id")
        instance_class: str | None = raw_name if raw_name in _CLASS_IDS else None
        if raw_name and instance_class is None:
            return None, (), f"Manifest 实例 {instance_id} 的 class_name 无效：{raw_name}"
        parsed_id: int | None = None
        if raw_id is not None and str(raw_id).strip() != "":
            try:
                parsed_id = int(raw_id)
            except (TypeError, ValueError):
                return None, (), f"Manifest 实例 {instance_id} 的 class_id 无效：{raw_id}"
            id_class = class_names_by_id.get(parsed_id)
            if id_class is None:
                return None, (), f"Manifest 实例 {instance_id} 的 class_id 超出7类范围：{parsed_id}"
            if instance_class is not None and instance_class != id_class:
                return None, (), f"Manifest 实例 {instance_id} 的 class_id/class_name 冲突，需人工确认"
            instance_class = id_class
        if instance_class is None:
            matches = [name for name in _CLASS_IDS if instance_id == name or instance_id.startswith(f"{name}_")]
            if len(matches) == 1:
                instance_class = matches[0]
            elif len(instances) == 1:
                instance_class = state.class_name
            else:
                return None, (), f"Manifest 实例 {instance_id} 缺少类别字段，且混合场景中不能自动猜测"
        expected_id = _CLASS_IDS[instance_class]
        if not raw_name:
            if not can_repair:
                return None, (), f"Manifest 实例 {instance_id} 缺少 class_name"
            instance["class_name"] = instance_class
            repairs.append(f"instances.{instance_id}.class_name")
        if parsed_id is None:
            if not can_repair:
                return None, (), f"Manifest 实例 {instance_id} 缺少 class_id"
            instance["class_id"] = expected_id
            repairs.append(f"instances.{instance_id}.class_id")

        mapped_class = classes.get(expected_id, classes.get(str(expected_id)))
        if mapped_class != instance_class:
            if not can_repair:
                return None, (), f"Manifest.classes 缺少实例类别 {instance_class}"
            classes[expected_id] = instance_class
            classes.pop(str(expected_id), None)
            repairs.append(f"classes.{expected_id}")

        tracker = str(instance.get("tracker") or "").strip()
        if tracker not in {"sam2", "foundationpose", "mask_sequence"}:
            return None, (), f"Manifest 实例 {instance_id} 的 tracker 无效：{tracker or '缺失'}"
        if tracker == "sam2" and not str(instance.get("key_mask_dir") or "").strip():
            if not can_repair:
                return None, (), f"Manifest 实例 {instance_id} 缺少 key_mask_dir"
            instance["key_mask_dir"] = _portable(
                manifest.parent, Path(project_root) / "data/key_masks" / state.scene_name / instance_id,
            )
            repairs.append(f"instances.{instance_id}.key_mask_dir")

    if not repairs:
        return None, (), None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    backup = manifest.with_name(f"{manifest.name}.auto_processing_backup_{stamp}")
    temporary = manifest.with_name(f".{manifest.name}.auto_processing.tmp")
    try:
        shutil.copy2(manifest, backup)
        temporary.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        temporary.replace(manifest)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        return backup if backup.exists() else None, (), f"Manifest 安全修复写入失败：{exc}"
    return backup, tuple(repairs), None


def _load_init_metadata(state: SceneWorkflowState) -> tuple[dict[str, str] | None, str | None]:
    path = state.scene_dir / "atec_capture_session.json"
    if not path.is_file():
        return None, "缺少 atec_capture_session.json，不能无歧义补建 Manifest"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return None, f"采集会话元数据无法读取：{exc}"
    if not isinstance(raw, dict):
        return None, "采集会话元数据格式错误"
    values = {key: str(raw.get(key, "")).strip() for key in (
        "scene_name", "class_name", "split", "capture_session_id", "source_video_id",
    )}
    missing = [key for key, value in values.items() if not value]
    if missing:
        return None, f"采集会话元数据缺少字段：{', '.join(missing)}"
    if values["scene_name"] != state.scene_name:
        return None, "采集会话 scene_name 与场景目录不一致"
    if values["class_name"] != state.class_name:
        return None, "采集会话 class_name 与类别目录不一致"
    if values["class_name"] not in _ALLOWED_CLASSES:
        return None, f"未知类别，不能自动补建 Manifest：{values['class_name']}"
    if values["split"] not in _ALLOWED_SPLITS:
        return None, f"未知 split，不能自动补建 Manifest：{values['split']}"
    return values, None


def _timestamp_integrity_error(scene_dir: Path) -> str | None:
    """Return a conservative metadata/timestamp error for one paired sequence."""
    scene = Path(scene_dir).expanduser().resolve()
    metadata_path = scene / "metadata.json"
    if not metadata_path.is_file():
        return "缺少 metadata.json，无法检查 RGB/Depth 时间戳"
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        frames = data.get("frames")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return f"metadata.json 无法读取，不能检查时间戳：{exc}"
    if not isinstance(frames, list):
        return "metadata.frames 不是列表，不能检查时间戳"

    rows: dict[str, dict[str, object]] = {}
    duplicates: list[str] = []
    for raw in frames:
        if not isinstance(raw, dict) or raw.get("id") is None:
            return "metadata.frames 存在缺少 id 的记录，不能检查时间戳"
        frame_id = str(raw["id"])
        if frame_id in rows:
            duplicates.append(frame_id)
        rows[frame_id] = raw
    if duplicates:
        return f"metadata.frames 存在重复帧 ID：{', '.join(sorted(set(duplicates)))}"

    rgb_ids = {path.stem for path in (scene / "rgb").glob("*.png") if path.is_file()}
    depth_ids = {path.stem for path in (scene / "depth").glob("*.png") if path.is_file()}
    paired_ids = sorted(rgb_ids & depth_ids)
    previous: tuple[float, float] | None = None
    for frame_id in paired_ids:
        row = rows.get(frame_id)
        if row is None:
            return f"帧 {frame_id} 缺少 metadata 时间戳记录"
        values: list[float] = []
        for key in ("color_timestamp_ms", "depth_timestamp_ms"):
            raw_value = row.get(key)
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                return f"帧 {frame_id} 缺少或包含无效 {key} 时间戳"
            if not math.isfinite(value) or value < 0:
                return f"帧 {frame_id} 的 {key} 时间戳无效：{raw_value}"
            values.append(value)
        current = (values[0], values[1])
        if previous is not None and (current[0] < previous[0] or current[1] < previous[1]):
            return f"帧 {frame_id} 的 RGB/Depth 时间戳发生倒退"
        previous = current
    return None


def preflight_scene(project_root: Path, state: SceneWorkflowState) -> AutoScenePlan:
    """Inspect and safely repair one scene, then choose its next action.

    Only proven-safe one-sided files are moved.  All ambiguous conditions are
    represented as ``manual`` and are never silently passed to export.
    """
    root = Path(project_root).expanduser().resolve()
    current = state
    quarantine_path: Path | None = None
    integrity = inspect_scene_integrity(current.scene_dir)
    if not integrity.is_complete and integrity.can_auto_repair:
        quarantine_path = repair_scene_integrity(integrity)
        current = scene_workflow_state(root, current.scene_dir)
        integrity = inspect_scene_integrity(current.scene_dir)

    if not integrity.is_complete or current.paired_frames <= 0:
        return _manual_plan(current, integrity.summary, quarantine_path=quarantine_path)

    timestamp_error = _timestamp_integrity_error(current.scene_dir)
    if timestamp_error:
        return _manual_plan(current, timestamp_error, quarantine_path=quarantine_path)

    manifest_backup_path: Path | None = None
    manifest_repairs: tuple[str, ...] = ()
    if current.manifest_path is not None:
        manifest_backup_path, manifest_repairs, manifest_error = _validate_or_repair_manifest(root, current)
        if manifest_error:
            plan = _manual_plan(current, manifest_error, quarantine_path=quarantine_path)
            return replace(plan, manifest_backup_path=manifest_backup_path)
        if manifest_repairs:
            current = scene_workflow_state(root, current.scene_dir)

    if current.code == "manifest_missing":
        metadata, error = _load_init_metadata(current)
        if error or metadata is None:
            return _manual_plan(current, error or "Manifest 不能安全自动补建", quarantine_path=quarantine_path)
        expected_manifest = root / "manifests" / f"{current.scene_name}_{metadata['split']}.yaml"
        return AutoScenePlan(
            current.scene_name, current.scene_dir, current.class_name,
            metadata["split"], expected_manifest, "init", "init",
            "采集会话元数据完整，可安全补建 Manifest",
            quarantine_path, metadata["capture_session_id"], metadata["source_video_id"],
        )

    common = dict(
        scene_name=current.scene_name,
        scene_dir=current.scene_dir,
        class_name=current.class_name,
        split=current.split,
        manifest_path=current.manifest_path,
        quarantine_path=quarantine_path,
        manifest_backup_path=manifest_backup_path,
    )
    repair_note = f"已安全修复 Manifest：{', '.join(manifest_repairs)}；" if manifest_repairs else ""
    if current.code == "segments_missing":
        return AutoScenePlan(
            **common, action="segment", stage="segment",
            reason=repair_note + "RGB-D 与 Manifest 完整，待自动分段",
        )
    if current.code == "masks_missing":
        return AutoScenePlan(
            **common, action="manual", stage="key_masks",
            reason=(f"关键帧 Mask 不完整：{current.mask_completed}/{current.mask_total}；"
                    "必须人工补齐后才能传播"),
        )
    if current.code in {"pending_export", "export_failed"}:
        reason = "关键帧完整，待 SAM2 传播/质量检查/YOLO 导出"
        if current.code == "export_failed":
            reason = "上次导出 accepted=0，按断点结果重新尝试自动处理"
        return AutoScenePlan(**common, action="run", stage="run", reason=repair_note + reason)
    if current.code == "dataset_ready":
        return AutoScenePlan(
            **common, action="skip", stage="skip",
            reason=f"已有有效导出：accepted={current.accepted}",
        )
    if current.code == "export_needs_review":
        return AutoScenePlan(
            **common, action="manual", stage="review",
            reason=(f"已有 accepted={current.accepted}，但仍有 {current.review} 帧需要人工 Review；"
                    "不自动把 review 帧写入正式标签"),
        )
    return AutoScenePlan(
        **common, action="manual", stage="preflight",
        reason=f"未识别或不能自动推进的场景状态：{current.code}（{current.detail}）",
    )


def plan_auto_scenes(
    project_root: Path,
    states: Iterable[SceneWorkflowState],
) -> tuple[AutoScenePlan, ...]:
    """Preflight every discovered scene in stable caller-provided order."""
    return tuple(preflight_scene(project_root, state) for state in states)


def build_manifest_init_args(project_root: Path, plan: AutoScenePlan) -> list[str]:
    """Build the existing ``atec-pipeline init`` arguments for a verified plan."""
    if plan.action != "init" or not plan.capture_session_id or not plan.source_video_id:
        raise ValueError("该场景没有经过可自动补建 Manifest 的预检")
    return [
        "init", str(Path(project_root).expanduser().resolve()), plan.scene_name,
        "--split", plan.split,
        "--only-class", plan.class_name,
        "--scene-class", plan.class_name,
        "--capture-session-id", plan.capture_session_id,
        "--source-video-id", plan.source_video_id,
    ]


def scene_is_locked(batch_active: bool, current_scene: str | None, selected_scene: str | None) -> bool:
    """Only the scene currently written by the batch worker is read-only."""
    return bool(batch_active and current_scene and selected_scene == current_scene)


def _record_dict(record: AutoBatchRecord) -> dict[str, object]:
    return {key: value for key, value in asdict(record).items() if value is not None}


def _summary(records: Iterable[AutoBatchRecord]) -> dict[str, int]:
    counts = {"success": 0, "skipped": 0, "failed": 0, "manual": 0, "cancelled": 0}
    for record in records:
        status = record.status.lower()
        if status == "success":
            counts["success"] += 1
        elif status in {"skip", "skipped"} or (status == "completed" and record.stage == "skip"):
            counts["skipped"] += 1
        elif status == "failed":
            counts["failed"] += 1
        elif status == "manual":
            counts["manual"] += 1
        elif status == "cancelled":
            counts["cancelled"] += 1
    return counts


def write_batch_report(
    project_root: Path,
    records: Iterable[AutoBatchRecord],
    *,
    started_at: datetime,
    finished_at: datetime | None = None,
    cancelled: bool = False,
) -> Path:
    """Atomically write a timestamped report and refresh ``latest.json``."""
    root = Path(project_root).expanduser().resolve()
    report_dir = root / "project_reports" / "auto_processing"
    report_dir.mkdir(parents=True, exist_ok=True)
    finished = finished_at or datetime.now(timezone.utc)
    record_list = tuple(records)
    payload = {
        "schema_version": 1,
        "started_at": started_at.astimezone(timezone.utc).isoformat(),
        "finished_at": finished.astimezone(timezone.utc).isoformat(),
        "cancelled": bool(cancelled),
        "summary": _summary(record_list),
        "scenes": [_record_dict(record) for record in record_list],
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    stamp = started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report_path = report_dir / f"{stamp}.json"
    for destination in (report_path, report_dir / "latest.json"):
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(destination)
    return report_path
