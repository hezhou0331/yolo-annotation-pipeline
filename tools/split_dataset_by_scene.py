#!/usr/bin/env python3
"""Split exported YOLO data by complete scene/source video, never by frames.

The default mode is a dry-run.  ``--apply`` moves selected train image/label
pairs to val, updates their project reports, and keeps the operation
leakage-safe by grouping reports by source_video_id/capture_session_id/scene.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import yaml

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from atec_pipeline.path_compat import infer_project_root, portable_path, resolve_compatible_path

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
ACCEPTED_STATUSES = {"accepted"}
REPORT_TRAIN_SUFFIX = "_train_report.json"
REPORT_VAL_SUFFIX = "_val_report.json"
STAGING_SPLIT_PARENTS = ("images", "labels", "quality_reports", "rendered_masks", "visualizations")


@dataclass
class ReportRecord:
    path: Path
    scene: str
    split: str
    capture_session_id: str | None
    source_video_id: str | None
    output_ids: list[str]
    class_names: set[str] = field(default_factory=set)
    stage_paths: tuple[Path, ...] = ()


@dataclass
class SceneGroup:
    key: str
    scene_names: set[str] = field(default_factory=set)
    capture_session_ids: set[str] = field(default_factory=set)
    source_video_ids: set[str] = field(default_factory=set)
    records: list[ReportRecord] = field(default_factory=list)

    @property
    def source_video_id(self) -> str | None:
        return next(iter(self.source_video_ids), None)

    @property
    def capture_session_id(self) -> str | None:
        return next(iter(self.capture_session_ids), None)

    @property
    def accepted_count(self) -> int:
        return sum(len(record.output_ids) for record in self.records)

    @property
    def class_names(self) -> set[str]:
        return {name for record in self.records for name in record.class_names}

    @property
    def newest_scene_name(self) -> str:
        return max(self.scene_names, default="")


@dataclass
class FileMove:
    image_src: Path
    image_dst: Path
    label_src: Path
    label_dst: Path


@dataclass(frozen=True)
class FileBackup:
    content: bytes
    mode: int
    atime_ns: int
    mtime_ns: int


def _load_dataset(dataset: Path) -> tuple[Path, dict[str, Any]]:
    dataset = dataset.expanduser().resolve()
    if dataset.is_dir():
        dataset = dataset / "dataset.yaml"
    if not dataset.is_file():
        raise ValueError(f"dataset.yaml不存在: {dataset}")
    data = yaml.safe_load(dataset.read_text(encoding="utf-8")) or {}
    root = resolve_compatible_path(
        data.get("path", dataset.parent),
        base=dataset.parent,
        repository_root=WORKSPACE,
        project_root=infer_project_root(dataset, repository_root=WORKSPACE),
    )
    return root, data


def _report_files(root: Path) -> list[Path]:
    report_dir = root / "project_reports"
    return sorted(p for p in report_dir.glob("*_report.json") if p.is_file())


def _read_report(path: Path) -> ReportRecord | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("scene"):
        return None
    frames = data.get("frames", [])
    output_ids = []
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict) or frame.get("status") not in ACCEPTED_STATUSES:
                continue
            output_id = str(frame.get("output_id", "")).strip()
            if output_id:
                output_ids.append(output_id)
    class_names: set[str] = set()
    for instance in data.get("instances") or []:
        if isinstance(instance, dict) and str(instance.get("class_name", "")).strip():
            class_names.add(str(instance["class_name"]).strip())
    project_root = infer_project_root(path, repository_root=WORKSPACE)
    scene = resolve_compatible_path(
        str(data["scene"]),
        base=path.parent,
        repository_root=WORKSPACE,
        project_root=project_root,
    )
    stage_paths: set[Path] = set()
    for instance in data.get("instances") or []:
        if not isinstance(instance, dict) or not str(instance.get("stage") or "").strip():
            continue
        stage_paths.add(resolve_compatible_path(
            str(instance["stage"]),
            base=path.parent,
            repository_root=WORKSPACE,
            project_root=project_root,
        ))
    return ReportRecord(
        path=path,
        scene=str(scene),
        split=str(data.get("split", "train")),
        capture_session_id=str(data["capture_session_id"]) if data.get("capture_session_id") else None,
        source_video_id=str(data["source_video_id"]) if data.get("source_video_id") else None,
        output_ids=sorted(set(output_ids)),
        class_names=class_names,
        stage_paths=tuple(sorted(stage_paths)),
    )


def _identifiers(record: ReportRecord) -> set[str]:
    identifiers = {f"scene:{Path(record.scene).expanduser().resolve()}"}
    if record.capture_session_id:
        identifiers.add(f"session:{record.capture_session_id}")
    if record.source_video_id:
        identifiers.add(f"video:{record.source_video_id}")
    return identifiers


def scan_groups(dataset: Path | str) -> list[SceneGroup]:
    root, _ = _load_dataset(Path(dataset))
    records = [record for path in _report_files(root) if (record := _read_report(path)) is not None]
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owner: dict[str, int] = {}
    for index, record in enumerate(records):
        for identifier in _identifiers(record):
            if identifier in owner:
                union(index, owner[identifier])
            else:
                owner[identifier] = index

    groups: dict[int, SceneGroup] = {}
    for index, record in enumerate(records):
        root_index = find(index)
        key = min(_identifiers(records[root_index]))
        group = groups.setdefault(key, SceneGroup(key=key))
        group.records.append(record)
        group.scene_names.add(Path(record.scene).name)
        if record.capture_session_id:
            group.capture_session_ids.add(record.capture_session_id)
        if record.source_video_id:
            group.source_video_ids.add(record.source_video_id)
    return sorted(groups.values(), key=lambda item: item.key)


def _find_image(image_dir: Path, output_id: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{output_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _selected_groups(groups: list[SceneGroup], scenes: list[str], videos: list[str]) -> list[SceneGroup]:
    scenes_set = set(scenes)
    videos_set = set(videos)
    selected = [
        group for group in groups
        if group.scene_names & scenes_set or group.source_video_ids & videos_set
    ]
    if not selected:
        available = sorted({name for group in groups for name in group.scene_names})
        available_videos = sorted({video for group in groups for video in group.source_video_ids})
        raise ValueError(
            f"未找到指定场次/采集视频；可用场次={available}，可用source_video_id={available_videos}"
        )
    return selected


def _project_root_for_scene(scene: Path) -> Path | None:
    for ancestor in scene.parents:
        if ancestor.name == "data":
            return ancestor.parent
    return None


def _validated_manifest_pair(scene: Path) -> tuple[Path, Path, Path]:
    """Return the selected scene's project root and exact train/val manifests."""
    project_root = _project_root_for_scene(scene)
    if project_root is None:
        raise ValueError(f"无法从选中场次解析project_root，拒绝切分: {scene}")

    old_manifest = project_root / "manifests" / f"{scene.name}_train.yaml"
    new_manifest = project_root / "manifests" / f"{scene.name}_val.yaml"
    if old_manifest.is_symlink() or not old_manifest.is_file():
        raise ValueError(f"选中场次train Manifest不存在或不是普通文件: {old_manifest}")
    if new_manifest.exists() or new_manifest.is_symlink():
        raise ValueError(f"val Manifest目标已存在，拒绝覆盖: {new_manifest}")

    try:
        manifest = yaml.safe_load(old_manifest.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"train Manifest无法读取或不是有效YAML: {old_manifest}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"train Manifest必须是YAML对象: {old_manifest}")
    project = manifest.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"train Manifest project必须是对象: {old_manifest}")
    if project.get("split") != "train":
        raise ValueError(f"train Manifest project.split必须严格为train: {old_manifest}")
    scene_value = project.get("scene")
    if not isinstance(scene_value, str) or not scene_value.strip():
        raise ValueError(f"train Manifest project.scene缺失或无效: {old_manifest}")
    try:
        manifest_scene = resolve_compatible_path(
            scene_value,
            base=old_manifest.parent,
            repository_root=WORKSPACE,
            project_root=project_root,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"train Manifest project.scene无法解析: {old_manifest}: {exc}") from exc
    if manifest_scene != scene:
        raise ValueError(
            f"train Manifest project.scene与选中场次不一致: {old_manifest}; "
            f"manifest={manifest_scene}, selected={scene}"
        )
    return project_root, old_manifest, new_manifest


def _val_report_path(report: Path) -> Path:
    if not report.name.endswith(REPORT_TRAIN_SUFFIX):
        raise ValueError(f"train导出报告文件名不规范，拒绝留下split残留: {report}")
    prefix = report.name[:-len(REPORT_TRAIN_SUFFIX)]
    return report.with_name(f"{prefix}{REPORT_VAL_SUFFIX}")


def _valid_review_marker(marker: Path, scene: Path, report: Path, project_root: Path) -> bool:
    """Return whether a marker is bound to the exact pre-split report version."""
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        if data.get("scene") != scene.name or data.get("class_name") != scene.parent.name:
            return False
        recorded_report = resolve_compatible_path(
            str(data["export_report"]),
            base=marker.parent,
            repository_root=WORKSPACE,
            project_root=project_root,
        )
        return (
            recorded_report == report.resolve()
            and int(data["export_report_mtime_ns"]) == report.stat().st_mtime_ns
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _formal_train_files(image_dir: Path, label_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError(f"正式train图片/标签目录不存在: {image_dir} / {label_dir}")
    images: dict[str, Path] = {}
    duplicate_images: list[str] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in images:
            duplicate_images.append(path.stem)
        else:
            images[path.stem] = path
    if duplicate_images:
        raise ValueError(f"正式train存在同stem多图片: {sorted(set(duplicate_images))[:10]}")
    labels = {path.stem: path for path in sorted(label_dir.glob("*.txt")) if path.is_file()}
    missing_labels = sorted(set(images) - set(labels))
    orphan_labels = sorted(set(labels) - set(images))
    if missing_labels or orphan_labels:
        raise ValueError(
            "正式train图片/标签不成对: "
            f"缺标签={missing_labels[:10]}，孤立标签={orphan_labels[:10]}"
        )
    return images, labels


def build_plan(dataset: Path | str, scenes: list[str], videos: list[str]) -> dict[str, Any]:
    dataset_path = Path(dataset)
    root, data = _load_dataset(dataset_path)
    groups = scan_groups(root)
    selected = _selected_groups(groups, scenes, videos)
    train_images = root / str(data.get("train", "images/train"))
    train_labels = root / "labels" / Path(str(data.get("train", "images/train"))).name
    val_images = root / str(data.get("val", "images/val"))
    val_labels = root / "labels" / Path(str(data.get("val", "images/val"))).name
    all_records = [record for group in groups for record in group.records]
    output_owners: dict[str, tuple[Path, Path]] = {}
    for record in all_records:
        owner = (record.path.resolve(), Path(record.scene).expanduser().resolve())
        for output_id in record.output_ids:
            previous = output_owners.get(output_id)
            if previous is not None and previous != owner:
                raise ValueError(
                    f"accepted output_id被不同report/scene重复认领: {output_id}; "
                    f"{previous[0]} ({previous[1]}) vs {owner[0]} ({owner[1]})"
                )
            output_owners[output_id] = owner
    train_image_files, train_label_files = _formal_train_files(train_images, train_labels)

    selected_records_by_scene: dict[Path, list[ReportRecord]] = {}
    for group in selected:
        for record in group.records:
            scene = Path(record.scene).expanduser().resolve()
            selected_records_by_scene.setdefault(scene, []).append(record)
    for scene, records in selected_records_by_scene.items():
        expected = {output_id for record in records for output_id in record.output_ids}
        prefix = f"{scene.name}_"
        actual_images = {stem for stem in train_image_files if stem.startswith(prefix)}
        actual_labels = {stem for stem in train_label_files if stem.startswith(prefix)}
        if actual_images != expected or actual_labels != expected:
            extra = sorted((actual_images | actual_labels) - expected)
            missing = sorted(expected - (actual_images & actual_labels))
            raise ValueError(
                f"场次正式train输出与report accepted集合不一致: {scene.name}; "
                f"extra={extra[:10]}，missing={missing[:10]}"
            )
    moves: list[tuple[Path, Path, Path, Path]] = []
    missing: list[str] = []
    report_paths: list[Path] = []
    selected_keys = {group.key for group in selected}
    for group in selected:
        if any(record.split != "train" for record in group.records):
            raise ValueError(f"选中的组不是纯train来源，拒绝混合切分: {group.key}")
        report_paths.extend(record.path for record in group.records)
        for record in group.records:
            for output_id in record.output_ids:
                image = _find_image(train_images, output_id)
                label = train_labels / f"{output_id}.txt"
                if image is None or not label.is_file():
                    missing.append(output_id)
                    continue
                moves.append((image, val_images / image.name, label, val_labels / label.name))
    if missing:
        raise ValueError(f"选中场次存在缺失的已接受图片/标签，拒绝部分切分: {sorted(set(missing))[:10]}")
    if not moves:
        raise ValueError("选中场次没有已导出的可用图片；请先完成Mask、SAM2传播和YOLO导出")
    source_names = {src.name for src, _, _, _ in moves}
    destination_names = {dst.name for _, dst, _, _ in moves}
    if len(source_names) != len(moves) or len(destination_names) != len(moves):
        raise ValueError("场次之间存在重复output_id，拒绝切分")
    conflicts = [str(dst) for _, dst, _, _ in moves if dst.exists()]
    if conflicts:
        raise ValueError(f"val目标已有文件，拒绝覆盖: {conflicts[:10]}")
    manifest_updates: set[tuple[Path, Path]] = set()
    manifest_by_scene: dict[Path, tuple[Path, Path]] = {}
    project_by_scene: dict[Path, Path] = {}
    session_metadata: list[Path] = []
    for scene in sorted(selected_records_by_scene):
        metadata = scene / "atec_capture_session.json"
        if metadata.is_file():
            session_metadata.append(metadata)
        project_root, old_manifest, new_manifest = _validated_manifest_pair(scene)
        project_by_scene[scene] = project_root
        manifest_updates.add((old_manifest, new_manifest))
        manifest_by_scene[scene] = (old_manifest, new_manifest)

    report_updates: list[tuple[Path, Path, Path, Path]] = []
    report_sidecar_updates: set[tuple[Path, Path]] = set()
    segments_updates: set[tuple[Path, Path, Path]] = set()
    review_marker_updates: set[tuple[Path, Path, Path, Path, Path]] = set()
    staging_updates: set[tuple[Path, Path]] = set()
    staging_metadata: set[Path] = set()
    staging_root = (root / "_staging").resolve()
    for group in selected:
        for record in group.records:
            scene = Path(record.scene).expanduser().resolve()
            project_root = project_by_scene[scene]
            new_manifest = manifest_by_scene[scene][1]
            new_report = _val_report_path(record.path)
            if new_report.exists():
                raise ValueError(f"val导出报告目标已存在，拒绝覆盖: {new_report}")
            report_updates.append((record.path, new_report, new_manifest, project_root))

            old_sidecar = record.path.with_suffix(".csv")
            new_sidecar = new_report.with_suffix(".csv")
            if new_sidecar.exists():
                raise ValueError(f"val导出报告CSV目标已存在，拒绝覆盖: {new_sidecar}")
            if old_sidecar.is_file():
                report_sidecar_updates.add((old_sidecar, new_sidecar))

            segments = scene / "project_reports" / "segments.json"
            if segments.is_file():
                segments_updates.add((segments, new_manifest, project_root))

            marker = scene / "project_reports" / "manual_review_complete.json"
            if marker.is_file() and _valid_review_marker(marker, scene, record.path, project_root):
                review_marker_updates.add((
                    marker, record.path, new_report, scene, project_root,
                ))

            for stage in record.stage_paths:
                resolved_stage = stage.resolve()
                try:
                    resolved_stage.relative_to(staging_root)
                except ValueError as exc:
                    raise ValueError(f"导出报告stage不在数据集_staging内，拒绝移动: {stage}") from exc
                if not resolved_stage.exists():
                    continue
                for parent_name in STAGING_SPLIT_PARENTS:
                    source = resolved_stage / parent_name / "train"
                    destination = resolved_stage / parent_name / "val"
                    if destination.exists():
                        raise ValueError(f"staging val目标已存在，拒绝覆盖: {destination}")
                    if source.is_dir():
                        staging_updates.add((source, destination))
                        if parent_name == "quality_reports":
                            staging_metadata.update(
                                path for path in source.rglob("quality_report.json") if path.is_file()
                            )
                    elif source.exists():
                        raise ValueError(f"staging split来源不是目录: {source}")
    return {
        "dataset": root,
        "dataset_yaml": (dataset_path if dataset_path.is_file() else root / "dataset.yaml").resolve(),
        "groups": selected,
        "files": moves,
        "reports": sorted(set(report_paths)),
        "selected_keys": selected_keys,
        "val_images": val_images,
        "val_labels": val_labels,
        "manifest_updates": sorted(manifest_updates),
        "session_metadata": sorted(set(session_metadata)),
        "report_updates": sorted(report_updates, key=lambda item: str(item[0])),
        "report_sidecar_updates": sorted(report_sidecar_updates),
        "segments_updates": sorted(segments_updates),
        "review_marker_updates": sorted(review_marker_updates),
        "staging_updates": sorted(staging_updates),
        "staging_metadata": sorted(staging_metadata),
    }


def build_auto_plan(dataset: Path | str, target_ratio: float = 0.20) -> dict[str, Any]:
    """Select deterministic, class-stratified whole groups for validation."""
    if not 0.0 < target_ratio < 1.0:
        raise ValueError("target_ratio必须在0到1之间")
    groups = scan_groups(dataset)
    train_groups = [
        group for group in groups
        if group.accepted_count > 0 and group.class_names and all(record.split == "train" for record in group.records)
    ]
    existing_val_classes = {
        name for group in groups for record in group.records if record.split == "val" for name in record.class_names
    }
    classes = sorted({name for group in train_groups for name in group.class_names})
    selected: set[str] = set()
    insufficient: list[str] = []

    for class_name in classes:
        if class_name in existing_val_classes:
            continue
        class_groups = [group for group in train_groups if class_name in group.class_names]
        if len(class_groups) < 2:
            insufficient.append(class_name)
            continue
        if any(group.key in selected for group in class_groups):
            continue
        # First satisfy class stratification with the complete scene whose
        # accepted-frame count is closest to this class's target share. Sort
        # newest-first before ``min`` so equally good candidates deterministically
        # prefer the newer independent capture without allowing recency alone to
        # select a huge scene that badly overshoots the requested ratio.
        candidates = sorted(
            class_groups, key=lambda group: (group.newest_scene_name, group.key), reverse=True
        )
        viable = [
            candidate for candidate in candidates
            if any(
                other.key not in selected | {candidate.key}
                for other in class_groups
            )
        ]
        if not viable:
            insufficient.append(class_name)
            continue
        class_target = sum(group.accepted_count for group in class_groups) * target_ratio
        candidate = min(viable, key=lambda group: abs(group.accepted_count - class_target))
        selected.add(candidate.key)

    if not selected:
        if existing_val_classes:
            return {
                "dataset": _load_dataset(Path(dataset))[0],
                "dataset_yaml": (Path(dataset).resolve() if Path(dataset).is_file() else _load_dataset(Path(dataset))[0] / "dataset.yaml"),
                "groups": [], "files": [], "reports": [], "selected_keys": set(),
                "val_images": _load_dataset(Path(dataset))[0] / "images/val",
                "val_labels": _load_dataset(Path(dataset))[0] / "labels/val",
                "manifest_updates": [], "session_metadata": [],
                "report_updates": [], "report_sidecar_updates": [],
                "segments_updates": [], "review_marker_updates": [],
                "staging_updates": [], "staging_metadata": [],
                "auto_summary": {"target_ratio": target_ratio, "insufficient_classes": insufficient},
            }
        details = "、".join(insufficient or classes or ["没有有效类别"])
        raise ValueError(f"无法自动划分val：类别 {details} 至少两个独立有效场次")

    chosen = [group for group in train_groups if group.key in selected]
    # Mandatory one-per-class selection takes precedence. Add another newest
    # group only when it moves the global ratio closer to the requested target
    # and every affected class still keeps at least one complete train group.
    total = sum(group.accepted_count for group in train_groups)
    selected_frames = sum(group.accepted_count for group in chosen)
    for candidate in sorted(
        (group for group in train_groups if group.key not in selected),
        key=lambda group: (group.newest_scene_name, group.key), reverse=True,
    ):
        can_move = all(
            any(other.key not in selected | {candidate.key} and class_name in other.class_names for other in train_groups)
            for class_name in candidate.class_names
        )
        if not can_move:
            continue
        before = abs(selected_frames / max(1, total) - target_ratio)
        after = abs((selected_frames + candidate.accepted_count) / max(1, total) - target_ratio)
        if after < before:
            selected.add(candidate.key)
            selected_frames += candidate.accepted_count

    selected_scenes = sorted({name for group in train_groups if group.key in selected for name in group.scene_names})
    plan = build_plan(dataset, selected_scenes, [])
    plan["auto_summary"] = {
        "target_ratio": target_ratio,
        "selected_frames": len(plan["files"]),
        "train_candidate_frames": total,
        "selected_scenes": selected_scenes,
        "insufficient_classes": insufficient,
        "existing_val_classes": sorted(existing_val_classes),
    }
    return plan


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(text, encoding="utf-8")
        if existing_mode is not None:
            os.chmod(temp_path, existing_mode)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _backup_file(path: Path) -> FileBackup:
    stat = path.stat()
    return FileBackup(
        content=path.read_bytes(),
        mode=stat.st_mode,
        atime_ns=stat.st_atime_ns,
        mtime_ns=stat.st_mtime_ns,
    )


def _restore_file(path: Path, backup: FileBackup) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(backup.content)
    os.chmod(path, backup.mode & 0o7777)
    os.utime(path, ns=(backup.atime_ns, backup.mtime_ns))


def _mkdir_tracked(path: Path, created: list[Path]) -> None:
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"目标父路径不是目录: {path}")
        return
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    path.mkdir(parents=True, exist_ok=False)
    created.extend(reversed(missing))


def _move_tracked(source: Path, destination: Path, moved: list[tuple[Path, Path]], created: list[Path]) -> None:
    _mkdir_tracked(destination.parent, created)
    # Record the intent before entering shutil.move: an asynchronous
    # interruption may arrive after the underlying rename succeeds but before
    # this function regains control.  Rollback resolves the resulting state.
    moved.append((source, destination))
    shutil.move(str(source), str(destination))


def apply_plan(plan: dict[str, Any]) -> None:
    moves: list[tuple[Path, Path, Path, Path]] = plan["files"]
    dataset_yaml: Path = plan["dataset_yaml"]
    manifest_updates: list[tuple[Path, Path]] = plan.get("manifest_updates", [])
    session_metadata: list[Path] = plan.get("session_metadata", [])
    report_updates: list[tuple[Path, Path, Path, Path]] = plan.get("report_updates", [])
    report_sidecar_updates: list[tuple[Path, Path]] = plan.get("report_sidecar_updates", [])
    segments_updates: list[tuple[Path, Path, Path]] = plan.get("segments_updates", [])
    review_marker_updates: list[tuple[Path, Path, Path, Path, Path]] = plan.get(
        "review_marker_updates", []
    )
    staging_updates: list[tuple[Path, Path]] = plan.get("staging_updates", [])
    staging_metadata: list[Path] = plan.get("staging_metadata", [])
    # A second preflight protects against a destination appearing after planning.
    for image_src, image_dst, label_src, label_dst in moves:
        if not image_src.is_file() or not label_src.is_file():
            raise RuntimeError(f"源文件在执行前消失: {image_src} / {label_src}")
        if image_dst.exists() or label_dst.exists():
            raise RuntimeError(f"拒绝覆盖val文件: {image_dst} / {label_dst}")
    for old_manifest, new_manifest in manifest_updates:
        if not old_manifest.is_file():
            raise RuntimeError(f"Manifest在执行前消失: {old_manifest}")
        if new_manifest.exists():
            raise RuntimeError(f"拒绝覆盖val Manifest: {new_manifest}")
    for old_report, new_report, _new_manifest, _project_root in report_updates:
        if not old_report.is_file():
            raise RuntimeError(f"导出报告在执行前消失: {old_report}")
        if new_report.exists():
            raise RuntimeError(f"拒绝覆盖val导出报告: {new_report}")
    for source, destination in [*report_sidecar_updates, *staging_updates]:
        if not source.exists():
            raise RuntimeError(f"事务来源在执行前消失: {source}")
        if destination.exists():
            raise RuntimeError(f"拒绝覆盖事务目标: {destination}")
    for marker, old_report, _new_report, scene, project_root in review_marker_updates:
        if not _valid_review_marker(marker, scene, old_report, project_root):
            raise RuntimeError(f"人工Review标记在计划后发生变化，拒绝伪造完成状态: {marker}")

    mutable_files = sorted(set(
        [dataset_yaml]
        + session_metadata
        + [old for old, _ in manifest_updates]
        + [old for old, _new, _manifest, _root in report_updates]
        + [path for path, _manifest, _root in segments_updates]
        + [marker for marker, _old, _new, _scene, _root in review_marker_updates]
        + staging_metadata
    ))
    missing_mutable = [path for path in mutable_files if not path.is_file()]
    if missing_mutable:
        raise RuntimeError(f"事务元数据在执行前消失: {missing_mutable[:10]}")
    backups = {path: _backup_file(path) for path in mutable_files}
    moved_paths: list[tuple[Path, Path]] = []
    created_directories: list[Path] = []
    try:
        for directory in (plan["val_images"], plan["val_labels"]):
            _mkdir_tracked(directory, created_directories)
        for image_src, image_dst, label_src, label_dst in moves:
            _move_tracked(image_src, image_dst, moved_paths, created_directories)
            _move_tracked(label_src, label_dst, moved_paths, created_directories)
        for old_report, new_report, new_manifest, project_root in report_updates:
            data = json.loads(old_report.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"导出报告必须是JSON对象: {old_report}")
            data["split"] = "val"
            data["manifest"] = portable_path(
                new_manifest, relative_to=old_report.parent,
                repository_root=WORKSPACE, project_root=project_root,
            )
            _atomic_write_text(old_report, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            _move_tracked(old_report, new_report, moved_paths, created_directories)
        for old_sidecar, new_sidecar in report_sidecar_updates:
            _move_tracked(old_sidecar, new_sidecar, moved_paths, created_directories)
        for metadata_path in session_metadata:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"采集会话元数据必须是JSON对象: {metadata_path}")
            data["split"] = "val"
            _atomic_write_text(metadata_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        for segments_path, new_manifest, project_root in segments_updates:
            data = json.loads(segments_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"segments.json必须是JSON对象: {segments_path}")
            data["manifest"] = portable_path(
                new_manifest, relative_to=segments_path.parent,
                repository_root=WORKSPACE, project_root=project_root,
            )
            _atomic_write_text(segments_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        for metadata_path in staging_metadata:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("split") != "train":
                raise ValueError(f"staging质量报告split不是train: {metadata_path}")
            data["split"] = "val"
            _atomic_write_text(metadata_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        for source, destination in staging_updates:
            _move_tracked(source, destination, moved_paths, created_directories)
        for old_manifest, new_manifest in manifest_updates:
            data = yaml.safe_load(old_manifest.read_text(encoding="utf-8")) or {}
            project = data.setdefault("project", {})
            if not isinstance(project, dict):
                raise ValueError(f"Manifest project字段无效: {old_manifest}")
            project["split"] = "val"
            _atomic_write_text(old_manifest, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
            _move_tracked(old_manifest, new_manifest, moved_paths, created_directories)
        for marker, _old_report, new_report, _scene, project_root in review_marker_updates:
            data = json.loads(marker.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"人工Review标记必须是JSON对象: {marker}")
            data["export_report"] = portable_path(
                new_report, relative_to=marker.parent,
                repository_root=WORKSPACE, project_root=project_root,
            )
            data["export_report_mtime_ns"] = new_report.stat().st_mtime_ns
            _atomic_write_text(marker, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        _root, data = _load_dataset(dataset_yaml)
        data["val"] = "images/val"
        if data.get("test") == data["val"]:
            data.pop("test", None)
        _atomic_write_text(dataset_yaml, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    except BaseException as error:
        rollback_errors: list[str] = []
        for source, destination in reversed(moved_paths):
            try:
                source_exists = source.exists() or source.is_symlink()
                destination_exists = destination.exists() or destination.is_symlink()
                if source_exists and not destination_exists:
                    # The interruption happened before the move took effect.
                    continue
                if not source_exists and destination_exists:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(destination), str(source))
                    continue
                raise RuntimeError(
                    "移动intent处于不一致状态，无法安全回滚: "
                    f"source_exists={source_exists}, destination_exists={destination_exists}; "
                    f"{source} -> {destination}"
                )
            except BaseException as rollback_error:  # pragma: no cover - catastrophic filesystem failure
                rollback_errors.append(str(rollback_error))
        for path, backup in backups.items():
            try:
                _restore_file(path, backup)
            except BaseException as rollback_error:  # pragma: no cover - catastrophic filesystem failure
                rollback_errors.append(f"恢复{path}失败: {rollback_error}")
        for directory in reversed(created_directories):
            try:
                directory.rmdir()
            except OSError:
                pass
        if rollback_errors:
            raise RuntimeError(f"切分失败且回滚不完整: {rollback_errors[:5]}") from error
        raise


def _print_groups(groups: list[SceneGroup], dataset: Path) -> None:
    root, data = _load_dataset(dataset)
    train_images = root / str(data.get("train", "images/train"))
    for group in groups:
        accepted = sum(len(record.output_ids) for record in group.records)
        present = sum(1 for record in group.records for output_id in record.output_ids if _find_image(train_images, output_id))
        print(json.dumps({
            "group": group.key,
            "scenes": sorted(group.scene_names),
            "capture_session_ids": sorted(group.capture_session_ids),
            "source_video_ids": sorted(group.source_video_ids),
            "reports": [str(record.path) for record in group.records],
            "accepted_frames_in_reports": accepted,
            "train_images_present": present,
        }, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="按完整场次/采集视频划分YOLO train/val，默认只做dry-run")
    parser.add_argument("--data", type=Path, required=True, help="dataset.yaml或数据集目录")
    parser.add_argument("--list", action="store_true", help="列出可用场次/采集视频")
    parser.add_argument("--val-scenes", nargs="*", default=[], help="完整场次目录名，可一次指定多个")
    parser.add_argument("--val-videos", nargs="*", default=[], help="source_video_id，可一次指定多个")
    parser.add_argument("--auto", action="store_true", help="按类别和完整场次自动选择约20%%验证集")
    parser.add_argument("--target-val-ratio", type=float, default=0.20, help="自动模式目标验证集比例")
    parser.add_argument("--apply", action="store_true", help="执行移动；不加此参数只打印计划")
    args = parser.parse_args()
    groups = scan_groups(args.data)
    if args.list:
        _print_groups(groups, args.data)
        return 0
    if args.auto:
        plan = build_auto_plan(args.data, target_ratio=args.target_val_ratio)
    elif args.val_scenes or args.val_videos:
        plan = build_plan(args.data, args.val_scenes, args.val_videos)
    else:
        _print_groups(groups, args.data)
        return 0
    print(f"计划将{len(plan['files'])}个完整帧样本从train移动到val")
    for image_src, image_dst, _, _ in plan["files"]:
        print(f"  {image_src.name}: train -> val")
    if args.apply:
        apply_plan(plan)
        print("切分已执行；请立即运行训练数据验证。")
    else:
        print("当前为dry-run，未修改任何文件；确认后加 --apply 执行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
