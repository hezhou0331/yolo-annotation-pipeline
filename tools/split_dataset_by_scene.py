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
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import yaml

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from atec_pipeline.path_compat import infer_project_root, resolve_compatible_path

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
ACCEPTED_STATUSES = {"accepted"}


@dataclass
class ReportRecord:
    path: Path
    scene: str
    split: str
    capture_session_id: str | None
    source_video_id: str | None
    output_ids: list[str]
    class_names: set[str] = field(default_factory=set)


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
    scene = resolve_compatible_path(
        str(data["scene"]),
        base=path.parent,
        repository_root=WORKSPACE,
        project_root=infer_project_root(path, repository_root=WORKSPACE),
    )
    return ReportRecord(
        path=path,
        scene=str(scene),
        split=str(data.get("split", "train")),
        capture_session_id=str(data["capture_session_id"]) if data.get("capture_session_id") else None,
        source_video_id=str(data["source_video_id"]) if data.get("source_video_id") else None,
        output_ids=sorted(set(output_ids)),
        class_names=class_names,
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


def build_plan(dataset: Path | str, scenes: list[str], videos: list[str]) -> dict[str, Any]:
    dataset_path = Path(dataset)
    root, data = _load_dataset(dataset_path)
    groups = scan_groups(root)
    selected = _selected_groups(groups, scenes, videos)
    train_images = root / str(data.get("train", "images/train"))
    train_labels = root / "labels" / Path(str(data.get("train", "images/train"))).name
    val_images = root / str(data.get("val", "images/val"))
    val_labels = root / "labels" / Path(str(data.get("val", "images/val"))).name
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
    manifest_updates: list[tuple[Path, Path]] = []
    session_metadata: list[Path] = []
    for group in selected:
        for record in group.records:
            scene = Path(record.scene).expanduser().resolve()
            metadata = scene / "atec_capture_session.json"
            if metadata.is_file():
                session_metadata.append(metadata)
            project_root = None
            for ancestor in scene.parents:
                if ancestor.name == "data":
                    project_root = ancestor.parent
                    break
            if project_root is None:
                continue
            old_manifest = project_root / "manifests" / f"{scene.name}_train.yaml"
            new_manifest = project_root / "manifests" / f"{scene.name}_val.yaml"
            if old_manifest.is_file():
                if new_manifest.exists():
                    raise ValueError(f"val Manifest目标已存在，拒绝覆盖: {new_manifest}")
                manifest_updates.append((old_manifest, new_manifest))
    return {
        "dataset": root,
        "dataset_yaml": (dataset_path if dataset_path.is_file() else root / "dataset.yaml").resolve(),
        "groups": selected,
        "files": moves,
        "reports": sorted(set(report_paths)),
        "selected_keys": selected_keys,
        "val_images": val_images,
        "val_labels": val_labels,
        "manifest_updates": sorted(set(manifest_updates)),
        "session_metadata": sorted(set(session_metadata)),
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
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def apply_plan(plan: dict[str, Any]) -> None:
    moves: list[tuple[Path, Path, Path, Path]] = plan["files"]
    reports: list[Path] = plan["reports"]
    dataset_yaml: Path = plan["dataset_yaml"]
    manifest_updates: list[tuple[Path, Path]] = plan.get("manifest_updates", [])
    session_metadata: list[Path] = plan.get("session_metadata", [])
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
    mutable_files = sorted(set(reports + session_metadata + [dataset_yaml] + [old for old, _ in manifest_updates]))
    backups = {path: path.read_bytes() for path in mutable_files}
    for directory in (plan["val_images"], plan["val_labels"]):
        directory.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[Path, Path, Path, Path]] = []
    renamed_manifests: list[tuple[Path, Path]] = []
    try:
        for image_src, image_dst, label_src, label_dst in moves:
            shutil.move(str(image_src), str(image_dst))
            shutil.move(str(label_src), str(label_dst))
            moved.append((image_src, image_dst, label_src, label_dst))
        for report_path in reports:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            data["split"] = "val"
            _atomic_write_text(report_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        for metadata_path in session_metadata:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            data["split"] = "val"
            _atomic_write_text(metadata_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        for old_manifest, new_manifest in manifest_updates:
            data = yaml.safe_load(old_manifest.read_text(encoding="utf-8")) or {}
            project = data.setdefault("project", {})
            if not isinstance(project, dict):
                raise ValueError(f"Manifest project字段无效: {old_manifest}")
            project["split"] = "val"
            _atomic_write_text(old_manifest, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
            old_manifest.replace(new_manifest)
            renamed_manifests.append((old_manifest, new_manifest))
        root, data = _load_dataset(dataset_yaml)
        data["val"] = "images/val"
        if data.get("test") == data["val"]:
            data.pop("test", None)
        _atomic_write_text(dataset_yaml, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    except Exception:
        for old_manifest, new_manifest in reversed(renamed_manifests):
            if new_manifest.exists() and not old_manifest.exists():
                new_manifest.replace(old_manifest)
        for path, content in backups.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        for image_src, image_dst, label_src, label_dst in reversed(moved):
            if image_dst.exists() and not image_src.exists():
                shutil.move(str(image_dst), str(image_src))
            if label_dst.exists() and not label_src.exists():
                shutil.move(str(label_dst), str(label_src))
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
