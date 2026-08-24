#!/usr/bin/env python3
"""Safety-check and train a YOLO11 segmentation dataset.

The validator is deliberately stricter than Ultralytics' loader: empty labels
are accepted only through an explicit reviewed-negative manifest; malformed
polygons, duplicate images and scene/session/video split leakage are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
OFFICIAL_CLASSES = {
    0: "can", 1: "watermelon_rind", 2: "meal_box", 3: "red_paper_bag",
    4: "blue_bin", 5: "green_bin", 6: "red_bin",
}


def resolve_active_training_names(
    names: dict[int, str], split_reports: dict[str, dict[str, Any]]
) -> dict[int, str]:
    """Return the contiguous class prefix that has labels in both train and val."""
    ids_by_split: dict[str, set[int]] = {}
    for split in ("train", "val"):
        instances = split_reports.get(split, {}).get("instances", {})
        ids_by_split[split] = {int(key) for key, value in instances.items() if int(value) > 0}
    active_ids = sorted(ids_by_split["train"] | ids_by_split["val"])
    if not active_ids:
        raise ValueError("train/val没有任何accepted类别实例")
    expected = list(range(active_ids[-1] + 1))
    if active_ids != expected:
        raise ValueError(f"阶段训练类别ID必须从0连续，实际为{active_ids}")
    unknown = [class_id for class_id in active_ids if class_id not in names]
    if unknown:
        raise ValueError(f"阶段训练包含dataset.yaml未声明的类别ID: {unknown}")
    for split in ("train", "val"):
        missing = [names[class_id] for class_id in active_ids if class_id not in ids_by_split[split]]
        if missing:
            raise ValueError(f"{split}缺少阶段训练类别实例: {', '.join(missing)}")
    return {class_id: names[class_id] for class_id in active_ids}


def write_training_dataset_yaml(
    source_path: Path, data: dict[str, Any], active_names: dict[int, str], run_dir: Path
) -> Path:
    """Write a run-local dataset YAML so the model head matches actual accepted classes."""
    source_path = source_path.expanduser().resolve()
    training_data = dict(data)
    root_value = training_data.get("path", source_path.parent)
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (source_path.parent / root).resolve()
    training_data["path"] = str(root)
    training_data["names"] = active_names
    if training_data.get("test") == training_data.get("val"):
        training_data.pop("test", None)
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "dataset_training.yaml"
    output.write_text(yaml.safe_dump(training_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return output


def parse_args():
    p = argparse.ArgumentParser(description="验证并训练YOLO11 Segmentation")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("models/yolo11s-seg.pt"), help="baseline使用的官方YOLO11-seg预训练权重")
    p.add_argument("--init-mode", choices=("baseline", "xcx-transfer"), default="baseline")
    p.add_argument("--xcx-detect-model", type=Path, default=Path("xcx/checkpoint/unified_yolo11s_640_15class_final.pt"))
    p.add_argument("--transfer-architecture", default="yolo11s-seg.yaml", help="xcx-transfer的目标分割结构；不允许scratch训练")
    p.add_argument("--transfer-report", type=Path, default=None)
    p.add_argument("--min-transfer-parameter-fraction", type=float, default=0.80)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", type=Path, default=Path("runs/segment"))
    p.add_argument("--name", default="atec_yolo11s_seg")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--report", type=Path, default=None, help="数据安全检查JSON；默认写入数据集project_reports")
    p.add_argument("--allow-same-split-dir", action="store_true", help="仅烟雾测试：允许train和val指向同一目录")
    p.add_argument("--allow-cross-split-duplicates", action="store_true", help="仅诊断用途：允许train/val出现内容完全相同的图片")
    p.add_argument("--require-project-reports", action="store_true", help="要求存在聚合器生成的场景报告，以验证场景没有跨split复用")
    p.add_argument("--require-source-ids", action="store_true", help="要求每个有效场景报告都有capture_session_id和source_video_id")
    p.add_argument("--reviewed-negatives", type=Path, default=None, help="人工确认纯干扰帧清单；默认读取数据集根目录reviewed_negatives.json（若存在）")
    return p.parse_args()


def resolve_dataset(data_path: Path):
    data_path = data_path.expanduser().resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_path.parent)).expanduser()
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    return data_path, data, root


def load_reviewed_negatives(path: Path | None, root: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    candidate = path.expanduser().resolve() if path else root / "reviewed_negatives.json"
    empty = {"train": set(), "val": set()}
    report: dict[str, Any] = {"path": str(candidate), "present": candidate.is_file(), "errors": [], "items": []}
    if not candidate.is_file():
        if path is not None:
            report["errors"].append(f"人工负样本清单不存在: {candidate}")
        return empty, report
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception as exc:
        report["errors"].append(f"无法读取人工负样本清单: {exc}")
        return empty, report
    items = data.get("items", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        report["errors"].append("人工负样本清单items必须是列表")
        return empty, report
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            report["errors"].append(f"人工负样本第{index + 1}项不是对象")
            continue
        split = str(item.get("split", ""))
        image = str(item.get("image", ""))
        stem = Path(image).stem
        if split not in empty:
            report["errors"].append(f"人工负样本split必须是train/val: {split}")
            continue
        if not image or not stem:
            report["errors"].append(f"人工负样本第{index + 1}项缺少image")
            continue
        if item.get("reviewed_no_targets") is not True:
            report["errors"].append(f"人工负样本未明确reviewed_no_targets=true: {split}/{image}")
            continue
        key = (split, stem)
        if key in seen:
            report["errors"].append(f"人工负样本重复: {split}/{stem}")
            continue
        seen.add(key)
        empty[split].add(stem)
        report["items"].append({
            "split": split, "image": image, "stem": stem, "reviewed_no_targets": True,
            "capture_session_id": str(item.get("capture_session_id") or ""),
            "source_video_id": str(item.get("source_video_id") or ""),
        })
    return empty, report


def tensor_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", ())
    return tuple(int(x) for x in shape)


def tensor_numel(value: Any) -> int:
    method = getattr(value, "numel", None)
    if callable(method):
        return int(method())
    size = getattr(value, "size", 0)
    return int(size) if isinstance(size, (int, float)) else 0


def select_compatible_transfer_state(
    target_state: dict[str, Any], source_state: dict[str, Any], excluded_prefixes: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected: dict[str, Any] = {}
    shape_mismatch: list[str] = []
    missing_source: list[str] = []
    eligible_parameters = 0
    loaded_parameters = 0
    for key, target_value in target_state.items():
        if key.endswith("num_batches_tracked") or key.startswith(excluded_prefixes):
            continue
        eligible_parameters += tensor_numel(target_value)
        source_value = source_state.get(key)
        if source_value is None:
            missing_source.append(key)
            continue
        if tensor_shape(source_value) != tensor_shape(target_value):
            shape_mismatch.append(key)
            continue
        selected[key] = source_value
        loaded_parameters += tensor_numel(target_value)
    fraction = loaded_parameters / eligible_parameters if eligible_parameters else 0.0
    return selected, {
        "eligible_tensor_count": sum(
            1 for key in target_state if not key.endswith("num_batches_tracked") and not key.startswith(excluded_prefixes)
        ),
        "loaded_tensor_count": len(selected),
        "eligible_parameter_count": eligible_parameters,
        "loaded_parameter_count": loaded_parameters,
        "loaded_parameter_fraction": fraction,
        "shape_mismatch": shape_mismatch,
        "missing_source": missing_source,
        "excluded_prefixes": list(excluded_prefixes),
    }


def build_xcx_transfer_model(args: argparse.Namespace):
    from ultralytics import YOLO

    source_path = args.xcx_detect_model.expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit(f"xcx统一检测权重不存在: {source_path}")
    source = YOLO(str(source_path))
    target = YOLO(str(args.transfer_architecture))
    if getattr(source, "task", None) != "detect":
        raise SystemExit(f"xcx迁移源必须是detect模型，实际为: {getattr(source, 'task', None)}")
    if getattr(target, "task", None) != "segment":
        raise SystemExit(f"迁移目标必须是segment模型，实际为: {getattr(target, 'task', None)}")
    target_layers = getattr(target.model, "model", None)
    if target_layers is None or len(target_layers) < 2:
        raise SystemExit("无法识别YOLO11-seg模块结构")
    head_prefix = f"model.{len(target_layers) - 1}."
    selected, transfer = select_compatible_transfer_state(
        target.model.state_dict(), source.model.state_dict(), (head_prefix,)
    )
    load_result = target.model.load_state_dict(selected, strict=False)
    transfer.update({
        "init_mode": "xcx-transfer",
        "source_model": str(source_path),
        "target_architecture": str(args.transfer_architecture),
        "segmentation_head_prefix": head_prefix,
        "segmentation_head_reinitialized": True,
        "load_missing_keys": list(load_result.missing_keys),
        "load_unexpected_keys": list(load_result.unexpected_keys),
        "minimum_required_fraction": args.min_transfer_parameter_fraction,
    })
    if transfer["loaded_parameter_fraction"] < args.min_transfer_parameter_fraction:
        raise SystemExit(
            "xcx迁移兼容参数比例过低: "
            f"{transfer['loaded_parameter_fraction']:.3f} < {args.min_transfer_parameter_fraction:.3f}"
        )
    return target, transfer


def polygon_area(coords: list[float]) -> float:
    points = list(zip(coords[0::2], coords[1::2]))
    return abs(sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )) / 2.0


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def validate_split(root: Path, rel: str, names: dict[int, str], reviewed_negative_stems: set[str] | None = None):
    image_dir = (root / rel).resolve()
    split_name = image_dir.name
    label_dir = root / "labels" / split_name
    images = image_files(image_dir)
    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter[int] = Counter()
    polygon_count = 0
    image_stems = {p.stem for p in images}
    reviewed_negative_stems = set(reviewed_negative_stems or set())
    reviewed_negative_used: list[str] = []

    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            errors.append(f"缺少标签: {image.name}")
            continue
        lines = [x.strip() for x in label.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not lines:
            if image.stem in reviewed_negative_stems:
                reviewed_negative_used.append(image.stem)
            else:
                errors.append(f"未人工确认的空标签: {label.name}")
        elif image.stem in reviewed_negative_stems:
            errors.append(f"reviewed_no_targets帧却含有正式标签: {label.name}")
        for line_no, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) < 7 or (len(parts) - 1) % 2:
                errors.append(f"分割标签点数/格式错误: {label.name}:{line_no}")
                continue
            try:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                errors.append(f"非数值标签: {label.name}:{line_no}")
                continue
            if class_id not in names:
                errors.append(f"未知类别ID {class_id}: {label.name}:{line_no}")
            if any(not math.isfinite(x) for x in coords):
                errors.append(f"标签包含NaN/Inf: {label.name}:{line_no}")
                continue
            if any(x < 0 or x > 1 for x in coords):
                errors.append(f"坐标超出0到1: {label.name}:{line_no}")
            if polygon_area(coords) <= 1e-8:
                errors.append(f"退化或零面积多边形: {label.name}:{line_no}")
            counts[class_id] += 1
            polygon_count += 1

    if label_dir.exists():
        orphan_labels = sorted(p.name for p in label_dir.glob("*.txt") if p.stem not in image_stems)
        errors.extend(f"标签没有对应图片: {name}" for name in orphan_labels)
    else:
        errors.append(f"标签目录不存在: {label_dir}")

    for stem in sorted(reviewed_negative_stems - image_stems):
        errors.append(f"人工负样本清单在{split_name}中找不到图片: {stem}")

    missing_classes = [names[i] for i in names if counts[i] == 0]
    if missing_classes:
        warnings.append(f"该split没有以下类别实例: {', '.join(missing_classes)}")
    return {
        "images": len(images),
        "polygons": polygon_count,
        "instances": {str(k): v for k, v in sorted(counts.items())},
        "missing_classes": missing_classes,
        "errors": errors,
        "warnings": warnings,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "files": [str(p) for p in images],
        "reviewed_negative_count": len(reviewed_negative_used),
        "reviewed_negative_stems": sorted(reviewed_negative_used),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_cross_split_duplicates(train_files: list[str], val_files: list[str]) -> list[dict]:
    by_size: dict[int, dict[str, list[Path]]] = defaultdict(lambda: {"train": [], "val": []})
    for split, files in (("train", train_files), ("val", val_files)):
        for value in files:
            path = Path(value)
            by_size[path.stat().st_size][split].append(path)

    duplicates: list[dict] = []
    for groups in by_size.values():
        if not groups["train"] or not groups["val"]:
            continue
        train_hashes: dict[str, list[Path]] = defaultdict(list)
        for path in groups["train"]:
            train_hashes[sha256_file(path)].append(path)
        for path in groups["val"]:
            digest = sha256_file(path)
            for train_path in train_hashes.get(digest, []):
                duplicates.append({"sha256": digest, "train": str(train_path), "val": str(path)})
    return duplicates


def inspect_scene_reports(root: Path) -> dict:
    report_dir = root / "project_reports"
    records = []
    scene_splits: dict[str, set[str]] = defaultdict(set)
    session_splits: dict[str, set[str]] = defaultdict(set)
    video_splits: dict[str, set[str]] = defaultdict(set)
    missing_source_ids = []
    for path in sorted(report_dir.glob("*_report.json")) if report_dir.exists() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            records.append({"report": str(path), "error": str(exc)})
            continue
        scene_raw = data.get("scene")
        split = str(data.get("split", ""))
        accepted = int(data.get("frame_status_counts", {}).get("accepted", 0))
        scene = str(Path(scene_raw).expanduser().resolve()) if scene_raw else ""
        session_id = str(data.get("capture_session_id") or "")
        video_id = str(data.get("source_video_id") or "")
        record = {
            "report": str(path), "scene": scene, "split": split, "accepted": accepted,
            "capture_session_id": session_id, "source_video_id": video_id,
        }
        records.append(record)
        if split and accepted > 0:
            if scene:
                scene_splits[scene].add(split)
            if session_id:
                session_splits[session_id].add(split)
            if video_id:
                video_splits[video_id].add(split)
            if not session_id or not video_id:
                missing_source_ids.append({
                    "report": str(path), "capture_session_id": session_id or None,
                    "source_video_id": video_id or None,
                })

    def reused(groups: dict[str, set[str]], key_name: str) -> list[dict]:
        return [
            {key_name: key, "splits": sorted(splits)}
            for key, splits in sorted(groups.items())
            if "train" in splits and "val" in splits
        ]

    return {
        "report_dir": str(report_dir),
        "records": records,
        "reused_train_val_scenes": reused(scene_splits, "scene"),
        "reused_train_val_capture_sessions": reused(session_splits, "capture_session_id"),
        "reused_train_val_source_videos": reused(video_splits, "source_video_id"),
        "reports_missing_source_ids": missing_source_ids,
    }


def public_split_report(split_report: dict) -> dict:
    return {k: v for k, v in split_report.items() if k != "files"}


def main():
    args = parse_args()
    data_path, data, root = resolve_dataset(args.data)
    raw_names = data.get("names", {})
    names = {int(k): str(v) for k, v in (enumerate(raw_names) if isinstance(raw_names, list) else raw_names.items())}
    if not names:
        raise SystemExit("dataset.yaml没有names")
    if sorted(names) != list(range(len(names))):
        raise SystemExit("dataset.yaml的类别ID必须从0连续编号")
    if names != OFFICIAL_CLASSES:
        raise SystemExit(f"正式训练只允许当前固定7类，实际names={names}")

    reviewed_negatives, negative_report = load_reviewed_negatives(args.reviewed_negatives, root)
    report: dict = {
        "dataset": str(data_path), "root": str(root), "names": names,
        "reviewed_negatives": negative_report,
    }
    split_reports = {}
    for split in ("train", "val"):
        rel = data.get(split)
        if not rel:
            raise SystemExit(f"dataset.yaml缺少{split}")
        split_reports[split] = validate_split(root, rel, names, reviewed_negatives[split])
        report[split] = public_split_report(split_reports[split])

    errors = negative_report["errors"] + report["train"]["errors"] + report["val"]["errors"]
    warnings = report["train"]["warnings"] + report["val"]["warnings"]
    active_names: dict[int, str] = {}
    try:
        active_names = resolve_active_training_names(names, split_reports)
    except ValueError as exc:
        errors.append(str(exc))
    report["active_training_classes"] = active_names
    report["active_training_class_count"] = len(active_names)
    train_dir = Path(report["train"]["image_dir"])
    val_dir = Path(report["val"]["image_dir"])
    if report["train"]["images"] == 0:
        errors.append("训练集没有图片")
    if report["val"]["images"] == 0:
        errors.append("验证集没有图片；正式训练必须先采集不同背景的val场景")
    if train_dir == val_dir and not args.allow_same_split_dir:
        errors.append("train和val指向同一目录；正式训练禁止这样做")

    duplicates = find_cross_split_duplicates(split_reports["train"]["files"], split_reports["val"]["files"])
    report["cross_split_duplicate_images"] = duplicates
    if duplicates and not args.allow_cross_split_duplicates:
        errors.append(f"train/val存在{len(duplicates)}对内容完全相同的图片")

    scene_report = inspect_scene_reports(root)
    report["scene_reports"] = scene_report
    if args.require_project_reports and not scene_report["records"]:
        errors.append("缺少project_reports场景来源报告，无法证明train/val来自不同录制场景")
    if scene_report["reused_train_val_scenes"]:
        errors.append("同一个RGB-D场景被同时用于train和val")
    if scene_report["reused_train_val_capture_sessions"]:
        errors.append("同一个capture_session_id被同时用于train和val；第4场次必须完整隔离")
    if scene_report["reused_train_val_source_videos"]:
        errors.append("同一个source_video_id被同时用于train和val；禁止相邻帧或复制视频跨集合")
    if args.require_source_ids and scene_report["reports_missing_source_ids"]:
        errors.append("有效project report缺少capture_session_id/source_video_id")

    combined_sessions: dict[str, set[str]] = defaultdict(set)
    combined_videos: dict[str, set[str]] = defaultdict(set)
    for record in scene_report["records"]:
        if int(record.get("accepted", 0)) <= 0:
            continue
        if record.get("capture_session_id"):
            combined_sessions[record["capture_session_id"]].add(record.get("split", ""))
        if record.get("source_video_id"):
            combined_videos[record["source_video_id"]].add(record.get("split", ""))
    negative_missing_ids = []
    for item in negative_report["items"]:
        if item.get("capture_session_id"):
            combined_sessions[item["capture_session_id"]].add(item["split"])
        if item.get("source_video_id"):
            combined_videos[item["source_video_id"]].add(item["split"])
        if not item.get("capture_session_id") or not item.get("source_video_id"):
            negative_missing_ids.append(f"{item['split']}/{item['image']}")
    report["combined_source_id_leakage"] = {
        "capture_sessions": [key for key, splits in combined_sessions.items() if "train" in splits and "val" in splits],
        "source_videos": [key for key, splits in combined_videos.items() if "train" in splits and "val" in splits],
        "reviewed_negatives_missing_source_ids": negative_missing_ids,
    }
    if report["combined_source_id_leakage"]["capture_sessions"]:
        errors.append("project report/人工负样本中有capture_session_id跨train/val")
    if report["combined_source_id_leakage"]["source_videos"]:
        errors.append("project report/人工负样本中有source_video_id跨train/val")
    if args.require_source_ids and negative_missing_ids:
        errors.append("人工负样本缺少capture_session_id/source_video_id")
    if scene_report["records"] and not any((
        scene_report["reused_train_val_scenes"],
        scene_report["reused_train_val_capture_sessions"],
        scene_report["reused_train_val_source_videos"],
    )):
        warnings.append("路径和来源ID未跨split复用；场次是否真正独立仍需人工确认")

    report["errors"] = errors
    report["warnings"] = warnings
    report_path = (args.report.expanduser().resolve() if args.report else root / "project_reports" / "dataset_validation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"检查报告：{report_path}")
    if errors:
        print("前20个错误：")
        for error in errors[:20]:
            print(" -", error)
        raise SystemExit(f"数据检查失败，共{len(errors)}项")
    if warnings:
        print("警告：")
        for warning in warnings:
            print(" -", warning)
    if args.validate_only:
        print("数据验证通过。")
        return 0

    from ultralytics import YOLO

    project = args.project.expanduser().resolve()
    transfer_report = None
    if args.init_mode == "baseline":
        model_path = args.model.expanduser().resolve()
        if not model_path.is_file():
            raise SystemExit(f"官方YOLO11-seg预训练权重不存在: {model_path}")
        model = YOLO(str(model_path))
        if getattr(model, "task", None) != "segment":
            raise SystemExit(f"baseline权重必须是segment模型，实际为: {getattr(model, 'task', None)}")
    else:
        model, transfer_report = build_xcx_transfer_model(args)
        transfer_path = (
            args.transfer_report.expanduser().resolve()
            if args.transfer_report else project / args.name / "transfer_report.json"
        )
        transfer_path.parent.mkdir(parents=True, exist_ok=True)
        transfer_path.write_text(json.dumps(transfer_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"迁移报告：{transfer_path}")

    cache = False if args.cache == "false" else args.cache
    training_data_path = write_training_dataset_yaml(
        data_path, data, active_names, project / args.name
    )
    print(
        f"阶段训练类别({len(active_names)}): "
        + ", ".join(active_names.values())
        + f"；训练配置：{training_data_path}"
    )
    model.train(
        data=str(training_data_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, amp=True, cache=cache,
        project=str(project), name=args.name, seed=args.seed, deterministic=True,
        patience=args.patience, task="segment", exist_ok=True, plots=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
