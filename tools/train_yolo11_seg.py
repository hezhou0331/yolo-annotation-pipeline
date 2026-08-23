#!/usr/bin/env python3
"""Safety-check and train a YOLO11 segmentation dataset.

The validator is deliberately stricter than Ultralytics' loader: it rejects
empty labels, malformed/degenerate polygons, train/val directory reuse,
content-identical images across train and validation, and reuse of the exact
same recorded RGB-D scene in multiple splits when project reports are present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args():
    p = argparse.ArgumentParser(description="验证并训练YOLO11 Segmentation")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("models/yolo11n-seg.pt"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", type=Path, default=Path("runs/segment"))
    p.add_argument("--name", default="atec_yolo11n_seg")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--report", type=Path, default=None, help="数据安全检查JSON；默认写入数据集project_reports")
    p.add_argument("--allow-same-split-dir", action="store_true", help="仅烟雾测试：允许train和val指向同一目录")
    p.add_argument("--allow-cross-split-duplicates", action="store_true", help="仅诊断用途：允许train/val出现内容完全相同的图片")
    p.add_argument("--require-project-reports", action="store_true", help="要求存在聚合器生成的场景报告，以验证场景没有跨split复用")
    return p.parse_args()


def resolve_dataset(data_path: Path):
    data_path = data_path.expanduser().resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_path.parent)).expanduser()
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    return data_path, data, root


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


def validate_split(root: Path, rel: str, names: dict[int, str]):
    image_dir = (root / rel).resolve()
    split_name = image_dir.name
    label_dir = root / "labels" / split_name
    images = image_files(image_dir)
    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter[int] = Counter()
    polygon_count = 0
    image_stems = {p.stem for p in images}

    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            errors.append(f"缺少标签: {image.name}")
            continue
        lines = [x.strip() for x in label.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not lines:
            errors.append(f"空标签: {label.name}")
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
        records.append({"report": str(path), "scene": scene, "split": split, "accepted": accepted})
        if scene and split and accepted > 0:
            scene_splits[scene].add(split)
    reused = [
        {"scene": scene, "splits": sorted(splits)}
        for scene, splits in sorted(scene_splits.items())
        if "train" in splits and "val" in splits
    ]
    return {"report_dir": str(report_dir), "records": records, "reused_train_val_scenes": reused}


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

    report: dict = {"dataset": str(data_path), "root": str(root), "names": names}
    split_reports = {}
    for split in ("train", "val"):
        rel = data.get(split)
        if not rel:
            raise SystemExit(f"dataset.yaml缺少{split}")
        split_reports[split] = validate_split(root, rel, names)
        report[split] = public_split_report(split_reports[split])

    errors = report["train"]["errors"] + report["val"]["errors"]
    warnings = report["train"]["warnings"] + report["val"]["warnings"]
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
    if scene_report["records"] and not scene_report["reused_train_val_scenes"]:
        warnings.append("场景路径未跨split复用；背景是否真正不同仍需按现场布置人工确认")

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

    model_path = args.model.expanduser().resolve()
    cache = False if args.cache == "false" else args.cache
    model = YOLO(str(model_path))
    model.train(
        data=str(data_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, amp=True, cache=cache,
        project=str(args.project.expanduser().resolve()), name=args.name,
        patience=args.patience, task="segment", exist_ok=True, plots=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
