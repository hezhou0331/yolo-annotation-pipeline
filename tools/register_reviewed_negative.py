#!/usr/bin/env python3
"""Safely add a manually reviewed distractor-only image to a YOLO dataset."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="登记人工确认的纯干扰帧（空YOLO标签）")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output-id", required=True)
    p.add_argument("--capture-session-id", required=True)
    p.add_argument("--source-video-id", required=True)
    p.add_argument("--reviewer", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_path.parent)).expanduser()
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    image = args.image.expanduser().resolve()
    if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
        raise SystemExit(f"源图片不存在或格式不支持: {image}")
    if not args.output_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.output_id):
        raise SystemExit("--output-id只允许字母、数字、-和_")
    image_dir = (root / data[args.split]).resolve()
    label_dir = root / "labels" / image_dir.name
    destination = image_dir / f"{args.output_id}{image.suffix.lower()}"
    label = label_dir / f"{args.output_id}.txt"
    manifest_path = root / "reviewed_negatives.json"
    if destination.exists() or label.exists():
        raise SystemExit(f"目标图片或标签已存在，拒绝覆盖: {destination}")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"format_version": 1, "items": []}
    keys = {(str(x.get("split")), Path(str(x.get("image", ""))).stem) for x in manifest.get("items", [])}
    if (args.split, args.output_id) in keys:
        raise SystemExit("人工负样本清单已存在同名项目")
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, destination)
    label.write_text("", encoding="utf-8")
    manifest.setdefault("items", []).append({
        "split": args.split,
        "image": destination.name,
        "reviewed_no_targets": True,
        "capture_session_id": args.capture_session_id,
        "source_video_id": args.source_video_id,
        "reviewer": args.reviewer,
        "source_image": str(image),
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已登记人工纯干扰帧：{destination}")
    print(f"空标签：{label}")
    print(f"清单：{manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
