#!/usr/bin/env python3
"""Evaluate YOLO11-seg candidates on real val, distractors and end-to-end FPS."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atec_pipeline.object_config import is_supported_class_prefix, load_class_map

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
OFFICIAL_CLASSES = load_class_map(ROOT / "configs" / "atec_objects.yaml")


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("模型必须写成NAME=/path/to/best.pt")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("模型名称和路径不能为空")
    return name, Path(path)


def parse_args():
    p = argparse.ArgumentParser(description="比较真实val精度、干扰误检和RTX端到端FPS")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", dest="models", type=parse_model, action="append", required=True)
    p.add_argument("--distractor-dir", type=Path, required=True)
    p.add_argument("--fps-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-fps-frames", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20)
    return p.parse_args()


def image_files(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def normalize_names(raw: Any) -> dict[int, str]:
    return {int(k): str(v) for k, v in (enumerate(raw) if isinstance(raw, list) else raw.items())}


def sync_cuda(device: str) -> None:
    import torch

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def validated_model_names(model: Any) -> dict[int, str]:
    names = normalize_names(getattr(model, "names", {}))
    if not is_supported_class_prefix(names, OFFICIAL_CLASSES):
        raise RuntimeError(f"候选模型类别不是当前9类配置的连续前缀: {names}")
    return names


def validate_metrics(model: Any, data: Path, device: str, imgsz: int) -> dict[str, Any]:
    metrics = model.val(data=str(data), split="val", device=device, imgsz=imgsz, plots=False, verbose=False)
    names = validated_model_names(model)
    seg = getattr(metrics, "seg", None)
    if seg is None:
        raise RuntimeError("Ultralytics验证结果没有seg指标")
    metric_index = {int(class_id): index for index, class_id in enumerate(seg.ap_class_index)}
    missing_val_classes = [name for class_id, name in names.items() if class_id not in metric_index]
    if missing_val_classes:
        raise RuntimeError(f"真实val缺少正式类别，无法比较每类召回: {missing_val_classes}")
    per_class = {}
    for class_id, class_name in names.items():
        precision, recall, map50, map5095 = seg.class_result(metric_index[class_id])
        per_class[class_name] = {
            "precision": float(precision), "recall": float(recall),
            "mask_map50": float(map50), "mask_map50_95": float(map5095),
        }
    return {
        "mask_map50_95": float(seg.map), "mask_map50": float(seg.map50), "mask_map75": float(seg.map75),
        "mean_precision": float(seg.mp), "mean_recall": float(seg.mr),
        "per_class": per_class,
        "results_dict": {str(k): float(v) for k, v in getattr(metrics, "results_dict", {}).items()},
    }


def distractor_metrics(model: Any, files: list[Path], device: str, imgsz: int, conf: float) -> dict[str, Any]:
    model_names = validated_model_names(model)
    false_instances = 0
    frames_with_false_positive = 0
    per_class = {name: 0 for name in model_names.values()}
    for result in model.predict(
        source=[str(p) for p in files], device=device, imgsz=imgsz, conf=conf,
        stream=True, verbose=False, save=False,
    ):
        boxes = getattr(result, "boxes", None)
        count = len(boxes) if boxes is not None else 0
        false_instances += count
        frames_with_false_positive += int(count > 0)
        if boxes is not None:
            for class_id in boxes.cls.detach().cpu().numpy().tolist():
                per_class[model_names[int(class_id)]] += 1
    total = len(files)
    return {
        "frames": total,
        "false_positive_instances": false_instances,
        "frames_with_false_positive": frames_with_false_positive,
        "false_positive_frames_fraction": frames_with_false_positive / total if total else None,
        "false_positive_instances_per_frame": false_instances / total if total else None,
        "per_class_false_positive_instances": per_class,
    }


def fps_metrics(model: Any, files: list[Path], device: str, imgsz: int, conf: float, warmup: int) -> dict[str, Any]:
    if not files:
        raise RuntimeError("FPS输入没有图片")
    warmup_files = files[: min(warmup, len(files))]
    for path in warmup_files:
        model.predict(source=str(path), device=device, imgsz=imgsz, conf=conf, verbose=False, save=False)
    sync_cuda(device)
    started = time.perf_counter()
    for path in files:
        model.predict(source=str(path), device=device, imgsz=imgsz, conf=conf, verbose=False, save=False)
    sync_cuda(device)
    elapsed = time.perf_counter() - started
    return {
        "frames": len(files), "elapsed_seconds": elapsed,
        "end_to_end_fps": len(files) / elapsed if elapsed > 0 else None,
        "target_fps_min": 20, "target_fps_max": 30,
        "meets_20_fps": bool(elapsed > 0 and len(files) / elapsed >= 20),
    }


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    distractors = image_files(args.distractor_dir)
    fps_files = image_files(args.fps_source)[: args.max_fps_frames]
    if not distractors:
        raise SystemExit("纯干扰目录没有图片")
    if len(fps_files) < 500:
        print(f"[警告] FPS连续测试只有{len(fps_files)}帧，正式验收要求至少500帧")
    report = {
        "format_version": 1,
        "data": str(args.data.expanduser().resolve()),
        "device": args.device, "imgsz": args.imgsz, "confidence": args.conf,
        "models": {},
        "selection_rule": (
            "B只有在独立真实val的Mask mAP50-95和每类召回稳定优于A，且纯干扰误检不增加时才采用；"
            "否则xcx只保留为标注教师。"
        ),
    }
    for name, path in args.models:
        model_path = path.expanduser().resolve()
        if not model_path.is_file():
            raise SystemExit(f"模型不存在: {model_path}")
        print(f"评估{name}: {model_path}", flush=True)
        model = YOLO(str(model_path))
        report["models"][name] = {
            "path": str(model_path),
            "validation": validate_metrics(model, args.data.expanduser().resolve(), args.device, args.imgsz),
            "distractors": distractor_metrics(model, distractors, args.device, args.imgsz, args.conf),
            "fps": fps_metrics(model, fps_files, args.device, args.imgsz, args.conf, args.warmup),
        }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"候选比较报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
