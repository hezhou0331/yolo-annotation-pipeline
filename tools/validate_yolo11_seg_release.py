#!/usr/bin/env python3
"""Validate one trained YOLO11-seg run before publishing its best weights."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atec_pipeline.object_config import load_class_map


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
EXPECTED_CLASSES = load_class_map(ROOT / "configs" / "atec_objects.yaml")
REQUIRED_MASK_METRICS = (
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
)
CHECKPOINT_FITNESS_METRICS = (
    "metrics/mAP50-95(B)",
    "metrics/mAP50-95(M)",
)
STRIPPED_CHECKPOINT_FIELDS = (
    "optimizer",
    "best_fitness",
    "ema",
    "updates",
    "scaler",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="发布前严格验收YOLO11-seg best.pt、训练指标和九类val推理",
    )
    parser.add_argument("--run", type=Path, required=True, help="训练run目录")
    parser.add_argument("--data", type=Path, required=True, help="九类dataset.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, help="可选JSON验收报告")
    return parser.parse_args()


def normalize_names(raw: Any) -> dict[int, str]:
    if isinstance(raw, list):
        return {index: str(name) for index, name in enumerate(raw)}
    if isinstance(raw, dict):
        try:
            return {int(class_id): str(name) for class_id, name in raw.items()}
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"类别ID必须是整数: {raw}") from exc
    raise RuntimeError(f"无法解析模型或数据集类别: {raw!r}")


def finite_float(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label}不是数值: {value!r}") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{label}不是有限数值: {value!r}")
    return number


def inspect_results_csv(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"训练指标不存在: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise RuntimeError(f"训练指标为空: {path}")
    required = tuple(dict.fromkeys(REQUIRED_MASK_METRICS + CHECKPOINT_FITNESS_METRICS))
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise RuntimeError(f"results.csv缺少发布验收指标列: {missing}")

    metric_columns = [name for name in rows[0] if name.startswith("metrics/")]
    parsed_rows: list[dict[str, float]] = []
    for row_number, row in enumerate(rows, start=2):
        parsed = {
            name: finite_float(row.get(name), label=f"results.csv第{row_number}行 {name}")
            for name in metric_columns
        }
        parsed["epoch"] = finite_float(
            row.get("epoch"), label=f"results.csv第{row_number}行 epoch"
        )
        parsed_rows.append(parsed)

    box_map = "metrics/mAP50-95(B)"
    mask_map = "metrics/mAP50-95(M)"
    best = max(parsed_rows, key=lambda row: row[box_map] + row[mask_map])
    mask_peak = max(parsed_rows, key=lambda row: row[mask_map])
    return {
        "epochs_recorded": len(parsed_rows),
        "best_epoch": int(best["epoch"]),
        "best_metrics": {name: best[name] for name in REQUIRED_MASK_METRICS},
        "selection_fitness": best[box_map] + best[mask_map],
        "selection_metrics": {
            box_map: best[box_map],
            mask_map: best[mask_map],
        },
        "mask_peak_epoch": int(mask_peak["epoch"]),
        "mask_peak_map50_95": mask_peak[mask_map],
    }


def validate_model_identity(model: Any) -> dict[int, str]:
    task = str(getattr(model, "task", ""))
    if task != "segment":
        raise RuntimeError(f"模型任务必须为segment，实际为: {task or '<empty>'}")
    names = normalize_names(getattr(model, "names", None))
    if names != EXPECTED_CLASSES:
        raise RuntimeError(f"模型类别必须严格等于0-8九类配置: {names}")
    return names


def _resolve_dataset_root(data_path: Path, config: dict[str, Any]) -> Path:
    configured = Path(str(config.get("path", "."))).expanduser()
    return configured.resolve() if configured.is_absolute() else (data_path.parent / configured).resolve()


def _images_from_spec(spec: str, *, dataset_root: Path) -> list[Path]:
    path = Path(spec).expanduser()
    path = path.resolve() if path.is_absolute() else (dataset_root / path).resolve()
    if path.is_dir():
        return sorted(
            item for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_file() and path.suffix.lower() == ".txt":
        images: list[Path] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            value = raw_line.strip()
            if not value:
                continue
            image = Path(value).expanduser()
            image = image.resolve() if image.is_absolute() else (path.parent / image).resolve()
            images.append(image)
        return sorted(images)
    raise RuntimeError(f"无法解析val图片来源: {path}")


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    image_indexes = [index for index, value in enumerate(parts) if value == "images"]
    if not image_indexes:
        raise RuntimeError(f"图片路径不含images目录，无法定位标签: {image}")
    parts[image_indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def select_val_samples(data_path: Path) -> dict[int, Path]:
    data_path = data_path.expanduser().resolve()
    if not data_path.is_file():
        raise RuntimeError(f"dataset.yaml不存在: {data_path}")
    config = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    data_names = normalize_names(config.get("names"))
    if data_names != EXPECTED_CLASSES:
        raise RuntimeError(f"数据集类别必须严格等于0-8九类配置: {data_names}")
    val_specs = config.get("val")
    if isinstance(val_specs, str):
        val_specs = [val_specs]
    if not isinstance(val_specs, list) or not val_specs:
        raise RuntimeError("dataset.yaml必须配置非空val")

    dataset_root = _resolve_dataset_root(data_path, config)
    images: list[Path] = []
    for spec in val_specs:
        images.extend(_images_from_spec(str(spec), dataset_root=dataset_root))
    if not images:
        raise RuntimeError("val没有图片")

    selected: dict[int, Path] = {}
    for image in images:
        label = label_path_for_image(image)
        if not image.is_file() or not label.is_file():
            raise RuntimeError(f"val图片/标签不成对: {image} / {label}")
        for line_number, raw_line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            fields = raw_line.split()
            if not fields:
                continue
            try:
                class_id = int(fields[0])
            except ValueError as exc:
                raise RuntimeError(f"标签类别ID无效: {label}:{line_number}") from exc
            if class_id not in EXPECTED_CLASSES:
                raise RuntimeError(f"标签类别ID越界: {label}:{line_number}: {class_id}")
            selected.setdefault(class_id, image)
        if len(selected) == len(EXPECTED_CLASSES):
            break
    missing = [EXPECTED_CLASSES[class_id] for class_id in EXPECTED_CLASSES if class_id not in selected]
    if missing:
        raise RuntimeError(f"val无法为每类选择至少一张样本: {missing}")
    return selected


def validate_metrics(
    model: Any,
    *,
    data_path: Path,
    device: str,
    imgsz: int,
    project: Path,
) -> dict[str, Any]:
    metrics = model.val(
        data=str(data_path),
        split="val",
        device=device,
        imgsz=imgsz,
        project=str(project),
        name="val",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    seg = getattr(metrics, "seg", None)
    if seg is None:
        raise RuntimeError("Ultralytics验证结果没有seg指标")
    class_indexes = [int(value) for value in seg.ap_class_index]
    missing = [class_id for class_id in EXPECTED_CLASSES if class_id not in class_indexes]
    if missing:
        raise RuntimeError(f"val指标缺少类别: {missing}")

    overall_fields = {
        "mask_map50_95": getattr(seg, "map", None),
        "mask_map50": getattr(seg, "map50", None),
        "mask_map75": getattr(seg, "map75", None),
        "mean_precision": getattr(seg, "mp", None),
        "mean_recall": getattr(seg, "mr", None),
    }
    overall = {
        name: finite_float(value, label=f"val {name}")
        for name, value in overall_fields.items()
    }
    per_class: dict[str, dict[str, float]] = {}
    for class_id, class_name in EXPECTED_CLASSES.items():
        index = class_indexes.index(class_id)
        values = seg.class_result(index)
        if len(values) != 4:
            raise RuntimeError(f"val类别指标字段数异常: {class_name}: {values}")
        per_class[class_name] = {
            name: finite_float(value, label=f"val {class_name} {name}")
            for name, value in zip(
                ("precision", "recall", "mask_map50", "mask_map50_95"), values
            )
        }
    for name, value in getattr(metrics, "results_dict", {}).items():
        finite_float(value, label=f"val results_dict {name}")
    return {**overall, "per_class": per_class}


def smoke_predict(
    model: Any,
    samples: dict[int, Path],
    *,
    device: str,
    imgsz: int,
) -> dict[str, str]:
    completed: dict[str, str] = {}
    for class_id, class_name in EXPECTED_CLASSES.items():
        source = samples[class_id]
        results = model.predict(
            source=str(source),
            device=device,
            imgsz=imgsz,
            save=False,
            verbose=False,
        )
        if results is None or len(results) < 1:
            raise RuntimeError(f"val推理没有返回结果: {class_name}: {source}")
        completed[class_name] = str(source)
    return completed


def validate_finalized_checkpoint(
    checkpoint: Any,
    *,
    label: str,
    expected_run_name: str,
) -> dict[str, Any]:
    """Require the stripped checkpoint shape produced by Ultralytics final_eval."""
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{label}不是Ultralytics checkpoint字典")
    if checkpoint.get("model") is None:
        raise RuntimeError(f"{label}缺少checkpoint model")

    epoch = checkpoint.get("epoch")
    active_fields = [name for name in STRIPPED_CHECKPOINT_FIELDS if checkpoint.get(name) is not None]
    if epoch != -1 or active_fields:
        details = [f"epoch={epoch!r}"]
        if active_fields:
            details.append("仍含训练态字段=" + ",".join(active_fields))
        raise RuntimeError(
            f"{label}: 训练未正常完成，禁止晋升；" + "；".join(details)
        )

    train_args = checkpoint.get("train_args")
    if not isinstance(train_args, dict) or not train_args:
        raise RuntimeError(f"{label}缺少有效train_args")
    task = str(train_args.get("task") or "").strip()
    name = str(train_args.get("name") or "").strip()
    if task != "segment":
        raise RuntimeError(f"{label} train_args.task必须为segment，实际为: {task or '<empty>'}")
    if name != expected_run_name:
        raise RuntimeError(
            f"{label} train_args.name与目标run不一致: {name or '<empty>'} != {expected_run_name}"
        )
    return {
        "epoch": -1,
        "stripped": True,
        "train_args": {"task": task, "name": name},
    }


def load_checkpoint_cpu(path: Path) -> Any:
    """Load a trusted local Ultralytics checkpoint without moving it to CUDA."""
    try:
        from ultralytics.utils.patches import torch_load
    except ImportError:
        import torch

        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")
    try:
        return torch_load(str(path), map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"无法读取checkpoint: {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    run_dir = args.run.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    missing_weights = [path for path in (weights, last_weights) if not path.is_file()]
    if missing_weights:
        raise SystemExit("Release验收要求best.pt和last.pt都存在: " + ", ".join(map(str, missing_weights)))

    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = validate_model_identity(model)
    try:
        checkpoints = {
            "best.pt": validate_finalized_checkpoint(
                getattr(model, "ckpt", None),
                label="best.pt",
                expected_run_name=run_dir.name,
            ),
            "last.pt": validate_finalized_checkpoint(
                load_checkpoint_cpu(last_weights),
                label="last.pt",
                expected_run_name=run_dir.name,
            ),
        }
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    training = inspect_results_csv(run_dir / "results.csv")
    samples = select_val_samples(data_path)
    with tempfile.TemporaryDirectory(prefix="atec_yolo_release_val_") as temp:
        validation = validate_metrics(
            model,
            data_path=data_path,
            device=args.device,
            imgsz=args.imgsz,
            project=Path(temp),
        )
    inference = smoke_predict(
        model,
        samples,
        device=args.device,
        imgsz=args.imgsz,
    )
    report = {
        "format_version": 1,
        "run": str(run_dir),
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "task": "segment",
        "names": names,
        "checkpoints": checkpoints,
        "training": training,
        "validation": validation,
        "inference_samples": inference,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"验收报告: {output}")
    else:
        print(rendered, end="")
    print("九类YOLO11-seg Release验收通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
