#!/usr/bin/env python3
"""Use local xcx YOLO11 detectors to propose SAM2 key masks for manual review.

The tool deliberately writes only to a proposal staging directory.  It never
assigns a manifest instance automatically and never overwrites an accepted key
mask.  ``promote`` is an explicit reviewer action.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CLASSES = {
    0: "can",
    1: "watermelon_rind",
    2: "meal_box",
    3: "red_paper_bag",
    4: "blue_bin",
    5: "green_bin",
    6: "red_bin",
}


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML顶层必须是映射: {path}")
    return data


def normalized_classes(raw: Any) -> dict[int, str]:
    if not isinstance(raw, dict):
        raise ValueError("classes必须是ID到名称的映射")
    return {int(key): str(value) for key, value in raw.items()}


def validate_teacher_config(config: dict[str, Any]) -> None:
    classes = normalized_classes(config.get("official_classes"))
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"教师配置必须严格保持当前7类，实际为: {classes}")
    mapping = {str(k): str(v) for k, v in config.get("source_to_official", {}).items()}
    unknown_targets = sorted(set(mapping.values()) - set(EXPECTED_CLASSES.values()))
    if unknown_targets:
        raise ValueError(f"映射包含非正式类别: {unknown_targets}")
    negatives = {str(x) for x in config.get("hard_negative_classes", [])}
    overlap = sorted(set(mapping) & negatives)
    if overlap:
        raise ValueError(f"同一xcx类别不能同时是正式映射和负样本: {overlap}")
    if not config.get("detectors"):
        raise ValueError("教师配置没有detectors")


def load_context(manifest_path: Path, segments_path: Path, config_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    segments_path = segments_path.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    manifest = load_yaml(manifest_path)
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    config = load_yaml(config_path)
    validate_teacher_config(config)
    manifest_classes = normalized_classes(manifest.get("classes"))
    if manifest_classes != EXPECTED_CLASSES:
        raise ValueError("manifest不是当前固定7类，拒绝生成候选Mask")
    project = manifest.get("project", {})
    scene = resolve_path(project["scene"], manifest_path.parent)
    instances = manifest.get("instances", [])
    if not instances:
        raise ValueError("manifest没有instances，无法进行人工实例归属")
    instance_by_id = {str(item["instance_id"]): item for item in instances}
    if len(instance_by_id) != len(instances):
        raise ValueError("manifest存在重复instance_id")
    return {
        "manifest_path": manifest_path,
        "segments_path": segments_path,
        "config_path": config_path,
        "manifest": manifest,
        "segments": segments,
        "config": config,
        "scene": scene,
        "instance_by_id": instance_by_id,
    }


def missing_instances_by_class(segment: dict[str, Any], instance_by_id: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    missing = set(str(x) for x in segment.get("missing_key_masks", []))
    if not missing:
        for instance_id, path in segment.get("required_key_mask_paths", {}).items():
            if path and not Path(path).exists():
                missing.add(str(instance_id))
    for instance_id in sorted(missing):
        item = instance_by_id.get(instance_id)
        if item is not None:
            result[str(item["class_name"])].append(instance_id)
    return dict(result)


def classify_detection(
    source_class: str,
    detector_config: dict[str, Any],
    mapping: dict[str, str],
    hard_negatives: set[str],
    missing_by_class: dict[str, list[str]],
) -> dict[str, Any]:
    if source_class in hard_negatives:
        return {"kind": "hard_negative"}
    official_class = mapping.get(source_class)
    if official_class is None:
        return {"kind": "crosscheck", "classification": "unmapped_source_class"}
    allowed = detector_config.get("allowed_official_classes")
    if allowed and official_class not in {str(x) for x in allowed}:
        return {"kind": "crosscheck", "official_class": official_class, "classification": "detector_scope_mismatch"}
    if not detector_config.get("proposal_source", False):
        return {"kind": "crosscheck", "official_class": official_class, "classification": "crosscheck_only"}
    candidate_instances = list(missing_by_class.get(official_class, []))
    if not candidate_instances:
        return {"kind": "crosscheck", "official_class": official_class, "classification": "no_missing_manifest_instance"}
    return {"kind": "proposal", "official_class": official_class, "candidate_instance_ids": candidate_instances}


def extract_detections(result: Any, detector_name: str) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", {})
    if boxes is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    detections = []
    for bbox, class_id, confidence in zip(xyxy, classes, confidences):
        cid = int(class_id)
        name = str(names[cid] if isinstance(names, (dict, list, tuple)) else cid)
        detections.append({
            "detector": detector_name,
            "source_class_id": cid,
            "source_class": name,
            "confidence": float(confidence),
            "bbox_xyxy": [float(x) for x in bbox.tolist()],
        })
    return detections


def sam_mask_from_box(sam: Any, image_path: Path, bbox: list[float], device: str, imgsz: int) -> np.ndarray:
    results = sam.predict(
        source=str(image_path), bboxes=[bbox], device=device, imgsz=imgsz,
        verbose=False, save=False,
    )
    if not results:
        raise RuntimeError("SAM2没有返回结果")
    masks = getattr(results[0], "masks", None)
    data = getattr(masks, "data", None)
    if data is None or len(data) == 0:
        raise RuntimeError("SAM2没有为候选框返回Mask")
    array = data[0].detach().cpu().numpy()
    if array.ndim != 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise RuntimeError(f"SAM2 Mask维度异常: {array.shape}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取关键帧: {image_path}")
    height, width = image.shape[:2]
    if array.shape != (height, width):
        array = cv2.resize(array.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    mask = (array > 0.5).astype(np.uint8) * 255
    if not np.any(mask):
        raise RuntimeError("SAM2返回空Mask")
    return mask


def render_overlay(image: np.ndarray, mask: np.ndarray, bbox: list[float], label: str) -> np.ndarray:
    overlay = image.copy()
    colored = np.zeros_like(image)
    colored[:, :, 1] = 255
    alpha = (mask > 0)[:, :, None]
    overlay = np.where(alpha, (0.55 * overlay + 0.45 * colored).astype(np.uint8), overlay)
    x1, y1, x2, y2 = (int(round(x)) for x in bbox)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.putText(overlay, label, (max(0, x1), max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    return overlay


def proposal_root(context: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    value = context["config"].get("proposal_policy", {}).get("staging_root", "../projects/atec_real/data/candidate_masks")
    return resolve_path(value, context["config_path"].parent) / context["scene"].name


def check_assets(context: dict[str, Any]) -> list[str]:
    missing = []
    config = context["config"]
    for detector in config["detectors"]:
        path = resolve_path(detector["weights"], context["config_path"].parent)
        if not path.is_file():
            missing.append(f"检测权重不存在: {path}")
    sam_model = resolve_path(config["sam2"]["model"], context["config_path"].parent)
    if not sam_model.is_file():
        missing.append(f"SAM2权重不存在: {sam_model}")
    if not (context["scene"] / "rgb").is_dir():
        missing.append(f"场景RGB目录不存在: {context['scene'] / 'rgb'}")
    return missing


def propose(args: argparse.Namespace) -> int:
    context = load_context(args.manifest, args.segments, args.teacher_config)
    if str(context["manifest"].get("project", {}).get("split", "train")) == "val":
        raise SystemExit("验证集必须保持人工独立真值，拒绝在val manifest上生成xcx候选Mask")
    missing_assets = check_assets(context)
    summary = {
        "manifest": str(context["manifest_path"]),
        "segments": str(context["segments_path"]),
        "teacher_config": str(context["config_path"]),
        "scene": str(context["scene"]),
        "asset_errors": missing_assets,
    }
    if args.check_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if not missing_assets else 2
    if missing_assets:
        raise SystemExit("\n".join(missing_assets))

    from ultralytics import SAM, YOLO

    config = context["config"]
    inference = config.get("inference", {})
    mapping = {str(k): str(v) for k, v in config["source_to_official"].items()}
    hard_negatives = {str(x) for x in config.get("hard_negative_classes", [])}
    detector_models = []
    for detector in config["detectors"]:
        weights = resolve_path(detector["weights"], context["config_path"].parent)
        detector_models.append((detector, YOLO(str(weights))))
    sam_cfg = config["sam2"]
    sam = SAM(str(resolve_path(sam_cfg["model"], context["config_path"].parent)))

    root = proposal_root(context, args.output)
    root.mkdir(parents=True, exist_ok=True)
    proposals: list[dict[str, Any]] = []
    hard_negative_records: list[dict[str, Any]] = []
    crosschecks: list[dict[str, Any]] = []
    selected_segments = set(args.segment_id or [])

    for segment in context["segments"].get("segments", []):
        if selected_segments and int(segment["segment_id"]) not in selected_segments:
            continue
        missing_by_class = missing_instances_by_class(segment, context["instance_by_id"])
        if not missing_by_class:
            continue
        frame_id = str(segment["start_id"])
        image_path = context["scene"] / "rgb" / f"{frame_id}.png"
        if not image_path.is_file():
            raise RuntimeError(f"分段关键帧不存在: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取分段关键帧: {image_path}")
        frame_dir = root / frame_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        for detector_cfg, model in detector_models:
            detector_name = str(detector_cfg["name"])
            result_list = model.predict(
                source=str(image_path), conf=float(inference.get("confidence", 0.25)),
                iou=float(inference.get("iou", 0.60)), imgsz=int(inference.get("imgsz", 640)),
                device=str(inference.get("device", 0)), verbose=False, save=False,
            )
            detections = extract_detections(result_list[0], detector_name) if result_list else []
            for det_index, detection in enumerate(detections):
                source_class = detection["source_class"]
                base_record = {"segment_id": int(segment["segment_id"]), "frame_id": frame_id, **detection}
                classification = classify_detection(
                    source_class, detector_cfg, mapping, hard_negatives, missing_by_class
                )
                if classification["kind"] == "hard_negative":
                    hard_negative_records.append(base_record)
                    continue
                if classification["kind"] == "crosscheck":
                    crosschecks.append({**base_record, **{k: v for k, v in classification.items() if k != "kind"}})
                    continue
                official_class = classification["official_class"]
                candidate_instances = classification["candidate_instance_ids"]

                proposal_id = f"{frame_id}__{official_class}__{detector_name}__{det_index:02d}"
                mask = sam_mask_from_box(
                    sam, image_path, detection["bbox_xyxy"], str(sam_cfg.get("device", 0)), int(sam_cfg.get("imgsz", 640))
                )
                mask_path = frame_dir / f"{proposal_id}.png"
                overlay_path = frame_dir / f"{proposal_id}_overlay.jpg"
                proposal_path = frame_dir / f"{proposal_id}.json"
                if mask_path.exists() or overlay_path.exists() or proposal_path.exists():
                    raise FileExistsError(f"候选文件已存在，拒绝覆盖: {proposal_path}")
                cv2.imwrite(str(mask_path), mask)
                overlay = render_overlay(image, mask, detection["bbox_xyxy"], f"{official_class} {detection['confidence']:.2f}")
                cv2.imwrite(str(overlay_path), overlay)
                record = {
                    "format_version": 1,
                    "proposal_id": proposal_id,
                    "status": "candidate_requires_manual_review",
                    "manifest": str(context["manifest_path"]),
                    "scene": str(context["scene"]),
                    "segment_id": int(segment["segment_id"]),
                    "frame_id": frame_id,
                    "image": str(image_path),
                    "candidate_mask": str(mask_path),
                    "overlay": str(overlay_path),
                    "official_class": official_class,
                    "candidate_instance_ids": candidate_instances,
                    "requires_manual_instance_assignment": True,
                    "teacher_detection": detection,
                    "review_command_template": (
                        f"{ROOT / 'scripts/atec-pipeline'} mask {image_path} REPLACE_ACCEPTED_MASK_PATH "
                        f"--mask-input {mask_path}"
                    ),
                }
                proposal_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                record["proposal_json"] = str(proposal_path)
                proposals.append(record)

    report = {
        **summary,
        "format_version": 1,
        "proposal_root": str(root),
        "proposal_count": len(proposals),
        "hard_negative_detection_count": len(hard_negative_records),
        "crosscheck_detection_count": len(crosschecks),
        "proposals": proposals,
        "hard_negative_detections": hard_negative_records,
        "crosscheck_detections": crosschecks,
        "safety_rules": [
            "候选Mask不是训练真值",
            "负样本类别不生成正式Mask",
            "同类多实例必须由人工选择instance_id",
            "promote命令拒绝覆盖已接受Mask",
        ],
    }
    report_path = root / "proposal_report.json"
    if report_path.exists():
        raise FileExistsError(f"候选报告已存在，拒绝覆盖: {report_path}")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"候选生成完成：Mask={len(proposals)}，干扰检测={len(hard_negative_records)}，交叉检查={len(crosschecks)}")
    print(f"报告：{report_path}")
    return 0


def promote(args: argparse.Namespace) -> int:
    proposal_path = args.proposal_json.expanduser().resolve()
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    manifest_path = Path(proposal["manifest"]).expanduser().resolve()
    manifest = load_yaml(manifest_path)
    instances = {str(item["instance_id"]): item for item in manifest.get("instances", [])}
    instance = instances.get(args.instance_id)
    if instance is None:
        raise SystemExit(f"manifest不存在instance_id: {args.instance_id}")
    if str(instance["class_name"]) != str(proposal["official_class"]):
        raise SystemExit("候选类别与所选instance类别不一致，拒绝晋升")
    candidates = [str(x) for x in proposal.get("candidate_instance_ids", [])]
    if args.instance_id not in candidates:
        raise SystemExit("所选instance不在该分段缺失实例候选中，拒绝晋升")
    key_dir_value = instance.get("key_mask_dir") or instance.get("registration_mask_dir")
    if not key_dir_value:
        raise SystemExit("所选instance没有key_mask_dir/registration_mask_dir")
    key_dir = resolve_path(key_dir_value, manifest_path.parent)
    destination = key_dir / f"{proposal['frame_id']}.png"
    if destination.exists():
        raise SystemExit(f"已接受Mask存在，拒绝覆盖: {destination}")
    source = Path(proposal["candidate_mask"]).expanduser().resolve()
    mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if mask is None or not np.any(mask > 0):
        raise SystemExit(f"候选Mask不存在或为空: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    receipt = {
        "format_version": 1,
        "action": "promote_reviewed_candidate",
        "proposal_json": str(proposal_path),
        "instance_id": args.instance_id,
        "official_class": proposal["official_class"],
        "destination": str(destination),
        "reviewer_confirmation_required": True,
    }
    receipt_path = destination.with_suffix(".promotion.json")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已晋升人工确认候选Mask：{destination}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="xcx检测框 + SAM2关键Mask候选工具")
    sub = p.add_subparsers(dest="action", required=True)
    x = sub.add_parser("propose", help="在缺少关键Mask的分段起始帧生成候选")
    x.add_argument("--manifest", type=Path, required=True)
    x.add_argument("--segments", type=Path, required=True)
    x.add_argument("--teacher-config", type=Path, default=ROOT / "configs/xcx_teacher.yaml")
    x.add_argument("--output", type=Path)
    x.add_argument("--segment-id", type=int, action="append", help="可重复，仅处理指定segment_id")
    x.add_argument("--check-only", action="store_true", help="只校验配置、权重和输入，不加载模型")
    x.set_defaults(func=propose)
    x = sub.add_parser("promote", help="将人工确认的候选显式晋升到某个manifest实例")
    x.add_argument("--proposal-json", type=Path, required=True)
    x.add_argument("--instance-id", required=True)
    x.set_defaults(func=promote)
    return p


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
