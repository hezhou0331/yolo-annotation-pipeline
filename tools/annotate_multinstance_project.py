#!/usr/bin/env python3
"""Manifest-driven multi-instance RGB-D annotation and safe YOLO aggregation.

A frame enters the formal dataset only when every declared active instance has
an eligible label and masks do not have a severe overlap conflict.  This avoids
turning a tracking failure into an unlabelled object (a false negative).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

WORKSPACE = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ATEC多实例自动标注")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--skip-tracking", action="store_true", help="复用_staging结果，只重新聚合")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--include-review", action="store_true", help="将review状态视作可进入正式数据；默认关闭")
    p.add_argument("--keep-conflicts", action="store_true", help="严重mask重叠仍进入正式数据；默认关闭")
    return p.parse_args()


def resolve_path(value: str | Path | None, base: Path, required: bool = False) -> Path | None:
    if value in (None, ""):
        if required:
            raise ValueError("缺少必填路径")
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value) or "instance"


def place_image(src: Path, dst: Path, mode: str = "hardlink") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def add_optional(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def add_review_override_arg(cmd: list[str], scene: Path, instance_id: str) -> Path:
    """Attach the stable, source-scene manual review sidecar to an export command."""
    path = Path(scene).expanduser().resolve() / "project_reports" / "manual_mask_review" / f"{safe_name(instance_id)}.json"
    cmd.extend(["--review-overrides", str(path)])
    return path


def run_tracker(
    instance: dict[str, Any],
    common: dict[str, Any],
    manifest_dir: Path,
    stage: Path,
    log_path: Path,
    dry_run: bool,
) -> None:
    tracker = instance.get("tracker", "foundationpose")
    scene = resolve_path(common["scene"], manifest_dir, required=True)
    prefix = common.get("name_prefix", f"{scene.name}_")
    split = common.get("split", "train")
    start = int(instance.get("start", 0))
    max_frames = int(instance.get("max_frames", 0))
    class_id = int(instance["class_id"])
    class_name = str(instance["class_name"])
    instance_id = safe_name(str(instance["instance_id"]))

    def mask_quality_command(mask_dir: Path) -> list[str]:
        cmd = [
            sys.executable, str(WORKSPACE / "tools" / "annotate_mask_sequence_yolo.py"),
            "--scene", str(scene), "--mask-dir", str(mask_dir),
            "--output", str(stage), "--class-id", str(class_id), "--class-name", class_name,
            "--instance-id", instance_id, "--split", split, "--start", str(start),
            "--max-frames", str(max_frames), "--name-prefix", str(prefix), "--image-mode", "none",
        ]
        quality = common.get("mask_quality", {}) | instance.get("quality", {})
        for key, flag in {
            "min_mask_area": "--min-mask-area",
            "min_depth_coverage": "--min-depth-coverage",
            "min_area_ratio": "--min-area-ratio",
            "max_area_ratio": "--max-area-ratio",
            "max_center_shift_norm": "--max-center-shift-norm",
            "min_dominant_component_ratio": "--min-dominant-component-ratio",
        }.items():
            if key in quality:
                add_optional(cmd, flag, quality[key])
        if common.get("include_review", False):
            cmd.append("--include-review")
        add_review_override_arg(cmd, scene, instance_id)
        return cmd

    commands: list[list[str]] = []
    if tracker == "foundationpose":
        script = WORKSPACE / "tools" / "annotate_foundationpose_quality.py"
        mesh = resolve_path(instance.get("mesh"), manifest_dir, required=True)
        first_mask = resolve_path(instance.get("first_mask"), manifest_dir)
        registration_dir = resolve_path(instance.get("registration_mask_dir"), manifest_dir)
        fp_dir = resolve_path(common.get("foundationpose_dir", WORKSPACE / "third_party/FoundationPose"), manifest_dir, required=True)
        cmd = [
            sys.executable, str(script), "--scene", str(scene), "--mesh", str(mesh),
            "--output", str(stage), "--foundationpose-dir", str(fp_dir),
            "--class-id", str(class_id), "--class-name", class_name,
            "--instance-id", instance_id, "--mesh-unit", str(instance.get("mesh_unit", "m")),
            "--task", "segmentation", "--split", split, "--start", str(start),
            "--max-frames", str(max_frames), "--name-prefix", str(prefix), "--image-mode", "none",
            "--inference-max-side", str(common.get("inference_max_side", 640)),
            "--max-consecutive-rejects", str(common.get("max_consecutive_rejects", 3)),
        ]
        add_optional(cmd, "--first-mask", first_mask)
        add_optional(cmd, "--registration-mask-dir", registration_dir)
        quality = common.get("quality", {}) | instance.get("quality", {})
        flag_map = {
            "min_mask_area": "--min-mask-area",
            "min_area_fraction": "--min-area-fraction",
            "max_area_fraction": "--max-area-fraction",
            "min_depth_coverage": "--min-depth-coverage",
            "max_depth_median_abs_m": "--max-depth-median-abs-m",
            "max_depth_rmse_m": "--max-depth-rmse-m",
            "min_area_ratio": "--min-area-ratio",
            "max_area_ratio": "--max-area-ratio",
            "max_center_shift_norm": "--max-center-shift-norm",
            "max_translation_jump_m": "--max-translation-jump-m",
            "max_rotation_jump_deg": "--max-rotation-jump-deg",
            "min_dominant_component_ratio": "--min-dominant-component-ratio",
            "min_registration_iou": "--min-registration-iou",
        }
        for key, flag in flag_map.items():
            if key in quality:
                add_optional(cmd, flag, quality[key])
        if common.get("include_review", False):
            cmd.append("--include-review")
        if common.get("no_visualization", False):
            cmd.append("--no-visualization")
        commands.append(cmd)
    elif tracker == "mask_sequence":
        mask_dir = resolve_path(instance.get("mask_dir"), manifest_dir, required=True)
        commands.append(mask_quality_command(mask_dir))
    elif tracker == "sam2":
        key_mask_dir = resolve_path(instance.get("key_mask_dir") or instance.get("registration_mask_dir"), manifest_dir, required=True)
        generated_mask_dir = stage / "_sam2_masks"
        default_python = Path.home() / "miniforge3" / "envs" / "yolo11" / "bin" / "python"
        sam2_python = resolve_path(common.get("sam2_python", default_python), manifest_dir, required=True)
        sam2_model = resolve_path(common.get("sam2_model", WORKSPACE / "models" / "sam2.1_t.pt"), manifest_dir, required=True)
        sam_cmd = [
            str(sam2_python), str(WORKSPACE / "tools" / "propagate_masks_sam2.py"),
            "--scene", str(scene), "--key-mask-dir", str(key_mask_dir),
            "--output-mask-dir", str(generated_mask_dir), "--model", str(sam2_model),
            "--device", str(common.get("sam2_device", 0)),
            "--imgsz", str(common.get("sam2_imgsz", 640)),
            "--start", str(start), "--max-frames", str(max_frames),
            "--memory-update-interval", str(instance.get("memory_update_interval", common.get("sam2_memory_update_interval", 5))),
            "--max-consecutive-failures", str(common.get("max_consecutive_rejects", 3)),
            "--auto-reregister-after-failures", str(instance.get("auto_reregister_after_failures", common.get("sam2_auto_reregister_after_failures", 1))),
            "--min-recovery-seed-iou", str(instance.get("min_recovery_seed_iou", common.get("sam2_min_recovery_seed_iou", 0.35))),
            "--max-flow-shift-norm", str(instance.get("max_flow_shift_norm", common.get("sam2_max_flow_shift_norm", 0.35))),
        ]
        if instance.get("auto_reregister", common.get("sam2_auto_reregister", True)):
            sam_cmd.append("--auto-reregister")
        else:
            sam_cmd.append("--no-auto-reregister")
        for key, flag in {
            "min_mask_area": "--min-mask-area",
            "min_area_ratio": "--min-area-ratio",
            "max_area_ratio": "--max-area-ratio",
            "min_depth_coverage": "--min-depth-coverage",
        }.items():
            value = (common.get("mask_quality", {}) | instance.get("quality", {})).get(key)
            if value is not None:
                add_optional(sam_cmd, flag, value)
        commands.extend([sam_cmd, mask_quality_command(generated_mask_dir)])
    else:
        raise ValueError(f"不支持的tracker：{tracker}")

    for cmd in commands:
        print("运行：", " ".join(cmd))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8") as log:
        for index, cmd in enumerate(commands, 1):
            log.write(f"\n===== step {index}/{len(commands)}: {' '.join(cmd)} =====\n")
            log.flush()
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)

def load_report(stage: Path, split: str, class_id: int, instance_id: str) -> tuple[dict, Path, Path]:
    instance_dir = f"class_{class_id:03d}_{safe_name(instance_id)}"
    report_path = stage / "quality_reports" / split / instance_dir / "quality_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"缺少质量报告：{report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report, stage / "labels" / split, stage / "rendered_masks" / split / instance_dir


def overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0; b = mask_b > 0
    denom = min(int(a.sum()), int(b.sum()))
    if denom <= 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / denom)


def render_review_image(rgb_path: Path, items: list[dict], output_path: Path) -> None:
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    colors = [(0, 0, 255), (0, 180, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]
    for index, item in enumerate(items):
        mask = cv2.imread(str(item["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        color = colors[index % len(colors)]
        overlay = image.copy(); overlay[mask > 0] = color
        image = cv2.addWeighted(image, 0.78, overlay, 0.22, 0)
        ys, xs = np.where(mask > 0)
        if xs.size:
            cv2.putText(image, item["instance_id"], (int(xs.min()), max(18, int(ys.min()))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest_dir = manifest_path.parent
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit("manifest必须是YAML对象")
    common = manifest.get("project", {})
    instances = manifest.get("instances", [])
    if not instances:
        raise SystemExit("manifest中没有instances")
    scene = resolve_path(common.get("scene"), manifest_dir, required=True)
    output = resolve_path(common.get("output"), manifest_dir, required=True)
    split = common.get("split", "train")
    prefix = common.get("name_prefix", f"{scene.name}_")
    stage_root = output / "_staging" / safe_name(scene.name)
    log_root = output / "logs" / safe_name(scene.name)
    output.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    classes: dict[int, str] = {}
    raw_classes = manifest.get("classes", {})
    if isinstance(raw_classes, list):
        classes = {index: str(name) for index, name in enumerate(raw_classes)}
    elif isinstance(raw_classes, dict):
        classes = {int(class_id): str(name) for class_id, name in raw_classes.items()}
    elif raw_classes:
        raise SystemExit("manifest.classes必须是列表或ID到名称的映射")
    stages: dict[str, Path] = {}
    for instance in instances:
        for key in ("instance_id", "class_id", "class_name"):
            if key not in instance:
                raise SystemExit(f"instance缺少{key}：{instance}")
        instance_id = safe_name(str(instance["instance_id"]))
        if instance_id in seen_ids:
            raise SystemExit(f"instance_id重复：{instance_id}")
        seen_ids.add(instance_id)
        class_id = int(instance["class_id"])
        class_name = str(instance["class_name"])
        if class_id in classes and classes[class_id] != class_name:
            raise SystemExit(f"class_id {class_id}名称冲突")
        classes[class_id] = class_name
        stage = stage_root / instance_id
        stages[instance_id] = stage
        if not args.skip_tracking:
            run_tracker(instance, common, manifest_dir, stage, log_root / f"{instance_id}.log", args.dry_run)
    if args.dry_run:
        return 0

    # 根据scene帧和每个实例的start/max_frames推导“本帧应出现哪些实例”。
    # 不能只相信tracker实际写出的report，否则某实例整帧漏报会被误当成完整标注。
    rgb_files = sorted((scene / "rgb").glob("*.png"))
    expected_instances: dict[str, set[str]] = defaultdict(set)
    source_frame_by_output: dict[str, str] = {}
    for instance in instances:
        instance_id = safe_name(str(instance["instance_id"]))
        start = max(0, int(instance.get("start", 0)))
        max_frames = int(instance.get("max_frames", 0))
        end = len(rgb_files) if max_frames <= 0 else min(len(rgb_files), start + max_frames)
        for rgb_path in rgb_files[start:end]:
            output_id = f"{prefix}{rgb_path.stem}"
            expected_instances[output_id].add(instance_id)
            source_frame_by_output[output_id] = rgb_path.stem

    frame_items: dict[str, list[dict]] = defaultdict(list)
    all_instance_reports = []
    for instance in instances:
        instance_id = safe_name(str(instance["instance_id"]))
        class_id = int(instance["class_id"])
        report, label_dir, mask_dir = load_report(stages[instance_id], split, class_id, instance_id)
        all_instance_reports.append({
            "instance_id": instance_id,
            "class_id": class_id,
            "class_name": instance["class_name"],
            "tracker": instance.get("tracker", "foundationpose"),
            "stage": str(stages[instance_id]),
            "status_counts": report.get("status_counts", {}),
        })
        for record in report["records"]:
            output_id = record["output_id"]
            label_path = label_dir / f"{output_id}.txt"
            label = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else ""
            frame_items[output_id].append({
                "instance_id": instance_id,
                "class_id": class_id,
                "class_name": instance["class_name"],
                "status": record["status"],
                "has_label": bool(record.get("has_label")) and bool(label),
                "label": label,
                "mask_path": mask_dir / f"{output_id}.png",
                "source_frame": record["id"],
                "reject_reasons": record.get("reject_reasons", []),
                "review_reasons": record.get("review_reasons", []),
            })

    final_images = output / "images" / split
    final_labels = output / "labels" / split
    review_images = output / "review" / split / "images"
    review_labels = output / "review" / split / "candidate_labels"
    for d in (final_images, final_labels, review_images, review_labels):
        d.mkdir(parents=True, exist_ok=True)

    frame_records = []
    overlap_limit = float(common.get("max_instance_overlap", 0.20))
    image_mode = common.get("image_mode", "hardlink")
    all_output_ids = sorted(set(expected_instances) | set(frame_items))
    for output_id in all_output_ids:
        items = frame_items.get(output_id, [])
        stale_paths = (
            list(final_labels.glob(f"{output_id}.*")) + list(final_images.glob(f"{output_id}.*"))
            + list(review_labels.glob(f"{output_id}.*")) + list(review_images.glob(f"{output_id}.*"))
        )
        for existing in stale_paths:
            existing.unlink()
        reasons = []
        expected_ids = expected_instances.get(output_id, set())
        observed_ids = {item["instance_id"] for item in items}
        for missing_id in sorted(expected_ids - observed_ids):
            reasons.append(f"{missing_id}:record_missing")
        accepted_statuses = {"accepted"} | ({"review"} if (args.include_review or common.get("include_review", False)) else set())
        for item in items:
            if item["status"] not in accepted_statuses or not item["has_label"]:
                reasons.append(f"{item['instance_id']}:{item['status']}")

        overlaps = []
        for i in range(len(items)):
            mask_i = cv2.imread(str(items[i]["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if mask_i is None:
                reasons.append(f"{items[i]['instance_id']}:mask_missing")
                continue
            for j in range(i + 1, len(items)):
                mask_j = cv2.imread(str(items[j]["mask_path"]), cv2.IMREAD_GRAYSCALE)
                if mask_j is None:
                    continue
                ratio = overlap_ratio(mask_i, mask_j)
                if ratio > overlap_limit:
                    overlaps.append({"a": items[i]["instance_id"], "b": items[j]["instance_id"], "ratio": ratio})
        if overlaps and not (args.keep_conflicts or common.get("keep_conflicts", False)):
            reasons.append("instance_mask_overlap")

        source_stem = items[0]["source_frame"] if items else source_frame_by_output.get(output_id)
        if source_stem is None:
            reasons.append("source_frame_unknown")
            rgb_path = scene / "rgb" / "__missing__.png"
        else:
            rgb_path = scene / "rgb" / f"{source_stem}.png"
        if not rgb_path.exists():
            reasons.append("source_image_missing")
        candidate_lines = [item["label"] for item in items if item["label"]]
        if not reasons and candidate_lines:
            (final_labels / f"{output_id}.txt").write_text("\n".join(candidate_lines) + "\n", encoding="utf-8")
            place_image(rgb_path, final_images / f"{output_id}.png", image_mode)
            status = "accepted"
        else:
            status = "review" if candidate_lines else "rejected"
            if candidate_lines:
                (review_labels / f"{output_id}.txt").write_text("\n".join(candidate_lines) + "\n", encoding="utf-8")
                render_review_image(rgb_path, items, review_images / f"{output_id}.jpg")
        frame_records.append({
            "output_id": output_id, "source_frame": source_stem, "status": status,
            "reasons": reasons, "overlaps": overlaps,
            "expected_instances": sorted(expected_ids),
            "observed_instances": sorted(observed_ids),
            "instances": [{k: v for k, v in item.items() if k not in ("label", "mask_path")} for item in items],
        })

    classes = dict(sorted(classes.items()))
    (output / "classes.json").write_text(json.dumps({str(k): v for k, v in classes.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    for required_id in range(max(classes) + 1):
        if required_id not in classes:
            raise SystemExit(f"YOLO类别ID必须连续，缺少{required_id}")
    val_has = any((output / "images" / "val").glob("*.png"))
    test_has = any((output / "images" / "test").glob("*.png"))
    allow_val_fallback = bool(common.get("allow_val_fallback_for_smoke", False))
    val_rel = "images/val" if val_has or not allow_val_fallback else "images/train"
    test_rel = "images/test" if test_has else None
    names_yaml = "".join(f"  {k}: {v}\n" for k, v in classes.items())
    test_yaml = f"test: {test_rel}\n" if test_rel else ""
    (output / "dataset.yaml").write_text(
        f"path: {output}\ntrain: images/train\nval: {val_rel}\n{test_yaml}names:\n{names_yaml}", encoding="utf-8"
    )

    counts = {s: sum(r["status"] == s for r in frame_records) for s in ("accepted", "review", "rejected")}
    project_report = {
        "manifest": str(manifest_path), "scene": str(scene), "output": str(output), "split": split,
        "capture_session_id": common.get("capture_session_id"),
        "source_video_id": common.get("source_video_id"),
        "classes": classes, "instances": all_instance_reports, "frame_status_counts": counts,
        "max_instance_overlap": overlap_limit, "frames": frame_records,
        "dataset_split_policy": {
            "val_has_images": val_has,
            "test_has_images": test_has,
            "allow_val_fallback_for_smoke": allow_val_fallback,
            "dataset_yaml_val": val_rel,
            "dataset_yaml_test": test_rel,
        },
        "safety_rule": "只有所有声明实例均通过质量门槛且无严重mask冲突的帧才进入正式YOLO数据集。",
    }
    report_dir = output / "project_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"{safe_name(scene.name)}_{split}_report"
    (report_dir / f"{report_name}.json").write_text(json.dumps(project_report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (report_dir / f"{report_name}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["output_id", "source_frame", "status", "reasons", "overlaps", "instance_count"])
        writer.writeheader()
        for record in frame_records:
            writer.writerow({
                "output_id": record["output_id"], "source_frame": record["source_frame"], "status": record["status"],
                "reasons": "|".join(record["reasons"]), "overlaps": json.dumps(record["overlaps"], ensure_ascii=False),
                "instance_count": len(record["instances"]),
            })
    print(f"聚合完成：accepted={counts['accepted']} review={counts['review']} rejected={counts['rejected']}，输出={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
