#!/usr/bin/env python3
"""Create an ATEC annotation project without overwriting user data."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atec_pipeline.object_config import load_class_map
CLASSES = load_class_map(ROOT / "configs" / "atec_objects.yaml")


def parse_args():
    p = argparse.ArgumentParser(description="创建ATEC多实例标注项目目录")
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--scene-name", required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--capture-session-id", help="不可变采集场次ID，例如session_01；正式训练前必须填写")
    p.add_argument("--source-video-id", help="不可变源视频ID，例如session_01_mixed_01；复制目录后也不得修改")
    p.add_argument(
        "--can-tracker",
        choices=("sam2", "foundationpose"),
        default="sam2",
        help="易拉罐默认也使用SAM2，只有已经准备好米制mesh时才选择foundationpose",
    )
    p.add_argument("--instances-per-class", type=int, default=1, help="为启用的比赛物品预建多少个实例")
    p.add_argument("--only-class", choices=tuple(CLASSES.values()), help="只启用一个类别；适合按类别分别采集")
    p.add_argument("--scene-class", choices=tuple(CLASSES.values()), help="将场景写入 data/scenes/<scene-class>/<scene-name>")
    p.add_argument("--include-bins", action="store_true", help="同时加入蓝/绿/红垃圾桶各一个SAM2实例")
    p.add_argument("--force-manifest", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.instances_per_class < 1:
        raise SystemExit("--instances-per-class必须至少为1")
    root = args.project_root.expanduser().resolve()
    scene = root / "data" / "scenes" / args.scene_name
    if args.scene_class:
        scene = root / "data" / "scenes" / args.scene_class / args.scene_name
    for path in (
        scene / "rgb", scene / "depth",
        root / "data" / "key_masks" / args.scene_name,
        root / "data" / "tracked_masks" / args.scene_name,
        root / "assets" / "meshes", root / "manifests", root / "datasets", root / "reports",
    ):
        path.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifests" / f"{args.scene_name}_{args.split}.yaml"
    if manifest_path.exists() and not args.force_manifest:
        raise SystemExit(f"manifest已存在，不覆盖：{manifest_path}；如确认覆盖请加--force-manifest")
    manifest_dir = manifest_path.parent

    def portable(path: Path) -> str:
        return os.path.relpath(path, manifest_dir)

    instances = []
    defaults = []
    object_trackers = {class_id: "sam2" for class_id in CLASSES}
    object_trackers[0] = args.can_tracker
    if args.only_class:
        selected_class_ids = {next(class_id for class_id, name in CLASSES.items() if name == args.only_class)}
        for class_id in selected_class_ids:
            for number in range(1, args.instances_per_class + 1):
                defaults.append((f"{CLASSES[class_id]}_{number:02d}", class_id, object_trackers[class_id]))
    else:
        selected_class_ids = set(range(4)) | ({4, 5, 6} if args.include_bins else set())
        for class_id in range(4):
            for number in range(1, args.instances_per_class + 1):
                defaults.append((f"{CLASSES[class_id]}_{number:02d}", class_id, object_trackers[class_id]))
        if args.include_bins:
            for class_id in (4, 5, 6):
                defaults.append((f"{CLASSES[class_id]}_01", class_id, object_trackers[class_id]))
    for instance_id, class_id, tracker in defaults:
        entry = {
            "instance_id": instance_id, "class_id": class_id, "class_name": CLASSES[class_id],
            "tracker": tracker, "start": 0, "max_frames": 0,
        }
        if tracker == "foundationpose":
            entry.update({
                "mesh": portable(root / "assets" / "meshes" / instance_id / "textured_mesh.obj"),
                "mesh_unit": "m",
                "registration_mask_dir": portable(root / "data" / "key_masks" / args.scene_name / instance_id),
            })
        elif tracker == "sam2":
            entry["key_mask_dir"] = portable(root / "data" / "key_masks" / args.scene_name / instance_id)
        else:
            entry["mask_dir"] = portable(root / "data" / "tracked_masks" / args.scene_name / instance_id)
        instances.append(entry)
    manifest = {
        "classes": CLASSES,
        "project": {
            "scene": portable(scene), "output": portable(root / "datasets" / "atec_yolo11_seg"),
            "foundationpose_dir": portable(ROOT / "third_party/FoundationPose"),
            "sam2_model": portable(ROOT / "models/sam2.1_t.pt"), "sam2_device": 0, "sam2_imgsz": 640,
            "sam2_memory_update_interval": 5,
            "sam2_auto_reregister": True, "sam2_auto_reregister_after_failures": 1,
            "sam2_min_recovery_seed_iou": 0.35, "sam2_max_flow_shift_norm": 0.35,
            "split": args.split,
            "capture_session_id": args.capture_session_id,
            "source_video_id": args.source_video_id,
            "name_prefix": f"{args.scene_name}_", "image_mode": "hardlink",
            "inference_max_side": 640, "max_consecutive_rejects": 3,
            "max_instance_overlap": 0.20, "include_review": False, "keep_conflicts": False,
            "allow_val_fallback_for_smoke": False,
            "quality": {
                "min_mask_area": 80, "min_area_fraction": 0.00015, "max_area_fraction": 0.65,
                "min_depth_coverage": 0.45, "max_depth_median_abs_m": 0.05, "max_depth_rmse_m": 0.08,
                "min_area_ratio": 0.45, "max_area_ratio": 2.20, "max_center_shift_norm": 0.35,
                "max_translation_jump_m": 0.20, "max_rotation_jump_deg": 55,
                "min_dominant_component_ratio": 0.75, "min_registration_iou": 0.45,
            },
            "mask_quality": {
                "min_mask_area": 80, "min_depth_coverage": 0.25, "min_area_ratio": 0.35,
                "max_area_ratio": 2.80, "max_center_shift_norm": 0.40,
                "min_dominant_component_ratio": 0.70,
            },
        },
        "instances": instances,
    }
    manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"项目已创建：{root}")
    print(f"采集目录：{scene}")
    print(f"manifest：{manifest_path}")
    print(f"manifest已预建：{len(selected_class_ids)}个类别、每类{args.instances_per_class}个实例。")
    print("请删除本段视频里实际不可见的实例；声明存在但跟丢的实例会使整帧进入review。")
    if not args.capture_session_id or not args.source_video_id:
        print("[警告] 尚未填写capture_session_id/source_video_id；严格数据验证会拒绝缺少来源ID的报告。")
    if args.can_tracker == "sam2":
        print("无CAD安全默认：易拉罐当前使用SAM2。准备好尺寸正确的mesh后，再把tracker改为foundationpose。")
    else:
        print("已选择FoundationPose易拉罐分支：运行前必须准备米制mesh和每段关键帧mask。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
