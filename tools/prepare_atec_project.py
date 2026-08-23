#!/usr/bin/env python3
"""Create an ATEC annotation project without overwriting user data."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {0: "can", 1: "watermelon_rind", 2: "meal_box", 3: "red_paper_bag", 4: "blue_bin", 5: "green_bin", 6: "red_bin"}


def parse_args():
    p = argparse.ArgumentParser(description="创建ATEC多实例标注项目目录")
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--scene-name", required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument(
        "--can-tracker",
        choices=("sam2", "foundationpose"),
        default="sam2",
        help="易拉罐默认也使用SAM2，只有已经准备好米制mesh时才选择foundationpose",
    )
    p.add_argument("--instances-per-class", type=int, default=1, help="为4类比赛物品各预建多少个实例")
    p.add_argument("--include-bins", action="store_true", help="同时加入蓝/绿/红垃圾桶各一个SAM2实例")
    p.add_argument("--force-manifest", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.instances_per_class < 1:
        raise SystemExit("--instances-per-class必须至少为1")
    root = args.project_root.expanduser().resolve()
    scene = root / "data" / "scenes" / args.scene_name
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
    instances = []
    defaults = []
    object_trackers = {0: args.can_tracker, 1: "sam2", 2: "sam2", 3: "sam2"}
    for class_id, tracker in object_trackers.items():
        for number in range(1, args.instances_per_class + 1):
            defaults.append((f"{CLASSES[class_id]}_{number:02d}", class_id, tracker))
    if args.include_bins:
        for class_id in (4, 5, 6):
            defaults.append((f"{CLASSES[class_id]}_01", class_id, "sam2"))
    for instance_id, class_id, tracker in defaults:
        entry = {
            "instance_id": instance_id, "class_id": class_id, "class_name": CLASSES[class_id],
            "tracker": tracker, "start": 0, "max_frames": 0,
        }
        if tracker == "foundationpose":
            entry.update({
                "mesh": str(root / "assets" / "meshes" / instance_id / "textured_mesh.obj"),
                "mesh_unit": "m",
                "registration_mask_dir": str(root / "data" / "key_masks" / args.scene_name / instance_id),
            })
        elif tracker == "sam2":
            entry["key_mask_dir"] = str(root / "data" / "key_masks" / args.scene_name / instance_id)
        else:
            entry["mask_dir"] = str(root / "data" / "tracked_masks" / args.scene_name / instance_id)
        instances.append(entry)
    manifest = {
        "classes": CLASSES,
        "project": {
            "scene": str(scene), "output": str(root / "datasets" / "atec_yolo11_seg"),
            "foundationpose_dir": str(ROOT / "third_party/FoundationPose"),
            "sam2_python": str(Path.home() / "miniforge3/envs/yolo11/bin/python"),
            "sam2_model": str(ROOT / "models/sam2.1_t.pt"), "sam2_device": 0, "sam2_imgsz": 640,
            "sam2_memory_update_interval": 5,
            "sam2_auto_reregister": True, "sam2_auto_reregister_after_failures": 1,
            "sam2_min_recovery_seed_iou": 0.35, "sam2_max_flow_shift_norm": 0.35,
            "split": args.split,
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
    shutil.copy2(ROOT / "configs" / "atec_objects.yaml", root / "atec_objects.yaml")
    print(f"项目已创建：{root}")
    print(f"采集目录：{scene}")
    print(f"manifest：{manifest_path}")
    print(f"manifest已预建：4类物品各{args.instances_per_class}个实例" + ("，以及3个垃圾桶实例。" if args.include_bins else "。"))
    print("请删除本段视频里实际不可见的实例；声明存在但跟丢的实例会使整帧进入review。")
    if args.can_tracker == "sam2":
        print("无CAD安全默认：易拉罐当前使用SAM2。准备好尺寸正确的mesh后，再把tracker改为foundationpose。")
    else:
        print("已选择FoundationPose易拉罐分支：运行前必须准备米制mesh和每段关键帧mask。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
