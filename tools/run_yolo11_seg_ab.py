#!/usr/bin/env python3
"""Run a controlled YOLO11s-seg baseline/xcx-transfer A/B experiment."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="同设置运行YOLO11s-seg A/B实验")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--official-model", type=Path, default=ROOT / "models/yolo11s-seg.pt")
    p.add_argument("--xcx-detect-model", type=Path, default=ROOT / "xcx/checkpoint/unified_yolo11s_640_15class_final.pt")
    p.add_argument("--project", type=Path, default=ROOT / "runs/segment_ab")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    p.add_argument("--reviewed-negatives", type=Path)
    p.add_argument("--execute", action="store_true", help="默认只写实验计划；确认后加此参数顺序执行A和B")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    common = [
        sys.executable, str(ROOT / "tools/train_yolo11_seg.py"), "--data", str(args.data),
        "--epochs", str(args.epochs), "--imgsz", str(args.imgsz), "--batch", str(args.batch),
        "--device", args.device, "--workers", str(args.workers), "--project", str(args.project),
        "--patience", str(args.patience), "--seed", str(args.seed), "--cache", args.cache,
        "--require-project-reports", "--require-source-ids",
    ]
    if args.reviewed_negatives:
        common += ["--reviewed-negatives", str(args.reviewed_negatives)]
    commands = {
        "A_official_baseline": common + [
            "--init-mode", "baseline", "--model", str(args.official_model), "--name", "A_official_yolo11s_seg",
        ],
        "B_xcx_transfer": common + [
            "--init-mode", "xcx-transfer", "--model", str(args.official_model),
            "--xcx-detect-model", str(args.xcx_detect_model), "--name", "B_xcx_transfer_yolo11s_seg",
        ],
    }
    project = args.project.expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)
    plan_path = project / "ab_experiment_plan.json"
    plan_path.write_text(json.dumps({
        "fixed_settings": {
            "data": str(args.data), "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
            "device": args.device, "workers": args.workers, "patience": args.patience, "seed": args.seed,
            "cache": args.cache,
        },
        "commands": commands,
        "selection_rule": "B仅在真实val的Mask mAP50-95/每类召回稳定更好且干扰误检不增加时采用",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"A/B实验计划：{plan_path}")
    for name, command in commands.items():
        print(name + ":", " ".join(command))
    if not args.execute:
        print("当前仅生成计划；确认数据验证通过后加--execute执行。")
        return 0
    for name, command in commands.items():
        print(f"开始{name}", flush=True)
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
