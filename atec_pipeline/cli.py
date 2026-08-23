#!/usr/bin/env python3
"""One command-line entry point for the ATEC RGB-D annotation workflow."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORBBEC_PY = Path.home() / "miniforge3/envs/orbbec/bin/python"
DEFAULT_FP_PY = Path.home() / "miniforge3/envs/foundationpose/bin/python"
DEFAULT_YOLO_PY = Path.home() / "miniforge3/envs/yolo11/bin/python"


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def interpreters() -> dict[str, Path]:
    return {
        "orbbec": env_path("ATEC_ORBBEC_PY", DEFAULT_ORBBEC_PY),
        "foundationpose": env_path("ATEC_FP_PY", DEFAULT_FP_PY),
        "yolo11": env_path("ATEC_YOLO_PY", DEFAULT_YOLO_PY),
    }


def run(cmd: list[str | Path], *, check: bool = True) -> int:
    printable = [str(x) for x in cmd]
    print("运行：", " ".join(printable), flush=True)
    result = subprocess.run(printable, check=check)
    return int(result.returncode)


def add_bool_flag(cmd: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def doctor(args: argparse.Namespace) -> int:
    print(f"ATEC工作区: {ROOT}")
    ok = True
    for name, path in interpreters().items():
        exists = path.is_file() and os.access(path, os.X_OK)
        print(f"[{'OK' if exists else '缺失'}] {name}: {path}")
        ok &= exists
    required = {
        "Gemini 336L采集脚本": ROOT / "tools/capture_orbbec_rgbd.py",
        "SAM2权重": ROOT / "models/sam2.1_t.pt",
        "YOLO11-seg权重": ROOT / "models/yolo11n-seg.pt",
        "FoundationPose": ROOT / "third_party/FoundationPose/run_demo.py",
    }
    missing_assets = []
    for name, path in required.items():
        exists = path.exists()
        print(f"[{'OK' if exists else '缺失'}] {name}: {path}")
        ok &= exists
        if not exists:
            missing_assets.append(name)
    if missing_assets:
        print("[提示] 第三方源码、模型和真实数据默认不进入Git。")
        print("[提示] 第三方源码：scripts/bootstrap_third_party.sh --install")
        print("[提示] 模型位置与校验：models/README.md、scripts/verify_local_assets.sh")
    usage = shutil.disk_usage(ROOT)
    print(f"[磁盘] 可用 {usage.free / 1024**3:.1f} GiB / 总计 {usage.total / 1024**3:.1f} GiB")
    if usage.free < 15 * 1024**3:
        print("[警告] 可用空间低于15 GiB，不建议长时间保存无压缩PNG序列。")
    print(f"[GitHub CLI] {shutil.which('gh') or '未安装'}")
    print(f"[Git] {'已初始化' if (ROOT / '.git').exists() else '尚未初始化'}")
    if args.deep:
        run([ROOT / "scripts/check_all_environments.sh"])
    return 0 if ok else 1


def init_project(args: argparse.Namespace) -> int:
    py = interpreters()["foundationpose"]
    cmd = [py, ROOT / "tools/prepare_atec_project.py", "--project-root", args.project_root,
           "--scene-name", args.scene_name, "--split", args.split,
           "--can-tracker", args.can_tracker, "--instances-per-class", str(args.instances_per_class)]
    add_bool_flag(cmd, args.include_bins, "--include-bins")
    add_bool_flag(cmd, args.force_manifest, "--force-manifest")
    return run(cmd)


def capture(args: argparse.Namespace) -> int:
    cmd = [ROOT / "scripts/capture_orbbec.sh", args.output,
           "--width", str(args.width), "--height", str(args.height), "--fps", str(args.fps),
           "--warmup", str(args.warmup), "--interval", str(args.interval),
           "--max-frames", str(args.max_frames), "--min-depth", str(args.min_depth),
           "--max-depth", str(args.max_depth)]
    add_bool_flag(cmd, args.auto, "--auto")
    add_bool_flag(cmd, args.no_preview, "--no-preview")
    return run(cmd)


def segment(args: argparse.Namespace) -> int:
    py = interpreters()["foundationpose"]
    cmd = [py, ROOT / "tools/segment_rgbd_sequence.py", "--manifest", args.manifest,
           "--max-segment-frames", str(args.max_segment_frames),
           "--scene-cut-threshold", str(args.scene_cut_threshold),
           "--timestamp-gap-ms", str(args.timestamp_gap_ms),
           "--min-depth-valid-ratio", str(args.min_depth_valid_ratio)]
    if args.output:
        cmd.extend(["--output", args.output])
    add_bool_flag(cmd, args.require_ready, "--require-ready")
    return run(cmd)


def checklist(args: argparse.Namespace) -> int:
    return run([interpreters()["foundationpose"], ROOT / "tools/print_mask_checklist.py",
                "--segments", args.segments, "--python", str(interpreters()["yolo11"])])


def mask(args: argparse.Namespace) -> int:
    cmd = [interpreters()["yolo11"], ROOT / "tools/draw_first_mask.py",
           "--image", args.image, "--output", args.output, "--brush-size", str(args.brush_size)]
    if args.mask_input:
        cmd.extend(["--mask-input", args.mask_input])
    add_bool_flag(cmd, args.no_resume, "--no-resume")
    return run(cmd)


def annotate(args: argparse.Namespace) -> int:
    cmd = [interpreters()["foundationpose"], ROOT / "tools/annotate_multinstance_project.py",
           "--manifest", args.manifest]
    add_bool_flag(cmd, args.skip_tracking, "--skip-tracking")
    add_bool_flag(cmd, args.dry_run, "--dry-run")
    add_bool_flag(cmd, args.include_review, "--include-review")
    add_bool_flag(cmd, args.keep_conflicts, "--keep-conflicts")
    return run(cmd)


def full_run(args: argparse.Namespace) -> int:
    cmd = [ROOT / "scripts/run_auto_annotation_pipeline.sh", args.manifest]
    if args.allow_missing_key_masks:
        cmd.append("--allow-missing-key-masks")
    if args.dry_run:
        cmd.append("--dry-run")
    return run(cmd)


def validate(args: argparse.Namespace) -> int:
    cmd = [interpreters()["yolo11"], ROOT / "tools/train_yolo11_seg.py", "--data", args.data,
           "--model", args.model, "--validate-only", "--require-project-reports"]
    return run(cmd)


def train(args: argparse.Namespace) -> int:
    cmd = [interpreters()["yolo11"], ROOT / "tools/train_yolo11_seg.py", "--data", args.data,
           "--model", args.model, "--epochs", str(args.epochs), "--imgsz", str(args.imgsz),
           "--batch", str(args.batch), "--device", args.device, "--workers", str(args.workers),
           "--project", args.project, "--name", args.name, "--patience", str(args.patience),
           "--cache", args.cache, "--require-project-reports"]
    return run(cmd)


def smoke_test(_: argparse.Namespace) -> int:
    jobs = [
        [interpreters()["yolo11"], ROOT / "tests/test_sam2_recovery.py"],
        [interpreters()["yolo11"], ROOT / "tests/test_annotation_pipeline.py"],
        [interpreters()["foundationpose"], ROOT / "tests/test_dataset_safety.py"],
    ]
    for cmd in jobs:
        run(cmd)
    print("全部Pipeline烟雾测试通过。")
    return 0


def status(args: argparse.Namespace) -> int:
    project = Path(args.project_root).expanduser().resolve()
    print(f"项目: {project}")
    usage = shutil.disk_usage(project if project.exists() else ROOT)
    print(f"磁盘可用: {usage.free / 1024**3:.1f} GiB")
    patterns = {
        "RGB帧": "data/scenes/*/rgb/*.png",
        "深度帧": "data/scenes/*/depth/*.png",
        "关键Mask": "data/key_masks/**/*.png",
        "YOLO标签": "datasets/**/labels/**/*.txt",
        "审核报告": "datasets/**/project_reports/*.json",
    }
    for label, pattern in patterns.items():
        print(f"{label}: {sum(1 for _ in project.glob(pattern))}")
    if (ROOT / ".git").exists():
        subprocess.run(["git", "status", "--short", "--branch"], cwd=ROOT, check=False)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atec-pipeline", description="Gemini 336L → SAM2/FoundationPose → YOLO11 Seg流水线")
    sub = p.add_subparsers(dest="command", required=True)

    x = sub.add_parser("doctor", help="检查三个Conda环境、权重、磁盘和Git")
    x.add_argument("--deep", action="store_true", help="额外运行完整环境检查")
    x.set_defaults(func=doctor)

    x = sub.add_parser("init", help="创建一个场景项目和manifest")
    x.add_argument("project_root", type=Path); x.add_argument("scene_name")
    x.add_argument("--split", choices=["train", "val", "test"], default="train")
    x.add_argument("--can-tracker", choices=["sam2", "foundationpose"], default="sam2")
    x.add_argument("--instances-per-class", type=int, default=1)
    x.add_argument("--include-bins", action="store_true"); x.add_argument("--force-manifest", action="store_true")
    x.set_defaults(func=init_project)

    x = sub.add_parser("capture", help="用Gemini 336L采集对齐RGB-D")
    x.add_argument("output", type=Path); x.add_argument("--width", type=int, default=640)
    x.add_argument("--height", type=int, default=480); x.add_argument("--fps", type=int, default=30)
    x.add_argument("--warmup", type=int, default=20); x.add_argument("--auto", action="store_true")
    x.add_argument("--interval", type=float, default=0.5); x.add_argument("--max-frames", type=int, default=0)
    x.add_argument("--no-preview", action="store_true"); x.add_argument("--min-depth", type=int, default=200)
    x.add_argument("--max-depth", type=int, default=3000); x.set_defaults(func=capture)

    x = sub.add_parser("segment", help="自动切分RGB-D序列并检查关键Mask")
    x.add_argument("manifest", type=Path); x.add_argument("--output", type=Path)
    x.add_argument("--max-segment-frames", type=int, default=180)
    x.add_argument("--scene-cut-threshold", type=float, default=0.18)
    x.add_argument("--timestamp-gap-ms", type=float, default=0.0)
    x.add_argument("--min-depth-valid-ratio", type=float, default=0.70)
    x.add_argument("--require-ready", action="store_true"); x.set_defaults(func=segment)

    x = sub.add_parser("checklist", help="打印所有缺失关键Mask的绘制命令")
    x.add_argument("segments", type=Path); x.set_defaults(func=checklist)

    x = sub.add_parser("mask", help="绘制或修订一个关键帧Mask")
    x.add_argument("image", type=Path); x.add_argument("output", type=Path)
    x.add_argument("--mask-input", type=Path); x.add_argument("--brush-size", type=int, default=25)
    x.add_argument("--no-resume", action="store_true"); x.set_defaults(func=mask)

    x = sub.add_parser("annotate", help="SAM2/FoundationPose传播、质量过滤和YOLO聚合")
    x.add_argument("manifest", type=Path); x.add_argument("--skip-tracking", action="store_true")
    x.add_argument("--dry-run", action="store_true"); x.add_argument("--include-review", action="store_true")
    x.add_argument("--keep-conflicts", action="store_true"); x.set_defaults(func=annotate)

    x = sub.add_parser("run", help="执行分段检查和完整自动标注")
    x.add_argument("manifest", type=Path); x.add_argument("--allow-missing-key-masks", action="store_true")
    x.add_argument("--dry-run", action="store_true", help="只检查分段和命令，不运行跟踪器或覆盖标签")
    x.set_defaults(func=full_run)

    x = sub.add_parser("validate", help="只做YOLO数据安全检查，不训练")
    x.add_argument("data", type=Path); x.add_argument("--model", type=Path, default=ROOT / "models/yolo11n-seg.pt")
    x.set_defaults(func=validate)

    x = sub.add_parser("train", help="训练YOLO11 Segmentation")
    x.add_argument("data", type=Path); x.add_argument("--model", type=Path, default=ROOT / "models/yolo11n-seg.pt")
    x.add_argument("--epochs", type=int, default=100); x.add_argument("--imgsz", type=int, default=640)
    x.add_argument("--batch", type=int, default=4); x.add_argument("--device", default="0")
    x.add_argument("--workers", type=int, default=4); x.add_argument("--project", type=Path, default=ROOT / "runs/segment")
    x.add_argument("--name", default="atec_yolo11n_seg"); x.add_argument("--patience", type=int, default=30)
    x.add_argument("--cache", choices=["false", "ram", "disk"], default="false"); x.set_defaults(func=train)

    x = sub.add_parser("smoke-test", help="运行CPU/轻量回归测试")
    x.set_defaults(func=smoke_test)
    x = sub.add_parser("status", help="汇总磁盘、帧、Mask、标签和Git状态")
    x.add_argument("project_root", nargs="?", type=Path, default=ROOT / "projects/atec_real"); x.set_defaults(func=status)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))
