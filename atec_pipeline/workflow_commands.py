"""Pure command planning for the thin ATEC desktop App.

This module intentionally has no Qt, camera, SAM2, YOLO, or filesystem-write
side effects.  The GUI asks it what the next action is and which existing CLI
command to launch; the algorithms continue to live behind ``atec-pipeline``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ContinueAction = Literal["segment", "mask", "run"]


@dataclass(frozen=True)
class ContinuePlan:
    """One operator-visible step selected from the saved-scene state."""

    action: ContinueAction
    task_kind: str | None
    feedback: str


_TASK_LABELS = {
    "capture": "RGB-D采集",
    "init": "创建Manifest",
    "segment": "自动分段",
    "mask": "关键帧Mask标记",
    "run": "SAM2传播与YOLO导出",
    "validate": "数据集验证",
    "prepare_validate": "训练数据验证",
    "prepare_split": "按完整场次划分验证集",
    "review": "人工Review",
    "review_rerun": "局部SAM2重新传播",
    "review_export": "Review结果重新导出",
    "train": "YOLO训练",
}


def task_display_name(kind: str) -> str:
    """Return a stable Chinese task name for GUI status and logs."""

    return _TASK_LABELS.get(kind, kind or "任务")


def plan_continue_processing(
    manifest_path: Path,
    segments_report: Path,
    *,
    masks_complete: bool,
) -> ContinuePlan:
    """Choose the next CLI/UI action without executing it."""

    manifest = Path(manifest_path).expanduser().resolve()
    segments = Path(segments_report).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest不存在：{manifest}")
    if not segments.is_file():
        return ContinuePlan(
            action="segment",
            task_kind="segment",
            feedback="正在启动自动分段；完成后会进入关键帧标记。",
        )
    if not masks_complete:
        return ContinuePlan(
            action="mask",
            task_kind=None,
            feedback="仍有关键帧 Mask 未完成；现在打开下一个缺失关键帧。",
        )
    return ContinuePlan(
        action="run",
        task_kind="run",
        feedback="正在启动 SAM2 传播与 YOLO 导出；详细输出已展开到运行日志。",
    )


def build_pipeline_command(
    repository_root: Path,
    subcommand: str,
    manifest_path: Path,
    *extra_args: str,
) -> tuple[str, list[str]]:
    """Build one existing ``scripts/atec-pipeline`` invocation."""

    root = Path(repository_root).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    return str(root / "scripts" / "atec-pipeline"), [subcommand, str(manifest), *map(str, extra_args)]


def build_split_command(
    repository_root: Path,
    dataset_path: Path,
    *,
    val_scenes: tuple[str, ...] = (),
    auto: bool = False,
    target_val_ratio: float = 0.20,
    apply: bool = False,
) -> tuple[str, list[str]]:
    """Build the existing CLI dataset-split command without mutating data."""

    root = Path(repository_root).expanduser().resolve()
    dataset = Path(dataset_path).expanduser().resolve()
    args = ["split", str(dataset)]
    if auto:
        args.extend(["--auto", "--target-val-ratio", str(target_val_ratio)])
    scenes = tuple(sorted({str(name).strip() for name in val_scenes if str(name).strip()}))
    if scenes:
        args.extend(["--val-scenes", *scenes])
    if not auto and not scenes:
        raise ValueError("切分命令必须指定--auto或至少一个val场次")
    if apply:
        args.append("--apply")
    return str(root / "scripts" / "atec-pipeline"), args
