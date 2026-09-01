#!/usr/bin/env python3
"""Prepare a portable ATEC RGB-D and YOLO11-seg release tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXCLUDED_DATA_DIRS = {".staging"}
EXCLUDED_DATASET_DIRS = {"_staging", "logs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="复制完整 ATEC 标注数据，排除暂存/缓存并改写本机绝对路径。"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="源代码仓库根目录（默认自动定位）",
    )
    parser.add_argument("--output", type=Path, required=True, help="空的快照输出目录")
    parser.add_argument("--tag", default="", help="写入快照清单的 Release tag")
    return parser.parse_args()


def _ensure_empty_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"输出目录必须为空：{output}")
    output.mkdir(parents=True, exist_ok=True)


def _copy_tree(
    source: Path,
    destination: Path,
    *,
    excluded_top_dirs: set[str],
    inode_targets: dict[tuple[int, int], Path],
) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    for current, dirnames, filenames in os.walk(source):
        current_path = Path(current)
        relative = current_path.relative_to(source)
        if relative == Path("."):
            dirnames[:] = [name for name in dirnames if name not in excluded_top_dirs]
        target_dir = destination / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            if filename.endswith(".cache"):
                continue
            src = current_path / filename
            dst = target_dir / filename
            if src.is_symlink():
                dst.symlink_to(os.readlink(src))
                continue
            stat = src.stat()
            inode_key = (stat.st_dev, stat.st_ino)
            prior = inode_targets.get(inode_key)
            if prior is not None:
                os.link(prior, dst)
            else:
                shutil.copy2(src, dst)
                inode_targets[inode_key] = dst


def _mapped_path(value: str, *, repo_root: Path, output: Path) -> Path | None:
    if not os.path.isabs(value):
        return None
    normalized = value.replace("\\", "/")
    project_marker = "/projects/atec_real/"
    if project_marker in normalized:
        suffix = normalized.rsplit(project_marker, 1)[1]
        return output / "projects" / "atec_real" / suffix
    source_prefix = repo_root.resolve().as_posix().rstrip("/") + "/"
    if normalized.startswith(source_prefix):
        return output / normalized[len(source_prefix) :]
    repo_marker = f"/{repo_root.name}/"
    if repo_marker in normalized:
        suffix = normalized.rsplit(repo_marker, 1)[1]
        return output / suffix
    return None


def _rewrite_value(value: Any, *, json_path: Path, repo_root: Path, output: Path) -> tuple[Any, int]:
    if isinstance(value, dict):
        rewritten: dict[str, Any] = {}
        changes = 0
        for key, item in value.items():
            rewritten[key], count = _rewrite_value(
                item, json_path=json_path, repo_root=repo_root, output=output
            )
            changes += count
        return rewritten, changes
    if isinstance(value, list):
        rewritten_list: list[Any] = []
        changes = 0
        for item in value:
            new_item, count = _rewrite_value(
                item, json_path=json_path, repo_root=repo_root, output=output
            )
            rewritten_list.append(new_item)
            changes += count
        return rewritten_list, changes
    if isinstance(value, str):
        mapped = _mapped_path(value, repo_root=repo_root, output=output)
        if mapped is not None:
            return os.path.relpath(mapped, json_path.parent), 1
    return value, 0


def _rewrite_json_paths(root: Path, *, repo_root: Path, output: Path) -> int:
    total = 0
    for json_path in root.rglob("*.json"):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        rewritten, changes = _rewrite_value(
            payload, json_path=json_path, repo_root=repo_root, output=output
        )
        if changes:
            json_path.write_text(
                json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            total += changes
    return total


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_split_counts(dataset: Path, split: str) -> dict[str, int]:
    image_stems = {path.stem for path in (dataset / "images" / split).glob("*") if path.is_file()}
    label_stems = {path.stem for path in (dataset / "labels" / split).glob("*.txt")}
    if image_stems != label_stems:
        missing_labels = sorted(image_stems - label_stems)[:5]
        missing_images = sorted(label_stems - image_stems)[:5]
        raise RuntimeError(
            f"{split} 图片/标签不匹配：missing_labels={missing_labels}, "
            f"missing_images={missing_images}"
        )
    return {"images": len(image_stems), "labels": len(label_stems)}


def _write_snapshot_manifest(output: Path, *, tag: str, rewritten_paths: int) -> Path:
    project = output / "projects" / "atec_real"
    data = project / "data"
    dataset = project / "datasets" / "atec_yolo11_seg"
    scene_dirs = [path for class_dir in (data / "scenes").iterdir() if class_dir.is_dir() for path in class_dir.iterdir() if path.is_dir()]
    train = _formal_split_counts(dataset, "train")
    val = _formal_split_counts(dataset, "val")
    manifest = {
        "tag": tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contents": {
            "source_scenes": len(scene_dirs),
            "key_mask_pngs": sum(1 for _ in (data / "key_masks").rglob("*.png")),
            "formal_project_reports": sum(1 for _ in (dataset / "project_reports").glob("*_report.json")),
            "formal_train": train,
            "formal_val": val,
            "formal_total_pairs": train["images"] + val["images"],
        },
        "excluded": [
            "projects/atec_real/data/.staging",
            "projects/atec_real/datasets/atec_yolo11_seg/_staging",
            "projects/atec_real/datasets/atec_yolo11_seg/logs",
            "*.cache",
        ],
        "rewritten_absolute_paths": rewritten_paths,
        "dataset_yaml_sha256": _sha256(dataset / "dataset.yaml"),
    }
    target = project / "data_snapshot.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    _ensure_empty_output(output)
    source_project = repo_root / "projects" / "atec_real"
    inode_targets: dict[tuple[int, int], Path] = {}
    _copy_tree(
        source_project / "data",
        output / "projects" / "atec_real" / "data",
        excluded_top_dirs=EXCLUDED_DATA_DIRS,
        inode_targets=inode_targets,
    )
    _copy_tree(
        source_project / "datasets" / "atec_yolo11_seg",
        output / "projects" / "atec_real" / "datasets" / "atec_yolo11_seg",
        excluded_top_dirs=EXCLUDED_DATASET_DIRS,
        inode_targets=inode_targets,
    )
    rewritten = _rewrite_json_paths(output, repo_root=repo_root, output=output)
    snapshot_manifest = _write_snapshot_manifest(output, tag=args.tag, rewritten_paths=rewritten)
    print(snapshot_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
