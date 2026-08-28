#!/usr/bin/env python3
"""Migrate legacy flat scene directories into data/scenes/<class>/<scene>."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable

import yaml

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from atec_pipeline.path_compat import infer_project_root, portable_path, resolve_compatible_path

CLASS_PREFIXES = (
    ("watermelon_rind", "watermelon_rind"),
    ("red_paper_bag", "red_paper_bag"),
    ("meal_box", "meal_box"),
    ("yellow_can", "can"),
    ("can", "can"),
    ("blue_bin", "blue_bin"),
    ("green_bin", "green_bin"),
    ("red_bin", "red_bin"),
)
CATEGORY_NAMES = frozenset(class_name for _prefix, class_name in CLASS_PREFIXES)


@dataclass(frozen=True)
class MigrationItem:
    source: str
    target: str
    class_name: str
    manifests: tuple[str, ...]


def infer_class(scene_name: str) -> str | None:
    for prefix, class_name in CLASS_PREFIXES:
        if scene_name == prefix or scene_name.startswith(prefix + "_"):
            return class_name
    return None


def _resolve_scene(manifest_path: Path, value: str) -> Path:
    return resolve_compatible_path(
        value,
        base=manifest_path.parent,
        repository_root=WORKSPACE,
        project_root=infer_project_root(manifest_path, repository_root=WORKSPACE),
    )


def find_manifests(project_root: Path, source: Path) -> tuple[Path, ...]:
    result = []
    manifest_dir = project_root / "manifests"
    for path in sorted(manifest_dir.glob("*.y*ml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            scene = data.get("project", {}).get("scene")
            if scene and _resolve_scene(path, str(scene)) == source.resolve():
                result.append(path)
        except (OSError, yaml.YAMLError, AttributeError):
            continue
    return tuple(result)


def scan(project_root: Path) -> tuple[MigrationItem, ...]:
    scenes_root = (project_root / "data" / "scenes").resolve()
    if not scenes_root.exists():
        return ()
    items = []
    for source in sorted(p for p in scenes_root.iterdir() if p.is_dir()):
        # Already migrated category directories are ignored; only direct scene dirs are candidates.
        if source.name in CATEGORY_NAMES:
            continue
        class_name = infer_class(source.name)
        if class_name is None:
            continue
        target = scenes_root / class_name / source.name
        manifests = tuple(str(p.resolve()) for p in find_manifests(project_root, source))
        items.append(MigrationItem(str(source), str(target), class_name, manifests))
    return tuple(items)


def validate(items: Iterable[MigrationItem]) -> list[str]:
    errors: list[str] = []
    seen_targets: set[Path] = set()
    for item in items:
        source, target = Path(item.source), Path(item.target)
        if not source.exists() or not source.is_dir():
            errors.append(f"源目录不存在或不是目录: {source}")
        if target in seen_targets:
            errors.append(f"目标路径重复: {target}")
        seen_targets.add(target)
        if target.exists():
            errors.append(f"目标路径已存在，不覆盖: {target}")
    return errors


def _rewrite_manifest(path: Path, source: Path, target: Path) -> bool:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    project = data.get("project") or {}
    if "scene" not in project or _resolve_scene(path, str(project["scene"])) != source.resolve():
        return False
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    scene_pattern = re.compile(r"^(\s*scene:\s*)(.*?)(\s*)$")
    for index, line in enumerate(lines):
        match = scene_pattern.match(line.rstrip("\n"))
        if match:
            newline = "\n" if line.endswith("\n") else ""
            value = portable_path(
                target,
                relative_to=path.parent,
                repository_root=WORKSPACE,
                project_root=infer_project_root(path, repository_root=WORKSPACE),
            )
            if match.group(2).strip().startswith(("'", '"')):
                quote = match.group(2).strip()[0]
                value = quote + value + quote
            lines[index] = f"{match.group(1)}{value}{newline}"
            path.write_text("".join(lines), encoding="utf-8")
            return True
    # Valid YAML without a simple scene line: preserve filename and semantics via safe_dump.
    project["scene"] = portable_path(
        target,
        relative_to=path.parent,
        repository_root=WORKSPACE,
        project_root=infer_project_root(path, repository_root=WORKSPACE),
    )
    data["project"] = project
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return True


def apply(items: Iterable[MigrationItem]) -> None:
    items = tuple(items)
    errors = validate(items)
    if errors:
        raise RuntimeError("迁移已停止，未移动任何目录:\n" + "\n".join(errors))
    for item in items:
        source, target = Path(item.source), Path(item.target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        for manifest in item.manifests:
            _rewrite_manifest(Path(manifest), source, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="将旧扁平场景迁移到按类别分类目录")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="确认无冲突后实际移动")
    parser.add_argument("--report", type=Path, help="输出JSON报告路径")
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    items = scan(root)
    errors = validate(items)
    report = {
        "project_root": str(root),
        "apply": bool(args.apply),
        "items": [asdict(item) for item in items],
        "errors": errors,
    }
    if args.apply:
        if errors:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print("迁移已停止：存在冲突；未移动任何目录。")
            return 2
        apply(items)
        report["applied"] = True
    else:
        report["applied"] = False
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.report:
        args.report.expanduser().resolve().write_text(text, encoding="utf-8")
    if not items:
        print("没有发现可迁移的旧扁平场景。")
    elif not args.apply:
        print("以上为dry-run报告；确认无冲突后加 --apply 才会移动。")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
