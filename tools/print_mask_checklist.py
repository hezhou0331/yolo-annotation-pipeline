#!/usr/bin/env python3
"""Print copy-ready manual-mask commands from segments.json."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from atec_pipeline.path_compat import infer_project_root, resolve_compatible_path
from atec_pipeline.runtime import interpreter_path


def main():
    parser = argparse.ArgumentParser(description="输出每段缺失关键mask的绘制清单")
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--python", default=str(interpreter_path("foundationpose")))
    args = parser.parse_args()
    path = args.segments.expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    project_root = infer_project_root(path, repository_root=WORKSPACE)
    scene = resolve_compatible_path(
        report["scene"], base=path.parent,
        repository_root=WORKSPACE, project_root=project_root,
    )
    tool = Path(__file__).resolve().parent / "draw_first_mask.py"
    count = 0
    for segment in report["segments"]:
        stem = segment["start_id"]
        for instance in segment.get("missing_key_masks", []):
            output = segment.get("required_key_mask_paths", {}).get(instance)
            print(f"# 分段 {segment['segment_id']}，实例 {instance}，首帧 {stem}")
            if output:
                output_path = resolve_compatible_path(
                    output, base=path.parent,
                    repository_root=WORKSPACE, project_root=project_root,
                )
                cmd = [args.python, str(tool), "--image", str(scene / "rgb" / f"{stem}.png"), "--output", str(output_path)]
                print(" ".join(shlex.quote(x) for x in cmd))
            else:
                print("# manifest未配置该实例的key_mask_dir/registration_mask_dir，请先补路径。")
            print()
            count += 1
    if count == 0:
        print("所有分段关键mask已齐备，可以运行自动标注。")
    else:
        print(f"共缺{count}个关键mask。逐条执行上面的命令并保存后，再继续处理当前场次。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
