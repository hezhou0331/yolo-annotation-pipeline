#!/usr/bin/env python3
"""Print copy-ready manual-mask commands from segments.json."""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="输出每段缺失关键mask的绘制清单")
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--python", default="/home/hezhou/miniforge3/envs/foundationpose/bin/python")
    args = parser.parse_args()
    path = args.segments.expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    scene = Path(report["scene"])
    tool = Path(__file__).resolve().parent / "draw_first_mask.py"
    count = 0
    for segment in report["segments"]:
        stem = segment["start_id"]
        for instance in segment.get("missing_key_masks", []):
            output = segment.get("required_key_mask_paths", {}).get(instance)
            print(f"# 分段 {segment['segment_id']}，实例 {instance}，首帧 {stem}")
            if output:
                cmd = [args.python, str(tool), "--image", str(scene / "rgb" / f"{stem}.png"), "--output", output]
                print(" ".join(shlex.quote(x) for x in cmd))
            else:
                print("# manifest未配置该实例的key_mask_dir/registration_mask_dir，请先补路径。")
            print()
            count += 1
    if count == 0:
        print("所有分段关键mask已齐备，可以运行自动标注。")
    else:
        print(f"共缺{count}个关键mask。逐条执行上面的命令并保存后，重新运行一键流水线。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
