#!/usr/bin/env python3
"""Generate a metric primitive OBJ for FoundationPose when no CAD model exists.

This is intentionally limited to geometry that can be measured reliably.  A
primitive is useful for cans and rigid closed boxes, but it is not a substitute
for a scan of deformable bags, food contents, or irregular watermelon rinds.
The generated OBJ is always written in metres so manifests should use
``mesh_unit: m``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("尺寸必须是有限正数")
    return number


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="用实测尺寸生成FoundationPose可用的米制基础Mesh")
    p.add_argument("--shape", choices=("cylinder", "box", "truncated-cone"), required=True)
    p.add_argument("--output", type=Path, required=True, help="输出OBJ路径，例如.../textured_mesh.obj")
    p.add_argument("--unit", choices=("m", "cm", "mm"), default="mm", help="输入尺寸单位；输出始终为米")
    p.add_argument("--height", type=positive, required=True)
    p.add_argument("--diameter", type=positive, help="圆柱直径")
    p.add_argument("--bottom-diameter", type=positive, help="截锥底部直径")
    p.add_argument("--top-diameter", type=positive, help="截锥顶部直径")
    p.add_argument("--length", type=positive, help="盒体X方向长度")
    p.add_argument("--width", type=positive, help="盒体Y方向宽度")
    p.add_argument("--segments", type=int, default=64, help="圆周分段数，至少12")
    p.add_argument("--name", default="atec_primitive")
    p.add_argument("--force", action="store_true", help="允许覆盖现有OBJ和元数据")
    return p.parse_args()


def scale_for(unit: str) -> float:
    return {"m": 1.0, "cm": 0.01, "mm": 0.001}[unit]


def box_mesh(length: float, width: float, height: float):
    x, y, z = length / 2, width / 2, height / 2
    vertices = [
        (-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
        (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z),
    ]
    faces = [
        (1, 3, 2), (1, 4, 3),       # bottom
        (5, 6, 7), (5, 7, 8),       # top
        (1, 2, 6), (1, 6, 5),       # -Y
        (2, 3, 7), (2, 7, 6),       # +X
        (3, 4, 8), (3, 8, 7),       # +Y
        (4, 1, 5), (4, 5, 8),       # -X
    ]
    return vertices, faces


def frustum_mesh(bottom_radius: float, top_radius: float, height: float, segments: int):
    z0, z1 = -height / 2, height / 2
    vertices: list[tuple[float, float, float]] = []
    for radius, z in ((bottom_radius, z0), (top_radius, z1)):
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
    bottom_center = len(vertices) + 1
    vertices.append((0.0, 0.0, z0))
    top_center = len(vertices) + 1
    vertices.append((0.0, 0.0, z1))

    faces: list[tuple[int, int, int]] = []
    for i in range(segments):
        j = (i + 1) % segments
        b0, b1 = i + 1, j + 1
        t0, t1 = segments + i + 1, segments + j + 1
        faces.extend(((b0, b1, t1), (b0, t1, t0)))
        faces.append((bottom_center, b1, b0))
        faces.append((top_center, t0, t1))
    return vertices, faces


def write_obj(path: Path, name: str, vertices, faces) -> None:
    lines = [
        "# ATEC metric primitive mesh",
        "# Units: metres",
        f"o {name}",
    ]
    lines.extend(f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in vertices)
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extents(vertices) -> list[float]:
    return [
        max(v[axis] for v in vertices) - min(v[axis] for v in vertices)
        for axis in range(3)
    ]


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    metadata = output.with_suffix(output.suffix + ".json")
    if not args.force and (output.exists() or metadata.exists()):
        raise SystemExit(f"输出已存在，不覆盖：{output}；确认后加--force")
    if args.segments < 12:
        raise SystemExit("--segments至少为12")

    s = scale_for(args.unit)
    height = args.height * s
    dimensions: dict[str, float | int | str]
    if args.shape == "box":
        if args.length is None or args.width is None:
            raise SystemExit("box必须提供--length和--width")
        length, width = args.length * s, args.width * s
        vertices, faces = box_mesh(length, width, height)
        dimensions = {"length_m": length, "width_m": width, "height_m": height}
    elif args.shape == "cylinder":
        if args.diameter is None:
            raise SystemExit("cylinder必须提供--diameter")
        diameter = args.diameter * s
        vertices, faces = frustum_mesh(diameter / 2, diameter / 2, height, args.segments)
        dimensions = {"diameter_m": diameter, "height_m": height, "segments": args.segments}
    else:
        if args.bottom_diameter is None or args.top_diameter is None:
            raise SystemExit("truncated-cone必须提供--bottom-diameter和--top-diameter")
        bottom, top = args.bottom_diameter * s, args.top_diameter * s
        vertices, faces = frustum_mesh(bottom / 2, top / 2, height, args.segments)
        dimensions = {
            "bottom_diameter_m": bottom,
            "top_diameter_m": top,
            "height_m": height,
            "segments": args.segments,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    write_obj(output, args.name, vertices, faces)
    info = {
        "format_version": 1,
        "shape": args.shape,
        "input_unit": args.unit,
        "output_unit": "m",
        "dimensions": dimensions,
        "vertex_count": len(vertices),
        "triangle_count": len(faces),
        "extents_m": extents(vertices),
        "foundationpose_manifest": {"mesh": str(output), "mesh_unit": "m"},
        "limitations": [
            "参数化mesh只适合尺寸可测且近似刚性的物体。",
            "易拉罐拉环、凹陷和印刷纹理未建模，首次使用必须与SAM2结果对比。",
            "纸袋、可变形餐盒内容物和不规则西瓜皮不应仅依赖此mesh。",
        ],
    }
    metadata.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成米制Mesh：{output}")
    print(f"尺寸范围XYZ（米）：{info['extents_m']}")
    print(f"元数据：{metadata}")
    print("manifest中使用：mesh_unit: m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
