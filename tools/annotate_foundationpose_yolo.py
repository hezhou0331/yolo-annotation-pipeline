#!/usr/bin/env python3
"""Track one CAD object with FoundationPose and export YOLO labels.

Input dataset layout:
  scene/rgb/000000.png
  scene/depth/000000.png       # uint16 millimetres by default
  scene/masks/000000.png       # only the registration frame is required
  scene/cam_K.txt

The script is intentionally optimized for an 8 GiB GPU: inference images are
scaled down when their longest side exceeds --inference-max-side. Labels and
rendered masks are still generated at the original image resolution.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_FP_DIR = WORKSPACE / "third_party/FoundationPose"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FoundationPose 跟踪并导出 YOLO 标注")
    p.add_argument("--scene", type=Path, required=True, help="含 rgb/depth/masks/cam_K.txt 的采集目录")
    p.add_argument("--mesh", type=Path, required=True, help="目标物体 CAD/mesh：OBJ/PLY/STL 等")
    p.add_argument("--output", type=Path, required=True, help="标注结果目录")
    p.add_argument("--foundationpose-dir", type=Path, default=DEFAULT_FP_DIR)
    p.add_argument("--class-id", type=int, default=0)
    p.add_argument("--class-name", default="object")
    p.add_argument("--first-mask", type=Path, default=None, help="首帧 mask；默认 scene/masks/<首帧名>.png")
    p.add_argument("--name-prefix", default=None, help="YOLO 图片名前缀；默认使用场景目录名，传空字符串可关闭")
    p.add_argument("--task", choices=("detection", "segmentation"), default="detection")
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--mesh-unit", choices=("m", "cm", "mm"), default="m")
    p.add_argument("--depth-to-metre", type=float, default=None, help="覆盖 metadata.json 中的深度换算；毫米图填 0.001")
    p.add_argument("--start", type=int, default=0, help="从第几张图开始注册")
    p.add_argument("--max-frames", type=int, default=0, help="0 表示处理到序列末尾")
    p.add_argument("--est-refine-iter", type=int, default=3)
    p.add_argument("--track-refine-iter", type=int, default=2)
    p.add_argument("--inference-max-side", type=int, default=640, help="RTX 4060 8GB 建议 640；0 表示不缩放")
    p.add_argument("--mask-mode", choices=("visible", "full"), default="visible")
    p.add_argument("--occlusion-tolerance", type=float, default=0.03, help="可见性深度容差，单位米")
    p.add_argument("--min-mask-area", type=int, default=30)
    p.add_argument("--polygon-epsilon", type=float, default=0.002, help="轮廓简化比例")
    p.add_argument("--image-mode", choices=("hardlink", "copy", "none"), default="hardlink")
    p.add_argument("--append-labels", action="store_true", help="多目标分次标注时追加而不是覆盖标签")
    p.add_argument("--no-visualization", action="store_true")
    return p.parse_args()


def load_mesh(path: Path, unit: str):
    import trimesh

    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [g for g in loaded.geometry.values()]
        if not geometries:
            raise RuntimeError(f"mesh 场景为空：{path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"不支持的 mesh 类型：{type(loaded).__name__}")
    scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[unit]
    mesh = loaded.copy()
    mesh.vertices = np.asarray(mesh.vertices) * scale
    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise RuntimeError("mesh 顶点/三角面数量过少")
    return mesh


def find_depth_scale(scene: Path, override: float | None) -> float:
    if override is not None:
        return override
    metadata_path = scene / "metadata.json"
    if metadata_path.exists():
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "depth_to_metre" in data:
            return float(data["depth_to_metre"])
    return 0.001


def load_frame(rgb_path: Path, depth_path: Path, depth_scale: float):
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise RuntimeError(f"无法读取 RGB：{rgb_path}")
    if depth_raw is None:
        raise RuntimeError(f"无法读取深度图：{depth_path}")
    if depth_raw.ndim != 2:
        raise RuntimeError(f"深度图必须为单通道：{depth_path}")
    if bgr.shape[:2] != depth_raw.shape:
        raise RuntimeError(f"RGB/深度尺寸不一致：{bgr.shape[:2]} vs {depth_raw.shape}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_m = depth_raw.astype(np.float32) * depth_scale
    depth_m[~np.isfinite(depth_m)] = 0
    return bgr, rgb, depth_m


def resize_for_inference(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray | None, K: np.ndarray, max_side: int):
    h, w = depth.shape
    if max_side <= 0 or max(h, w) <= max_side:
        return rgb, depth, mask, K.copy(), 1.0
    scale = max_side / float(max(h, w))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    rgb_small = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    mask_small = None if mask is None else cv2.resize(mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
    K_small = K.copy()
    K_small[0, :] *= out_w / float(w)
    K_small[1, :] *= out_h / float(h)
    return rgb_small, depth_small, mask_small, K_small, scale


def render_mask(mesh_tensors, pose: np.ndarray, K: np.ndarray, h: int, w: int, glctx, scene_depth: np.ndarray, mode: str, tolerance: float):
    import torch
    from Utils import nvdiffrast_render

    pose_t = torch.as_tensor(pose[None], device="cuda", dtype=torch.float32)
    _, rendered_depth, _ = nvdiffrast_render(
        K=K,
        H=h,
        W=w,
        ob_in_cams=pose_t,
        glctx=glctx,
        mesh_tensors=mesh_tensors,
        output_size=np.asarray([h, w]),
    )
    rd = rendered_depth[0].detach().float().cpu().numpy()
    full = np.isfinite(rd) & (rd > 0)
    if mode == "full":
        mask = full
    else:
        observed = np.isfinite(scene_depth) & (scene_depth > 0)
        mask = full & (~observed | (rd <= scene_depth + tolerance))
        # Remove isolated single-pixel raster/depth comparison noise.
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return mask, rd


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def detection_line(class_id: int, bbox, w: int, h: int) -> str:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def segmentation_line(class_id: int, mask: np.ndarray, epsilon_ratio: float) -> str | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(0.5, epsilon_ratio * cv2.arcLength(contour, True))
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(polygon) < 3:
        return None
    h, w = mask.shape
    coords = []
    for x, y in polygon:
        coords.extend((np.clip(x / w, 0, 1), np.clip(y / h, 0, 1)))
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in coords)


def place_image(src: Path, dst: Path, mode: str):
    if mode == "none":
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_label(path: Path, line: str | None, append: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if line is None:
        if not append:
            path.write_text("", encoding="utf-8")
        return
    if append and path.exists() and path.stat().st_size > 0:
        with path.open("a", encoding="utf-8") as f:
            f.write("\n" + line)
    else:
        path.write_text(line + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    scene = args.scene.expanduser().resolve()
    mesh_path = args.mesh.expanduser().resolve()
    output = args.output.expanduser().resolve()
    fp_dir = args.foundationpose_dir.expanduser().resolve()
    if not fp_dir.exists():
        raise SystemExit(f"FoundationPose 目录不存在：{fp_dir}")
    sys.path.insert(0, str(fp_dir))
    os.chdir(fp_dir)

    import torch
    import trimesh
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    from Utils import make_mesh_tensors, set_logging_format, set_seed

    set_logging_format()
    set_seed(0)

    rgb_files = sorted((scene / "rgb").glob("*.png"))
    if not rgb_files:
        raise SystemExit(f"没有找到 RGB 图片：{scene / 'rgb'}")
    if args.start < 0 or args.start >= len(rgb_files):
        raise SystemExit(f"--start 超出范围：共有 {len(rgb_files)} 帧")
    end = len(rgb_files) if args.max_frames <= 0 else min(len(rgb_files), args.start + args.max_frames)
    selected = rgb_files[args.start:end]

    K_path = scene / "cam_K.txt"
    if not K_path.exists():
        K_path = scene / "K.txt"
    K = np.loadtxt(K_path).reshape(3, 3).astype(np.float64)
    depth_scale = find_depth_scale(scene, args.depth_to_metre)
    mesh = load_mesh(mesh_path, args.mesh_unit)

    first_stem = selected[0].stem
    first_mask_path = args.first_mask.expanduser().resolve() if args.first_mask else scene / "masks" / f"{first_stem}.png"
    first_mask_raw = cv2.imread(str(first_mask_path), cv2.IMREAD_UNCHANGED)
    if first_mask_raw is None:
        raise SystemExit(
            f"缺少注册帧 mask：{first_mask_path}\n"
            "请先运行 tools/draw_first_mask.py 绘制第一帧目标轮廓。"
        )
    if first_mask_raw.ndim == 3:
        first_mask_raw = first_mask_raw.max(axis=2)
    first_mask = first_mask_raw > 0

    print(f"场景：{scene}")
    print(f"模型：{mesh_path}，单位={args.mesh_unit}，顶点={len(mesh.vertices)}，面={len(mesh.faces)}")
    print(f"处理帧：{args.start}..{end - 1}（共 {len(selected)} 帧）")
    print(f"深度换算：PNG 数值 × {depth_scale} = 米")
    print(f"任务：YOLO {args.task}，类别 {args.class_id}:{args.class_name}")

    output.mkdir(parents=True, exist_ok=True)
    name_prefix = f"{scene.name}_" if args.name_prefix is None else args.name_prefix
    class_dir = f"class_{args.class_id:03d}"
    poses_dir = output / "poses" / args.split / class_dir
    masks_dir = output / "rendered_masks" / args.split / class_dir
    vis_dir = output / "visualizations" / args.split / class_dir
    images_dir = output / "images" / args.split
    labels_dir = output / "labels" / args.split
    all_split_dirs = []
    for split_name in ("train", "val", "test"):
        all_split_dirs.extend((output / "images" / split_name, output / "labels" / split_name))
    for d in (poses_dir, masks_dir, *all_split_dirs):
        d.mkdir(parents=True, exist_ok=True)
    if not args.no_visualization:
        vis_dir.mkdir(parents=True, exist_ok=True)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(output / "foundationpose_debug" / args.split / class_dir),
        debug=0,
        glctx=glctx,
    )
    render_mesh_tensors = make_mesh_tensors(mesh)

    records = []
    started = time.time()
    for local_i, rgb_path in enumerate(selected):
        stem = rgb_path.stem
        output_stem = f"{name_prefix}{stem}"
        depth_path = scene / "depth" / f"{stem}.png"
        bgr, rgb, depth_m = load_frame(rgb_path, depth_path, depth_scale)
        mask_for_frame = first_mask if local_i == 0 else None
        rgb_inf, depth_inf, mask_inf, K_inf, scale = resize_for_inference(
            rgb, depth_m, mask_for_frame, K, args.inference_max_side
        )

        frame_started = time.time()
        if local_i == 0:
            if mask_inf is None or int(mask_inf.sum()) < 4:
                raise RuntimeError("第一帧 mask 有效像素过少")
            pose = estimator.register(
                K=K_inf,
                rgb=rgb_inf,
                depth=depth_inf,
                ob_mask=mask_inf.astype(bool),
                iteration=args.est_refine_iter,
            )
            mode = "register"
        else:
            pose = estimator.track_one(
                rgb=rgb_inf,
                depth=depth_inf,
                K=K_inf,
                iteration=args.track_refine_iter,
            )
            mode = "track"

        np.savetxt(poses_dir / f"{output_stem}.txt", pose.reshape(4, 4), fmt="%.10f")
        mask, rendered_depth = render_mask(
            render_mesh_tensors,
            pose,
            K,
            bgr.shape[0],
            bgr.shape[1],
            glctx,
            depth_m,
            args.mask_mode,
            args.occlusion_tolerance,
        )
        area = int(mask.sum())
        bbox = mask_bbox(mask) if area >= args.min_mask_area else None
        line = None
        if bbox is not None:
            if args.task == "detection":
                line = detection_line(args.class_id, bbox, bgr.shape[1], bgr.shape[0])
            else:
                line = segmentation_line(args.class_id, mask, args.polygon_epsilon)

        cv2.imwrite(str(masks_dir / f"{output_stem}.png"), mask.astype(np.uint8) * 255, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        label_path = labels_dir / f"{output_stem}.txt"
        write_label(label_path, line, args.append_labels)
        place_image(rgb_path, images_dir / f"{output_stem}{rgb_path.suffix.lower()}", args.image_mode)

        if not args.no_visualization:
            vis = bgr.copy()
            overlay = vis.copy()
            overlay[mask] = (0, 0, 255)
            vis = cv2.addWeighted(vis, 0.72, overlay, 0.28, 0)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
            cv2.putText(vis, f"{stem} {mode} area={area}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imwrite(str(vis_dir / f"{output_stem}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

        elapsed = time.time() - frame_started
        records.append({
            "id": stem,
            "output_id": output_stem,
            "mode": mode,
            "seconds": elapsed,
            "inference_scale": scale,
            "mask_area": area,
            "has_label": line is not None,
            "bbox_xyxy": list(bbox) if bbox is not None else None,
        })
        print(f"[{local_i + 1}/{len(selected)}] {stem}: {mode}, {elapsed:.2f}s, mask={area}, label={'yes' if line else 'no'}")
        torch.cuda.empty_cache()
        gc.collect()

    classes_path = output / "classes.json"
    classes = {}
    if classes_path.exists():
        classes = json.loads(classes_path.read_text(encoding="utf-8"))
    classes[str(args.class_id)] = args.class_name
    classes = dict(sorted(classes.items(), key=lambda item: int(item[0])))
    classes_path.write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")

    # A first training run may not have a separate validation recording yet.
    # In that case dataset.yaml remains immediately usable by falling back to
    # train images; after a --split val run it switches to images/val.
    val_has_images = any((output / "images" / "val").glob("*.png"))
    test_has_images = any((output / "images" / "test").glob("*.png"))
    val_rel = "images/val" if val_has_images else "images/train"
    test_rel = "images/test" if test_has_images else val_rel
    names_yaml = "".join(f"  {class_id}: {name}\n" for class_id, name in classes.items())
    dataset_yaml = output / "dataset.yaml"
    yaml_text = (
        f"path: {output}\n"
        "train: images/train\n"
        f"val: {val_rel}\n"
        f"test: {test_rel}\n"
        "names:\n"
        f"{names_yaml}"
    )
    dataset_yaml.write_text(yaml_text, encoding="utf-8")
    summary = {
        "scene": str(scene),
        "mesh": str(mesh_path),
        "mesh_unit": args.mesh_unit,
        "task": args.task,
        "class_id": args.class_id,
        "class_name": args.class_name,
        "split": args.split,
        "depth_to_metre": depth_scale,
        "camera_matrix": K.tolist(),
        "mask_mode": args.mask_mode,
        "frames_requested": len(selected),
        "frames_labeled": sum(bool(r["has_label"]) for r in records),
        "total_seconds": time.time() - started,
        "records": records,
        "note": "YOLO segmentation uses the largest visible external contour; visually inspect rendered masks and labels.",
    }
    summary_path = output / f"annotation_summary_{args.split}_class_{args.class_id:03d}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{summary['frames_labeled']}/{len(selected)} 帧生成标签，输出 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
