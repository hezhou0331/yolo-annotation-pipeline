#!/usr/bin/env python3
"""Regression checks for no-CAD defaults and train/val leakage guards."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
TMP = Path('/tmp/atec_dataset_safety')


def run(command, expect=0):
    completed = subprocess.run([str(x) for x in command], text=True, capture_output=True)
    if completed.returncode != expect:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise AssertionError((completed.returncode, expect, command))
    return completed


def write_image(path: Path, value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((48, 64, 3), value, np.uint8)
    cv2.rectangle(image, (8, 8), (35, 35), (value // 2, 255 - value, 100), -1)
    assert cv2.imwrite(str(path), image)


def write_label(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('0 0.125 0.166667 0.5625 0.166667 0.5625 0.75 0.125 0.75\n', encoding='utf-8')


def write_scene_report(path: Path, scene: Path, split: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'scene': str(scene), 'split': split,
        'frame_status_counts': {'accepted': 1, 'review': 0, 'rejected': 0},
    }, ensure_ascii=False), encoding='utf-8')


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)


    # 关键帧编辑器的核心操作应支持多块添加、擦除和续画。
    sys.path.insert(0, str(ROOT / 'tools'))
    from draw_first_mask import apply_polygon, load_initial_mask, save_mask
    edited = np.zeros((40, 50), np.uint8)
    apply_polygon(edited, [(3, 3), (20, 3), (20, 20), (3, 20)], erase=False)
    apply_polygon(edited, [(25, 5), (40, 5), (40, 18), (25, 18)], erase=False)
    before_erase = int((edited > 0).sum())
    apply_polygon(edited, [(8, 8), (15, 8), (15, 15), (8, 15)], erase=True)
    assert 0 < int((edited > 0).sum()) < before_erase
    edited_path = TMP / 'edited_mask.png'
    save_mask(edited_path, edited)
    resumed = load_initial_mask(edited_path, edited.shape)
    assert np.array_equal(edited, resumed)

    # 无CAD项目必须默认可运行：易拉罐使用SAM2，而不是强制要求不存在的mesh。
    project = TMP / 'project_default'
    run([sys.executable, ROOT / 'tools/prepare_atec_project.py',
         '--project-root', project, '--scene-name', 'train_bg01', '--split', 'train'])
    manifest = yaml.safe_load((project / 'manifests/train_bg01_train.yaml').read_text(encoding='utf-8'))
    can = next(x for x in manifest['instances'] if x['instance_id'] == 'can_01')
    assert can['tracker'] == 'sam2', can
    assert 'key_mask_dir' in can and 'mesh' not in can, can
    assert manifest['project']['allow_val_fallback_for_smoke'] is False

    project_multi = TMP / 'project_multi'
    run([sys.executable, ROOT / 'tools/prepare_atec_project.py',
         '--project-root', project_multi, '--scene-name', 'train_multi', '--split', 'train',
         '--instances-per-class', '3', '--include-bins'])
    manifest_multi = yaml.safe_load((project_multi / 'manifests/train_multi_train.yaml').read_text(encoding='utf-8'))
    assert len(manifest_multi['instances']) == 15, len(manifest_multi['instances'])
    assert len({x['instance_id'] for x in manifest_multi['instances']}) == 15
    assert sum(x['class_id'] == 0 for x in manifest_multi['instances']) == 3
    assert {x['class_id'] for x in manifest_multi['instances'] if x['class_id'] >= 4} == {4, 5, 6}

    project_fp = TMP / 'project_fp'
    run([sys.executable, ROOT / 'tools/prepare_atec_project.py',
         '--project-root', project_fp, '--scene-name', 'train_bg01', '--split', 'train',
         '--can-tracker', 'foundationpose'])
    manifest_fp = yaml.safe_load((project_fp / 'manifests/train_bg01_train.yaml').read_text(encoding='utf-8'))
    can_fp = next(x for x in manifest_fp['instances'] if x['instance_id'] == 'can_01')
    assert can_fp['tracker'] == 'foundationpose' and can_fp['mesh_unit'] == 'm'

    # 参数化易拉罐mesh应是米制、闭合且尺寸与实测值一致。
    mesh_path = TMP / 'meshes/can_01/textured_mesh.obj'
    run([sys.executable, ROOT / 'tools/generate_primitive_mesh.py', '--shape', 'cylinder',
         '--diameter', '66', '--height', '122', '--unit', 'mm', '--output', mesh_path])
    mesh = trimesh.load(mesh_path, process=False)
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) >= 100 and len(mesh.faces) >= 200
    assert np.allclose(mesh.extents, [0.066, 0.066, 0.122], atol=1e-6), mesh.extents
    metadata = json.loads(mesh_path.with_suffix('.obj.json').read_text(encoding='utf-8'))
    assert metadata['output_unit'] == 'm'
    assert metadata['foundationpose_manifest']['mesh_unit'] == 'm'

    # 构造不同内容、不同场景的train/val，严格检查应通过。
    dataset = TMP / 'dataset'
    train_image = dataset / 'images/train/train_bg01_000000.png'
    val_image = dataset / 'images/val/val_bg01_000000.png'
    write_image(train_image, 40)
    write_image(val_image, 180)
    write_label(dataset / 'labels/train/train_bg01_000000.txt')
    write_label(dataset / 'labels/val/val_bg01_000000.txt')
    (dataset / 'dataset.yaml').write_text(yaml.safe_dump({
        'path': str(dataset), 'train': 'images/train', 'val': 'images/val',
        'test': 'images/val', 'names': {0: 'can', 1: 'watermelon_rind'},
    }, sort_keys=False), encoding='utf-8')
    write_scene_report(dataset / 'project_reports/train_bg01_train_report.json', TMP / 'scenes/train_bg01', 'train')
    write_scene_report(dataset / 'project_reports/val_bg01_val_report.json', TMP / 'scenes/val_bg01', 'val')
    passed = run([sys.executable, ROOT / 'tools/train_yolo11_seg.py', '--data', dataset / 'dataset.yaml',
                  '--validate-only', '--require-project-reports'])
    assert '数据验证通过' in passed.stdout

    # 即使文件名不同，内容相同也必须拒绝，防止train/val泄漏。
    shutil.copy2(train_image, val_image)
    duplicated = run([sys.executable, ROOT / 'tools/train_yolo11_seg.py', '--data', dataset / 'dataset.yaml',
                      '--validate-only', '--require-project-reports'], expect=1)
    assert '内容完全相同' in duplicated.stdout + duplicated.stderr

    # 恢复不同图片后，同一RGB-D场景跨split复用也必须拒绝。
    write_image(val_image, 180)
    write_scene_report(dataset / 'project_reports/val_bg01_val_report.json', TMP / 'scenes/train_bg01', 'val')
    reused = run([sys.executable, ROOT / 'tools/train_yolo11_seg.py', '--data', dataset / 'dataset.yaml',
                  '--validate-only', '--require-project-reports'], expect=1)
    assert '同一个RGB-D场景' in reused.stdout + reused.stderr

    print('DATASET_SAFETY_ASSERTIONS_PASSED')
    print(json.dumps({
        'default_can_tracker': can['tracker'],
        'foundationpose_opt_in': can_fp['tracker'],
        'multi_instance_count': len(manifest_multi['instances']),
        'mesh_extents_m': mesh.extents.tolist(),
        'dataset': str(dataset),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
