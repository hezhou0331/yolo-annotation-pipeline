#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
TMP = Path('/tmp/atec_annotation_pipeline_smoke')


def rect_mask(shape, xyxy):
    mask = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def run(cmd):
    print('RUN', ' '.join(map(str, cmd)))
    subprocess.run([str(x) for x in cmd], check=True)


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    scene = TMP / 'scene'
    rgb_dir = scene / 'rgb'
    depth_dir = scene / 'depth'
    masks_a = TMP / 'masks_a'
    masks_b = TMP / 'masks_b'
    for d in (rgb_dir, depth_dir, masks_a, masks_b):
        d.mkdir(parents=True, exist_ok=True)

    h, w = 120, 160
    for i in range(7):
        stem = f'{i:06d}'
        image = np.full((h, w, 3), (35, 45, 55), dtype=np.uint8)
        cv2.putText(image, stem, (4, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        depth = np.full((h, w), 1000, dtype=np.uint16)

        a_boxes = {
            0: (20, 30, 50, 60),
            1: (20, 30, 50, 60),
            2: (4, 8, 115, 95),       # 面积突变 -> rejected
            4: (0, 30, 30, 60),       # 触边 -> review
            5: (20, 30, 50, 60),      # 深度无效 -> rejected
            6: (20, 30, 50, 60),
        }
        if i in a_boxes:
            mask_a = rect_mask((h, w), a_boxes[i])
            cv2.imwrite(str(masks_a / f'{stem}.png'), mask_a)
            image[mask_a > 0] = (20, 40, 210)
            if i == 5:
                depth[mask_a > 0] = 0
        # i==3 故意不写A mask -> mask_file_missing

        b_box = (52, 30, 82, 60) if i != 1 else (48, 30, 78, 60)
        mask_b = rect_mask((h, w), b_box)
        cv2.imwrite(str(masks_b / f'{stem}.png'), mask_b)
        image[mask_b > 0] = (210, 120, 20)

        cv2.imwrite(str(rgb_dir / f'{stem}.png'), image)
        cv2.imwrite(str(depth_dir / f'{stem}.png'), depth)

    (scene / 'camera.json').write_text(json.dumps({
        'depth_scale_to_metre': 0.001,
        'intrinsics': {'fx': 120.0, 'fy': 120.0, 'cx': 80.0, 'cy': 60.0}
    }, indent=2), encoding='utf-8')

    mask_output = TMP / 'mask_output'
    run([
        sys.executable, ROOT / 'tools/annotate_mask_sequence_yolo.py',
        '--scene', scene, '--mask-dir', masks_a, '--output', mask_output,
        '--class-id', '0', '--class-name', 'can', '--instance-id', 'a',
        '--name-prefix', 'smoke_', '--image-mode', 'hardlink'
    ])
    report_a_path = mask_output / 'quality_reports/train/class_000_a/quality_report.json'
    report_a = json.loads(report_a_path.read_text(encoding='utf-8'))
    assert report_a['status_counts'] == {'accepted': 3, 'review': 1, 'rejected': 3}, report_a['status_counts']
    statuses_a = {r['id']: r['status'] for r in report_a['records']}
    assert statuses_a == {
        '000000': 'accepted', '000001': 'accepted', '000002': 'rejected',
        '000003': 'rejected', '000004': 'review', '000005': 'rejected',
        '000006': 'accepted'
    }, statuses_a
    assert not (mask_output / 'labels/train/smoke_000003.txt').exists()
    assert not any(p.stat().st_size == 0 for p in (mask_output / 'labels/train').glob('*.txt'))

    project_output = TMP / 'dataset'
    manifest = {
        'classes': {0: 'can', 1: 'watermelon_rind', 2: 'meal_box', 3: 'red_paper_bag', 4: 'blue_bin', 5: 'green_bin', 6: 'red_bin'},
        'project': {
            'scene': str(scene), 'output': str(project_output), 'split': 'train',
            'name_prefix': 'smoke_', 'image_mode': 'hardlink',
            'max_instance_overlap': 0.05, 'include_review': False,
            'mask_quality': {
                'min_mask_area': 80, 'min_depth_coverage': 0.25,
                'min_area_ratio': 0.35, 'max_area_ratio': 2.8,
                'max_center_shift_norm': 0.40, 'min_dominant_component_ratio': 0.70
            }
        },
        'instances': [
            {'instance_id': 'a', 'class_id': 0, 'class_name': 'can', 'tracker': 'mask_sequence', 'mask_dir': str(masks_a)},
            {'instance_id': 'b', 'class_id': 1, 'class_name': 'watermelon_rind', 'tracker': 'mask_sequence', 'mask_dir': str(masks_b)},
        ]
    }
    manifest_path = TMP / 'project.yaml'
    manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding='utf-8')
    run([sys.executable, ROOT / 'tools/annotate_multinstance_project.py', '--manifest', manifest_path])

    report_path = project_output / 'project_reports/scene_train_report.json'
    project_report = json.loads(report_path.read_text(encoding='utf-8'))
    statuses = {r['source_frame']: r['status'] for r in project_report['frames']}
    assert statuses['000000'] == 'accepted', statuses
    assert statuses['000001'] == 'review', statuses  # 两mask超过5%重叠阈值
    assert statuses['000006'] == 'accepted', statuses
    for stem in ('000002', '000003', '000004', '000005'):
        assert statuses[stem] == 'review', (stem, statuses)
    formal_labels = sorted((project_output / 'labels/train').glob('*.txt'))
    assert [p.stem for p in formal_labels] == ['smoke_000000', 'smoke_000006'], formal_labels
    assert all(len([x for x in p.read_text().splitlines() if x.strip()]) == 2 for p in formal_labels)
    assert not any(p.stat().st_size == 0 for p in project_output.rglob('*.txt'))
    assert len(list((project_output / 'images/train').glob('*.png'))) == 2

    # 安全回归：模拟B实例report整帧漏报。即使遗留label/mask还在，聚合也必须移除正式帧。
    b_report_path = project_output / '_staging/scene/b/quality_reports/train/class_001_b/quality_report.json'
    b_report = json.loads(b_report_path.read_text(encoding='utf-8'))
    b_report['records'] = [r for r in b_report['records'] if r['id'] != '000006']
    b_report_path.write_text(json.dumps(b_report, ensure_ascii=False, indent=2), encoding='utf-8')
    run([sys.executable, ROOT / 'tools/annotate_multinstance_project.py', '--manifest', manifest_path, '--skip-tracking'])
    project_report = json.loads(report_path.read_text(encoding='utf-8'))
    frame6 = next(r for r in project_report['frames'] if r['source_frame'] == '000006')
    assert frame6['status'] == 'review', frame6
    assert 'b:record_missing' in frame6['reasons'], frame6
    assert not (project_output / 'labels/train/smoke_000006.txt').exists()
    assert not (project_output / 'images/train/smoke_000006.png').exists()

    dataset_yaml = yaml.safe_load((project_output / 'dataset.yaml').read_text(encoding='utf-8'))
    assert dataset_yaml['names'][6] == 'red_bin', dataset_yaml
    assert dataset_yaml['val'] == 'images/val', dataset_yaml
    assert dataset_yaml['test'] == 'images/val', dataset_yaml
    print('ALL_ASSERTIONS_PASSED')
    print(json.dumps({
        'mask_sequence_counts': report_a['status_counts'],
        'initial_project_counts': {'accepted': 2, 'review': 5, 'rejected': 0},
        'record_missing_guard': frame6['reasons'],
        'output': str(TMP)
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
