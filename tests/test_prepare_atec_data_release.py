#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "prepare_atec_data_release.py"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


with tempfile.TemporaryDirectory(prefix="atec_release_prepare_") as tmp:
    fake_repo = Path(tmp) / "YOLO_Annotation_Pipeline"
    project = fake_repo / "projects" / "atec_real"
    data = project / "data"
    dataset = project / "datasets" / "atec_yolo11_seg"
    write(data / "scenes" / "meal_box" / "scene_01" / "metadata.json", json.dumps({"scene": str(data / "scenes" / "meal_box" / "scene_01")}))
    write(data / "key_masks" / "scene_01" / "box_01" / "000000.png", "mask")
    write(data / ".staging" / "unfinished" / "frame.png", "skip")
    write(dataset / "images" / "train" / "scene_01_000000.png", "image")
    os.link(dataset / "images" / "train" / "scene_01_000000.png", data / "scenes" / "meal_box" / "scene_01" / "000000.png")
    write(dataset / "labels" / "train" / "scene_01_000000.txt", "2 0.1 0.1 0.2 0.2 0.3 0.3\n")
    write(dataset / "images" / "val" / "scene_02_000000.png", "val-image")
    write(dataset / "labels" / "val" / "scene_02_000000.txt", "2 0.1 0.1 0.2 0.2 0.3 0.3\n")
    write(dataset / "project_reports" / "scene_01_report.json", json.dumps({"scene": str(data / "scenes" / "meal_box" / "scene_01")}))
    write(dataset / "dataset.yaml", "path: .\ntrain: images/train\nval: images/val\n")
    write(dataset / "labels" / "train.cache", "skip")
    write(dataset / "_staging" / "scene_01" / "temp.txt", "skip")
    write(dataset / "logs" / "scene_01" / "run.log", "skip")

    output = Path(tmp) / "release"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(fake_repo), "--output", str(output), "--tag", "data-test"],
        check=True,
    )
    snapshot = json.loads((output / "projects" / "atec_real" / "data_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["tag"] == "data-test"
    assert snapshot["contents"]["source_scenes"] == 1
    assert snapshot["contents"]["formal_total_pairs"] == 2
    assert snapshot["rewritten_absolute_paths"] == 2
    assert not (output / "projects" / "atec_real" / "data" / ".staging").exists()
    assert not (output / "projects" / "atec_real" / "datasets" / "atec_yolo11_seg" / "_staging").exists()
    assert not (output / "projects" / "atec_real" / "datasets" / "atec_yolo11_seg" / "logs").exists()
    assert not (output / "projects" / "atec_real" / "datasets" / "atec_yolo11_seg" / "labels" / "train.cache").exists()
    copied_image = output / "projects" / "atec_real" / "datasets" / "atec_yolo11_seg" / "images" / "train" / "scene_01_000000.png"
    copied_scene = output / "projects" / "atec_real" / "data" / "scenes" / "meal_box" / "scene_01" / "000000.png"
    assert copied_image.stat().st_ino == copied_scene.stat().st_ino
    metadata = json.loads((output / "projects" / "atec_real" / "data" / "scenes" / "meal_box" / "scene_01" / "metadata.json").read_text(encoding="utf-8"))
    assert not os.path.isabs(metadata["scene"])

print("PREPARE_ATEC_DATA_RELEASE_ASSERTIONS_PASSED")
