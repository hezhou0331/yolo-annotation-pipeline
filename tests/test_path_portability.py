#!/usr/bin/env python3
"""Regression coverage for repository relocation and portable generated paths."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atec_pipeline.path_compat import (  # noqa: E402
    infer_project_root,
    portable_path,
    resolve_compatible_path,
)
from tools.migrate_scene_layout import apply, scan  # noqa: E402
from tools.split_dataset_by_scene import _load_dataset  # noqa: E402
from tools.train_yolo11_seg import resolve_dataset  # noqa: E402


def run(*args: str | Path) -> None:
    subprocess.run([sys.executable, *(str(arg) for arg in args)], check=True)


def assert_relative_to(value: str, *, base: Path, expected: Path) -> None:
    assert not Path(value).is_absolute(), value
    assert (base / value).resolve() == expected.resolve(), (value, base, expected)


def write_frame(scene: Path, stem: str = "000000") -> None:
    (scene / "rgb").mkdir(parents=True, exist_ok=True)
    (scene / "depth").mkdir(parents=True, exist_ok=True)
    rgb = np.full((16, 20, 3), 96, dtype=np.uint8)
    depth = np.full((16, 20), 1000, dtype=np.uint16)
    assert cv2.imwrite(str(scene / "rgb" / f"{stem}.png"), rgb)
    assert cv2.imwrite(str(scene / "depth" / f"{stem}.png"), depth)


def check_shared_helpers(tmp: Path) -> None:
    repository = tmp / "data" / "current" / "YOLO_Annotation_Pipeline"
    project = repository / "projects" / "portable"
    project.mkdir(parents=True)
    legacy = Path("/home/old/data/archive/YOLO_Annotation_Pipeline")

    cases = {
        legacy / "data/scenes/can/scene_01": project / "data/scenes/can/scene_01",
        legacy / "datasets/atec/images/train": project / "datasets/atec/images/train",
        legacy / "assets/meshes/can.obj": project / "assets/meshes/can.obj",
        legacy / "manifests/can_train.yaml": project / "manifests/can_train.yaml",
        legacy / "reports/summary.json": project / "reports/summary.json",
        legacy / "models/sam2.1_t.pt": repository / "models/sam2.1_t.pt",
        legacy / "third_party/FoundationPose/run_demo.py": repository / "third_party/FoundationPose/run_demo.py",
    }
    for old_path, expected in cases.items():
        actual = resolve_compatible_path(
            old_path,
            repository_root=repository,
            project_root=project,
        )
        assert actual == expected.resolve(), (old_path, actual, expected)

    project_only_legacy = Path("/mnt/archive") / project.name / "data/scenes/can/scene_02"
    assert resolve_compatible_path(
        project_only_legacy,
        repository_root=repository,
        project_root=project,
    ) == (project / "data/scenes/can/scene_02").resolve()

    canonical_project_legacy = Path("/producer/projects/atec_real/reports/summary.json")
    assert resolve_compatible_path(
        canonical_project_legacy,
        repository_root=repository,
        project_root=project,
    ) == (project / "reports/summary.json").resolve()

    nested_marker = legacy / "data/archive/models/future.pt"
    assert resolve_compatible_path(
        nested_marker,
        repository_root=repository,
        project_root=project,
    ) == (project / "data/archive/models/future.pt").resolve()

    # Missing external outputs are valid paths too. Common directory names must
    # not make them look like an old checkout without a repo/project-name anchor.
    missing_external_root = Path("/srv") / f"{tmp.name}_missing_external"
    assert not missing_external_root.exists()
    for marker in ("data", "datasets", "manifests", "assets", "reports", "models", "third_party"):
        external_output = missing_external_root / marker / "future/output.bin"
        assert resolve_compatible_path(
            external_output,
            repository_root=repository,
            project_root=project,
        ) == external_output.resolve(), (marker, external_output)

    unrelated_project = missing_external_root / "projects/unrelated/datasets/future.bin"
    assert resolve_compatible_path(
        unrelated_project,
        repository_root=repository,
        project_root=project,
    ) == unrelated_project.resolve()
    backup_name = missing_external_root / f"{repository.name}_backup/data/future.bin"
    assert resolve_compatible_path(
        backup_name,
        repository_root=repository,
        project_root=project,
    ) == backup_name.resolve()

    missing_report = missing_external_root / "reports/future.json"
    inferred = infer_project_root(missing_report, repository_root=repository, default=project)
    assert inferred == project.resolve()
    assert portable_path(
        missing_report,
        relative_to=project / "reports",
        repository_root=repository,
        project_root=inferred,
    ) == str(missing_report.resolve())

    external = tmp / "external/data/assets/model.bin"
    external.parent.mkdir(parents=True)
    external.write_bytes(b"external")
    assert resolve_compatible_path(
        external,
        repository_root=repository,
        project_root=project,
    ) == external.resolve()

    existing_legacy_like = tmp / "external" / repository.name / "models/external.pt"
    existing_legacy_like.parent.mkdir(parents=True)
    existing_legacy_like.write_bytes(b"external")
    assert resolve_compatible_path(
        existing_legacy_like,
        repository_root=repository,
        project_root=project,
    ) == existing_legacy_like.resolve()

    current_model = repository / "models/new.pt"
    assert resolve_compatible_path(
        current_model,
        repository_root=repository,
        project_root=project,
    ) == current_model.resolve()
    assert resolve_compatible_path(
        "../data/scenes/can/scene_01",
        base=project / "manifests",
        repository_root=repository,
        project_root=project,
    ) == (project / "data/scenes/can/scene_01").resolve()
    assert portable_path(
        project / "data/scenes/can/scene_01",
        relative_to=project / "manifests",
        repository_root=repository,
        project_root=project,
    ) == "../data/scenes/can/scene_01"
    assert portable_path(
        "/opt/external/model.pt",
        relative_to=project / "manifests",
        repository_root=repository,
        project_root=project,
    ) == "/opt/external/model.pt"

    dataset_path = project / "datasets/portable/dataset.yaml"
    dataset_path.parent.mkdir(parents=True)
    legacy_dataset_root = legacy / "datasets/portable"
    dataset_path.write_text(
        yaml.safe_dump({"path": str(legacy_dataset_root), "train": "images/train", "val": "images/val"}),
        encoding="utf-8",
    )
    split_root, _ = _load_dataset(dataset_path)
    _, _, training_root = resolve_dataset(dataset_path)
    assert split_root == dataset_path.parent.resolve()
    assert training_root == dataset_path.parent.resolve()


def check_generated_files(tmp: Path) -> None:
    project = tmp / "project"
    source = project / "data/scenes/yellow_can_portable_01"
    write_frame(source)
    manifest_path = project / "manifests/portable.yaml"
    manifest_path.parent.mkdir(parents=True)
    legacy_scene = "/home/old/data/archive/YOLO_Annotation_Pipeline/data/scenes/yellow_can_portable_01"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "classes": {0: "can"},
                "project": {
                    "scene": legacy_scene,
                    "output": "../datasets/portable",
                    "split": "train",
                    "name_prefix": "portable_",
                    "image_mode": "copy",
                },
                "instances": [
                    {
                        "instance_id": "can_01",
                        "class_id": 0,
                        "class_name": "can",
                        "tracker": "mask_sequence",
                        "key_mask_dir": "../data/key_masks/can_01",
                    }
                ],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    items = scan(project)
    assert len(items) == 1, items
    apply(items)
    scene = project / "data/scenes/can/yellow_can_portable_01"
    migrated = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert_relative_to(migrated["project"]["scene"], base=manifest_path.parent, expected=scene)

    key_mask = project / "data/key_masks/can_01/000000.png"
    key_mask.parent.mkdir(parents=True)
    mask = np.full((16, 20), 255, dtype=np.uint8)
    assert cv2.imwrite(str(key_mask), mask)
    segment_report_path = scene / "project_reports/segments.json"
    run(
        ROOT / "tools/segment_rgbd_sequence.py",
        "--manifest",
        manifest_path,
        "--output",
        segment_report_path,
        "--require-ready",
    )
    segment_report = json.loads(segment_report_path.read_text(encoding="utf-8"))
    segment_base = segment_report_path.parent
    assert segment_report["format_version"] == 2
    assert_relative_to(segment_report["scene"], base=segment_base, expected=scene)
    assert_relative_to(segment_report["manifest"], base=segment_base, expected=manifest_path)
    assert_relative_to(segment_report["frames"][0]["rgb"], base=segment_base, expected=scene / "rgb/000000.png")
    assert_relative_to(segment_report["frames"][0]["depth"], base=segment_base, expected=scene / "depth/000000.png")
    segment = segment_report["segments"][0]
    assert_relative_to(segment["key_masks"]["can_01"], base=segment_base, expected=key_mask)
    assert_relative_to(segment["required_key_mask_paths"]["can_01"], base=segment_base, expected=key_mask)

    output = project / "datasets/portable"
    stage = output / "_staging/yellow_can_portable_01/can_01"
    quality_dir = stage / "quality_reports/train/class_000_can_01"
    label_dir = stage / "labels/train"
    mask_dir = stage / "rendered_masks/train/class_000_can_01"
    for directory in (quality_dir, label_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)
    quality_dir.joinpath("quality_report.json").write_text(
        json.dumps(
            {
                "status_counts": {"accepted": 1},
                "records": [
                    {
                        "id": "000000",
                        "output_id": "portable_000000",
                        "status": "accepted",
                        "has_label": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    label_dir.joinpath("portable_000000.txt").write_text(
        "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n",
        encoding="utf-8",
    )
    assert cv2.imwrite(str(mask_dir / "portable_000000.png"), mask)
    run(ROOT / "tools/annotate_multinstance_project.py", "--manifest", manifest_path, "--skip-tracking")

    dataset = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset["path"] == ".", dataset
    export_report_path = output / "project_reports/yellow_can_portable_01_train_report.json"
    export_report = json.loads(export_report_path.read_text(encoding="utf-8"))
    export_base = export_report_path.parent
    assert export_report["format_version"] == 2
    assert_relative_to(export_report["manifest"], base=export_base, expected=manifest_path)
    assert_relative_to(export_report["scene"], base=export_base, expected=scene)
    assert_relative_to(export_report["output"], base=export_base, expected=output)
    assert_relative_to(export_report["instances"][0]["stage"], base=export_base, expected=stage)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="atec_path_portability_") as directory:
        tmp = Path(directory)
        check_shared_helpers(tmp)
        check_generated_files(tmp)
    print("PATH_PORTABILITY_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
