#!/usr/bin/env python3
"""Tests for safe flat-scene migration."""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.migrate_scene_layout import apply, scan, validate  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="atec_migrate_") as tmp:
        root = Path(tmp) / "project"
        scene = root / "data/scenes/yellow_can_train_01"
        scene.mkdir(parents=True)
        (scene / "rgb").mkdir()
        (scene / "rgb/000001.png").write_bytes(b"keep")
        manifest = root / "manifests/yellow_can_train_01_train.yaml"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            "project:\n  scene: ../data/scenes/yellow_can_train_01\n  split: train\n",
            encoding="utf-8",
        )
        second_scene = root / "data/scenes/can_train_01"
        second_scene.mkdir(parents=True)
        (second_scene / "rgb").mkdir()
        (second_scene / "rgb/000002.png").write_bytes(b"keep-second")
        second_manifest = root / "manifests/can_train_01_train.yaml"
        second_manifest.write_text(
            "project:\n  scene: ../data/scenes/can_train_01\n  split: train\n",
            encoding="utf-8",
        )
        items = scan(root)
        assert len(items) == 2
        assert all(item.class_name == "can" for item in items)
        assert not validate(items)
        apply(items)
        target = root / "data/scenes/can/yellow_can_train_01"
        second_target = root / "data/scenes/can/can_train_01"
        assert (target / "rgb/000001.png").read_bytes() == b"keep"
        assert (second_target / "rgb/000002.png").read_bytes() == b"keep-second"
        assert not scene.exists()
        assert not second_scene.exists()
        assert "scene: ../data/scenes/can/yellow_can_train_01" in manifest.read_text(encoding="utf-8")
        assert "scene: ../data/scenes/can/can_train_01" in second_manifest.read_text(encoding="utf-8")
        assert not scan(root), "already-migrated class directories must not be rescanned as scenes"

        conflict = root / "data/scenes/can/yellow_can_test_01"
        conflict.mkdir(parents=True)
        source = root / "data/scenes/yellow_can_test_01"
        source.mkdir(parents=True)
        conflict_items = scan(root)
        assert any(validate(conflict_items))
        try:
            apply(conflict_items)
        except RuntimeError:
            pass
        else:
            raise AssertionError("migration must stop on target conflict")
        assert source.exists()

    print("SCENE_MIGRATION_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
