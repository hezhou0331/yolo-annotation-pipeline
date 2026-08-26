#!/usr/bin/env python3
"""Offscreen GUI tests for the independent one-click batch controller."""
from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt5.QtCore import QProcess  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402
import atec_pipeline.gui_app as gui_app_module  # noqa: E402
from atec_pipeline.gui_app import AtecMainWindow  # noqa: E402


def make_scene(root: Path, name: str, *, masks: bool, exported: bool = False) -> Path:
    scene = root / "data/scenes/can" / name
    (scene / "rgb").mkdir(parents=True)
    (scene / "depth").mkdir()
    (scene / "rgb/000000.png").write_bytes(b"rgb")
    (scene / "depth/000000.png").write_bytes(b"depth")
    (scene / "metadata.json").write_text(json.dumps({"frames": [{
        "id": "000000", "color_timestamp_ms": 1000.0, "depth_timestamp_ms": 1000.0,
    }]}), encoding="utf-8")
    manifest = root / "manifests" / f"{name}_train.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "project:\n"
        f"  scene: ../data/scenes/can/{name}\n"
        "  output: ../datasets/atec_yolo11_seg\n"
        "  split: train\n"
        f"  capture_session_id: session_{name}\n"
        f"  source_video_id: {name}_clip_01\n"
        f"  name_prefix: {name}_\n"
        "classes:\n  0: can\n"
        "instances:\n- instance_id: can_01\n  class_id: 0\n  class_name: can\n  tracker: sam2\n"
        f"  key_mask_dir: ../data/key_masks/{name}/can_01\n",
        encoding="utf-8",
    )
    mask = root / "data/key_masks" / name / "can_01/000000.png"
    report = scene / "project_reports/segments.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"segments": [{
        "segment_id": 0, "start_id": "000000", "end_id": "000000",
        "required_key_mask_paths": {"can_01": str(mask)},
    }]}), encoding="utf-8")
    if masks:
        mask.parent.mkdir(parents=True)
        mask.write_bytes(b"mask")
    if exported:
        write_export(root, name, accepted=1)
    return scene


def write_export(root: Path, name: str, *, accepted: int) -> None:
    path = root / "datasets/atec_yolo11_seg/project_reports" / f"{name}_train_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"frame_status_counts": {
        "accepted": accepted, "review": 0, "rejected": 0 if accepted else 1,
    }}), encoding="utf-8")


def scene_item(window: AtecMainWindow, name: str):
    return next(
        window.scene_list.item(index)
        for index in range(window.scene_list.count())
        if name in window.scene_list.item(index).text()
    )


class FakeCloseEvent:
    def __init__(self) -> None:
        self.accepted = False
        self.ignored = False

    def accept(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.ignored = True


class FakeBatchProcess:
    def __init__(self, *, state: QProcess.ProcessState = QProcess.NotRunning, pid: int = 0) -> None:
        self.current_state = state
        self.pid = pid
        self.started: list[tuple[str, list[str]]] = []
        self.terminate_calls = 0

    def state(self) -> QProcess.ProcessState:
        return self.current_state

    def start(self, program: str, args: list[str]) -> None:
        self.started.append((program, args))

    def processId(self) -> int:
        return self.pid

    def terminate(self) -> None:
        self.terminate_calls += 1


def main() -> int:
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="atec_gui_auto_") as tmp:
        root = Path(tmp) / "project"
        make_scene(root, "can_01_pending", masks=True)
        make_scene(root, "can_02_pending", masks=True)
        make_scene(root, "can_03_manual", masks=False)
        make_scene(root, "can_04_ready", masks=True, exported=True)

        window = AtecMainWindow(project_root=root, config_path=ROOT / "configs/atec_objects.yaml")
        assert window.auto_prepare_button.text() == "一键自动处理所有场景"
        assert window.stop_auto_button.text() == "停止自动处理"
        assert not window.advanced_panel.isAncestorOf(window.auto_prepare_button), (
            "daily one-click processing must remain visible when advanced controls are folded"
        )
        assert not window.advanced_panel.isAncestorOf(window.auto_progress)
        assert window.auto_process is not window.process
        assert window.auto_progress.minimum() == 0
        assert "尚未开始" in window.auto_stage_label.text()

        launched: list[tuple[str, str, list[str]]] = []
        window._start_auto_task = lambda kind, program, args: launched.append((kind, program, args))  # type: ignore[method-assign]
        window.start_auto_prepare()
        assert window._auto_batch_active
        assert window._auto_total == 4
        assert launched and launched[0][0] == "auto_run"
        first_scene = window._auto_current_plan.scene_name
        assert first_scene == "can_01_pending"
        assert "can_01_pending" in window.auto_scene_label.text()
        assert "2/4" in window.auto_overall_label.text(), "manual and skipped scenes are terminal immediately"

        window._load_scene_item(scene_item(window, first_scene))
        window._refresh_state(refresh_scenes=False)
        assert not window.mark_button.isEnabled()
        assert not window.integrity_button.isEnabled()
        assert "只读" in window.guidance_label.text()

        window._load_scene_item(scene_item(window, "can_03_manual"))
        window._refresh_state(refresh_scenes=False)
        assert window.mark_button.isEnabled(), "other scenes must remain editable during batch processing"

        window._auto_process_finished(1, QProcess.NormalExit)
        assert len(launched) == 2, "one scene failure must advance to the next scene"
        assert window._auto_current_plan.scene_name == "can_02_pending"
        write_export(root, "can_02_pending", accepted=3)
        window._auto_process_finished(0, QProcess.NormalExit)
        assert not window._auto_batch_active
        latest = root / "project_reports/auto_processing/latest.json"
        payload = json.loads(latest.read_text(encoding="utf-8"))
        assert payload["summary"]["success"] == 1
        assert payload["summary"]["failed"] == 1
        assert payload["summary"]["manual"] == 1
        assert payload["summary"]["skipped"] == 1
        assert payload["cancelled"] is False

        # A new run re-scans filesystem state. Stopping does not delete prior
        # reports or accepted export, and leaves unfinished work for the next run.
        launched.clear()
        window.start_auto_prepare()
        assert window._auto_batch_active and launched
        window.stop_auto_processing()
        assert not window._auto_batch_active
        cancelled = json.loads(latest.read_text(encoding="utf-8"))
        assert cancelled["cancelled"] is True
        assert (root / "datasets/atec_yolo11_seg/project_reports/can_02_pending_train_report.json").is_file()

        window.close()

        # FailedToStart is reported through errorOccurred.  A later stale
        # finished signal from that same launch must not fail the next scene.
        failed_start_root = Path(tmp) / "failed_start_project"
        make_scene(failed_start_root, "can_01_pending", masks=True)
        make_scene(failed_start_root, "can_02_pending", masks=True)
        failed_start_window = AtecMainWindow(
            project_root=failed_start_root, config_path=ROOT / "configs/atec_objects.yaml"
        )
        failed_start_launches: list[tuple[str, str, list[str]]] = []
        failed_start_window._start_auto_task = (  # type: ignore[method-assign]
            lambda kind, program, args: failed_start_launches.append((kind, program, args))
        )
        failed_start_window.start_auto_prepare()
        failed_scene = failed_start_window._auto_current_plan.scene_name
        failed_start_window._auto_process_error(QProcess.FailedToStart)
        failed_start_window._auto_process_finished(-1, QProcess.CrashExit)
        app.processEvents()
        assert failed_scene == "can_01_pending"
        assert failed_start_window._auto_batch_active
        assert failed_start_window._auto_current_plan.scene_name == "can_02_pending"
        assert len(failed_start_launches) == 2
        failed_records = [
            record for record in failed_start_window._auto_batch_records if record.status == "failed"
        ]
        assert [record.scene_name for record in failed_records] == ["can_01_pending"]
        failed_start_window.stop_auto_processing()
        failed_start_window.close()

        # Closing the App while a batch is active must stop it and atomically
        # persist a cancelled report, even if no foreground/live task exists.
        close_root = Path(tmp) / "close_project"
        make_scene(close_root, "can_close_pending", masks=True)
        close_window = AtecMainWindow(
            project_root=close_root, config_path=ROOT / "configs/atec_objects.yaml"
        )
        close_window._start_auto_task = lambda *_args: None  # type: ignore[method-assign]
        close_window.start_auto_prepare()
        close_event = FakeCloseEvent()
        original_question = QMessageBox.question
        QMessageBox.question = lambda *_args, **_kwargs: QMessageBox.Yes  # type: ignore[assignment]
        try:
            close_window.closeEvent(close_event)  # type: ignore[arg-type]
        finally:
            QMessageBox.question = original_question  # type: ignore[assignment]
        assert close_event.accepted and not close_event.ignored
        assert not close_window._auto_batch_active
        close_payload = json.loads(
            (close_root / "project_reports/auto_processing/latest.json").read_text(encoding="utf-8")
        )
        assert close_payload["cancelled"] is True
        assert close_payload["summary"]["cancelled"] == 1

        # The batch wrapper itself spawns SAM2/annotation children.  It must run
        # in a separate process group and Stop must signal that whole group.
        process_root = Path(tmp) / "process_group_project"
        process_window = AtecMainWindow(
            project_root=process_root, config_path=ROOT / "configs/atec_objects.yaml"
        )
        original_auto_process = process_window.auto_process
        fake_process = FakeBatchProcess()
        process_window.auto_process = fake_process  # type: ignore[assignment]
        process_window._start_auto_task("auto_run", "/tmp/worker", ["run", "scene.yaml"])
        assert fake_process.started == [("setsid", ["/tmp/worker", "run", "scene.yaml"])]

        fake_process.current_state = QProcess.Running
        fake_process.pid = 4242
        process_window._auto_batch_active = True
        signalled: list[tuple[int, int]] = []
        original_killpg = gui_app_module.os.killpg
        gui_app_module.os.killpg = lambda pid, sig: signalled.append((pid, sig))  # type: ignore[assignment]
        try:
            process_window.stop_auto_processing()
        finally:
            gui_app_module.os.killpg = original_killpg  # type: ignore[assignment]
        assert signalled == [(4242, signal.SIGTERM)]
        assert fake_process.terminate_calls == 0
        process_window._auto_batch_active = False
        process_window.auto_process = original_auto_process
        process_window.close()
    app.processEvents()
    print("GUI_AUTO_PROCESSING_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
