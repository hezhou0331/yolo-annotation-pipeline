#!/usr/bin/env python3
"""Offscreen smoke tests for the thin PyQt5 coordinator; no camera is started."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from types import SimpleNamespace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt5.QtCore import QProcess, Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMessageBox, QTabWidget  # noqa: E402
from atec_pipeline.gui_app import AtecMainWindow  # noqa: E402
from atec_pipeline.gui_state import manifest_for_session  # noqa: E402


def main() -> int:
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="atec_gui_app_") as tmp:
        project_root = Path(tmp) / "project"
        window = AtecMainWindow(project_root=project_root, config_path=ROOT / "configs/atec_objects.yaml")
        assert window.epochs.value() == 100
        assert window.batch.value() == 4
        assert window.device.text() == "0"
        assert window.init_mode.currentText() == "baseline"
        assert window.experiment.text() == "atec_9class_reviewed_20260829"
        train_program, train_args = window.command_for_train()
        assert train_program == str(ROOT / "scripts/atec-pipeline")
        assert train_args[train_args.index("--name") + 1] == "atec_9class_reviewed_20260829"

        # Daily operation is one thin page, not three workflow tabs.
        assert not isinstance(window.centralWidget(), QTabWidget)
        assert len(window.class_buttons) == 9
        assert window.class_buttons[0].text().startswith("1  易拉罐")
        assert window.class_buttons[7].text().startswith("8  紫色袋子")
        assert window.class_buttons[8].text().startswith("9  沙瓶")
        assert window.start_button.text() == "开始采集"
        assert window.stop_button.text().startswith("停止采集")
        assert window.stop_capture_shortcut.key().toString() == "Ctrl+C"
        assert not window.stop_capture_shortcut.isEnabled()
        assert window.mark_button.text().startswith("开始标记")
        assert window.continue_button.text() == "继续自动处理"
        assert window.prepare_button.text() == "准备训练数据"
        assert not hasattr(window, "auto_process")
        assert not hasattr(window, "auto_prepare_button")
        assert window.mask_help_button.text() == "标注器使用说明"
        assert "Enter" in window.mask_quick_help.text() and "S 保存" in window.mask_quick_help.text()
        assert "尚未运行" in window.propagation_status_label.text()
        assert window.advanced_panel.isHidden(), "algorithm/training controls must be folded by default"
        assert window.log.isHidden(), "logs should not clutter the daily workflow by default"

        window._select_class(1)
        assert window.selected_class.name == "can"
        assert window.session is not None
        program, args = window.command_for_capture()
        assert program == "setsid"
        assert "capture_orbbec.sh" in args[0]
        init_program, init_args = window.command_for_init()
        assert init_args[0] == "init"
        assert "--scene-class" in init_args and "can" in init_args
        dataset_yaml = project_root / "datasets/atec_yolo11_seg/dataset.yaml"
        assert window._dataset_path() == dataset_yaml
        validate_program, validate_args = window.command_for_validate()
        assert validate_args == ["validate", str(dataset_yaml)]
        assert not window.has_independent_val()
        assert not window.train_button.isEnabled()

        def scene_list_texts() -> list[str]:
            return [window.scene_list.item(index).text() for index in range(window.scene_list.count())]

        def make_scene(
            class_name: str,
            scene_name: str,
            *,
            masks_complete: bool,
            export_failed: bool = False,
            split: str = "train",
        ) -> Path:
            scene_dir = project_root / "data/scenes" / class_name / scene_name
            (scene_dir / "rgb").mkdir(parents=True)
            (scene_dir / "depth").mkdir()
            (scene_dir / "rgb/000000.png").write_bytes(b"rgb")
            (scene_dir / "depth/000000.png").write_bytes(b"depth")
            manifest = project_root / "manifests" / f"{scene_name}_{split}.yaml"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                f"project:\n  scene: {scene_name}\n  split: {split}\n",
                encoding="utf-8",
            )
            mask_path = project_root / "data/key_masks" / scene_name / "000000.png"
            segments = scene_dir / "project_reports/segments.json"
            segments.parent.mkdir(parents=True)
            segments.write_text(
                "{\n"
                '  "segments": [{"segment_id": 0, "start_id": "000000", "end_id": "000000", '
                f'"required_key_mask_paths": {{"{class_name}_01": "{mask_path}"}}}}]\n'
                "}\n",
                encoding="utf-8",
            )
            if masks_complete:
                mask_path.parent.mkdir(parents=True)
                mask_path.write_bytes(b"mask")
            if export_failed:
                report = (
                    project_root
                    / "datasets/atec_yolo11_seg/project_reports"
                    / f"{scene_name}_{split}_report.json"
                )
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(
                    '{"frame_status_counts": {"accepted": 0, "review": 0, "rejected": 1}}',
                    encoding="utf-8",
                )
            return scene_dir

        # Even with no saved scenes, both fixed groups remain visible for the selected class.
        assert window.saved_scenes_group.title() == "易拉罐（can）的已保存场次"
        empty_texts = scene_list_texts()
        assert "── 未标记完成（0）──" in empty_texts
        assert "暂无未标记完成的场次" in empty_texts
        assert "── 已标记完成（0）──" in empty_texts
        assert "暂无已标记完成的场次" in empty_texts
        assert "── 已经过人工 Review（0）──" in empty_texts
        assert "暂无已经过人工 Review的场次" in empty_texts

        can_incomplete = make_scene("can", "can_filter_incomplete", masks_complete=False)
        can_failed = make_scene("can", "can_filter_failed", masks_complete=True, export_failed=True)
        watermelon_complete = make_scene(
            "watermelon_rind", "watermelon_filter_complete", masks_complete=True, split="val"
        )

        window._select_class(1)
        can_texts = scene_list_texts()
        assert window.saved_scenes_group.title() == "易拉罐（can）的已保存场次"
        assert any("can_filter_incomplete" in text for text in can_texts)
        assert any("can_filter_failed" in text for text in can_texts)
        assert not any("watermelon_filter_complete" in text for text in can_texts)
        assert "── 未标记完成（1）──" in can_texts
        assert "── 已标记完成（1）──" in can_texts
        assert "── 已经过人工 Review（0）──" in can_texts
        assert "场次 2" in window.class_frame_summary.text()
        assert "RGB-D 总帧 2" in window.class_frame_summary.text()
        assert "已处理 1" in window.class_frame_summary.text()
        assert "accepted 0" in window.class_frame_summary.text()
        assert "rejected 1" in window.class_frame_summary.text()
        assert "待人工标记 1 场" in window.class_frame_summary.text()
        failed_item = next(
            window.scene_list.item(index)
            for index in range(window.scene_list.count())
            if "can_filter_failed" in window.scene_list.item(index).text()
        )
        assert "自动处理失败" in failed_item.text()
        assert failed_item.background().color().name() == "#ffd6d6"

        window._select_class(2)
        watermelon_texts = scene_list_texts()
        assert window.saved_scenes_group.title() == "仿真西瓜皮（watermelon_rind）的已保存场次"
        assert any("watermelon_filter_complete" in text for text in watermelon_texts)
        assert not any("can_filter_incomplete" in text for text in watermelon_texts)
        assert not any("can_filter_failed" in text for text in watermelon_texts)
        assert "── 未标记完成（0）──" in watermelon_texts
        assert "暂无未标记完成的场次" in watermelon_texts
        assert "── 已标记完成（1）──" in watermelon_texts
        assert "── 已经过人工 Review（0）──" in watermelon_texts
        assert "场次 1" in window.class_frame_summary.text()
        assert "RGB-D 总帧 1" in window.class_frame_summary.text()
        assert "待 SAM2 传播 1 场" in window.class_frame_summary.text()

        watermelon_item = next(
            window.scene_list.item(index)
            for index in range(window.scene_list.count())
            if "watermelon_filter_complete" in window.scene_list.item(index).text()
        )
        assert isinstance(watermelon_item.data(Qt.UserRole), dict)
        window._load_scene_item(watermelon_item)
        assert window.selected_class.name == "watermelon_rind"
        assert window.session is not None
        assert window.session.scene_name == watermelon_complete.name
        assert window.session.split == "val"

        all_states = window._all_scene_states()
        assert {state.class_name for state in all_states} == {"can", "watermelon_rind"}
        assert {state.scene_dir for state in all_states} == {can_incomplete, can_failed, watermelon_complete}

        # A saved scene with an interrupted final RGB write can be checked and
        # safely repaired from the daily-use scene panel without a terminal.
        integrity_scene = project_root / "data/scenes/can/can_integrity_repair"
        (integrity_scene / "rgb").mkdir(parents=True)
        (integrity_scene / "depth").mkdir()
        (integrity_scene / "rgb/000000.png").write_bytes(b"rgb")
        (integrity_scene / "depth/000000.png").write_bytes(b"depth")
        (integrity_scene / "rgb/000001.png").write_bytes(b"orphan")
        (integrity_scene / "metadata.json").write_text(
            '{"frames": [{"id": "000000"}]}', encoding="utf-8"
        )
        window._select_class(1)
        integrity_item = next(
            window.scene_list.item(index)
            for index in range(window.scene_list.count())
            if "can_integrity_repair" in window.scene_list.item(index).text()
        )
        window._load_scene_item(integrity_item)
        assert window.integrity_button.text() == "检查/修复数据完整性"
        assert window.integrity_button.isEnabled()
        original_question = QMessageBox.question
        original_information = QMessageBox.information
        notices: list[tuple[str, str]] = []
        QMessageBox.question = staticmethod(lambda *args, **kwargs: QMessageBox.Yes)  # type: ignore[method-assign]
        QMessageBox.information = staticmethod(  # type: ignore[method-assign]
            lambda _parent, title, text, *args, **kwargs: notices.append((title, text)) or QMessageBox.Ok
        )
        try:
            window.check_or_repair_integrity()
        finally:
            QMessageBox.question = original_question  # type: ignore[method-assign]
            QMessageBox.information = original_information  # type: ignore[method-assign]
        assert not (integrity_scene / "rgb/000001.png").exists()
        quarantined = list((integrity_scene / ".integrity_quarantine").glob("*/rgb/000001.png"))
        assert len(quarantined) == 1 and quarantined[0].read_bytes() == b"orphan"
        assert notices and "保留 1 对 RGB-D" in notices[-1][1]

        # Keep the remainder of this smoke test on the original can workflow.
        window._select_class(1)

        # Merely declaring a val key is not evidence that independent val data exists.
        dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
        dataset_yaml.write_text(
            "path: .\ntrain: images/train\nval: images/val\nnames: [can]\n",
            encoding="utf-8",
        )
        assert not window.has_independent_val()
        (dataset_yaml.parent / "images/train").mkdir(parents=True)
        (dataset_yaml.parent / "images/val").mkdir(parents=True)
        (dataset_yaml.parent / "images/train/train.png").write_bytes(b"train")
        (dataset_yaml.parent / "images/val/val.png").write_bytes(b"val")
        assert window.has_independent_val()
        window._refresh_state()
        assert not window.train_button.isEnabled(), "val existence alone must not bypass validation"
        window.validation_passed = True
        window._refresh_state()
        assert window.train_button.isEnabled()
        window.validation_passed = False

        # A failed launch with no frame data should not leave empty staging junk.
        failed_session = window.session
        (failed_session.staging_dir / "rgb").mkdir(parents=True)
        (failed_session.staging_dir / "depth").mkdir()
        window.task_kind = "capture"
        window._process_finished(1, QProcess.NormalExit)
        assert not failed_session.staging_dir.exists()
        assert "未就绪" in window.camera_status.text()
        assert not window.log.isHidden(), "errors should reveal the log automatically"

        # The single "开始标记" action automatically runs segmentation first.
        window._new_session_preview()
        assert window.session is not None
        window.session.scene_dir.mkdir(parents=True)
        (window.session.scene_dir / "rgb").mkdir()
        (window.session.scene_dir / "depth").mkdir()
        (window.session.scene_dir / "rgb/000000.png").write_bytes(b"rgb")
        (window.session.scene_dir / "depth/000000.png").write_bytes(b"depth")
        started: list[tuple[str, str, list[str]]] = []
        window._start_task = lambda kind, program, args: started.append((kind, program, args))  # type: ignore[method-assign]
        window._refresh_state()
        assert window.mark_button.isEnabled(), "a saved scene must allow retrying missing Manifest initialization"
        original_warning = QMessageBox.warning
        QMessageBox.warning = lambda *args, **kwargs: QMessageBox.Ok  # type: ignore[method-assign]
        try:
            window.start_annotation()
        finally:
            QMessageBox.warning = original_warning  # type: ignore[method-assign]
        assert started and started[-1][0] == "init"
        assert window.pending_after_task == "mark"

        resumed: list[str] = []
        window.start_annotation = lambda: resumed.append("mark")  # type: ignore[method-assign]
        window.task_kind = "init"
        window.pending_after_task = "mark"
        window._process_finished(0, QProcess.NormalExit)
        app.processEvents()
        assert resumed == ["mark"], "successful init must continue into segmentation/marking"

        manifest = manifest_for_session(window.session)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("project: {}\n", encoding="utf-8")
        started.clear()
        window.start_annotation = AtecMainWindow.start_annotation.__get__(window, AtecMainWindow)  # type: ignore[method-assign]
        window.start_annotation()
        assert started and started[-1][0] == "segment"
        assert window.pending_after_task == "mark"

        # The daily page shows per-segment completion based on actual mask files.
        report = window._segments_report()
        report.parent.mkdir(parents=True, exist_ok=True)
        mask_path = project_root / "data/key_masks/test/can_01/000000.png"
        report.write_text(
            "{\n"
            '  "segments": [{"segment_id": 0, "start_id": "000000", "end_id": "000114", '
            f'"required_key_mask_paths": {{"can_01": "{mask_path}"}}, "missing_key_masks": ["can_01"]}}]\n'
            "}",
            encoding="utf-8",
        )
        window._refresh_state()
        assert "0/1" in window.mask_progress_summary.text()
        assert "未完成" in window.mask_progress_list.item(0).text()
        mask_path.parent.mkdir(parents=True)
        mask_path.write_bytes(b"mask")
        window._refresh_state()
        assert "1/1" in window.mask_progress_summary.text()
        assert "已完成" in window.mask_progress_list.item(0).text()
        window._start_task = AtecMainWindow._start_task.__get__(window, AtecMainWindow)  # type: ignore[method-assign]

        # Continue is a thin GUI action: it starts one CLI process and makes
        # progress visible immediately instead of appearing to do nothing.
        original_command_for_run = window.command_for_run
        window.command_for_run = lambda: ("/bin/sh", ["-c", "sleep 0.2"])  # type: ignore[method-assign]
        window.continue_button.click()
        assert not window.log.isHidden()
        assert any(state in window.process_status.text() for state in ("正在启动", "运行中"))
        assert "SAM2" in window.guidance_label.text() and "YOLO" in window.guidance_label.text()
        assert window.process.waitForStarted(1000)
        app.processEvents()
        assert "运行中" in window.process_status.text()
        assert window.process.waitForFinished(2000)
        app.processEvents()
        window.command_for_run = original_command_for_run  # type: ignore[method-assign]

        # A failed executable must surface the concrete startup error in the
        # daily UI, not only append an opaque enum to a hidden log.
        original_critical = QMessageBox.critical
        startup_errors: list[tuple[str, str]] = []
        QMessageBox.critical = staticmethod(  # type: ignore[method-assign]
            lambda _parent, title, text, *args, **kwargs: startup_errors.append((title, text)) or QMessageBox.Ok
        )
        try:
            window._start_task("run", "/definitely/missing/atec-pipeline", [])
            deadline = time.monotonic() + 1.0
            while not startup_errors and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
        finally:
            QMessageBox.critical = original_critical  # type: ignore[method-assign]
        assert startup_errors and "启动" in startup_errors[-1][0]
        assert "/definitely/missing/atec-pipeline" in startup_errors[-1][1]
        assert "启动失败" in window.process_status.text()

        export_report = dataset_yaml.parent / "project_reports" / f"{window.session.scene_name}_{window.session.split}_report.json"
        export_report.parent.mkdir(parents=True, exist_ok=True)
        export_report.write_text(
            '{"frame_status_counts": {"accepted": 90, "review": 20, "rejected": 5}, "frames": [], '
            '"instances": [{"instance_id": "can_01", "class_id": 0, "class_name": "can", '
            f'"tracker": "sam2", "stage": "{dataset_yaml.parent / "_staging" / window.session.scene_name / "can_01"}"}}]}}',
            encoding="utf-8",
        )
        stage = dataset_yaml.parent / "_staging" / window.session.scene_name / "can_01"
        quality_report = stage / "quality_reports/train/class_000_can_01/quality_report.json"
        quality_report.parent.mkdir(parents=True)
        quality_report.write_text(
            '{"scene": "scene", "instance_id": "can_01", "records": '
            '[{"id": "000000", "status": "accepted", "reject_reasons": [], "review_reasons": []}]}',
            encoding="utf-8",
        )
        (stage / "_sam2_masks").mkdir(parents=True)
        (stage / "_sam2_masks/000000.png").write_bytes(b"mask")
        manifest.write_text(
            "project:\n"
            f"  scene: {window.session.scene_dir}\n"
            f"  output: {dataset_yaml.parent}\n"
            "  sam2_python: ~/miniforge3/envs/yolo11/bin/python\n"
            "  split: train\n"
            "instances:\n"
            "- instance_id: can_01\n"
            "  class_id: 0\n"
            "  class_name: can\n"
            "  tracker: sam2\n"
            f"  key_mask_dir: {mask_path.parent}\n",
            encoding="utf-8",
        )
        window._refresh_state()
        assert "accepted 90" in window.propagation_status_label.text()
        assert "待检查 25" in window.propagation_status_label.text()
        assert window.review_button.text() == "检查当前场次标注效果"
        assert window.review_button.isEnabled()
        assert window.mark_review_complete_button.text() == "标记当前场次 Review 完成"
        assert window.mark_review_complete_button.isEnabled()
        assert not window.clear_review_complete_button.isEnabled()

        marker = window.session.scene_dir / "project_reports/manual_review_complete.json"
        original_question = QMessageBox.question
        original_information = QMessageBox.information
        review_notices: list[tuple[str, str]] = []
        QMessageBox.question = staticmethod(lambda *args, **kwargs: QMessageBox.Yes)  # type: ignore[method-assign]
        QMessageBox.information = staticmethod(  # type: ignore[method-assign]
            lambda _parent, title, text, *args, **kwargs: review_notices.append((title, text)) or QMessageBox.Ok
        )
        try:
            window.mark_current_scene_review_complete()
        finally:
            QMessageBox.question = original_question  # type: ignore[method-assign]
            QMessageBox.information = original_information  # type: ignore[method-assign]
        assert marker.is_file()
        reviewed_texts = scene_list_texts()
        assert "── 已经过人工 Review（1）──" in reviewed_texts
        assert any(window.session.scene_name in text for text in reviewed_texts)
        assert not window.mark_review_complete_button.isEnabled()
        assert window.clear_review_complete_button.isEnabled()
        assert any("Review完成" in title.replace(" ", "") for title, _text in review_notices)

        original_question = QMessageBox.question
        QMessageBox.question = staticmethod(lambda *args, **kwargs: QMessageBox.Yes)  # type: ignore[method-assign]
        try:
            window.clear_current_scene_review_complete()
        finally:
            QMessageBox.question = original_question  # type: ignore[method-assign]
        assert not marker.exists()
        unreviewed_texts = scene_list_texts()
        assert "── 已经过人工 Review（0）──" in unreviewed_texts
        assert window.mark_review_complete_button.isEnabled()
        assert not window.clear_review_complete_button.isEnabled()

        contexts = window._review_contexts()
        assert len(contexts) == 1 and contexts[0]["instance_id"] == "can_01"
        review_program, review_args = window.command_for_review(contexts[0], ("0",))
        assert review_program.endswith("envs/yolo11/bin/python")
        assert not review_program.startswith("~"), "portable home paths must be expanded before QProcess launch"
        assert "review_mask_sequence.py" in review_args[0]
        assert str(quality_report) in review_args
        assert "--segments" in review_args and review_args[review_args.index("--segments") + 1] == "0"
        assert str(window.session.scene_dir / "project_reports/manual_mask_review/can_01.json") in review_args
        assert window._review_action_file is not None
        window._review_action_file.parent.mkdir(parents=True, exist_ok=True)
        window._review_action_file.write_text('{"action": "review_changed"}', encoding="utf-8")
        review_tasks: list[tuple[str, str, list[str]]] = []
        original_start_task = window._start_task
        window._start_task = lambda kind, program, args: review_tasks.append((kind, program, list(args)))  # type: ignore[method-assign]
        window.task_kind = "review"
        window._process_finished(0, QProcess.NormalExit)
        app.processEvents()
        assert len(review_tasks) == 1
        kind, program, args = review_tasks.pop()
        assert kind == "review_export"
        assert program.endswith("scripts/atec-pipeline")
        assert args[:2] == ["annotate", str(manifest)]
        assert "--skip-tracking" in args
        assert "run" not in args, "A/R/X changes must never rerun full SAM2"

        batch_action = {
            "action": "rerun_ranges",
            "instance_id": "can_01",
            "ranges": [
                {
                    "segment_id": "0",
                    "start_frame": "000000",
                    "end_before_frame": None,
                    "last_frame": "000000",
                    "boundary_reason": "segment_end",
                    "key_mask": str(mask_path),
                }
            ],
            "selected_segments": ["0"],
            "resume_frame": "000000",
        }
        window._review_action_file.write_text(json.dumps(batch_action), encoding="utf-8")
        window.task_kind = "review"
        window._process_finished(0, QProcess.NormalExit)
        app.processEvents()
        assert len(review_tasks) == 1
        kind, program, args = review_tasks.pop()
        assert kind == "review_rerun"
        assert program.endswith("scripts/atec-pipeline")
        assert args[:2] == ["rerun-range", str(manifest)]
        assert args[args.index("--instance-id") + 1] == "can_01"
        assert args[args.index("--ranges-file") + 1] == str(window._review_action_file)
        assert "run" not in args, "closing Review must start one local batch, not the full pipeline"
        window._start_task = original_start_task  # type: ignore[method-assign]

        # Training result metrics are visible without reading the terminal log.
        run_dir = ROOT / "runs/segment" / window.experiment.text()
        # Use an isolated experiment name rooted in the temporary project via method override.
        isolated_runs = project_root / "runs/segment"
        window._training_runs_root = lambda: isolated_runs  # type: ignore[method-assign]
        run_dir = isolated_runs / window.experiment.text()
        (run_dir / "weights").mkdir(parents=True)
        (run_dir / "weights/best.pt").write_bytes(b"best")
        (run_dir / "results.csv").write_text(
            "epoch,metrics/precision(M),metrics/recall(M),metrics/mAP50(M),metrics/mAP50-95(M)\n"
            "0,0.80,0.70,0.75,0.60\n",
            encoding="utf-8",
        )
        window.refresh_training_result()
        assert "Mask mAP50-95: 0.600" in window.training_result_label.text()
        assert window.open_training_result_button.isEnabled()
        assert window.live_start_button.isEnabled()
        assert str(run_dir / "weights/best.pt") in window.live_model_label.text()
        assert window.live_source_kind.currentData() == "opencv"
        assert window.live_source.isEnabled(), "external webcam mode must expose its source index"
        assert window.live_source.text() == "0"
        assert "外接摄像头" in window.live_start_button.text()
        live_program, live_args = window.command_for_live()
        assert live_program.endswith("envs/yolo11/bin/python")
        assert "live_yolo11_seg.py" in live_args[0]
        assert str(run_dir / "weights/best.pt") in live_args
        assert "--source" in live_args and live_args[live_args.index("--source") + 1] == "0"
        assert "--conf" in live_args and live_args[live_args.index("--conf") + 1] == "0.25"
        assert "yellow_can" not in " ".join(live_args), "live inference must be generic, not hard-coded to yellow can"

        orbbec_index = window.live_source_kind.findData("orbbec")
        assert orbbec_index >= 0
        window.live_source_kind.setCurrentIndex(orbbec_index)
        assert not window.live_source.isEnabled(), "Orbbec SDK mode must not use a V4L2 source index"
        assert "RGB-D" in window.live_source_kind.currentText()
        assert "Orbbec RGB-D" in window.live_start_button.text()
        orbbec_program, orbbec_args = window.command_for_live()
        assert orbbec_program == live_program
        assert "live_yolo11_seg_orbbec.py" in orbbec_args[0]

        # Dataset mutation must remain behind the CLI: GUI previews read-only,
        # then launches ``atec-pipeline split ... --apply`` as a subprocess.
        import atec_pipeline.gui_app as gui_module

        original_build_auto_plan = gui_module.build_auto_plan
        original_question = QMessageBox.question
        direct_apply_called: list[bool] = []
        original_apply_plan = getattr(gui_module, "apply_plan", None)
        if original_apply_plan is not None:
            gui_module.apply_plan = lambda _plan: direct_apply_called.append(True)  # type: ignore[attr-defined]
        auto_started: list[tuple[str, str, list[str]]] = []
        original_all_scene_states = window._all_scene_states
        original_start_task = window._start_task
        try:
            gui_module.build_auto_plan = lambda *_args, **_kwargs: {
                "groups": [SimpleNamespace(scene_names={"can_scene_02"})],
                "files": [(Path("a.png"), Path("b.png"), Path("a.txt"), Path("b.txt"))],
            }
            QMessageBox.question = staticmethod(lambda *args, **kwargs: QMessageBox.Yes)  # type: ignore[method-assign]
            window._all_scene_states = lambda: [
                SimpleNamespace(training_eligible=True, split="train", scene_name="can_scene_01", accepted=3, class_name="can"),
                SimpleNamespace(training_eligible=True, split="train", scene_name="can_scene_02", accepted=1, class_name="can"),
            ]
            window._start_task = lambda kind, program, args: auto_started.append((kind, program, list(args)))  # type: ignore[method-assign]
            window._preview_training_split()
        finally:
            gui_module.build_auto_plan = original_build_auto_plan
            QMessageBox.question = original_question  # type: ignore[method-assign]
            window._all_scene_states = original_all_scene_states  # type: ignore[method-assign]
            window._start_task = original_start_task  # type: ignore[method-assign]
            if original_apply_plan is not None:
                gui_module.apply_plan = original_apply_plan  # type: ignore[attr-defined]
        assert not direct_apply_called, "GUI must not apply dataset mutations in-process"
        assert auto_started and auto_started[-1][0] == "prepare_split"
        assert auto_started[-1][2] == [
            "split", str(dataset_yaml.resolve()), "--val-scenes", "can_scene_02", "--apply"
        ]

        window.close()
    app.processEvents()
    print("GUI_APP_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
