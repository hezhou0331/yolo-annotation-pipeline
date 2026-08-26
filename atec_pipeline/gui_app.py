#!/usr/bin/env python3
"""Thin local PyQt5 shell for ATEC RGB-D capture and annotation commands."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys

from PyQt5.QtCore import QProcess, Qt, QTimer, QUrl
from PyQt5.QtGui import QColor, QCloseEvent, QDesktopServices, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
import yaml

from .gui_state import (
    CaptureSession,
    ObjectClass,
    discard_failed_empty_capture,
    find_best_weights,
    has_paired_rgbd_frames,
    inspect_scene_integrity,
    load_export_summary,
    load_mask_progress,
    load_object_classes,
    load_scene_human_review_completion,
    load_training_summary,
    mark_scene_human_review_complete,
    clear_scene_human_review_complete,
    make_capture_session,
    manifest_for_session,
    paired_rgbd_frame_count,
    repair_scene_integrity,
    scene_workflow_state,
    summarize_scene_states,
    write_session_metadata,
)
from tools.split_dataset_by_scene import apply_plan, build_auto_plan

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "projects" / "atec_real"
CONFIG_PATH = ROOT / "configs" / "atec_objects.yaml"


class AtecMainWindow(QMainWindow):
    """A deliberately thin GUI around the repository's existing CLI tools."""

    def __init__(self, project_root: Path = PROJECT_ROOT, config_path: Path = CONFIG_PATH):
        super().__init__()
        self.project_root = Path(project_root).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.classes = load_object_classes(self.config_path)
        self.selected_class: ObjectClass | None = None
        self.session: CaptureSession | None = None
        self.task_kind = ""
        self.pending_after_task: str | None = None
        self.validation_passed = False
        self._auto_prepare_queue: list[tuple[str, Path]] = []
        self._auto_prepare_failures: list[str] = []
        self._review_action_file: Path | None = None

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.started.connect(self._process_started)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)

        self.live_process = QProcess(self)
        self.live_process.setWorkingDirectory(str(ROOT))
        self.live_process.readyReadStandardOutput.connect(self._read_live_stdout)
        self.live_process.readyReadStandardError.connect(self._read_live_stderr)
        self.live_process.started.connect(lambda: self._append_log("[实时识别] 已启动"))
        self.live_process.finished.connect(self._live_process_finished)
        self.live_process.errorOccurred.connect(lambda error: self._append_log(f"[实时识别进程错误] {error}"))
        self._live_model_path: Path | None = None

        self.frame_timer = QTimer(self)
        self.frame_timer.setInterval(500)
        self.frame_timer.timeout.connect(self._refresh_capture_metrics)

        self.setWindowTitle("ATEC RGB-D 采集与标注")
        self.resize(1000, 900)
        self._build_ui()
        self._select_class(1)
        self._refresh_state()
        self.frame_timer.start()

    # ---------- command builders: algorithms remain in existing CLI ----------
    def command_for_capture(self) -> tuple[str, list[str]]:
        if not self.session:
            raise RuntimeError("请先选择类别并生成场次")
        return "setsid", [
            str(ROOT / "scripts" / "capture_orbbec.sh"),
            str(self.session.staging_dir),
            "--auto",
            "--interval",
            "0.1",
        ]

    def command_for_init(self) -> tuple[str, list[str]]:
        self._require_session()
        return str(ROOT / "scripts" / "atec-pipeline"), [
            "init",
            str(self.project_root),
            self.session.scene_name,
            "--split",
            self.session.split,
            "--only-class",
            self.session.object_class.name,
            "--scene-class",
            self.session.object_class.name,
            "--capture-session-id",
            self.session.capture_session_id,
            "--source-video-id",
            self.session.source_video_id,
        ]

    def command_for_segment(self) -> tuple[str, list[str]]:
        manifest = self._require_manifest()
        return str(ROOT / "scripts" / "atec-pipeline"), [
            "segment",
            str(manifest),
            "--output",
            str(self._segments_report()),
        ]

    def command_for_mask(self, image: Path, output: Path) -> tuple[str, list[str]]:
        return str(ROOT / "scripts" / "atec-pipeline"), ["mask", str(image), str(output)]

    def command_for_run(self) -> tuple[str, list[str]]:
        return str(ROOT / "scripts" / "atec-pipeline"), ["run", str(self._require_manifest())]

    def command_for_validate(self) -> tuple[str, list[str]]:
        return str(ROOT / "scripts" / "atec-pipeline"), ["validate", str(self._dataset_path())]

    def command_for_train(self) -> tuple[str, list[str]]:
        return str(ROOT / "scripts" / "atec-pipeline"), [
            "train",
            str(self._dataset_path()),
            "--init-mode",
            self.init_mode.currentText(),
            "--epochs",
            str(self.epochs.value()),
            "--batch",
            str(self.batch.value()),
            "--device",
            self.device.text().strip() or "0",
            "--name",
            self.experiment.text().strip() or "atec_gui_run",
        ]

    def command_for_review(self, context: dict, segment_ids: tuple[str, ...] = ()) -> tuple[str, list[str]]:
        action_file = self._require_session().scene_dir / "project_reports" / "manual_mask_review" / "player_action.json"
        self._review_action_file = action_file
        python = str(Path(context.get("python") or os.environ.get("ATEC_YOLO_PYTHON", "~/miniforge3/envs/yolo11/bin/python")).expanduser())
        args = [
            str(ROOT / "tools" / "review_mask_sequence.py"),
            "--scene", str(context["scene"]),
            "--mask-dir", str(context["mask_dir"]),
            "--quality-report", str(context["quality_report"]),
            "--segments-report", str(context["segments_report"]),
            "--review-overrides", str(context["review_overrides"]),
            "--instance-id", str(context["instance_id"]),
            "--key-mask-dir", str(context["key_mask_dir"]),
            "--action-file", str(action_file),
            "--editor-python", python,
        ]
        if segment_ids:
            args.extend(["--segments", ",".join(segment_ids)])
        return python, args

    def command_for_review_export(self) -> tuple[str, list[str]]:
        return str(ROOT / "scripts" / "atec-pipeline"), [
            "annotate", str(self._require_manifest()), "--skip-tracking",
        ]

    def command_for_review_rerun(self, action: dict) -> tuple[str, list[str]]:
        instance_id = str(action.get("instance_id") or "").strip()
        ranges = action.get("ranges")
        if not instance_id:
            raise ValueError("Review动作缺少instance_id")
        if action.get("action") == "rerun_ranges" and (not isinstance(ranges, list) or not ranges):
            raise ValueError("Review动作没有待处理的局部范围")
        if self._review_action_file is None or not self._review_action_file.is_file():
            raise FileNotFoundError("Review局部传播动作文件不存在")
        return str(ROOT / "scripts" / "atec-pipeline"), [
            "rerun-range", str(self._require_manifest()),
            "--instance-id", instance_id,
            "--ranges-file", str(self._review_action_file),
        ]

    # ---------- one-page UI ----------
    def _build_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        title = QLabel("ATEC RGB-D 数据采集与标注")
        title.setStyleSheet("font-size:22px; font-weight:600;")
        subtitle = QLabel("日常只需：选择类别 → 开始/停止采集 → 保存或丢弃 → 开始标记")
        subtitle.setStyleSheet("color:#555;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        layout.addWidget(self._build_session_group())
        layout.addWidget(self._build_capture_group())
        layout.addWidget(self._build_annotation_group())
        layout.addWidget(self._build_scene_group())

        self.advanced_toggle = QPushButton("显示高级处理与训练")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.advanced_toggle)
        self.advanced_panel = self._build_advanced_panel()
        self.advanced_panel.hide()
        layout.addWidget(self.advanced_panel)

        self.log_toggle = QPushButton("显示运行日志")
        self.log_toggle.setCheckable(True)
        self.log_toggle.toggled.connect(self._toggle_log)
        layout.addWidget(self.log_toggle)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("logOutput")
        self.log.setMinimumHeight(180)
        self.log.hide()
        layout.addWidget(self.log)
        layout.addStretch(1)

        scroll.setWidget(page)
        self.setCentralWidget(scroll)

    def _build_session_group(self) -> QGroupBox:
        group = QGroupBox("1. 选择物品")
        layout = QVBoxLayout(group)
        grid = QGridLayout()
        self.class_buttons: list[QPushButton] = []
        for index, obj in enumerate(self.classes, start=1):
            button = QPushButton(f"{index}  {obj.chinese_name} ({obj.name})")
            button.setObjectName(f"classButton{index}")
            button.setShortcut(str(index))
            button.clicked.connect(lambda _checked=False, n=index: self._select_class(n))
            self.class_buttons.append(button)
            grid.addWidget(button, (index - 1) // 4, (index - 1) % 4)
        layout.addLayout(grid)

        form = QFormLayout()
        self.split_combo = QComboBox()
        self.split_combo.addItems(["train", "val", "test"])
        self.split_combo.currentTextChanged.connect(self._new_session_preview)
        self.remark_edit = QLineEdit()
        self.remark_edit.setPlaceholderText("可选：光照、摆放、动作等备注")
        self.session_label = QLabel("尚未生成场次")
        self.session_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("数据划分", self.split_combo)
        form.addRow("备注", self.remark_edit)
        form.addRow("当前场次", self.session_label)
        layout.addLayout(form)
        return group

    def _build_capture_group(self) -> QGroupBox:
        group = QGroupBox("2. 采集 RGB-D")
        layout = QGridLayout(group)
        self.camera_status = QLabel("相机未启动")
        self.process_status = QLabel("空闲")
        self.frame_count = QLabel("0 对")
        self.start_button = QPushButton("开始采集")
        self.stop_button = QPushButton("停止采集（Ctrl+C）")
        self.stop_capture_shortcut = QShortcut(QKeySequence("Ctrl+C"), self)
        self.stop_capture_shortcut.setContext(Qt.ApplicationShortcut)
        self.stop_capture_shortcut.activated.connect(self.stop_capture)
        self.save_button = QPushButton("保存本次")
        self.discard_button = QPushButton("丢弃本次")
        self.start_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)
        self.save_button.clicked.connect(lambda: self.finalize_capture("save"))
        self.discard_button.clicked.connect(lambda: self.finalize_capture("discard"))
        layout.addWidget(QLabel("相机"), 0, 0)
        layout.addWidget(self.camera_status, 0, 1, 1, 3)
        layout.addWidget(QLabel("任务"), 1, 0)
        layout.addWidget(self.process_status, 1, 1)
        layout.addWidget(QLabel("RGB/Depth 配对帧"), 1, 2)
        layout.addWidget(self.frame_count, 1, 3)
        layout.addWidget(self.start_button, 2, 0)
        layout.addWidget(self.stop_button, 2, 1)
        layout.addWidget(self.save_button, 2, 2)
        layout.addWidget(self.discard_button, 2, 3)
        note = QLabel("采集时会自动打开 Orbbec RGB-D Capture 预览；停止后再选择保存或丢弃。")
        note.setWordWrap(True)
        layout.addWidget(note, 3, 0, 1, 4)
        return group

    def _build_annotation_group(self) -> QGroupBox:
        group = QGroupBox("3. 标记与继续处理")
        layout = QVBoxLayout(group)
        self.guidance_label = QLabel("完成一次采集并保存后，即可开始标记。")
        self.guidance_label.setWordWrap(True)
        layout.addWidget(self.guidance_label)

        self.mask_quick_help = QLabel(
            "标注器：左键依次点轮廓 → Enter 应用为绿色区域 → S 保存 → Q/ESC 退出。"
            "不要使用窗口顶部工具栏的软盘图标。"
        )
        self.mask_quick_help.setWordWrap(True)
        self.mask_quick_help.setStyleSheet("background:#fff4cc; padding:8px; border:1px solid #e0c56e;")
        layout.addWidget(self.mask_quick_help)

        self.mask_progress_summary = QLabel("关键帧进度：尚未分段")
        self.mask_progress_summary.setStyleSheet("font-weight:600;")
        layout.addWidget(self.mask_progress_summary)
        self.mask_progress_list = QListWidget()
        self.mask_progress_list.setMaximumHeight(110)
        self.mask_progress_list.addItem("保存场次并自动分段后，这里会显示每个分段是否完成。")
        layout.addWidget(self.mask_progress_list)
        self.propagation_status_label = QLabel("后续视频帧：SAM2 传播与 YOLO 导出尚未运行。")
        self.propagation_status_label.setWordWrap(True)
        layout.addWidget(self.propagation_status_label)

        row = QHBoxLayout()
        self.mark_button = QPushButton("开始标记 / 标下一个")
        self.continue_button = QPushButton("继续自动处理")
        self.mask_help_button = QPushButton("标注器使用说明")
        self.mark_button.clicked.connect(self.start_annotation)
        self.continue_button.clicked.connect(self.continue_processing)
        self.mask_help_button.clicked.connect(self.show_mask_help)
        row.addWidget(self.mark_button)
        row.addWidget(self.continue_button)
        row.addWidget(self.mask_help_button)
        layout.addLayout(row)
        return group

    def _build_scene_group(self) -> QGroupBox:
        group = QGroupBox("已保存场次（单击可继续标记）")
        self.saved_scenes_group = group
        layout = QVBoxLayout(group)
        self.class_frame_summary = QLabel("当前物品数据概览：请先选择物品。")
        self.class_frame_summary.setObjectName("classFrameSummary")
        self.class_frame_summary.setWordWrap(True)
        self.class_frame_summary.setStyleSheet(
            "background:#eef6ff; padding:8px; border:1px solid #9ec5e8; font-weight:600;"
        )
        layout.addWidget(self.class_frame_summary)
        self.scene_list = QListWidget()
        self.scene_list.setObjectName("sceneList")
        self.scene_list.setMinimumHeight(180)
        self.scene_list.setMaximumHeight(260)
        self.scene_list.itemClicked.connect(self._load_scene_item)
        layout.addWidget(self.scene_list)
        self.integrity_button = QPushButton("检查/修复数据完整性")
        self.integrity_button.setToolTip(
            "检查 RGB/Depth 是否逐帧配对；仅将未写入 metadata 的孤立末帧移动到可恢复隔离目录"
        )
        self.integrity_button.clicked.connect(self.check_or_repair_integrity)
        layout.addWidget(self.integrity_button)
        self.review_button = QPushButton("检查当前场次标注效果")
        self.review_button.setToolTip("按原始帧顺序播放 RGB + 半透明 Mask + 轮廓，并人工修改逐帧状态")
        self.review_button.clicked.connect(self.start_review)
        layout.addWidget(self.review_button)
        review_state_row = QHBoxLayout()
        self.mark_review_complete_button = QPushButton("标记当前场次 Review 完成")
        self.mark_review_complete_button.setToolTip(
            "仅在完整检查当前采集场次后点击；不会移动或修改RGB-D、Mask和YOLO数据"
        )
        self.clear_review_complete_button = QPushButton("取消 Review 完成标记")
        self.clear_review_complete_button.setToolTip("只删除人工Review完成元数据，不修改标注和训练数据")
        self.mark_review_complete_button.clicked.connect(self.mark_current_scene_review_complete)
        self.clear_review_complete_button.clicked.connect(self.clear_current_scene_review_complete)
        review_state_row.addWidget(self.mark_review_complete_button)
        review_state_row.addWidget(self.clear_review_complete_button)
        layout.addLayout(review_state_row)
        return group

    def _build_advanced_panel(self) -> QGroupBox:
        panel = QGroupBox("高级处理与训练（仍调用现有 atec-pipeline 命令）")
        layout = QVBoxLayout(panel)
        self.workflow_status = QListWidget()
        self.workflow_status.setMaximumHeight(105)
        layout.addWidget(self.workflow_status)

        commands = QHBoxLayout()
        self.auto_prepare_button = QPushButton("自动准备训练数据")
        self.segment_button = QPushButton("仅运行自动分段")
        self.run_button = QPushButton("仅运行 SAM2/YOLO 导出")
        self.validate_button = QPushButton("验证数据集")
        self.auto_prepare_button.clicked.connect(self.start_auto_prepare)
        self.segment_button.clicked.connect(self.run_segment)
        self.run_button.clicked.connect(self.run_pipeline)
        self.validate_button.clicked.connect(self.run_validate)
        commands.addWidget(self.auto_prepare_button)
        commands.addWidget(self.segment_button)
        commands.addWidget(self.run_button)
        commands.addWidget(self.validate_button)
        layout.addLayout(commands)

        form = QFormLayout()
        self.dataset_label = QLabel(str(self._dataset_path()))
        self.init_mode = QComboBox()
        self.init_mode.addItems(["baseline", "xcx-transfer"])
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(100)
        self.batch = QSpinBox()
        self.batch.setRange(1, 512)
        self.batch.setValue(4)
        self.device = QLineEdit("0")
        self.experiment = QLineEdit("atec_4class_baseline_moredata_20260824")
        form.addRow("数据集", self.dataset_label)
        form.addRow("初始化", self.init_mode)
        form.addRow("epochs", self.epochs)
        form.addRow("batch", self.batch)
        form.addRow("device", self.device)
        form.addRow("实验名", self.experiment)
        layout.addLayout(form)
        self.train_status = QLabel("没有独立 val 时，正式训练按钮保持禁用。")
        self.train_status.setWordWrap(True)
        layout.addWidget(self.train_status)
        row = QHBoxLayout()
        self.train_button = QPushButton("开始训练")
        self.stop_train_button = QPushButton("停止训练")
        self.train_button.clicked.connect(self.run_train)
        self.stop_train_button.clicked.connect(self.stop_task)
        row.addWidget(self.train_button)
        row.addWidget(self.stop_train_button)
        layout.addLayout(row)

        self.training_result_label = QLabel("训练效果：尚未发现训练结果。训练后重点看验证集 Mask mAP50-95。")
        self.training_result_label.setWordWrap(True)
        layout.addWidget(self.training_result_label)
        result_row = QHBoxLayout()
        self.refresh_training_result_button = QPushButton("刷新训练效果")
        self.open_training_result_button = QPushButton("打开训练结果目录")
        self.refresh_training_result_button.clicked.connect(self.refresh_training_result)
        self.open_training_result_button.clicked.connect(self.open_training_result)
        result_row.addWidget(self.refresh_training_result_button)
        result_row.addWidget(self.open_training_result_button)
        layout.addLayout(result_row)

        live_form = QFormLayout()
        self.live_model_label = QLabel("尚未发现真实训练 best.pt")
        self.live_model_label.setWordWrap(True)
        self.live_source_kind = QComboBox()
        self.live_source_kind.addItem("外接 HD Webcam（/dev/video0）", "opencv")
        self.live_source_kind.addItem("Orbbec SDK RGB（与采集相机一致）", "orbbec")
        self.live_source_kind.currentIndexChanged.connect(self._refresh_live_source_controls)
        self.live_source = QLineEdit("0")
        self.live_source.setPlaceholderText("普通摄像头编号或视频文件路径")
        self.live_conf = QDoubleSpinBox()
        self.live_conf.setRange(0.01, 1.0)
        self.live_conf.setSingleStep(0.05)
        self.live_conf.setValue(0.25)
        live_form.addRow("实时测试模型", self.live_model_label)
        live_form.addRow("实时数据源", self.live_source_kind)
        live_form.addRow("普通摄像头/视频源", self.live_source)
        live_form.addRow("置信度", self.live_conf)
        self._refresh_live_source_controls()
        layout.addLayout(live_form)
        live_row = QHBoxLayout()
        self.live_start_button = QPushButton("启动外接摄像头实时识别")
        self.live_stop_button = QPushButton("停止实时识别")
        self.live_start_button.clicked.connect(self.start_live_recognition)
        self.live_stop_button.clicked.connect(self.stop_live_recognition)
        live_row.addWidget(self.live_start_button)
        live_row.addWidget(self.live_stop_button)
        layout.addLayout(live_row)
        self._refresh_live_source_controls()
        return panel

    def show_mask_help(self) -> None:
        QMessageBox.information(
            self,
            "关键帧 Mask 标注说明",
            "推荐使用多边形模式：\n\n"
            "1. 用鼠标左键沿物体外轮廓依次点选，至少 3 个点。\n"
            "2. 按 Enter，把轮廓真正应用成绿色 Mask。\n"
            "3. 边缘不准时按 B 切换画笔；右键拖动可临时擦除。\n"
            "4. 确认绿色区域只覆盖目标后，按 S 保存。\n"
            "5. 看到日志“已保存”后，按 Q 或 Esc 退出编辑器。\n"
            "6. 回到 App 查看每个分段的“已完成/未完成”；有未完成时点击“开始标记 / 标下一个”。\n\n"
            "重要：顶部工具栏是 OpenCV 图片工具，不负责保存 Mask；Mask 必须用键盘 S 保存。"
        )

    def _training_runs_root(self) -> Path:
        return ROOT / "runs" / "segment"

    def refresh_training_result(self) -> None:
        runs_root = self._training_runs_root()
        experiment = self.experiment.text().strip()
        summary = load_training_summary(runs_root, experiment)
        best_weights = find_best_weights(runs_root, experiment)
        self._live_model_path = best_weights
        self.live_model_label.setText(str(best_weights) if best_weights else "尚未发现真实训练 best.pt")
        self._last_training_run_dir = summary.run_dir if summary else (best_weights.parents[1] if best_weights else None)
        if summary is None:
            if best_weights:
                self.training_result_label.setText(f"发现真实训练权重：{best_weights}；results.csv 不存在，暂无验证指标摘要。")
            else:
                self.training_result_label.setText(
                    "训练效果：尚未发现训练结果。先准备独立 val，再开始训练；训练后重点看验证集 Mask mAP50-95。"
                )
            self.open_training_result_button.setEnabled(bool(self._last_training_run_dir))
        else:
            def shown(value: float | None) -> str:
                return "—" if value is None else f"{value:.3f}"
            weights = str(best_weights) if best_weights else "best.pt 尚未生成"
            self.training_result_label.setText(
                f"最佳 epoch {summary.best_epoch} | Mask mAP50-95: {shown(summary.mask_map50_95)} | "
                f"mAP50: {shown(summary.mask_map50)} | Precision: {shown(summary.mask_precision)} | "
                f"Recall: {shown(summary.mask_recall)}\n最佳权重：{weights}"
            )
            self.open_training_result_button.setEnabled(summary.run_dir.is_dir())
        if hasattr(self, "live_start_button"):
            idle = self.process.state() == QProcess.NotRunning and self.live_process.state() == QProcess.NotRunning
            self.live_start_button.setEnabled(bool(best_weights) and idle)

    def open_training_result(self) -> None:
        run_dir = getattr(self, "_last_training_run_dir", None)
        if not run_dir or not Path(run_dir).is_dir():
            QMessageBox.information(self, "训练结果", "尚未发现可打开的训练结果目录。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(run_dir)))

    def _refresh_live_source_controls(self, _index: int | None = None) -> None:
        if not hasattr(self, "live_source_kind"):
            return
        is_external_webcam = self.live_source_kind.currentData() == "opencv"
        self.live_source.setEnabled(is_external_webcam)
        if hasattr(self, "live_start_button"):
            self.live_start_button.setText(
                "启动外接摄像头实时识别" if is_external_webcam else "启动 Orbbec 实时识别"
            )

    def command_for_live(self) -> tuple[str, list[str]]:
        weights = self._live_model_path or find_best_weights(
            self._training_runs_root(), self.experiment.text().strip()
        )
        if weights is None or not weights.is_file():
            raise FileNotFoundError("尚未发现真实训练生成的 best.pt；不会使用官方基础权重冒充。")
        python = Path(os.environ.get("ATEC_YOLO_PYTHON", "~/miniforge3/envs/yolo11/bin/python")).expanduser()
        if not python.is_file():
            raise FileNotFoundError(f"YOLO11 Python环境不存在: {python}")

        common_args = [
            "--model", str(weights),
            "--conf", f"{self.live_conf.value():.2f}",
            "--imgsz", "640",
            "--device", self.device.text().strip() or "0",
        ]
        if self.live_source_kind.currentData() == "orbbec":
            orbbec_python = Path(os.environ.get("ATEC_ORBBEC_PYTHON", "~/miniforge3/envs/orbbec/bin/python")).expanduser()
            if not orbbec_python.is_file():
                raise FileNotFoundError(f"Orbbec Python环境不存在: {orbbec_python}")
            return str(python), [
                str(ROOT / "tools" / "live_yolo11_seg_orbbec.py"),
                *common_args,
                "--orbbec-python", str(orbbec_python),
            ]
        return str(python), [
            str(ROOT / "tools" / "live_yolo11_seg.py"),
            *common_args,
            "--source", self.live_source.text().strip() or "0",
        ]

    def start_live_recognition(self) -> None:
        if self.process.state() != QProcess.NotRunning or self.live_process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "资源占用", "请先停止采集、训练、SAM2 或已有实时识别任务。")
            return
        try:
            program, args = self.command_for_live()
        except Exception as exc:
            QMessageBox.warning(self, "无法启动实时识别", str(exc))
            return
        self._append_log("$ " + " ".join([program, *args]))
        self.live_process.start(program, args)
        self._refresh_state()

    def stop_live_recognition(self) -> None:
        if self.live_process.state() != QProcess.NotRunning:
            self.live_process.terminate()
            QTimer.singleShot(5000, self._kill_live_if_needed)

    def _kill_live_if_needed(self) -> None:
        if self.live_process.state() != QProcess.NotRunning:
            self.live_process.kill()

    def _read_live_stdout(self) -> None:
        self._append_log(bytes(self.live_process.readAllStandardOutput()).decode(errors="replace"))

    def _read_live_stderr(self) -> None:
        self._append_log(bytes(self.live_process.readAllStandardError()).decode(errors="replace"))

    def _live_process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._append_log(f"[实时识别] 已结束 exit={exit_code}")
        if exit_code != 0:
            self._show_log()
        self._refresh_state()

    def _toggle_advanced(self, visible: bool) -> None:
        self.advanced_panel.setVisible(visible)
        self.advanced_toggle.setText("隐藏高级处理与训练" if visible else "显示高级处理与训练")

    def _toggle_log(self, visible: bool) -> None:
        self.log.setVisible(visible)
        self.log_toggle.setText("隐藏运行日志" if visible else "显示运行日志")

    def _show_log(self) -> None:
        if not self.log_toggle.isChecked():
            self.log_toggle.setChecked(True)

    # ---------- state ----------
    def _select_class(self, number: int) -> None:
        self.selected_class = self.classes[number - 1]
        for button, obj in zip(self.class_buttons, self.classes):
            button.setStyleSheet("background:#b8e0ff; font-weight:600;" if obj == self.selected_class else "")
        self._new_session_preview()

    def _new_session_preview(self) -> None:
        if self.selected_class is None:
            return
        self.session = make_capture_session(
            self.project_root,
            self.selected_class,
            self.split_combo.currentText(),
            remark=self.remark_edit.text().strip(),
        )
        self.session_label.setText(f"{self.session.scene_name}  |  {self.session.capture_session_id}")
        self._refresh_state()

    def _load_scene_item(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.UserRole)
        if not isinstance(data, dict):
            return
        class_name = str(data.get("class_name", ""))
        selected = next((obj for obj in self.classes if obj.name == class_name), None)
        if selected is None:
            return
        self.selected_class = selected
        for button, obj in zip(self.class_buttons, self.classes):
            button.setStyleSheet("background:#b8e0ff; font-weight:600;" if obj == selected else "")
        split = str(data.get("split", "train"))
        self.split_combo.blockSignals(True)
        self.split_combo.setCurrentText(split)
        self.split_combo.blockSignals(False)
        self.session = make_capture_session(
            self.project_root,
            self.selected_class,
            split,
            scene_name=str(data["scene_name"]),
        )
        metadata = self.session.scene_dir / "atec_capture_session.json"
        if metadata.exists():
            try:
                saved = json.loads(metadata.read_text(encoding="utf-8"))
                self.session.remark = str(saved.get("remark", ""))
                self.remark_edit.setText(self.session.remark)
            except (OSError, ValueError, TypeError):
                pass
        self.session_label.setText(f"{self.session.scene_name}  |  已保存 {split}")
        self._append_log(f"[场次] 已选择 {self.session.scene_dir}")
        self._refresh_state(refresh_scenes=False)

    def _require_session(self) -> CaptureSession:
        if self.session is None:
            raise RuntimeError("请先选择类别")
        return self.session

    def check_or_repair_integrity(self) -> None:
        """Inspect the selected formal scene and quarantine proven-safe orphans."""
        try:
            session = self._require_session()
            scene = session.scene_dir
            if not scene.is_dir():
                raise FileNotFoundError("请先在“已保存场次”中选择一个正式场次")
            report = inspect_scene_integrity(scene)
        except Exception as exc:
            QMessageBox.warning(self, "无法检查数据完整性", str(exc))
            return

        if report.is_complete:
            QMessageBox.information(self, "数据完整性检查通过", report.summary)
            self._append_log(f"[数据完整性] {session.scene_name}: {report.summary}")
            return

        if not report.can_auto_repair:
            details = list(report.unsafe_recorded_missing)
            if report.metadata_error:
                details.append(report.metadata_error)
            orphan_paths = [*report.orphan_rgb, *report.orphan_depth]
            if orphan_paths:
                details.append(
                    "未配对文件：" + "、".join(str(path.relative_to(scene)) for path in orphan_paths)
                )
            message = (
                "发现不能安全自动处理的数据不一致。App 不会移动或删除这些文件。\n\n"
                + "\n".join(f"- {item}" for item in details)
            )
            QMessageBox.warning(self, "需要人工检查数据完整性", message)
            self._append_log(f"[数据完整性/需人工] {session.scene_name}: {report.summary}")
            return

        relative_files = "\n".join(f"- {path.relative_to(scene)}" for path in report.safe_orphans)
        prompt = (
            f"发现 {len(report.safe_orphans)} 个未配对且未记录到 metadata 的文件。\n"
            f"保留 {report.paired_count} 对完整 RGB-D。\n\n"
            "这些文件将移动到场次内的可恢复隔离目录，不会永久删除：\n"
            f"{relative_files}\n\n是否执行安全修复？"
        )
        answer = QMessageBox.question(
            self,
            "安全修复数据完整性",
            prompt,
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            self._append_log(f"[数据完整性] 用户取消修复 {session.scene_name}")
            return
        try:
            quarantine = repair_scene_integrity(report)
            checked = inspect_scene_integrity(scene)
        except Exception as exc:
            QMessageBox.critical(self, "数据完整性修复失败", str(exc))
            self._append_log(f"[数据完整性/失败] {session.scene_name}: {exc}")
            return

        message = (
            f"数据完整性已修复：保留 {checked.paired_count} 对 RGB-D，"
            f"隔离 {len(report.safe_orphans)} 个孤立文件。\n\n"
            f"隔离目录：{quarantine}"
        )
        QMessageBox.information(self, "数据完整性已修复", message)
        self._append_log(f"[数据完整性/已修复] {session.scene_name}: {message}")
        self._refresh_state()

    def _require_manifest(self) -> Path:
        session = self._require_session()
        path = manifest_for_session(session)
        if not path.exists():
            raise FileNotFoundError(f"Manifest 不存在，请先保存采集：{path}")
        return path

    def _dataset_path(self) -> Path:
        return self.project_root / "datasets" / "atec_yolo11_seg" / "dataset.yaml"

    def _segments_report(self) -> Path:
        return self._require_session().scene_dir / "project_reports" / "segments.json"

    def _scene_export_report(self) -> Path:
        session = self._require_session()
        return self._dataset_path().parent / "project_reports" / f"{session.scene_name}_{session.split}_report.json"

    @staticmethod
    def _manifest_path(value: str | Path, base: Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (base / path).resolve()

    def _review_contexts(self) -> list[dict]:
        if not self.session:
            return []
        manifest_path = manifest_for_session(self.session)
        export_path = self._scene_export_report()
        segments_path = self._segments_report()
        if not (manifest_path.is_file() and export_path.is_file() and segments_path.is_file()):
            return []
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        export = json.loads(export_path.read_text(encoding="utf-8"))
        project = manifest.get("project") or {}
        manifest_instances = {
            str(item.get("instance_id")): item
            for item in (manifest.get("instances") or [])
            if item.get("instance_id")
        }
        scene = self._manifest_path(project.get("scene", self.session.scene_dir), manifest_path.parent)
        python = str(Path(project.get("sam2_python") or "~/miniforge3/envs/yolo11/bin/python").expanduser())
        split = str(export.get("split") or project.get("split") or self.session.split)
        contexts: list[dict] = []
        for item in export.get("instances") or []:
            instance_id = str(item.get("instance_id", ""))
            manifest_instance = manifest_instances.get(instance_id) or {}
            if str(item.get("tracker") or manifest_instance.get("tracker") or "") != "sam2":
                continue
            stage_raw = item.get("stage")
            key_raw = manifest_instance.get("key_mask_dir")
            if not stage_raw or not key_raw:
                continue
            stage = self._manifest_path(stage_raw, export_path.parent)
            class_id = int(item.get("class_id", manifest_instance.get("class_id", 0)))
            instance_dir = f"class_{class_id:03d}_{instance_id}"
            quality = stage / "quality_reports" / split / instance_dir / "quality_report.json"
            if not quality.is_file():
                candidates = sorted(stage.glob("quality_reports/*/*/quality_report.json"))
                quality = candidates[0] if candidates else quality
            mask_dir = stage / "_sam2_masks"
            if not (quality.is_file() and mask_dir.is_dir()):
                continue
            contexts.append({
                "scene": scene,
                "instance_id": instance_id,
                "class_name": str(item.get("class_name") or manifest_instance.get("class_name") or ""),
                "mask_dir": mask_dir,
                "quality_report": quality,
                "segments_report": segments_path,
                "review_overrides": scene / "project_reports" / "manual_mask_review" / f"{instance_id}.json",
                "key_mask_dir": self._manifest_path(key_raw, manifest_path.parent),
                "python": python,
            })
        return contexts

    def _review_segment_choices(self) -> list[tuple[str, tuple[str, ...]]]:
        data = json.loads(self._segments_report().read_text(encoding="utf-8"))
        segments = [
            (str(item.get("segment_id", index)), str(item.get("start_id", "?")), str(item.get("end_id", "?")))
            for index, item in enumerate(data.get("segments") or [])
        ]
        choices: list[tuple[str, tuple[str, ...]]] = [(f"全部分段（{len(segments)} 段）", ())]
        if len(segments) > 3:
            for start in range(0, len(segments), 3):
                batch = segments[start:start + 3]
                ids = tuple(item[0] for item in batch)
                choices.append((f"分批检查：分段 {ids[0]}–{ids[-1]}（{len(ids)} 段）", ids))
        choices.extend((f"只检查分段 {sid}：{start}–{end}", (sid,)) for sid, start, end in segments)
        return choices

    def _refresh_propagation_status(self) -> None:
        if not self.session:
            self.propagation_status_label.setText("后续视频帧：SAM2 传播与 YOLO 导出尚未运行。")
            return
        try:
            summary = load_export_summary(self._scene_export_report())
        except (OSError, ValueError, TypeError) as exc:
            self.propagation_status_label.setText(f"后续视频帧：导出报告读取失败：{exc}")
            return
        if summary is None:
            self.propagation_status_label.setText("后续视频帧：SAM2 传播与 YOLO 导出尚未运行。")
        elif summary.all_accepted:
            self.propagation_status_label.setText(
                f"后续视频帧：已处理 {summary.total} 帧，accepted {summary.accepted}，全部通过自动质量门。"
            )
        else:
            self.propagation_status_label.setText(
                f"后续视频帧：已处理 {summary.total} 帧；accepted {summary.accepted}，"
                f"review {summary.review}，rejected {summary.rejected}，待检查 {summary.needs_review}。"
            )

    def _active_scene_path(self) -> Path | None:
        if not self.session:
            return None
        if self.session.scene_dir.exists():
            return self.session.scene_dir
        return self.session.staging_dir

    def _has_rgbd_frames(self) -> bool:
        path = self._active_scene_path()
        return bool(path and has_paired_rgbd_frames(path))

    def _frame_count(self) -> int:
        path = self._active_scene_path()
        return paired_rgbd_frame_count(path) if path else 0

    def _refresh_capture_metrics(self) -> None:
        self.frame_count.setText(f"{self._frame_count()} 对")

    def _refresh_scene_list(self) -> None:
        self.scene_list.clear()
        states = []
        selected = self.selected_class
        if selected is not None:
            self.saved_scenes_group.setTitle(f"{selected.chinese_name}（{selected.name}）的已保存场次")
            class_root = self.project_root / "data" / "scenes" / selected.name
            if class_root.is_dir():
                for scene in sorted((path for path in class_root.iterdir() if path.is_dir()), reverse=True):
                    try:
                        states.append(scene_workflow_state(self.project_root, scene))
                    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                        self._append_log(f"[场次状态] {scene.name}: {exc}")
        else:
            self.saved_scenes_group.setTitle("已保存场次（请先选择物品）")

        summary = summarize_scene_states(states)
        if selected is None:
            self.class_frame_summary.setText("当前物品数据概览：请先选择物品。")
        else:
            self.class_frame_summary.setText(
                f"当前物品数据概览：场次 {summary.scene_count} | RGB-D 总帧 {summary.paired_frames} | "
                f"已处理 {summary.processed_frames}（accepted {summary.accepted} / "
                f"review {summary.review} / rejected {summary.rejected}） | "
                f"待 SAM2 传播 {summary.pending_propagation_scenes} 场 | "
                f"待人工标记 {summary.needs_manual_scenes} 场"
            )

        groups = (
            ("needs_manual", "未标记完成"),
            ("keyframes_complete", "已标记完成"),
            ("human_reviewed", "已经过人工 Review"),
        )
        colors = {
            "blue": QColor("#d6ecff"),
            "yellow": QColor("#fff3b0"),
            "green": QColor("#d8f3dc"),
            "red": QColor("#ffd6d6"),
            "gray": QColor("#e5e7eb"),
        }
        for group_code, title in groups:
            group_states = [state for state in states if state.group == group_code]
            header = QListWidgetItem(f"── {title}（{len(group_states)}）──")
            header.setFlags(Qt.NoItemFlags)
            header.setBackground(QColor("#cbd5e1"))
            self.scene_list.addItem(header)
            if not group_states:
                empty = QListWidgetItem(f"暂无{title}的场次")
                empty.setFlags(Qt.NoItemFlags)
                empty.setForeground(QColor("#6b7280"))
                self.scene_list.addItem(empty)
                continue
            for state in sorted(group_states, key=lambda item: item.scene_name, reverse=True):
                item = QListWidgetItem(f"{state.scene_name}  |  {state.detail}")
                item.setBackground(colors[state.color])
                item.setToolTip(f"类别：{state.class_name}；split：{state.split}；状态：{state.code}")
                item.setData(Qt.UserRole, {
                    "scene_name": state.scene_name,
                    "class_name": state.class_name,
                    "split": state.split,
                    "status": state.code,
                })
                self.scene_list.addItem(item)

    def _all_scene_states(self):
        states = []
        scenes_root = self.project_root / "data" / "scenes"
        for object_class in self.classes:
            class_root = scenes_root / object_class.name
            if not class_root.is_dir():
                continue
            for scene in sorted(path for path in class_root.iterdir() if path.is_dir()):
                states.append(scene_workflow_state(self.project_root, scene))
        return states

    def _missing_masks_for_segment(self, segment: dict, report: Path) -> list[str]:
        required = segment.get("required_key_mask_paths") or {}
        if not required:
            return list(segment.get("missing_key_masks") or [])
        missing: list[str] = []
        for instance, raw_path in required.items():
            path = Path(raw_path)
            if not path.is_absolute():
                path = (report.parent / path).resolve()
            if not path.exists():
                missing.append(str(instance))
        return missing

    def _masks_complete(self) -> bool:
        try:
            return load_mask_progress(self._segments_report()).complete
        except (OSError, ValueError, TypeError):
            return False

    def _refresh_mask_progress(self) -> None:
        self.mask_progress_list.clear()
        try:
            progress = load_mask_progress(self._segments_report())
        except (OSError, ValueError, TypeError) as exc:
            self.mask_progress_summary.setText("关键帧进度：分段报告读取失败")
            self.mask_progress_list.addItem(str(exc))
            return
        if not progress.segments:
            self.mask_progress_summary.setText("关键帧进度：尚未分段")
            self.mask_progress_list.addItem("点击“开始标记 / 标下一个”，App 会先自动分段。")
            return
        self.mask_progress_summary.setText(
            f"关键帧进度：{progress.completed_required}/{progress.total_required} 个必需 Mask 已保存"
        )
        for item in progress.segments:
            state = "已完成" if item.complete else f"未完成：{', '.join(item.missing_instances) or '缺少 Mask'}"
            self.mask_progress_list.addItem(
                f"分段 {item.segment_id}（帧 {item.start_id}–{item.end_id}）：{state}"
            )

    def has_independent_val(self) -> bool:
        yaml_path = self._dataset_path()
        if not yaml_path.is_file():
            return False
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            root = Path(data.get("path", yaml_path.parent)).expanduser()
            if not root.is_absolute():
                root = (yaml_path.parent / root).resolve()

            def split_dir(name: str) -> Path | None:
                raw = data.get(name)
                if not isinstance(raw, str) or not raw.strip():
                    return None
                path = Path(raw).expanduser()
                return path.resolve() if path.is_absolute() else (root / path).resolve()

            train_dir = split_dir("train")
            val_dir = split_dir("val")
            if not train_dir or not val_dir or train_dir == val_dir:
                return False
            image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            return train_dir.is_dir() and val_dir.is_dir() and any(
                path.is_file() and path.suffix.lower() in image_suffixes
                for path in val_dir.iterdir()
            )
        except (OSError, ValueError, TypeError, yaml.YAMLError):
            return False

    def _refresh_state(self, *, refresh_scenes: bool = True) -> None:
        main_active = self.process.state() != QProcess.NotRunning
        live_active = self.live_process.state() != QProcess.NotRunning
        active = main_active or live_active
        has_staging = bool(self.session and self.session.staging_dir.exists())
        has_scene = bool(self.session and self.session.scene_dir.exists())
        has_manifest = bool(self.session and manifest_for_session(self.session).exists())
        report_exists = bool(self.session and self._segments_report().exists())
        masks = self._masks_complete() if self.session else False

        self.start_button.setEnabled(not active and self.session is not None)
        self.stop_button.setEnabled(main_active and self.task_kind == "capture")
        self.stop_capture_shortcut.setEnabled(main_active and self.task_kind == "capture")
        self.save_button.setEnabled(not active and has_staging and not has_scene and self._has_rgbd_frames())
        self.discard_button.setEnabled(not active and has_staging and not has_scene)
        self.mark_button.setEnabled(not active and has_scene)
        self.continue_button.setEnabled(not active and has_scene and has_manifest)
        self.segment_button.setEnabled(not active and has_scene and has_manifest)
        self.run_button.setEnabled(not active and has_manifest and masks)
        self.validate_button.setEnabled(not active and has_manifest and self._dataset_path().exists())
        self.auto_prepare_button.setEnabled(not active)
        self.integrity_button.setEnabled(not active and has_scene)
        try:
            can_review = bool(self._review_contexts())
        except (OSError, ValueError, TypeError, json.JSONDecodeError, yaml.YAMLError):
            can_review = False
        self.review_button.setEnabled(not active and can_review)
        active_scene = self.session.scene_dir if self.session and self.session.scene_dir.is_dir() else None
        if active_scene is not None:
            completion = load_scene_human_review_completion(self.project_root, active_scene)
            marker_exists = completion.marker_path.is_file()
        else:
            completion = None
            marker_exists = False
        self.mark_review_complete_button.setEnabled(
            not active and can_review and completion is not None and not completion.valid
        )
        self.clear_review_complete_button.setEnabled(not active and marker_exists)
        self.train_button.setEnabled(not active and self.validation_passed and self.has_independent_val())
        self.stop_train_button.setEnabled(main_active and self.task_kind == "train")
        self.live_start_button.setEnabled(not active and bool(self._live_model_path))
        self.live_stop_button.setEnabled(live_active)
        self._refresh_mask_progress()
        self._refresh_propagation_status()

        if active and self.task_kind == "capture":
            guidance = "正在采集：观察独立 RGB-D 预览，完成后点击“停止采集”。"
        elif has_staging and not has_scene:
            guidance = "采集已暂存：确认配对帧后保存，或丢弃本次数据。"
        elif has_scene and not has_manifest:
            guidance = "场次已保存但 Manifest 尚未创建；点击“开始标记”会先重试初始化。"
        elif has_scene and has_manifest and not report_exists:
            guidance = "可以点击“开始标记”；App 会先自动分段，再打开 Mask 标记器。"
        elif report_exists and not masks:
            progress = load_mask_progress(self._segments_report())
            guidance = (
                f"关键帧 Mask 已保存 {progress.completed_required}/{progress.total_required}；"
                "在编辑器中按 Enter 应用、S 保存、Q 退出，然后点击“开始标记 / 标下一个”。"
            )
        elif masks:
            guidance = "关键帧 Mask 已完成；点击“继续自动处理”运行 SAM2 传播和 YOLO 导出。"
        else:
            guidance = "选择类别后点击“开始采集”。"
        self.guidance_label.setText(guidance)

        self.workflow_status.clear()
        self.workflow_status.addItems([
            f"采集：{'已保存' if has_scene else ('暂存中' if has_staging else '未开始')}",
            f"Manifest：{'已创建' if has_manifest else '未创建'}",
            f"关键帧 Mask：{'已完成' if masks else '未完成'}",
            f"训练 val：{'已检测到独立 val' if self.has_independent_val() else '缺少独立 val，禁止正式训练'}",
            f"数据验证：{'通过，可以训练' if self.validation_passed else '尚未通过，训练保持锁定'}",
        ])
        try:
            useful_classes = sorted({state.class_name for state in self._all_scene_states() if state.training_eligible})
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            useful_classes = []
        missing_classes = [obj.name for obj in self.classes if obj.name not in useful_classes]
        self.train_status.setText(
            f"本次有效标签类别：{', '.join(useful_classes) if useful_classes else '暂无'}\n"
            f"尚无有效标签：{', '.join(missing_classes) if missing_classes else '无'}\n"
            + ("训练数据已验证，可以开始阶段性训练。" if self.validation_passed else "点击“自动准备训练数据”并通过验证后解锁训练。")
        )
        self.dataset_label.setText(str(self._dataset_path()))
        self._refresh_capture_metrics()
        self.refresh_training_result()
        if refresh_scenes:
            self._refresh_scene_list()

    # ---------- process handling ----------
    def _append_log(self, text: str) -> None:
        cleaned = text.rstrip("\n")
        if cleaned:
            self.log.appendPlainText(cleaned)

    def _read_stdout(self) -> None:
        self._append_log(bytes(self.process.readAllStandardOutput()).decode(errors="replace"))

    def _read_stderr(self) -> None:
        self._append_log(bytes(self.process.readAllStandardError()).decode(errors="replace"))

    def _start_task(self, kind: str, program: str, args: list[str]) -> None:
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "任务进行中", "请先停止或等待当前任务完成。")
            return
        self.task_kind = kind
        self.process_status.setText(kind)
        self._append_log("$ " + " ".join([program, *args]))
        self.process.start(program, args)
        self._refresh_state()

    def _process_started(self) -> None:
        self._append_log(f"[开始] {self.task_kind}")
        if self.task_kind == "capture":
            self.camera_status.setText("采集程序运行中，等待 RGB-D 预览")
        self._refresh_state()

    def _process_error(self, error: QProcess.ProcessError) -> None:
        self._append_log(f"[进程错误] {error}")
        if self.task_kind == "capture":
            self.camera_status.setText("相机未就绪（查看日志）")
        self._show_log()
        self._refresh_state()

    def _process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        del exit_status
        kind = self.task_kind
        self._append_log(f"[结束] {kind}: exit={exit_code}")
        self.process_status.setText("空闲")
        succeeded = exit_code == 0 or (kind == "capture" and exit_code == 130) or (kind == "review" and exit_code == 20)

        if kind == "capture":
            if succeeded:
                self.camera_status.setText("已停止")
                self._refresh_state()
                self._ask_save_after_capture()
            else:
                self.camera_status.setText("相机未就绪（查看日志）")
                cleaned = bool(self.session and discard_failed_empty_capture(self.session))
                if cleaned:
                    self._append_log("[清理] 采集未产生 RGB/Depth PNG，已删除空暂存目录。")
                else:
                    self._append_log("[保留] 检测到部分采集文件，暂存目录保留供检查或手动丢弃。")
                self._show_log()
                self._refresh_state()
            return

        pending = self.pending_after_task if succeeded else None
        self.pending_after_task = None
        if not succeeded:
            self._show_log()
        if kind in {"validate", "auto_validate"}:
            self.validation_passed = succeeded
        self._refresh_state()
        if kind == "auto_run":
            scene_name = getattr(self, "_auto_prepare_current_scene", "未知场次")
            if not succeeded:
                self._auto_prepare_failures.append(scene_name)
                self._append_log(f"[自动准备] {scene_name} 自动处理失败，已跳过；不会进入训练集。")
            QTimer.singleShot(0, self._run_next_auto_scene)
            return
        if kind == "auto_validate":
            if succeeded:
                QMessageBox.information(self, "训练数据已就绪", "train/val 数据与安全检查已通过，可以开始阶段性训练。")
            else:
                QMessageBox.warning(self, "训练数据验证失败", "训练按钮保持禁用；请查看日志中的准确原因。")
            return
        if kind == "validate":
            if succeeded:
                QMessageBox.information(self, "验证通过", "训练数据验证通过，已解锁开始训练。")
            return
        if kind == "review":
            action = {}
            if self._review_action_file and self._review_action_file.is_file():
                try:
                    action = json.loads(self._review_action_file.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    action = {}
            if succeeded and action.get("action") in {"rerun_ranges", "rerun_range"}:
                self.validation_passed = False
                try:
                    command = self.command_for_review_rerun(action)
                except Exception as exc:
                    self._append_log(f"[Review/失败] 无法启动局部传播：{exc}")
                    QMessageBox.warning(self, "无法局部传播", str(exc))
                    return
                count = len(action.get("ranges") or [action])
                self._append_log(
                    f"[Review] 已关闭检查窗口；本次新增 {count} 个关键帧范围，"
                    "现在统一执行一次局部SAM2传播和YOLO刷新。旧验证结果已失效。"
                )
                QTimer.singleShot(0, lambda command=command: self._start_task("review_rerun", *command))
            elif succeeded and action.get("action") == "review_changed":
                self.validation_passed = False
                try:
                    command = self.command_for_review_export()
                except Exception as exc:
                    self._append_log(f"[Review/失败] 无法重新导出：{exc}")
                    QMessageBox.warning(self, "无法重新导出", str(exc))
                    return
                self._append_log(
                    "[Review] A/R/X人工状态已更新；只重新执行质量检查和YOLO聚合，"
                    "不会重跑SAM2。旧验证结果已失效。"
                )
                QTimer.singleShot(0, lambda command=command: self._start_task("review_export", *command))
            elif succeeded:
                self._append_log("[Review] 检查结束；没有检测到人工修改。")
            return
        if kind == "review_rerun":
            if succeeded:
                self._append_log("[Review] 本次批量局部传播和YOLO刷新已完成；可再次打开检查效果。")
            else:
                self._append_log("[Review/失败] 局部传播未完成；工具不会自动回退为整场重跑，请查看上方根因。")
                QMessageBox.warning(self, "局部传播失败", "旧结果已尽量保留；请查看日志中的准确失败原因。")
            return
        if kind == "review_export":
            if succeeded:
                self._append_log("[Review] 人工A/R/X状态已重新聚合到YOLO数据。")
            else:
                QMessageBox.warning(self, "重新导出失败", "SAM2 Mask未重跑；请查看日志中的准确失败原因。")
            return
        if kind == "mask":
            if succeeded:
                progress = load_mask_progress(self._segments_report())
                if progress.complete:
                    QMessageBox.information(
                        self, "关键帧已完成",
                        f"已保存 {progress.completed_required}/{progress.total_required} 个关键 Mask。现在可以点击“继续自动处理”。"
                    )
                else:
                    QMessageBox.information(
                        self, "本张 Mask 已保存",
                        f"当前已保存 {progress.completed_required}/{progress.total_required}。点击“开始标记 / 标下一个”继续。"
                    )
            else:
                QMessageBox.warning(
                    self, "Mask 未保存",
                    "编辑器没有正常保存。请重新打开，并按 Enter 应用轮廓、S 保存、Q 退出。"
                )
            return
        if kind == "train" and succeeded:
            self.refresh_training_result()
            self.advanced_toggle.setChecked(True)
        if kind == "init" and pending == "mark":
            QTimer.singleShot(0, self.start_annotation)
            return
        if kind == "segment" and pending == "mark":
            QTimer.singleShot(0, self.mark_next_missing)

    def stop_task(self) -> None:
        if self.process.state() != QProcess.NotRunning:
            if self.task_kind == "capture":
                self.stop_capture()
            else:
                self.process.terminate()

    def stop_capture(self) -> None:
        if self.process.state() == QProcess.NotRunning or self.task_kind != "capture":
            return
        pid = int(self.process.processId())
        if pid:
            try:
                os.killpg(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        self.process_status.setText("正在等待采集程序收尾")
        self.stop_button.setEnabled(False)

    # ---------- daily actions ----------
    def start_capture(self) -> None:
        self._new_session_preview()
        assert self.session is not None
        if self.session.scene_dir.exists():
            QMessageBox.warning(self, "场次冲突", f"正式场次已存在：{self.session.scene_dir}")
            return
        if self.session.staging_dir.exists():
            if not discard_failed_empty_capture(self.session):
                QMessageBox.warning(self, "暂存冲突", f"暂存目录已有数据，请先处理：{self.session.staging_dir}")
                return
        self.session.staging_dir.mkdir(parents=True, exist_ok=False)
        self.camera_status.setText("正在启动；预览窗口即将出现")
        self._start_task("capture", *self.command_for_capture())

    def _ask_save_after_capture(self) -> None:
        if not self.session or not self.session.staging_dir.exists():
            return
        if not self._has_rgbd_frames():
            cleaned = discard_failed_empty_capture(self.session)
            if cleaned:
                self._append_log("[清理] 本次没有产生 RGB/Depth PNG，空暂存目录已删除。")
            else:
                self._append_log("[采集] RGB/Depth 不成对，已保留暂存目录供检查；不能保存为正式场次。")
                self._show_log()
            self._refresh_state()
            return
        box = QMessageBox(self)
        box.setWindowTitle("保存本次采集？")
        box.setText(f"已采集 {self._frame_count()} 对 RGB/Depth 帧，是否保存？")
        save = box.addButton("保存", QMessageBox.AcceptRole)
        discard = box.addButton("丢弃", QMessageBox.DestructiveRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.exec_()
        if box.clickedButton() is save:
            self.finalize_capture("save")
        elif box.clickedButton() is discard:
            self.finalize_capture("discard")

    def finalize_capture(self, action: str) -> None:
        if not self.session:
            return
        if action == "save":
            try:
                if not has_paired_rgbd_frames(self.session.staging_dir):
                    raise RuntimeError("没有完整配对的 RGB/Depth PNG，拒绝保存正式场次。")
                target = self.session.save()
                write_session_metadata(self.session, target)
                self._append_log(f"[保存] {target}")
                self._start_task("init", *self.command_for_init())
            except Exception as exc:
                self._append_log(f"[保存失败] {exc}")
                self._show_log()
                QMessageBox.critical(self, "保存失败", str(exc))
        elif action == "discard":
            try:
                self.session.discard()
                self._append_log("[丢弃] 本次暂存已删除")
                self._new_session_preview()
            except Exception as exc:
                self._show_log()
                QMessageBox.critical(self, "丢弃失败", str(exc))

    def start_annotation(self) -> None:
        try:
            session = self._require_session()
            if not session.scene_dir.exists():
                raise FileNotFoundError(f"正式场次不存在，请先保存采集：{session.scene_dir}")
            if not manifest_for_session(session).exists():
                self.pending_after_task = "mark"
                self._start_task("init", *self.command_for_init())
                return
            if not self._segments_report().exists():
                self.pending_after_task = "mark"
                self.run_segment()
                return
            self.mark_next_missing()
        except Exception as exc:
            QMessageBox.warning(self, "无法开始标记", str(exc))

    def continue_processing(self) -> None:
        try:
            self._require_manifest()
            if not self._segments_report().exists():
                self.pending_after_task = "mark"
                self.run_segment()
            elif not self._masks_complete():
                self.mark_next_missing()
            else:
                self.run_pipeline()
        except Exception as exc:
            QMessageBox.warning(self, "无法继续处理", str(exc))

    def start_review(self) -> None:
        try:
            contexts = self._review_contexts()
            if not contexts:
                raise FileNotFoundError("当前场次还没有可播放的 SAM2 Mask 和逐帧质量报告，请先完成自动处理。")
            context = contexts[0]
            if len(contexts) > 1:
                labels = [f"{item['instance_id']} ({item['class_name']})" for item in contexts]
                selected, ok = QInputDialog.getItem(self, "选择实例", "要检查哪个实例？", labels, 0, False)
                if not ok:
                    return
                context = contexts[labels.index(selected)]
            choices = self._review_segment_choices()
            labels = [label for label, _ids in choices]
            selected, ok = QInputDialog.getItem(
                self, "选择检查范围", "可检查全部、每批3段，或单独一个分段：", labels, 0, False,
            )
            if not ok:
                return
            segment_ids = choices[labels.index(selected)][1]
            program, args = self.command_for_review(context, segment_ids)
            if self._review_action_file and self._review_action_file.exists():
                self._review_action_file.unlink()
            self._append_log(
                "[Review] 空格播放/暂停，左右逐帧，A/R/X修改状态；"
                "K可连续增加关键帧，保存后继续检查，关闭Review时才统一局部传播。"
            )
            self._start_task("review", program, args)
        except Exception as exc:
            QMessageBox.warning(self, "无法检查标注效果", str(exc))

    def mark_current_scene_review_complete(self) -> None:
        """Explicitly mark the entire selected scene as human reviewed."""
        try:
            session = self._require_session()
            if not session.scene_dir.is_dir():
                raise FileNotFoundError("当前场次尚未正式保存")
            if not self._review_contexts():
                raise FileNotFoundError("当前场次没有可播放的SAM2 Mask和质量报告，请先完成自动处理")
            answer = QMessageBox.question(
                self,
                "确认整场人工 Review 完成",
                "请确认你已经检查了当前采集场次的全部必要分段，而不是只检查其中一小段。\n\n"
                "此操作只写入完成标记，不会移动或修改RGB-D、Mask和YOLO数据。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            marker = mark_scene_human_review_complete(self.project_root, session.scene_dir)
            self._append_log(f"[Review] 已标记整场人工Review完成：{marker}")
            self._refresh_state()
            QMessageBox.information(
                self,
                "人工 Review 完成",
                "当前场次已移入“已经过人工 Review”。如果之后重新传播或重新导出，标记会自动失效。",
            )
        except Exception as exc:
            QMessageBox.warning(self, "无法标记 Review 完成", str(exc))

    def clear_current_scene_review_complete(self) -> None:
        """Clear only the selected scene's completion marker."""
        try:
            session = self._require_session()
            marker = session.scene_dir / "project_reports" / "manual_review_complete.json"
            if not marker.is_file():
                self._refresh_state()
                return
            answer = QMessageBox.question(
                self,
                "取消人工 Review 完成标记",
                "只取消完成状态，不会删除或修改RGB-D、Mask、Manifest和YOLO标签。是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            clear_scene_human_review_complete(session.scene_dir)
            self._append_log(f"[Review] 已取消整场人工Review完成标记：{session.scene_name}")
            self._refresh_state()
        except Exception as exc:
            QMessageBox.warning(self, "无法取消 Review 完成标记", str(exc))

    # ---------- advanced command actions ----------
    def run_segment(self) -> None:
        try:
            command = self.command_for_segment()
        except Exception as exc:
            self.pending_after_task = None
            QMessageBox.warning(self, "无法分段", str(exc))
            return
        self._start_task("segment", *command)

    def _next_missing_mask(self) -> tuple[Path, Path] | None:
        report = self._segments_report()
        if not report.exists():
            return None
        data = json.loads(report.read_text(encoding="utf-8"))
        scene = Path(data.get("scene", self.session.scene_dir))
        if not scene.is_absolute():
            scene = (report.parent / scene).resolve()
        for segment in data.get("segments", []):
            missing = self._missing_masks_for_segment(segment, report)
            if not missing:
                continue
            instance = missing[0]
            output_value = (segment.get("required_key_mask_paths") or {}).get(instance)
            if not output_value:
                continue
            output = Path(output_value)
            if not output.is_absolute():
                output = (report.parent / output).resolve()
            frame_id = str(segment["start_id"])
            image_name = frame_id if frame_id.lower().endswith(".png") else f"{frame_id}.png"
            return scene / "rgb" / image_name, output
        return None

    def mark_next_missing(self) -> None:
        try:
            next_item = self._next_missing_mask()
        except Exception as exc:
            QMessageBox.warning(self, "读取分段失败", str(exc))
            return
        if not next_item:
            self._append_log("[Mask] 没有发现缺失关键帧；可以继续自动处理。")
            self._refresh_state()
            return
        image, output = next_item
        self._append_log("[Mask 操作] Enter 应用多边形 → S 保存 → Q/ESC 退出。")
        self._start_task("mask", *self.command_for_mask(image, output))

    def run_pipeline(self) -> None:
        try:
            command = self.command_for_run()
        except Exception as exc:
            QMessageBox.warning(self, "无法运行", str(exc))
            return
        self.validation_passed = False
        self._start_task("run", *command)

    def start_auto_prepare(self) -> None:
        """Process completed keyframes, split whole scenes, then validate."""
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "任务进行中", "请等待当前任务完成后再准备训练数据。")
            return
        self.validation_passed = False
        self._auto_prepare_failures = []
        try:
            states = self._all_scene_states()
        except Exception as exc:
            QMessageBox.warning(self, "扫描场次失败", str(exc))
            return
        manual = [state for state in states if state.group == "needs_manual"]
        pending = [state for state in states if state.code in {"pending_export", "export_failed"}]
        usable = [state for state in states if state.training_eligible]
        self._append_log(
            f"[自动准备] 扫描 {len(states)} 个场次：待人工 {len(manual)}，"
            f"待/需重跑自动处理 {len(pending)}，已有 accepted {len(usable)}。"
        )
        self._auto_prepare_queue = [
            (state.scene_name, state.manifest_path)
            for state in pending if state.manifest_path is not None
        ]
        if self._auto_prepare_queue:
            self._run_next_auto_scene()
        else:
            self._preview_auto_split()

    def _run_next_auto_scene(self) -> None:
        if not self._auto_prepare_queue:
            self._preview_auto_split()
            return
        scene_name, manifest = self._auto_prepare_queue.pop(0)
        self._auto_prepare_current_scene = scene_name
        self._append_log(f"[自动准备] 处理场次 {scene_name}")
        self._start_task(
            "auto_run", str(ROOT / "scripts" / "atec-pipeline"), ["run", str(manifest)]
        )

    def _preview_auto_split(self) -> None:
        try:
            plan = build_auto_plan(self._dataset_path(), target_ratio=0.20)
        except Exception as exc:
            self._refresh_state()
            QMessageBox.warning(self, "无法准备验证集", str(exc))
            self._append_log(f"[自动准备失败] {exc}")
            return
        if not plan.get("files"):
            self._append_log("[自动准备] 已存在固定 val，无需重新洗牌；开始验证。")
            self._start_task("auto_validate", *self.command_for_validate())
            return
        val_scenes = sorted({name for group in plan["groups"] for name in group.scene_names})
        all_groups = [state for state in self._all_scene_states() if state.training_eligible and state.split == "train"]
        train_scenes = sorted(state.scene_name for state in all_groups if state.scene_name not in val_scenes)
        val_frames = len(plan["files"])
        train_frames = sum(state.accepted for state in all_groups) - val_frames
        preview = (
            "准备按完整场次划分验证集（不会拆分同一采集视频）\n\n"
            "train:\n- " + ("\n- ".join(train_scenes) or "无")
            + "\n\nval:\n- " + "\n- ".join(val_scenes)
            + f"\n\n预计：train {max(0, train_frames)} 张，val {val_frames} 张\n\n确认划分？"
        )
        answer = QMessageBox.question(
            self, "自动准备训练数据", preview,
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            self._append_log("[自动准备] 用户取消验证集划分，未修改文件。")
            self._refresh_state()
            return
        try:
            apply_plan(plan)
        except Exception as exc:
            QMessageBox.critical(self, "验证集划分失败", f"事务已回滚：{exc}")
            self._append_log(f"[自动准备失败/已回滚] {exc}")
            self._show_log()
            self._refresh_state()
            return
        self._append_log(f"[自动准备] 已将 {val_frames} 个 accepted 样本按完整场次划入 val。")
        self._refresh_state()
        self._start_task("auto_validate", *self.command_for_validate())

    def run_validate(self) -> None:
        try:
            command = self.command_for_validate()
        except Exception as exc:
            QMessageBox.warning(self, "无法验证", str(exc))
            return
        self.validation_passed = False
        self._start_task("validate", *command)

    def run_train(self) -> None:
        if not self.validation_passed or not self.has_independent_val():
            QMessageBox.warning(self, "禁止训练", "训练数据尚未通过独立 val 与数据安全验证。")
            return
        self._start_task("train", *self.command_for_train())

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.process.state() != QProcess.NotRunning or self.live_process.state() != QProcess.NotRunning:
            answer = QMessageBox.question(self, "任务运行中", "仍有后台任务或实时识别，确定退出吗？")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.stop_task()
            self.stop_live_recognition()
        self.frame_timer.stop()
        event.accept()


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv or sys.argv)
    window = AtecMainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
