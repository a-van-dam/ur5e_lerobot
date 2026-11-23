#!/usr/bin/env python3
"""
UR5e Policy Training Studio — LeRobot Imitation Learning GUI
============================================================
A graphical user interface wrapping train_robot.py for training, fine-tuning,
and resuming imitation learning policies (ACT, Diffusion Policy, VQ-BeT) on UR5e datasets.

Features:
- Dataset auto-detection with episode, frame, and camera metadata badges.
- Dynamic model naming (default: <dataset>_<policy>) to train different policies on the same dataset.
- Automatic folder creation (outputs/train/<model_name>/checkpoint/...) with zero manual setup.
- Checkpoint resumption & fine-tuning: pick any previous model and continue training for additional steps.
- Comprehensive hyperparameter tuning (steps, batch size, learning rates, optimizer, augmentations, chunk size, VAE).
- Live training dashboard: real-time loss graph, KPI metrics (train/val loss, speed, ETA), and color console.
- Interactive Web Browser Dashboard: one-click button to open rich Chart.js loss curves in Chrome/Firefox.
- One-click launch into GUI Tester (gui_tester.py) to immediately evaluate trained policies on the robot.
- Real-time GPU VRAM & hardware utilization monitor.
"""

from __future__ import annotations

import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
import webbrowser

import numpy as np
import torch

from PyQt6.QtCore import (
    QMutex,
    QMutexLocker,
    QObject,
    QPoint,
    QRect,
    QRectF,
    QSize,
    QThread,
    QTimer,
    QUrl,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Force PyAV backend to avoid conda torchcodec C++ glib symbol mismatch
try:
    import lerobot.utils.import_utils
    lerobot.utils.import_utils.get_safe_default_video_backend = lambda: "pyav"
except Exception:
    pass

# Import backend training manager from train_robot.py
try:
    import train_robot
    from train_robot import (
        AUG_COLOR,
        AUG_CUTOUT,
        AUG_NOISE,
        AUG_ROTATION,
        AUG_TRANSLATION,
        AVAILABLE_OPTIMIZERS,
        AVAILABLE_POLICIES,
        BATCH_SIZE,
        DATASET_PATH,
        DEFAULT_CHUNK_SIZE,
        DEVICE,
        EVAL_FREQ,
        GRAD_CLIP_NORM,
        LEARNING_RATE,
        LEROBOT_AVAILABLE,
        LOG_FREQ,
        LR_BACKBONE,
        LR_MIN,
        NUM_WORKERS,
        OUTPUT_DIR,
        SAVE_CHECKPOINT_FREQ,
        SEED,
        TRAINING_STEPS,
        USE_AUGMENTATION,
        VAL_SPLIT_RATIO,
        WARMUP_STEPS,
        WEIGHT_DECAY,
        TEMPORAL_ENSEMBLE_COEFF,
        RobotTrainer,
    )
    LEROBOT_IMPORT_ERROR = getattr(train_robot, "LEROBOT_IMPORT_ERROR", None)
except Exception as err:
    LEROBOT_AVAILABLE = False
    LEROBOT_IMPORT_ERROR = str(err)
    AVAILABLE_POLICIES = ["act", "diffusion", "vqbet"]
    AVAILABLE_OPTIMIZERS = ["AdamW", "Adam", "SGD"]
    TRAINING_STEPS = 20000
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    LR_BACKBONE = 1e-5
    LR_MIN = 1e-6
    WARMUP_STEPS = 1000
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 10.0
    DEFAULT_CHUNK_SIZE = 60
    USE_AUGMENTATION = True
    TEMPORAL_ENSEMBLE_COEFF = 0.01
    AUG_ROTATION = True
    AUG_TRANSLATION = True
    AUG_COLOR = True
    AUG_CUTOUT = False
    AUG_NOISE = False
    SAVE_CHECKPOINT_FREQ = 2500
    LOG_FREQ = 50
    EVAL_FREQ = 1000
    VAL_SPLIT_RATIO = 0.1
    OUTPUT_DIR = "outputs/train/model"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    SEED = 42


# =============================================================================
# 1. Stdout / Stderr Stream Redirection to GUI Console
# =============================================================================

class ConsoleLogEmitter(QObject):
    log_signal = pyqtSignal(str)

    def __init__(self, original_stream):
        super().__init__()
        self.original_stream = original_stream

    def write(self, text):
        if self.original_stream:
            try:
                self.original_stream.write(text)
                self.original_stream.flush()
            except Exception:
                pass
        if text.strip():
            self.log_signal.emit(text)

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass


# =============================================================================
# 2. Asynchronous Training Background Worker Thread
# =============================================================================

class TrainingWorker(QThread):
    step_progress = pyqtSignal(dict)
    plot_updated = pyqtSignal(str)
    training_finished = pyqtSignal(bool, str, dict)

    def __init__(self, trainer_kwargs: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.trainer_kwargs = dict(trainer_kwargs)
        self.trainer: Optional[RobotTrainer] = None
        self._is_aborted = False

    def abort(self):
        self._is_aborted = True
        if self.trainer:
            self.trainer.abort()

    def run(self):
        try:
            # Wire up step callback
            def on_step(metrics: Dict[str, Any]):
                self.step_progress.emit(metrics)

            self.trainer_kwargs["step_callback"] = on_step
            self.trainer = RobotTrainer(**self.trainer_kwargs)

            # Check if loss plot already exists or as it updates
            loss_file = self.trainer.loss_plot_file

            results = self.trainer.run_training()

            if loss_file.exists():
                self.plot_updated.emit(str(loss_file))

            success = (results.get("status") == "COMPLETED")
            msg = f"Training {results.get('status')} after {results.get('steps_completed', 0)} steps."
            self.training_finished.emit(success, msg, results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.training_finished.emit(False, f"Training failed with error: {e}", {})


# =============================================================================
# 3. Dynamic Loss Plot Display Widget
# =============================================================================

class LossPlotWidget(QFrame):
    """Displays the loss convergence curve image generated by training."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            LossPlotWidget {
                background-color: #0b1120;
                border: 1px solid #1e293b;
                border-radius: 8px;
            }
        """)
        self.image_path: Optional[Path] = None
        self.pixmap: Optional[QPixmap] = None
        self.setMinimumHeight(240)

    def set_image(self, path: str | Path):
        p = Path(path)
        if p.exists():
            self.image_path = p
            self.pixmap = QPixmap(str(p))
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = self.rect()

        if self.pixmap and not self.pixmap.isNull():
            # Scale pixmap maintaining aspect ratio to fit widget
            scaled = self.pixmap.scaled(
                rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (rect.width() - scaled.width()) // 2
            y = (rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            # Draw placeholder canvas
            painter.setPen(QPen(QColor("#334155"), 1, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(rect.adjusted(12, 12, -12, -12), 6, 6)

            painter.setPen(QColor("#64748b"))
            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignCenter,
                "📉 Loss Curve Display\n(Live chart will render once first checkpoint is saved)",
            )


# =============================================================================
# 4. Main Training Studio Window
# =============================================================================

class UR5eTrainerStudio(QMainWindow):
    """
    Complete Graphical User Interface for training imitation learning models
    with the Hugging Face LeRobot framework on UR5e manipulation datasets.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("UR5e Policy Training Studio — LeRobot Imitation Learning")
        self.resize(1300, 880)
        self.setMinimumSize(1150, 750)

        self.project_root = Path(__file__).resolve().parent
        self.datasets_dir = self.project_root / "datasets"
        self.outputs_dir = self.project_root / "outputs" / "train"

        # User custom naming tracking
        self._user_edited_name = False

        # Background worker
        self.training_worker: Optional[TrainingWorker] = None
        self.is_training_active = False
        self.training_start_time = 0.0

        # Redirect standard output to GUI terminal console
        self.log_emitter = ConsoleLogEmitter(sys.stdout)
        self.log_emitter.log_signal.connect(self.append_log_message)
        sys.stdout = self.log_emitter

        # Setup Theme & Interface
        self.apply_theme()
        self.setup_ui()

        # Initial scans
        self.rescan_datasets()
        self.rescan_checkpoints()

        # Telemetry timer for GPU stats
        self.gpu_timer = QTimer(self)
        self.gpu_timer.timeout.connect(self.update_gpu_telemetry)
        self.gpu_timer.start(2500)
        self.update_gpu_telemetry()

    # -------------------------------------------------------------------------
    # Visual Theme & Styling
    # -------------------------------------------------------------------------
    def apply_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#090d16"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#f8fafc"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#0f172a"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1e293b"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#f8fafc"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#1e293b"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#0284c7"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        self.setPalette(palette)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #090d16;
            }
            QWidget {
                color: #f8fafc;
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
                letter-spacing: 0.5px;
                border: 1px solid #1e293b;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 14px;
                background-color: #0f172a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #38bdf8;
            }
            QLabel {
                color: #cbd5e1;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 5px;
                padding: 4px 8px;
                color: #ffffff;
                font-size: 12px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #38bdf8;
                background-color: #1a2538;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e293b;
                border: 1px solid #334155;
                selection-background-color: #0284c7;
                selection-color: #ffffff;
            }
            QPushButton {
                background-color: #1e293b;
                color: #ffffff;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #334155;
                border-color: #64748b;
            }
            QPushButton:pressed {
                background-color: #0f172a;
            }
            QPushButton:disabled {
                background-color: #111827;
                color: #475569;
                border-color: #1f2937;
            }
            QTabWidget::pane {
                border: 1px solid #1e293b;
                border-radius: 6px;
                background-color: #0f172a;
            }
            QTabBar::tab {
                background: #1e293b;
                color: #94a3b8;
                padding: 6px 16px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #0f172a;
                color: #38bdf8;
                border-bottom: 2px solid #38bdf8;
            }
            QProgressBar {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #38bdf8);
                border-radius: 5px;
            }
            QScrollBar:vertical {
                background: #0f172a;
                width: 10px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #475569;
            }
        """)

    # -------------------------------------------------------------------------
    # Layout Setup
    # -------------------------------------------------------------------------
    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        root_layout = QHBoxLayout(main_widget)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        # Main splitter dividing Configuration Panel (Left) and Live Dashboard (Right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        # ─── LEFT COLUMN: Configuration Tabs & Controls ───
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # App Title Banner
        banner = QFrame()
        banner.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 10px;")
        banner_layout = QVBoxLayout(banner)
        banner_layout.setContentsMargins(4, 4, 4, 4)
        banner_title = QLabel("🤖 UR5e Policy Training Studio")
        banner_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #ffffff;")
        banner_sub = QLabel("Imitation Learning • ACT & Diffusion Models • Local GPU Acceleration")
        banner_sub.setStyleSheet("font-size: 11px; color: #94a3b8;")
        banner_layout.addWidget(banner_title)
        banner_layout.addWidget(banner_sub)
        left_layout.addWidget(banner)

        # Tabs for Configuration
        tabs = QTabWidget()
        tabs.addTab(self.create_dataset_model_tab(), "1. Dataset & Model")
        tabs.addTab(self.create_hyperparameters_tab(), "2. Hyperparameters")
        tabs.addTab(self.create_advanced_tab(), "3. Hardware & Tuning")
        left_layout.addWidget(tabs, stretch=1)

        splitter.addWidget(left_container)

        # ─── RIGHT COLUMN: Dashboard, Curves & Live Logs ───
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Top Action Bar (Start / Stop, Browser, Test Buttons)
        right_layout.addWidget(self.create_action_bar())

        # Progress & KPI Summary Card
        right_layout.addWidget(self.create_kpi_card())

        # Mid Splitter: Loss Curve Plot (Top) and Live Terminal Log (Bottom)
        mid_splitter = QSplitter(Qt.Orientation.Vertical)

        # Real-time Loss Curve View
        self.loss_plot_widget = LossPlotWidget()
        mid_splitter.addWidget(self.loss_plot_widget)

        # Console Log Widget
        console_box = QGroupBox("TRAINING EXECUTION TERMINAL")
        console_layout = QVBoxLayout(console_box)
        console_layout.setContentsMargins(8, 8, 8, 8)
        console_layout.setSpacing(6)

        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("""
            QTextEdit {
                background-color: #0b1120;
                color: #e2e8f0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #1e293b;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        console_layout.addWidget(self.console_output, stretch=1)

        console_toolbar = QHBoxLayout()
        self.chk_autoscroll = QCheckBox("Auto-scroll output")
        self.chk_autoscroll.setChecked(True)
        console_toolbar.addWidget(self.chk_autoscroll)

        console_toolbar.addStretch()

        self.lbl_gpu_stats = QLabel("GPU: Monitoring...")
        self.lbl_gpu_stats.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        console_toolbar.addWidget(self.lbl_gpu_stats)

        btn_clear_log = QPushButton("Clear Console")
        btn_clear_log.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        btn_clear_log.clicked.connect(self.console_output.clear)
        console_toolbar.addWidget(btn_clear_log)

        console_layout.addLayout(console_toolbar)
        mid_splitter.addWidget(console_box)

        mid_splitter.setStretchFactor(0, 4)
        mid_splitter.setStretchFactor(1, 5)
        right_layout.addWidget(mid_splitter, stretch=1)

        splitter.addWidget(right_container)

        # Splitter sizing ratio: 42% configuration, 58% dashboard/logs
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Select dataset and click 'Start Training'.")

    # -------------------------------------------------------------------------
    # Tab 1: Dataset & Model Identification
    # -------------------------------------------------------------------------
    def create_dataset_model_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # 1. Dataset Selection
        grp_ds = QGroupBox("TRAINING DATASET")
        ds_layout = QVBoxLayout(grp_ds)
        ds_layout.setSpacing(6)

        ds_row = QHBoxLayout()
        self.combo_dataset = QComboBox()
        self.combo_dataset.currentIndexChanged.connect(self.on_dataset_selection_changed)
        ds_row.addWidget(self.combo_dataset, stretch=1)

        btn_rescan_ds = QPushButton("🔄 Refresh")
        btn_rescan_ds.setToolTip("Rescan datasets directory")
        btn_rescan_ds.clicked.connect(self.rescan_datasets)
        ds_row.addWidget(btn_rescan_ds)

        btn_browse_ds = QPushButton("📁 Browse...")
        btn_browse_ds.clicked.connect(self.on_browse_custom_dataset)
        ds_row.addWidget(btn_browse_ds)
        ds_layout.addLayout(ds_row)

        self.lbl_dataset_meta = QLabel("Scanning dataset metadata...")
        self.lbl_dataset_meta.setStyleSheet("color: #38bdf8; font-size: 11px; padding: 2px 4px;")
        self.lbl_dataset_meta.setWordWrap(True)
        ds_layout.addWidget(self.lbl_dataset_meta)

        meta_row = QHBoxLayout()
        self.lbl_channel_mode = QLabel("● RGB Mode (3-Ch)")
        self.lbl_channel_mode.setStyleSheet("color: #a855f7; font-weight: bold; font-size: 11px; padding: 2px 4px;")
        meta_row.addWidget(self.lbl_channel_mode)

        self.chk_grayscale = QCheckBox("Train in Grayscale Mode (Convert to 1-Ch on GPU)")
        self.chk_grayscale.setChecked(False)
        self.chk_grayscale.setStyleSheet("font-size: 11px; color: #cbd5e1; font-weight: bold;")
        self.chk_grayscale.setToolTip("When checked, RGB dataset frames are converted to 1-channel luminance on GPU during training and model is saved with '_gs'. Default is unchecked (RGB color, '_col').")
        self.chk_grayscale.toggled.connect(self.on_grayscale_toggled)
        meta_row.addWidget(self.chk_grayscale)
        meta_row.addStretch()
        ds_layout.addLayout(meta_row)

        layout.addWidget(grp_ds)

        # 2. Policy Architecture Selection
        grp_policy = QGroupBox("POLICY ARCHITECTURE")
        pol_layout = QVBoxLayout(grp_policy)
        pol_layout.setSpacing(6)

        self.combo_policy = QComboBox()
        self.combo_policy.addItem("ACT — Action Chunking with Transformers (Recommended)", "act")
        self.combo_policy.addItem("Diffusion Policy — Multi-modal Denoising Diffusion", "diffusion")
        self.combo_policy.addItem("VQ-BeT — Vector-Quantized Behavior Transformer", "vqbet")
        self.combo_policy.currentIndexChanged.connect(self.on_policy_selection_changed)
        pol_layout.addWidget(self.combo_policy)

        self.lbl_policy_desc = QLabel(
            "ACT handles complex multi-modal trajectories with high sample efficiency and fast GPU inference."
        )
        self.lbl_policy_desc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.lbl_policy_desc.setWordWrap(True)
        pol_layout.addWidget(self.lbl_policy_desc)

        layout.addWidget(grp_policy)

        # 3. Model Naming & Output Destination
        grp_naming = QGroupBox("TRAINED MODEL NAMING & DESTINATION")
        name_layout = QVBoxLayout(grp_naming)
        name_layout.setSpacing(6)

        lbl_name_hint = QLabel("Model Name (Unique folder will be created in outputs/train/):")
        lbl_name_hint.setStyleSheet("font-size: 11px; color: #94a3b8;")
        name_layout.addWidget(lbl_name_hint)

        name_row = QHBoxLayout()
        self.edit_model_name = QLineEdit()
        self.edit_model_name.setPlaceholderText("e.g. move_to_red_cube_act")
        self.edit_model_name.textEdited.connect(self.on_model_name_manually_edited)
        name_row.addWidget(self.edit_model_name, stretch=1)

        btn_reset_name = QPushButton("Reset Default")
        btn_reset_name.setToolTip("Reset to <dataset>_<policy>")
        btn_reset_name.clicked.connect(self.reset_default_model_name)
        name_row.addWidget(btn_reset_name)
        name_layout.addLayout(name_row)

        lbl_out_dir = QLabel("Full Output Path:")
        lbl_out_dir.setStyleSheet("font-size: 11px; color: #94a3b8;")
        name_layout.addWidget(lbl_out_dir)

        out_row = QHBoxLayout()
        self.edit_output_dir = QLineEdit()
        self.edit_output_dir.setReadOnly(True)
        self.edit_output_dir.setStyleSheet("background-color: #0b1120; color: #94a3b8;")
        out_row.addWidget(self.edit_output_dir, stretch=1)

        btn_browse_out = QPushButton("Browse...")
        btn_browse_out.clicked.connect(self.on_browse_output_dir)
        out_row.addWidget(btn_browse_out)
        name_layout.addLayout(out_row)

        layout.addWidget(grp_naming)

        # 4. Checkpoint Resumption & Fine-tuning
        grp_resume = QGroupBox("CONTINUE TRAINING / RESUME FROM CHECKPOINT")
        resume_layout = QVBoxLayout(grp_resume)
        resume_layout.setSpacing(8)

        self.chk_resume = QCheckBox("Resume / Fine-tune from an existing trained checkpoint")
        self.chk_resume.toggled.connect(self.on_resume_toggled)
        resume_layout.addWidget(self.chk_resume)

        self.resume_container = QWidget()
        res_cont_layout = QVBoxLayout(self.resume_container)
        res_cont_layout.setContentsMargins(0, 0, 0, 0)
        res_cont_layout.setSpacing(6)

        res_row = QHBoxLayout()
        self.combo_resume_ckpt = QComboBox()
        self.combo_resume_ckpt.currentIndexChanged.connect(self.on_resume_checkpoint_selected)
        res_row.addWidget(self.combo_resume_ckpt, stretch=1)

        btn_rescan_ckpts = QPushButton("🔄 Scan")
        btn_rescan_ckpts.clicked.connect(self.rescan_checkpoints)
        res_row.addWidget(btn_rescan_ckpts)

        btn_browse_ckpt = QPushButton("Browse...")
        btn_browse_ckpt.clicked.connect(self.on_browse_custom_checkpoint)
        res_row.addWidget(btn_browse_ckpt)
        res_cont_layout.addLayout(res_row)

        self.lbl_resume_info = QLabel("Select checkpoint to view prior training metrics.")
        self.lbl_resume_info.setStyleSheet("color: #f59e0b; font-size: 11px;")
        res_cont_layout.addWidget(self.lbl_resume_info)

        resume_layout.addWidget(self.resume_container)
        self.resume_container.setVisible(False)

        layout.addWidget(grp_resume)
        layout.addStretch()
        return widget

    # -------------------------------------------------------------------------
    # Tab 2: Core Optimization Hyperparameters
    # -------------------------------------------------------------------------
    def create_hyperparameters_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # Training Steps & Batch Size
        grp_steps = QGroupBox("OPTIMIZATION STEPS & BATCHING")
        grid_steps = QGridLayout(grp_steps)
        grid_steps.setSpacing(8)

        grid_steps.addWidget(QLabel("Training Steps:"), 0, 0)
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(100, 1000000)
        self.spin_steps.setSingleStep(1000)
        self.spin_steps.setValue(TRAINING_STEPS)
        grid_steps.addWidget(self.spin_steps, 0, 1)

        grid_steps.addWidget(QLabel("Batch Size:"), 0, 2)
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 256)
        self.spin_batch_size.setValue(BATCH_SIZE)
        grid_steps.addWidget(self.spin_batch_size, 0, 3)

        grid_steps.addWidget(QLabel("Optimizer:"), 1, 0)
        self.combo_optimizer = QComboBox()
        for opt in AVAILABLE_OPTIMIZERS:
            self.combo_optimizer.addItem(opt)
        grid_steps.addWidget(self.combo_optimizer, 1, 1)

        grid_steps.addWidget(QLabel("Warmup Steps:"), 1, 2)
        self.spin_warmup = QSpinBox()
        self.spin_warmup.setRange(0, 50000)
        self.spin_warmup.setSingleStep(250)
        self.spin_warmup.setValue(WARMUP_STEPS)
        grid_steps.addWidget(self.spin_warmup, 1, 3)

        layout.addWidget(grp_steps)

        # Learning Rates & Schedules
        grp_lr = QGroupBox("LEARNING RATES & REGULARIZATION")
        grid_lr = QGridLayout(grp_lr)
        grid_lr.setSpacing(8)

        grid_lr.addWidget(QLabel("Policy / Head LR:"), 0, 0)
        self.edit_lr = QLineEdit("1e-4")
        self.edit_lr.setToolTip("Learning rate for transformer attention & action heads")
        grid_lr.addWidget(self.edit_lr, 0, 1)

        grid_lr.addWidget(QLabel("Vision Backbone LR:"), 0, 2)
        self.edit_lr_backbone = QLineEdit("1e-5")
        self.edit_lr_backbone.setToolTip("Differential learning rate for pretrained ResNet visual backbone")
        grid_lr.addWidget(self.edit_lr_backbone, 0, 3)

        grid_lr.addWidget(QLabel("Minimum LR (Cosine):"), 1, 0)
        self.edit_lr_min = QLineEdit("1e-6")
        grid_lr.addWidget(self.edit_lr_min, 1, 1)

        grid_lr.addWidget(QLabel("Weight Decay (L2):"), 1, 2)
        self.edit_weight_decay = QLineEdit("1e-4")
        grid_lr.addWidget(self.edit_weight_decay, 1, 3)

        grid_lr.addWidget(QLabel("Grad Clip Norm:"), 2, 0)
        self.spin_grad_clip = QDoubleSpinBox()
        self.spin_grad_clip.setRange(0.0, 100.0)
        self.spin_grad_clip.setValue(GRAD_CLIP_NORM)
        grid_lr.addWidget(self.spin_grad_clip, 2, 1)

        layout.addWidget(grp_lr)

        # Checkpoint Frequencies & Augmentations
        grp_freq = QGroupBox("CHECKPOINTS & EVALUATION")
        grid_freq = QGridLayout(grp_freq)
        grid_freq.setSpacing(8)

        grid_freq.addWidget(QLabel("Save Checkpoint Every:"), 0, 0)
        self.spin_save_freq = QSpinBox()
        self.spin_save_freq.setRange(100, 100000)
        self.spin_save_freq.setSingleStep(500)
        self.spin_save_freq.setValue(SAVE_CHECKPOINT_FREQ)
        grid_freq.addWidget(self.spin_save_freq, 0, 1)

        grid_freq.addWidget(QLabel("Eval Val Loss Every:"), 0, 2)
        self.spin_eval_freq = QSpinBox()
        self.spin_eval_freq.setRange(100, 100000)
        self.spin_eval_freq.setSingleStep(500)
        self.spin_eval_freq.setValue(EVAL_FREQ)
        grid_freq.addWidget(self.spin_eval_freq, 0, 3)

        grid_freq.addWidget(QLabel("Validation Split Ratio:"), 1, 0)
        self.spin_val_ratio = QDoubleSpinBox()
        self.spin_val_ratio.setRange(0.0, 0.5)
        self.spin_val_ratio.setSingleStep(0.05)
        self.spin_val_ratio.setValue(VAL_SPLIT_RATIO)
        grid_freq.addWidget(self.spin_val_ratio, 1, 1)

        grid_freq.addWidget(QLabel("Console Log Every:"), 1, 2)
        self.spin_log_freq = QSpinBox()
        self.spin_log_freq.setRange(1, 500)
        self.spin_log_freq.setValue(LOG_FREQ)
        grid_freq.addWidget(self.spin_log_freq, 1, 3)

        layout.addWidget(grp_freq)

        # Real-time Visual Data Augmentation Group
        grp_aug = QGroupBox("REAL-TIME VISUAL DATA AUGMENTATION (GPU)")
        layout_aug = QVBoxLayout(grp_aug)
        layout_aug.setSpacing(10)

        self.chk_aug_master = QCheckBox("Enable Visual Data Augmentation (Master Switch)")
        self.chk_aug_master.setStyleSheet("font-weight: bold; color: #38bdf8;")
        self.chk_aug_master.setChecked(USE_AUGMENTATION)
        self.chk_aug = self.chk_aug_master  # Backwards compatibility alias
        layout_aug.addWidget(self.chk_aug_master)

        aug_sub_widget = QWidget()
        grid_sub_aug = QGridLayout(aug_sub_widget)
        grid_sub_aug.setContentsMargins(16, 2, 4, 4)
        grid_sub_aug.setSpacing(8)

        self.chk_aug_rotation = QCheckBox("Random Rotation (±10°)")
        self.chk_aug_rotation.setToolTip("Applies random rotations up to ±10 degrees directly on GPU tensors to simulate camera tilt and robot vibration.")
        self.chk_aug_rotation.setChecked(AUG_ROTATION)
        grid_sub_aug.addWidget(self.chk_aug_rotation, 0, 0)

        self.chk_aug_translation = QCheckBox("Random Translation / Shifts (±8% X/Y)")
        self.chk_aug_translation.setToolTip("Applies subtle 2D translations (±8% along X and Y) with border padding to make policy robust against robot mounting shifts (DrQ-v2 / RAD standard).")
        self.chk_aug_translation.setChecked(AUG_TRANSLATION)
        grid_sub_aug.addWidget(self.chk_aug_translation, 0, 1)

        self.chk_aug_color = QCheckBox("Lighting & Color Jitter (Brightness/Contrast ±20%)")
        self.chk_aug_color.setToolTip("Randomized brightness and contrast perturbations on GPU to prevent policy overfitting to ambient room lighting and shadows.")
        self.chk_aug_color.setChecked(AUG_COLOR)
        grid_sub_aug.addWidget(self.chk_aug_color, 1, 0)

        self.chk_aug_cutout = QCheckBox("Random Cutout / Occlusions (Dropout Patches)")
        self.chk_aug_cutout.setToolTip("Dropout random rectangular patches (5-15% area) to prevent policy relying on single visual landmarks and handle gripper/arm occlusions.")
        self.chk_aug_cutout.setChecked(AUG_CUTOUT)
        grid_sub_aug.addWidget(self.chk_aug_cutout, 1, 1)

        self.chk_aug_noise = QCheckBox("Sensor Noise / Gaussian Noise (Sensor Grain)")
        self.chk_aug_noise.setToolTip("Mild additive Gaussian noise and grain to simulate low-light sensor noise and sensor variation.")
        self.chk_aug_noise.setChecked(AUG_NOISE)
        grid_sub_aug.addWidget(self.chk_aug_noise, 2, 0, 1, 2)

        layout_aug.addWidget(aug_sub_widget)

        self.chk_aug_master.toggled.connect(aug_sub_widget.setEnabled)
        aug_sub_widget.setEnabled(self.chk_aug_master.isChecked())

        layout.addWidget(grp_aug)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(widget)
        return scroll

    # -------------------------------------------------------------------------
    # Tab 3: Hardware & Policy Tuning
    # -------------------------------------------------------------------------
    def create_advanced_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # Hardware & DataLoader
        grp_hw = QGroupBox("HARDWARE & EXECUTION ENGINE")
        grid_hw = QGridLayout(grp_hw)
        grid_hw.setSpacing(8)

        grid_hw.addWidget(QLabel("Compute Device:"), 0, 0)
        self.combo_device = QComboBox()
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            self.combo_device.addItem(f"CUDA:0 ({gpu_name})", "cuda")
        self.combo_device.addItem("CPU (Safe fallback)", "cpu")
        grid_hw.addWidget(self.combo_device, 0, 1)

        grid_hw.addWidget(QLabel("DataLoader Workers:"), 0, 2)
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 16)
        self.spin_workers.setValue(NUM_WORKERS)
        grid_hw.addWidget(self.spin_workers, 0, 3)

        grid_hw.addWidget(QLabel("Random Seed:"), 1, 0)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(SEED)
        grid_hw.addWidget(self.spin_seed, 1, 1)

        layout.addWidget(grp_hw)

        # Policy Architecture Hyperparameters (Horizon & Interval Tuning)
        grp_act = QGroupBox("POLICY HORIZON & TEMPORAL ENSEMBLING")
        grid_act = QGridLayout(grp_act)
        grid_act.setSpacing(8)

        grid_act.addWidget(QLabel("Chunk Size (Horizon Steps):"), 0, 0)
        self.spin_chunk_size = QSpinBox()
        self.spin_chunk_size.setRange(10, 300)
        self.spin_chunk_size.setValue(DEFAULT_CHUNK_SIZE)
        self.spin_chunk_size.setToolTip("Total future action steps predicted at each neural network snapshot")
        self.spin_chunk_size.valueChanged.connect(self.update_decision_interval)
        grid_act.addWidget(self.spin_chunk_size, 0, 1)

        grid_act.addWidget(QLabel("Action Steps (Execution Chunk):"), 0, 2)
        self.spin_n_action_steps = QSpinBox()
        self.spin_n_action_steps.setRange(1, 100)
        self.spin_n_action_steps.setValue(15)
        self.spin_n_action_steps.setToolTip("How many action steps to execute before taking a new camera observation (used when Temporal Ensembling is disabled)")
        self.spin_n_action_steps.valueChanged.connect(self.update_decision_interval)
        grid_act.addWidget(self.spin_n_action_steps, 0, 3)

        self.chk_temporal_ensemble = QCheckBox("Enable Temporal Ensembling (Smooth 30 Hz Moving Average for ACT)")
        self.chk_temporal_ensemble.setChecked(True)
        self.chk_temporal_ensemble.setStyleSheet("font-weight: bold; color: #38bdf8;")
        self.chk_temporal_ensemble.setToolTip("ACT Algorithm 2: Continuously queries policy at 30 Hz and blends overlapping chunks via exponential decay weights w_i = exp(-m*i). Eliminates open-loop pauses and boundary jerks.")
        self.chk_temporal_ensemble.toggled.connect(self.on_temporal_ensemble_toggled)
        self.spin_n_action_steps.setEnabled(False)
        grid_act.addWidget(self.chk_temporal_ensemble, 1, 0, 1, 2)

        te_box = QHBoxLayout()
        lbl_coeff = QLabel("Ensemble Coeff (m):")
        lbl_coeff.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        te_box.addWidget(lbl_coeff)
        self.spin_temporal_coeff = QDoubleSpinBox()
        self.spin_temporal_coeff.setRange(0.001, 0.200)
        self.spin_temporal_coeff.setSingleStep(0.005)
        self.spin_temporal_coeff.setValue(0.010)
        self.spin_temporal_coeff.setDecimals(3)
        self.spin_temporal_coeff.setToolTip("Exponential weight decay coefficient (ACT default: 0.010). Lower values average more smoothly across time.")
        self.spin_temporal_coeff.valueChanged.connect(self.update_decision_interval)
        te_box.addWidget(self.spin_temporal_coeff)
        grid_act.addLayout(te_box, 1, 2, 1, 2)

        grid_act.addWidget(QLabel("KL Divergence Weight:"), 2, 0)
        self.spin_kl_weight = QDoubleSpinBox()
        self.spin_kl_weight.setRange(0.1, 100.0)
        self.spin_kl_weight.setValue(10.0)
        grid_act.addWidget(self.spin_kl_weight, 2, 1)

        grid_act.addWidget(QLabel("Transformer Dropout:"), 2, 2)
        self.spin_dropout = QDoubleSpinBox()
        self.spin_dropout.setRange(0.0, 0.5)
        self.spin_dropout.setSingleStep(0.05)
        self.spin_dropout.setValue(0.1)
        grid_act.addWidget(self.spin_dropout, 2, 3)

        self.lbl_decision_interval = QLabel("Replanning Mode: Temporal Ensembling (m=0.010, 30 Hz Continuous Smooth Blend)")
        self.lbl_decision_interval.setStyleSheet("color: #38bdf8; font-family: monospace; font-weight: bold; font-size: 11px; padding: 4px;")
        grid_act.addWidget(self.lbl_decision_interval, 3, 0, 1, 4)

        layout.addWidget(grp_act)
        layout.addStretch()
        return widget

    # -------------------------------------------------------------------------
    # Top Action Bar
    # -------------------------------------------------------------------------
    def create_action_bar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 6px;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        # Primary Start / Stop Button
        self.btn_train = QPushButton("▶  START TRAINING")
        self.btn_train.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px 24px;
                border: 1px solid #22c55e;
            }
            QPushButton:hover {
                background-color: #15803d;
            }
        """)
        self.btn_train.clicked.connect(self.toggle_training)
        layout.addWidget(self.btn_train)

        # Open in Browser Button
        self.btn_browser = QPushButton("🌐  View in Browser")
        self.btn_browser.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
                padding: 10px 18px;
                border-radius: 6px;
                border: 1px solid #38bdf8;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
        """)
        self.btn_browser.setToolTip("Open live interactive HTML dashboard with Chart.js loss curves in your web browser")
        self.btn_browser.clicked.connect(self.open_dashboard_in_browser)
        layout.addWidget(self.btn_browser)

        # Test in GUI Tester Button
        self.btn_test_policy = QPushButton("🚀  Test in GUI Tester")
        self.btn_test_policy.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
                padding: 10px 18px;
                border-radius: 6px;
                border: 1px solid #10b981;
            }
            QPushButton:hover {
                background-color: #047857;
            }
        """)
        self.btn_test_policy.setToolTip("Launch gui_tester.py pre-loaded with this checkpoint")
        self.btn_test_policy.clicked.connect(self.launch_gui_tester)
        layout.addWidget(self.btn_test_policy)

        # Open Folder in File Manager
        self.btn_open_folder = QPushButton("📁 Folder")
        self.btn_open_folder.setToolTip("Open output directory in Linux file manager")
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        layout.addWidget(self.btn_open_folder)

        layout.addStretch()
        return bar

    # -------------------------------------------------------------------------
    # Progress & KPI Summary Card
    # -------------------------------------------------------------------------
    def create_kpi_card(self) -> QWidget:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Top line: Step count + Status badge
        top_line = QHBoxLayout()
        self.lbl_progress_text = QLabel("Step: 0 / 15,000 (0.0%)")
        self.lbl_progress_text.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        top_line.addWidget(self.lbl_progress_text)

        top_line.addStretch()

        self.lbl_status_badge = QLabel("IDLE")
        self.lbl_status_badge.setStyleSheet("""
            background-color: #334155;
            color: #94a3b8;
            font-weight: bold;
            font-size: 11px;
            padding: 3px 10px;
            border-radius: 10px;
        """)
        top_line.addWidget(self.lbl_status_badge)
        layout.addLayout(top_line)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # Metrics grid
        grid = QGridLayout()
        grid.setSpacing(12)

        def make_kpi_box(title: str, default_val: str, val_color: str = "#ffffff"):
            box = QFrame()
            box.setStyleSheet("background-color: #1e293b; border-radius: 6px; padding: 6px;")
            b_layout = QVBoxLayout(box)
            b_layout.setContentsMargins(6, 4, 6, 4)
            b_layout.setSpacing(2)
            lbl_t = QLabel(title.upper())
            lbl_t.setStyleSheet("font-size: 10px; color: #94a3b8; font-weight: 600;")
            lbl_v = QLabel(default_val)
            lbl_v.setStyleSheet(f"font-size: 18px; font-weight: 800; color: {val_color}; font-family: 'Consolas', monospace;")
            b_layout.addWidget(lbl_t)
            b_layout.addWidget(lbl_v)
            return box, lbl_v

        self.box_t_loss, self.lbl_val_train_loss = make_kpi_box("Train Loss", "--", "#38bdf8")
        self.box_v_loss, self.lbl_val_val_loss = make_kpi_box("Val Loss", "--", "#f59e0b")
        self.box_speed, self.lbl_val_speed = make_kpi_box("Speed", "0.0 st/s", "#ffffff")
        self.box_eta, self.lbl_val_eta = make_kpi_box("ETA (Elapsed)", "--:--", "#ffffff")

        grid.addWidget(self.box_t_loss, 0, 0)
        grid.addWidget(self.box_v_loss, 0, 1)
        grid.addWidget(self.box_speed, 0, 2)
        grid.addWidget(self.box_eta, 0, 3)

        layout.addLayout(grid)
        return card

    # -------------------------------------------------------------------------
    # Dataset Scanning & Path Helpers
    # -------------------------------------------------------------------------
    def rescan_datasets(self):
        self.combo_dataset.blockSignals(True)
        self.combo_dataset.clear()

        found = []
        if self.datasets_dir.exists():
            for p in sorted(self.datasets_dir.iterdir()):
                if p.is_dir() and (p / "meta" / "info.json").exists():
                    found.append(p)

        for p in found:
            self.combo_dataset.addItem(f"📁 {p.name}", str(p))

        self.combo_dataset.blockSignals(False)

        if self.combo_dataset.count() > 0:
            self.combo_dataset.setCurrentIndex(0)
            self.on_dataset_selection_changed(0)
        else:
            self.lbl_dataset_meta.setText("⚠ No valid LeRobot datasets found in datasets/ directory.")

    def rescan_checkpoints(self):
        self.combo_resume_ckpt.blockSignals(True)
        self.combo_resume_ckpt.clear()

        found = []
        if self.outputs_dir.exists():
            for model_dir in sorted(self.outputs_dir.iterdir()):
                best_dir = model_dir / "best_checkpoint"
                if best_dir.exists() and (best_dir / "config.json").exists():
                    found.append((f"{model_dir.name} [BEST VAL]", best_dir))
                ckpt_dir = model_dir / "checkpoint"
                if ckpt_dir.exists() and (ckpt_dir / "config.json").exists():
                    found.append((model_dir.name, ckpt_dir))
                steps_dir = model_dir / "checkpoints"
                if steps_dir.exists() and steps_dir.is_dir():
                    for step_sub in sorted(steps_dir.iterdir()):
                        if step_sub.is_dir() and (step_sub / "config.json").exists():
                            found.append((f"{model_dir.name}/{step_sub.name}", step_sub))

        for name, ckpt in found:
            self.combo_resume_ckpt.addItem(f"💾 {name}", str(ckpt))

        self.combo_resume_ckpt.blockSignals(False)
        if self.combo_resume_ckpt.count() > 0:
            self.on_resume_checkpoint_selected(0)
        else:
            self.lbl_resume_info.setText("No previous checkpoints found in outputs/train/.")

    def on_temporal_ensemble_toggled(self, checked: bool):
        if hasattr(self, "spin_temporal_coeff"):
            self.spin_temporal_coeff.setEnabled(checked)
        if hasattr(self, "spin_n_action_steps"):
            self.spin_n_action_steps.setEnabled(not checked)
        self.update_decision_interval()

    def update_decision_interval(self):
        fps = getattr(self, "dataset_fps", 30)
        chunk = self.spin_chunk_size.value() if hasattr(self, "spin_chunk_size") else 100
        if hasattr(self, "chk_temporal_ensemble") and self.chk_temporal_ensemble.isChecked():
            m_coeff = self.spin_temporal_coeff.value() if hasattr(self, "spin_temporal_coeff") else 0.010
            if hasattr(self, "lbl_decision_interval"):
                self.lbl_decision_interval.setText(
                    f"Replanning Mode: Temporal Ensembling (m={m_coeff:.3f}) | Smooth 30 Hz Moving Average (Horizon={chunk})"
                )
                self.lbl_decision_interval.setStyleSheet("color: #38bdf8; font-family: monospace; font-weight: bold; font-size: 11px; padding: 4px;")
        else:
            steps = self.spin_n_action_steps.value() if hasattr(self, "spin_n_action_steps") else 15
            dt = steps / max(1, fps)
            if hasattr(self, "lbl_decision_interval"):
                self.lbl_decision_interval.setText(
                    f"Replanning Mode: Open-Loop Chunks every {steps} steps ({dt:.2f}s / {int(dt*1000)}ms @ {fps} FPS) of {chunk}"
                )
                self.lbl_decision_interval.setStyleSheet("color: #f59e0b; font-family: monospace; font-weight: bold; font-size: 11px; padding: 4px;")

    def on_grayscale_toggled(self, checked: bool):
        if checked:
            self.lbl_channel_mode.setText("● Grayscale Mode (1-Ch GPU Filter)")
            self.lbl_channel_mode.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px; padding: 2px 4px;")
        else:
            self.lbl_channel_mode.setText("● RGB Color Mode (3-Ch Full Spectrum)")
            self.lbl_channel_mode.setStyleSheet("color: #a855f7; font-weight: bold; font-size: 11px; padding: 2px 4px;")

        if not self._user_edited_name:
            self.update_auto_model_name()
        else:
            cur = self.edit_model_name.text().strip()
            if cur.endswith("_col") and checked:
                new_n = cur[:-4] + "_gs"
                self.edit_model_name.setText(new_n)
                self.update_output_directory(new_n)
            elif cur.endswith("_gs") and not checked:
                new_n = cur[:-3] + "_col"
                self.edit_model_name.setText(new_n)
                self.update_output_directory(new_n)

    def on_dataset_selection_changed(self, index: int):
        ds_path_str = self.combo_dataset.currentData()
        if not ds_path_str:
            return
        ds_path = Path(ds_path_str)
        info_file = ds_path / "meta" / "info.json"

        meta_text = f"Dataset: {ds_path.name}"
        if info_file.exists():
            try:
                with open(info_file, "r") as f:
                    data = json.load(f)
                eps = data.get("total_episodes", "?")
                frames = data.get("total_frames", "?")
                fps = data.get("fps", 30)
                self.dataset_fps = int(fps)
                self.update_decision_interval()
                meta_text = f"✔ {eps} Episodes | {frames:,} Total Frames | {fps} FPS | Root: {ds_path.name}"
            except Exception as e:
                meta_text = f"✔ {ds_path.name} (Metadata: {e})"

        self.lbl_dataset_meta.setText(meta_text)

        # Inspect if dataset is legacy grayscale
        is_gray = False
        for cfg_p in [ds_path / "meta" / "recording_config.json", ds_path / "recording_config.json"]:
            if cfg_p.exists():
                try:
                    with open(cfg_p, "r") as cf:
                        cfg = json.load(cf)
                        if cfg.get("grayscale") is True:
                            is_gray = True
                            break
                except Exception:
                    pass

        if hasattr(self, "lbl_channel_mode") and hasattr(self, "chk_grayscale"):
            self.chk_grayscale.blockSignals(True)
            self.chk_grayscale.setChecked(is_gray)
            self.chk_grayscale.blockSignals(False)
            if is_gray:
                self.lbl_channel_mode.setText("● Grayscale Mode (1-Ch)")
                self.lbl_channel_mode.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px; padding: 2px 4px;")
            else:
                self.lbl_channel_mode.setText("● RGB Color Mode (3-Ch)")
                self.lbl_channel_mode.setStyleSheet("color: #a855f7; font-weight: bold; font-size: 11px; padding: 2px 4px;")

        if not self._user_edited_name:
            self.update_auto_model_name()

    def on_policy_selection_changed(self, index: int):
        pol = self.combo_policy.currentData()
        if pol == "act":
            self.lbl_policy_desc.setText("ACT: Action Chunking with Transformers. Best for precise UR5e manipulation.")
        elif pol == "diffusion":
            self.lbl_policy_desc.setText("Diffusion Policy: Generates multi-modal actions via iterative denoising.")
        elif pol == "vqbet":
            self.lbl_policy_desc.setText("VQ-BeT: Vector-Quantized Behavior Transformer for complex long-horizon tasks.")

        if not self._user_edited_name:
            self.update_auto_model_name()

    def update_auto_model_name(self):
        if self.chk_resume.isChecked() and self.combo_resume_ckpt.currentData():
            ckpt_p = Path(self.combo_resume_ckpt.currentData())
            existing_name = ckpt_p.parent.name
            self.edit_model_name.setText(existing_name)
            self.update_output_directory(existing_name)
            return

        ds_path_str = self.combo_dataset.currentData()
        if ds_path_str:
            ds_name = Path(ds_path_str).name
        else:
            ds_name = "model"
        pol = self.combo_policy.currentData() or "act"
        color_tag = "gs" if (hasattr(self, "chk_grayscale") and self.chk_grayscale.isChecked()) else "col"

        default_name = f"{ds_name}_{pol}_{color_tag}"
        self.edit_model_name.setText(default_name)
        self.update_output_directory(default_name)

    def on_model_name_manually_edited(self, text: str):
        self._user_edited_name = bool(text.strip())
        self.update_output_directory(text.strip())

    def reset_default_model_name(self):
        self._user_edited_name = False
        self.update_auto_model_name()

    def update_output_directory(self, model_name: str):
        if not model_name:
            model_name = "default_model"
        target_dir = self.outputs_dir / model_name
        self.edit_output_dir.setText(str(target_dir))

        # Check if loss plot already exists
        loss_p = target_dir / "loss_curve.png"
        if loss_p.exists():
            self.loss_plot_widget.set_image(loss_p)

    def on_browse_custom_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select LeRobot Dataset Folder", str(self.datasets_dir))
        if folder:
            p = Path(folder)
            self.combo_dataset.addItem(f"📁 {p.name} (Custom)", str(p))
            self.combo_dataset.setCurrentIndex(self.combo_dataset.count() - 1)

    def on_browse_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", str(self.outputs_dir))
        if folder:
            self.edit_output_dir.setText(folder)
            self._user_edited_name = True

    def on_resume_toggled(self, checked: bool):
        self.resume_container.setVisible(checked)
        if checked:
            self.rescan_checkpoints()
            if self.combo_resume_ckpt.currentData():
                self.on_resume_checkpoint_selected(self.combo_resume_ckpt.currentIndex())
        else:
            self._user_edited_name = False
            self.update_auto_model_name()

    def on_resume_checkpoint_selected(self, index: int):
        ckpt_str = self.combo_resume_ckpt.currentData()
        if not ckpt_str:
            return
        ckpt_p = Path(ckpt_str)
        existing_model_name = ckpt_p.parent.name

        # When continuing training, reuse the exact existing model folder
        self.edit_model_name.blockSignals(True)
        self.edit_model_name.setText(existing_model_name)
        self.edit_model_name.blockSignals(False)
        self.update_output_directory(existing_model_name)

        # Pre-load existing loss curve if present
        existing_plot = ckpt_p.parent / "loss_curve.png"
        if existing_plot.exists():
            self.loss_plot_widget.set_image(existing_plot)

        info_f = ckpt_p / "training_info.json"
        if info_f.exists():
            try:
                with open(info_f, "r") as f:
                    data = json.load(f)
                step = data.get("step", 0)
                tot = data.get("training_steps", step)
                pol = data.get("policy_type", "unknown")
                self.lbl_resume_info.setText(
                    f"✔ Checkpoint: {existing_model_name} | Step {step:,} / {tot:,} | Policy: {pol.upper()}"
                )
                # Auto-align policy architecture dropdown with checkpoint
                if pol:
                    idx = self.combo_policy.findData(pol.lower())
                    if idx >= 0:
                        self.combo_policy.blockSignals(True)
                        self.combo_policy.setCurrentIndex(idx)
                        self.combo_policy.blockSignals(False)
            except Exception as e:
                self.lbl_resume_info.setText(f"Checkpoint info: {e}")
        else:
            self.lbl_resume_info.setText(f"Found model weights at: {ckpt_p.name}")

    def on_browse_custom_checkpoint(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Checkpoint Directory", str(self.outputs_dir))
        if folder:
            p = Path(folder)
            self.combo_resume_ckpt.addItem(f"💾 {p.name} (Custom)", str(p))
            self.combo_resume_ckpt.setCurrentIndex(self.combo_resume_ckpt.count() - 1)

    # -------------------------------------------------------------------------
    # Training Execution & Process Lifecycle
    # -------------------------------------------------------------------------
    def toggle_training(self):
        if self.is_training_active:
            self.stop_training()
        else:
            self.start_training()

    def start_training(self):
        if not LEROBOT_AVAILABLE:
            QMessageBox.critical(self, "LeRobot Missing", f"LeRobot is not installed:\n{LEROBOT_IMPORT_ERROR}")
            return

        ds_path = self.combo_dataset.currentData()
        if not ds_path or not Path(ds_path).exists():
            QMessageBox.warning(self, "Invalid Dataset", "Please select a valid dataset folder first.")
            return

        out_dir = Path(self.edit_output_dir.text()).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Parse learning rates
        try:
            lr = float(self.edit_lr.text())
            lr_backbone = float(self.edit_lr_backbone.text())
            lr_min = float(self.edit_lr_min.text())
            weight_decay = float(self.edit_weight_decay.text())
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Hyperparameter", f"Could not parse numeric rate: {e}")
            return

        # Prepare kwargs
        use_te = hasattr(self, "chk_temporal_ensemble") and self.chk_temporal_ensemble.isChecked()
        te_coeff = self.spin_temporal_coeff.value() if use_te else None
        n_act_steps = 1 if use_te else self.spin_n_action_steps.value()

        trainer_kwargs = {
            "dataset_path": ds_path,
            "policy_name": self.combo_policy.currentData(),
            "optimizer_name": self.combo_optimizer.currentText(),
            "training_steps": self.spin_steps.value(),
            "batch_size": self.spin_batch_size.value(),
            "learning_rate": lr,
            "lr_backbone": lr_backbone,
            "lr_min": lr_min,
            "warmup_steps": self.spin_warmup.value(),
            "weight_decay": weight_decay,
            "grad_clip_norm": self.spin_grad_clip.value(),
            "use_augmentation": self.chk_aug_master.isChecked(),
            "aug_rotation": self.chk_aug_rotation.isChecked(),
            "aug_translation": self.chk_aug_translation.isChecked(),
            "aug_color": self.chk_aug_color.isChecked(),
            "aug_cutout": self.chk_aug_cutout.isChecked(),
            "aug_noise": self.chk_aug_noise.isChecked(),
            "save_checkpoint_freq": self.spin_save_freq.value(),
            "log_freq": self.spin_log_freq.value(),
            "eval_freq": self.spin_eval_freq.value(),
            "val_split_ratio": self.spin_val_ratio.value(),
            "output_dir": str(out_dir),
            "device": self.combo_device.currentData(),
            "num_workers": self.spin_workers.value(),
            "seed": self.spin_seed.value(),
            "chunk_size": self.spin_chunk_size.value(),
            "n_action_steps": n_act_steps,
            "kl_weight": self.spin_kl_weight.value(),
            "dropout": self.spin_dropout.value(),
            "grayscale": self.chk_grayscale.isChecked(),
            "temporal_ensemble_coeff": te_coeff,
        }

        if self.chk_resume.isChecked():
            ckpt = self.combo_resume_ckpt.currentData()
            if ckpt:
                trainer_kwargs["resume_checkpoint"] = ckpt

        # Update UI state to Training
        self.is_training_active = True
        self.training_start_time = time.time()
        self.btn_train.setText("⏹  STOP / ABORT TRAINING")
        self.btn_train.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px 24px;
                border: 1px solid #ef4444;
            }
            QPushButton:hover {
                background-color: #b91c1c;
            }
        """)
        self.lbl_status_badge.setText("TRAINING")
        self.lbl_status_badge.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; font-size: 11px; padding: 3px 10px; border-radius: 10px;")
        self.status_bar.showMessage(f"Training active for {out_dir.name}...")

        # Start thread
        self.training_worker = TrainingWorker(trainer_kwargs)
        self.training_worker.step_progress.connect(self.on_training_step_progress)
        self.training_worker.plot_updated.connect(self.loss_plot_widget.set_image)
        self.training_worker.training_finished.connect(self.on_training_finished)
        self.training_worker.start()

    def stop_training(self):
        if self.training_worker and self.training_worker.isRunning():
            self.status_bar.showMessage("Aborting training... saving safe emergency checkpoint...")
            self.training_worker.abort()
            self.btn_train.setEnabled(False)

    def on_training_step_progress(self, metrics: Dict[str, Any]):
        step = metrics.get("step", 0)
        tot = metrics.get("total_steps", 1)
        t_loss = metrics.get("train_loss", 0.0)
        v_loss = metrics.get("val_loss")
        speed = metrics.get("steps_per_sec", 0.0)
        eta_s = metrics.get("eta_sec", 0.0)

        # Update progress bar
        pct = min(100.0, (step / max(1, tot)) * 100.0)
        self.progress_bar.setValue(int(pct * 10))
        self.lbl_progress_text.setText(f"Step: {step:,} / {tot:,} ({pct:.1f}%)")

        # Update KPI cards
        self.lbl_val_train_loss.setText(f"{t_loss:.4f}")
        self.lbl_val_val_loss.setText(f"{v_loss:.4f}" if v_loss is not None else "--")
        self.lbl_val_speed.setText(f"{speed:.1f} st/s")

        elapsed_s = time.time() - self.training_start_time
        el_str = str(datetime.timedelta(seconds=int(elapsed_s)))
        eta_str = str(datetime.timedelta(seconds=int(eta_s))) if eta_s > 0 else "00:00"
        self.lbl_val_eta.setText(f"{eta_str} ({el_str})")

        # Auto-update loss plot every checkpoint save or eval
        out_dir = Path(self.edit_output_dir.text())
        plot_p = out_dir / "loss_curve.png"
        if plot_p.exists() and (step % self.spin_eval_freq.value() == 0 or step % self.spin_save_freq.value() == 0):
            self.loss_plot_widget.set_image(plot_p)

    def on_training_finished(self, success: bool, message: str, results: Dict[str, Any]):
        self.is_training_active = False
        self.btn_train.setEnabled(True)
        self.btn_train.setText("▶  START TRAINING")
        self.btn_train.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px 24px;
                border: 1px solid #22c55e;
            }
            QPushButton:hover {
                background-color: #15803d;
            }
        """)

        badge_text = "COMPLETED" if success else "STOPPED"
        badge_bg = "#10b981" if success else "#f59e0b"
        self.lbl_status_badge.setText(badge_text)
        self.lbl_status_badge.setStyleSheet(f"background-color: {badge_bg}; color: #ffffff; font-weight: bold; font-size: 11px; padding: 3px 10px; border-radius: 10px;")

        self.status_bar.showMessage(message, 8000)

        # Update final loss curve
        out_dir = Path(self.edit_output_dir.text())
        plot_p = out_dir / "loss_curve.png"
        if plot_p.exists():
            self.loss_plot_widget.set_image(plot_p)

        self.rescan_checkpoints()

        if success:
            QMessageBox.information(
                self,
                "Training Complete",
                f"Policy training completed successfully!\n\n"
                f"• Checkpoint: {out_dir / 'checkpoint'}\n"
                f"• Metrics CSV: {out_dir / 'train_log.csv'}\n"
                f"• HTML Dashboard: {out_dir / 'training_dashboard.html'}\n\n"
                f"Click 'View in Browser' or 'Test in GUI Tester' to evaluate the model.",
            )

    # -------------------------------------------------------------------------
    # Browser Dashboard & External Integrations
    # -------------------------------------------------------------------------
    def open_dashboard_in_browser(self):
        out_dir = Path(self.edit_output_dir.text())
        html_file = out_dir / "training_dashboard.html"

        if not html_file.exists():
            # Check if any other checkpoint in outputs has one
            all_dashboards = list(self.outputs_dir.glob("*/training_dashboard.html"))
            if all_dashboards:
                html_file = all_dashboards[0]

        if html_file.exists():
            url = html_file.resolve().as_uri()
            print(f"[DASHBOARD] Opening interactive web dashboard: {url}")
            webbrowser.open(url)
            self.status_bar.showMessage(f"Opened dashboard in browser: {html_file.name}", 4000)
        else:
            QMessageBox.information(
                self,
                "Dashboard Not Found",
                f"No training_dashboard.html generated yet for {out_dir.name}.\n"
                "It will be created automatically once training starts and the first checkpoint is saved.",
            )

    def launch_gui_tester(self):
        out_dir = Path(self.edit_output_dir.text())
        ckpt_dir = out_dir / "checkpoint"
        if not ckpt_dir.exists():
            # Search available checkpoints
            ckpts = list(self.outputs_dir.glob("*/checkpoint"))
            if ckpts:
                ckpt_dir = ckpts[0]
            else:
                QMessageBox.warning(self, "No Checkpoint", "No trained checkpoint found to test.")
                return

        tester_script = self.project_root / "gui_tester.py"
        if not tester_script.exists():
            QMessageBox.critical(self, "Tester Not Found", f"Cannot find gui_tester.py at {tester_script}")
            return

        python_bin = sys.executable
        print(f"[LAUNCH] Launching GUI Tester with model: {ckpt_dir.parent.name}")
        subprocess.Popen([python_bin, str(tester_script)])
        self.status_bar.showMessage("Launched Policy Testing Studio.", 4000)

    def open_output_folder(self):
        out_dir = Path(self.edit_output_dir.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(out_dir)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(out_dir)])
        elif sys.platform == "win32":
            os.startfile(str(out_dir))

    # -------------------------------------------------------------------------
    # Hardware Telemetry & Logging
    # -------------------------------------------------------------------------
    def update_gpu_telemetry(self):
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.lbl_gpu_stats.setText(f"GPU VRAM: {allocated:.2f} GB / {total:.1f} GB")
            except Exception:
                self.lbl_gpu_stats.setText("GPU: Active (CUDA)")
        else:
            self.lbl_gpu_stats.setText("CPU Execution Mode")

    def append_log_message(self, message: str):
        # Colorize log text based on severity tags
        clean = message.rstrip()
        if not clean:
            return

        if "[ERROR]" in clean or "CRITICAL" in clean or "Exception" in clean:
            color = "#ef4444"
        elif "[WARNING]" in clean or "WARN" in clean:
            color = "#f59e0b"
        elif "[CHECKPOINT]" in clean or "[SUCCESS]" in clean:
            color = "#10b981"
        elif "[Step" in clean:
            color = "#38bdf8"
        elif "[DASHBOARD]" in clean or "[POLICY]" in clean or "[OPTIMIZER]" in clean:
            color = "#a78bfa"
        else:
            color = "#cbd5e1"

        self.console_output.append(f'<span style="color: {color};">{clean}</span>')

        if self.chk_autoscroll.isChecked():
            cursor = self.console_output.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.console_output.setTextCursor(cursor)

    def closeEvent(self, event):
        if self.is_training_active:
            ans = QMessageBox.question(
                self,
                "Confirm Exit",
                "Training is currently running! Are you sure you want to stop training and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.Yes:
                if self.training_worker:
                    self.training_worker.abort()
                    self.training_worker.wait(1500)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# =============================================================================
# 5. Application Entrypoint
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("UR5e Policy Training Studio")

    studio = UR5eTrainerStudio()
    studio.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
