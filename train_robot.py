#!/usr/bin/env python3
"""
UR5e LeRobot Local Demonstration Training Script
=================================================
Trains imitation learning policies (ACT, Diffusion Policy, VQ-BeT) on locally recorded
UR5e demonstration datasets using Hugging Face's LeRobot framework.

Designed to be simple and straightforward to configure via top-of-file constants or CLI,
while modular enough to be seamlessly wrapped by a graphical user interface (GUI).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Try importing LeRobot components
try:
    import lerobot
    import lerobot.utils.import_utils
    # Force PyAV backend to avoid conda torchcodec C++ glib symbol mismatch
    lerobot.utils.import_utils.get_safe_default_video_backend = lambda: "pyav"

    from lerobot.configs import FeatureType
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTConfig, ACTPolicy
    from lerobot.utils.feature_utils import dataset_to_policy_features
    LEROBOT_AVAILABLE = True
    LEROBOT_IMPORT_ERROR = None
except ImportError as e:
    LEROBOT_AVAILABLE = False
    LEROBOT_IMPORT_ERROR = str(e)


# =============================================================================
# 1. User Configuration & Hyperparameters (Adjust freely here)
# =============================================================================

PROJECT_NAME = "move_to_red_cube"

# Path to local dataset folder (relative to this script or absolute)
DATASET_PATH = "datasets/" + PROJECT_NAME

# Available Policy Architectures
AVAILABLE_POLICIES = [
    "act",        # [0] Recommended: Action Chunking with Transformers (fast, handles multi-modal arm tasks)
    "diffusion",  # [1] Diffusion Policy (requires 'diffusers' package: pip install diffusers)
    "vqbet",      # [2] Vector-Quantized Behavior Transformer
]
SELECTED_POLICY_INDEX = 0  # <--- Select policy by index (0: ACT, 1: Diffusion, 2: VQ-BeT)

# Available Optimizers
AVAILABLE_OPTIMIZERS = [
    "AdamW",      # [0] Standard for Transformers / ACT (adaptive with decoupled weight decay)
    "Adam",       # [1] Standard adaptive optimizer
    "SGD",        # [2] Stochastic Gradient Descent with momentum
]
SELECTED_OPTIMIZER_INDEX = 0  # <--- Select optimizer by index (0: AdamW, 1: Adam, 2: SGD)

# Training Hyperparameters
TRAINING_STEPS = 15000      # Total optimization steps (increased for robust task mastery)
BATCH_SIZE = 32               # Samples per training batch (RTX 3080 16GB handles 16-64 easily)
LEARNING_RATE = 1e-4          # Initial optimizer learning rate for transformer heads
LR_BACKBONE = 1e-5            # Differential learning rate for pretrained ResNet vision backbone
LR_MIN = 1e-6                 # Minimum learning rate at end of Cosine Annealing schedule
WARMUP_STEPS = 1000           # Linear warmup steps to stabilize transformer attention layers
WEIGHT_DECAY = 1e-4           # L2 regularization weight decay
GRAD_CLIP_NORM = 10.0         # Gradient clipping max norm to prevent exploding gradients
USE_AUGMENTATION = True       # Apply mild visual jitter to prevent overfitting to lighting

# Checkpoint & Disk Space Management
# Overwrites the same folder every save to prevent gigabytes of redundant copies
SAVE_CHECKPOINT_FREQ = 2500   # Save policy weights every N steps
OUTPUT_DIR = "outputs/train/" + PROJECT_NAME  # Directory for model checkpoint, logs, and plots

# Validation & Evaluation
VAL_SPLIT_RATIO = 0.1         # Reserve 10% of frames/episodes for validation
EVAL_FREQ = 1000              # Compute validation loss every N steps

# Logging & Hardware
LOG_FREQ = 20                 # Console progress print interval (in steps)
NUM_WORKERS = 4               # DataLoader background subprocess workers (0 for single-thread)
SEED = 42                     # Random seed for reproducibility
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 2. Helper Functions: Timestamps, Plotting & Logging
# =============================================================================

def set_seed(seed: int):
    """Sets random seeds across Python, NumPy, and PyTorch for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """Converts relative frame delta indices into floating-point second offsets."""
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]


def plot_loss_curve(
    log_file: Path,
    output_png: Path,
    policy_name: str,
    steps_data: List[int],
    train_loss_data: List[float],
    val_loss_data: List[Tuple[int, float]],
):
    """
    Renders the training & validation loss curves to a PNG image.
    Prefers Matplotlib if installed; otherwise falls back cleanly to a built-in
    OpenCV/PIL chart renderer so no extra package installation is mandatory.
    """
    if not steps_data or not train_loss_data:
        return

    # Option A: Use Matplotlib if available
    try:
        import matplotlib
        matplotlib.use("Agg")  # Headless backend
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 5), dpi=120)
        plt.style.use("dark_background")

        plt.plot(steps_data, train_loss_data, label="Train Loss", color="#38bdf8", alpha=0.85, linewidth=1.8)

        if val_loss_data:
            val_steps, val_losses = zip(*val_loss_data)
            plt.plot(val_steps, val_losses, label="Val Loss", color="#f59e0b", marker="o", markersize=4, linewidth=2.0)

        plt.title(f"UR5e Imitation Learning — {policy_name.upper()} Policy Training", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Optimization Step", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.3, color="#475569")
        plt.legend(loc="upper right", framealpha=0.8)
        plt.tight_layout()
        plt.savefig(output_png)
        plt.close()
        return
    except ImportError:
        pass

    # Option B: Built-in OpenCV fallback renderer (always available in current environment)
    try:
        import cv2

        w, h = 900, 500
        canvas = np.full((h, w, 3), (30, 24, 15), dtype=np.uint8)  # Dark slate background #0f172a

        margin_left, margin_right = 80, 50
        margin_top, margin_bottom = 70, 70
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom

        # Filter valid finite losses
        valid_losses = [l for l in train_loss_data if np.isfinite(l)]
        if not valid_losses:
            return

        min_loss = max(0.0, min(valid_losses) * 0.9)
        max_loss = max(valid_losses) * 1.1 if max(valid_losses) > min_loss else min_loss + 1.0
        max_step = max(steps_data) if max(steps_data) > 0 else 1

        # Draw grid
        for i in range(5):
            gy = int(margin_top + i * (plot_h / 4))
            val_y = max_loss - i * ((max_loss - min_loss) / 4)
            cv2.line(canvas, (margin_left, gy), (w - margin_right, gy), (50, 45, 35), 1)
            cv2.putText(canvas, f"{val_y:.2f}", (10, gy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        # Draw Axes
        cv2.line(canvas, (margin_left, margin_top), (margin_left, h - margin_bottom), (180, 180, 180), 2)
        cv2.line(canvas, (margin_left, h - margin_bottom), (w - margin_right, h - margin_bottom), (180, 180, 180), 2)

        # Plot training loss line
        pts = []
        for step, l in zip(steps_data, train_loss_data):
            if not np.isfinite(l):
                continue
            px = int(margin_left + (step / max_step) * plot_w)
            norm_y = (l - min_loss) / (max_loss - min_loss)
            py = int((h - margin_bottom) - np.clip(norm_y, 0.0, 1.0) * plot_h)
            pts.append([px, py])

        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (248, 189, 56), 2, cv2.LINE_AA)

        # Plot validation points if present
        for v_step, v_loss in val_loss_data:
            if not np.isfinite(v_loss):
                continue
            px = int(margin_left + (v_step / max_step) * plot_w)
            norm_y = (v_loss - min_loss) / (max_loss - min_loss)
            py = int((h - margin_bottom) - np.clip(norm_y, 0.0, 1.0) * plot_h)
            cv2.circle(canvas, (px, py), 5, (11, 158, 245), -1, cv2.LINE_AA)

        # Labels & Header
        title = f"UR5e Policy Training: {policy_name.upper()} Loss Curve"
        cv2.putText(canvas, title, (margin_left, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(canvas, f"Step (Total: {max_step})", (w // 2 - 50, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Legend
        cv2.line(canvas, (w - 280, 35), (w - 240, 35), (248, 189, 56), 2)
        cv2.putText(canvas, "Train Loss", (w - 230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if val_loss_data:
            cv2.circle(canvas, (w - 150, 35), 5, (11, 158, 245), -1)
            cv2.putText(canvas, "Val Loss", (w - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imwrite(str(output_png), canvas)
    except Exception as e:
        print(f"[LOG] Note: Loss curve image generation skipped ({e}). Metrics saved in CSV.")


def generate_html_dashboard(
    output_html: Path,
    model_name: str,
    policy_name: str,
    dataset_name: str,
    current_step: int,
    total_steps: int,
    train_loss: float,
    val_loss: Optional[float],
    steps_per_sec: float,
    elapsed_sec: float,
    eta_sec: float,
    learning_rate: float,
    device_name: str,
    status: str,
    history_steps: List[int],
    history_train_loss: List[float],
    history_val_loss: List[Tuple[int, float]],
):
    """Generates a standalone, beautiful HTML dashboard with interactive Chart.js graphs."""
    val_steps_dict = {s: l for s, l in history_val_loss}
    val_loss_series = [val_steps_dict.get(s, None) for s in history_steps]

    pct = min(100.0, (current_step / max(1, total_steps)) * 100.0)
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_sec)))
    eta_str = str(datetime.timedelta(seconds=int(eta_sec))) if eta_sec > 0 else "00:00:00"
    val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"

    badge_color = "#10b981" if status == "COMPLETED" else ("#38bdf8" if status == "TRAINING" else "#f59e0b")

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Monitor — {model_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #0b1120;
            color: #f8fafc;
            padding: 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .title-group h1 {{
            font-size: 24px;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.5px;
        }}
        .title-group p {{
            color: #94a3b8;
            font-size: 13px;
            margin-top: 4px;
        }}
        .badge {{
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background-color: {badge_color}20;
            color: {badge_color};
            border: 1px solid {badge_color}60;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 16px;
        }}
        .metric-label {{
            font-size: 11px;
            color: #9ca3af;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        .metric-val {{
            font-size: 26px;
            font-weight: 800;
            color: #f3f4f6;
            margin-top: 6px;
            font-family: 'SF Mono', Consolas, monospace;
        }}
        .progress-bar-container {{
            background: #1f2937;
            border-radius: 8px;
            height: 12px;
            overflow: hidden;
            margin-bottom: 24px;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #0284c7, #38bdf8);
            width: {pct:.1f}%;
            transition: width 0.3s ease;
        }}
        .chart-box {{
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        .chart-title {{
            font-size: 16px;
            font-weight: 700;
            color: #e5e7eb;
        }}
        .refresh-toggle {{
            font-size: 12px;
            color: #94a3b8;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title-group">
            <h1>UR5e LeRobot Training Dashboard</h1>
            <p>Model: <strong>{model_name}</strong> | Policy: <strong>{policy_name.upper()}</strong> | Dataset: <strong>{dataset_name}</strong></p>
        </div>
        <div class="badge">{status}</div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Step Progress</div>
            <div class="metric-val">{current_step:,} <span style="font-size: 14px; color: #94a3b8;">/ {total_steps:,}</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Train Loss</div>
            <div class="metric-val" style="color: #38bdf8;">{train_loss:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Val Loss</div>
            <div class="metric-val" style="color: #f59e0b;">{val_loss_str}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Speed</div>
            <div class="metric-val">{steps_per_sec:.1f} <span style="font-size: 13px; color: #94a3b8;">steps/s</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">ETA / Elapsed</div>
            <div class="metric-val" style="font-size: 19px;">{eta_str} <span style="font-size: 12px; color: #94a3b8;">({elapsed_str})</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Learning Rate</div>
            <div class="metric-val" style="font-size: 19px;">{learning_rate:.2e}</div>
        </div>
    </div>

    <div class="progress-bar-container">
        <div class="progress-fill"></div>
    </div>

    <div class="chart-box">
        <div class="chart-header">
            <div class="chart-title">Loss Convergence Curves</div>
            <label class="refresh-toggle">
                <input type="checkbox" id="autoRefresh" {"checked" if status == "TRAINING" else ""}> Auto-refresh (5s)
            </label>
        </div>
        <div style="position: relative; height: 420px; width: 100%;">
            <canvas id="lossChart"></canvas>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('lossChart').getContext('2d');
        const steps = {json.dumps(history_steps[-600:])};
        const trainLoss = {json.dumps([round(x, 5) for x in history_train_loss[-600:]])};
        const valLoss = {json.dumps([round(x, 5) if x is not None else None for x in val_loss_series[-600:]])};

        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: steps,
                datasets: [
                    {{
                        label: 'Training Loss',
                        data: trainLoss,
                        borderColor: '#38bdf8',
                        backgroundColor: 'rgba(56, 189, 248, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.2,
                        fill: true
                    }},
                    {{
                        label: 'Validation Loss',
                        data: valLoss,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.2)',
                        borderWidth: 2.5,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        showLine: true,
                        spanGaps: true
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ intersect: false, mode: 'index' }},
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Optimization Step', color: '#94a3b8' }},
                        grid: {{ color: '#1e293b' }},
                        ticks: {{ color: '#64748b' }}
                    }},
                    y: {{
                        title: {{ display: true, text: 'Loss', color: '#94a3b8' }},
                        grid: {{ color: '#1e293b' }},
                        ticks: {{ color: '#64748b' }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#e2e8f0', font: {{ weight: 'bold' }} }} }}
                }}
            }}
        }});

        const autoRefreshCheckbox = document.getElementById('autoRefresh');
        if (autoRefreshCheckbox) {{
            setInterval(() => {{
                if (autoRefreshCheckbox.checked) {{
                    window.location.reload();
                }}
            }}, 5000);
        }}
    </script>
</body>
</html>
"""
    try:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_html.write_text(html_code, encoding="utf-8")
    except Exception as err:
        print(f"[DASHBOARD] Warning: Failed to write HTML dashboard: {err}")



# =============================================================================
# 3. Modular RobotTrainer Class
# =============================================================================

def detect_dataset_grayscale(dataset_path: Path) -> bool:
    """Checks dataset metadata for grayscale recording mode."""
    for candidate in [
        Path(dataset_path) / "meta" / "recording_config.json",
        Path(dataset_path) / "recording_config.json",
    ]:
        if candidate.exists():
            try:
                with open(candidate, "r") as f:
                    cfg = json.load(f)
                    if cfg.get("grayscale") is True:
                        return True
            except Exception:
                pass
    return False


class RobotTrainer:
    """
    Core imitation learning training manager for UR5e LeRobot datasets.
    Can be invoked from CLI or embedded directly in a PyQt GUI QThread.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        policy_name: str = "act",
        optimizer_name: str = "AdamW",
        training_steps: int = TRAINING_STEPS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        lr_backbone: float = LR_BACKBONE,
        lr_min: float = LR_MIN,
        warmup_steps: int = WARMUP_STEPS,
        weight_decay: float = WEIGHT_DECAY,
        grad_clip_norm: float = GRAD_CLIP_NORM,
        use_augmentation: bool = USE_AUGMENTATION,
        save_checkpoint_freq: int = SAVE_CHECKPOINT_FREQ,
        log_freq: int = LOG_FREQ,
        eval_freq: int = EVAL_FREQ,
        val_split_ratio: float = VAL_SPLIT_RATIO,
        output_dir: str | Path = OUTPUT_DIR,
        device: str = DEVICE,
        num_workers: int = NUM_WORKERS,
        seed: int = SEED,
        step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        resume_checkpoint: Optional[str | Path] = None,
        chunk_size: Optional[int] = None,
        n_action_steps: Optional[int] = None,
        kl_weight: Optional[float] = None,
        dropout: Optional[float] = None,
        grayscale: Optional[bool] = None,
    ):
        if not LEROBOT_AVAILABLE:
            raise RuntimeError(f"LeRobot could not be loaded in this Python environment: {LEROBOT_IMPORT_ERROR}")

        self.dataset_path = Path(dataset_path).resolve()
        self.policy_name = policy_name.lower().strip()
        self.optimizer_name = optimizer_name.strip()
        self.training_steps = int(training_steps)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.lr_backbone = float(lr_backbone)
        self.lr_min = float(lr_min)
        self.warmup_steps = int(warmup_steps)
        self.weight_decay = float(weight_decay)
        self.grad_clip_norm = float(grad_clip_norm)
        self.use_augmentation = bool(use_augmentation)
        self.save_checkpoint_freq = int(save_checkpoint_freq)
        self.log_freq = int(log_freq)
        self.eval_freq = int(eval_freq)
        self.val_split_ratio = float(val_split_ratio)
        self.output_dir = Path(output_dir).resolve()
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.step_callback = step_callback

        # Internal state
        self.is_aborted = False
        self.checkpoint_dir = self.output_dir / "checkpoint"
        self.log_csv_file = self.output_dir / "train_log.csv"
        self.loss_plot_file = self.output_dir / "loss_curve.png"
        self.dashboard_html_file = self.output_dir / "training_dashboard.html"
        self.resume_checkpoint = Path(resume_checkpoint).resolve() if resume_checkpoint else None
        self.chunk_size = int(chunk_size) if chunk_size is not None else None
        self.n_action_steps = int(n_action_steps) if n_action_steps is not None else None
        self.kl_weight = float(kl_weight) if kl_weight is not None else None
        self.dropout = float(dropout) if dropout is not None else None
        if grayscale is not None:
            self.is_grayscale = bool(grayscale)
        else:
            self.is_grayscale = detect_dataset_grayscale(self.dataset_path)

        # History tracking
        self.history_steps: List[int] = []
        self.history_train_loss: List[float] = []
        self.history_val_loss: List[Tuple[int, float]] = []

    def abort(self):
        """Signals the training loop to safely stop early."""
        self.is_aborted = True

    def _validate_dataset(self) -> LeRobotDatasetMetadata:
        """Verifies existence of dataset metadata, files, and camera features."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset directory not found at: {self.dataset_path}\n"
                f"Please ensure you record demonstrations using 'python gui_recorder.py' first."
            )

        info_file = self.dataset_path / "meta" / "info.json"
        if not info_file.exists():
            raise FileNotFoundError(
                f"Missing 'meta/info.json' at {self.dataset_path}.\n"
                f"The folder does not appear to be a valid finalized LeRobot dataset."
            )

        repo_id = f"local/{self.dataset_path.name}"
        meta = LeRobotDatasetMetadata(repo_id, root=self.dataset_path)

        if meta.total_episodes <= 0 or meta.total_frames <= 0:
            raise ValueError(f"Dataset '{self.dataset_path.name}' has 0 episodes or frames recorded.")

        return meta

    def _build_policy_config(self, meta: LeRobotDatasetMetadata) -> Tuple[Any, Any, Any]:
        """Constructs policy configuration, policy model, and pre/post processors."""
        features = dataset_to_policy_features(meta.features)
        output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {k: ft for k, ft in features.items() if k not in output_features}

        print(f"[POLICY] Identified Input Features:  {list(input_features.keys())}")
        print(f"[POLICY] Identified Output Features: {list(output_features.keys())}")

        if self.policy_name == "act":
            kwargs = {}
            if self.chunk_size is not None:
                kwargs["chunk_size"] = self.chunk_size
            if self.n_action_steps is not None:
                kwargs["n_action_steps"] = self.n_action_steps
            if self.kl_weight is not None:
                kwargs["kl_weight"] = self.kl_weight
            if self.dropout is not None:
                kwargs["dropout"] = self.dropout
            cfg = ACTConfig(input_features=input_features, output_features=output_features, **kwargs)
            if self.resume_checkpoint:
                resume_p = self.resume_checkpoint
                if (resume_p / "checkpoint").exists():
                    resume_p = resume_p / "checkpoint"
                print(f"[RESUME] Loading pretrained ACT weights from: {resume_p}")
                policy = ACTPolicy.from_pretrained(resume_p)
            else:
                policy = ACTPolicy(cfg)
        elif self.policy_name == "diffusion":
            try:
                from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
            except ImportError as err:
                raise ImportError(
                    "Diffusion Policy requires the 'diffusers' library.\n"
                    "Install it in your environment using: pip install diffusers\n"
                    f"Original error: {err}"
                )
            kwargs = {}
            if self.n_action_steps is not None:
                kwargs["n_action_steps"] = self.n_action_steps
            cfg = DiffusionConfig(input_features=input_features, output_features=output_features, **kwargs)
            if self.resume_checkpoint:
                resume_p = self.resume_checkpoint
                if (resume_p / "checkpoint").exists():
                    resume_p = resume_p / "checkpoint"
                print(f"[RESUME] Loading pretrained Diffusion weights from: {resume_p}")
                policy = DiffusionPolicy.from_pretrained(resume_p)
            else:
                policy = DiffusionPolicy(cfg)
        elif self.policy_name == "vqbet":
            try:
                from lerobot.policies.vqbet import VQBeTConfig, VQBeTPolicy
            except ImportError as err:
                raise ImportError(
                    f"VQ-BeT Policy dependencies missing: {err}\n"
                    "Ensure torchvision and transformers are up to date."
                )
            cfg = VQBeTConfig(input_features=input_features, output_features=output_features)
            policy = VQBeTPolicy(cfg)
        else:
            raise ValueError(
                f"Unsupported policy '{self.policy_name}'. Choose from: {AVAILABLE_POLICIES}"
            )

        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)
        return cfg, policy, (preprocessor, postprocessor)

    def _build_optimizer(self, policy: nn.Module) -> torch.optim.Optimizer:
        """Instantiates the selected PyTorch optimizer with differential learning rates."""
        backbone_params = []
        transformer_params = []

        for name, param in policy.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                transformer_params.append(param)

        if backbone_params and self.policy_name == "act":
            print(f"[OPTIMIZER] Using differential learning rates:")
            print(f"            • Vision Backbone (ResNet): {self.lr_backbone:.1e}")
            print(f"            • Policy Heads/Transformer: {self.learning_rate:.1e}")
            param_groups = [
                {"params": transformer_params, "lr": self.learning_rate, "weight_decay": self.weight_decay},
                {"params": backbone_params, "lr": self.lr_backbone, "weight_decay": self.weight_decay},
            ]
        else:
            param_groups = [{"params": [p for p in policy.parameters() if p.requires_grad], "lr": self.learning_rate, "weight_decay": self.weight_decay}]

        opt_name = self.optimizer_name.upper()
        if opt_name == "ADAMW":
            return torch.optim.AdamW(param_groups)
        elif opt_name == "ADAM":
            return torch.optim.Adam(param_groups)
        elif opt_name == "SGD":
            return torch.optim.SGD(param_groups, momentum=0.9)
        else:
            print(f"[WARNING] Unknown optimizer '{self.optimizer_name}'. Defaulting to AdamW.")
            return torch.optim.AdamW(param_groups)

    def _save_checkpoint(self, policy: Any, preprocessor: Any, postprocessor: Any, step: int):
        """
        Saves the model weights and processors to disk.
        Overwrites the same directory every time to minimize disk consumption.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[CHECKPOINT] Saving latest policy weights to: {self.checkpoint_dir} (Step {step})")

        # Save model and normalization processors
        policy.save_pretrained(self.checkpoint_dir)
        preprocessor.save_pretrained(self.checkpoint_dir)
        postprocessor.save_pretrained(self.checkpoint_dir)

        # Save metadata info file
        info_dict = {
            "policy_type": self.policy_name,
            "step": step,
            "training_steps": self.training_steps,
            "dataset": str(self.dataset_path.name),
            "grayscale": self.is_grayscale,
            "image_channels": 1 if self.is_grayscale else 3,
            "updated_at": datetime.datetime.now().isoformat(),
        }
        with open(self.checkpoint_dir / "training_info.json", "w") as f:
            json.dump(info_dict, f, indent=2)

        # Update visual loss curve plot
        plot_loss_curve(
            self.log_csv_file,
            self.loss_plot_file,
            self.policy_name,
            self.history_steps,
            self.history_train_loss,
            self.history_val_loss,
        )

        # Generate live HTML browser dashboard
        latest_t_loss = self.history_train_loss[-1] if self.history_train_loss else 0.0
        latest_v_loss = self.history_val_loss[-1][1] if self.history_val_loss else None
        generate_html_dashboard(
            output_html=self.dashboard_html_file,
            model_name=self.output_dir.name,
            policy_name=self.policy_name,
            dataset_name=self.dataset_path.name,
            current_step=step,
            total_steps=self.training_steps,
            train_loss=latest_t_loss,
            val_loss=latest_v_loss,
            steps_per_sec=0.0,
            elapsed_sec=0.0,
            eta_sec=0.0,
            learning_rate=0.0,
            device_name=str(self.device),
            status="COMPLETED" if step >= self.training_steps else ("ABORTED" if self.is_aborted else "TRAINING"),
            history_steps=self.history_steps,
            history_train_loss=self.history_train_loss,
            history_val_loss=self.history_val_loss,
        )
        print(f"[CHECKPOINT] ✔ Checkpoint, loss plot & HTML dashboard updated ({self.dashboard_html_file.name}).")

    def run_training(self) -> Dict[str, Any]:
        """Executes the complete offline imitation learning training loop."""
        set_seed(self.seed)
        start_time = time.time()

        print("\n" + "=" * 70)
        print("  UR5e LeRobot Imitation Learning Training Engine")
        print("=" * 70)
        print(f"  • Dataset:           {self.dataset_path}")
        print(f"  • Selected Policy:   {self.policy_name.upper()}")
        print(f"  • Optimizer:         {self.optimizer_name} (lr={self.learning_rate}, wd={self.weight_decay})")
        print(f"  • Image Mode:        {'Grayscale (1-Ch luminance)' if self.is_grayscale else 'RGB (3-Ch color)'}")
        print(f"  • Total Steps:       {self.training_steps:,}")
        print(f"  • Batch Size:        {self.batch_size}")
        print(f"  • Device:            {self.device} ({torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'})")
        print(f"  • Save Interval:     Every {self.save_checkpoint_freq} steps (In-place overwrite)")
        print(f"  • Output Directory:  {self.output_dir}")
        print("=" * 70 + "\n")

        # 1. Dataset Verification
        meta = self._validate_dataset()
        print(f"[DATASET] Loaded '{self.dataset_path.name}': {meta.total_episodes} episodes, {meta.total_frames} frames @ {meta.fps} FPS")

        # 2. Policy Setup
        cfg, policy, (preprocessor, postprocessor) = self._build_policy_config(meta)
        policy.to(self.device)
        policy.train()
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"[MODEL] Policy initialized. Total parameters: {total_params:,} (Trainable: {trainable_params:,})")

        # 3. Delta Timestamps Alignment
        delta_timestamps = {
            "action": make_delta_timestamps(getattr(cfg, "action_delta_indices", None), meta.fps),
        }
        if hasattr(cfg, "image_features") and cfg.image_features:
            for k in cfg.image_features:
                delta_timestamps[k] = make_delta_timestamps(getattr(cfg, "observation_delta_indices", None), meta.fps)

        # 4. Instantiate LeRobotDataset
        repo_id = f"local/{self.dataset_path.name}"
        dataset = LeRobotDataset(repo_id, root=self.dataset_path, delta_timestamps=delta_timestamps, video_backend="pyav")
        total_samples = len(dataset)
        print(f"[DATASET] Ready with {total_samples} temporal-aligned samples.")

        # 5. Train / Validation Split
        indices = list(range(total_samples))
        random.shuffle(indices)
        num_val = int(total_samples * self.val_split_ratio) if self.val_split_ratio > 0 and total_samples > 10 else 0

        val_dataset = None
        if num_val > 0:
            val_indices = indices[:num_val]
            train_indices = indices[num_val:]
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            print(f"[SPLIT] Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
        else:
            train_dataset = dataset
            print(f"[SPLIT] Using all {len(train_dataset)} samples for training.")

        # 6. DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=(len(train_dataset) > self.batch_size),
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=max(0, min(2, self.num_workers)),
                pin_memory=(self.device.type == "cuda"),
                drop_last=False,
            )

        # 7. Optimizer & Learning Rate Scheduler Setup
        optimizer = self._build_optimizer(policy)

        # Cosine Annealing Learning Rate Scheduler with Linear Warmup
        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(max(1, self.training_steps - self.warmup_steps))
            cos_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(self.lr_min / self.learning_rate, cos_mult)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"[SCHEDULER] Cosine Annealing initialized: {self.warmup_steps} warmup steps -> min lr {self.lr_min:.1e}")
        if self.use_augmentation:
            print("[AUGMENT] Visual data augmentation active (mild randomized brightness/contrast jitter on GPU).")

        # Check for starting step if resuming
        start_step = 0
        if self.resume_checkpoint:
            resume_p = self.resume_checkpoint
            if (resume_p / "checkpoint").exists():
                resume_p = resume_p / "checkpoint"
            info_f = resume_p / "training_info.json"
            if info_f.exists():
                try:
                    with open(info_f, "r") as f:
                        info_data = json.load(f)
                    start_step = int(info_data.get("step", 0))
                    print(f"[RESUME] Found prior training record at step {start_step:,}.")
                except Exception as err:
                    print(f"[RESUME] Note: Could not read training_info.json: {err}")

        if self.training_steps <= start_step:
            print(f"[RESUME] Specified training_steps ({self.training_steps}) <= start_step ({start_step}).")
            self.training_steps = start_step + 5000
            print(f"[RESUME] Automatically adjusted target to {self.training_steps:,} steps.")

        # 8. Setup Output Logging Directory & CSV
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_mode = "a" if (self.resume_checkpoint and self.log_csv_file.exists()) else "w"
        if csv_mode == "w":
            with open(self.log_csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "train_loss", "val_loss", "learning_rate", "elapsed_sec", "steps_per_sec"])

        # 9. Main Optimization Loop
        print("\n[TRAIN] >>> Commencing Optimization Steps <<<")
        step = start_step
        running_loss = 0.0
        log_window_steps = 0
        step_start_time = time.time()
        train_iter = iter(train_loader)

        while step < self.training_steps and not self.is_aborted:
            step += 1

            # Fetch next batch (loop dataloader infinitely across epochs)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Preprocess (normalization & tensor device transfer)
            batch = preprocessor(batch)
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Grayscale channel consistency
            if self.is_grayscale:
                for k in batch:
                    if "image" in k and isinstance(batch[k], torch.Tensor) and batch[k].ndim == 4:
                        if batch[k].shape[1] == 3:
                            lum = 0.2989 * batch[k][:, 0:1] + 0.5870 * batch[k][:, 1:2] + 0.1140 * batch[k][:, 2:3]
                            batch[k] = lum.repeat(1, 3, 1, 1)
                        elif batch[k].shape[1] == 1:
                            batch[k] = batch[k].repeat(1, 3, 1, 1)

            # Visual Data Augmentation (mild random brightness/contrast on images)
            if self.use_augmentation:
                for k in batch:
                    if "image" in k and isinstance(batch[k], torch.Tensor) and batch[k].ndim == 4:
                        bright = 1.0 + (torch.rand(batch[k].shape[0], 1, 1, 1, device=batch[k].device) - 0.5) * 0.25
                        contrast = 1.0 + (torch.rand(batch[k].shape[0], 1, 1, 1, device=batch[k].device) - 0.5) * 0.25
                        img = batch[k] * bright
                        mean = img.mean(dim=(-2, -1), keepdim=True)
                        img = (img - mean) * contrast + mean
                        batch[k] = torch.clamp(img, 0.0, 1.0)

            # Forward & Backward Pass
            optimizer.zero_grad()
            loss, _ = policy.forward(batch)
            loss_val = loss.item()
            loss.backward()

            # Gradient Clipping & Optimizer Step
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            running_loss += loss_val
            log_window_steps += 1

            # Periodic Validation Check
            val_loss_scalar: Optional[float] = None
            if val_loader is not None and (step % self.eval_freq == 0 or step == self.training_steps):
                policy.eval()
                val_losses = []
                with torch.no_grad():
                    for v_idx, v_batch in enumerate(val_loader):
                        v_batch = preprocessor(v_batch)
                        v_batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in v_batch.items()}
                        if self.is_grayscale:
                            for k in v_batch:
                                if "image" in k and isinstance(v_batch[k], torch.Tensor) and v_batch[k].ndim == 4:
                                    if v_batch[k].shape[1] == 3:
                                        lum = 0.2989 * v_batch[k][:, 0:1] + 0.5870 * v_batch[k][:, 1:2] + 0.1140 * v_batch[k][:, 2:3]
                                        v_batch[k] = lum.repeat(1, 3, 1, 1)
                                    elif v_batch[k].shape[1] == 1:
                                        v_batch[k] = v_batch[k].repeat(1, 3, 1, 1)
                        v_loss, _ = policy.forward(v_batch)
                        val_losses.append(v_loss.item())
                        if v_idx >= 30:  # Evaluate up to 30 batches for rapid, non-blocking evaluation
                            break
                if val_losses:
                    val_loss_scalar = float(np.mean(val_losses))
                    self.history_val_loss.append((step, val_loss_scalar))
                policy.train()

            # Logging & Progress Display
            if step % self.log_freq == 0 or step == self.training_steps:
                avg_train_loss = running_loss / max(1, log_window_steps)
                self.history_steps.append(step)
                self.history_train_loss.append(avg_train_loss)

                now = time.time()
                elapsed = now - start_time
                steps_per_sec = log_window_steps / max(1e-5, (now - step_start_time))
                remaining_steps = self.training_steps - step
                eta_sec = remaining_steps / max(1e-5, steps_per_sec)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                current_lr = optimizer.param_groups[0]["lr"]

                # Write to CSV log
                with open(self.log_csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        step,
                        f"{avg_train_loss:.5f}",
                        f"{val_loss_scalar:.5f}" if val_loss_scalar is not None else "",
                        f"{current_lr:.6e}",
                        f"{elapsed:.1f}",
                        f"{steps_per_sec:.2f}",
                    ])

                # Console line
                val_text = f" | val_loss: {val_loss_scalar:.4f}" if val_loss_scalar is not None else ""
                print(
                    f"[Step {step:5d}/{self.training_steps}] "
                    f"train_loss: {avg_train_loss:.4f}{val_text} | "
                    f"lr: {current_lr:.2e} | "
                    f"Speed: {steps_per_sec:.1f} steps/s | "
                    f"ETA: {eta_str}"
                )

                if self.step_callback is not None:
                    self.step_callback({
                        "step": step,
                        "total_steps": self.training_steps,
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss_scalar,
                        "steps_per_sec": steps_per_sec,
                        "eta_sec": eta_sec,
                    })

                running_loss = 0.0
                log_window_steps = 0
                step_start_time = time.time()

            # In-Place Periodic Checkpointing
            if step % self.save_checkpoint_freq == 0 or step == self.training_steps:
                self._save_checkpoint(policy, preprocessor, postprocessor, step)

        total_time = time.time() - start_time
        status = "ABORTED" if self.is_aborted else "COMPLETED"
        print("\n" + "=" * 70)
        print(f"  Training {status} in {datetime.timedelta(seconds=int(total_time))}")
        print(f"  • Final Model Saved: {self.checkpoint_dir}")
        print(f"  • Loss Curve:        {self.loss_plot_file}")
        print(f"  • CSV Metrics Log:   {self.log_csv_file}")
        print("=" * 70 + "\n")

        return {
            "status": status,
            "steps_completed": step,
            "final_checkpoint": str(self.checkpoint_dir),
            "loss_plot": str(self.loss_plot_file),
            "csv_log": str(self.log_csv_file),
            "total_time_s": total_time,
        }


# =============================================================================
# 4. Command Line Interface (CLI) & Script Entrypoint
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train UR5e Imitation Learning Policy (LeRobot)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Path to local LeRobot dataset folder")
    parser.add_argument(
        "--policy",
        type=str,
        default=AVAILABLE_POLICIES[SELECTED_POLICY_INDEX],
        choices=AVAILABLE_POLICIES,
        help=f"Policy architecture to train (options: {AVAILABLE_POLICIES})",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=AVAILABLE_OPTIMIZERS[SELECTED_OPTIMIZER_INDEX],
        choices=AVAILABLE_OPTIMIZERS,
        help=f"Optimizer to use (options: {AVAILABLE_OPTIMIZERS})",
    )
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS, help="Number of training optimization steps")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate for policy heads/transformer")
    parser.add_argument("--lr-backbone", type=float, default=LR_BACKBONE, help="Learning rate for ResNet vision backbone")
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS, help="Linear warmup steps for scheduler")
    parser.add_argument("--no-aug", action="store_true", help="Disable visual data augmentation")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="L2 weight decay")
    parser.add_argument("--save-freq", type=int, default=SAVE_CHECKPOINT_FREQ, help="Checkpoint save step frequency (in-place overwrite)")
    parser.add_argument("--log-freq", type=int, default=LOG_FREQ, help="Console logging interval")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for model weights and loss curves")
    parser.add_argument("--device", type=str, default=DEVICE, help="Execution device (cuda or cpu)")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="DataLoader subprocess workers")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--grayscale", action="store_true", default=None, help="Train policy on single-channel Grayscale imagery")
    return parser.parse_args()


def main():
    args = parse_args()

    trainer = RobotTrainer(
        dataset_path=args.dataset,
        policy_name=args.policy,
        optimizer_name=args.optimizer,
        training_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_backbone=args.lr_backbone,
        warmup_steps=args.warmup_steps,
        use_augmentation=not args.no_aug,
        weight_decay=args.weight_decay,
        save_checkpoint_freq=args.save_freq,
        log_freq=args.log_freq,
        output_dir=args.output_dir,
        device=args.device,
        num_workers=args.workers,
        seed=args.seed,
        grayscale=args.grayscale,
    )

    try:
        trainer.run_training()
    except KeyboardInterrupt:
        print("\n[TRAIN] Training interrupted by user (Ctrl+C). Saving emergency checkpoint...")
        trainer.abort()


if __name__ == "__main__":
    main()

