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

# Training Hyperparameters (tuned for 320x240 dual-cam 7-DoF cube pick-and-place)
TRAINING_STEPS = 20000        # Total optimization steps (~5 epochs on 130k frames; avoids overfitting)
BATCH_SIZE = 32               # Samples per training batch (RTX 3080 16GB sweet spot)
LEARNING_RATE = 1e-4          # Initial optimizer learning rate for transformer heads
LR_BACKBONE = 1e-5            # Differential learning rate for pretrained ResNet vision backbone
LR_MIN = 1e-6                 # Minimum learning rate at end of Cosine Annealing schedule
WARMUP_STEPS = 1000           # Linear warmup steps to stabilize transformer attention layers
WEIGHT_DECAY = 1e-4           # L2 regularization weight decay
GRAD_CLIP_NORM = 10.0         # Gradient clipping max norm to prevent exploding gradients
DEFAULT_CHUNK_SIZE = 60       # 2.0s action chunk horizon @ 30 FPS (crisp pickup/close/drop transitions)
USE_AUGMENTATION = True       # Master switch for GPU visual data augmentation
AUG_ROTATION = True           # Slight random rotation (±10 deg)
AUG_TRANSLATION = True        # Slight random translation (±8% X/Y)
AUG_COLOR = True              # Mild brightness & contrast jitter (±20%)
AUG_CUTOUT = False            # Random cutout / occlusion patches (5-15% area)
AUG_NOISE = False             # Mild Gaussian sensor noise
TEMPORAL_ENSEMBLE_COEFF = 0.01  # ACT Algorithm 2: Exponential moving average across overlapping chunks (0.01 standard)

# Checkpoint Management
# Keeps latest checkpoint/, lowest-val-loss best_checkpoint/, and periodic checkpoints/step_XXXXXX/
SAVE_CHECKPOINT_FREQ = 2500   # Save periodic policy snapshot every N steps
OUTPUT_DIR = "outputs/train/" + PROJECT_NAME  # Directory for model checkpoint, logs, and plots

# Validation & Evaluation
VAL_SPLIT_RATIO = 0.1         # Reserve 10% of frames/episodes for validation
EVAL_FREQ = 1000              # Compute validation loss every N steps

# Logging & Hardware
LOG_FREQ = 50                 # Console progress print interval (in steps)
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


def apply_visual_augmentation(
    img: torch.Tensor,
    aug_rotation: bool = True,
    aug_translation: bool = True,
    aug_color: bool = True,
    aug_cutout: bool = False,
    aug_noise: bool = False,
    max_rotation_deg: float = 10.0,
    max_translation_ratio: float = 0.08,
) -> torch.Tensor:
    """
    High-performance GPU-accelerated visual data augmentation for imitation & RL policies.
    Supports 4D tensors (B, C, H, W) and 5D temporal sequences (B, T, C, H, W).
    Executes under torch.no_grad() to eliminate autograd graph overhead.
    """
    if not (aug_rotation or aug_translation or aug_color or aug_cutout or aug_noise):
        return img

    with torch.no_grad():
        orig_shape = img.shape
        if img.ndim == 5:
            B, T, C, H, W = img.shape
            flat_img = img.view(B * T, C, H, W)
        elif img.ndim == 4:
            B, C, H, W = img.shape
            flat_img = img
            T = 1
        else:
            return img

        N = B * T
        if N == 0:
            return img

        # 1. Spatial Affine Transformation (Random Rotation ±10 deg, Shift ±8% X/Y)
        if aug_rotation or aug_translation:
            if aug_rotation:
                angles = torch.empty(N, device=flat_img.device).uniform_(-max_rotation_deg, max_rotation_deg)
                rad = angles * (3.141592653589793 / 180.0)
                cos = torch.cos(rad)
                sin = torch.sin(rad)
            else:
                cos = torch.ones(N, device=flat_img.device)
                sin = torch.zeros(N, device=flat_img.device)

            if aug_translation:
                max_t = 2.0 * max_translation_ratio
                tx = torch.empty(N, device=flat_img.device).uniform_(-max_t, max_t)
                ty = torch.empty(N, device=flat_img.device).uniform_(-max_t, max_t)
            else:
                tx = torch.zeros(N, device=flat_img.device)
                ty = torch.zeros(N, device=flat_img.device)

            theta = torch.zeros(N, 2, 3, device=flat_img.device, dtype=flat_img.dtype)
            theta[:, 0, 0] = cos
            theta[:, 0, 1] = -sin
            theta[:, 0, 2] = tx
            theta[:, 1, 0] = sin
            theta[:, 1, 1] = cos
            theta[:, 1, 2] = ty

            grid = torch.nn.functional.affine_grid(theta, flat_img.size(), align_corners=False)
            flat_img = torch.nn.functional.grid_sample(flat_img, grid, mode='bilinear', padding_mode='border', align_corners=False)

        # 2. Lighting & Color Jitter (Brightness ±20%, Contrast ±20%)
        if aug_color:
            bright = 1.0 + (torch.rand(N, 1, 1, 1, device=flat_img.device) - 0.5) * 0.3
            contrast = 1.0 + (torch.rand(N, 1, 1, 1, device=flat_img.device) - 0.5) * 0.3
            flat_img = flat_img * bright
            mean = flat_img.mean(dim=(-2, -1), keepdim=True)
            flat_img = (flat_img - mean) * contrast + mean
            flat_img = torch.clamp(flat_img, 0.0, 1.0)

        # 3. Gaussian Sensor Noise
        if aug_noise:
            noise_mask = (torch.rand(N, 1, 1, 1, device=flat_img.device) < 0.5).float()
            noise = torch.randn_like(flat_img) * 0.02 * noise_mask
            flat_img = torch.clamp(flat_img + noise, 0.0, 1.0)

        # 4. Random Cutout / Occlusion Dropout Patches (simulates gripper/tool/arm occlusions)
        if aug_cutout:
            cutout_probs = torch.rand(N, device=flat_img.device)
            box_ws = (torch.rand(N, device=flat_img.device) * 0.15 + 0.05) * W
            box_hs = (torch.rand(N, device=flat_img.device) * 0.15 + 0.05) * H
            xs = torch.rand(N, device=flat_img.device) * (W - 1)
            ys = torch.rand(N, device=flat_img.device) * (H - 1)

            for i in range(N):
                if cutout_probs[i] < 0.5:
                    bw = int(box_ws[i].item())
                    bh = int(box_hs[i].item())
                    cx = int(xs[i].item())
                    cy = int(ys[i].item())
                    x1 = max(0, cx - bw // 2)
                    x2 = min(W, cx + bw // 2)
                    y1 = max(0, cy - bh // 2)
                    y2 = min(H, cy + bh // 2)
                    flat_img[i, :, y1:y2, x1:x2] = 0.0

        if orig_shape != flat_img.shape:
            flat_img = flat_img.view(orig_shape)
        return flat_img


def repair_dataset_indices_if_needed(dataset_path: Path) -> bool:
    """
    Checks if dataset parquet index columns, episode indices, and info.json total_frames
    are strictly contiguous and in-bounds (e.g. after episode deletions). If a desynchronization
    is detected, reindexes all parquet rows starting from 0 and updates metadata/stats.
    """
    data_dir = dataset_path / "data"
    meta_info_file = dataset_path / "meta" / "info.json"
    if not data_dir.exists() or not meta_info_file.exists():
        return False

    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
        import numpy as np

        data_files = sorted(data_dir.glob("**/*.parquet"))
        if not data_files:
            return False

        with open(meta_info_file, "r") as f:
            info = json.load(f)

        meta_frames = info.get("total_frames", 0)
        total_rows = sum(pq.read_metadata(df).num_rows for df in data_files)

        # Check first index
        first_table = pq.read_table(data_files[0])
        first_idx = first_table.column("index")[0].as_py() if "index" in first_table.column_names else 0

        # If meta_frames matches total_rows and first index is 0, no repair needed
        if meta_frames == total_rows and first_idx == 0:
            return False

        print(f"[REPAIR] Dataset '{dataset_path.name}' index desynchronization detected:")
        print(f"         meta total_frames={meta_frames}, actual rows={total_rows}, first_index={first_idx}")
        print(f"         Repairing contiguous frame indices (0 -> {total_rows - 1}) and updating metadata...")

        ep_meta_dir = dataset_path / "meta" / "episodes"
        ep_files = sorted(ep_meta_dir.glob("**/*.parquet")) if ep_meta_dir.exists() else []

        curr_offset = 0
        all_ep_stats = []

        for i, df_p in enumerate(data_files):
            d_table = pq.read_table(df_p)
            num_rows = len(d_table)
            new_index = pa.array(np.arange(curr_offset, curr_offset + num_rows, dtype=np.int64))
            new_ep_idx = pa.array(np.full(num_rows, i, dtype=np.int64))

            d_cols = {}
            for name in d_table.column_names:
                if name == "index":
                    d_cols[name] = new_index
                elif name == "episode_index":
                    d_cols[name] = new_ep_idx
                else:
                    d_cols[name] = d_table.column(name)

            new_d_table = pa.Table.from_arrays(list(d_cols.values()), names=list(d_cols.keys()))
            pq.write_table(new_d_table, df_p)

            # Repair corresponding episode meta parquet if available
            if i < len(ep_files):
                ep_p = ep_files[i]
                e_table = pq.read_table(ep_p)
                idx_arr = np.arange(curr_offset, curr_offset + num_rows, dtype=np.float64)

                e_dict = {col: e_table.column(col)[0].as_py() for col in e_table.column_names}
                e_dict["episode_index"] = i
                e_dict["data/file_index"] = i
                if "meta/episodes/file_index" in e_dict:
                    e_dict["meta/episodes/file_index"] = i
                e_dict["dataset_from_index"] = curr_offset
                e_dict["dataset_to_index"] = curr_offset + num_rows

                for k in list(e_dict.keys()):
                    if k.startswith("videos/") and k.endswith("/file_index"):
                        e_dict[k] = i

                if "stats/index/min" in e_dict:
                    e_dict["stats/index/min"] = [float(curr_offset)]
                    e_dict["stats/index/max"] = [float(curr_offset + num_rows - 1)]
                    e_dict["stats/index/mean"] = [float(np.mean(idx_arr))]
                    e_dict["stats/index/std"] = [float(np.std(idx_arr))]
                    e_dict["stats/index/count"] = [num_rows]
                    e_dict["stats/index/q01"] = [float(np.percentile(idx_arr, 1))]
                    e_dict["stats/index/q10"] = [float(np.percentile(idx_arr, 10))]
                    e_dict["stats/index/q50"] = [float(np.percentile(idx_arr, 50))]
                    e_dict["stats/index/q90"] = [float(np.percentile(idx_arr, 90))]
                    e_dict["stats/index/q99"] = [float(np.percentile(idx_arr, 99))]

                if "stats/episode_index/min" in e_dict:
                    e_dict["stats/episode_index/min"] = [float(i)]
                    e_dict["stats/episode_index/max"] = [float(i)]
                    e_dict["stats/episode_index/mean"] = [float(i)]
                    e_dict["stats/episode_index/std"] = [0.0]
                    e_dict["stats/episode_index/count"] = [num_rows]
                    e_dict["stats/episode_index/q01"] = [float(i)]
                    e_dict["stats/episode_index/q10"] = [float(i)]
                    e_dict["stats/episode_index/q50"] = [float(i)]
                    e_dict["stats/episode_index/q90"] = [float(i)]
                    e_dict["stats/episode_index/q99"] = [float(i)]

                new_e_arrays = []
                for f in e_table.schema:
                    val = e_dict[f.name]
                    new_e_arrays.append(pa.array([val], type=f.type))
                new_e_table = pa.Table.from_arrays(new_e_arrays, schema=e_table.schema)
                pq.write_table(new_e_table, ep_p)

                # Collect stats for aggregate_stats
                ep_stat = {}
                for col in e_table.column_names:
                    if col.startswith("stats/"):
                        parts = col.split("/")
                        feat_name = "/".join(parts[1:-1])
                        stat_name = parts[-1]
                        if feat_name not in ep_stat:
                            ep_stat[feat_name] = {}
                        val = new_e_table.column(col)[0].as_py()
                        ep_stat[feat_name][stat_name] = np.array(val)
                all_ep_stats.append(ep_stat)

            curr_offset += num_rows

        # Update info.json
        info["total_episodes"] = len(data_files)
        info["total_frames"] = curr_offset
        with open(meta_info_file, "w") as f:
            json.dump(info, f, indent=2)

        # Update stats.json if ep_stats available
        if all_ep_stats:
            try:
                from lerobot.datasets.compute_stats import aggregate_stats
                global_stats = aggregate_stats(all_ep_stats)
                def to_json_serializable(d):
                    if isinstance(d, dict):
                        return {k: to_json_serializable(v) for k, v in d.items()}
                    elif isinstance(d, np.ndarray):
                        return d.tolist()
                    return d
                with open(dataset_path / "meta" / "stats.json", "w") as f:
                    json.dump(to_json_serializable(global_stats), f, indent=2)
            except Exception:
                pass

        print(f"[REPAIR] ✔ Dataset '{dataset_path.name}' repaired successfully ({len(data_files)} episodes, {curr_offset} frames).")
        return True
    except Exception as err:
        print(f"[WARNING] Could not auto-repair dataset indices: {err}")
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
        aug_rotation: bool = AUG_ROTATION,
        aug_translation: bool = AUG_TRANSLATION,
        aug_color: bool = AUG_COLOR,
        aug_cutout: bool = AUG_CUTOUT,
        aug_noise: bool = AUG_NOISE,
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
        temporal_ensemble_coeff: Optional[float] = TEMPORAL_ENSEMBLE_COEFF,
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
        self.aug_rotation = bool(aug_rotation)
        self.aug_translation = bool(aug_translation)
        self.aug_color = bool(aug_color)
        self.aug_cutout = bool(aug_cutout)
        self.aug_noise = bool(aug_noise)
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
        self.best_checkpoint_dir = self.output_dir / "best_checkpoint"
        self.periodic_checkpoints_dir = self.output_dir / "checkpoints"
        self.best_val_loss: float = float("inf")
        self.log_csv_file = self.output_dir / "train_log.csv"
        self.loss_plot_file = self.output_dir / "loss_curve.png"
        self.dashboard_html_file = self.output_dir / "training_dashboard.html"
        self.resume_checkpoint = Path(resume_checkpoint).resolve() if resume_checkpoint else None
        self.chunk_size = int(chunk_size) if chunk_size is not None else DEFAULT_CHUNK_SIZE
        self.temporal_ensemble_coeff = float(temporal_ensemble_coeff) if temporal_ensemble_coeff is not None else None
        if self.policy_name == "act" and self.temporal_ensemble_coeff is not None:
            self.n_action_steps = 1
        else:
            self.n_action_steps = int(n_action_steps) if n_action_steps is not None else None
        self.kl_weight = float(kl_weight) if kl_weight is not None else None
        self.dropout = float(dropout) if dropout is not None else None
        if grayscale is not None:
            self.is_grayscale = bool(grayscale)
        else:
            self.is_grayscale = False  # Default: Train in RGB Color

        # Extract native dataset FPS
        self.dataset_fps = 30
        for info_c in [self.dataset_path / "meta" / "info.json", self.dataset_path / "meta" / "recording_config.json"]:
            if info_c.exists():
                try:
                    with open(info_c, "r") as f_meta:
                        self.dataset_fps = int(json.load(f_meta).get("fps", 30))
                        break
                except Exception:
                    pass

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

        # Self-healing index alignment (repairs any frame gaps or off-by-N indices from episode deletions)
        repair_dataset_indices_if_needed(self.dataset_path)

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
            if self.temporal_ensemble_coeff is not None:
                kwargs["temporal_ensemble_coeff"] = self.temporal_ensemble_coeff
                kwargs["n_action_steps"] = 1
            elif self.n_action_steps is not None:
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

    def _write_checkpoint_to_dir(
        self,
        target_dir: Path,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        step: int,
        val_loss: Optional[float] = None,
    ):
        """Writes model weights, processors, and training_info.json into target_dir."""
        target_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(target_dir)
        preprocessor.save_pretrained(target_dir)
        postprocessor.save_pretrained(target_dir)

        eff_chunk = getattr(policy.config, "chunk_size", self.chunk_size or DEFAULT_CHUNK_SIZE)
        eff_n_actions = getattr(policy.config, "n_action_steps", self.n_action_steps or (50 if self.policy_name == 'act' else 10))
        dec_interval = float(eff_n_actions / max(1, self.dataset_fps))

        info_dict = {
            "policy_type": self.policy_name,
            "step": step,
            "training_steps": self.training_steps,
            "dataset": str(self.dataset_path.name),
            "fps": self.dataset_fps,
            "chunk_size": eff_chunk,
            "n_action_steps": eff_n_actions,
            "decision_interval_s": dec_interval,
            "temporal_ensemble_coeff": self.temporal_ensemble_coeff,
            "val_loss": val_loss,
            "best_val_loss": None if not math.isfinite(self.best_val_loss) else self.best_val_loss,
            "grayscale": self.is_grayscale,
            "image_mode": "gs" if self.is_grayscale else "col",
            "image_channels": 1 if self.is_grayscale else 3,
            "augmentation": {
                "enabled": self.use_augmentation,
                "rotation": self.aug_rotation,
                "translation": self.aug_translation,
                "color": self.aug_color,
                "cutout": self.aug_cutout,
                "noise": self.aug_noise,
            },
            "updated_at": datetime.datetime.now().isoformat(),
        }
        with open(target_dir / "training_info.json", "w") as f:
            json.dump(info_dict, f, indent=2)

    def _save_checkpoint(self, policy: Any, preprocessor: Any, postprocessor: Any, step: int, save_periodic: bool = True):
        """
        Saves the latest model weights to checkpoint/ and also keeps a periodic snapshot
        in checkpoints/step_XXXXXX/ so the best generalization checkpoint can be chosen later.
        """
        latest_v_loss = self.history_val_loss[-1][1] if self.history_val_loss else None
        print(f"\n[CHECKPOINT] Saving policy weights to: {self.checkpoint_dir} (Step {step})")
        self._write_checkpoint_to_dir(self.checkpoint_dir, policy, preprocessor, postprocessor, step, val_loss=latest_v_loss)

        if save_periodic:
            step_dir = self.periodic_checkpoints_dir / f"step_{step:06d}"
            self._write_checkpoint_to_dir(step_dir, policy, preprocessor, postprocessor, step, val_loss=latest_v_loss)
            print(f"[CHECKPOINT] Saved periodic snapshot: {step_dir}")

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
        print(f"  • Dataset Frequency: {self.dataset_fps} FPS (dt = {1.0/max(1, self.dataset_fps):.3f}s)")
        if self.policy_name == "act" and self.temporal_ensemble_coeff is not None:
            print(f"  • Replanning Mode:   Temporal Ensembling Active (m={self.temporal_ensemble_coeff:.3f}, 30 Hz continuous blend)")
            print(f"  • Decision Horizon:  Chunk={self.chunk_size or 100}, Step=1 (Queried every frame @ {self.dataset_fps} FPS)")
        else:
            eff_steps = self.n_action_steps or (50 if self.policy_name == 'act' else 10)
            print(f"  • Decision Horizon:  Chunk={self.chunk_size or 100}, Execution={eff_steps} steps ({eff_steps/max(1, self.dataset_fps):.2f}s interval)")
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
            active_augs = []
            if self.aug_rotation: active_augs.append("Rotation (±10°)")
            if self.aug_translation: active_augs.append("Translation (±8% X/Y)")
            if self.aug_color: active_augs.append("Color/Brightness Jitter (±20%)")
            if self.aug_cutout: active_augs.append("Occlusion Cutout")
            if self.aug_noise: active_augs.append("Sensor Noise")
            if active_augs:
                print(f"[AUGMENT] Visual data augmentation active on GPU: {', '.join(active_augs)}")
            else:
                print("[AUGMENT] Visual data augmentation enabled, but no sub-augmentations selected.")
        else:
            print("[AUGMENT] Visual data augmentation disabled.")

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

            # Visual Data Augmentation (GPU Real-Time)
            if self.use_augmentation and (self.aug_rotation or self.aug_translation or self.aug_color or self.aug_cutout or self.aug_noise):
                for k in batch:
                    if "image" in k and isinstance(batch[k], torch.Tensor) and batch[k].ndim in (4, 5):
                        batch[k] = apply_visual_augmentation(
                            batch[k],
                            aug_rotation=self.aug_rotation,
                            aug_translation=self.aug_translation,
                            aug_color=self.aug_color,
                            aug_cutout=self.aug_cutout,
                            aug_noise=self.aug_noise,
                        )

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
                    if val_loss_scalar < self.best_val_loss:
                        self.best_val_loss = val_loss_scalar
                        self._write_checkpoint_to_dir(
                            self.best_checkpoint_dir,
                            policy,
                            preprocessor,
                            postprocessor,
                            step,
                            val_loss=val_loss_scalar,
                        )
                        print(f"[CHECKPOINT] ★ New Best Val Loss ({val_loss_scalar:.5f} at Step {step}) -> Saved {self.best_checkpoint_dir}")
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
        print(f"  • Final Model Saved:     {self.checkpoint_dir}")
        if self.best_checkpoint_dir.exists():
            print(f"  • Best Val Model Saved:  {self.best_checkpoint_dir} (val_loss={self.best_val_loss:.5f})")
        print(f"  • Periodic Snapshots:    {self.periodic_checkpoints_dir}")
        print(f"  • Loss Curve:            {self.loss_plot_file}")
        print(f"  • CSV Metrics Log:       {self.log_csv_file}")
        print("=" * 70 + "\n")

        return {
            "status": status,
            "steps_completed": step,
            "final_checkpoint": str(self.checkpoint_dir),
            "best_checkpoint": str(self.best_checkpoint_dir) if self.best_checkpoint_dir.exists() else None,
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
    parser.add_argument("--aug-rotation", action=argparse.BooleanOptionalAction, default=AUG_ROTATION, help="Random rotation (±10 deg)")
    parser.add_argument("--aug-translation", action=argparse.BooleanOptionalAction, default=AUG_TRANSLATION, help="Random translation / shifts (±8%% X/Y)")
    parser.add_argument("--aug-color", action=argparse.BooleanOptionalAction, default=AUG_COLOR, help="Random color & brightness jitter (±20%%)")
    parser.add_argument("--aug-cutout", action=argparse.BooleanOptionalAction, default=AUG_CUTOUT, help="Random cutout / occlusion patches")
    parser.add_argument("--aug-noise", action=argparse.BooleanOptionalAction, default=AUG_NOISE, help="Random Gaussian sensor noise")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="L2 weight decay")
    parser.add_argument("--save-freq", type=int, default=SAVE_CHECKPOINT_FREQ, help="Checkpoint save step frequency (latest + periodic snapshots)")
    parser.add_argument("--log-freq", type=int, default=LOG_FREQ, help="Console logging interval")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for model weights and loss curves")
    parser.add_argument("--device", type=str, default=DEVICE, help="Execution device (cuda or cpu)")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="DataLoader subprocess workers")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--grayscale", action="store_true", default=None, help="Train policy on single-channel Grayscale imagery")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Action prediction horizon steps")
    parser.add_argument("--n-action-steps", type=int, default=None, help="Action execution steps before re-querying visual observation")
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=TEMPORAL_ENSEMBLE_COEFF, help="ACT Temporal Ensembling coefficient (0.01 standard)")
    parser.add_argument("--no-temporal-ensemble", action="store_true", help="Disable ACT temporal ensembling")
    return parser.parse_args()


def main():
    args = parse_args()
    te_coeff = None if args.no_temporal_ensemble else args.temporal_ensemble_coeff

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
        aug_rotation=args.aug_rotation,
        aug_translation=args.aug_translation,
        aug_color=args.aug_color,
        aug_cutout=args.aug_cutout,
        aug_noise=args.aug_noise,
        weight_decay=args.weight_decay,
        save_checkpoint_freq=args.save_freq,
        log_freq=args.log_freq,
        output_dir=args.output_dir,
        device=args.device,
        num_workers=args.workers,
        seed=args.seed,
        grayscale=args.grayscale,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=te_coeff,
    )

    try:
        trainer.run_training()
    except KeyboardInterrupt:
        print("\n[TRAIN] Training interrupted by user (Ctrl+C). Saving emergency checkpoint...")
        trainer.abort()


if __name__ == "__main__":
    main()

