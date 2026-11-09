#!/usr/bin/env python3
"""
UR5e LeRobot — Policy Testing & Evaluation Studio (PyQt6)
=========================================================
Standalone graphical desktop application for evaluating trained imitation
learning policy checkpoints (ACT, Diffusion, VQ-BeT) on the Universal Robots UR5e.

Key Features:
- Model & Checkpoint Browser: Folder picker, auto-scan of outputs/train/*/checkpoint,
  model architecture inspection, camera requirements detection.
- Live Camera Feeds: 1 or 2 cameras (Workspace Primary + Wrist Flange Secondary)
  with device dropdowns, resolution selection, and live FPS HUD.
- Autonomous Trajectory Rollout: Multithreaded closed-loop 30 Hz execution (zero GUI freezing)
  with initial trajectory plan preview, safety jump guards, and progress bar.
- Manual Axis Control & Precision Jogging: 6-DoF joint sliders, direct numeric inputs,
  per-joint Move buttons, Move to Target, Home pose, custom Start pose saving/recall,
  and sync-to-robot.
- Compliant Freedrive Mode: Toggle freedrive for manual guidance with automatic
  interruption during commanded moves and auto-restore after rollouts.
- Live Console Log Output: Embedded timestamped terminal console with stdout/stderr redirection.
- Simulation / Mock Fallback: `--mock` flag for full offline testing without physical arm.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# PyQt6 Core & Widgets
from PyQt6.QtCore import (
    QMutex,
    QMutexLocker,
    QObject,
    QSize,
    QThread,
    QTimer,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
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
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

# Optional PyTorch & LeRobot imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy
    from lerobot.robots import UR5eRobotConfig, make_robot_from_config
    from ur5e.robot_ur5e import UR5eRobot
    LEROBOT_AVAILABLE = True
except ImportError as err:
    LEROBOT_AVAILABLE = False
    LEROBOT_IMPORT_ERR = str(err)

# Optional Gamepad Teleoperation Controller
try:
    from gamepad_controller import (
        GamepadConfig,
        UR5eGamepadTeleop,
        get_recommended_config_for_name,
    )
    GAMEPAD_AVAILABLE = True
except ImportError:
    GAMEPAD_AVAILABLE = False

# OnRobot 2FG Gripper Modbus TCP Controller
try:
    from onrobot_gripper import get_shared_gripper, OnRobot2FGModbus
    ONROBOT_GRIPPER_AVAILABLE = True
except ImportError:
    get_shared_gripper = None
    OnRobot2FGModbus = None
    ONROBOT_GRIPPER_AVAILABLE = False

# =============================================================================
# Defaults & Global Constants
# =============================================================================
DEFAULT_ROBOT_IP = "192.168.1.102"
DEFAULT_CAMERA_INDEX = 4         # HD Webcam C525 (Workspace)
DEFAULT_CAMERA_2_INDEX = 6       # eFace 1325R (Wrist Flange)
DEFAULT_FPS = 30
DEFAULT_TRAIN_DIR = Path(__file__).resolve().parent / "outputs" / "train"

UR5E_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

# Canonical UR5e Ready/Home Pose: [0°, -90°, 90°, -90°, -90°, 0°]
HOME_JOINTS = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
HOME_JOINTS_DICT = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -90.0,
    "elbow": 90.0,
    "wrist_1": -90.0,
    "wrist_2": -90.0,
    "wrist_3": 0.0,
}

# Default Start Position: [-85°, -72°, -60°, 145°, -84°, 275°]
DEFAULT_START_JOINTS = [-85.0, -72.0, -60.0, 145.0, -84.0, 275.0]
DEFAULT_START_JOINTS_DICT = {
    "shoulder_pan": -85.0,
    "shoulder_lift": -72.0,
    "elbow": -60.0,
    "wrist_1": 145.0,
    "wrist_2": -84.0,
    "wrist_3": 275.0,
}


def detect_available_cameras() -> List[Tuple[int, str]]:
    """Scans Linux v4l2 device nodes and returns (index, name)."""
    cameras = []
    v4l_dir = Path("/sys/class/video4linux")
    if v4l_dir.exists():
        for dev in sorted(v4l_dir.glob("video*"), key=lambda p: int(p.name.replace("video", "")) if p.name.replace("video", "").isdigit() else 999):
            idx_file = dev / "index"
            if idx_file.exists():
                try:
                    if idx_file.read_text().strip() != "0":
                        continue
                except Exception:
                    pass
            idx = int(dev.name.replace("video", ""))
            name_file = dev / "name"
            dev_name = name_file.read_text().strip() if name_file.exists() else f"Camera {idx}"
            clean_name = dev_name.split(":")[0].strip()
            cameras.append((idx, f"/dev/video{idx}: {clean_name}"))

    if not cameras:
        cameras = [(0, "/dev/video0: Default Camera")]
    return cameras


def scan_available_checkpoints() -> List[Tuple[str, Path]]:
    """Scans outputs/train/ for available model checkpoints."""
    checkpoints = []
    if DEFAULT_TRAIN_DIR.exists():
        for proj in sorted(DEFAULT_TRAIN_DIR.iterdir()):
            if proj.is_dir():
                ckpt = proj / "checkpoint"
                if ckpt.exists() and (ckpt / "config.json").exists():
                    checkpoints.append((proj.name, ckpt))
                elif (proj / "config.json").exists():
                    checkpoints.append((proj.name, proj))
    return checkpoints


# =============================================================================
# 1. Thread-Safe Live Log Redirection
# =============================================================================

class LogEmitter(QObject):
    log_signal = pyqtSignal(str, str)


class StreamToLogger:
    """Redirects writes to both stdout/stderr and Qt log terminal."""

    def __init__(self, emitter: LogEmitter, level: str = "INFO", original_stream=None):
        self.emitter = emitter
        self.level = level
        self.original_stream = original_stream

    def write(self, text: str):
        if text.strip():
            self.emitter.log_signal.emit(self.level, text.rstrip())
        if self.original_stream:
            self.original_stream.write(text)
            self.original_stream.flush()

    def flush(self):
        if self.original_stream:
            self.original_stream.flush()


# =============================================================================
# 2. Camera Capture Worker & Video Widget
# =============================================================================

class CameraWorker(QThread):
    frame_ready = pyqtSignal(int, np.ndarray)
    fps_updated = pyqtSignal(float)
    status_signal = pyqtSignal(int, bool, str)

    def __init__(self, camera_index: int, target_fps: int = 30, width: int = 320, height: int = 240, mock: bool = False, grayscale: bool = False):
        super().__init__()
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.mock = mock
        self._grayscale = grayscale
        self._running = False
        self._mutex = QMutex()
        self.cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None

    def set_grayscale(self, enabled: bool):
        with QMutexLocker(self._mutex):
            self._grayscale = bool(enabled)

    def update_resolution(self, width: int, height: int):
        with QMutexLocker(self._mutex):
            self.width = width
            self.height = height

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with QMutexLocker(self._mutex):
            return None if self._latest_frame is None else self._latest_frame.copy()

    def run(self):
        self._running = True
        if not self.mock:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.status_signal.emit(self.camera_index, False, f"Could not open device /dev/video{self.camera_index}")
                self.mock = True
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.status_signal.emit(self.camera_index, True, f"Opened /dev/video{self.camera_index}")

        frame_interval = 1.0 / self.target_fps
        frame_counter = 0
        fps_start = time.perf_counter()
        sim_angle = 0.0

        while self._running:
            t0 = time.perf_counter()

            with QMutexLocker(self._mutex):
                cur_w, cur_h = self.width, self.height

            if not self.mock and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    if frame.shape[1] != cur_w or frame.shape[0] != cur_h:
                        frame = cv2.resize(frame, (cur_w, cur_h), interpolation=cv2.INTER_AREA)
                    with QMutexLocker(self._mutex):
                        is_gray = self._grayscale
                    if is_gray and len(frame.shape) == 3:
                        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                    with QMutexLocker(self._mutex):
                        self._latest_frame = frame
                    self.frame_ready.emit(self.camera_index, frame)
                else:
                    time.sleep(0.01)
            else:
                # Synthetic mock camera frame
                sim_angle += 0.05
                frame = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)
                frame[:, :, 0] = int(35 + 20 * math.sin(sim_angle))
                frame[:, :, 1] = int(45 + 20 * math.cos(sim_angle))
                frame[:, :, 2] = int(55 + 15 * math.sin(sim_angle * 0.5))

                cx, cy = int(cur_w / 2 + 50 * math.cos(sim_angle)), int(cur_h / 2 + 30 * math.sin(sim_angle))
                cv2.circle(frame, (cx, cy), 18, (56, 189, 248), -1)
                cv2.putText(frame, f"CAM {self.camera_index} [SIM]", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                with QMutexLocker(self._mutex):
                    is_gray = self._grayscale
                if is_gray and len(frame.shape) == 3:
                    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                with QMutexLocker(self._mutex):
                    self._latest_frame = frame
                self.frame_ready.emit(self.camera_index, frame)

            frame_counter += 1
            t_now = time.perf_counter()
            if t_now - fps_start >= 1.0:
                calc_fps = frame_counter / (t_now - fps_start)
                self.fps_updated.emit(calc_fps)
                frame_counter = 0
                fps_start = t_now

            elapsed = time.perf_counter() - t0
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        if self.cap:
            self.cap.release()
            self.cap = None

    def stop(self):
        self._running = False
        self.wait(1500)


class VideoFeedWidget(QWidget):
    """Renders real-time camera frames with responsive scaling, aspect ratio, and HUD."""

    def __init__(self, title: str = "Primary Camera", parent=None):
        super().__init__(parent)
        self.title = title
        self._pixmap: Optional[QPixmap] = None
        self._fps: float = 0.0
        self._is_evaluating: bool = False
        self._is_grayscale: bool = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)

    def set_evaluating(self, active: bool):
        self._is_evaluating = active
        self.update()

    def set_grayscale(self, enabled: bool):
        self._is_grayscale = bool(enabled)
        self.update()

    def update_frame(self, cam_id: int, frame: np.ndarray):
        if len(frame.shape) == 2:
            h, w = frame.shape
            bytes_per_line = w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(q_img)
        self.update()

    def update_fps(self, fps: float):
        self._fps = fps
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#0b1120"))

        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QColor("#64748b"))
            painter.setFont(QFont("sans-serif", 12, QFont.Weight.Bold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{self.title}\nWaiting for video feed...")

        # HUD Overlay Banner
        hud_rect_w = 210
        hud_rect_h = 24
        painter.fillRect(10, 10, hud_rect_w, hud_rect_h, QColor(15, 23, 42, 210))
        painter.setPen(QColor("#38bdf8"))
        painter.setFont(QFont("monospace", 10, QFont.Weight.Bold))
        mode_str = "GRAY" if self._is_grayscale else "RGB"
        painter.drawText(18, 27, f"{self.title} | {self._fps:4.1f} FPS [{mode_str}]")

        if self._is_evaluating:
            # Active Rollout Badge
            eval_badge_w = 155
            painter.fillRect(self.width() - eval_badge_w - 12, 10, eval_badge_w, hud_rect_h, QColor(22, 163, 74, 230))
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("sans-serif", 9, QFont.Weight.Bold))
            painter.drawText(self.width() - eval_badge_w - 4, 27, "● EVALUATION ACTIVE")

        painter.end()


# =============================================================================
# 3. Robot RTDE Worker (Telemetry, Freedrive, Jogging)
# =============================================================================

class RobotRTDEWorker(QThread):
    telemetry_signal = pyqtSignal(dict)
    freedrive_state_changed = pyqtSignal(bool)
    connection_state_changed = pyqtSignal(bool, str)

    def __init__(self, robot_ip: str = DEFAULT_ROBOT_IP, mock: bool = False, parent=None):
        super().__init__(parent)
        self.robot_ip = robot_ip
        self.mock = mock
        self._running = False
        self._robot = None
        self._freedrive_active = False

        # Command queues & thread synchronization
        self._cmd_lock = QMutex()
        self._cmd_toggle_freedrive = False
        self._cmd_disconnect = False
        self._cmd_target_pose: Optional[Dict[str, float]] = None

        # Commanded motion state & automatic freedrive restoration
        self._motion_in_progress = False
        self._motion_restore_freedrive = False
        self._motion_target_q = [0.0] * 6
        self._motion_start_time = 0.0
        self._motion_timeout = 5.0
        self._motion_last_disp = 999.0
        self._motion_last_progress_time = 0.0

        # Latest joint states (radians)
        self._latest_q = {j: 0.0 for j in UR5E_JOINTS}
        self._mock_cartesian_v: List[float] = [0.0] * 6

    def get_latest_joints(self) -> Dict[str, float]:
        with QMutexLocker(self._cmd_lock):
            return dict(self._latest_q)

    def set_mock_cartesian_velocity(self, v: List[float]):
        with QMutexLocker(self._cmd_lock):
            self._mock_cartesian_v = list(v)

    @property
    def is_freedrive_active(self) -> bool:
        return self._freedrive_active

    def request_toggle_freedrive(self):
        with QMutexLocker(self._cmd_lock):
            self._cmd_toggle_freedrive = True

    def request_move_pose(self, pose: Dict[str, float]):
        with QMutexLocker(self._cmd_lock):
            self._cmd_target_pose = dict(pose)

    def request_disconnect(self):
        with QMutexLocker(self._cmd_lock):
            self._cmd_disconnect = True

    def stream_servoj(self, target_q: List[float], dt: float = 0.033):
        """Streams a servoj position command smoothly to the arm."""
        if not self.mock and self._robot and hasattr(self._robot, "sock") and self._robot.sock:
            q_str = ", ".join(f"{val:.5f}" for val in target_q)
            cmd = f"servoj([{q_str}], t={dt:.3f}, lookahead_time=0.1, gain=300)\n"
            try:
                self._robot.sock.sendall(cmd.encode("utf-8"))
            except Exception as e:
                print(f"[ERROR] servoj transmission failed: {e}")
        elif self.mock:
            with QMutexLocker(self._cmd_lock):
                for idx, j_name in enumerate(UR5E_JOINTS):
                    self._latest_q[j_name] = float(target_q[idx])

    def stop_motion(self):
        """Halts motion immediately on the robot."""
        if not self.mock and self._robot and hasattr(self._robot, "sock") and self._robot.sock:
            try:
                self._robot.sock.sendall(b"stopj(2.0)\n")
            except Exception:
                pass

    def run(self):
        self._running = True

        # Connect to UR5e or fallback to simulated mock
        if not self.mock and LEROBOT_AVAILABLE:
            try:
                from ur5e.configuration_ur5e import UR5eRobotConfig
                from ur5e.robot_ur5e import UR5eRobot
                config = UR5eRobotConfig(robot_ip=self.robot_ip)
                self._robot = UR5eRobot(config)
                self._robot.connect()
                self.connection_state_changed.emit(True, f"Connected to UR5e at {self.robot_ip}")
            except Exception as e:
                self.connection_state_changed.emit(False, f"Connection failed ({e}). Entering Mock Robot Mode.")
                self.mock = True
                self._robot = None
        else:
            self.connection_state_changed.emit(True, f"Mock Robot Mode active (Simulated UR5e on {self.robot_ip})")

        sim_t = 0.0
        mock_angles = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]

        target_period = 1.0 / 125.0
        last_emit_time = time.perf_counter()
        emit_interval = 1.0 / 30.0

        sample_count = 0
        perf_window_start = time.perf_counter()
        measured_rate = 125.0
        read_latency_ms = 1.2

        while self._running:
            t0 = time.perf_counter()

            # Process queued commands
            with QMutexLocker(self._cmd_lock):
                if self._cmd_disconnect:
                    break
                if self._cmd_toggle_freedrive:
                    self._cmd_toggle_freedrive = False
                    self._motion_restore_freedrive = False
                    new_state = not self._freedrive_active
                    try:
                        if self._robot and hasattr(self._robot, "enable_freedrive"):
                            if new_state:
                                self._robot.enable_freedrive()
                            else:
                                self._robot.disable_freedrive()
                        self._freedrive_active = new_state
                        self.freedrive_state_changed.emit(self._freedrive_active)
                    except Exception as err:
                        print(f"[ERROR] Failed to toggle Freedrive: {err}")

                if self._cmd_target_pose is not None:
                    target_pose = self._cmd_target_pose
                    self._cmd_target_pose = None

                    was_freedrive = self._freedrive_active or self._motion_restore_freedrive
                    if self._freedrive_active:
                        try:
                            if self._robot and hasattr(self._robot, "disable_freedrive"):
                                self._robot.disable_freedrive()
                            self._freedrive_active = False
                            self.freedrive_state_changed.emit(False)
                            print("[MOTION] Freedrive temporarily paused for commanded motion.")
                        except Exception as err:
                            print(f"[ERROR] Failed to disable Freedrive before move: {err}")

                    self._motion_restore_freedrive = was_freedrive
                    self._motion_target_q = [float(target_pose[f"{j}.pos"]) for j in UR5E_JOINTS]
                    self._motion_in_progress = True
                    self._motion_start_time = time.perf_counter()
                    self._motion_last_progress_time = self._motion_start_time

                    max_disp = max(abs(self._latest_q[j] - target_pose[f"{j}.pos"]) for j in UR5E_JOINTS)
                    self._motion_last_disp = max_disp

                    # Timeout based on actual UR5e velocity (0.2 rad/s) with generous headroom
                    cmd_v = getattr(getattr(self._robot, "config", None), "velocity", 0.2) or 0.2
                    self._motion_timeout = max(15.0, (max_disp / max(0.08, cmd_v * 0.6)) + 8.0)

                    if self._robot and hasattr(self._robot, "send_action"):
                        try:
                            self._robot.send_action(target_pose)
                        except Exception as err:
                            print(f"[ERROR] Motion command failed: {err}")
                            self._motion_in_progress = False
                            self._motion_restore_freedrive = False

            # Telemetry acquisition
            if not self.mock and self._robot and self._robot.rx:
                try:
                    q_list = self._robot.rx.getActualQ()
                    for idx, j_name in enumerate(UR5E_JOINTS):
                        self._latest_q[j_name] = float(q_list[idx])
                except Exception as err:
                    print(f"[WARNING] RTDE read exception: {err}")

                # Monitor motion completion
                if self._motion_in_progress:
                    current_disp = max(abs(self._latest_q[j] - self._motion_target_q[idx]) for idx, j in enumerate(UR5E_JOINTS))
                    now = time.perf_counter()
                    elapsed = now - self._motion_start_time

                    # Refresh progress timestamp whenever the arm moves closer by >= 0.003 rad (~0.17°)
                    if current_disp < self._motion_last_disp - 0.003:
                        self._motion_last_disp = current_disp
                        self._motion_last_progress_time = now

                    time_since_progress = now - self._motion_last_progress_time

                    if current_disp < 0.02:
                        self._motion_in_progress = False
                        print(f"[MOTION] Target reached in {elapsed:.2f}s.")
                        if self._motion_restore_freedrive:
                            self._motion_restore_freedrive = False
                            try:
                                if hasattr(self._robot, "enable_freedrive"):
                                    self._robot.enable_freedrive()
                                self._freedrive_active = True
                                self.freedrive_state_changed.emit(True)
                                print("[MOTION] Freedrive automatically restored.")
                            except Exception as err:
                                print(f"[ERROR] Failed to restore Freedrive: {err}")
                    elif elapsed >= self._motion_timeout and time_since_progress >= 4.0:
                        # Only time out if total timeout expired AND no progress made for 4.0 seconds
                        self._motion_in_progress = False
                        self._motion_restore_freedrive = False
                        print(f"[WARNING] Motion timed out after {elapsed:.1f}s without progress (max joint err: {math.degrees(current_disp):.1f}°).")
            else:
                # Synthetic realistic UR5e telemetry
                sim_t += target_period
                if self._motion_in_progress:
                    max_err = 0.0
                    for idx in range(6):
                        err = self._motion_target_q[idx] - mock_angles[idx]
                        max_err = max(max_err, abs(err))
                        step = math.copysign(min(abs(err), 1.5 * target_period), err)
                        mock_angles[idx] += step
                    elapsed = time.perf_counter() - self._motion_start_time
                    if max_err < 0.005:
                        self._motion_in_progress = False
                        print(f"[MOTION] Mock target reached ({elapsed:.2f}s).")
                        if self._motion_restore_freedrive:
                            self._motion_restore_freedrive = False
                            self._freedrive_active = True
                            self.freedrive_state_changed.emit(True)
                            print("[MOTION] Freedrive automatically restored.")
                    elif elapsed >= self._motion_timeout:
                        self._motion_in_progress = False
                        self._motion_restore_freedrive = False
                        print(f"[WARNING] Mock motion timed out after {elapsed:.1f}s.")
                elif self._freedrive_active:
                    for i in range(6):
                        mock_angles[i] += 0.001 * math.sin(sim_t * (1 + i * 0.4))
                elif self._mock_cartesian_v and any(abs(x) > 1e-4 for x in self._mock_cartesian_v):
                    dt = target_period
                    vx, vy, vz, w_base, w_wrist2, w_wrist3 = self._mock_cartesian_v
                    theta_base = mock_angles[0]
                    vfwd = vx * math.cos(theta_base) + vy * math.sin(theta_base)
                    vlat = -vx * math.sin(theta_base) + vy * math.cos(theta_base)
                    mock_angles[0] += (vlat * 1.2 + w_base) * dt
                    mock_angles[1] += vfwd * dt * 1.2
                    mock_angles[2] -= vfwd * dt * 1.2
                    mock_angles[1] -= vz * dt * 1.2
                    mock_angles[4] += w_wrist2 * dt
                    mock_angles[5] += w_wrist3 * dt
                for idx, j_name in enumerate(UR5E_JOINTS):
                    self._latest_q[j_name] = mock_angles[idx]

            sample_count += 1
            t_now = time.perf_counter()
            read_latency_ms = (t_now - t0) * 1000.0

            if t_now - perf_window_start >= 0.5:
                measured_rate = sample_count / (t_now - perf_window_start)
                sample_count = 0
                perf_window_start = t_now

            if t_now - last_emit_time >= emit_interval:
                self.telemetry_signal.emit({
                    "joints": dict(self._latest_q),
                    "rtde_hz": measured_rate,
                    "latency_ms": read_latency_ms,
                    "freedrive": self._freedrive_active,
                })
                last_emit_time = t_now

            elapsed = time.perf_counter() - t0
            sleep_t = target_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        if self._robot:
            try:
                if self._freedrive_active:
                    self._robot.disable_freedrive()
                self._robot.disconnect()
            except Exception:
                pass
            self._robot = None
        self._running = False

    def get_latest_joints(self) -> Dict[str, float]:
        return dict(self._latest_q)

    def stop(self):
        self._running = False
        self.request_disconnect()
        self.wait(2000)


# =============================================================================
# 4. Asynchronous Policy Rollout Evaluation Worker
# =============================================================================

class PolicyRolloutWorker(QThread):
    """
    Executes trained policy evaluation in a background worker thread.
    Manages inference loops, safe servoj streaming, progress updates,
    and automatic post-test resets without freezing the Qt UI.
    """
    plan_preview_ready = pyqtSignal(dict)
    step_progress = pyqtSignal(int, int, float, list)
    rollout_finished = pyqtSignal(bool, str)

    def __init__(
        self,
        model_path: Path,
        robot_worker: RobotRTDEWorker,
        camera_worker_1: CameraWorker,
        camera_worker_2: Optional[CameraWorker],
        duration_s: float = 10.0,
        control_fps: int = 30,
        max_jump_deg: float = 45.0,
        n_action_steps: Optional[int] = None,
        temporal_ensemble: bool = True,
        temporal_ensemble_coeff: float = 0.01,
        auto_freedrive_after: bool = True,
        return_start_after: bool = False,
        start_pose_deg: Optional[Dict[str, float]] = None,
        device_name: str = "cuda" if (torch and torch.cuda.is_available()) else "cpu",
        mock: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.model_path = model_path
        self.robot_worker = robot_worker
        self.camera_worker_1 = camera_worker_1
        self.camera_worker_2 = camera_worker_2
        self.duration_s = duration_s
        self.control_fps = control_fps
        self.max_jump_deg = max_jump_deg
        self.n_action_steps = n_action_steps
        self.temporal_ensemble = temporal_ensemble
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        self.auto_freedrive_after = auto_freedrive_after
        self.return_start_after = return_start_after
        self.start_pose_deg = start_pose_deg or DEFAULT_START_JOINTS_DICT
        self.device_name = device_name
        self.mock = mock
        self._abort_requested = False

    def request_abort(self):
        self._abort_requested = True

    def run(self):
        print("\n" + "=" * 75)
        print("      UR5e Autonomous Policy Evaluation & Trajectory Rollout")
        print("=" * 75)
        print(f"  • Checkpoint Path:    {self.model_path}")
        print(f"  • Target Duration:    {self.duration_s:.1f}s ({int(self.duration_s * self.control_fps)} steps @ {self.control_fps} Hz)")
        print(f"  • Max Joint Jump:     ±{self.max_jump_deg}°")
        print(f"  • Post-Test Reset:    {'Return to Start Pose' if self.return_start_after else ('Auto-Restore Freedrive' if self.auto_freedrive_after else 'Position Hold')}")
        print("=" * 75 + "\n")

        # 1. Load Policy and Pre/Post Processors
        print("[1/4] Loading policy weights and normalizers...")
        device = torch.device(self.device_name if (self.device_name == "cuda" and torch.cuda.is_available()) else "cpu")
        try:
            policy = ACTPolicy.from_pretrained(self.model_path).to(device)
            policy.eval()
            preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=self.model_path)
            print(f"      ✔ Policy loaded: {policy.__class__.__name__} on {device}")

            if self.temporal_ensemble and hasattr(policy, "temporal_ensembler"):
                try:
                    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
                    policy.config.temporal_ensemble_coeff = float(self.temporal_ensemble_coeff)
                    policy.config.n_action_steps = 1
                    policy.temporal_ensembler = ACTTemporalEnsembler(
                        float(self.temporal_ensemble_coeff),
                        getattr(policy.config, "chunk_size", 100),
                    )
                    policy.reset()
                    print(f"  • Replanning Mode:    Temporal Ensembling Active (m={self.temporal_ensemble_coeff:.3f}, 30 Hz continuous blend)")
                except Exception as e:
                    print(f"  • Note: Temporal ensembling initialization fallback ({e}). Using open-loop chunks.")
                    if hasattr(policy.config, "temporal_ensemble_coeff"):
                        policy.config.temporal_ensemble_coeff = None
            else:
                if hasattr(policy.config, "temporal_ensemble_coeff"):
                    policy.config.temporal_ensemble_coeff = None
                if self.n_action_steps is not None and hasattr(policy.config, "n_action_steps"):
                    policy.config.n_action_steps = int(self.n_action_steps)
                if hasattr(policy, "drop_queued_actions"):
                    policy.drop_queued_actions()
                policy.reset()
                curr_steps = getattr(policy.config, "n_action_steps", 10)
                dt_dec = curr_steps / max(1, self.control_fps)
                print(f"  • Replanning Mode:    Open-Loop Chunks ({curr_steps} steps, {dt_dec:.2f}s @ {self.control_fps} FPS)")
        except Exception as err:
            msg = f"Failed to load policy checkpoint: {err}"
            print(f"[ERROR] {msg}")
            self.rollout_finished.emit(False, msg)
            return

        # Check if model was trained in grayscale mode
        is_model_grayscale = False
        model_dir = Path(self.model_path)
        for info_p in [model_dir / "training_info.json", model_dir.parent / "training_info.json"]:
            if info_p.exists():
                try:
                    with open(info_p, "r") as f_info:
                        t_info = json.load(f_info)
                        if t_info.get("grayscale") is True or t_info.get("image_channels", 3) == 1:
                            is_model_grayscale = True
                            break
                except Exception:
                    pass

        # Inspect camera requirements
        img_features = getattr(policy.config, "image_features", {}) or getattr(policy.config, "input_features", {})
        has_wrist_cam = "observation.images.wrist" in img_features
        req_h, req_w = 240, 320
        if "observation.images.workspace" in img_features:
            shape = getattr(img_features["observation.images.workspace"], "shape", None)
            if shape and len(shape) >= 2:
                req_h, req_w = shape[-2], shape[-1]

        # 2. Capture Initial Observation & Trajectory Preview
        print("\n[2/4] Capturing current observation and computing trajectory plan preview...")
        cam1_frame = self.camera_worker_1.get_latest_frame()
        if cam1_frame is None:
            cam1_frame = np.zeros((req_h, req_w, 3), dtype=np.uint8)
        if cam1_frame.shape[1] != req_w or cam1_frame.shape[0] != req_h:
            cam1_frame = cv2.resize(cam1_frame, (req_w, req_h), interpolation=cv2.INTER_AREA)
        if is_model_grayscale and len(cam1_frame.shape) == 3:
            cam1_frame = cv2.cvtColor(cv2.cvtColor(cam1_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        current_joints = self.robot_worker.get_latest_joints()
        start_q_rad = [current_joints.get(j, 0.0) for j in UR5E_JOINTS]
        start_q_deg = [math.degrees(r) for r in start_q_rad]

        # Single inference step for trajectory preview
        img_rgb = cv2.cvtColor(cam1_frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        state_tensor = torch.from_numpy(np.array(start_q_rad, dtype=np.float32)).unsqueeze(0).to(device)

        batch = {
            "observation.state": state_tensor,
            "observation.images.workspace": img_tensor,
        }
        if has_wrist_cam and self.camera_worker_2:
            cam2_frame = self.camera_worker_2.get_latest_frame()
            if cam2_frame is None:
                cam2_frame = np.zeros((req_h, req_w, 3), dtype=np.uint8)
            if cam2_frame.shape[1] != req_w or cam2_frame.shape[0] != req_h:
                cam2_frame = cv2.resize(cam2_frame, (req_w, req_h), interpolation=cv2.INTER_AREA)
            if is_model_grayscale and len(cam2_frame.shape) == 3:
                cam2_frame = cv2.cvtColor(cv2.cvtColor(cam2_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            wrist_rgb = cv2.cvtColor(cam2_frame, cv2.COLOR_BGR2RGB)
            batch["observation.images.wrist"] = torch.from_numpy(wrist_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            pre_batch = preprocessor(batch)
            pred_raw = policy.select_action(pre_batch)
            pred_post = postprocessor(pred_raw)

        first_target_rad = pred_post.squeeze().detach().cpu().numpy()
        if first_target_rad.ndim > 1:
            first_target_rad = first_target_rad[0].tolist()
        else:
            first_target_rad = first_target_rad.tolist()

        first_target_deg = [math.degrees(r) for r in first_target_rad]
        deltas_deg = [t - c for t, c in zip(first_target_deg, start_q_deg)]
        max_delta_deg = max(abs(d) for d in deltas_deg)

        # Print clean trajectory plan preview table to live console
        print("\n" + "=" * 75)
        print("                    >>> PREDICTED TRAJECTORY PREVIEW <<<")
        print("=" * 75)
        print(f" {'Joint':<16} | {'Current Pos':<16} | {'First Target':<16} | {'Initial Shift':<14}")
        print("-" * 75)
        for j_name, c_deg, c_rad, t_deg, t_rad, d_deg in zip(
            UR5E_JOINTS, start_q_deg, start_q_rad, first_target_deg, first_target_rad, deltas_deg
        ):
            status = "WARN" if abs(d_deg) > self.max_jump_deg else "OK"
            print(f" {j_name:<16} | {c_deg:+7.2f}° ({c_rad:+.3f}) | {t_deg:+7.2f}° ({t_rad:+.3f}) | {d_deg:+6.2f}° [{status}]")
        print("-" * 75)
        print(f" Max Initial Shift:           {max_delta_deg:.2f}°")
        print(f" Rollout Duration:            {self.duration_s:.1f}s ({int(self.duration_s * self.control_fps)} steps)")
        print("=" * 75 + "\n")

        self.plan_preview_ready.emit({
            "current_deg": start_q_deg,
            "target_deg": first_target_deg,
            "deltas_deg": deltas_deg,
            "max_delta_deg": max_delta_deg,
        })

        if max_delta_deg > self.max_jump_deg:
            print(f"[SAFETY WARNING] Initial predicted shift of {max_delta_deg:.1f}° exceeds safety limit (±{self.max_jump_deg}°).")

        # 3. Disable Freedrive for Autonomous Rollout
        print("[3/4] Ensuring Freedrive mode is deactivated for autonomous execution...")
        if self.robot_worker.is_freedrive_active:
            self.robot_worker.request_toggle_freedrive()
            time.sleep(0.3)

        # 4. Continuous Closed-Loop Rollout Loop
        total_steps = max(1, int(self.duration_s * self.control_fps))
        loop_period = 1.0 / self.control_fps
        policy.reset()
        last_target_q = list(start_q_rad)
        max_jump_rad = math.radians(self.max_jump_deg)

        print(f"[4/4] Executing autonomous rollout ({total_steps} steps)...")
        step_count = 0
        t_rollout_start = time.perf_counter()

        try:
            while step_count < total_steps and not self._abort_requested:
                t_tick_start = time.perf_counter()
                step_count += 1

                # 1. Grab camera frame
                frame = self.camera_worker_1.get_latest_frame()
                if frame is None:
                    frame = np.zeros((req_h, req_w, 3), dtype=np.uint8)
                if frame.shape[1] != req_w or frame.shape[0] != req_h:
                    frame = cv2.resize(frame, (req_w, req_h), interpolation=cv2.INTER_AREA)

                # 2. Grab current joint angles
                obs = self.robot_worker.get_latest_joints()
                cur_q = [obs.get(j, 0.0) for j in UR5E_JOINTS]

                # 3. Model inference
                if is_model_grayscale and len(frame.shape) == 3:
                    frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                state_t = torch.from_numpy(np.array(cur_q, dtype=np.float32)).unsqueeze(0).to(device)

                step_batch = {
                    "observation.state": state_t,
                    "observation.images.workspace": img_t,
                }
                if has_wrist_cam and self.camera_worker_2:
                    w_frame = self.camera_worker_2.get_latest_frame()
                    if w_frame is None:
                        w_frame = np.zeros((req_h, req_w, 3), dtype=np.uint8)
                    if w_frame.shape[1] != req_w or w_frame.shape[0] != req_h:
                        w_frame = cv2.resize(w_frame, (req_w, req_h), interpolation=cv2.INTER_AREA)
                    if is_model_grayscale and len(w_frame.shape) == 3:
                        w_frame = cv2.cvtColor(cv2.cvtColor(w_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                    step_batch["observation.images.wrist"] = torch.from_numpy(
                        cv2.cvtColor(w_frame, cv2.COLOR_BGR2RGB)
                    ).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

                with torch.no_grad():
                    p_batch = preprocessor(step_batch)
                    a_raw = policy.select_action(p_batch)
                    a_post = postprocessor(a_raw)

                step_target = a_post.squeeze().detach().cpu().numpy()
                if step_target.ndim > 1:
                    step_target = step_target[0].tolist()
                else:
                    step_target = step_target.tolist()

                # 4. Clamp single-step delta against safety limits
                clamped_target = []
                for tgt, last in zip(step_target, last_target_q):
                    d = tgt - last
                    d = max(-max_jump_rad, min(max_jump_rad, d))
                    clamped_target.append(last + d)
                last_target_q = list(clamped_target)

                # 5. Stream action to robot
                self.robot_worker.stream_servoj(clamped_target, dt=loop_period)

                elapsed_s = time.perf_counter() - t_rollout_start
                self.step_progress.emit(step_count, total_steps, elapsed_s, clamped_target)

                # Sleep to maintain control loop frequency
                t_elapsed = time.perf_counter() - t_tick_start
                sleep_rem = loop_period - t_elapsed
                if sleep_rem > 0:
                    time.sleep(sleep_rem)

            # End of rollout
            self.robot_worker.stop_motion()

            if self._abort_requested:
                msg = f"Rollout aborted by user at step {step_count}/{total_steps}."
                print(f"\n[ABORT] {msg}")
                success = False
            else:
                elapsed_total = time.perf_counter() - t_rollout_start
                msg = f"Rollout completed successfully! {step_count}/{total_steps} steps in {elapsed_total:.2f}s."
                print(f"\n[SUCCESS] {msg}")
                success = True

        except Exception as err:
            self.robot_worker.stop_motion()
            msg = f"Exception during rollout: {err}"
            print(f"[ERROR] {msg}")
            success = False

        # 5. Post-Test Behavior
        if self.return_start_after:
            print("\n[POST-TEST] Returning arm to stored Start Position...")
            start_action = {f"{j}.pos": math.radians(self.start_pose_deg.get(j, 0.0)) for j in UR5E_JOINTS}
            self.robot_worker.request_move_pose(start_action)
            time.sleep(1.0)
        elif self.auto_freedrive_after:
            print("\n[POST-TEST] Automatically restoring Freedrive mode for manual arm guidance...")
            if not self.robot_worker.is_freedrive_active:
                self.robot_worker.request_toggle_freedrive()

        self.rollout_finished.emit(success, msg)


# =============================================================================
# 5. Main Policy Testing Studio Window
# =============================================================================

class UR5ePolicyTesterStudio(QMainWindow):
    """
    Main Policy Testing & Evaluation Studio GUI Window.
    Coordinates model selection, live camera HUD, RTDE manual jogging,
    and background closed-loop rollout execution.
    """

    def __init__(self, robot_ip: str = DEFAULT_ROBOT_IP, model_path: Optional[str] = None, mock: bool = False):
        super().__init__()
        self.robot_ip = robot_ip
        self.mock_mode = mock
        self.target_fps = DEFAULT_FPS
        self.camera_1_index = DEFAULT_CAMERA_INDEX
        self.camera_2_index = DEFAULT_CAMERA_2_INDEX
        self.enable_dual_cameras = False
        self.is_grayscale = False
        self.model_required_cams = 1
        self.model_camera_names = ["Workspace"]

        # Model Checkpoint Path
        if model_path:
            self.active_model_path = Path(model_path).resolve()
        else:
            detected = scan_available_checkpoints()
            if detected:
                self.active_model_path = detected[0][1]
            else:
                self.active_model_path = Path(DEFAULT_MODEL_PATH).resolve() if "DEFAULT_MODEL_PATH" in globals() else (DEFAULT_TRAIN_DIR / "checkpoint")

        # Custom Start Position (defaults to DEFAULT_START_JOINTS_DICT: -85°, -72°, -60°, 145°, -84°, 275°)
        self.start_position: Dict[str, float] = dict(DEFAULT_START_JOINTS_DICT)
        self.freedrive_active: bool = False
        self._user_dragging_joint: Optional[str] = None
        self._user_modified_joints: set[str] = set()
        self._user_targets: Dict[str, float] = {}

        self.jog_sliders: Dict[str, QSlider] = {}
        self.jog_inputs: Dict[str, QLineEdit] = {}
        self.jog_move_buttons: Dict[str, QPushButton] = {}
        self.joint_labels: Dict[str, QLabel] = {}
        self.joint_bars: Dict[str, QProgressBar] = {}

        # Rollout thread reference
        self.rollout_worker: Optional[PolicyRolloutWorker] = None

        # Window properties
        self.setWindowTitle("UR5e LeRobot — Policy Testing & Autonomous Rollout Studio")
        self.resize(1420, 980)
        self.setMinimumSize(1150, 820)

        self.apply_theme()
        self.setup_ui()

        # Log redirection
        self.log_emitter = LogEmitter()
        self.log_emitter.log_signal.connect(self.append_log)
        self.stdout_redirect = StreamToLogger(self.log_emitter, "INFO", original_stream=sys.__stdout__)
        self.stderr_redirect = StreamToLogger(self.log_emitter, "ERROR", original_stream=sys.__stderr__)
        sys.stdout = self.stdout_redirect
        sys.stderr = self.stderr_redirect

        # Catch unhandled exceptions
        def excepthook(exc_type, exc_value, exc_tb):
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.__stderr__)
            self.log_emitter.log_signal.emit("CRITICAL", f"Unhandled exception: {exc_value}")
        sys.excepthook = excepthook

        # Setup Hotkeys
        self.setup_shortcuts()

        # Start Camera Worker (Primary)
        self.camera_worker_1 = CameraWorker(self.camera_1_index, self.target_fps, 320, 240, self.mock_mode, self.is_grayscale)
        self.camera_worker_1.frame_ready.connect(self.cam_widget_1.update_frame)
        self.camera_worker_1.fps_updated.connect(self.cam_widget_1.update_fps)
        self.camera_worker_1.start()

        # Camera Worker (Secondary)
        self.camera_worker_2: Optional[CameraWorker] = None

        # Start Robot RTDE Worker
        self.robot_worker = RobotRTDEWorker(self.robot_ip, self.mock_mode)
        self.robot_worker.telemetry_signal.connect(self.on_telemetry_updated)
        self.robot_worker.freedrive_state_changed.connect(self.on_freedrive_state_changed)
        self.robot_worker.connection_state_changed.connect(self.on_robot_connection_status)
        self.robot_worker.start()

        # Initialize OnRobot 2FG Gripper
        self._user_dragging_gripper: bool = False
        self.gripper: Optional[OnRobot2FGModbus] = None
        if ONROBOT_GRIPPER_AVAILABLE and get_shared_gripper is not None:
            try:
                self.gripper = get_shared_gripper(auto_connect=not self.mock_mode)
                if self.gripper and self.gripper.is_connected:
                    init_w = self.gripper.get_width()
                    if hasattr(self, "gripper_slider"):
                        self.gripper_slider.blockSignals(True)
                        self.gripper_slider.setValue(int(round(init_w)))
                        self.gripper_slider.blockSignals(False)
                    if hasattr(self, "gripper_input"):
                        self.gripper_input.setText(f"{init_w:.1f}")
            except Exception as e:
                print(f"[GRIPPER] Init notice: {e}")

        # Start Gamepad Teleoperation Controller
        self.gamepad_teleop: Optional[UR5eGamepadTeleop] = None
        self.gamepad_teleop_active: bool = False
        if GAMEPAD_AVAILABLE:
            try:
                self.gamepad_teleop = UR5eGamepadTeleop(
                    robot_ip=self.robot_ip,
                    mock=self.mock_mode,
                    parent=self,
                )
                if self.gripper is not None:
                    self.gamepad_teleop.set_gripper(self.gripper)
                    self.gamepad_teleop.gripper_width_updated.connect(self.on_gamepad_gripper_width_updated)
                self.gamepad_teleop.set_base_angle_provider(
                    lambda: self.robot_worker.get_latest_joints().get("shoulder_pan", 0.0)
                )
                self.gamepad_teleop.set_joint_positions_provider(
                    lambda: self.robot_worker.get_latest_joints()
                )
                if hasattr(self, "slider_gamepad_speed"):
                    self.gamepad_teleop.set_speed_percentage(self.slider_gamepad_speed.value())
                if hasattr(self, "slider_gamepad_accel"):
                    self.gamepad_teleop.set_accel_percentage(self.slider_gamepad_accel.value())
                if self.mock_mode:
                    self.gamepad_teleop.set_velocity_callback(
                        lambda v: self.robot_worker.set_mock_cartesian_velocity(v)
                    )
                self.gamepad_teleop.status_updated.connect(self.on_gamepad_status_updated)
                self.gamepad_teleop.velocity_updated.connect(self.on_gamepad_velocity_updated)
                self.gamepad_teleop.teleop_state_changed.connect(self.on_gamepad_state_changed)
                self.gamepad_teleop.button_select_pressed.connect(self.on_gamepad_select_clicked)
                self.gamepad_teleop.button_start_pressed.connect(self.on_gamepad_start_clicked)
                self.gamepad_teleop.start()
            except Exception as e:
                print(f"[GAMEPAD WARNING] Could not start Gamepad Teleoperation: {e}", file=sys.__stderr__)
                self.gamepad_teleop = None

        # Inspect initial model path
        self.inspect_selected_model()

        print(f"[SYSTEM] Policy Testing Studio Initialized | Robot IP: {self.robot_ip}")
        if self.mock_mode:
            print("[INFO] Application running in SIMULATION/MOCK mode. No physical arm required.")

    # -------------------------------------------------------------------------
    # Styling & Visual Theme
    # -------------------------------------------------------------------------
    def apply_theme(self):
        palette_css = """
        QMainWindow, QWidget {
            background-color: #0f172a;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 12px;
        }
        QGroupBox {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            margin-top: 24px;
            font-weight: bold;
            font-size: 11px;
            color: #94a3b8;
            padding-top: 14px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            top: 6px;
            padding: 0 6px;
            background-color: #1e293b;
            border-radius: 4px;
        }
        QLineEdit, QComboBox, QPlainTextEdit, QDoubleSpinBox {
            background-color: #0b1120;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 5px;
            color: #f1f5f9;
            selection-background-color: #2563eb;
        }
        QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QDoubleSpinBox:focus {
            border: 1px solid #38bdf8;
        }
        QPushButton {
            background-color: #334155;
            color: #ffffff;
            border: 1px solid #475569;
            border-radius: 6px;
            padding: 6px 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #475569;
        }
        QPushButton:pressed {
            background-color: #1e293b;
        }
        QProgressBar {
            background-color: #0b1120;
            border: 1px solid #334155;
            border-radius: 4px;
            text-align: center;
            font-size: 10px;
            color: #cbd5e1;
        }
        QProgressBar::chunk {
            background-color: #38bdf8;
            border-radius: 3px;
        }
        QSplitter::handle {
            background-color: #1e293b;
            height: 4px;
            width: 4px;
        }
        QStatusBar {
            background-color: #0b1120;
            border-top: 1px solid #1e293b;
        }
        """
        self.setStyleSheet(palette_css)

    # -------------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------------
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(8)

        # Horizontal Main Splitter: Left (Model & Rollout & Video) | Right (Telemetry, Jog, Console)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ---------------------------------------------------------------------
        # LEFT COLUMN: Model Selection, Evaluation Controls, Camera Feeds
        # ---------------------------------------------------------------------
        left_widget = QWidget()
        left_column = QVBoxLayout(left_widget)
        left_column.setContentsMargins(4, 4, 4, 4)
        left_column.setSpacing(8)

        # 1. Model & Checkpoint Selection Card
        model_box = QGroupBox("TRAINED MODEL & POLICY CHECKPOINT")
        model_layout = QVBoxLayout(model_box)
        model_layout.setContentsMargins(10, 10, 10, 10)
        model_layout.setSpacing(6)

        row_picker = QHBoxLayout()
        lbl_model = QLabel("Model Checkpoint:")
        lbl_model.setStyleSheet("font-weight: bold; color: #94a3b8;")
        self.txt_model_path = QLineEdit(str(self.active_model_path))
        self.txt_model_path.setToolTip("Path to the checkpoint folder containing config.json and model.safetensors")
        self.txt_model_path.returnPressed.connect(self.on_model_path_committed)

        self.btn_browse_model = QPushButton("📁 Browse...")
        self.btn_browse_model.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_browse_model.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                font-size: 11px;
                padding: 5px 10px;
                border: 1px solid #38bdf8;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
        """)
        self.btn_browse_model.clicked.connect(self.on_browse_model_folder)

        row_picker.addWidget(lbl_model)
        row_picker.addWidget(self.txt_model_path, stretch=1)
        row_picker.addWidget(self.btn_browse_model)
        model_layout.addLayout(row_picker)

        # Quick-Selector for Detected Checkpoints
        row_detect = QHBoxLayout()
        lbl_quick = QLabel("Quick Select:")
        lbl_quick.setStyleSheet("color: #64748b; font-size: 11px;")
        self.combo_checkpoints = QComboBox()
        self.combo_checkpoints.setToolTip("Select from automatically detected trained models under outputs/train/")
        self.populate_detected_checkpoints()
        self.combo_checkpoints.currentIndexChanged.connect(self.on_quick_checkpoint_selected)

        self.btn_reload_model = QPushButton("⚡ Reload Model")
        self.btn_reload_model.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_reload_model.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #f1f5f9;
                font-size: 11px;
                padding: 5px 10px;
                border: 1px solid #475569;
            }
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        self.btn_reload_model.clicked.connect(self.inspect_selected_model)

        row_detect.addWidget(lbl_quick)
        row_detect.addWidget(self.combo_checkpoints, stretch=1)
        row_detect.addWidget(self.btn_reload_model)
        model_layout.addLayout(row_detect)

        # Model Info Chip / Badge
        self.lbl_model_info = QLabel("Model: Inspecting checkpoint...")
        self.lbl_model_info.setStyleSheet("""
            background-color: #0b1120;
            color: #38bdf8;
            border: 1px solid #1e293b;
            border-radius: 4px;
            padding: 4px 8px;
            font-family: monospace;
            font-size: 10px;
            font-weight: bold;
        """)
        model_layout.addWidget(self.lbl_model_info)
        left_column.addWidget(model_box)

        # 2. Evaluation & Autonomous Rollout Controls Card
        eval_box = QGroupBox("AUTONOMOUS POLICY EVALUATION & ROLLOUT CONTROLS")
        eval_layout = QVBoxLayout(eval_box)
        eval_layout.setContentsMargins(10, 10, 10, 10)
        eval_layout.setSpacing(8)

        # Execution Buttons Row (Run & Stop)
        btn_eval_row = QHBoxLayout()
        btn_eval_row.setSpacing(10)

        self.btn_run_test = QPushButton("▶  RUN POLICY EVALUATION (Enter)")
        self.btn_run_test.setMinimumHeight(50)
        self.btn_run_test.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                border: 1px solid #22c55e;
            }
            QPushButton:hover {
                background-color: #15803d;
            }
            QPushButton:disabled {
                background-color: #14532d;
                color: #86efac;
            }
        """)
        self.btn_run_test.clicked.connect(self.start_policy_evaluation)
        btn_eval_row.addWidget(self.btn_run_test, stretch=3)

        self.btn_stop_test = QPushButton("⏹  STOP / ABORT (Esc)")
        self.btn_stop_test.setMinimumHeight(50)
        self.btn_stop_test.setEnabled(False)
        self.btn_stop_test.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
                border-radius: 6px;
                border: 1px solid #ef4444;
            }
            QPushButton:hover {
                background-color: #b91c1c;
            }
            QPushButton:disabled {
                background-color: #291518;
                color: #553034;
                border: 1px solid #3f1d24;
            }
        """)
        self.btn_stop_test.clicked.connect(self.abort_policy_evaluation)
        btn_eval_row.addWidget(self.btn_stop_test, stretch=2)

        eval_layout.addLayout(btn_eval_row)

        # Rollout Settings Row (Duration, Max Jump, Reset Mode)
        opts_row = QHBoxLayout()

        lbl_dur = QLabel("Duration:")
        lbl_dur.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px;")
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(1.0, 60.0)
        self.spin_duration.setValue(10.0)
        self.spin_duration.setSuffix(" s")
        self.spin_duration.setSingleStep(1.0)
        self.spin_duration.setToolTip("Execution duration in seconds (30 Hz closed loop)")

        lbl_jump = QLabel("Max Jump:")
        lbl_jump.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px; margin-left: 6px;")
        self.spin_max_jump = QDoubleSpinBox()
        self.spin_max_jump.setRange(5.0, 90.0)
        self.spin_max_jump.setValue(45.0)
        self.spin_max_jump.setSuffix(" °")
        self.spin_max_jump.setSingleStep(5.0)
        self.spin_max_jump.setToolTip("Maximum allowed delta per joint step before safety warning")

        opts_row.addWidget(lbl_dur)
        opts_row.addWidget(self.spin_duration)
        opts_row.addWidget(lbl_jump)
        opts_row.addWidget(self.spin_max_jump)
        opts_row.addStretch()

        self.chk_auto_freedrive = QCheckBox("Auto-enable Freedrive after test")
        self.chk_auto_freedrive.setChecked(True)
        self.chk_auto_freedrive.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        self.chk_auto_freedrive.setToolTip("Automatically unlock arm in Freedrive once rollout completes")
        opts_row.addWidget(self.chk_auto_freedrive)

        self.chk_return_start = QCheckBox("Return to Start Pose")
        self.chk_return_start.setChecked(False)
        self.chk_return_start.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        self.chk_return_start.setToolTip("Automatically command arm back to active Start Position when test finishes")
        opts_row.addWidget(self.chk_return_start)

        eval_layout.addLayout(opts_row)

        # Temporal Ensembling & Decision Horizon Controls
        ensemble_row = QHBoxLayout()
        self.chk_temporal_ensemble = QCheckBox("Temporal Ensembling (Smooth 30 Hz continuous blend)")
        self.chk_temporal_ensemble.setChecked(True)
        self.chk_temporal_ensemble.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        self.chk_temporal_ensemble.setToolTip("Enable ACT Temporal Ensembling with exponential moving average weights (recommended for smooth continuous control)")
        self.chk_temporal_ensemble.toggled.connect(self.on_temporal_ensemble_toggled)
        ensemble_row.addWidget(self.chk_temporal_ensemble)

        lbl_coeff = QLabel("Coeff (m):")
        lbl_coeff.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px; margin-left: 8px;")
        self.spin_ensemble_coeff = QDoubleSpinBox()
        self.spin_ensemble_coeff.setRange(0.001, 0.500)
        self.spin_ensemble_coeff.setDecimals(3)
        self.spin_ensemble_coeff.setSingleStep(0.005)
        self.spin_ensemble_coeff.setValue(0.010)
        self.spin_ensemble_coeff.setToolTip("Temporal ensembling exponential decay weight coefficient: w_i = exp(-m * i). Default 0.010")
        self.spin_ensemble_coeff.valueChanged.connect(lambda val: self.on_temporal_ensemble_toggled(self.chk_temporal_ensemble.isChecked()))
        ensemble_row.addWidget(lbl_coeff)
        ensemble_row.addWidget(self.spin_ensemble_coeff)
        ensemble_row.addStretch()
        eval_layout.addLayout(ensemble_row)

        # Decision Interval & Action Steps Horizon Row (for Open-Loop Chunking when ensembling disabled)
        self.interval_container = QWidget()
        interval_row = QHBoxLayout(self.interval_container)
        interval_row.setContentsMargins(0, 0, 0, 0)
        self.lbl_steps = QLabel("Re-plan Interval (Open-Loop):")
        self.lbl_steps.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px;")
        interval_row.addWidget(self.lbl_steps)

        self.slider_action_steps = QSlider(Qt.Orientation.Horizontal)
        self.slider_action_steps.setRange(1, 50)
        self.slider_action_steps.setValue(10)
        self.slider_action_steps.setEnabled(False)
        self.slider_action_steps.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #1e293b;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #38bdf8;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0284c7;
                border: 1px solid #38bdf8;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
        """)
        self.slider_action_steps.valueChanged.connect(self.on_action_steps_changed)
        interval_row.addWidget(self.slider_action_steps, stretch=2)

        self.lbl_interval_hud = QLabel("Ensemble Active (m=0.010, 30 Hz)")
        self.lbl_interval_hud.setStyleSheet("color: #38bdf8; font-weight: bold; font-family: monospace; font-size: 11px; min-width: 220px;")
        interval_row.addWidget(self.lbl_interval_hud)

        eval_layout.addWidget(self.interval_container)

        # Rollout Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 300)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setFormat("Ready for Evaluation")
        eval_layout.addWidget(self.progress_bar)

        left_column.addWidget(eval_box)

        # 3. Camera Streams & Device Controls Bar
        cam_ctrl_bar = QHBoxLayout()
        cam_ctrl_bar.setSpacing(6)

        available_cameras = detect_available_cameras()

        lbl_cam1 = QLabel("Cam 1:")
        lbl_cam1.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px;")
        cam_ctrl_bar.addWidget(lbl_cam1)

        self.combo_cam1 = QComboBox()
        for idx, label in available_cameras:
            self.combo_cam1.addItem(label, idx)
        idx_match = self.combo_cam1.findData(self.camera_1_index)
        if idx_match >= 0:
            self.combo_cam1.setCurrentIndex(idx_match)
        self.combo_cam1.currentIndexChanged.connect(self.on_cam1_device_changed)
        cam_ctrl_bar.addWidget(self.combo_cam1)

        self.btn_toggle_cam2 = QPushButton("+ 2nd Camera")
        self.btn_toggle_cam2.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_toggle_cam2.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #94a3b8;
                font-size: 10px;
                font-weight: bold;
                padding: 4px 8px;
                border: 1px solid #334155;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #ffffff;
            }
        """)
        self.btn_toggle_cam2.clicked.connect(self.toggle_dual_cameras)
        cam_ctrl_bar.addWidget(self.btn_toggle_cam2)

        self.lbl_cam2 = QLabel("Cam 2:")
        self.lbl_cam2.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px; margin-left: 4px;")
        self.lbl_cam2.setVisible(False)
        cam_ctrl_bar.addWidget(self.lbl_cam2)

        self.combo_cam2 = QComboBox()
        for idx, label in available_cameras:
            self.combo_cam2.addItem(label, idx)
        idx2_match = self.combo_cam2.findData(self.camera_2_index)
        if idx2_match >= 0:
            self.combo_cam2.setCurrentIndex(idx2_match)
        self.combo_cam2.setVisible(False)
        self.combo_cam2.currentIndexChanged.connect(self.on_cam2_device_changed)
        cam_ctrl_bar.addWidget(self.combo_cam2)

        self.chk_grayscale = QCheckBox("Grayscale (1-Ch)")
        self.chk_grayscale.setStyleSheet("""
            QCheckBox {
                color: #cbd5e1;
                font-size: 11px;
                font-weight: bold;
                margin-left: 6px;
            }
        """)
        self.chk_grayscale.setToolTip("Toggle Grayscale camera preview and tensor processing (auto-synced from model metadata)")
        self.chk_grayscale.toggled.connect(self.on_grayscale_toggled)
        cam_ctrl_bar.addWidget(self.chk_grayscale)

        cam_ctrl_bar.addStretch()
        left_column.addLayout(cam_ctrl_bar)

        # Video Displays Container
        self.video_container = QWidget()
        self.video_layout = QHBoxLayout(self.video_container)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_layout.setSpacing(6)

        self.cam_widget_1 = VideoFeedWidget("Workspace Camera (Cam 1)")
        self.video_layout.addWidget(self.cam_widget_1, stretch=1)

        self.cam_widget_2 = VideoFeedWidget("Wrist Camera (Cam 2)")
        self.cam_widget_2.setVisible(False)
        self.video_layout.addWidget(self.cam_widget_2, stretch=1)

        left_column.addWidget(self.video_container, stretch=3)
        splitter.addWidget(left_widget)

        # ---------------------------------------------------------------------
        # RIGHT COLUMN: Freedrive Status, Joint Telemetry, Jogging, Live Log
        # ---------------------------------------------------------------------
        right_widget = QWidget()
        right_column = QVBoxLayout(right_widget)
        right_column.setContentsMargins(4, 4, 4, 4)
        right_column.setSpacing(8)

        # 4. Freedrive Mode Card
        fd_box = QGroupBox("FREEDRIVE & ARM COMPLIANCE CONTROL")
        fd_layout = QVBoxLayout(fd_box)
        fd_layout.setContentsMargins(10, 8, 10, 8)
        fd_layout.setSpacing(6)

        fd_row = QHBoxLayout()
        self.lbl_fd_status = QLabel("ROBOT STATE: LOCKED (Position Hold)")
        self.lbl_fd_status.setStyleSheet("""
            color: #94a3b8;
            font-weight: bold;
            font-size: 11px;
            padding: 4px 8px;
            background-color: #0b1120;
            border-radius: 4px;
            border: 1px solid #1e293b;
        """)
        fd_row.addWidget(self.lbl_fd_status, stretch=1)

        self.btn_toggle_freedrive = QPushButton("⚡ TOGGLE FREEDRIVE MODE")
        self.btn_toggle_freedrive.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_toggle_freedrive.setMinimumHeight(34)
        self.btn_toggle_freedrive.setStyleSheet("""
            QPushButton {
                background-color: #475569;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
                border: 1px solid #64748b;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #64748b;
            }
        """)
        self.btn_toggle_freedrive.clicked.connect(self.on_freedrive_button_clicked)
        fd_row.addWidget(self.btn_toggle_freedrive)

        fd_layout.addLayout(fd_row)
        right_column.addWidget(fd_box)

        # Gamepad Teleoperation & Manual Positioning Card
        gamepad_box = QGroupBox("GAMEPAD TELEOPERATION & MANUAL POSITIONING")
        gamepad_layout = QVBoxLayout(gamepad_box)
        gamepad_layout.setContentsMargins(10, 8, 10, 8)
        gamepad_layout.setSpacing(6)

        gamepad_row1 = QHBoxLayout()
        self.btn_toggle_gamepad = QPushButton("🎮 ENABLE GAMEPAD TELEOP")
        self.btn_toggle_gamepad.setCheckable(True)
        self.btn_toggle_gamepad.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_toggle_gamepad.setMinimumHeight(32)
        self.btn_toggle_gamepad.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #94a3b8;
                font-size: 11px;
                font-weight: bold;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #f1f5f9;
            }
            QPushButton:checked {
                background-color: #0284c7;
                color: #ffffff;
                border: 1px solid #38bdf8;
            }
        """)
        self.btn_toggle_gamepad.setToolTip(
            "Toggle real-time Cartesian teleoperation & direct joint control via gamepad.\n"
            "• Left Stick (XY): TCP Forward/Backward (+X/-X) and Left/Right (+Y/-Y)\n"
            "• Right Stick (Y): TCP Vertical Up/Down (+Z/-Z)\n"
            "• Right Stick (X): Direct Base Motor Rotation (shoulder_pan)\n"
            "• Shoulders (LB/RB / L1/R1): Direct Wrist 2 Rotation\n"
            "• Triggers (LT/RT): Direct Wrist 3 Rotation\n"
            "• Neutral sticks/buttons: Immediate braking (0 drift)"
        )
        self.btn_toggle_gamepad.toggled.connect(self.on_gamepad_toggle_clicked)
        gamepad_row1.addWidget(self.btn_toggle_gamepad)

        self.lbl_gamepad_badge = QLabel("🎮 No Gamepad Detected")
        self.lbl_gamepad_badge.setStyleSheet("""
            color: #64748b;
            font-size: 11px;
            font-weight: bold;
            padding: 4px 8px;
            background-color: #0b1120;
            border: 1px solid #1e293b;
            border-radius: 4px;
        """)
        gamepad_row1.addWidget(self.lbl_gamepad_badge)
        gamepad_layout.addLayout(gamepad_row1)

        gamepad_row2 = QHBoxLayout()
        self.lbl_gamepad_vel = QLabel("TCP: 0.00 m/s")
        self.lbl_gamepad_vel.setStyleSheet("""
            color: #38bdf8;
            font-family: monospace;
            font-size: 11px;
            font-weight: bold;
            padding: 4px 8px;
            background-color: #0b1120;
            border: 1px solid #1e293b;
            border-radius: 4px;
        """)
        gamepad_row2.addWidget(self.lbl_gamepad_vel)

        # Gripper Gamepad Shortcut Badge
        self.lbl_gamepad_gripper_hint = QLabel("🤏 Gripper: [A] Close | [B] Open")
        self.lbl_gamepad_gripper_hint.setStyleSheet("""
            color: #10b981;
            font-family: monospace;
            font-size: 11px;
            font-weight: bold;
            padding: 4px 8px;
            background-color: #064e3b;
            border: 1px solid #059669;
            border-radius: 4px;
        """)
        self.lbl_gamepad_gripper_hint.setToolTip("Xbox Controller: Hold [A] to slowly CLOSE gripper | Hold [B] to slowly OPEN gripper")
        gamepad_row2.addWidget(self.lbl_gamepad_gripper_hint)
        gamepad_layout.addLayout(gamepad_row2)

        # Inversion Checkboxes Row
        gamepad_row3 = QHBoxLayout()
        self.chk_invert_left_lr = QCheckBox("⇄ Invert Left (L/R)")
        self.chk_invert_left_lr.setChecked(False)
        self.chk_invert_left_lr.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_invert_left_lr.setStyleSheet("""
            QCheckBox {
                color: #cbd5e1;
                font-size: 11px;
                font-weight: bold;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #475569;
                border-radius: 3px;
                background-color: #0f172a;
            }
            QCheckBox::indicator:checked {
                background-color: #0284c7;
                border: 1px solid #38bdf8;
            }
            QCheckBox:hover {
                color: #38bdf8;
            }
        """)
        self.chk_invert_left_lr.setToolTip(
            "Invert Left Stick lateral translation direction (Left/Right):\n"
            "• Normal: Left Stick Right -> +Lateral (Right relative to arm heading)\n"
            "• Inverted: Left Stick Right -> -Lateral (Left relative to arm heading)"
        )
        self.chk_invert_left_lr.toggled.connect(self.on_invert_left_lr_toggled)
        gamepad_row3.addWidget(self.chk_invert_left_lr)

        self.chk_invert_right_lr = QCheckBox("⇄ Invert Right (Base)")
        self.chk_invert_right_lr.setChecked(False)
        self.chk_invert_right_lr.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_invert_right_lr.setStyleSheet("""
            QCheckBox {
                color: #cbd5e1;
                font-size: 11px;
                font-weight: bold;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #475569;
                border-radius: 3px;
                background-color: #0f172a;
            }
            QCheckBox::indicator:checked {
                background-color: #0284c7;
                border: 1px solid #38bdf8;
            }
            QCheckBox:hover {
                color: #38bdf8;
            }
        """)
        self.chk_invert_right_lr.setToolTip(
            "Invert Right Stick horizontal base rotation direction (shoulder_pan):\n"
            "• Normal: Right Stick Right -> +Base rotation\n"
            "• Inverted: Right Stick Right -> -Base rotation"
        )
        self.chk_invert_right_lr.toggled.connect(self.on_invert_right_lr_toggled)
        gamepad_row3.addWidget(self.chk_invert_right_lr)

        # Backwards compatibility alias
        self.chk_invert_lr = self.chk_invert_left_lr
        gamepad_layout.addLayout(gamepad_row3)

        # Row 4: Speed and Acceleration Sliders with Live Readouts
        gamepad_row4 = QHBoxLayout()
        gamepad_row4.setSpacing(8)

        # Master Speed Slider
        self.lbl_gamepad_speed = QLabel("⚡ Speed: 50%")
        self.lbl_gamepad_speed.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold;")
        self.slider_gamepad_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_gamepad_speed.setRange(10, 100)
        self.slider_gamepad_speed.setValue(50)
        self.slider_gamepad_speed.setFixedWidth(110)
        self.slider_gamepad_speed.setFixedHeight(18)
        self.slider_gamepad_speed.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #334155;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #38bdf8;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #f1f5f9;
                width: 12px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 6px;
            }
        """)
        self.slider_gamepad_speed.setToolTip("Scale maximum gamepad teleoperation speed (10% to 100%).")
        self.slider_gamepad_speed.valueChanged.connect(self.on_gamepad_speed_changed)
        gamepad_row4.addWidget(self.lbl_gamepad_speed)
        gamepad_row4.addWidget(self.slider_gamepad_speed)

        gamepad_row4.addSpacing(8)

        # Master Acceleration Slider
        self.lbl_gamepad_accel = QLabel("🚀 Accel: 50%")
        self.lbl_gamepad_accel.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold;")
        self.slider_gamepad_accel = QSlider(Qt.Orientation.Horizontal)
        self.slider_gamepad_accel.setRange(10, 100)
        self.slider_gamepad_accel.setValue(50)
        self.slider_gamepad_accel.setFixedWidth(110)
        self.slider_gamepad_accel.setFixedHeight(18)
        self.slider_gamepad_accel.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #334155;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #4ade80;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #f1f5f9;
                width: 12px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 6px;
            }
        """)
        self.slider_gamepad_accel.setToolTip("Scale acceleration ramp rate (10% to 100%). Lower values prevent Wrist 1 protective stops.")
        self.slider_gamepad_accel.valueChanged.connect(self.on_gamepad_accel_changed)
        gamepad_row4.addWidget(self.lbl_gamepad_accel)
        gamepad_row4.addWidget(self.slider_gamepad_accel)

        gamepad_row4.addStretch()
        gamepad_layout.addLayout(gamepad_row4)

        right_column.addWidget(gamepad_box)

        # 5. Real-Time 6-DoF Joint Telemetry HUD
        telem_box = QGroupBox("6-DOF JOINT TELEMETRY & ENCODER MONITOR")
        telem_layout = QVBoxLayout(telem_box)
        telem_layout.setContentsMargins(10, 8, 10, 8)
        telem_layout.setSpacing(4)

        telem_header = QHBoxLayout()
        self.lbl_rtde_rate = QLabel("RTDE: 125.0 Hz")
        self.lbl_rtde_rate.setStyleSheet("color: #38bdf8; font-family: monospace; font-size: 10px; font-weight: bold;")
        self.lbl_rtde_latency = QLabel("Latency: 1.2 ms")
        self.lbl_rtde_latency.setStyleSheet("color: #94a3b8; font-family: monospace; font-size: 10px;")
        telem_header.addWidget(self.lbl_rtde_rate)
        telem_header.addSpacing(12)
        telem_header.addWidget(self.lbl_rtde_latency)
        telem_header.addStretch()
        telem_layout.addLayout(telem_header)

        grid_joints = QGridLayout()
        grid_joints.setSpacing(3)

        for idx, j_name in enumerate(UR5E_JOINTS):
            name_lbl = QLabel(j_name.replace("_", " ").title())
            name_lbl.setStyleSheet("color: #cbd5e1; font-size: 10px; font-weight: 500;")
            name_lbl.setMinimumWidth(80)

            val_lbl = QLabel("+0.0°")
            val_lbl.setFixedWidth(52)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setStyleSheet("color: #38bdf8; font-family: monospace; font-size: 10px; font-weight: bold;")
            self.joint_labels[j_name] = val_lbl

            pbar = QProgressBar()
            pbar.setRange(-360, 360)
            pbar.setValue(0)
            pbar.setTextVisible(False)
            pbar.setFixedHeight(6)
            self.joint_bars[j_name] = pbar

            grid_joints.addWidget(name_lbl, idx, 0)
            grid_joints.addWidget(val_lbl, idx, 1)
            grid_joints.addWidget(pbar, idx, 2)

        telem_layout.addLayout(grid_joints)
        right_column.addWidget(telem_box)

        # 6. Manual Axis Control & Jogging Panel (With Staged Targets & Navigation)
        jog_box = QGroupBox("MANUAL AXIS CONTROL & JOGGING")
        self.jog_group = jog_box
        jog_layout = QVBoxLayout(jog_box)
        jog_layout.setContentsMargins(10, 8, 10, 8)
        jog_layout.setSpacing(5)

        jog_header = QHBoxLayout()
        self.lbl_jog_notice = QLabel("Freedrive LOCKED — Manual jogging active")
        self.lbl_jog_notice.setStyleSheet("color: #4ade80; font-size: 10px; font-weight: bold;")
        jog_header.addWidget(self.lbl_jog_notice, stretch=1)

        # Home Pose Button
        self.btn_move_home = QPushButton("⌂ Home")
        self.btn_move_home.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_move_home.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                font-size: 10px;
                font-weight: bold;
                padding: 3px 7px;
                border-radius: 4px;
                border: 1px solid #38bdf8;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
        """)
        self.btn_move_home.setToolTip("Command arm to move to default Home Pose [0°, -90°, 90°, -90°, -90°, 0°]")
        self.btn_move_home.clicked.connect(self.move_to_home_pose)
        jog_header.addWidget(self.btn_move_home)

        # Go to Start Position Button
        self.btn_go_start = QPushButton("▶ Go to Start")
        self.btn_go_start.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_go_start.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: #ffffff;
                font-size: 10px;
                font-weight: bold;
                padding: 3px 7px;
                border-radius: 4px;
                border: 1px solid #22c55e;
            }
            QPushButton:hover {
                background-color: #15803d;
            }
        """)
        self.btn_go_start.setToolTip("Command arm to move to stored Custom Start Position")
        self.btn_go_start.clicked.connect(self.move_to_start_pose)
        jog_header.addWidget(self.btn_go_start)

        # Set Start Position Button
        self.btn_set_start = QPushButton("💾 Set Start")
        self.btn_set_start.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_set_start.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #f1f5f9;
                font-size: 10px;
                font-weight: bold;
                padding: 3px 7px;
                border-radius: 4px;
                border: 1px solid #64748b;
            }
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        self.btn_set_start.setToolTip("Save robot's current 6 joint coordinates as the active Start Position")
        self.btn_set_start.clicked.connect(self.set_current_as_start_position)
        jog_header.addWidget(self.btn_set_start)

        # Move to Target Button
        self.btn_move_target = QPushButton("🎯 Move to Target")
        self.btn_move_target.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_move_target.setStyleSheet("""
            QPushButton {
                background-color: #4f46e5;
                color: #ffffff;
                font-size: 10px;
                font-weight: bold;
                padding: 3px 7px;
                border-radius: 4px;
                border: 1px solid #818cf8;
            }
            QPushButton:hover {
                background-color: #4338ca;
            }
        """)
        self.btn_move_target.setToolTip("Command arm to move all joints to the currently specified numeric target values")
        self.btn_move_target.clicked.connect(self.move_to_all_targets)
        jog_header.addWidget(self.btn_move_target)

        # Sync Button
        self.btn_sync_targets = QPushButton("⟳ Sync")
        self.btn_sync_targets.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_sync_targets.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #94a3b8;
                font-size: 10px;
                font-weight: bold;
                padding: 3px 7px;
                border-radius: 4px;
                border: 1px solid #475569;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #f1f5f9;
            }
        """)
        self.btn_sync_targets.setToolTip("Reset all jog input fields to match current robot joint telemetry")
        self.btn_sync_targets.clicked.connect(self.sync_jog_to_robot)
        jog_header.addWidget(self.btn_sync_targets)

        jog_layout.addLayout(jog_header)

        # Start Position Display Banner
        self.lbl_start_pose_info = QLabel()
        self.lbl_start_pose_info.setStyleSheet("""
            background-color: #0b1120;
            color: #38bdf8;
            border: 1px solid #1e293b;
            border-radius: 4px;
            padding: 2px 6px;
            font-family: monospace;
            font-size: 10px;
            font-weight: bold;
        """)
        jog_layout.addWidget(self.lbl_start_pose_info)
        self.update_start_pose_display()

        grid_jog = QGridLayout()
        grid_jog.setSpacing(4)

        for idx, j_name in enumerate(UR5E_JOINTS):
            name_lbl = QLabel(j_name.replace("_", " ").title())
            name_lbl.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 500;")
            name_lbl.setMinimumWidth(80)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-360, 360)
            slider.setValue(int(round(self.start_position.get(j_name, 0.0))))
            slider.setFixedHeight(18)
            slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 4px;
                    background: #334155;
                    border-radius: 2px;
                }
                QSlider::sub-page:horizontal {
                    background: #38bdf8;
                    border-radius: 2px;
                }
                QSlider::handle:horizontal {
                    background: #f1f5f9;
                    width: 12px;
                    margin-top: -4px;
                    margin-bottom: -4px;
                    border-radius: 6px;
                }
            """)

            input_box = QLineEdit(f"{self.start_position.get(j_name, 0.0):+.1f}")
            input_box.setFixedWidth(56)
            input_box.setFixedHeight(22)
            input_box.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            input_box.setStyleSheet("""
                QLineEdit {
                    background-color: #0b1120;
                    color: #f1f5f9;
                    font-family: monospace;
                    font-size: 10px;
                    font-weight: bold;
                    border: 1px solid #334155;
                    border-radius: 3px;
                    padding: 1px 3px;
                }
                QLineEdit:focus {
                    border: 1px solid #38bdf8;
                    background-color: #0f172a;
                }
            """)
            input_box.setToolTip(f"Type target angle in degrees for {j_name} and press Enter (or click Move)")

            btn_move_joint = QPushButton("Move")
            btn_move_joint.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn_move_joint.setFixedWidth(46)
            btn_move_joint.setFixedHeight(22)
            btn_move_joint.setStyleSheet("""
                QPushButton {
                    background-color: #0284c7;
                    color: #ffffff;
                    font-size: 10px;
                    font-weight: bold;
                    border-radius: 3px;
                    border: 1px solid #38bdf8;
                    padding: 1px 4px;
                }
                QPushButton:hover {
                    background-color: #0369a1;
                }
            """)
            btn_move_joint.setToolTip(f"Move {j_name} to the entered target angle")

            self.jog_sliders[j_name] = slider
            self.jog_inputs[j_name] = input_box
            self.jog_move_buttons[j_name] = btn_move_joint

            slider.sliderPressed.connect(lambda j=j_name: self.on_jog_slider_pressed(j))
            slider.valueChanged.connect(lambda val, j=j_name: self.on_jog_slider_value_changed(j, val))
            slider.sliderReleased.connect(lambda j=j_name: self.on_jog_slider_released(j))

            input_box.textEdited.connect(lambda text, j=j_name: self.on_jog_input_edited(j, text))
            input_box.returnPressed.connect(lambda j=j_name: self.on_jog_input_committed(j))
            btn_move_joint.clicked.connect(lambda _, j=j_name: self.on_jog_input_committed(j))

            grid_jog.addWidget(name_lbl, idx, 0)
            grid_jog.addWidget(slider, idx, 1)
            grid_jog.addWidget(input_box, idx, 2)
            grid_jog.addWidget(btn_move_joint, idx, 3)

        # Gripper Control Row (OnRobot 2FG14)
        sep_line = QFrame()
        sep_line.setFrameShape(QFrame.Shape.HLine)
        sep_line.setFrameShadow(QFrame.Shadow.Sunken)
        sep_line.setStyleSheet("background-color: #334155; margin-top: 4px; margin-bottom: 4px;")
        grid_jog.addWidget(sep_line, 6, 0, 1, 4)

        gripper_lbl = QLabel("🤏 Gripper")
        gripper_lbl.setStyleSheet("color: #10b981; font-weight: bold; font-size: 11px;")
        gripper_lbl.setToolTip("OnRobot 2FG14 Gripper Opening Width (55mm to 105mm)")

        self.gripper_slider = QSlider(Qt.Orientation.Horizontal)
        self.gripper_slider.setMinimum(55)
        self.gripper_slider.setMaximum(105)
        self.gripper_slider.setValue(72)
        self.gripper_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #1e293b;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #10b981;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #f1f5f9;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
                border: 1px solid #10b981;
            }
        """)
        self.gripper_slider.setToolTip("Drag slider to open / close OnRobot 2FG gripper (55 - 105 mm)")

        self.gripper_input = QLineEdit("72.0")
        self.gripper_input.setFixedWidth(56)
        self.gripper_input.setFixedHeight(22)
        self.gripper_input.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.gripper_input.setStyleSheet("""
            QLineEdit {
                background-color: #0b1120;
                color: #10b981;
                font-family: monospace;
                font-size: 10px;
                font-weight: bold;
                border: 1px solid #334155;
                border-radius: 3px;
                padding: 1px 3px;
            }
            QLineEdit:focus {
                border: 1px solid #10b981;
                background-color: #0f172a;
            }
        """)
        self.gripper_input.setToolTip("Type target width in mm (55 - 105) and press Enter")

        self.btn_move_gripper = QPushButton("Move")
        self.btn_move_gripper.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_move_gripper.setFixedWidth(46)
        self.btn_move_gripper.setFixedHeight(22)
        self.btn_move_gripper.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: #ffffff;
                font-size: 10px;
                font-weight: bold;
                border-radius: 3px;
                border: 1px solid #10b981;
                padding: 1px 4px;
            }
            QPushButton:hover {
                background-color: #047857;
            }
        """)
        self.btn_move_gripper.setToolTip("Command gripper to the entered target width")

        self.gripper_slider.sliderPressed.connect(self.on_gripper_slider_pressed)
        self.gripper_slider.valueChanged.connect(self.on_gripper_slider_value_changed)
        self.gripper_slider.sliderReleased.connect(self.on_gripper_slider_released)
        self.gripper_input.returnPressed.connect(self.on_gripper_input_committed)
        self.btn_move_gripper.clicked.connect(self.on_gripper_input_committed)

        grid_jog.addWidget(gripper_lbl, 7, 0)
        grid_jog.addWidget(self.gripper_slider, 7, 1)
        grid_jog.addWidget(self.gripper_input, 7, 2)
        grid_jog.addWidget(self.btn_move_gripper, 7, 3)

        # Quick Gripper Action Buttons Row (Close 55mm / Half 80mm / Open 105mm)
        row_quick_grip = QHBoxLayout()
        row_quick_grip.setSpacing(6)
        self.btn_grip_close = QPushButton("🤏 Close (55mm)")
        self.btn_grip_close.setFixedHeight(22)
        self.btn_grip_close.setStyleSheet("background-color: #1e293b; color: #f87171; font-size: 10px; font-weight: bold; border: 1px solid #334155; border-radius: 3px;")
        self.btn_grip_close.setToolTip("Quick Close gripper to minimum width (55mm) - Xbox Button [A]")
        self.btn_grip_close.clicked.connect(lambda: self.command_gripper_width(55.0))

        self.btn_grip_half = QPushButton("↔ 80mm")
        self.btn_grip_half.setFixedHeight(22)
        self.btn_grip_half.setStyleSheet("background-color: #1e293b; color: #38bdf8; font-size: 10px; font-weight: bold; border: 1px solid #334155; border-radius: 3px;")
        self.btn_grip_half.setToolTip("Move gripper to mid-aperture (80mm)")
        self.btn_grip_half.clicked.connect(lambda: self.command_gripper_width(80.0))

        self.btn_grip_open = QPushButton("👐 Open (105mm)")
        self.btn_grip_open.setFixedHeight(22)
        self.btn_grip_open.setStyleSheet("background-color: #1e293b; color: #34d399; font-size: 10px; font-weight: bold; border: 1px solid #334155; border-radius: 3px;")
        self.btn_grip_open.setToolTip("Quick Open gripper to maximum width (105mm) - Xbox Button [B]")
        self.btn_grip_open.clicked.connect(lambda: self.command_gripper_width(105.0))

        row_quick_grip.addWidget(self.btn_grip_close)
        row_quick_grip.addWidget(self.btn_grip_half)
        row_quick_grip.addWidget(self.btn_grip_open)
        grid_jog.addLayout(row_quick_grip, 8, 0, 1, 4)

        jog_layout.addLayout(grid_jog)
        right_column.addWidget(jog_box)

        # 7. Live Console Log Terminal
        log_box = QGroupBox("LIVE EVALUATION TERMINAL CONSOLE")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(10, 8, 10, 8)
        log_layout.setSpacing(6)

        log_ctrl = QHBoxLayout()
        lbl_term = QLabel("Console Output:")
        lbl_term.setStyleSheet("color: #64748b; font-size: 11px;")
        log_ctrl.addWidget(lbl_term)
        log_ctrl.addStretch()

        btn_copy = QPushButton("📋 Copy Log")
        btn_copy.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_copy.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        btn_copy.clicked.connect(self.copy_log_to_clipboard)
        log_ctrl.addWidget(btn_copy)

        btn_clear = QPushButton("🗑 Clear")
        btn_clear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_clear.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        btn_clear.clicked.connect(self.clear_log)
        log_ctrl.addWidget(btn_clear)
        log_layout.addLayout(log_ctrl)

        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(2000)
        self.txt_log.setStyleSheet("""
            QPlainTextEdit {
                background-color: #030712;
                color: #e2e8f0;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #1e293b;
                border-radius: 4px;
            }
        """)
        log_layout.addWidget(self.txt_log, stretch=1)
        right_column.addWidget(log_box, stretch=2)

        splitter.addWidget(right_widget)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready for Policy Evaluation.", 4000)

        # Equal initial split
        splitter.setSizes([750, 650])

    def setup_shortcuts(self):
        shortcut_enter = QShortcut(QKeySequence(Qt.Key.Key_Return), self)
        shortcut_enter.activated.connect(self.on_enter_hotkey)

        shortcut_esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        shortcut_esc.activated.connect(self.abort_policy_evaluation)

    def on_enter_hotkey(self):
        # Trigger rollout evaluation if not already running and no input box focused
        for inp in self.jog_inputs.values():
            if inp.hasFocus():
                return
        if self.rollout_worker is None or not self.rollout_worker.isRunning():
            self.start_policy_evaluation()

    # -------------------------------------------------------------------------
    # Model Checkpoint Management Slots
    # -------------------------------------------------------------------------
    def populate_detected_checkpoints(self):
        self.combo_checkpoints.blockSignals(True)
        self.combo_checkpoints.clear()
        detected = scan_available_checkpoints()
        for name, path in detected:
            self.combo_checkpoints.addItem(f"{name} ({path.parent.name})", str(path))
        self.combo_checkpoints.blockSignals(False)

    def on_browse_model_folder(self):
        start_dir = str(DEFAULT_TRAIN_DIR) if DEFAULT_TRAIN_DIR.exists() else str(Path.cwd())
        chosen_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Trained Policy Checkpoint Directory",
            start_dir,
            QFileDialog.Option.ShowDirsOnly,
        )
        if chosen_dir:
            self.active_model_path = Path(chosen_dir).resolve()
            self.txt_model_path.setText(str(self.active_model_path))
            self.inspect_selected_model()

    def on_model_path_committed(self):
        text = self.txt_model_path.text().strip()
        if text:
            self.active_model_path = Path(text).resolve()
            self.inspect_selected_model()

    def on_quick_checkpoint_selected(self, index: int):
        path_str = self.combo_checkpoints.itemData(index)
        if path_str:
            self.active_model_path = Path(path_str).resolve()
            self.txt_model_path.setText(str(self.active_model_path))
            self.inspect_selected_model()

    def inspect_selected_model(self):
        """Reads config.json from checkpoint and displays model properties."""
        p = self.active_model_path
        if not p.exists():
            self.lbl_model_info.setText(f"⚠ Directory not found: {p.name}")
            self.lbl_model_info.setStyleSheet("background-color: #291518; color: #f87171; border: 1px solid #7f1d1d;")
            return

        config_json = p / "config.json"
        if not config_json.exists():
            self.lbl_model_info.setText(f"⚠ Missing config.json in: {p.name}")
            self.lbl_model_info.setStyleSheet("background-color: #291518; color: #f87171; border: 1px solid #7f1d1d;")
            return

        try:
            import json
            data = json.loads(config_json.read_text())
            policy_type = data.get("type", "ACTPolicy")

            # Detect visual input features from both input_features and image_features
            features = {}
            if "input_features" in data and isinstance(data["input_features"], dict):
                features.update({k: v for k, v in data["input_features"].items() if "observation.images" in k})
            if "image_features" in data and isinstance(data["image_features"], dict):
                features.update({k: v for k, v in data["image_features"].items() if "observation.images" in k})

            cams = []
            if "observation.images.workspace" in features:
                cams.append("Workspace")
            if "observation.images.wrist" in features:
                cams.append("Wrist")

            # Fallback to training_info.json or info.json in parent folders if present
            if not cams:
                for info_p in [p / "training_info.json", p.parent / "training_info.json"]:
                    if info_p.exists():
                        try:
                            with open(info_p, "r") as f_info:
                                t_info = json.load(f_info)
                                if t_info.get("dual_cam") is True or t_info.get("num_cameras") == 2:
                                    cams = ["Workspace", "Wrist"]
                                    break
                                elif t_info.get("dual_cam") is False or t_info.get("num_cameras") == 1:
                                    cams = ["Workspace"]
                                    break
                        except Exception:
                            pass

            if not cams:
                cams = ["Workspace"]

            self.model_required_cams = len(cams)
            self.model_camera_names = list(cams)
            cam_str = f"{len(cams)}-Cam ({' + '.join(cams)})"

            device_str = "CUDA (RTX)" if (torch and torch.cuda.is_available()) else "CPU"

            # Check if checkpoint was trained in grayscale mode
            is_model_grayscale = False
            for info_p in [p / "training_info.json", p.parent / "training_info.json"]:
                if info_p.exists():
                    try:
                        with open(info_p, "r") as f_info:
                            t_info = json.load(f_info)
                            if t_info.get("grayscale") is True or t_info.get("image_channels", 3) == 1:
                                is_model_grayscale = True
                                break
                    except Exception:
                        pass

            self.set_grayscale_mode(is_model_grayscale)
            mode_badge = "Grayscale (1-Ch)" if is_model_grayscale else "RGB (3-Ch)"

            # Check for trained FPS, decision horizon, and temporal ensemble settings
            model_fps = 30
            trained_n_steps = 10
            trained_ensemble_coeff = None
            for info_p in [p / "training_info.json", p.parent / "training_info.json"]:
                if info_p.exists():
                    try:
                        with open(info_p, "r") as f_info:
                            t_info = json.load(f_info)
                            model_fps = int(t_info.get("fps", 30))
                            trained_n_steps = int(t_info.get("n_action_steps", 10))
                            if "temporal_ensemble_coeff" in t_info and t_info["temporal_ensemble_coeff"] is not None:
                                trained_ensemble_coeff = float(t_info["temporal_ensemble_coeff"])
                            break
                    except Exception:
                        pass
            if trained_ensemble_coeff is None:
                try:
                    with open(p / "config.json", "r") as f_cfg:
                        cfg_data = json.load(f_cfg)
                        if "temporal_ensemble_coeff" in cfg_data and cfg_data["temporal_ensemble_coeff"] is not None:
                            trained_ensemble_coeff = float(cfg_data["temporal_ensemble_coeff"])
                except Exception:
                    pass

            self.model_fps = model_fps
            if trained_ensemble_coeff is not None and hasattr(self, "spin_ensemble_coeff"):
                self.spin_ensemble_coeff.blockSignals(True)
                self.spin_ensemble_coeff.setValue(trained_ensemble_coeff)
                self.spin_ensemble_coeff.blockSignals(False)

            if hasattr(self, "slider_action_steps"):
                self.slider_action_steps.blockSignals(True)
                self.slider_action_steps.setValue(trained_n_steps)
                self.slider_action_steps.blockSignals(False)

            if hasattr(self, "chk_temporal_ensemble"):
                self.on_temporal_ensemble_toggled(self.chk_temporal_ensemble.isChecked())
            else:
                self.on_action_steps_changed(trained_n_steps)

            info_text = f"✔ {policy_type.upper()} | Cams: {cam_str} | Mode: {mode_badge} | Device: {device_str} | Checkpoint: {p.parent.name}"
            self.lbl_model_info.setText(info_text)
            self.lbl_model_info.setStyleSheet("background-color: #0b1120; color: #4ade80; border: 1px solid #166534; font-weight: bold;")
            print(f"[MODEL] Inspected checkpoint: {info_text}")

            # Auto-align secondary camera with policy requirement upon loading
            if "Wrist" in cams and not self.enable_dual_cameras:
                print(f"[MODEL] Model requires dual cameras ({cam_str}). Auto-enabling 2-Camera Mode.")
                self.set_dual_cameras_enabled(True)
            elif "Wrist" not in cams and self.enable_dual_cameras:
                print(f"[MODEL] Model requires single camera ({cam_str}). Auto-setting 1-Camera Mode.")
                self.set_dual_cameras_enabled(False)
        except Exception as e:
            self.lbl_model_info.setText(f"Error parsing model config: {e}")

    # -------------------------------------------------------------------------
    # Camera Controls & Devices
    # -------------------------------------------------------------------------
    def on_temporal_ensemble_toggled(self, checked: bool):
        if hasattr(self, "spin_ensemble_coeff"):
            self.spin_ensemble_coeff.setEnabled(checked)
        if hasattr(self, "slider_action_steps"):
            self.slider_action_steps.setEnabled(not checked)
        if hasattr(self, "lbl_interval_hud"):
            fps = getattr(self, "model_fps", 30)
            if checked:
                coeff = self.spin_ensemble_coeff.value() if hasattr(self, "spin_ensemble_coeff") else 0.010
                self.lbl_interval_hud.setText(f"Ensemble Active (m={coeff:.3f}, {fps} Hz)")
                self.lbl_interval_hud.setStyleSheet("color: #38bdf8; font-weight: bold; font-family: monospace; font-size: 11px; min-width: 220px;")
            else:
                steps = self.slider_action_steps.value() if hasattr(self, "slider_action_steps") else 10
                dt = steps / max(1, fps)
                self.lbl_interval_hud.setText(f"{steps} steps ({dt:.2f}s / {int(dt*1000)}ms @ {fps} FPS)")
                self.lbl_interval_hud.setStyleSheet("color: #f59e0b; font-weight: bold; font-family: monospace; font-size: 11px; min-width: 220px;")

    def on_grayscale_toggled(self, checked: bool):
        self.set_grayscale_mode(checked)

    def set_grayscale_mode(self, enabled: bool):
        self.is_grayscale = enabled
        if hasattr(self, "chk_grayscale"):
            self.chk_grayscale.blockSignals(True)
            self.chk_grayscale.setChecked(enabled)
            self.chk_grayscale.blockSignals(False)
        if self.camera_worker_1 is not None:
            self.camera_worker_1.set_grayscale(enabled)
        if self.camera_worker_2 is not None:
            self.camera_worker_2.set_grayscale(enabled)
        if hasattr(self, "cam_widget_1"):
            self.cam_widget_1.set_grayscale(enabled)
        if hasattr(self, "cam_widget_2"):
            self.cam_widget_2.set_grayscale(enabled)

    def on_action_steps_changed(self, steps: int):
        fps = getattr(self, "model_fps", 30)
        dt = steps / max(1, fps)
        if hasattr(self, "lbl_interval_hud"):
            if hasattr(self, "chk_temporal_ensemble") and self.chk_temporal_ensemble.isChecked():
                coeff = self.spin_ensemble_coeff.value() if hasattr(self, "spin_ensemble_coeff") else 0.010
                self.lbl_interval_hud.setText(f"Ensemble Active (m={coeff:.3f}, {fps} Hz)")
                self.lbl_interval_hud.setStyleSheet("color: #38bdf8; font-weight: bold; font-family: monospace; font-size: 11px; min-width: 220px;")
            else:
                self.lbl_interval_hud.setText(f"{steps} steps ({dt:.2f}s / {int(dt*1000)}ms @ {fps} FPS)")
                self.lbl_interval_hud.setStyleSheet("color: #f59e0b; font-weight: bold; font-family: monospace; font-size: 11px; min-width: 220px;")

    def toggle_dual_cameras(self):
        self.set_dual_cameras_enabled(not self.enable_dual_cameras)

    def set_dual_cameras_enabled(self, enabled: bool):
        self.enable_dual_cameras = enabled
        self.cam_widget_2.setVisible(enabled)
        self.lbl_cam2.setVisible(enabled)
        self.combo_cam2.setVisible(enabled)

        if enabled:
            self.btn_toggle_cam2.setText("✖ 1-Camera Mode")
            if self.camera_worker_2 is None:
                self.camera_worker_2 = CameraWorker(self.camera_2_index, self.target_fps, 320, 240, self.mock_mode, self.is_grayscale)
                self.camera_worker_2.frame_ready.connect(self.cam_widget_2.update_frame)
                self.camera_worker_2.fps_updated.connect(self.cam_widget_2.update_fps)
                self.camera_worker_2.start()
        else:
            self.btn_toggle_cam2.setText("+ 2nd Camera")
            if self.camera_worker_2 is not None:
                self.camera_worker_2.stop()
                self.camera_worker_2 = None

    def on_cam1_device_changed(self, index: int):
        dev_idx = self.combo_cam1.itemData(index)
        if dev_idx is not None and dev_idx != self.camera_1_index:
            self.camera_1_index = dev_idx
            self.camera_worker_1.stop()
            self.camera_worker_1 = CameraWorker(self.camera_1_index, self.target_fps, 320, 240, self.mock_mode, self.is_grayscale)
            self.camera_worker_1.frame_ready.connect(self.cam_widget_1.update_frame)
            self.camera_worker_1.fps_updated.connect(self.cam_widget_1.update_fps)
            self.camera_worker_1.start()

    def on_cam2_device_changed(self, index: int):
        dev_idx = self.combo_cam2.itemData(index)
        if dev_idx is not None and dev_idx != self.camera_2_index:
            self.camera_2_index = dev_idx
            if self.camera_worker_2:
                self.camera_worker_2.stop()
                self.camera_worker_2 = CameraWorker(self.camera_2_index, self.target_fps, 320, 240, self.mock_mode, self.is_grayscale)
                self.camera_worker_2.frame_ready.connect(self.cam_widget_2.update_frame)
                self.camera_worker_2.fps_updated.connect(self.cam_widget_2.update_fps)
                self.camera_worker_2.start()

    # -------------------------------------------------------------------------
    # Robot Telemetry & Freedrive Slots
    # -------------------------------------------------------------------------
    @pyqtSlot(bool, str)
    def on_robot_connection_status(self, is_connected: bool, message: str):
        level = "INFO" if is_connected else "ERROR"
        print(f"[{level}] [UR5e] {message}")
        self.status_bar.showMessage(message, 5000)
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop:
            if is_connected and hasattr(self.robot_worker, "_robot") and self.robot_worker._robot:
                if hasattr(self.robot_worker._robot, "sock") and self.robot_worker._robot.sock:
                    self.gamepad_teleop.set_external_socket(self.robot_worker._robot.sock)
            elif not is_connected and "Mock" in message:
                self.mock_mode = True
                self.gamepad_teleop.mock = True
                self.gamepad_teleop.set_velocity_callback(self.robot_worker.apply_cartesian_velocity)

    @pyqtSlot(bool)
    def on_freedrive_state_changed(self, active: bool):
        self.freedrive_active = active
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop:
            if active:
                self.gamepad_teleop.set_inhibited(True, "Freedrive active")
            else:
                self.gamepad_teleop.set_inhibited(False)
        if active:
            self.lbl_fd_status.setText("ROBOT STATE: FREEDRIVE ACTIVE (Hand-Guiding)")
            self.lbl_fd_status.setStyleSheet("""
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
                padding: 4px 8px;
                background-color: #d97706;
                border-radius: 4px;
            """)
            self.btn_toggle_freedrive.setText("🔒 LOCK ROBOT JOINTS")
            self.btn_toggle_freedrive.setStyleSheet("""
                QPushButton {
                    background-color: #d97706;
                    color: #ffffff;
                    font-size: 11px;
                    font-weight: bold;
                    border: 2px solid #f59e0b;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #b45309;
                }
            """)
            if hasattr(self, "lbl_jog_notice"):
                self.lbl_jog_notice.setText("Freedrive ACTIVE — Commanded motion will pause & restore freedrive")
                self.lbl_jog_notice.setStyleSheet("color: #f59e0b; font-size: 10px; font-weight: bold;")
            print("[INFO] Freedrive ENABLED — Robot arm is compliant for hand guiding.")
        else:
            self.lbl_fd_status.setText("ROBOT STATE: LOCKED (Position Hold)")
            self.lbl_fd_status.setStyleSheet("""
                color: #94a3b8;
                font-weight: bold;
                font-size: 11px;
                padding: 4px 8px;
                background-color: #0b1120;
                border-radius: 4px;
                border: 1px solid #1e293b;
            """)
            self.btn_toggle_freedrive.setText("⚡ TOGGLE FREEDRIVE MODE")
            self.btn_toggle_freedrive.setStyleSheet("""
                QPushButton {
                    background-color: #475569;
                    color: #ffffff;
                    font-size: 11px;
                    font-weight: bold;
                    border: 1px solid #64748b;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #64748b;
                }
            """)
            if hasattr(self, "lbl_jog_notice"):
                self.lbl_jog_notice.setText("Freedrive LOCKED — Manual jogging active")
                self.lbl_jog_notice.setStyleSheet("color: #4ade80; font-size: 10px; font-weight: bold;")
            print("[INFO] Freedrive DISABLED — Robot arm is locked in place.")

    @pyqtSlot(dict)
    def on_telemetry_updated(self, telem: dict):
        joints = telem.get("joints", {})
        rtde_hz = telem.get("rtde_hz", 0.0)
        latency_ms = telem.get("latency_ms", 0.0)

        self.lbl_rtde_rate.setText(f"RTDE: {rtde_hz:5.1f} Hz")
        self.lbl_rtde_latency.setText(f"Latency: {latency_ms:4.1f} ms")

        for j_name, rad_val in joints.items():
            deg_val = math.degrees(rad_val)
            if j_name in self.joint_labels:
                self.joint_labels[j_name].setText(f"{deg_val:+6.1f}°")
                self.joint_bars[j_name].setValue(int(deg_val))

            # Only sync jog sliders and inputs if this joint has NOT been modified by the user,
            # is not currently being dragged, and does not currently have keyboard focus
            if j_name not in self._user_modified_joints and self._user_dragging_joint != j_name:
                if hasattr(self, "jog_inputs") and j_name in self.jog_inputs:
                    inp = self.jog_inputs[j_name]
                    if not inp.hasFocus():
                        inp.setText(f"{deg_val:+.1f}")
                if hasattr(self, "jog_sliders") and j_name in self.jog_sliders:
                    slider = self.jog_sliders[j_name]
                    slider.blockSignals(True)
                    slider.setValue(int(round(deg_val)))
                    slider.blockSignals(False)

        # Sync live base motor angle and 6-DoF joint state to gamepad teleop
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop:
            if "shoulder_pan" in joints:
                self.gamepad_teleop.set_base_angle(joints["shoulder_pan"])
            if len(joints) >= 6:
                self.gamepad_teleop.set_joint_positions(joints)

    def on_freedrive_button_clicked(self):
        self.robot_worker.request_toggle_freedrive()

    # -------------------------------------------------------------------------
    # Gamepad Teleoperation Slots
    # -------------------------------------------------------------------------
    @pyqtSlot(bool, str)
    def on_gamepad_status_updated(self, is_connected: bool, name: str):
        if not hasattr(self, "lbl_gamepad_badge"):
            return
        if is_connected:
            self.lbl_gamepad_badge.setText(f"🎮 Connected: {name}")
            self.lbl_gamepad_badge.setStyleSheet("""
                color: #4ade80;
                font-size: 11px;
                font-weight: bold;
                padding: 4px 8px;
                background-color: #052e16;
                border: 1px solid #166534;
                border-radius: 4px;
            """)
        else:
            self.lbl_gamepad_badge.setText("🎮 No Gamepad Detected")
            self.lbl_gamepad_badge.setStyleSheet("""
                color: #64748b;
                font-size: 11px;
                font-weight: bold;
                padding: 4px 8px;
                background-color: #0b1120;
                border: 1px solid #1e293b;
                border-radius: 4px;
            """)
            if hasattr(self, "btn_toggle_gamepad") and self.btn_toggle_gamepad.isChecked():
                self.btn_toggle_gamepad.setChecked(False)

    @pyqtSlot(bool)
    def on_gamepad_toggle_clicked(self, checked: bool):
        if not hasattr(self, "gamepad_teleop") or self.gamepad_teleop is None:
            QMessageBox.warning(
                self,
                "Gamepad Unavailable",
                "Gamepad controller module is not available or pygame is missing.\n"
                "Please run: pip install pygame"
            )
            if hasattr(self, "btn_toggle_gamepad"):
                self.btn_toggle_gamepad.setChecked(False)
            return

        if checked:
            if not self.gamepad_teleop.is_controller_connected():
                QMessageBox.warning(
                    self,
                    "No Gamepad Detected",
                    "No USB or Bluetooth gamepad is currently detected.\n\n"
                    "Please connect a controller (Xbox, PlayStation, Logitech, etc.) and try again."
                )
                self.btn_toggle_gamepad.setChecked(False)
                return

            # Safety Interlock: Mutual exclusivity with Freedrive
            if self.freedrive_active:
                print("[GAMEPAD] Automatically disabling Freedrive mode to engage Gamepad Teleoperation.")
                self.robot_worker.request_toggle_freedrive()

            # Ensure shared socket is provided to gamepad worker
            if hasattr(self.robot_worker, "_robot") and self.robot_worker._robot:
                if hasattr(self.robot_worker._robot, "sock") and self.robot_worker._robot.sock:
                    self.gamepad_teleop.set_external_socket(self.robot_worker._robot.sock)

            self.gamepad_teleop.set_enabled(True)
            self.gamepad_teleop.set_inhibited(False)
            self.gamepad_teleop_active = True
            self.btn_toggle_gamepad.setText("🎮 GAMEPAD TELEOP: ACTIVE")
            self.btn_toggle_gamepad.setStyleSheet("""
                QPushButton {
                    background-color: #0284c7;
                    color: #ffffff;
                    font-size: 11px;
                    font-weight: bold;
                    border: 2px solid #38bdf8;
                    border-radius: 4px;
                    padding: 4px 10px;
                }
                QPushButton:hover {
                    background-color: #0369a1;
                }
            """)
            print(f"[GAMEPAD] Cartesian Teleoperation ENABLED with {self.gamepad_teleop.get_controller_name()}.")
            self.status_bar.showMessage(f"Gamepad Teleoperation Active ({self.gamepad_teleop.get_controller_name()}).", 5000)
        else:
            self.gamepad_teleop.set_enabled(False)
            self.gamepad_teleop_active = False
            self.btn_toggle_gamepad.setText("🎮 ENABLE GAMEPAD TELEOP")
            self.btn_toggle_gamepad.setStyleSheet("""
                QPushButton {
                    background-color: #1e293b;
                    color: #94a3b8;
                    font-size: 11px;
                    font-weight: bold;
                    border: 1px solid #334155;
                    border-radius: 4px;
                    padding: 4px 10px;
                }
                QPushButton:hover {
                    background-color: #334155;
                    color: #f1f5f9;
                }
            """)
            if hasattr(self, "lbl_gamepad_vel"):
                self.lbl_gamepad_vel.setText("TCP: 0.00 m/s")
            print("[GAMEPAD] Cartesian Teleoperation DISABLED.")
            self.status_bar.showMessage("Gamepad Teleoperation Disabled.", 3000)

    @pyqtSlot(list)
    def on_gamepad_velocity_updated(self, v: list):
        if not hasattr(self, "lbl_gamepad_vel"):
            return
        vx, vy, vz, w_base, w_w2, w_w3 = v
        v_lin = math.sqrt(vx * vx + vy * vy + vz * vz)
        has_joint = any(abs(x) > 1e-4 for x in (w_base, w_w2, w_w3))
        has_cart = v_lin > 1e-4

        if has_joint:
            parts = []
            if abs(w_base) > 1e-4:
                theta_deg = math.degrees(self.gamepad_teleop.get_base_angle()) if hasattr(self, "gamepad_teleop") and self.gamepad_teleop else 0.0
                parts.append(f"Base:{w_base:+5.2f} [{theta_deg:+.0f}°]")
            if abs(w_w2) > 1e-4:
                parts.append(f"W2:{w_w2:+5.2f}")
            if abs(w_w3) > 1e-4:
                parts.append(f"W3:{w_w3:+5.2f}")
            self.lbl_gamepad_vel.setText(" | ".join(parts) + " rad/s")
            self.lbl_gamepad_vel.setStyleSheet("""
                color: #a855f7;
                font-family: monospace;
                font-size: 11px;
                font-weight: bold;
                padding: 4px 8px;
                background-color: #2e1065;
                border: 1px solid #7e22ce;
                border-radius: 4px;
            """)
        elif has_cart:
            theta_deg = math.degrees(self.gamepad_teleop.get_base_angle()) if hasattr(self, "gamepad_teleop") and self.gamepad_teleop else 0.0
            self.lbl_gamepad_vel.setText(f"Vx:{vx:+5.2f} Vy:{vy:+5.2f} Vz:{vz:+5.2f} | {v_lin:4.2f} m/s [Heading:{theta_deg:+.0f}°]")
            self.lbl_gamepad_vel.setStyleSheet("""
                color: #38bdf8;
                font-family: monospace;
                font-size: 11px;
                font-weight: bold;
                padding: 4px 8px;
                background-color: #082f49;
                border: 1px solid #0284c7;
                border-radius: 4px;
            """)
        else:
            self.lbl_gamepad_vel.setText("TCP: 0.00 m/s")
            self.lbl_gamepad_vel.setStyleSheet("""
                color: #64748b;
                font-family: monospace;
                font-size: 11px;
                font-weight: bold;
                padding: 4px 8px;
                background-color: #0b1120;
                border: 1px solid #1e293b;
                border-radius: 4px;
            """)

    @pyqtSlot(bool, str)
    def on_gamepad_state_changed(self, is_active: bool, message: str):
        if hasattr(self, "lbl_gamepad_vel") and not is_active and hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.lbl_gamepad_vel.setText(f"PAUSED ({message})")

    @pyqtSlot(bool)
    def on_invert_left_lr_toggled(self, checked: bool):
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop is not None:
            self.gamepad_teleop.set_invert_left_horizontal(checked)
            mode_str = "INVERTED" if checked else "NORMAL"
            print(f"[GAMEPAD] Left Stick lateral control inverted: {checked} ({mode_str} mode)")
            self.status_bar.showMessage(f"Gamepad Left Stick lateral: {mode_str}", 3000)

    @pyqtSlot(bool)
    def on_invert_right_lr_toggled(self, checked: bool):
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop is not None:
            self.gamepad_teleop.set_invert_right_horizontal(checked)
            mode_str = "INVERTED" if checked else "NORMAL"
            print(f"[GAMEPAD] Right Stick base rotation inverted: {checked} ({mode_str} mode)")
            self.status_bar.showMessage(f"Gamepad Right Stick base rotation: {mode_str}", 3000)

    @pyqtSlot(bool)
    def on_invert_lr_toggled(self, checked: bool):
        self.on_invert_left_lr_toggled(checked)
        self.on_invert_right_lr_toggled(checked)

    @pyqtSlot(int)
    def on_gamepad_speed_changed(self, value: int):
        if hasattr(self, "lbl_gamepad_speed"):
            self.lbl_gamepad_speed.setText(f"⚡ Speed: {value}%")
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop is not None:
            self.gamepad_teleop.set_speed_percentage(value)

    @pyqtSlot(int)
    def on_gamepad_accel_changed(self, value: int):
        if hasattr(self, "lbl_gamepad_accel"):
            self.lbl_gamepad_accel.setText(f"🚀 Accel: {value}%")
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop is not None:
            self.gamepad_teleop.set_accel_percentage(value)

    @pyqtSlot()
    def on_gamepad_select_clicked(self):
        """Gamepad Select button pressed -> Trigger Go to Start / Home position."""
        if hasattr(self, "policy_running") and self.policy_running:
            print("[GAMEPAD] Cannot move to Start position while policy rollout is active.")
            self.status_bar.showMessage("Cannot Go to Start while rollout is active.", 3000)
            return
        print("[GAMEPAD BUTTON] 'Select' pressed -> Triggering Go to Start Pose.")
        if hasattr(self, "btn_go_start"):
            self.btn_go_start.click()
        else:
            self.move_to_start_pose()

    @pyqtSlot()
    def on_gamepad_start_clicked(self):
        """Gamepad Start button pressed -> Toggle Policy Evaluation (same as Enter)."""
        print("[GAMEPAD BUTTON] 'Start' pressed -> Toggling Policy Evaluation.")
        if hasattr(self, "btn_run_test"):
            self.btn_run_test.click()
        else:
            self.start_policy_evaluation()

    # -------------------------------------------------------------------------
    # Manual Jogging, Start Pose & Home Navigation Slots
    # -------------------------------------------------------------------------
    def sync_jog_to_robot(self):
        self._user_modified_joints.clear()
        self._user_targets.clear()
        current_rads = self.robot_worker.get_latest_joints()
        for j in UR5E_JOINTS:
            deg_val = math.degrees(current_rads.get(j, 0.0))
            if hasattr(self, "jog_inputs") and j in self.jog_inputs:
                self.jog_inputs[j].setText(f"{deg_val:+.1f}")
                self.jog_inputs[j].clearFocus()
            if hasattr(self, "jog_sliders") and j in self.jog_sliders:
                slider = self.jog_sliders[j]
                slider.blockSignals(True)
                slider.setValue(int(round(deg_val)))
                slider.blockSignals(False)
        print("[JOG] Synchronized all jog target fields to current robot position.")
        self.status_bar.showMessage("Jog inputs synchronized to current robot position.", 3000)

    def update_start_pose_display(self):
        pose_str = ", ".join(f"{self.start_position.get(j, 0.0):+.1f}°" for j in UR5E_JOINTS)
        if hasattr(self, "lbl_start_pose_info"):
            self.lbl_start_pose_info.setText(f"Start Pose: [{pose_str}]")
            lines = ["Stored Start Position:"]
            for j in UR5E_JOINTS:
                lines.append(f"  • {j.replace('_', ' ').title()}: {self.start_position.get(j, 0.0):+.1f}°")
            self.lbl_start_pose_info.setToolTip("\n".join(lines))

    def set_current_as_start_position(self):
        current_rads = self.robot_worker.get_latest_joints()
        for j in UR5E_JOINTS:
            self.start_position[j] = round(math.degrees(current_rads.get(j, 0.0)), 1)
        self.update_start_pose_display()
        pose_str = ", ".join(f"{self.start_position[j]:+.1f}°" for j in UR5E_JOINTS)
        print(f"[POSITION] Active Start Position set to: [{pose_str}]")
        self.status_bar.showMessage(f"Active Start Position saved: [{pose_str}]", 4000)

    def move_to_start_pose(self):
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.gamepad_teleop.set_inhibited(True, "Trajectory motion in progress")
        self._user_modified_joints.clear()
        self._user_targets.clear()
        action = {f"{j}.pos": math.radians(self.start_position.get(j, 0.0)) for j in UR5E_JOINTS}
        pose_str = ", ".join(f"{self.start_position.get(j, 0.0):+.1f}°" for j in UR5E_JOINTS)
        print(f"[MOTION] Commanding UR5e arm to Start Position [{pose_str}]...")
        self.status_bar.showMessage("Moving arm to Start Position...", 4000)
        self.robot_worker.request_move_pose(action)

    def move_to_home_pose(self):
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.gamepad_teleop.set_inhibited(True, "Trajectory motion in progress")
        self._user_modified_joints.clear()
        self._user_targets.clear()
        home_action = {f"{j}.pos": math.radians(HOME_JOINTS_DICT[j]) for j in UR5E_JOINTS}
        print("[MOTION] Commanding UR5e arm to Home position (0°, -90°, 90°, -90°, -90°, 0°)...")
        self.status_bar.showMessage("Moving arm to Home position...", 4000)
        self.robot_worker.request_move_pose(home_action)

    def get_numeric_input_value(self, j_name: str) -> float:
        if j_name in self._user_targets:
            return self._user_targets[j_name]
        if hasattr(self, "jog_inputs") and j_name in self.jog_inputs:
            text = self.jog_inputs[j_name].text().replace("°", "").strip()
            try:
                val = float(text)
                return max(-360.0, min(360.0, val))
            except ValueError:
                pass
        if hasattr(self, "jog_sliders") and j_name in self.jog_sliders:
            return float(self.jog_sliders[j_name].value())
        return 0.0

    def move_to_all_targets(self):
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.gamepad_teleop.set_inhibited(True, "Trajectory motion in progress")
        for j in UR5E_JOINTS:
            if hasattr(self, "jog_inputs") and j in self.jog_inputs:
                text = self.jog_inputs[j].text().replace("°", "").strip()
                try:
                    self._user_targets[j] = max(-360.0, min(360.0, float(text)))
                except ValueError:
                    pass

        target_rads = {}
        for j in UR5E_JOINTS:
            val_deg = self.get_numeric_input_value(j)
            target_rads[f"{j}.pos"] = math.radians(val_deg)
            if hasattr(self, "jog_inputs") and j in self.jog_inputs:
                self.jog_inputs[j].setText(f"{val_deg:+.1f}")
                self.jog_inputs[j].clearFocus()
            if hasattr(self, "jog_sliders") and j in self.jog_sliders:
                slider = self.jog_sliders[j]
                slider.blockSignals(True)
                slider.setValue(int(round(val_deg)))
                slider.blockSignals(False)

        self._user_modified_joints.clear()
        self._user_targets.clear()

        print("[MOTION] Commanding UR5e arm to all specified target joint positions...")
        self.status_bar.showMessage("Moving arm to target joint positions...", 4000)
        self.robot_worker.request_move_pose(target_rads)

    def on_jog_slider_pressed(self, j_name: str):
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.gamepad_teleop.set_inhibited(True, "Jogging slider in progress")
        self._user_dragging_joint = j_name
        self._user_modified_joints.add(j_name)

    def on_jog_slider_value_changed(self, j_name: str, val: int):
        if self._user_dragging_joint == j_name:
            self._user_targets[j_name] = float(val)
            self._user_modified_joints.add(j_name)
        if hasattr(self, "jog_inputs") and j_name in self.jog_inputs:
            self.jog_inputs[j_name].setText(f"{float(val):+.1f}")

    def on_jog_slider_released(self, j_name: str):
        if not hasattr(self, "jog_sliders") or j_name not in self.jog_sliders:
            self._user_dragging_joint = None
            return

        target_deg = float(self.jog_sliders[j_name].value())
        self._user_targets.pop(j_name, None)
        self._user_modified_joints.discard(j_name)
        target_rad = math.radians(target_deg)

        if hasattr(self, "jog_inputs") and j_name in self.jog_inputs:
            self.jog_inputs[j_name].setText(f"{target_deg:+.1f}")

        current_joints = self.robot_worker.get_latest_joints()
        action = {f"{j}.pos": current_joints.get(j, 0.0) for j in UR5E_JOINTS}
        action[f"{j_name}.pos"] = target_rad

        print(f"[MOTION] Jogging {j_name} to {target_deg:+5.1f}° ({target_rad:+.4f} rad)...")
        self.status_bar.showMessage(f"Jogging {j_name} to {target_deg:+5.1f}°...", 3000)
        self.robot_worker.request_move_pose(action)
        self._user_dragging_joint = None

    def on_jog_input_edited(self, j_name: str, text: str):
        clean_text = text.replace("°", "").strip()
        try:
            val = float(clean_text)
            clamped = max(-360.0, min(360.0, val))
            self._user_targets[j_name] = clamped
            self._user_modified_joints.add(j_name)
            if hasattr(self, "jog_sliders") and j_name in self.jog_sliders:
                slider = self.jog_sliders[j_name]
                slider.blockSignals(True)
                slider.setValue(int(round(clamped)))
                slider.blockSignals(False)
        except ValueError:
            pass

    def on_jog_input_committed(self, j_name: str):
        if hasattr(self, "jog_inputs") and j_name in self.jog_inputs:
            text = self.jog_inputs[j_name].text().replace("°", "").strip()
            try:
                self._user_targets[j_name] = max(-360.0, min(360.0, float(text)))
            except ValueError:
                pass

        val_deg = self.get_numeric_input_value(j_name)
        self._user_targets.pop(j_name, None)
        self._user_modified_joints.discard(j_name)

        if hasattr(self, "jog_inputs") and j_name in self.jog_inputs:
            self.jog_inputs[j_name].setText(f"{val_deg:+.1f}")
            self.jog_inputs[j_name].clearFocus()
        if hasattr(self, "jog_sliders") and j_name in self.jog_sliders:
            slider = self.jog_sliders[j_name]
            slider.blockSignals(True)
            slider.setValue(int(round(val_deg)))
            slider.blockSignals(False)

        target_rad = math.radians(val_deg)
        current_joints = self.robot_worker.get_latest_joints()
        action = {f"{j}.pos": current_joints.get(j, 0.0) for j in UR5E_JOINTS}
        action[f"{j_name}.pos"] = target_rad

        print(f"[MOTION] Jogging {j_name} to {val_deg:+5.1f}° ({target_rad:+.4f} rad)...")
        self.status_bar.showMessage(f"Jogging {j_name} to {val_deg:+5.1f}°...", 3000)
        self.robot_worker.request_move_pose(action)

    # -------------------------------------------------------------------------
    # Gripper Manual Control Slots (Slider, Input Box & Gamepad Teleop)
    # -------------------------------------------------------------------------
    def on_gripper_slider_pressed(self):
        self._user_dragging_gripper = True

    def on_gripper_slider_value_changed(self, val: int):
        if hasattr(self, "gripper_input"):
            self.gripper_input.setText(f"{float(val):.1f}")

    def on_gripper_slider_released(self):
        self._user_dragging_gripper = False
        if hasattr(self, "gripper_slider"):
            val = float(self.gripper_slider.value())
            self.command_gripper_width(val)

    def on_gripper_input_committed(self):
        if hasattr(self, "gripper_input"):
            text = self.gripper_input.text().replace("mm", "").strip()
            try:
                val = float(text)
                self.command_gripper_width(val)
            except ValueError:
                pass

    def command_gripper_width(self, width_mm: float):
        min_w = getattr(self.gripper, "min_width_mm", 55.0) if self.gripper else 55.0
        max_w = getattr(self.gripper, "max_width_mm", 105.0) if self.gripper else 105.0
        clamped = max(min_w, min(max_w, float(width_mm)))

        if hasattr(self, "gripper_slider"):
            self.gripper_slider.blockSignals(True)
            self.gripper_slider.setValue(int(round(clamped)))
            self.gripper_slider.blockSignals(False)

        if hasattr(self, "gripper_input"):
            self.gripper_input.setText(f"{clamped:.1f}")

        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop is not None:
            self.gamepad_teleop.set_gripper_target_width(clamped)

        if self.gripper is not None and self.gripper.is_connected:
            try:
                self.gripper.grip(clamped, force_n=30.0, speed_percent=50, wait=False)
                print(f"[GRIPPER] Commanded width to {clamped:.1f} mm (30N force).")
                self.status_bar.showMessage(f"Gripper moving to {clamped:.1f} mm...", 3000)
            except Exception as e:
                print(f"[GRIPPER ERROR] Failed to command gripper: {e}")
                self.status_bar.showMessage(f"Gripper command failed: {e}", 3000)
        else:
            print(f"[GRIPPER SIM] Commanded width: {clamped:.1f} mm.")
            self.status_bar.showMessage(f"Gripper set to {clamped:.1f} mm (Simulation).", 3000)

    @pyqtSlot(float)
    def on_gamepad_gripper_width_updated(self, width_mm: float):
        """Called live when the user holds Xbox [A] (close) or [B] (open)."""
        if not getattr(self, "_user_dragging_gripper", False):
            if hasattr(self, "gripper_slider"):
                self.gripper_slider.blockSignals(True)
                self.gripper_slider.setValue(int(round(width_mm)))
                self.gripper_slider.blockSignals(False)
            if hasattr(self, "gripper_input"):
                self.gripper_input.setText(f"{width_mm:.1f}")

    # -------------------------------------------------------------------------
    # Policy Evaluation Rollout Workflow
    # -------------------------------------------------------------------------
    def start_policy_evaluation(self):
        if self.rollout_worker is not None and self.rollout_worker.isRunning():
            return

        if not self.active_model_path.exists() or not (self.active_model_path / "config.json").exists():
            QMessageBox.critical(self, "Model Not Found", f"Cannot find valid checkpoint directory at:\n{self.active_model_path}")
            return

        # =====================================================================
        # Camera Count & Configuration Validation Gate
        # =====================================================================
        required_cams = getattr(self, "model_required_cams", 1)
        req_cam_names = getattr(self, "model_camera_names", ["Workspace"])
        active_cams = 2 if (self.enable_dual_cameras and self.camera_worker_2 is not None) else 1

        if required_cams == 2 and active_cams != 2:
            warning_msg = (
                f"<h3>⚠️ Camera Setup Mismatch</h3>"
                f"<p>The loaded model was trained with <b>2 Camera Inputs</b>:</p>"
                f"<ul>"
                f"<li><b>Camera 1:</b> Primary Workspace View</li>"
                f"<li><b>Camera 2:</b> Wrist Flange View</li>"
                f"</ul>"
                f"<p>However, currently only <b>{active_cams} camera</b> is active in the studio.</p>"
                f"<p style='color: #ef4444; font-weight: bold;'>Autonomous policy evaluation has been prevented to avoid feeding incomplete observation tensors to the neural network.</p>"
                f"<hr>"
                f"<p><b>How to resolve this:</b></p>"
                f"<ol>"
                f"<li>Click the <b>'+ 2nd Camera'</b> button in the Camera Controls bar.</li>"
                f"<li>Select the correct <code>/dev/video</code> node for your Wrist camera.</li>"
                f"<li>Confirm that both live video windows are displaying active camera feeds before starting the evaluation.</li>"
                f"</ol>"
            )
            print(f"[POLICY VALIDATION ERROR] Cannot start evaluation: Model was trained with 2 cameras ({' + '.join(req_cam_names)}), but only {active_cams} camera is active.", file=sys.__stderr__)
            self.status_bar.showMessage("Evaluation blocked: Model requires 2 camera inputs (Workspace + Wrist).", 8000)
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Camera Input Mismatch")
            msg_box.setTextFormat(Qt.TextFormat.RichText)
            msg_box.setText(warning_msg)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            return

        if required_cams == 1 and active_cams != 1:
            warning_msg = (
                f"<h3>⚠️ Camera Setup Mismatch</h3>"
                f"<p>The loaded model was trained with only <b>1 Camera Input</b> (Workspace View).</p>"
                f"<p>However, <b>2 Cameras (Dual-Cam Mode)</b> are currently enabled in the studio.</p>"
                f"<p style='color: #ef4444; font-weight: bold;'>Autonomous policy evaluation has been prevented to avoid unexpected observation dimensionality mismatch.</p>"
                f"<hr>"
                f"<p><b>How to resolve this:</b></p>"
                f"<ol>"
                f"<li>Click the <b>'✖ 1-Camera Mode'</b> button in the Camera Controls bar to disable the secondary wrist camera.</li>"
                f"<li>Confirm that only the Primary Workspace Camera feed is visible before starting the evaluation.</li>"
                f"</ol>"
            )
            print(f"[POLICY VALIDATION ERROR] Cannot start evaluation: Model was trained with 1 camera ({' + '.join(req_cam_names)}), but 2 cameras are currently active.", file=sys.__stderr__)
            self.status_bar.showMessage("Evaluation blocked: Model requires 1 camera input (Workspace only).", 8000)
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Camera Input Mismatch")
            msg_box.setTextFormat(Qt.TextFormat.RichText)
            msg_box.setText(warning_msg)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            return

        if self.camera_worker_1 is None:
            QMessageBox.critical(self, "Camera Error", "Primary Workspace camera is not initialized.")
            return

        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.gamepad_teleop.set_inhibited(True, "Autonomous rollout in progress")

        duration = self.spin_duration.value()
        max_jump = self.spin_max_jump.value()
        auto_fd = self.chk_auto_freedrive.isChecked()
        return_start = self.chk_return_start.isChecked()

        # Update UI Controls to Active Rollout State
        self.btn_run_test.setEnabled(False)
        self.btn_run_test.setText("⏳  EVALUATION IN PROGRESS...")
        self.btn_stop_test.setEnabled(True)
        self.cam_widget_1.set_evaluating(True)
        self.cam_widget_2.set_evaluating(True)

        total_steps = int(duration * 30)
        self.progress_bar.setRange(0, total_steps)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"Starting Rollout (0 / {total_steps} steps)")

        self.status_bar.showMessage("Executing autonomous policy evaluation rollout...", 10000)

        # Launch background rollout thread
        n_steps = self.slider_action_steps.value() if hasattr(self, "slider_action_steps") else 10
        use_ensemble = self.chk_temporal_ensemble.isChecked() if hasattr(self, "chk_temporal_ensemble") else True
        ens_coeff = self.spin_ensemble_coeff.value() if hasattr(self, "spin_ensemble_coeff") else 0.01
        self.rollout_worker = PolicyRolloutWorker(
            model_path=self.active_model_path,
            robot_worker=self.robot_worker,
            camera_worker_1=self.camera_worker_1,
            camera_worker_2=self.camera_worker_2,
            duration_s=duration,
            control_fps=getattr(self, "model_fps", 30),
            max_jump_deg=max_jump,
            n_action_steps=n_steps,
            temporal_ensemble=use_ensemble,
            temporal_ensemble_coeff=ens_coeff,
            auto_freedrive_after=auto_fd,
            return_start_after=return_start,
            start_pose_deg=self.start_position,
            device_name="cuda" if (torch and torch.cuda.is_available()) else "cpu",
            mock=self.mock_mode,
        )
        self.rollout_worker.step_progress.connect(self.on_rollout_step_progress)
        self.rollout_worker.rollout_finished.connect(self.on_rollout_finished)
        self.rollout_worker.start()

    def abort_policy_evaluation(self):
        if self.rollout_worker is not None and self.rollout_worker.isRunning():
            print("[USER] Abort evaluation requested by user.")
            self.btn_stop_test.setText("⏳ Aborting...")
            self.btn_stop_test.setEnabled(False)
            self.rollout_worker.request_abort()

    @pyqtSlot(int, int, float, list)
    def on_rollout_step_progress(self, step: int, total_steps: int, elapsed_s: float, current_q: list):
        self.progress_bar.setValue(step)
        self.progress_bar.setFormat(f"Step {step}/{total_steps} ({elapsed_s:.1f}s) | Active Rollout")

    @pyqtSlot(bool, str)
    def on_rollout_finished(self, success: bool, message: str):
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.gamepad_teleop.set_inhibited(False)
        self.btn_run_test.setEnabled(True)
        self.btn_run_test.setText("▶  RUN POLICY EVALUATION (Enter)")
        self.btn_stop_test.setEnabled(False)
        self.btn_stop_test.setText("⏹  STOP / ABORT (Esc)")
        self.cam_widget_1.set_evaluating(False)
        self.cam_widget_2.set_evaluating(False)

        if success:
            self.progress_bar.setFormat("✔ Rollout Completed Successfully")
            self.status_bar.showMessage("Evaluation rollout completed successfully.", 5000)
        else:
            self.progress_bar.setFormat("Rollout Stopped / Aborted")
            self.status_bar.showMessage(message, 6000)

    # -------------------------------------------------------------------------
    # Log Redirection & Console Helper Slots
    # -------------------------------------------------------------------------
    @pyqtSlot(str, str)
    def append_log(self, level: str, message: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        color = "#e2e8f0"
        if level == "ERROR" or "ERROR" in message or "CRITICAL" in message:
            color = "#f87171"
        elif level == "WARNING" or "WARNING" in message or "WARN" in message:
            color = "#fbbf24"
        elif "MOTION" in message:
            color = "#38bdf8"
        elif "SUCCESS" in message:
            color = "#4ade80"
        elif "RECORD" in message or "EVAL" in message:
            color = "#a78bfa"

        html = f'<span style="color: #64748b;">[{timestamp}]</span> <span style="color: {color};">{message}</span>'
        self.txt_log.appendHtml(html)
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def copy_log_to_clipboard(self):
        QApplication.clipboard().setText(self.txt_log.toPlainText())
        self.status_bar.showMessage("Console log copied to clipboard.", 3000)

    def clear_log(self):
        self.txt_log.clear()

    # -------------------------------------------------------------------------
    # Window Close Event Cleanup
    # -------------------------------------------------------------------------
    def closeEvent(self, event):
        print("[SYSTEM] Shutting down Policy Testing Studio...")
        if self.rollout_worker and self.rollout_worker.isRunning():
            self.rollout_worker.request_abort()
            self.rollout_worker.wait(1000)

        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop is not None:
            try:
                self.gamepad_teleop.stop()
            except Exception:
                pass

        if self.camera_worker_1:
            self.camera_worker_1.stop()
        if self.camera_worker_2:
            self.camera_worker_2.stop()
        if self.robot_worker:
            self.robot_worker.stop()

        event.accept()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="UR5e LeRobot — Policy Testing & Evaluation Studio")
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP, help="IP address of UR5e arm (Port 30004/30002)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained policy checkpoint directory")
    parser.add_argument("--mock", action="store_true", help="Launch in Simulation/Mock Mode without physical arm")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("UR5e Policy Testing Studio")

    studio = UR5ePolicyTesterStudio(
        robot_ip=args.robot_ip,
        model_path=args.model_path,
        mock=args.mock,
    )
    studio.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

