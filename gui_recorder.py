#!/usr/bin/env python3
"""
UR5e LeRobot Demonstration Dataset Studio (PyQt6)
==================================================
Standalone desktop application for recording demonstration datasets on a
Universal Robots UR5e arm with integrated camera feeds, RTDE telemetry (125 Hz),
Freedrive physical teleoperation, and LeRobot dataset management.

Features:
- Decoupled multithreading (UI, RTDE telemetry polling, OpenCV camera, recorder)
- Modular camera feed layout (workspace primary + expandable secondary feed)
- Real-time 6-DoF joint telemetry dashboard & RTDE benchmark HUD
- Dataset & Episode Management (scan existing folder, count, inspect, delete last)
- Hotkey-driven workflow (Space = Start/End, 'c'/Left = Cancel, Esc = Stop)
- Strict Freedrive safety enforcement before recording
- Hardware diagnostics runner (executes test_lerobot_ur5e.py in background thread)
- Embedded timestamped live log terminal console with stdout/stderr redirection
- Automatic Simulation/Mock fallback mode for offline GUI development
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime
import json
import math
import os
from pathlib import Path
import pickle
import queue
import shutil
import subprocess
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
    QFileDialog,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
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

# Optional PyTorch & LeRobot imports with graceful fallback
try:
    import torch
except ImportError:
    torch = None

try:
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots import UR5eRobotConfig, make_robot_from_config
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

# Gamepad Cartesian Teleoperation Controller
try:
    from gamepad_controller import GamepadConfig, UR5eGamepadTeleop, PYGAME_AVAILABLE
    GAMEPAD_AVAILABLE = True
except ImportError:
    GAMEPAD_AVAILABLE = False
    UR5eGamepadTeleop = None
    GamepadConfig = None
    PYGAME_AVAILABLE = False

# OnRobot 2FG Gripper Modbus TCP Controller
try:
    from onrobot_gripper import get_shared_gripper, OnRobot2FGModbus
    ONROBOT_GRIPPER_AVAILABLE = True
except ImportError:
    get_shared_gripper = None
    OnRobot2FGModbus = None
    ONROBOT_GRIPPER_AVAILABLE = False

# Hybrid Trajectory Assist & Min-Jerk Recovery
try:
    from path_manager import PathManager, TrajectoryPoint
    from hybrid_path_runner import HybridPathWorker, HybridState
    HYBRID_AVAILABLE = True
except ImportError as err:
    HYBRID_AVAILABLE = False
    PathManager = None
    TrajectoryPoint = None
    HybridPathWorker = None
    HybridState = None
    print(f"[HYBRID WARNING] Could not import hybrid components: {err}")



# Default Configuration
DEFAULT_ROBOT_IP = "192.168.1.102"
DEFAULT_CAMERA_INDEX = 4         # HD Webcam C525 (Workspace)
DEFAULT_CAMERA_2_INDEX = 6       # eFace 1325R (Wrist Flange)
DEFAULT_FPS = 30
DEFAULT_DATASET_NAME = "ur5e_assembly"
DEFAULT_DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

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

# Default Start Position loaded dynamically from default_start_position.json
try:
    from start_position_config import (
        get_default_start_joints_dict,
        get_default_start_joints_deg,
        get_default_start_gripper_mm,
        save_start_position as persist_start_position,
    )
    DEFAULT_START_JOINTS = get_default_start_joints_deg()
    DEFAULT_START_JOINTS_DICT = get_default_start_joints_dict()
    DEFAULT_START_GRIPPER_MM = get_default_start_gripper_mm()
except ImportError:
    DEFAULT_START_JOINTS = [-90.0, -40.0, -100.0, 120.0, -90.0, 120.0]
    DEFAULT_START_JOINTS_DICT = {
        "shoulder_pan": -90.0,
        "shoulder_lift": -40.0,
        "elbow": -100.0,
        "wrist_1": 120.0,
        "wrist_2": -90.0,
        "wrist_3": 120.0,
    }
    DEFAULT_START_GRIPPER_MM = 105.0
    persist_start_position = None


def detect_available_cameras() -> List[Tuple[int, str]]:
    """
    Scans Linux v4l2 device nodes and returns a list of (camera_index, user_friendly_label).
    Filters out unsupported or metadata-only sub-nodes using the sysfs 'index' attribute.
    """
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


# =============================================================================
# 1. Thread-Safe Live Log Redirection
# =============================================================================

class LogEmitter(QObject):
    log_signal = pyqtSignal(str, str)  # (level, message)


class StreamToLogger:
    """Redirects writes from stdout/stderr to Qt signal emitter while preserving terminal output."""
    def __init__(self, emitter: LogEmitter, level: str = "INFO", original_stream=None):
        self.emitter = emitter
        self.level = level
        self.original_stream = original_stream
        self._buffer = ""

    def write(self, text: str):
        if self.original_stream:
            try:
                self.original_stream.write(text)
                self.original_stream.flush()
            except Exception:
                pass
        if not text:
            return
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.emitter.log_signal.emit(self.level, line)

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass
        if self._buffer.strip():
            self.emitter.log_signal.emit(self.level, self._buffer.strip())
            self._buffer = ""


# =============================================================================
# 2. Asynchronous Camera Capture Thread
# =============================================================================

class CameraWorker(QThread):
    """
    Dedicated QThread capturing frames from OpenCV camera.
    Runs at target FPS asynchronously to prevent UI stutter.
    """
    frame_ready = pyqtSignal(int, np.ndarray)  # (camera_id, bgr_image)
    fps_updated = pyqtSignal(int, float)       # (camera_id, measured_fps)
    status_signal = pyqtSignal(int, bool, str) # (camera_id, is_connected, message)

    def __init__(self, camera_id: int, target_fps: int = 30, width: int = 320, height: int = 240, mock: bool = False, grayscale: bool = False, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.target_width = width
        self.target_height = height
        self.mock = mock
        self._grayscale = grayscale
        self._running = False
        self._cap = None
        self._mutex = QMutex()
        self._reconnect_requested = False

    def set_grayscale(self, enabled: bool):
        self._mutex.lock()
        self._grayscale = bool(enabled)
        self._mutex.unlock()

    def set_resolution(self, width: int, height: int):
        self._mutex.lock()
        self.target_width = width
        self.target_height = height
        cap = self._cap
        self._mutex.unlock()
        if cap is not None and cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            except Exception:
                pass

    def request_reconnect(self):
        """Requests the camera capture loop to cleanly release and reopen the video device."""
        self._mutex.lock()
        self._reconnect_requested = True
        self._mutex.unlock()

    def run(self):
        self._running = True

        if not self.mock:
            self._cap = cv2.VideoCapture(self.camera_id)
            # Request capture parameters from driver
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            if not self._cap.isOpened():
                self.status_signal.emit(
                    self.camera_id,
                    False,
                    f"Camera /dev/video{self.camera_id} unavailable (Falling back to test pattern)"
                )
                self.mock = True
            else:
                self.status_signal.emit(
                    self.camera_id,
                    True,
                    f"Camera /dev/video{self.camera_id} connected @ {self.target_fps} FPS"
                )

        frame_interval = 1.0 / max(1, self.target_fps)
        last_frame_time = time.perf_counter()
        fps_counter = 0
        fps_start = time.perf_counter()

        sim_t = 0.0
        consecutive_failures = 0

        while self._running:
            t0 = time.perf_counter()

            self._mutex.lock()
            tw, th = self.target_width, self.target_height
            reconnect_needed = self._reconnect_requested
            self._reconnect_requested = False
            self._mutex.unlock()

            if reconnect_needed and not self.mock:
                if self._cap is not None:
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None
                time.sleep(0.05)
                self._cap = cv2.VideoCapture(self.camera_id)
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, tw)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, th)
                self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                consecutive_failures = 0
                if self._cap.isOpened():
                    self.status_signal.emit(
                        self.camera_id,
                        True,
                        f"Camera /dev/video{self.camera_id} reconnected @ {self.target_fps} FPS"
                    )

            if not self.mock and self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    # Auto-recover if video frame acquisition stalls (e.g. V4L2 queue drops during disk save/h264 encode)
                    if consecutive_failures >= 15:
                        try:
                            self._cap.release()
                        except Exception:
                            pass
                        time.sleep(0.05)
                        self._cap = cv2.VideoCapture(self.camera_id)
                        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, tw)
                        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, th)
                        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                        consecutive_failures = 0
                    time.sleep(0.01)
                    continue
                consecutive_failures = 0
                # Resize to target resolution if different
                if frame.shape[1] != tw or frame.shape[0] != th:
                    frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            else:
                # Generate synthetic test frame with timestamp and visual animation matching current resolution
                frame = np.zeros((th, tw, 3), dtype=np.uint8)
                # Subtle gradient background
                sim_t += 0.03
                gx = int(128 + 60 * math.sin(sim_t))
                gy = int(128 + 60 * math.cos(sim_t * 0.8))
                frame[:, :, 0] = 30
                frame[:, :, 1] = int(gx * 0.3)
                frame[:, :, 2] = int(gy * 0.4)

                # Scaled grid pattern
                step = max(20, min(tw, th) // 8)
                for x in range(0, tw, step):
                    cv2.line(frame, (x, 0), (x, th), (45, 55, 72), 1)
                for y in range(0, th, step):
                    cv2.line(frame, (0, y), (tw, y), (45, 55, 72), 1)

                # Simulated end-effector crosshair moving smoothly
                cx = int(tw / 2 + (tw / 4) * math.sin(sim_t * 1.2))
                cy = int(th / 2 + (th / 4) * math.cos(sim_t * 1.5))
                r = max(6, min(tw, th) // 20)
                cv2.circle(frame, (cx, cy), r, (0, 215, 255), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 215, 255), -1)

                font_scale = max(0.35, min(tw, th) / 600.0)
                cv2.putText(
                    frame,
                    f"CAM {self.camera_id} [{tw}x{th}]",
                    (10, max(18, int(th * 0.12))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (200, 220, 255),
                    1,
                )
                time_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-4]
                cv2.putText(
                    frame,
                    f"TIME: {time_str}",
                    (10, max(36, th - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.9,
                    (150, 180, 200),
                    1,
                )

            # Apply grayscale mode if enabled
            self._mutex.lock()
            is_gray = self._grayscale
            self._mutex.unlock()

            if is_gray:
                if len(frame.shape) == 3:
                    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

            # Emit frame
            self.frame_ready.emit(self.camera_id, frame)

            # Compute actual FPS
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_start >= 1.0:
                measured_fps = fps_counter / (now - fps_start)
                self.fps_updated.emit(self.camera_id, measured_fps)
                fps_counter = 0
                fps_start = now

            # Sleep to maintain requested FPS rate
            elapsed = time.perf_counter() - t0
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None

    def stop(self):
        self._running = False
        self.wait(1500)


# =============================================================================
# 3. Asynchronous Robot RTDE Telemetry & Control Worker (125 Hz)
# =============================================================================

class RobotRTDEWorker(QThread):
    """
    Dedicated QThread maintaining real-time RTDE communication with UR5e.
    Polls joint angles at 125 Hz, tracks loop latency, and safely processes
    Freedrive toggles via URScript socket.
    """
    telemetry_signal = pyqtSignal(dict)           # Joint positions & telemetry stats
    freedrive_state_changed = pyqtSignal(bool)   # True if active, False if locked
    connection_state_changed = pyqtSignal(bool, str) # (is_connected, message)

    def __init__(self, robot_ip: str = DEFAULT_ROBOT_IP, mock: bool = False, parent=None):
        super().__init__(parent)
        self.robot_ip = robot_ip
        self.mock = mock
        self._running = False
        self._robot = None
        self._freedrive_active = False

        # Queue commands for the worker thread
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

        # Latest joint states (radians) and TCP pose [x, y, z, rx, ry, rz]
        self._latest_q = {j: 0.0 for j in UR5E_JOINTS}
        self._latest_tcp: List[float] = [-0.1334, -0.0631, 0.5503, 1.0980, -1.8947, 1.3240]
        self._mock_cartesian_v: List[float] = [0.0] * 6

    @property
    def is_freedrive_active(self) -> bool:
        return self._freedrive_active

    @property
    def is_motion_in_progress(self) -> bool:
        return self._motion_in_progress

    def apply_cartesian_velocity(self, v: List[float]):
        """Applies Cartesian velocity updates to the simulated arm model."""
        with QMutexLocker(self._cmd_lock):
            self._mock_cartesian_v = list(v)

    def request_toggle_freedrive(self):
        with QMutexLocker(self._cmd_lock):
            self._cmd_toggle_freedrive = True

    def request_move_pose(self, pose: Dict[str, float]):
        with QMutexLocker(self._cmd_lock):
            self._cmd_target_pose = dict(pose)

    def request_disconnect(self):
        with QMutexLocker(self._cmd_lock):
            self._cmd_disconnect = True

    def stream_servoj(self, target_q: List[float], dt: float = 0.02):
        """Streams a servoj position command smoothly to the physical arm or mock model."""
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

    def stream_speedj(self, qd: List[float], a: float = 1.4, t: float = 0.06):
        """Streams a speedj joint velocity command smoothly to the physical arm or mock model."""
        if not self.mock and self._robot and hasattr(self._robot, "sock") and self._robot.sock:
            qd_str = ", ".join(f"{val:.4f}" for val in qd)
            cmd = f"speedj([{qd_str}], a={a:.2f}, t={t:.2f})\n"
            try:
                self._robot.sock.sendall(cmd.encode("utf-8"))
            except Exception as e:
                print(f"[ERROR] speedj transmission failed: {e}")
        elif self.mock:
            with QMutexLocker(self._cmd_lock):
                for idx, j_name in enumerate(UR5E_JOINTS):
                    self._latest_q[j_name] += float(qd[idx]) * t

    def stop_motion(self, a: float = 2.0):
        """Halts motion immediately on the physical arm."""
        if not self.mock and self._robot and hasattr(self._robot, "sock") and self._robot.sock:
            try:
                self._robot.sock.sendall(f"stopj({a:.2f})\n".encode("utf-8"))
            except Exception:
                pass

    def run(self):
        self._running = True

        # 1. Connect to UR5e or fallback to mock
        if not self.mock and LEROBOT_AVAILABLE:
            try:
                from ur5e.configuration_ur5e import UR5eRobotConfig
                from ur5e.robot_ur5e import UR5eRobot
                config = UR5eRobotConfig(robot_ip=self.robot_ip)
                self._robot = UR5eRobot(config)
                self._robot.connect()
                self.connection_state_changed.emit(True, f"Connected to UR5e at {self.robot_ip}")
            except Exception as e:
                self.connection_state_changed.emit(
                    False,
                    f"Hardware connection failed ({e}). Entering Mock Robot Mode."
                )
                self.mock = True
                self._robot = None
        else:
            self.connection_state_changed.emit(
                True,
                f"Mock Robot Mode active (Simulated UR5e on {self.robot_ip})"
            )

        # Simulation kinematics parameters
        sim_t = 0.0
        mock_angles = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]

        # 125 Hz polling loop (period = 8ms)
        target_period = 1.0 / 125.0
        last_emit_time = time.perf_counter()
        emit_interval = 1.0 / 30.0  # Throttle UI signal emissions to 30 Hz for UI fluidity

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
                    self._motion_restore_freedrive = False  # User manually toggled freedrive
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

                    # Check if freedrive was active (or already marked for restoration)
                    was_freedrive = self._freedrive_active or self._motion_restore_freedrive
                    if self._freedrive_active:
                        try:
                            if self._robot and hasattr(self._robot, "disable_freedrive"):
                                self._robot.disable_freedrive()
                            self._freedrive_active = False
                            self.freedrive_state_changed.emit(False)
                            print("[MOTION] Freedrive temporarily paused for commanded motion.")
                        except Exception as err:
                            print(f"[ERROR] Failed to disable Freedrive before motion: {err}")

                    self._motion_restore_freedrive = was_freedrive
                    self._motion_target_q = [float(target_pose[f"{j}.pos"]) for j in UR5E_JOINTS]
                    self._motion_in_progress = True
                    self._motion_start_time = time.perf_counter()
                    self._motion_last_progress_time = self._motion_start_time

                    # Timeout based on actual UR5e velocity (0.2 rad/s) with generous headroom
                    max_disp = max(abs(self._latest_q[j] - target_pose[f"{j}.pos"]) for j in UR5E_JOINTS)
                    self._motion_last_disp = max_disp
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

                try:
                    tcp_list = self._robot.rx.getActualTCPPose()
                    self._latest_tcp = [float(v) for v in tcp_list]
                except Exception:
                    pass

                # Monitor motion completion on physical arm
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
                        print(f"[MOTION] Target position reached in {elapsed:.2f}s.")
                        if self._motion_restore_freedrive:
                            self._motion_restore_freedrive = False
                            try:
                                if hasattr(self._robot, "enable_freedrive"):
                                    self._robot.enable_freedrive()
                                self._freedrive_active = True
                                self.freedrive_state_changed.emit(True)
                                print("[MOTION] Freedrive automatically restored after motion completion.")
                            except Exception as err:
                                print(f"[ERROR] Failed to restore Freedrive: {err}")
                    elif elapsed >= self._motion_timeout and time_since_progress >= 4.0:
                        # Only time out if total timeout expired AND no progress made for 4.0 seconds
                        self._motion_in_progress = False
                        self._motion_restore_freedrive = False
                        print(f"[WARNING] Motion wait timed out after {elapsed:.1f}s without progress (max joint err: {math.degrees(current_disp):.1f}°).")
            else:
                # Synthetic realistic UR5e telemetry
                sim_t += target_period
                if self._motion_in_progress:
                    # Smoothly step mock angles towards target_q at ~1.5 rad/s
                    max_err = 0.0
                    for idx in range(6):
                        err = self._motion_target_q[idx] - mock_angles[idx]
                        max_err = max(max_err, abs(err))
                        step = math.copysign(min(abs(err), 1.5 * target_period), err)
                        mock_angles[idx] += step
                    elapsed = time.perf_counter() - self._motion_start_time
                    if max_err < 0.005:
                        self._motion_in_progress = False
                        print(f"[MOTION] Mock target motion reached ({elapsed:.2f}s).")
                        if self._motion_restore_freedrive:
                            self._motion_restore_freedrive = False
                            self._freedrive_active = True
                            self.freedrive_state_changed.emit(True)
                            print("[MOTION] Freedrive automatically restored after motion completion.")
                    elif elapsed >= self._motion_timeout:
                        self._motion_in_progress = False
                        self._motion_restore_freedrive = False
                        print(f"[WARNING] Mock motion timed out after {elapsed:.1f}s.")
                elif self._freedrive_active:
                    # In freedrive, arm drifts or simulates compliance
                    noise = 0.002 * math.sin(sim_t * 5)
                    for i in range(6):
                        mock_angles[i] += 0.001 * math.sin(sim_t * (1 + i * 0.4))
                elif self._mock_cartesian_v and any(abs(x) > 1e-4 for x in self._mock_cartesian_v):
                    # Mock teleoperation velocity integration:
                    # [vx, vy, vz, w_base, w_wrist2, w_wrist3]
                    dt = target_period
                    vx, vy, vz, w_base, w_wrist2, w_wrist3 = self._mock_cartesian_v
                    theta_base = mock_angles[0]
                    # Deconstruct base-frame velocity into arm's reach and swing plane
                    vfwd = vx * math.cos(theta_base) + vy * math.sin(theta_base)
                    vlat = -vx * math.sin(theta_base) + vy * math.cos(theta_base)
                    mock_angles[0] += (vlat * 1.2 + w_base) * dt  # Base pan responds to swing & direct w_base
                    mock_angles[1] += vfwd * dt * 1.2             # Shoulder extends in arm's forward direction
                    mock_angles[2] -= vfwd * dt * 1.2             # Elbow compensates for reach
                    mock_angles[1] -= vz * dt * 1.2               # Shoulder moves vertical vz
                    mock_angles[4] += w_wrist2 * dt               # Wrist 2 rotates directly via shoulders
                    mock_angles[5] += w_wrist3 * dt               # Wrist 3 rotates directly via triggers
                for idx, j_name in enumerate(UR5E_JOINTS):
                    self._latest_q[j_name] = mock_angles[idx]

            sample_count += 1
            t_now = time.perf_counter()
            read_latency_ms = (t_now - t0) * 1000.0

            # Frequency benchmark calculation every 0.5s
            if t_now - perf_window_start >= 0.5:
                measured_rate = sample_count / (t_now - perf_window_start)
                sample_count = 0
                perf_window_start = t_now

            # Throttle UI update emission to 30 Hz to keep Qt queue light
            if t_now - last_emit_time >= emit_interval:
                telemetry_payload = {
                    "joints": dict(self._latest_q),
                    "rtde_hz": measured_rate,
                    "latency_ms": read_latency_ms,
                    "freedrive": self._freedrive_active,
                }
                self.telemetry_signal.emit(telemetry_payload)
                last_emit_time = t_now

            # Sleep to match 125 Hz cycle
            elapsed = time.perf_counter() - t0
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Cleanup robot connection
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

    def get_latest_tcp(self) -> List[float]:
        return list(self._latest_tcp)

    def stop(self):
        self._running = False
        self.request_disconnect()
        self.wait(2000)


# =============================================================================
# 4. Hardware Verification Diagnostic Worker (test_lerobot_ur5e.py)
# =============================================================================

class HardwareTestWorker(QThread):
    """
    Background worker executing the 5-phase test_lerobot_ur5e.py suite.
    Captures live output and streams each line directly to the terminal console.
    """
    output_line = pyqtSignal(str)
    test_finished = pyqtSignal(bool, str)

    def __init__(self, script_path: Path, robot_ip: str, parent=None):
        super().__init__(parent)
        self.script_path = script_path
        self.robot_ip = robot_ip
        self._process = None

    def run(self):
        self.output_line.emit("\n" + "=" * 70)
        self.output_line.emit(">>> INITIATING UR5e HARDWARE DIAGNOSTICS SUITE <<<")
        self.output_line.emit(f"Target: {self.script_path.name} | Robot IP: {self.robot_ip}")
        self.output_line.emit("=" * 70 + "\n")

        if not self.script_path.exists():
            msg = f"Diagnostic script not found at: {self.script_path}"
            self.output_line.emit(f"[ERROR] {msg}")
            self.test_finished.emit(False, msg)
            return

        cmd = [sys.executable, "-u", str(self.script_path)]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            for line in iter(self._process.stdout.readline, ""):
                cleaned = line.rstrip()
                if cleaned:
                    self.output_line.emit(cleaned)

            self._process.stdout.close()
            return_code = self._process.wait()

            if return_code == 0:
                self.output_line.emit("\n[SUCCESS] All diagnostic verification phases PASSED cleanly!\n")
                self.test_finished.emit(True, "Diagnostics passed successfully.")
            else:
                self.output_line.emit(f"\n[FAILURE] Diagnostic test exited with code {return_code}.\n")
                self.test_finished.emit(False, f"Exited with error code {return_code}")

        except Exception as e:
            self.output_line.emit(f"[ERROR] Subprocess execution error: {e}")
            self.test_finished.emit(False, str(e))

    def terminate_process(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self.wait(1000)


# =============================================================================
# 4b. Asynchronous LeRobot Dataset Episode Save Worker Thread
# =============================================================================

@dataclass
class EpisodeSaveTask:
    payload_file: Path
    payload_data: Optional[Dict[str, Any]]
    ep_idx: int
    num_frames: int
    duration_s: float
    task_str: str


class EpisodeSaveWorkerThread(QThread):
    """
    Dedicated background worker thread that serializes demonstration payloads
    and coordinates isolated subprocess video encoding via save_episode_worker.py.

    Key Benefits:
    1. Zero GUI freezing: encoding, PyAV video compression, and disk I/O run completely off the Qt main thread.
    2. Crash isolation: C-extension segmentation faults never terminate the main studio window.
    3. Sequential FIFO dataset writing: prevents concurrent file write collisions on Parquet / meta info.
    """
    save_progress = pyqtSignal(str)
    episode_saved = pyqtSignal(int, int, float, int, bool, str, str)  # ep_idx, num_frames, duration_s, total_eps, success, err_msg, payload_path

    def __init__(self, worker_script: Path, parent=None):
        super().__init__(parent)
        self.worker_script = Path(worker_script).resolve()
        self._queue: queue.Queue[Optional[EpisodeSaveTask]] = queue.Queue()
        self._stop_requested = False
        self._is_busy = False
        self._active_proc: Optional[subprocess.Popen] = None

    def enqueue_task(self, task: EpisodeSaveTask):
        self._queue.put(task)

    def is_busy(self) -> bool:
        return self._is_busy or not self._queue.empty()

    def wait_until_empty(self, timeout_s: float = 15.0):
        """Waits until all pending save tasks in the queue are completed."""
        t_start = time.time()
        while self.is_busy() and (time.time() - t_start < timeout_s):
            time.sleep(0.1)

    def stop(self):
        self._stop_requested = True
        self._queue.put(None)  # Sentinel to unblock queue.get
        if self._active_proc and self._active_proc.poll() is None:
            try:
                self._active_proc.terminate()
            except Exception:
                pass
        self.wait(3000)

    def run(self):
        while not self._stop_requested:
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if task is None or self._stop_requested:
                break

            self._is_busy = True
            ep_idx = task.ep_idx
            num_frames = task.num_frames
            duration_s = task.duration_s
            payload_file = task.payload_file

            try:
                self.save_progress.emit(f"Serializing Episode #{ep_idx:04d} payload...")
                if task.payload_data is not None:
                    payload_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(payload_file, "wb") as f:
                        pickle.dump(task.payload_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    task.payload_data = None  # Free RAM immediately

                if not self.worker_script.exists():
                    raise FileNotFoundError(f"Worker script not found: {self.worker_script}")

                env = os.environ.copy()
                conda_bin = str(Path(sys.executable).parent)
                env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"

                self.save_progress.emit(f"Encoding Episode #{ep_idx:04d} ({num_frames} frames)...")
                cmd = [sys.executable, "-u", str(self.worker_script), "--payload", str(payload_file)]

                self._active_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )

                stdout, stderr = self._active_proc.communicate(timeout=300)
                retcode = self._active_proc.returncode

                total_episodes = -1
                success = (retcode == 0)
                err_msg = ""

                if stdout:
                    for line in stdout.splitlines():
                        if "__WORKER_RESULT_JSON_START__" in line and "__WORKER_RESULT_JSON_END__" in line:
                            try:
                                json_str = line.split("__WORKER_RESULT_JSON_START__")[1].split("__WORKER_RESULT_JSON_END__")[0]
                                res_data = json.loads(json_str)
                                total_episodes = int(res_data.get("total_episodes", -1))
                            except Exception:
                                pass
                        else:
                            print(line)

                if not success:
                    err_msg = stderr or stdout or f"Worker exited with code {retcode}"
                    print(f"[WORKER ERROR] Exit code {retcode}:\n{err_msg}", file=sys.__stderr__)

                self.episode_saved.emit(
                    ep_idx,
                    num_frames,
                    duration_s,
                    total_episodes,
                    success,
                    err_msg,
                    str(payload_file),
                )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[SAVE WORKER THREAD ERROR] {e}\n{tb}", file=sys.__stderr__)
                self.episode_saved.emit(
                    ep_idx,
                    num_frames,
                    duration_s,
                    -1,
                    False,
                    str(e),
                    str(payload_file),
                )
            finally:
                self._active_proc = None
                self._is_busy = False
                self._queue.task_done()


# =============================================================================
# 5. Modular Camera Feed UI Widget
# =============================================================================

class CameraFeedWidget(QFrame):
    """
    Self-contained video feed widget with title bar, resolution/FPS badges,
    blinking recording indicator overlay, and smooth proportional scaling.
    Designed so secondary or wrist cameras can easily be added side-by-side.
    """
    def __init__(self, title: str, camera_id: int, parent=None):
        super().__init__(parent)
        self.title = title
        self.camera_id = camera_id
        self.is_recording = False
        self._fps = 0.0

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            CameraFeedWidget {
                background-color: #0b0f19;
                border: 2px solid #1e293b;
                border-radius: 8px;
            }
        """)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Header bar
        header = QHBoxLayout()
        self.lbl_title = QLabel(f"<b>{self.title}</b>")
        self.lbl_title.setStyleSheet("color: #94a3b8; font-size: 11px;")
        header.addWidget(self.lbl_title)

        header.addStretch()

        self.lbl_mode_badge = QLabel("RGB")
        self.lbl_mode_badge.setStyleSheet("""
            color: #a855f7;
            font-size: 10px;
            font-weight: bold;
            padding: 1px 5px;
            background-color: rgba(168, 85, 247, 0.15);
            border: 1px solid rgba(168, 85, 247, 0.3);
            border-radius: 3px;
        """)
        header.addWidget(self.lbl_mode_badge)

        self.lbl_fps = QLabel("0.0 FPS")
        self.lbl_fps.setStyleSheet("color: #38bdf8; font-family: monospace; font-size: 11px;")
        header.addWidget(self.lbl_fps)

        self.lbl_rec = QLabel("● REC")
        self.lbl_rec.setStyleSheet("""
            color: #ef4444;
            font-weight: bold;
            font-size: 11px;
            padding: 1px 6px;
            background-color: rgba(239, 68, 68, 0.2);
            border-radius: 4px;
        """)
        self.lbl_rec.setVisible(False)
        header.addWidget(self.lbl_rec)

        layout.addLayout(header)

        # Image display canvas
        self.video_canvas = QLabel("Connecting Camera...")
        self.video_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_canvas.setStyleSheet("""
            background-color: #050811;
            color: #64748b;
            font-weight: bold;
            font-size: 12px;
            border-radius: 6px;
        """)
        self.video_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_canvas.setMinimumSize(200, 200)
        layout.addWidget(self.video_canvas)

        # Blinking timer for REC badge
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_rec_blink)
        self._blink_visible = True

    def set_recording(self, recording: bool):
        self.is_recording = recording
        if recording:
            self.lbl_rec.setVisible(True)
            self._blink_timer.start(500)
        else:
            self._blink_timer.stop()
            self.lbl_rec.setVisible(False)

    def _toggle_rec_blink(self):
        self._blink_visible = not self._blink_visible
        self.lbl_rec.setVisible(self._blink_visible)

    @pyqtSlot(int, float)
    def update_fps(self, cam_id: int, fps: float):
        if cam_id == self.camera_id:
            self._fps = fps
            self.lbl_fps.setText(f"{fps:4.1f} FPS")

    def set_grayscale(self, enabled: bool):
        if enabled:
            self.lbl_mode_badge.setText("GRAY")
            self.lbl_mode_badge.setStyleSheet("""
                color: #38bdf8;
                font-size: 10px;
                font-weight: bold;
                padding: 1px 5px;
                background-color: rgba(56, 189, 248, 0.15);
                border: 1px solid rgba(56, 189, 248, 0.3);
                border-radius: 3px;
            """)
        else:
            self.lbl_mode_badge.setText("RGB")
            self.lbl_mode_badge.setStyleSheet("""
                color: #a855f7;
                font-size: 10px;
                font-weight: bold;
                padding: 1px 5px;
                background-color: rgba(168, 85, 247, 0.15);
                border: 1px solid rgba(168, 85, 247, 0.3);
                border-radius: 3px;
            """)

    @pyqtSlot(int, np.ndarray)
    def update_frame(self, cam_id: int, frame_bgr: np.ndarray):
        if cam_id != self.camera_id:
            return

        if len(frame_bgr.shape) == 2:
            h, w = frame_bgr.shape
            bytes_per_line = w
            qimg = QImage(frame_bgr.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = frame_bgr.shape
            bytes_per_line = ch * w
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale to canvas dimensions preserving aspect ratio
        canvas_w = max(10, self.video_canvas.width())
        canvas_h = max(10, self.video_canvas.height())
        scaled_pixmap = QPixmap.fromImage(qimg).scaled(
            canvas_w,
            canvas_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_canvas.setPixmap(scaled_pixmap)


# =============================================================================
# 6. Main Teleoperation & Recording Studio Application
# =============================================================================

class UR5eRecorderStudio(QMainWindow):
    """
    Main Teleoperation & Recording Studio Window.
    Coordinates RTDE Worker, Camera Worker, LeRobotDataset writing,
    Diagnostic subprocesses, hotkey workflows, and GUI layout.
    """

    def __init__(self, robot_ip: str, camera_index: int, fps: int, camera_2_index: int = DEFAULT_CAMERA_2_INDEX, dual_cam: bool = False, mock: bool = False):
        super().__init__()
        self.robot_ip = robot_ip
        self.camera_index = camera_index
        self.camera_1_index = camera_index
        self.camera_2_index = camera_2_index
        self.enable_dual_cameras = False
        self.target_fps = fps
        self.mock_mode = mock
        self.is_grayscale = False

        # State tracking
        self.is_recording = False
        self.freedrive_active = False
        self.record_start_time: Optional[float] = None
        self.recorded_frames_count = 0
        self.active_dataset: Optional[Any] = None
        self.dataset_root: Path = DEFAULT_DATASETS_DIR / DEFAULT_DATASET_NAME
        self.current_episode_buffer: List[Dict[str, Any]] = []
        self.latest_camera_frame_1: Optional[np.ndarray] = None
        self.latest_camera_frame_2: Optional[np.ndarray] = None
        self.latest_camera_frame: Optional[np.ndarray] = None  # Alias for camera 1

        # Window settings
        self.setWindowTitle("UR5e LeRobot — Teleoperation & Demonstration Recording Studio")
        self.resize(1380, 980)
        self.setMinimumSize(1100, 800)
        self._user_dragging_joint: Optional[str] = None

        # Custom Start Position (defaults to DEFAULT_START_JOINTS_DICT: -80°, -45°, -100°, 150°, -85°, 60°)
        self.start_position: Dict[str, float] = dict(DEFAULT_START_JOINTS_DICT)
        self.start_gripper_mm: float = DEFAULT_START_GRIPPER_MM
        self.jog_sliders: Dict[str, QSlider] = {}
        self.jog_inputs: Dict[str, QLineEdit] = {}
        self.jog_move_buttons: Dict[str, QPushButton] = {}
        self.jog_labels: Dict[str, QLineEdit] = self.jog_inputs
        self._user_modified_joints: set[str] = set()
        self._user_targets: Dict[str, float] = {}

        self.camera_width = 320
        self.camera_height = 240

        # Hybrid Control, Reference Path Playback & Min-Jerk Recovery
        self.path_manager: Optional[PathManager] = PathManager(robot_ip=self.robot_ip, mock=self.mock_mode) if PathManager else None
        self.loaded_ref_path: Optional[Path] = None
        self.loaded_ref_waypoints: List[Any] = []
        self.override_delay_s: float = 0.50
        self.hybrid_worker: Optional[Any] = None

        # Apply dark industrial visual styling
        self.apply_theme()

        # Build UI layout
        self.setup_ui()

        # Setup Logging redirection with terminal console mirroring
        self.log_emitter = LogEmitter()
        self.log_emitter.log_signal.connect(self.append_log)
        self.stdout_redirect = StreamToLogger(self.log_emitter, "INFO", original_stream=sys.__stdout__)
        self.stderr_redirect = StreamToLogger(self.log_emitter, "ERROR", original_stream=sys.__stderr__)
        sys.stdout = self.stdout_redirect
        sys.stderr = self.stderr_redirect

        # Catch and print unhandled exceptions to terminal stderr
        def excepthook(exc_type, exc_value, exc_tb):
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.__stderr__)
            if hasattr(self, "log_emitter"):
                self.log_emitter.log_signal.emit("CRITICAL", f"Unhandled exception: {exc_value}")
        sys.excepthook = excepthook

        # Setup Hotkeys
        self.setup_shortcuts()

        # Start Primary Camera Thread
        self.camera_worker_1 = CameraWorker(
            self.camera_1_index, self.target_fps, self.camera_width, self.camera_height, self.mock_mode
        )
        self.camera_worker_1.frame_ready.connect(self.on_cam1_frame_received)
        self.camera_worker_1.fps_updated.connect(self.cam_widget_1.update_fps)
        self.camera_worker_1.status_signal.connect(self.on_camera_status)
        self.camera_worker_1.start()
        self.camera_worker = self.camera_worker_1  # Backwards compatibility alias

        # Secondary camera worker reference
        self.camera_worker_2: Optional[CameraWorker] = None

        if dual_cam:
            self.set_dual_cameras_enabled(True)

        # Start Robot RTDE Thread
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

        # Initialize Gamepad Teleoperation Worker
        self.gamepad_teleop_active = False
        if GAMEPAD_AVAILABLE and UR5eGamepadTeleop is not None:
            self.gamepad_teleop = UR5eGamepadTeleop(
                robot_ip=self.robot_ip,
                mock=self.mock_mode,
                parent=self,
            )
            if self.gripper is not None:
                self.gamepad_teleop.set_gripper(self.gripper)
                self.gamepad_teleop.gripper_width_updated.connect(self.on_gamepad_gripper_width_updated)
            self.gamepad_teleop.status_updated.connect(self.on_gamepad_status_updated)
            self.gamepad_teleop.velocity_updated.connect(self.on_gamepad_velocity_updated)
            self.gamepad_teleop.teleop_state_changed.connect(self.on_gamepad_state_changed)
            # Controller Button Shortcuts: Select -> Go to Start, Start -> Toggle Record (Space)
            self.gamepad_teleop.button_select_pressed.connect(self.on_gamepad_select_clicked)
            self.gamepad_teleop.button_start_pressed.connect(self.on_gamepad_start_clicked)
            # Provide live base motor angle (shoulder_pan) for heading-relative Cartesian control
            self.gamepad_teleop.set_base_angle_provider(
                lambda: self.robot_worker.get_latest_joints().get("shoulder_pan", 0.0)
            )
            # Provide live 6-DoF joint angles for kinematics and Wrist 3 decoupling
            self.gamepad_teleop.set_joint_positions_provider(
                lambda: self.robot_worker.get_latest_joints()
            )
            # Provide live 6-DoF TCP pose for Tool-Gaze direction control
            self.gamepad_teleop.set_tcp_pose_provider(
                lambda: self.robot_worker.get_latest_tcp()
            )
            if hasattr(self, "slider_gamepad_speed"):
                self.gamepad_teleop.set_speed_percentage(self.slider_gamepad_speed.value())
            if hasattr(self, "slider_gamepad_accel"):
                self.gamepad_teleop.set_accel_percentage(self.slider_gamepad_accel.value())
            if self.mock_mode:
                self.gamepad_teleop.set_velocity_callback(self.robot_worker.apply_cartesian_velocity)
            self.gamepad_teleop.start()
        else:
            self.gamepad_teleop = None

        # High-resolution UI Timer for recording stopwatch
        self.recording_timer = QTimer(self)
        self.recording_timer.setInterval(50)  # 20 Hz update rate for timer label
        self.recording_timer.timeout.connect(self.update_recording_timer)

        # Dataset Recording Synced Timer (30 Hz frame capture)
        self.dataset_frame_timer = QTimer(self)
        self.dataset_frame_timer.setInterval(int(1000 / self.target_fps))
        self.dataset_frame_timer.timeout.connect(self.capture_recording_frame)

        # Diagnostic Worker reference
        self.diagnostic_worker: Optional[HardwareTestWorker] = None

        # Asynchronous LeRobot Dataset Save Worker Thread
        worker_script = Path(__file__).parent / "save_episode_worker.py"
        self.episode_save_thread = EpisodeSaveWorkerThread(worker_script=worker_script, parent=self)
        self.episode_save_thread.save_progress.connect(self._on_save_progress)
        self.episode_save_thread.episode_saved.connect(self._on_episode_saved)
        self.episode_save_thread.start()

        # Scan initial dataset directory
        self.rescan_dataset_directory()

        # Scan available pre-recorded reference paths
        self.rescan_reference_paths()

        print(f"[SYSTEM] UR5e Studio initialized. Target IP: {self.robot_ip} | Camera: /dev/video{self.camera_index}")
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
        QLineEdit, QComboBox, QListWidget, QPlainTextEdit {
            background-color: #0b1120;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 6px;
            color: #f1f5f9;
            selection-background-color: #2563eb;
        }
        QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {
            border: 1px solid #38bdf8;
        }
        QPushButton {
            background-color: #334155;
            color: #ffffff;
            border: 1px solid #475569;
            border-radius: 6px;
            padding: 8px 14px;
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
            color: #94a3b8;
            font-size: 11px;
        }
        """
        self.setStyleSheet(palette_css)

    # -------------------------------------------------------------------------
    # UI Layout Construction
    # -------------------------------------------------------------------------
    def setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left/Center Column: Cameras & Recording Workflow
        left_column = QVBoxLayout()
        left_column.setSpacing(8)

        # 1. Camera Feeds Area (Modular, side-by-side expandable)
        self.camera_container = QGroupBox("VIDEO TELEMETRY VIEWPORT (EXPANDABLE MULTI-CAM)")
        cam_layout = QHBoxLayout(self.camera_container)
        cam_layout.setContentsMargins(8, 8, 8, 8)
        cam_layout.setSpacing(8)

        # Primary Camera Widget
        self.cam_widget_1 = CameraFeedWidget(f"Camera 1: Workspace (/dev/video{self.camera_index})", self.camera_index)
        cam_layout.addWidget(self.cam_widget_1, stretch=2)

        # Secondary Camera Slot (Easily toggleable or plugged in)
        self.cam_widget_2 = CameraFeedWidget(f"Camera 2: Wrist Flange (/dev/video{self.camera_2_index})", self.camera_2_index)
        self.cam_widget_2.setVisible(False)  # Hidden by default, toggled via button below
        cam_layout.addWidget(self.cam_widget_2, stretch=2)

        left_column.addWidget(self.camera_container, stretch=5)

        # Camera Options Bar
        cam_ctrl_bar = QHBoxLayout()
        self.btn_toggle_cam2 = QPushButton("+ Enable Secondary Camera")
        self.btn_toggle_cam2.setStyleSheet("font-size: 11px; padding: 4px 10px;")
        self.btn_toggle_cam2.setToolTip("Toggle secondary camera (Wrist Flange). When enabled, both cameras stream and record into the dataset.")
        self.btn_toggle_cam2.clicked.connect(self.toggle_secondary_camera)
        cam_ctrl_bar.addWidget(self.btn_toggle_cam2)

        # Discovered Cameras List
        available_cameras = detect_available_cameras()

        # Cam 1 Device Selector
        lbl_cam1 = QLabel("Cam 1:")
        lbl_cam1.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px; margin-left: 6px;")
        cam_ctrl_bar.addWidget(lbl_cam1)

        self.combo_cam1 = QComboBox()
        self.combo_cam1.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        self.combo_cam1.setToolTip("Select device for Primary Camera (Workspace)")
        for idx, label in available_cameras:
            self.combo_cam1.addItem(label, idx)
        # Select active camera_1_index
        idx_match = self.combo_cam1.findData(self.camera_1_index)
        if idx_match >= 0:
            self.combo_cam1.setCurrentIndex(idx_match)
        self.combo_cam1.currentIndexChanged.connect(self.on_cam1_device_changed)
        cam_ctrl_bar.addWidget(self.combo_cam1)

        # Cam 2 Device Selector (Visible when dual camera is active)
        self.lbl_cam2 = QLabel("Cam 2:")
        self.lbl_cam2.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px; margin-left: 6px;")
        self.lbl_cam2.setVisible(False)
        cam_ctrl_bar.addWidget(self.lbl_cam2)

        self.combo_cam2 = QComboBox()
        self.combo_cam2.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        self.combo_cam2.setToolTip("Select device for Secondary Camera (Wrist Flange)")
        for idx, label in available_cameras:
            self.combo_cam2.addItem(label, idx)
        idx2_match = self.combo_cam2.findData(self.camera_2_index)
        if idx2_match >= 0:
            self.combo_cam2.setCurrentIndex(idx2_match)
        self.combo_cam2.setVisible(False)
        self.combo_cam2.currentIndexChanged.connect(self.on_cam2_device_changed)
        cam_ctrl_bar.addWidget(self.combo_cam2)

        # Resolution Dropdown
        lbl_res = QLabel("Res:")
        lbl_res.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px; margin-left: 6px;")
        cam_ctrl_bar.addWidget(lbl_res)

        self.RESOLUTION_PRESETS = [
            # Standard Landscape (Width x Height)
            ("320 x 240 (Landscape 4:3 - Recommended Fast)", 320, 240),
            ("640 x 480 (Landscape 4:3 - VGA Standard)", 640, 480),
            ("480 x 360 (Landscape 4:3 - Medium)", 480, 360),
            ("160 x 120 (Landscape 4:3 - Ultra Fast)", 160, 120),
            ("640 x 360 (Landscape 16:9 - Widescreen nHD)", 640, 360),
            ("1280 x 720 (Landscape 16:9 - HD 720p)", 1280, 720),
            ("848 x 480 (Landscape 16:9 - Wide 480p)", 848, 480),
            # Square (1:1)
            ("256 x 256 (Square 1:1 - Policy/ACT Standard)", 256, 256),
            ("224 x 224 (Square 1:1 - ResNet/ViT Standard)", 224, 224),
            ("384 x 384 (Square 1:1 - High-Res Vision)", 384, 384),
            ("192 x 192 (Square 1:1 - Compact)", 192, 192),
            ("128 x 128 (Square 1:1 - Ultra Fast)", 128, 128),
            # Portrait / Vertical (Height > Width)
            ("240 x 320 (Portrait 3:4 - Fast Training)", 240, 320),
            ("360 x 480 (Portrait 3:4 - Medium)", 360, 480),
            ("480 x 640 (Portrait 3:4 - High-Res)", 480, 640),
            ("120 x 160 (Portrait 3:4 - Ultra Fast)", 120, 160),
            ("270 x 480 (Portrait 9:16 - Vertical Fast)", 270, 480),
            ("360 x 640 (Portrait 9:16 - Vertical nHD)", 360, 640),
            ("540 x 960 (Portrait 9:16 - Vertical qHD)", 540, 960),
            ("720 x 1280 (Portrait 9:16 - Vertical HD 720p)", 720, 1280),
        ]
        self.combo_resolution = QComboBox()
        for label, _, _ in self.RESOLUTION_PRESETS:
            self.combo_resolution.addItem(label)
        self.combo_resolution.setCurrentIndex(0)  # Default: 320x240
        self.combo_resolution.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        self.combo_resolution.setMinimumWidth(260)
        self.combo_resolution.setToolTip("Select recording resolution for cameras (Landscape, Square, or Portrait). Smaller resolutions train faster and save disk space.")
        self.combo_resolution.currentIndexChanged.connect(self.on_resolution_changed)
        cam_ctrl_bar.addWidget(self.combo_resolution)

        self.lbl_color_badge = QLabel("● RGB Color Mode")
        self.lbl_color_badge.setStyleSheet("color: #a855f7; font-weight: bold; font-size: 11px; margin-left: 6px; padding: 2px 4px;")
        self.lbl_color_badge.setToolTip("Datasets are collected in full RGB color. Grayscale filtering can be applied dynamically during training.")
        cam_ctrl_bar.addWidget(self.lbl_color_badge)

        lbl_fps_tag = QLabel("Rate:")
        lbl_fps_tag.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px; margin-left: 6px;")
        cam_ctrl_bar.addWidget(lbl_fps_tag)

        self.FPS_PRESETS = [30, 20, 15, 10]
        self.combo_fps = QComboBox()
        for f_val in self.FPS_PRESETS:
            self.combo_fps.addItem(f"{f_val} FPS", f_val)
        self.combo_fps.setCurrentIndex(0)
        self.combo_fps.setStyleSheet("font-size: 11px; padding: 2px 4px;")
        self.combo_fps.setToolTip("Observation and control rate (FPS). Standard for UR5e imitation learning is 30 FPS.")
        self.combo_fps.currentIndexChanged.connect(self.on_fps_changed)
        cam_ctrl_bar.addWidget(self.combo_fps)

        self.lbl_dual_badge = QLabel("○ 1-Cam Mode")
        self.lbl_dual_badge.setStyleSheet("color: #64748b; font-size: 10px; margin-left: 6px;")
        cam_ctrl_bar.addWidget(self.lbl_dual_badge)

        self.lbl_cam_info = QLabel("Format: 320x240 RGB @ 30 FPS")
        self.lbl_cam_info.setStyleSheet("color: #64748b; font-size: 10px; margin-left: 4px;")
        cam_ctrl_bar.addWidget(self.lbl_cam_info)

        self.btn_refresh_cams = QPushButton("🔄 Refresh (F5)")
        self.btn_refresh_cams.setStyleSheet("font-size: 11px; padding: 2px 8px; font-weight: bold;")
        self.btn_refresh_cams.setToolTip("Reconnect camera streams if preview freezes (Shortcut: F5)")
        self.btn_refresh_cams.clicked.connect(self.refresh_camera_streams)
        cam_ctrl_bar.addWidget(self.btn_refresh_cams)

        cam_ctrl_bar.addStretch()

        left_column.addLayout(cam_ctrl_bar)

        # 2. Recording & Freedrive Control Card
        rec_box = QGroupBox("EPISODE RECORDING CONTROLS")
        rec_layout = QVBoxLayout(rec_box)
        rec_layout.setContentsMargins(12, 12, 12, 12)
        rec_layout.setSpacing(10)

        # Timer & Status Row
        timer_row = QHBoxLayout()
        self.lbl_timer = QLabel("00:00.0")
        self.lbl_timer.setStyleSheet("""
            font-family: 'Courier New', monospace;
            font-size: 38px;
            font-weight: bold;
            color: #38bdf8;
            background-color: #0b1120;
            border: 2px solid #1e293b;
            border-radius: 8px;
            padding: 4px 18px;
        """)
        timer_row.addWidget(self.lbl_timer)

        timer_row.addSpacing(16)

        # Freedrive Status Indicator & Big Toggle Button
        fd_status_box = QVBoxLayout()
        self.lbl_fd_status = QLabel("ROBOT STATE: LOCKED (Position Hold)")
        self.lbl_fd_status.setStyleSheet("""
            color: #94a3b8;
            font-weight: bold;
            font-size: 12px;
            padding: 4px 8px;
            background-color: #1e293b;
            border-radius: 4px;
        """)
        fd_status_box.addWidget(self.lbl_fd_status)

        self.btn_toggle_freedrive = QPushButton("⚡ TOGGLE FREEDRIVE MODE")
        self.btn_toggle_freedrive.setMinimumHeight(38)
        self.btn_toggle_freedrive.setStyleSheet("""
            QPushButton {
                background-color: #475569;
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #64748b;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #64748b;
            }
        """)
        self.btn_toggle_freedrive.clicked.connect(self.on_freedrive_button_clicked)
        fd_status_box.addWidget(self.btn_toggle_freedrive)

        timer_row.addLayout(fd_status_box, stretch=1)
        rec_layout.addLayout(timer_row)

        # Gamepad Cartesian Teleoperation Control Strip
        gamepad_panel = QFrame()
        gamepad_panel.setStyleSheet("""
            QFrame {
                background-color: #0b1120;
                border: 1px solid #1e293b;
                border-radius: 6px;
                padding: 2px;
            }
        """)
        gamepad_main_layout = QVBoxLayout(gamepad_panel)
        gamepad_main_layout.setContentsMargins(8, 6, 8, 6)
        gamepad_main_layout.setSpacing(6)

        # Row 1: Primary Controls, Status HUD, and Inversion Toggles
        gamepad_row1 = QHBoxLayout()
        gamepad_row1.setSpacing(10)

        # Toggle Button: "Enable Gamepad Teleop"
        self.btn_toggle_gamepad = QPushButton("🎮 ENABLE GAMEPAD TELEOP")
        self.btn_toggle_gamepad.setCheckable(True)
        self.btn_toggle_gamepad.setMinimumHeight(34)
        self.btn_toggle_gamepad.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #94a3b8;
                font-size: 12px;
                font-weight: bold;
                border: 1px solid #334155;
                border-radius: 5px;
                padding: 4px 12px;
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
            "• Left Stick (XY): Tool-Gaze Forward/Backward (gripper pointing direction) & Strafe Left/Right\n"
            "• Right Stick (Y): TCP Vertical Up/Down (+Z/-Z)\n"
            "• Right Stick (X): Direct Base Motor Rotation (shoulder_pan)\n"
            "• Shoulders (LB/RB / L1/R1): Direct Wrist 2 Rotation\n"
            "• Triggers (LT/RT): Direct Wrist 3 Rotation\n"
            "• Select Button: Go to Start / Home Position\n"
            "• Start Button: Start / End Episode Recording (Space)\n"
            "• Note: Wrist 3 is strictly decoupled from analog stick movement\n"
            "• Neutral sticks/buttons: Smooth deceleration ramping (0 drift)"
        )
        self.btn_toggle_gamepad.toggled.connect(self.on_gamepad_toggle_clicked)
        gamepad_row1.addWidget(self.btn_toggle_gamepad)

        # Gamepad Connection Status Badge
        self.lbl_gamepad_badge = QLabel("🎮 No Gamepad Detected")
        self.lbl_gamepad_badge.setStyleSheet("""
            color: #64748b;
            font-size: 11px;
            font-weight: bold;
            padding: 4px 8px;
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 4px;
        """)
        gamepad_row1.addWidget(self.lbl_gamepad_badge)

        # Commanded TCP Velocity HUD Readout
        self.lbl_gamepad_vel = QLabel("TCP: 0.00 m/s")
        self.lbl_gamepad_vel.setStyleSheet("""
            color: #38bdf8;
            font-family: monospace;
            font-size: 11px;
            font-weight: bold;
            padding: 4px 8px;
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 4px;
        """)
        gamepad_row1.addWidget(self.lbl_gamepad_vel)

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
        gamepad_row1.addWidget(self.lbl_gamepad_gripper_hint)

        # Invert Left Stick Horizontal (Lateral Translation) Checkbox
        self.chk_invert_left_lr = QCheckBox("⇄ Invert Left Stick (L/R)")
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
        gamepad_row1.addWidget(self.chk_invert_left_lr)

        # Invert Right Stick Horizontal (Base Motor Rotation) Checkbox
        self.chk_invert_right_lr = QCheckBox("⇄ Invert Right Stick (Base)")
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
        gamepad_row1.addWidget(self.chk_invert_right_lr)

        # Backwards compatibility alias
        self.chk_invert_lr = self.chk_invert_left_lr
        gamepad_row1.addStretch()
        gamepad_main_layout.addLayout(gamepad_row1)

        # Row 2: Speed and Acceleration Sliders with Live Readouts and Controller Shortcuts Badge
        gamepad_row2 = QHBoxLayout()
        gamepad_row2.setSpacing(10)

        # Master Speed Slider (Default 50%)
        self.lbl_gamepad_speed = QLabel("⚡ Speed: 50%")
        self.lbl_gamepad_speed.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold;")
        self.slider_gamepad_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_gamepad_speed.setRange(10, 100)
        self.slider_gamepad_speed.setValue(50)
        self.slider_gamepad_speed.setFixedWidth(130)
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
        self.slider_gamepad_speed.setToolTip("Scale maximum gamepad teleoperation speed (10% to 100%). Default: 50%.")
        self.slider_gamepad_speed.valueChanged.connect(self.on_gamepad_speed_changed)
        gamepad_row2.addWidget(self.lbl_gamepad_speed)
        gamepad_row2.addWidget(self.slider_gamepad_speed)

        gamepad_row2.addSpacing(10)

        # Master Acceleration Slider (Default 50%)
        self.lbl_gamepad_accel = QLabel("🚀 Accel: 50%")
        self.lbl_gamepad_accel.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold;")
        self.slider_gamepad_accel = QSlider(Qt.Orientation.Horizontal)
        self.slider_gamepad_accel.setRange(10, 100)
        self.slider_gamepad_accel.setValue(50)
        self.slider_gamepad_accel.setFixedWidth(130)
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
        self.slider_gamepad_accel.setToolTip("Scale acceleration ramp rate (10% to 100%). Lower values prevent Wrist 1 protective stops. Default: 50%.")
        self.slider_gamepad_accel.valueChanged.connect(self.on_gamepad_accel_changed)
        gamepad_row2.addWidget(self.lbl_gamepad_accel)
        gamepad_row2.addWidget(self.slider_gamepad_accel)

        gamepad_row2.addSpacing(12)

        # Gamepad Button Shortcuts Indicator Badge
        lbl_gamepad_shortcuts = QLabel("🎮 Select: Go to Start | Start: Record (Space)")
        lbl_gamepad_shortcuts.setStyleSheet("""
            color: #e2e8f0;
            background-color: #1e293b;
            border: 1px solid #475569;
            border-radius: 4px;
            padding: 2px 7px;
            font-size: 10px;
            font-weight: 600;
        """)
        lbl_gamepad_shortcuts.setToolTip("Hardware Gamepad Shortcuts:\n• Select / Back: Moves arm to Start / Home pose\n• Start / Options: Toggles episode demonstration recording")
        gamepad_row2.addWidget(lbl_gamepad_shortcuts)

        lbl_accel_hint = QLabel("(Smooth ramping protects Wrist 1)")
        lbl_accel_hint.setStyleSheet("color: #64748b; font-size: 10px; font-style: italic;")
        gamepad_row2.addWidget(lbl_accel_hint)

        gamepad_row2.addStretch()
        gamepad_main_layout.addLayout(gamepad_row2)

        rec_layout.addWidget(gamepad_panel)

        # =====================================================================
        # Hybrid Assist & Pre-Recorded Reference Path Panel
        # =====================================================================
        hybrid_panel = QFrame()
        hybrid_panel.setStyleSheet("""
            QFrame {
                background-color: #0b1120;
                border: 1px solid #1e293b;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        hybrid_main_layout = QVBoxLayout(hybrid_panel)
        hybrid_main_layout.setContentsMargins(8, 6, 8, 6)
        hybrid_main_layout.setSpacing(6)

        # Row 1: Header, Path Selector Combo, Browse Button, Clear Button
        hybrid_row1 = QHBoxLayout()
        hybrid_row1.setSpacing(8)

        lbl_hybrid_title = QLabel("🤖 HYBRID CONTROL:")
        lbl_hybrid_title.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        lbl_hybrid_title.setToolTip("Execute pre-recorded golden trajectory with seamless real-time gamepad override and minimum-jerk snapback.")
        hybrid_row1.addWidget(lbl_hybrid_title)

        self.combo_ref_path = QComboBox()
        self.combo_ref_path.setMinimumHeight(28)
        self.combo_ref_path.setStyleSheet("""
            QComboBox {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
                min-width: 220px;
            }
            QComboBox:hover {
                border-color: #38bdf8;
            }
        """)
        self.combo_ref_path.setToolTip("Select pre-recorded trajectory from reference_paths/")
        self.combo_ref_path.currentIndexChanged.connect(self.on_ref_path_combo_changed)
        hybrid_row1.addWidget(self.combo_ref_path, stretch=1)

        self.btn_browse_ref_path = QPushButton("📂 Browse...")
        self.btn_browse_ref_path.setMinimumHeight(28)
        self.btn_browse_ref_path.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #cbd5e1;
                border: 1px solid #475569;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                padding: 2px 10px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #ffffff;
            }
        """)
        self.btn_browse_ref_path.setToolTip("Open pre-recorded reference path (.csv or .json) from disk")
        self.btn_browse_ref_path.clicked.connect(self.browse_reference_path)
        hybrid_row1.addWidget(self.btn_browse_ref_path)

        self.btn_clear_ref_path = QPushButton("✕ Clear")
        self.btn_clear_ref_path.setMinimumHeight(28)
        self.btn_clear_ref_path.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #94a3b8;
                border: 1px solid #334155;
                border-radius: 4px;
                font-size: 11px;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: #7f1d1d;
                color: #fca5a5;
            }
        """)
        self.btn_clear_ref_path.setToolTip("Clear loaded reference path to switch to manual demonstration recording")
        self.btn_clear_ref_path.clicked.connect(self.clear_reference_path)
        hybrid_row1.addWidget(self.btn_clear_ref_path)

        hybrid_main_layout.addLayout(hybrid_row1)

        # Row 2: Return Delay Dropdown, State Badge, Progress Readout
        hybrid_row2 = QHBoxLayout()
        hybrid_row2.setSpacing(10)

        lbl_delay = QLabel("⏱ Return Delay:")
        lbl_delay.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;")
        lbl_delay.setToolTip("Grace period: duration of zero gamepad input required before robot transitions to State 3 (Recovery) and snaps back to the path.")
        hybrid_row2.addWidget(lbl_delay)

        self.combo_override_delay = QComboBox()
        self.combo_override_delay.setMinimumHeight(26)
        self.combo_override_delay.setStyleSheet("""
            QComboBox {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 11px;
                min-width: 125px;
            }
        """)
        self.combo_override_delay.addItem("300 ms", 0.30)
        self.combo_override_delay.addItem("500 ms (Default)", 0.50)
        self.combo_override_delay.addItem("750 ms", 0.75)
        self.combo_override_delay.addItem("1000 ms (1.0 s)", 1.00)
        self.combo_override_delay.addItem("1500 ms (1.5 s)", 1.50)
        self.combo_override_delay.addItem("2000 ms (2.0 s)", 2.00)
        self.combo_override_delay.addItem("0 ms (Immediate)", 0.00)
        self.combo_override_delay.setCurrentIndex(1)  # 500 ms default
        self.combo_override_delay.currentIndexChanged.connect(self.on_override_delay_changed)
        hybrid_row2.addWidget(self.combo_override_delay)

        self.lbl_hybrid_state = QLabel("HYBRID: IDLE")
        self.lbl_hybrid_state.setStyleSheet("""
            color: #64748b;
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 4px;
            padding: 3px 8px;
            font-size: 11px;
            font-weight: bold;
        """)
        hybrid_row2.addWidget(self.lbl_hybrid_state)

        self.lbl_hybrid_progress = QLabel("Path: None (Manual Demo Mode)")
        self.lbl_hybrid_progress.setStyleSheet("""
            color: #94a3b8;
            font-family: monospace;
            font-size: 11px;
        """)
        hybrid_row2.addWidget(self.lbl_hybrid_progress, stretch=1)

        hybrid_main_layout.addLayout(hybrid_row2)
        rec_layout.addWidget(hybrid_panel)

        # Task Description Input
        task_row = QHBoxLayout()
        task_lbl = QLabel("Task Description:")
        task_lbl.setStyleSheet("color: #94a3b8; font-weight: bold;")
        self.txt_task_desc = QLineEdit("Pick and place demonstration")
        self.txt_task_desc.setPlaceholderText("e.g. Grasp red peg and insert into base fixture")
        task_row.addWidget(task_lbl)
        task_row.addWidget(self.txt_task_desc, stretch=1)
        rec_layout.addLayout(task_row)

        # Side-by-side Action Buttons (Start/End [Space], Cancel [c/Left], Stop [Esc])
        btn_action_row = QHBoxLayout()
        btn_action_row.setSpacing(10)

        # Start / End Episode Button (Space)
        self.btn_record = QPushButton("▶  START EPISODE (Space)")
        self.btn_record.setMinimumHeight(52)
        self.btn_record.setStyleSheet("""
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
        """)
        self.btn_record.clicked.connect(self.on_record_button_clicked)
        btn_action_row.addWidget(self.btn_record, stretch=3)

        # Cancel Recording Button (c or Left Arrow)
        self.btn_cancel = QPushButton("✖  CANCEL (c / ←)")
        self.btn_cancel.setMinimumHeight(52)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("""
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
        self.btn_cancel.clicked.connect(self.cancel_recording)
        btn_action_row.addWidget(self.btn_cancel, stretch=2)

        # Stop Session Button (Esc)
        self.btn_stop_session = QPushButton("⏹  STOP SESSION (Esc)")
        self.btn_stop_session.setMinimumHeight(52)
        self.btn_stop_session.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
                border-radius: 6px;
                border: 1px solid #475569;
            }
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        self.btn_stop_session.clicked.connect(self.stop_session)
        btn_action_row.addWidget(self.btn_stop_session, stretch=2)

        rec_layout.addLayout(btn_action_row)
        left_column.addWidget(rec_box, stretch=2)

        # 3. Live Log / Terminal Console Widget
        log_box = QGroupBox("TERMINAL LOG CONSOLE & SYSTEM NOTIFICATIONS")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.setSpacing(6)

        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("DejaVu Sans Mono", 9))
        self.txt_log.setStyleSheet("""
            background-color: #050811;
            color: #94a3b8;
            border: 1px solid #1e293b;
            border-radius: 4px;
        """)
        log_layout.addWidget(self.txt_log)

        log_btn_bar = QHBoxLayout()
        self.btn_clear_log = QPushButton("Clear Console")
        self.btn_clear_log.setStyleSheet("font-size: 10px; padding: 3px 8px;")
        self.btn_clear_log.clicked.connect(self.txt_log.clear)
        log_btn_bar.addWidget(self.btn_clear_log)
        log_btn_bar.addStretch()
        log_layout.addLayout(log_btn_bar)

        left_column.addWidget(log_box, stretch=2)

        main_layout.addLayout(left_column, stretch=7)

        # Right Column: Telemetry Gauges, Dataset Management & Diagnostics
        right_column = QVBoxLayout()
        right_column.setSpacing(8)

        # 4. Dataset & Episode Management Panel
        ds_box = QGroupBox("DATASET & EPISODE MANAGEMENT")
        ds_layout = QVBoxLayout(ds_box)
        ds_layout.setContentsMargins(10, 10, 10, 10)
        ds_layout.setSpacing(8)

        # Dataset Name Input
        ds_name_lbl = QLabel("Active Dataset Name:")
        ds_name_lbl.setStyleSheet("color: #94a3b8; font-weight: bold;")
        ds_layout.addWidget(ds_name_lbl)

        ds_name_row = QHBoxLayout()
        self.txt_dataset_name = QLineEdit()
        self.txt_dataset_name.setPlaceholderText("Enter new dataset name (e.g. cube_pick_place)...")
        self.txt_dataset_name.setText(DEFAULT_DATASET_NAME)
        self.txt_dataset_name.returnPressed.connect(self.create_new_dataset)
        self.txt_dataset_name.textChanged.connect(self.on_dataset_name_edited)
        ds_name_row.addWidget(self.txt_dataset_name, stretch=2)

        self.btn_new_ds = QPushButton("➕ Add New")
        self.btn_new_ds.setStyleSheet("""
            QPushButton {
                background-color: #047857;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
                padding: 5px 10px;
                border: 1px solid #059669;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        self.btn_new_ds.setToolTip("Create a new dataset with the name typed above (warns if name already exists)")
        self.btn_new_ds.clicked.connect(self.create_new_dataset)
        ds_name_row.addWidget(self.btn_new_ds)

        self.btn_load_ds = QPushButton("📁 Load Existing...")
        self.btn_load_ds.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
                padding: 5px 10px;
                border: 1px solid #3b82f6;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
        """)
        self.btn_load_ds.setToolTip("Open file dialog to load an existing dataset folder to add new episodes")
        self.btn_load_ds.clicked.connect(self.load_existing_dataset)
        ds_name_row.addWidget(self.btn_load_ds)

        self.btn_refresh_ds = QPushButton("↺")
        self.btn_refresh_ds.setToolTip("Rescan dataset directory")
        self.btn_refresh_ds.clicked.connect(self.rescan_dataset_directory)
        ds_name_row.addWidget(self.btn_refresh_ds)
        ds_layout.addLayout(ds_name_row)

        self.lbl_dataset_path = QLabel(f"Path: datasets/{DEFAULT_DATASET_NAME}")
        self.lbl_dataset_path.setStyleSheet("color: #64748b; font-size: 10px;")
        ds_layout.addWidget(self.lbl_dataset_path)

        # Episode History List
        ep_lbl = QLabel("Recorded Episodes in Active Dataset:")
        ep_lbl.setStyleSheet("color: #94a3b8; font-weight: bold; margin-top: 4px;")
        ds_layout.addWidget(ep_lbl)

        self.list_episodes = QListWidget()
        self.list_episodes.setStyleSheet("""
            QListWidget {
                background-color: #0b1120;
                border: 1px solid #1e293b;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #1e293b;
            }
            QListWidget::item:selected {
                background-color: #1e3a8a;
                color: #ffffff;
            }
        """)
        ds_layout.addWidget(self.list_episodes, stretch=2)

        # Delete Selected or Last Episode Button
        self.btn_delete_ep = QPushButton("🗑  Delete Selected / Last Episode")
        self.btn_delete_ep.setToolTip("Deletes the selected episode in the list above, or defaults to the last episode if none is selected.")
        self.btn_delete_ep.setStyleSheet("""
            QPushButton {
                background-color: #451a1a;
                color: #f87171;
                border: 1px solid #7f1d1d;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7f1d1d;
                color: #ffffff;
            }
        """)
        self.btn_delete_ep.clicked.connect(self.delete_episode)
        ds_layout.addWidget(self.btn_delete_ep)

        # 3D Visualizer Console Command Section
        viz_sep = QFrame()
        viz_sep.setFrameShape(QFrame.Shape.HLine)
        viz_sep.setStyleSheet("color: #334155; margin-top: 6px; margin-bottom: 2px;")
        ds_layout.addWidget(viz_sep)

        viz_title_lbl = QLabel("3D Visualizer Console Command:")
        viz_title_lbl.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        ds_layout.addWidget(viz_title_lbl)

        viz_cmd_row = QHBoxLayout()
        self.txt_viz_command = QLineEdit()
        self.txt_viz_command.setReadOnly(True)
        self.txt_viz_command.setFont(QFont("DejaVu Sans Mono", 9))
        self.txt_viz_command.setStyleSheet("""
            QLineEdit {
                background-color: #050811;
                color: #38bdf8;
                border: 1px solid #1e293b;
                border-radius: 4px;
                padding: 6px;
                selection-background-color: #1e3a8a;
            }
        """)
        viz_cmd_row.addWidget(self.txt_viz_command, stretch=4)

        self.btn_copy_viz_cmd = QPushButton("📋 Copy")
        self.btn_copy_viz_cmd.setToolTip("Copy visualizer command to clipboard")
        self.btn_copy_viz_cmd.setStyleSheet("""
            QPushButton {
                background-color: #1e3a8a;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
                padding: 6px 12px;
                border: 1px solid #3b82f6;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        self.btn_copy_viz_cmd.clicked.connect(self.copy_visualizer_command)
        viz_cmd_row.addWidget(self.btn_copy_viz_cmd, stretch=1)

        self.btn_launch_viz = QPushButton("🌐 Launch Viewer")
        self.btn_launch_viz.setToolTip("Launch the LeRobot Rerun visualizer directly for the selected episode")
        self.btn_launch_viz.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
                padding: 6px 12px;
                border: 1px solid #10b981;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #047857;
            }
        """)
        self.btn_launch_viz.clicked.connect(self.launch_visualizer)
        viz_cmd_row.addWidget(self.btn_launch_viz, stretch=1)
        ds_layout.addLayout(viz_cmd_row)

        self.list_episodes.currentRowChanged.connect(lambda _: self.update_visualizer_command())

        right_column.addWidget(ds_box, stretch=3)

        # 5. Live Telemetry HUD (Joints + RTDE Stats)
        telem_box = QGroupBox("ROBOT RTDE TELEMETRY HUD (125 Hz)")
        telem_layout = QVBoxLayout(telem_box)
        telem_layout.setContentsMargins(10, 10, 10, 10)
        telem_layout.setSpacing(6)

        # Rate & Latency Badges
        stats_row = QHBoxLayout()
        self.lbl_rtde_rate = QLabel("RTDE: 0.0 Hz")
        self.lbl_rtde_rate.setStyleSheet("color: #38bdf8; font-family: monospace; font-weight: bold;")
        stats_row.addWidget(self.lbl_rtde_rate)

        stats_row.addStretch()

        self.lbl_rtde_latency = QLabel("Latency: 0.0 ms")
        self.lbl_rtde_latency.setStyleSheet("color: #4ade80; font-family: monospace; font-weight: bold;")
        stats_row.addWidget(self.lbl_rtde_latency)
        telem_layout.addLayout(stats_row)

        # 6-DoF Joint Values with Mini Bars
        self.joint_labels: Dict[str, QLabel] = {}
        self.joint_bars: Dict[str, QProgressBar] = {}

        grid_joints = QGridLayout()
        grid_joints.setSpacing(4)

        for idx, j_name in enumerate(UR5E_JOINTS):
            name_lbl = QLabel(j_name.replace("_", " ").title())
            name_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 500;")

            val_lbl = QLabel("+0.0°")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setStyleSheet("color: #f1f5f9; font-family: monospace; font-weight: bold;")
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
        right_column.addWidget(telem_box, stretch=2)

        # 6. Manual Axis Control & Jogging Panel
        jog_box = QGroupBox("MANUAL AXIS CONTROL & JOGGING")
        self.jog_group = jog_box
        jog_layout = QVBoxLayout(jog_box)
        jog_layout.setContentsMargins(10, 10, 10, 10)
        jog_layout.setSpacing(6)

        # Status & Navigation Toolbar
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

        # Move to Target Button (moves all joints to entered values)
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

        # Sync to Robot Button (discards staged target angles and snaps inputs back to current robot position)
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

            # Direct Numeric Input Field
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

            # Adjacent Move to Target Button for this joint
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

            # Synchronize slider and numeric input
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
        self.gripper_slider.setValue(int(round(DEFAULT_START_GRIPPER_MM)))
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

        self.gripper_input = QLineEdit(f"{DEFAULT_START_GRIPPER_MM:.1f}")
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
        right_column.addWidget(jog_box, stretch=3)

        # 7. Hardware Diagnostics Panel
        diag_box = QGroupBox("HARDWARE DIAGNOSTICS & VERIFICATION")
        diag_layout = QVBoxLayout(diag_box)
        diag_layout.setContentsMargins(10, 10, 10, 10)
        diag_layout.setSpacing(8)

        diag_desc = QLabel("Executes the 5-phase LeRobot UR5e verification test (introspection, telemetry rate benchmark, +2° wrist nudge, +5° base motion).")
        diag_desc.setWordWrap(True)
        diag_desc.setStyleSheet("color: #64748b; font-size: 11px;")
        diag_layout.addWidget(diag_desc)

        self.btn_test_hardware = QPushButton("🔬 Run Full Hardware Test (test_lerobot_ur5e.py)")
        self.btn_test_hardware.setMinimumHeight(36)
        self.btn_test_hardware.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                border: 1px solid #60a5fa;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #1e293b;
                color: #64748b;
                border: 1px solid #334155;
            }
        """)
        self.btn_test_hardware.clicked.connect(self.run_hardware_test)
        diag_layout.addWidget(self.btn_test_hardware)

        right_column.addWidget(diag_box, stretch=2)

        main_layout.addLayout(right_column, stretch=3)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Press 'Space' to Start Episode or toggle Freedrive.")

    # -------------------------------------------------------------------------
    # Hotkeys Configuration
    # -------------------------------------------------------------------------
    def setup_shortcuts(self):
        # Space: Start / End Episode
        self.shortcut_space = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.shortcut_space.activated.connect(self.on_space_hotkey)

        # 'c': Cancel Episode
        self.shortcut_c = QShortcut(QKeySequence(Qt.Key.Key_C), self)
        self.shortcut_c.activated.connect(self.cancel_recording)

        # Left Arrow: Cancel Episode
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.shortcut_left.activated.connect(self.cancel_recording)

        # Esc: Stop Session
        self.shortcut_esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.shortcut_esc.activated.connect(self.stop_session)

        # F5: Refresh Camera Streams
        self.shortcut_f5 = QShortcut(QKeySequence(Qt.Key.Key_F5), self)
        self.shortcut_f5.activated.connect(self.refresh_camera_streams)

    # -------------------------------------------------------------------------
    # Camera Slots & Multiplexing
    # -------------------------------------------------------------------------
    def toggle_secondary_camera(self):
        self.set_dual_cameras_enabled(not self.enable_dual_cameras)

    def set_dual_cameras_enabled(self, enabled: bool):
        self.enable_dual_cameras = enabled
        self.cam_widget_2.setVisible(enabled)
        if hasattr(self, "lbl_cam2"):
            self.lbl_cam2.setVisible(enabled)
        if hasattr(self, "combo_cam2"):
            self.combo_cam2.setVisible(enabled)

        if enabled:
            self.btn_toggle_cam2.setText("- Disable Secondary Camera")
            self.btn_toggle_cam2.setStyleSheet("""
                QPushButton {
                    background-color: #1e3a8a;
                    color: #93c5fd;
                    border: 1px solid #3b82f6;
                    font-size: 11px;
                    padding: 4px 10px;
                }
            """)
            if hasattr(self, "lbl_dual_badge"):
                self.lbl_dual_badge.setText("● 2-Cam Mode (Workspace+Wrist)")
                self.lbl_dual_badge.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 10px; margin-left: 6px;")

            # Start worker 2 if not active
            if self.camera_worker_2 is None:
                print(f"[CAMERA 2] Initializing Secondary Camera on /dev/video{self.camera_2_index}...")
                self.camera_worker_2 = CameraWorker(
                    self.camera_2_index, self.target_fps, self.camera_width, self.camera_height, self.mock_mode, self.is_grayscale
                )
                self.camera_worker_2.frame_ready.connect(self.on_cam2_frame_received)
                self.camera_worker_2.fps_updated.connect(self.cam_widget_2.update_fps)
                self.camera_worker_2.status_signal.connect(self.on_camera_status)
                self.camera_worker_2.start()
        else:
            self.btn_toggle_cam2.setText("+ Enable Secondary Camera")
            self.btn_toggle_cam2.setStyleSheet("""
                QPushButton {
                    background-color: #334155;
                    color: #ffffff;
                    border: 1px solid #475569;
                    font-size: 11px;
                    padding: 4px 10px;
                }
            """)
            if hasattr(self, "lbl_dual_badge"):
                self.lbl_dual_badge.setText("○ 1-Cam Mode")
                self.lbl_dual_badge.setStyleSheet("color: #64748b; font-size: 10px; margin-left: 6px;")

            if self.camera_worker_2 is not None:
                print(f"[CAMERA 2] Stopping Secondary Camera (/dev/video{self.camera_2_index})...")
                self.camera_worker_2.stop()
                self.camera_worker_2 = None
                self.latest_camera_frame_2 = None

        # Invalidate active dataset handle so next episode writes correct schema (1 or 2 cameras)
        self.active_dataset = None

    def on_grayscale_toggled(self, checked: bool):
        self.is_grayscale = checked
        if self.camera_worker_1 is not None:
            self.camera_worker_1.set_grayscale(checked)
        if self.camera_worker_2 is not None:
            self.camera_worker_2.set_grayscale(checked)
        if hasattr(self, "cam_widget_1"):
            self.cam_widget_1.set_grayscale(checked)
        if hasattr(self, "cam_widget_2"):
            self.cam_widget_2.set_grayscale(checked)
        ch_desc = "GRAY (1-Ch)" if checked else "RGB (3-Ch)"
        self.lbl_cam_info.setText(f"Format: {self.camera_width}x{self.camera_height} {ch_desc} @ {self.target_fps} FPS")
        print(f"[CAMERA] Vision mode switched to: {ch_desc}")
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(f"Camera mode set to {ch_desc}.", 3000)
        # Invalidate active dataset handle so any new recording starts with matching format
        self.active_dataset = None

    def on_fps_changed(self, index: int):
        val = self.combo_fps.itemData(index)
        if val is not None and val != self.target_fps:
            self.target_fps = int(val)
            if self.camera_worker_1 is not None:
                self.camera_worker_1.target_fps = self.target_fps
            if self.camera_worker_2 is not None:
                self.camera_worker_2.target_fps = self.target_fps
            interval_ms = int(1000.0 / max(1, self.target_fps))
            if hasattr(self, "dataset_frame_timer"):
                self.dataset_frame_timer.setInterval(interval_ms)
            ch_desc = "GRAY (1-Ch)" if self.is_grayscale else "RGB (3-Ch)"
            self.lbl_cam_info.setText(f"Format: {self.camera_width}x{self.camera_height} {ch_desc} @ {self.target_fps} FPS")
            print(f"[DATASET] Recording sampling rate updated to {self.target_fps} FPS (dt = {1.0/self.target_fps:.3f}s)")
            self.active_dataset = None

    def on_cam1_device_changed(self, index: int):
        cam_id = self.combo_cam1.itemData(index)
        if cam_id is not None and cam_id != self.camera_1_index:
            print(f"[CAMERA 1] Switching Workspace camera to /dev/video{cam_id}...")
            self.camera_1_index = cam_id
            self.camera_index = cam_id
            if self.camera_worker_1 is not None:
                self.camera_worker_1.stop()
                self.camera_worker_1 = CameraWorker(
                    self.camera_1_index, self.target_fps, self.camera_width, self.camera_height, self.mock_mode, self.is_grayscale
                )
                self.camera_worker_1.frame_ready.connect(self.on_cam1_frame_received)
                self.camera_worker_1.fps_updated.connect(self.cam_widget_1.update_fps)
                self.camera_worker_1.status_signal.connect(self.on_camera_status)
                self.camera_worker_1.start()
                self.camera_worker = self.camera_worker_1
            self.cam_widget_1.lbl_title.setText(f"Camera 1: Workspace (/dev/video{self.camera_1_index})")

    def on_cam2_device_changed(self, index: int):
        cam_id = self.combo_cam2.itemData(index)
        if cam_id is not None and cam_id != self.camera_2_index:
            print(f"[CAMERA 2] Switching Wrist camera to /dev/video{cam_id}...")
            self.camera_2_index = cam_id
            if self.camera_worker_2 is not None:
                self.camera_worker_2.stop()
                self.camera_worker_2 = CameraWorker(
                    self.camera_2_index, self.target_fps, self.camera_width, self.camera_height, self.mock_mode, self.is_grayscale
                )
                self.camera_worker_2.frame_ready.connect(self.on_cam2_frame_received)
                self.camera_worker_2.fps_updated.connect(self.cam_widget_2.update_fps)
                self.camera_worker_2.status_signal.connect(self.on_camera_status)
                self.camera_worker_2.start()
            self.cam_widget_2.lbl_title.setText(f"Camera 2: Wrist Flange (/dev/video{self.camera_2_index})")

    def on_resolution_changed(self, index: int):
        if index < 0 or index >= len(self.RESOLUTION_PRESETS):
            return
        desc, w, h = self.RESOLUTION_PRESETS[index]
        self.camera_width = w
        self.camera_height = h
        if self.camera_worker_1 is not None:
            self.camera_worker_1.set_resolution(w, h)
        if self.camera_worker_2 is not None:
            self.camera_worker_2.set_resolution(w, h)
        ch_desc = "GRAY (1-Ch)" if self.is_grayscale else "RGB (3-Ch)"
        self.lbl_cam_info.setText(f"Format: {w}x{h} {ch_desc} @ {self.target_fps} FPS")
        print(f"[CAMERA] Resolution updated to {w}x{h} ({desc})")
        self.status_bar.showMessage(f"Camera resolution set to {w}x{h}.", 3000)
        # Invalidate active dataset handle so any new recording starts with the chosen resolution
        self.active_dataset = None

    def _sync_resolution_ui(self, w: int, h: int):
        self.camera_width = w
        self.camera_height = h
        if self.camera_worker_1 is not None:
            self.camera_worker_1.set_resolution(w, h)
        if self.camera_worker_2 is not None:
            self.camera_worker_2.set_resolution(w, h)
        ch_desc = "GRAY (1-Ch)" if self.is_grayscale else "RGB (3-Ch)"
        self.lbl_cam_info.setText(f"Format: {w}x{h} {ch_desc} @ {self.target_fps} FPS")
        if hasattr(self, "combo_resolution"):
            self.combo_resolution.blockSignals(True)
            found_idx = -1
            for idx, (_, ow, oh) in enumerate(self.RESOLUTION_PRESETS):
                if ow == w and oh == h:
                    found_idx = idx
                    break
            if found_idx >= 0:
                self.combo_resolution.setCurrentIndex(found_idx)
            else:
                custom_label = f"{w} x {h} (Dataset Configured)"
                self.combo_resolution.addItem(custom_label)
                self.combo_resolution.setCurrentIndex(self.combo_resolution.count() - 1)
            self.combo_resolution.blockSignals(False)

    def refresh_camera_streams(self):
        """Flushes and reconnects active camera capture workers to clear any V4L2 queue stall."""
        print("[CAMERA] Refreshing camera streams...")
        if hasattr(self, "camera_worker_1") and self.camera_worker_1:
            self.camera_worker_1.request_reconnect()
        if hasattr(self, "camera_worker_2") and self.camera_worker_2:
            self.camera_worker_2.request_reconnect()
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage("Camera streams refreshed.", 3000)

    def update_visualizer_command(self):
        clean_name = self.dataset_root.name
        selected_idx = 0
        curr_item = self.list_episodes.currentItem()
        if curr_item is not None:
            text = curr_item.text()
            import re
            m = re.search(r"Episode #(\d+)", text)
            if m:
                selected_idx = int(m.group(1))
            else:
                row = self.list_episodes.currentRow()
                if row >= 0:
                    selected_idx = row
        abs_root = self.dataset_root.resolve()
        cmd = f"lerobot-dataset-viz --repo-id local/{clean_name} --root {abs_root} --episode-index {selected_idx}"
        if hasattr(self, "txt_viz_command"):
            self.txt_viz_command.setText(cmd)

    def launch_visualizer(self):
        clean_name = self.dataset_root.name
        abs_root = self.dataset_root.resolve()
        selected_idx = 0
        curr_item = self.list_episodes.currentItem()
        if curr_item is not None:
            text = curr_item.text()
            import re
            m = re.search(r"Episode #(\d+)", text)
            if m:
                selected_idx = int(m.group(1))
            else:
                row = self.list_episodes.currentRow()
                if row >= 0:
                    selected_idx = row

        info_path = abs_root / "meta" / "info.json"
        if not info_path.exists():
            QMessageBox.warning(
                self,
                "Dataset Not Found",
                f"No episode metadata found at:\n{info_path}\n\nPlease record and save at least one episode first.",
            )
            return

        env = os.environ.copy()
        conda_bin = str(Path(sys.executable).parent)
        env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"

        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_dataset_viz",
            "--repo-id",
            f"local/{clean_name}",
            "--root",
            str(abs_root),
            "--episode-index",
            str(selected_idx),
        ]
        try:
            subprocess.Popen(cmd, env=env)
            self.status_bar.showMessage(f"Opening LeRobot Visualizer for Episode #{selected_idx}...", 5000)
            print(f"[SYSTEM] Launched visualizer: {' '.join(cmd)}")
        except Exception as e:
            QMessageBox.critical(self, "Visualizer Error", f"Failed to launch visualizer:\n{e}")

    def copy_visualizer_command(self):
        cmd = self.txt_viz_command.text().strip()
        if not cmd:
            return
        QApplication.clipboard().setText(cmd)
        self.btn_copy_viz_cmd.setText("✔ Copied!")
        self.btn_copy_viz_cmd.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
                padding: 6px 12px;
                border: 1px solid #22c55e;
                border-radius: 4px;
            }
        """)
        QTimer.singleShot(2000, self._reset_copy_btn)
        self.status_bar.showMessage("Visualizer command copied to clipboard!", 3500)
        print(f"[SYSTEM] Copied to clipboard: {cmd}")

    def _reset_copy_btn(self):
        self.btn_copy_viz_cmd.setText("📋 Copy")
        self.btn_copy_viz_cmd.setStyleSheet("""
            QPushButton {
                background-color: #1e3a8a;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
                padding: 6px 12px;
                border: 1px solid #3b82f6;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)

    @pyqtSlot(int, np.ndarray)
    def on_cam1_frame_received(self, cam_id: int, frame: np.ndarray):
        self.latest_camera_frame_1 = frame
        self.latest_camera_frame = frame
        self.cam_widget_1.update_frame(cam_id, frame)

    @pyqtSlot(int, np.ndarray)
    def on_cam2_frame_received(self, cam_id: int, frame: np.ndarray):
        self.latest_camera_frame_2 = frame
        self.cam_widget_2.update_frame(cam_id, frame)

    @pyqtSlot(int, np.ndarray)
    def on_frame_received(self, cam_id: int, frame: np.ndarray):
        # Backwards compatibility slot
        self.on_cam1_frame_received(cam_id, frame)

    @pyqtSlot(int, bool, str)
    def on_camera_status(self, cam_id: int, is_connected: bool, message: str):
        level = "INFO" if is_connected else "WARNING"
        print(f"[{level}] [CAMERA {cam_id}] {message}")

    # -------------------------------------------------------------------------
    # Robot RTDE Telemetry Slots
    # -------------------------------------------------------------------------
    @pyqtSlot(bool, str)
    def on_robot_connection_status(self, is_connected: bool, message: str):
        level = "INFO" if is_connected else "ERROR"
        print(f"[{level}] [UR5e] {message}")
        self.status_bar.showMessage(message, 6000)
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
        # Keep jog controls accessible; commanded motion automatically interrupts and restores freedrive
        if active:
            self.lbl_fd_status.setText("ROBOT STATE: FREEDRIVE ACTIVE (Hand-Guiding)")
            self.lbl_fd_status.setStyleSheet("""
                color: #ffffff;
                font-weight: bold;
                font-size: 12px;
                padding: 4px 8px;
                background-color: #d97706;
                border-radius: 4px;
            """)
            self.btn_toggle_freedrive.setText("🔒 LOCK ROBOT JOINTS")
            self.btn_toggle_freedrive.setStyleSheet("""
                QPushButton {
                    background-color: #d97706;
                    color: #ffffff;
                    font-size: 13px;
                    font-weight: bold;
                    border: 2px solid #f59e0b;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #b45309;
                }
            """)
            if hasattr(self, "lbl_jog_notice"):
                self.lbl_jog_notice.setText("Freedrive ACTIVE — Commanded motion will pause & restore freedrive")
                self.lbl_jog_notice.setStyleSheet("color: #f59e0b; font-size: 10px; font-weight: bold;")
            print("[INFO] Freedrive ENABLED — Robot arm is now compliant for hand demonstration.")

            # Safety Interlock: Inhibit gamepad teleoperation while Freedrive is physically active
            if hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled():
                self.gamepad_teleop.set_inhibited(True, "Freedrive active")
                if hasattr(self, "lbl_gamepad_vel"):
                    self.lbl_gamepad_vel.setText("PAUSED (Freedrive)")
                print("[GAMEPAD] Gamepad teleoperation paused: Freedrive is active.")
        else:
            self.lbl_fd_status.setText("ROBOT STATE: LOCKED (Position Hold)")
            self.lbl_fd_status.setStyleSheet("""
                color: #94a3b8;
                font-weight: bold;
                font-size: 12px;
                padding: 4px 8px;
                background-color: #1e293b;
                border-radius: 4px;
            """)
            self.btn_toggle_freedrive.setText("⚡ TOGGLE FREEDRIVE MODE")
            self.btn_toggle_freedrive.setStyleSheet("""
                QPushButton {
                    background-color: #475569;
                    color: #ffffff;
                    font-size: 13px;
                    font-weight: bold;
                    border: 2px solid #64748b;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #64748b;
                }
            """)
            if hasattr(self, "lbl_jog_notice"):
                self.lbl_jog_notice.setText("Freedrive LOCKED — Manual jogging active")
                self.lbl_jog_notice.setStyleSheet("color: #4ade80; font-size: 10px; font-weight: bold;")
            print("[INFO] Freedrive DISABLED — Robot arm is re-locked in place.")

            # Safety Interlock: Resume gamepad teleoperation if it was enabled
            if hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled():
                self.gamepad_teleop.set_inhibited(False)
                if hasattr(self, "lbl_gamepad_vel"):
                    self.lbl_gamepad_vel.setText("TCP: 0.00 m/s")
                print("[GAMEPAD] Gamepad teleoperation resumed.")

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

        # Safety Interlock: Check motion completion to un-inhibit gamepad teleoperation
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            if not self.robot_worker.is_motion_in_progress and not self.freedrive_active:
                if self.gamepad_teleop.is_inhibited() and "Trajectory" in self.gamepad_teleop.get_inhibition_reason():
                    self.gamepad_teleop.set_inhibited(False)
                    if hasattr(self, "lbl_gamepad_vel"):
                        self.lbl_gamepad_vel.setText("TCP: 0.00 m/s")

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
                background-color: #0f172a;
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
                    font-size: 12px;
                    font-weight: bold;
                    border: 2px solid #38bdf8;
                    border-radius: 5px;
                    padding: 4px 12px;
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
                    font-size: 12px;
                    font-weight: bold;
                    border: 1px solid #334155;
                    border-radius: 5px;
                    padding: 4px 12px;
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
                background-color: #0f172a;
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
        if hasattr(self, "is_recording") and self.is_recording:
            print("[GAMEPAD] Cannot move to Start position while demonstration recording is active.")
            self.status_bar.showMessage("Cannot Go to Start while recording.", 3000)
            return
        print("[GAMEPAD BUTTON] 'Select' pressed -> Triggering Go to Start Pose.")
        if hasattr(self, "btn_go_start"):
            self.btn_go_start.click()
        else:
            self.move_to_start_pose()

    @pyqtSlot()
    def on_gamepad_start_clicked(self):
        """Gamepad Start button pressed -> Toggle Start/End Episode Recording (same as Space)."""
        print("[GAMEPAD BUTTON] 'Start' pressed -> Toggling Episode Recording.")
        if hasattr(self, "btn_record"):
            self.btn_record.click()
        else:
            self.on_record_button_clicked()

    # -------------------------------------------------------------------------
    # Manual Jogging, Start Pose & Home Pose Slots
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
        gripper_w = float(self.gripper_slider.value()) if hasattr(self, "gripper_slider") else DEFAULT_START_GRIPPER_MM
        if persist_start_position is not None:
            persist_start_position(self.start_position, gripper_w)
            print("[CONFIG] Persisted new start position to default_start_position.json")
        self.update_start_pose_display()
        pose_str = ", ".join(f"{self.start_position[j]:+.1f}°" for j in UR5E_JOINTS)
        print(f"[POSITION] Active Start Position set to: [{pose_str}]")
        self.status_bar.showMessage(f"Active Start Position saved: [{pose_str}]", 4000)

    def move_to_start_pose(self):
        # Inhibit gamepad teleoperation during discrete motion
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
        # Inhibit gamepad teleoperation during discrete motion
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.gamepad_teleop.set_inhibited(True, "Trajectory motion in progress")
        self._user_modified_joints.clear()
        self._user_targets.clear()
        home_action = {f"{j}.pos": math.radians(HOME_JOINTS_DICT[j]) for j in UR5E_JOINTS}
        print("[MOTION] Commanding UR5e arm to Home position (0°, -90°, 90°, -90°, -90°, 0°)...")
        self.status_bar.showMessage("Moving arm to Home position...", 4000)
        self.robot_worker.request_move_pose(home_action)

    def get_numeric_input_value(self, j_name: str) -> float:
        # 1. First priority: if user explicitly entered a target in _user_targets
        if j_name in self._user_targets:
            return self._user_targets[j_name]
        # 2. Second: read text currently in the input box
        if hasattr(self, "jog_inputs") and j_name in self.jog_inputs:
            text = self.jog_inputs[j_name].text().replace("°", "").strip()
            try:
                val = float(text)
                return max(-360.0, min(360.0, val))
            except ValueError:
                pass
        # 3. Third: slider position
        if hasattr(self, "jog_sliders") and j_name in self.jog_sliders:
            return float(self.jog_sliders[j_name].value())
        return 0.0

    def move_to_all_targets(self):
        # Inhibit gamepad teleoperation during discrete motion
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.gamepad_teleop.set_inhibited(True, "Trajectory motion in progress")

        # 1. Harvest all typed texts into _user_targets first
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

        # Clear staged target flags once motion is dispatched
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
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.gamepad_teleop.set_inhibited(True, "Trajectory motion in progress")
        if not hasattr(self, "jog_sliders") or j_name not in self.jog_sliders:
            self._user_dragging_joint = None
            return

        target_deg = float(self.jog_sliders[j_name].value())
        self._user_targets.pop(j_name, None)
        self._user_modified_joints.discard(j_name)
        target_rad = math.radians(target_deg)

        if hasattr(self, "jog_inputs") and j_name in self.jog_inputs:
            self.jog_inputs[j_name].setText(f"{target_deg:+.1f}")

        # Build complete action dictionary preserving other current joint positions
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
        # Ensure any text currently typed in the box is captured into _user_targets
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
    # Hybrid Control, Reference Path Management & Min-Jerk Snapback
    # -------------------------------------------------------------------------
    def rescan_reference_paths(self):
        """Scans the reference_paths/ directory and populates the combo box."""
        if not hasattr(self, "combo_ref_path"):
            return
        self.combo_ref_path.blockSignals(True)
        self.combo_ref_path.clear()
        self.combo_ref_path.addItem("None (Manual Recording Mode)", None)

        paths_dir = Path(__file__).resolve().parent / "reference_paths"
        if paths_dir.is_dir():
            csv_files = sorted(list(paths_dir.glob("*.csv")) + list(paths_dir.glob("*.json")), reverse=True)
            for f in csv_files:
                self.combo_ref_path.addItem(f"📁 {f.name}", str(f))

        # Reselect loaded path if still valid
        if self.loaded_ref_path and self.loaded_ref_path.is_file():
            for idx in range(self.combo_ref_path.count()):
                if self.combo_ref_path.itemData(idx) == str(self.loaded_ref_path):
                    self.combo_ref_path.setCurrentIndex(idx)
                    break
        else:
            self.combo_ref_path.setCurrentIndex(0)
        self.combo_ref_path.blockSignals(False)

    def on_ref_path_combo_changed(self, index: int):
        filepath_str = self.combo_ref_path.currentData()
        if not filepath_str:
            self.clear_reference_path()
        else:
            self.load_reference_path(Path(filepath_str))

    def browse_reference_path(self):
        default_dir = str(Path(__file__).resolve().parent / "reference_paths")
        filepath_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pre-Recorded Reference Path",
            default_dir,
            "Trajectory Files (*.csv *.json);;CSV Files (*.csv);;JSON Files (*.json);;All Files (*)",
        )
        if filepath_str:
            target_path = Path(filepath_str)
            self.load_reference_path(target_path)
            self.rescan_reference_paths()
            # Select it in combo
            for idx in range(self.combo_ref_path.count()):
                if self.combo_ref_path.itemData(idx) == str(target_path):
                    self.combo_ref_path.blockSignals(True)
                    self.combo_ref_path.setCurrentIndex(idx)
                    self.combo_ref_path.blockSignals(False)
                    break

    def load_reference_path(self, filepath: Path):
        """Loads reference path into memory and prepares hybrid assist."""
        if not filepath.is_file():
            QMessageBox.warning(self, "Load Error", f"Reference path file not found:\n{filepath}")
            return
        try:
            if not self.path_manager:
                self.path_manager = PathManager(robot_ip=self.robot_ip, mock=self.mock_mode)

            points = self.path_manager.load_path(filepath)
            if not points or len(points) < 2:
                QMessageBox.warning(self, "Load Error", f"Reference path has insufficient waypoints ({len(points)}).")
                return

            self.loaded_ref_path = filepath
            self.loaded_ref_waypoints = points
            dur = self.path_manager.duration_s

            msg = f"Path: {filepath.name} ({len(points)} pts, {dur:.1f}s)"
            self.lbl_hybrid_progress.setText(msg)
            self.lbl_hybrid_state.setText("HYBRID: READY")
            self.lbl_hybrid_state.setStyleSheet("""
                color: #38bdf8;
                background-color: #0369a1;
                border: 1px solid #38bdf8;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: bold;
            """)
            print(f"[HYBRID] Loaded reference path: {filepath.name} ({len(points)} waypoints, {dur:.2f}s). Ready for hybrid assist recording.")
            self.status_bar.showMessage(f"Loaded reference path: {filepath.name} ({len(points)} waypoints).", 5000)

            # Check if start pose is defined
            start_deg = self.path_manager.get_start_position_deg()
            if start_deg:
                print(f"[HYBRID] Reference path start pose: {[round(x, 1) for x in start_deg]}°")
        except Exception as e:
            print(f"[HYBRID ERROR] Failed to load reference path: {e}")
            QMessageBox.critical(self, "Load Error", f"Failed to load reference path:\n\n{e}")

    def clear_reference_path(self):
        """Clears loaded reference path and restores pure manual demonstration mode."""
        self.loaded_ref_path = None
        self.loaded_ref_waypoints = []
        if hasattr(self, "combo_ref_path"):
            self.combo_ref_path.blockSignals(True)
            self.combo_ref_path.setCurrentIndex(0)
            self.combo_ref_path.blockSignals(False)
        if hasattr(self, "lbl_hybrid_progress"):
            self.lbl_hybrid_progress.setText("Path: None (Manual Demo Mode)")
        if hasattr(self, "lbl_hybrid_state"):
            self.lbl_hybrid_state.setText("HYBRID: IDLE")
            self.lbl_hybrid_state.setStyleSheet("""
                color: #64748b;
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: bold;
            """)
        print("[HYBRID] Cleared reference path. Mode: Normal manual recording.")
        self.status_bar.showMessage("Reference path cleared. Normal manual recording mode active.", 3000)

    def on_override_delay_changed(self, index: int):
        val = self.combo_override_delay.currentData()
        if val is not None:
            self.override_delay_s = float(val)
            if self.hybrid_worker:
                self.hybrid_worker.set_override_delay_s(self.override_delay_s)
            print(f"[HYBRID] Override return delay set to {self.override_delay_s*1000:.0f} ms.")

    @pyqtSlot(str, str, str)
    def on_hybrid_state_changed(self, state_name: str, color_hex: str, desc: str):
        if hasattr(self, "lbl_hybrid_state"):
            self.lbl_hybrid_state.setText(state_name)
            self.lbl_hybrid_state.setStyleSheet(f"""
                color: #ffffff;
                background-color: {color_hex};
                border: 1px solid {color_hex};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: bold;
            """)
        self.status_bar.showMessage(desc, 4000)

    @pyqtSlot(int, int, float, float)
    def on_hybrid_progress_updated(self, curr_idx: int, total_pts: int, pct: float, dist_cm: float):
        if hasattr(self, "lbl_hybrid_progress"):
            self.lbl_hybrid_progress.setText(f"Waypoint {curr_idx}/{total_pts} ({pct:.1f}%) | Offset: {dist_cm:.1f} cm")

    @pyqtSlot(int, float, float)
    def on_hybrid_snapback_started(self, target_idx: int, duration_s: float, max_dq_deg: float):
        print(f"[HYBRID] >>> Snapping back to Waypoint #{target_idx} via Minimum-Jerk polynomial in {duration_s:.2f}s (max Δq: {max_dq_deg:.1f}°) <<<")

    @pyqtSlot(bool, str)
    def on_hybrid_playback_finished(self, success: bool, message: str):
        print(f"[HYBRID] Playback finished: success={success}, msg={message}")
        if self.is_recording and success:
            print("[HYBRID] Reference path playback completed. Automatically saving episode demonstration...")
            self.end_episode_and_save()
        elif hasattr(self, "lbl_hybrid_state"):
            self.lbl_hybrid_state.setText("COMPLETED" if success else "STOPPED")

    # -------------------------------------------------------------------------
    # Recording Workflow & Guard Logic
    # -------------------------------------------------------------------------
    def on_space_hotkey(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.end_episode_and_save()

    def on_record_button_clicked(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.end_episode_and_save()

    def start_recording(self):
        # Demonstrations can be recorded via Hybrid Assist (Pre-recorded Trajectory Playback),
        # Gamepad Cartesian Teleoperation, physical Freedrive hand-guiding, or precision axis controls.
        is_hybrid = bool(self.loaded_ref_waypoints and len(self.loaded_ref_waypoints) >= 2)
        if is_hybrid:
            mode_desc = f"Hybrid Path Assist ({len(self.loaded_ref_waypoints)} waypoints, return delay {self.override_delay_s*1000:.0f}ms)"
        elif hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled() and not self.gamepad_teleop.is_inhibited():
            mode_desc = f"Gamepad Teleoperation ({self.gamepad_teleop.get_controller_name()})"
        elif self.freedrive_active:
            mode_desc = "Freedrive Hand-Guiding"
        else:
            mode_desc = "Precision Axis Controls"
        print(f"[RECORD] Starting demonstration episode recording ({mode_desc})...")
        self.status_bar.showMessage(f"Recording episode active ({mode_desc}). Press Space to stop.", 5000)

        # Refresh camera streams on start to ensure clean driver buffers
        self.refresh_camera_streams()

        # Prepare dataset buffer
        self.is_recording = True
        self.record_start_time = time.perf_counter()
        self.recorded_frames_count = 0
        self.current_episode_buffer.clear()

        # Disable resolution and camera controls during active demonstration
        if hasattr(self, "combo_resolution"):
            self.combo_resolution.setEnabled(False)
        if hasattr(self, "combo_cam1"):
            self.combo_cam1.setEnabled(False)
        if hasattr(self, "combo_cam2"):
            self.combo_cam2.setEnabled(False)
        if hasattr(self, "btn_toggle_cam2"):
            self.btn_toggle_cam2.setEnabled(False)

        # Update UI Controls to Active Recording State
        btn_text = "⏸  END HYBRID EPISODE (Space)" if is_hybrid else "⏸  END EPISODE (Space)"
        self.btn_record.setText(btn_text)
        self.btn_record.setStyleSheet("""
            QPushButton {
                background-color: #d97706;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                border: 2px solid #f59e0b;
            }
            QPushButton:hover {
                background-color: #b45309;
            }
        """)
        self.btn_cancel.setEnabled(True)
        self.cam_widget_1.set_recording(True)
        self.cam_widget_2.set_recording(True)

        # Launch Hybrid Control Worker if a reference path is loaded
        if is_hybrid:
            # Auto-arm Gamepad Teleoperation if controller is plugged in so manual override is instantly ready
            if (
                hasattr(self, "btn_toggle_gamepad")
                and not self.btn_toggle_gamepad.isChecked()
                and self.gamepad_teleop
                and self.gamepad_teleop.is_controller_connected()
            ):
                self.btn_toggle_gamepad.setChecked(True)

            self.hybrid_worker = HybridPathWorker(
                waypoints=self.loaded_ref_waypoints,
                robot_ip=self.robot_ip,
                gamepad_teleop=self.gamepad_teleop,
                robot_worker=self.robot_worker,
                gripper=self.gripper,
                override_delay_s=self.override_delay_s,
                mock=self.mock_mode,
                parent=self,
            )
            self.hybrid_worker.state_changed.connect(self.on_hybrid_state_changed)
            self.hybrid_worker.progress_updated.connect(self.on_hybrid_progress_updated)
            self.hybrid_worker.snapback_started.connect(self.on_hybrid_snapback_started)
            self.hybrid_worker.log_message.connect(lambda lvl, msg: print(f"[{lvl}] {msg}"))
            self.hybrid_worker.finished.connect(self.on_hybrid_playback_finished)
            self.hybrid_worker.start()

        # Start Timers
        self.recording_timer.start()
        self.dataset_frame_timer.start()

        ep_idx = self.list_episodes.count()
        print(f"\n[RECORD] >>> RECORDING STARTED for Episode #{ep_idx} <<<")
        self.status_bar.showMessage(f"Recording Episode #{ep_idx}... Press Space to End or 'c' to Cancel.")

    def update_recording_timer(self):
        if not self.is_recording or self.record_start_time is None:
            return
        elapsed = time.perf_counter() - self.record_start_time
        mins = int(elapsed // 60)
        secs = elapsed % 60
        self.lbl_timer.setText(f"{mins:02d}:{secs:04.1f}")

    def capture_recording_frame(self):
        if not self.is_recording:
            return

        # Acquire synced telemetry and camera frame
        joints = self.robot_worker.get_latest_joints()
        q_vals = np.array([joints[j] for j in UR5E_JOINTS], dtype=np.float32)

        # Primary Camera (Workspace)
        cam1_frame = self.latest_camera_frame_1 if self.latest_camera_frame_1 is not None else self.latest_camera_frame
        if cam1_frame is not None:
            raw_1 = cam1_frame
            if raw_1.shape[1] != self.camera_width or raw_1.shape[0] != self.camera_height:
                raw_1 = cv2.resize(raw_1, (self.camera_width, self.camera_height), interpolation=cv2.INTER_AREA)
            if self.is_grayscale:
                if len(raw_1.shape) == 3:
                    g1 = cv2.cvtColor(raw_1, cv2.COLOR_BGR2GRAY)
                else:
                    g1 = raw_1
                img_workspace = cv2.cvtColor(g1, cv2.COLOR_GRAY2RGB)
            else:
                img_workspace = cv2.cvtColor(raw_1, cv2.COLOR_BGR2RGB)
        else:
            img_workspace = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)

        task_text = self.txt_task_desc.text().strip() or "Pick and place demonstration"

        frame_data = {
            "observation.state": q_vals,
            "action": q_vals,
            "observation.images.workspace": img_workspace,
            "task": task_text,
            "timestamp": time.perf_counter() - (self.record_start_time or time.perf_counter()),
        }

        # Secondary Camera (Wrist Flange) if dual mode is active
        if self.enable_dual_cameras:
            if self.latest_camera_frame_2 is not None:
                raw_2 = self.latest_camera_frame_2
                if raw_2.shape[1] != self.camera_width or raw_2.shape[0] != self.camera_height:
                    raw_2 = cv2.resize(raw_2, (self.camera_width, self.camera_height), interpolation=cv2.INTER_AREA)
                if self.is_grayscale:
                    if len(raw_2.shape) == 3:
                        g2 = cv2.cvtColor(raw_2, cv2.COLOR_BGR2GRAY)
                    else:
                        g2 = raw_2
                    img_wrist = cv2.cvtColor(g2, cv2.COLOR_GRAY2RGB)
                else:
                    img_wrist = cv2.cvtColor(raw_2, cv2.COLOR_BGR2RGB)
            else:
                img_wrist = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            frame_data["observation.images.wrist"] = img_wrist

        self.current_episode_buffer.append(frame_data)
        self.recorded_frames_count += 1

    def end_episode_and_save(self):
        if not self.is_recording:
            return

        try:
            # Stop Hybrid Assist worker if active
            if hasattr(self, "hybrid_worker") and self.hybrid_worker and self.hybrid_worker.isRunning():
                self.hybrid_worker.request_abort()
                self.hybrid_worker.wait(1500)

            # Stop recording timers
            self.dataset_frame_timer.stop()
            self.recording_timer.stop()
            self.is_recording = False
            duration_s = time.perf_counter() - (self.record_start_time or time.perf_counter())

            self.cam_widget_1.set_recording(False)
            self.cam_widget_2.set_recording(False)

            num_frames = len(self.current_episode_buffer)
            ep_idx = self.list_episodes.count()

            print(f"[RECORD] Queueing Episode #{ep_idx} ({num_frames} frames, {duration_s:.1f}s) for background save...")

            # Commit episode asynchronously via background save thread
            self.commit_episode_to_dataset(ep_idx, num_frames, duration_s)

            # Maintain arm in freedrive mode as requested
            print("[INFO] Episode recorded! Robot remains in Freedrive mode for the next demo.")
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed in end_episode_and_save: {e}", file=sys.__stderr__)
            import traceback
            traceback.print_exc(file=sys.__stderr__)
            self.status_bar.showMessage(f"Error saving episode: {e}", 8000)
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving the demonstration:\n\n{e}")
        finally:
            # Refresh camera streams to clear any V4L2 stalls caused by dataset disk I/O / ffmpeg encoding
            self.refresh_camera_streams()

            # Re-enable controls and restore green start state
            self.btn_record.setEnabled(True)
            self.btn_stop_session.setEnabled(True)
            if hasattr(self, "combo_resolution"):
                self.combo_resolution.setEnabled(True)
            if hasattr(self, "combo_cam1"):
                self.combo_cam1.setEnabled(True)
            if hasattr(self, "combo_cam2"):
                self.combo_cam2.setEnabled(True)
            if hasattr(self, "btn_toggle_cam2"):
                self.btn_toggle_cam2.setEnabled(True)
            self.btn_record.setText("▶  START EPISODE (Space)")
            self.btn_record.setStyleSheet("""
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
            """)

    def cancel_recording(self):
        if not self.is_recording:
            return

        if hasattr(self, "hybrid_worker") and self.hybrid_worker and self.hybrid_worker.isRunning():
            self.hybrid_worker.request_abort()
            self.hybrid_worker.wait(1500)

        self.dataset_frame_timer.stop()
        self.recording_timer.stop()
        self.is_recording = False

        self.cam_widget_1.set_recording(False)
        self.cam_widget_2.set_recording(False)
        if hasattr(self, "combo_resolution"):
            self.combo_resolution.setEnabled(True)
        if hasattr(self, "combo_cam1"):
            self.combo_cam1.setEnabled(True)
        if hasattr(self, "combo_cam2"):
            self.combo_cam2.setEnabled(True)
        if hasattr(self, "btn_toggle_cam2"):
            self.btn_toggle_cam2.setEnabled(True)

        discarded_count = len(self.current_episode_buffer)
        self.current_episode_buffer.clear()

        # Refresh camera streams when canceling
        self.refresh_camera_streams()

        # Reset timer and buttons
        self.lbl_timer.setText("00:00.0")
        self.btn_record.setText("▶  START EPISODE (Space)")
        self.btn_record.setStyleSheet("""
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
        """)
        self.btn_cancel.setEnabled(False)

        print(f"[WARNING] Recording CANCELED by user. Discarded {discarded_count} frames without saving.")
        self.status_bar.showMessage(f"Recording canceled. Arm left in Freedrive mode.", 5000)

    def stop_session(self):
        """Safely terminates recording, disables freedrive, locks the arm, and closes handles."""
        if self.is_recording:
            self.cancel_recording()

        print("\n[SYSTEM] >>> STOPPING TELEOPERATION SESSION <<<")
        if self.freedrive_active:
            print("[INFO] Re-locking robot joints safely...")
            self.robot_worker.request_toggle_freedrive()

        self.status_bar.showMessage("Session stopped. Robot locked.", 5000)

    # -------------------------------------------------------------------------
    # Dataset Management & Persistence
    # -------------------------------------------------------------------------
    def on_dataset_name_edited(self, text: str):
        """Automatically unlocks the grayscale toggle when typing a new unique dataset name."""
        clean_name = text.strip()
        if not clean_name:
            return
        target_path = DEFAULT_DATASETS_DIR / clean_name
        has_episodes = False
        target_info = target_path / "meta" / "info.json"
        if target_info.exists():
            try:
                with open(target_info, "r") as f:
                    meta = json.load(f)
                    has_episodes = meta.get("total_episodes", 0) > 0
            except Exception:
                has_episodes = True
        if hasattr(self, "chk_grayscale"):
            if not target_path.exists() or not has_episodes:
                self.chk_grayscale.setEnabled(True)
                self.chk_grayscale.setToolTip("Record dataset with 1-channel grayscale luminance (saves ~60% video disk space)")

    def create_new_dataset(self):
        """Creates a new dataset using the exact name typed in the text box.
        If a dataset with that name already exists with recorded episodes, warns the user and aborts."""
        clean_name = self.txt_dataset_name.text().strip()
        if not clean_name:
            QMessageBox.warning(self, "Missing Name", "Please type a dataset name in the text box first.")
            return

        target_dir = DEFAULT_DATASETS_DIR / clean_name
        # Check if the dataset already exists and contains data
        meta_info = target_dir / "meta" / "info.json"
        episodes_dir = target_dir / "meta" / "episodes"
        has_episodes = False
        if meta_info.exists():
            try:
                with open(meta_info, "r") as f:
                    meta = json.load(f)
                    has_episodes = meta.get("total_episodes", 0) > 0
            except Exception:
                has_episodes = True
        elif episodes_dir.exists() and any(episodes_dir.iterdir()):
            has_episodes = True

        if target_dir.exists() and has_episodes:
            msg = "The dataset '" + clean_name + "' already exists with recorded demonstrations!\n\n" +                   "A new dataset was NOT created to prevent accidental overwriting.\n\n" +                   "- To create a brand new dataset, please enter a different name.\n" +                   "- To add episodes to '" + clean_name + "', click 'Load Existing...'."
            QMessageBox.warning(self, "Dataset Already Exists", msg)
            return

        # Prepare new dataset
        self.active_dataset = None
        self.dataset_root = target_dir
        self.lbl_dataset_path.setText(f"Path: datasets/{clean_name}")
        self.list_episodes.clear()

        self.is_grayscale = False
        if hasattr(self, "lbl_color_badge"):
            self.lbl_color_badge.setText("● RGB Color Mode")
            self.lbl_color_badge.setStyleSheet("color: #a855f7; font-weight: bold; font-size: 11px; margin-left: 6px; padding: 2px 4px;")
        if hasattr(self, "combo_fps"):
            self.combo_fps.setEnabled(True)
            self.combo_fps.setToolTip("Observation and control rate (FPS). Standard for UR5e imitation learning is 30 FPS.")

        print(f"[DATASET] Prepared fresh new dataset: '{clean_name}' (Path: datasets/{clean_name})")
        self.status_bar.showMessage(f"New dataset '{clean_name}' ready in full RGB color.", 4000)
        self.update_visualizer_command()

    def load_existing_dataset(self):
        """Opens a file browser to select an existing dataset directory and locks the grayscale toggle to match."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Existing Dataset Folder",
            str(DEFAULT_DATASETS_DIR),
        )
        if not folder:
            return

        chosen_path = Path(folder).resolve()
        self.active_dataset = None
        self.dataset_root = chosen_path
        self.txt_dataset_name.blockSignals(True)
        self.txt_dataset_name.setText(chosen_path.name)
        self.txt_dataset_name.blockSignals(False)

        try:
            rel = chosen_path.relative_to(DEFAULT_DATASETS_DIR.parent)
            self.lbl_dataset_path.setText(f"Path: {rel}")
        except Exception:
            self.lbl_dataset_path.setText(f"Path: {chosen_path}")

        self.rescan_dataset_directory()
        print(f"[DATASET] Loaded existing dataset: '{chosen_path.name}' ({chosen_path})")
        self.status_bar.showMessage(f"Loaded dataset '{chosen_path.name}' ({self.list_episodes.count()} episodes).", 4000)

    def apply_dataset_name(self):
        """Alias pointing to create_new_dataset for backwards compatibility."""
        self.create_new_dataset()

    def rescan_dataset_directory(self):
        self.list_episodes.clear()

        if not self.dataset_root.exists():
            print(f"[DATASET] Directory '{self.dataset_root.name}' does not exist yet. It will be created when an episode is saved.")
            if hasattr(self, "chk_grayscale"):
                self.chk_grayscale.setEnabled(True)
                self.chk_grayscale.setToolTip("Record dataset with 1-channel grayscale luminance (saves ~60% video disk space)")
            self.update_visualizer_command()
            return

        meta_info_file = self.dataset_root / "meta" / "info.json"
        total_episodes = 0
        total_frames = 0

        if meta_info_file.exists():
            try:
                import json
                with open(meta_info_file, "r") as f:
                    meta = json.load(f)
                    total_episodes = meta.get("total_episodes", 0)
                    total_frames = meta.get("total_frames", 0)
                    cam_shape = meta.get("features", {}).get("observation.images.workspace", {}).get("shape", [])
                    if len(cam_shape) == 3:
                        h, w = cam_shape[0], cam_shape[1]
                        self._sync_resolution_ui(w, h)
                    has_wrist = "observation.images.wrist" in meta.get("features", {})
                    if has_wrist != self.enable_dual_cameras:
                        self.set_dual_cameras_enabled(has_wrist)
            except Exception as e:
                print(f"[WARNING] Could not parse {meta_info_file}: {e}")

        # Scan folder for raw video or chunk records if any
        if total_episodes > 0:
            avg_frames = total_frames // max(1, total_episodes)
            for i in range(total_episodes):
                approx_dur = avg_frames / self.target_fps
                item_text = f"Episode #{i:04d} — ~{avg_frames} frames ({approx_dur:.1f}s)"
                self.list_episodes.addItem(QListWidgetItem(item_text))
            self.list_episodes.setCurrentRow(total_episodes - 1)
        else:
            # Check for simulated local records file
            records_file = self.dataset_root / "episodes_index.txt"
            if records_file.exists():
                lines = records_file.read_text().strip().splitlines()
                for line in lines:
                    if line.strip():
                        self.list_episodes.addItem(QListWidgetItem(line.strip()))
                self.list_episodes.setCurrentRow(self.list_episodes.count() - 1)

        # Inspect dataset recording configuration (grayscale detection & episode consistency lock)
        detected_grayscale = None
        for cfg_candidate in [self.dataset_root / "meta" / "recording_config.json", self.dataset_root / "recording_config.json"]:
            if cfg_candidate.exists():
                try:
                    with open(cfg_candidate, "r") as cf:
                        cfg_json = json.load(cf)
                        if "grayscale" in cfg_json:
                            detected_grayscale = bool(cfg_json["grayscale"])
                            break
                except Exception as err:
                    print(f"[WARNING] Could not parse {cfg_candidate}: {err}")

        # Sync and lock FPS selector based on dataset metadata
        detected_fps = None
        if meta_info_file.exists():
            try:
                with open(meta_info_file, "r") as f:
                    detected_fps = json.load(f).get("fps")
            except Exception:
                pass
        if detected_fps is None:
            for cfg_c in [self.dataset_root / "meta" / "recording_config.json", self.dataset_root / "recording_config.json"]:
                if cfg_c.exists():
                    try:
                        with open(cfg_c, "r") as cf:
                            detected_fps = json.load(cf).get("fps")
                            if detected_fps:
                                break
                    except Exception:
                        pass

        if hasattr(self, "combo_fps"):
            if detected_fps is not None:
                self.combo_fps.blockSignals(True)
                fps_idx = self.combo_fps.findData(int(detected_fps))
                if fps_idx >= 0:
                    self.combo_fps.setCurrentIndex(fps_idx)
                else:
                    self.combo_fps.addItem(f"{detected_fps} FPS", int(detected_fps))
                    self.combo_fps.setCurrentIndex(self.combo_fps.count() - 1)
                self.target_fps = int(detected_fps)
                self.combo_fps.blockSignals(False)

                if total_episodes > 0:
                    self.combo_fps.setEnabled(False)
                    self.combo_fps.setToolTip(f"Dataset sampling rate locked to {detected_fps} FPS to match existing episodes.")
                else:
                    self.combo_fps.setEnabled(True)
            else:
                if total_episodes > 0:
                    self.combo_fps.setEnabled(False)
                else:
                    self.combo_fps.setEnabled(True)

        self.is_grayscale = False
        if hasattr(self, "lbl_color_badge"):
            if detected_grayscale is True:
                self.lbl_color_badge.setText("● Legacy Grayscale Detected")
                self.lbl_color_badge.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px; margin-left: 6px; padding: 2px 4px;")
            else:
                self.lbl_color_badge.setText("● RGB Color Mode")
                self.lbl_color_badge.setStyleSheet("color: #a855f7; font-weight: bold; font-size: 11px; margin-left: 6px; padding: 2px 4px;")

        print(f"[DATASET] Scanned '{self.dataset_root.name}': {self.list_episodes.count()} episodes found.")
        self.update_visualizer_command()

    def get_or_create_dataset(self) -> Optional[Any]:
        """
        Initializes or resumes the official LeRobotDataset instance for the active dataset folder.
        Automatically cleans up legacy dummy metadata and ensures standard LeRobot Parquet/video features.
        """
        if not LEROBOT_AVAILABLE or torch is None:
            print("[WARNING] 'lerobot' or 'torch' is not installed in the current environment.", file=sys.__stderr__)
            print("          Please ensure you are running in the 'lerobot' conda environment.", file=sys.__stderr__)
            return None

        if self.active_dataset is not None:
            if hasattr(self.active_dataset, "root") and Path(self.active_dataset.root).resolve() == self.dataset_root.resolve():
                return self.active_dataset

        meta_info_file = self.dataset_root / "meta" / "info.json"
        repo_id = f"local/{self.dataset_root.name}"

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": [f"{j}_pos" for j in UR5E_JOINTS],
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": [f"{j}_pos" for j in UR5E_JOINTS],
            },
            "observation.images.workspace": {
                "dtype": "video",
                "shape": (self.camera_height, self.camera_width, 3),
                "names": ["height", "width", "channels"],
            },
        }
        if self.enable_dual_cameras:
            features["observation.images.wrist"] = {
                "dtype": "video",
                "shape": (self.camera_height, self.camera_width, 3),
                "names": ["height", "width", "channels"],
            }

        # Check if an existing info.json is a valid LeRobot metadata file
        is_valid_lerobot = False
        if meta_info_file.exists():
            try:
                import json
                with open(meta_info_file, "r") as f:
                    meta = json.load(f)
                if "features" in meta and "codebase_version" in meta:
                    is_valid_lerobot = True
                    cam_shape = meta.get("features", {}).get("observation.images.workspace", {}).get("shape", [])
                    if len(cam_shape) == 3:
                        h, w = cam_shape[0], cam_shape[1]
                        self._sync_resolution_ui(w, h)
                    has_wrist = "observation.images.wrist" in meta.get("features", {})
                    if has_wrist != self.enable_dual_cameras:
                        self.set_dual_cameras_enabled(has_wrist)
            except Exception:
                is_valid_lerobot = False

        try:
            if is_valid_lerobot:
                print(f"[DATASET] Resuming official LeRobot dataset at: {self.dataset_root.resolve()}")
                self.active_dataset = LeRobotDataset.resume(repo_id=repo_id, root=self.dataset_root)
            else:
                # If directory already exists without valid metadata, remove it so LeRobotDataset.create() does not conflict
                if self.dataset_root.exists():
                    print(f"[DATASET] Cleaning pre-existing empty/invalid folder at {self.dataset_root.resolve()} for fresh creation...")
                    shutil.rmtree(self.dataset_root)

                mode_desc = "Workspace + Wrist (Dual Cam)" if self.enable_dual_cameras else "Workspace (Single Cam)"
                print(f"[DATASET] Creating fresh LeRobot dataset at: {self.dataset_root.resolve()} [{mode_desc}] ({self.camera_width}x{self.camera_height})")
                self.active_dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=self.target_fps,
                    features=features,
                    root=self.dataset_root,
                    robot_type="ur5e",
                    use_videos=True,
                )
            return self.active_dataset
        except Exception as e:
            print(f"[ERROR] Failed to initialize LeRobotDataset: {e}", file=sys.__stderr__)
            import traceback
            traceback.print_exc(file=sys.__stderr__)
            return None

    def _on_save_progress(self, msg: str):
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(msg, 3000)
        print(f"[DATASET] {msg}")

    def _on_episode_saved(
        self,
        ep_idx: int,
        num_frames: int,
        duration_s: float,
        total_episodes: int,
        success: bool,
        err_msg: str,
        payload_path: str,
    ):
        # Locate corresponding item in list_episodes
        target_item = None
        for i in range(self.list_episodes.count()):
            it = self.list_episodes.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == ep_idx:
                target_item = it
                break

        if success:
            saved_text = f"Episode #{ep_idx:04d} — {num_frames} frames ({duration_s:.1f}s)"
            if target_item is not None:
                target_item.setText(saved_text)
            print(f"[DATASET] ✔ Episode #{ep_idx:04d} saved successfully ({num_frames} frames). Total dataset episodes: {total_episodes}.")
            self.status_bar.showMessage(f"✔ Episode #{ep_idx:04d} saved! Total dataset episodes: {total_episodes}.", 6000)
        else:
            self._write_local_manifest_episode(ep_idx, num_frames, duration_s)
            failed_text = f"Episode #{ep_idx:04d} — {num_frames} frames ({duration_s:.1f}s) [UNENCODED]"
            if target_item is not None:
                target_item.setText(failed_text)
            print(f"[ERROR] Save worker failed for Episode #{ep_idx}!", file=sys.__stderr__)
            err_box_msg = (
                f"Failed to encode Episode #{ep_idx} into the dataset.\n\n"
                f"IMPORTANT: Your demonstration data was NOT lost!\n"
                f"Raw demonstration buffer is preserved at:\n{payload_path}\n\n"
                f"The studio has protected your session from crashing.\n"
                f"Details:\n{err_msg[:400]}"
            )
            QMessageBox.warning(self, "Episode Save Notice", err_box_msg)

        self.update_visualizer_command()

    def commit_episode_to_dataset(self, ep_idx: int, num_frames: int, duration_s: float):
        """
        Commits recorded demonstration frames to the LeRobot dataset.
        Executes saving via an asynchronous background subprocess worker (save_episode_worker.py)
        to guarantee:
        1. Fast, non-blocking UI responsiveness (no freeze/stutter).
        2. Safe crash isolation: video encoding faults never harm the main application.
        3. Raw demonstration buffer safety on disk.
        """
        if not self.current_episode_buffer:
            print("[WARNING] Episode buffer empty; nothing to save.")
            return

        task_str = self.txt_task_desc.text().strip() or "Pick and place demonstration"

        # 1. Prepare temporary save directory and serialize demonstration payload
        temp_dir = self.dataset_root / ".temp_save"
        temp_dir.mkdir(parents=True, exist_ok=True)
        payload_file = temp_dir / f"episode_{ep_idx:04d}_{int(time.time())}.pkl"

        payload = {
            "dataset_root": str(self.dataset_root.resolve()),
            "repo_id": f"local/{self.dataset_root.name}",
            "fps": self.target_fps,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
            "enable_dual_cameras": self.enable_dual_cameras,
            "is_grayscale": self.is_grayscale,
            "ep_idx": ep_idx,
            "num_frames": num_frames,
            "duration_s": duration_s,
            "task_str": task_str,
            "frames": list(self.current_episode_buffer),
        }

        # Clear buffer immediately to free memory in main application
        self.current_episode_buffer.clear()

        # Add pending item in the episode list with user-role metadata
        item_text = f"Episode #{ep_idx:04d} — {num_frames} frames ({duration_s:.1f}s) ⏳ [Encoding...]"
        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, ep_idx)
        self.list_episodes.addItem(item)
        self.list_episodes.setCurrentRow(self.list_episodes.count() - 1)

        task = EpisodeSaveTask(
            payload_file=payload_file,
            payload_data=payload,
            ep_idx=ep_idx,
            num_frames=num_frames,
            duration_s=duration_s,
            task_str=task_str,
        )

        if hasattr(self, "episode_save_thread") and self.episode_save_thread is not None and self.episode_save_thread.isRunning():
            self.episode_save_thread.enqueue_task(task)
            print(f"[DATASET] Enqueued Episode #{ep_idx:04d} for background H.264 streaming save...")
            self.status_bar.showMessage(f"Episode #{ep_idx:04d} recorded! Encoding video and parquet in background...", 4000)
        else:
            # Synchronous fallback if thread is not active
            print("[WARNING] Background save thread not active; falling back to direct write.", file=sys.__stderr__)
            try:
                with open(payload_file, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as dump_err:
                print(f"[ERROR] Failed to write temporary payload: {dump_err}", file=sys.__stderr__)
            self._write_local_manifest_episode(ep_idx, num_frames, duration_s)
            item.setText(f"Episode #{ep_idx:04d} — {num_frames} frames ({duration_s:.1f}s) [UNENCODED]")
            self.update_visualizer_command()

    def _write_local_manifest_episode(self, ep_idx: int, num_frames: int, duration_s: float):
        """Standard backup manifest recording episode records."""
        try:
            self.dataset_root.mkdir(parents=True, exist_ok=True)
            records_file = self.dataset_root / "episodes_index.txt"
            entry = f"Episode #{ep_idx:04d} — {num_frames} frames ({duration_s:.1f}s) | Task: {self.txt_task_desc.text().strip()}\n"
            with open(records_file, "a") as f:
                f.write(entry)

            meta_dir = self.dataset_root / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_info_file = meta_dir / "info.json"
            meta_data = {"total_episodes": ep_idx + 1, "fps": self.target_fps, "robot_type": "ur5e"}
            if meta_info_file.exists():
                try:
                    import json
                    with open(meta_info_file, "r") as f:
                        meta_data = json.load(f)
                    meta_data["total_episodes"] = max(meta_data.get("total_episodes", 0), ep_idx + 1)
                    meta_data["total_frames"] = meta_data.get("total_frames", 0) + num_frames
                except Exception:
                    pass
            with open(meta_info_file, "w") as f:
                import json
                json.dump(meta_data, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Error writing local manifest: {e}", file=sys.__stderr__)

    def delete_episode(self):
        """
        Deletes the episode currently highlighted in the list widget.
        If no episode is selected, defaults to deleting the last episode.
        """
        total_items = self.list_episodes.count()
        if total_items == 0:
            QMessageBox.information(self, "No Episodes", "The active dataset contains no recorded episodes to delete.")
            return

        current_row = self.list_episodes.currentRow()
        if current_row < 0 or current_row >= total_items:
            # Default to the most recent/last episode
            target_row = total_items - 1
        else:
            target_row = current_row

        item = self.list_episodes.item(target_row)
        if "[Encoding...]" in item.text():
            QMessageBox.warning(
                self,
                "Save in Progress",
                "This episode is currently being encoded and saved to disk in the background.\n\n"
                "Please wait a moment for saving to complete before deleting."
            )
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Delete Episode",
            f"Are you sure you want to permanently delete:\n\n<b>{item.text()}</b>\n\n(List position #{target_row + 1} of {total_items})?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if confirm != QMessageBox.StandardButton.Yes:
            return

        # Remove item from UI
        self.list_episodes.takeItem(target_row)

        # Update metadata in info.json
        meta_info_file = self.dataset_root / "meta" / "info.json"
        if meta_info_file.exists():
            try:
                import json
                with open(meta_info_file, "r") as f:
                    meta_data = json.load(f)
                meta_data["total_episodes"] = max(0, meta_data.get("total_episodes", 1) - 1)
                with open(meta_info_file, "w") as f:
                    json.dump(meta_data, f, indent=2)
            except Exception as e:
                print(f"[WARNING] Could not update meta/info.json: {e}")

        # Update manifest file if present
        records_file = self.dataset_root / "episodes_index.txt"
        if records_file.exists():
            lines = records_file.read_text().strip().splitlines()
            if lines and target_row < len(lines):
                del lines[target_row]
                records_file.write_text("\n".join(lines) + ("\n" if lines else ""))

        print(f"[INFO] Deleted episode at list position #{target_row + 1} from dataset '{self.dataset_root.name}'.")
        self.status_bar.showMessage(f"Episode deleted from '{self.dataset_root.name}'.", 4000)
        self.update_visualizer_command()

    # -------------------------------------------------------------------------
    # Hardware Diagnostics Runner
    # -------------------------------------------------------------------------
    def run_hardware_test(self):
        if self.is_recording:
            QMessageBox.warning(self, "Recording in Progress", "Cannot run hardware verification while recording an episode.")
            return

        test_script = Path(__file__).resolve().parent / "test_lerobot_ur5e.py"
        self.btn_test_hardware.setEnabled(False)
        self.btn_test_hardware.setText("⏳ Running Hardware Verification Suite...")

        self.diagnostic_worker = HardwareTestWorker(test_script, self.robot_ip)
        self.diagnostic_worker.output_line.connect(self.append_log_line)
        self.diagnostic_worker.test_finished.connect(self.on_hardware_test_completed)
        self.diagnostic_worker.start()

    @pyqtSlot(bool, str)
    def on_hardware_test_completed(self, success: bool, message: str):
        self.btn_test_hardware.setEnabled(True)
        self.btn_test_hardware.setText("🔬 Run Full Hardware Test (test_lerobot_ur5e.py)")
        if success:
            QMessageBox.information(self, "Hardware Verification Passed", "UR5e arm and RTDE telemetry successfully verified!")
        else:
            QMessageBox.warning(self, "Hardware Verification Failed", f"Verification encountered an issue:\n{message}")

    # -------------------------------------------------------------------------
    # Log Console Helpers
    # -------------------------------------------------------------------------
    @pyqtSlot(str, str)
    def append_log(self, level: str, message: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] [{level}] "
        self.txt_log.appendPlainText(prefix + message)
        # Scroll to bottom
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    @pyqtSlot(str)
    def append_log_line(self, raw_line: str):
        self.txt_log.appendPlainText(raw_line)
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # -------------------------------------------------------------------------
    # Safe Teardown on Window Close
    # -------------------------------------------------------------------------
    def closeEvent(self, event):
        print("\n[SYSTEM] Closing Teleoperation Studio window...")
        if hasattr(self, "hybrid_worker") and self.hybrid_worker and self.hybrid_worker.isRunning():
            try:
                self.hybrid_worker.request_abort()
                self.hybrid_worker.wait(1500)
            except Exception:
                pass

        if self.is_recording:
            self.cancel_recording()

        if self.diagnostic_worker and self.diagnostic_worker.isRunning():
            self.diagnostic_worker.terminate_process()

        # Stop workers
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop is not None:
            try:
                self.gamepad_teleop.stop()
                self.gamepad_teleop.wait(1000)
            except Exception:
                pass
        if self.camera_worker_1 is not None:
            self.camera_worker_1.stop()
        if self.camera_worker_2 is not None:
            self.camera_worker_2.stop()
        self.robot_worker.stop()

        # Stop background dataset save thread
        if hasattr(self, "episode_save_thread") and self.episode_save_thread is not None:
            if self.episode_save_thread.is_busy():
                print("[SYSTEM] Waiting for background episode encoding to finish before closing...")
                if hasattr(self, "status_bar"):
                    self.status_bar.showMessage("Finalizing dataset encoding before exit...", 5000)
                self.episode_save_thread.wait_until_empty(timeout_s=15)
            self.episode_save_thread.stop()

        if self.active_dataset is not None:
            try:
                self.active_dataset.finalize()
                self.active_dataset = None
            except Exception as e:
                print(f"[SYSTEM] Dataset finalize on close: {e}", file=sys.__stderr__)

        # Restore system standard streams
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        event.accept()


# =============================================================================
# 7. Application Entrypoint
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="UR5e LeRobot Demonstration Dataset Studio")
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP, help=f"UR5e PolyScope IP (default: {DEFAULT_ROBOT_IP})")
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX, help=f"OpenCV Camera 1 index (default: {DEFAULT_CAMERA_INDEX})")
    parser.add_argument("--camera2-index", type=int, default=DEFAULT_CAMERA_2_INDEX, help=f"OpenCV Camera 2 index (default: {DEFAULT_CAMERA_2_INDEX})")
    parser.add_argument("--dual-cam", action="store_true", help="Launch with secondary camera stream enabled")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"Target camera and recording FPS (default: {DEFAULT_FPS})")
    parser.add_argument("--mock", action="store_true", help="Launch in Simulation/Mock mode without physical UR5e or USB camera")
    return parser.parse_args()


def main():
    args = parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("UR5e LeRobot Studio")

    studio = UR5eRecorderStudio(
        robot_ip=args.robot_ip,
        camera_index=args.camera_index,
        camera_2_index=args.camera2_index,
        dual_cam=args.dual_cam,
        fps=args.fps,
        mock=args.mock,
    )
    studio.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

