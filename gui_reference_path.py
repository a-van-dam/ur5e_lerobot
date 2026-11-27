#!/usr/bin/env python3
"""
UR5e Reference Path Studio (Phase 1 "Golden" Trajectory Tool)
==============================================================
Dedicated graphical interface for recording, saving, loading, and playing
back deterministic reference trajectories for Hugging Face LeRobot on the
UR5e arm with OnRobot 2FG14 Modbus TCP gripper.

Features:
1. Dual Camera Live Feeds:
   - Simultaneous real-time feeds from Workspace (Cam 1) and Wrist Flange (Cam 2)
   - Visual guidance for operator control (NOT recorded as part of reference path)
   - Device selectors, resolution presets, live FPS counters, and mock test pattern fallbacks
2. Trajectory Start Position Tracking & Automatic Pre-Replay Alignment:
   - Start position is recorded and serialized into CSV comments and JSON metadata
   - Default Start Coordinates: -80°, -45°, -100°, 150°, -85°, 60°, Gripper 105.0 mm (wide open)
   - During playback, automatically detects if the arm is not at the file's start position
     and smoothly navigates there before waypoint streaming begins
3. Advanced Control & Workflow Toggles:
   - Checkboxes for inverting both Left Stick (L/R translation) and Right Stick (Base rotation)
   - Checkbox to automatically return the robot to the Start Position after recording finishes
4. Trajectory Management:
   - Save, load, and delete recorded reference paths via GUI button and confirmation dialog
   - Instant file inspection and selection
5. Status Indicators:
   - UR5e connection status, Gripper Modbus connection status, Gamepad hardware status
6. Emergency Safety STOP:
   - Instant halt sending stopj()/stopl() to UR5e and stopping the gripper
7. Jogging, Start/Home Poses & Gripper Aperture:
   - Canonical Home Pose, Go to Temp Start, Set New Start, Move to Target, Sync
   - 6-DoF Joint Jog Sliders (-360° to +360°) and numeric inputs
   - Gripper aperture slider (55 - 105 mm) with quick presets
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# PyQt6 UI Framework
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
    QImage,
    QKeySequence,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
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
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

# Reference Path Backend Logic
from path_manager import (
    CSV_HEADER,
    PathManager,
    TrajectoryPoint,
    UR5E_JOINT_NAMES,
)

# Optional Gamepad Teleoperation Controller
try:
    from gamepad_controller import GamepadConfig, UR5eGamepadTeleop, PYGAME_AVAILABLE
    GAMEPAD_AVAILABLE = True
except ImportError:
    GAMEPAD_AVAILABLE = False
    UR5eGamepadTeleop = None
    GamepadConfig = None
    PYGAME_AVAILABLE = False

# Optional OnRobot 2FG Modbus TCP Driver
try:
    from onrobot_gripper import OnRobot2FGModbus, get_shared_gripper
    ONROBOT_GRIPPER_AVAILABLE = True
except ImportError:
    OnRobot2FGModbus = None
    get_shared_gripper = None
    ONROBOT_GRIPPER_AVAILABLE = False

# Optional LeRobot UR5e Driver
try:
    from ur5e.configuration_ur5e import UR5eRobotConfig
    from ur5e.robot_ur5e import UR5eRobot
    LEROBOT_UR5E_AVAILABLE = True
except ImportError:
    UR5eRobotConfig = None
    UR5eRobot = None
    LEROBOT_UR5E_AVAILABLE = False


from camera_utils import (
    detect_available_cameras,
    open_video_capture,
    resolve_camera_indices,
)

# Default Network and Kinematic Configuration
DEFAULT_ROBOT_IP = "192.168.1.102"
DEFAULT_GRIPPER_IP = "192.168.1.1"
DEFAULT_COMMAND_PORT = 30002
DEFAULT_FPS = 30
DEFAULT_CAMERA_1_INDEX = 4       # Fallback preference; auto-resolved to connected cameras at startup
DEFAULT_CAMERA_2_INDEX = 6       # Fallback preference; auto-resolved to connected cameras at startup

HOME_JOINTS_DICT: Dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -90.0,
    "elbow": 90.0,
    "wrist_1": -90.0,
    "wrist_2": -90.0,
    "wrist_3": 0.0,
}

# Centralized start position loader from default_start_position.json
try:
    from start_position_config import (
        get_default_start_joints_dict,
        get_default_start_gripper_mm,
        save_start_position as persist_start_position,
        load_start_position,
        UR5E_JOINT_NAMES,
    )
    DEFAULT_START_JOINTS_DICT = get_default_start_joints_dict()
    DEFAULT_START_GRIPPER_MM = get_default_start_gripper_mm()
except ImportError:
    DEFAULT_START_JOINTS_DICT: Dict[str, float] = {
        "shoulder_pan": -90.0,
        "shoulder_lift": -40.0,
        "elbow": -100.0,
        "wrist_1": 120.0,
        "wrist_2": -90.0,
        "wrist_3": 120.0,
    }
    DEFAULT_START_GRIPPER_MM: float = 105.0
    persist_start_position = None


# Camera discovery is provided by camera_utils.detect_available_cameras


# =============================================================================
# Camera Capture Worker & Display Widget
# =============================================================================

class CameraWorker(QThread):
    """
    Dedicated QThread capturing frames from OpenCV camera device.
    Runs asynchronously at target FPS to provide smooth operator feedback.
    """

    frame_ready = pyqtSignal(int, np.ndarray)  # (camera_id, bgr_image)
    fps_updated = pyqtSignal(int, float)       # (camera_id, measured_fps)
    status_signal = pyqtSignal(int, bool, str) # (camera_id, is_connected, message)

    def __init__(
        self,
        camera_id: int,
        target_fps: int = 30,
        width: int = 320,
        height: int = 240,
        mock: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.target_width = width
        self.target_height = height
        self.mock = mock
        self._using_fallback = False
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._mutex = QMutex()
        self._reconnect_requested = False

    def set_camera_id(self, new_id: int):
        with QMutexLocker(self._mutex):
            self.camera_id = new_id
            self._reconnect_requested = True

    def set_resolution(self, width: int, height: int):
        with QMutexLocker(self._mutex):
            self.target_width = width
            self.target_height = height
            cap = self._cap
        if cap is not None and cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            except Exception:
                pass

    def request_reconnect(self):
        with QMutexLocker(self._mutex):
            self._reconnect_requested = True

    def run(self):
        self._running = True

        if not self.mock:
            self._cap = open_video_capture(
                self.camera_id, self.target_width, self.target_height, self.target_fps
            )

            if not self._cap.isOpened():
                self.status_signal.emit(
                    self.camera_id,
                    False,
                    f"Camera /dev/video{self.camera_id} unavailable (Fallback synthetic feed)",
                )
                self._using_fallback = True
            else:
                self._using_fallback = False
                self.status_signal.emit(
                    self.camera_id,
                    True,
                    f"Camera /dev/video{self.camera_id} connected @ {self.target_fps} FPS",
                )

        sim_t = 0.0
        fps_counter = 0
        fps_start = time.perf_counter()

        while self._running:
            t0 = time.perf_counter()

            with QMutexLocker(self._mutex):
                tw, th = self.target_width, self.target_height
                current_id = self.camera_id
                reconnect_needed = self._reconnect_requested
                self._reconnect_requested = False

            if reconnect_needed and not self.mock:
                if self._cap is not None:
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None
                time.sleep(0.05)
                self._cap = open_video_capture(current_id, tw, th, self.target_fps)
                if self._cap.isOpened():
                    self._using_fallback = False
                    self.status_signal.emit(
                        current_id,
                        True,
                        f"Camera /dev/video{current_id} reconnected @ {self.target_fps} FPS",
                    )
                else:
                    self._using_fallback = True
                    self.status_signal.emit(
                        current_id,
                        False,
                        f"Camera /dev/video{current_id} unavailable (Fallback synthetic feed)",
                    )

            if not self.mock and not self._using_fallback and self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                if frame.shape[1] != tw or frame.shape[0] != th:
                    frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            else:
                # Synthetic animated test frame with resolution and timestamp
                frame = np.zeros((th, tw, 3), dtype=np.uint8)
                sim_t += 0.04
                gx = int(128 + 60 * math.sin(sim_t + current_id))
                gy = int(128 + 60 * math.cos(sim_t * 0.8 + current_id))
                frame[:, :, 0] = 25
                frame[:, :, 1] = int(gx * 0.35)
                frame[:, :, 2] = int(gy * 0.45)

                step = max(24, min(tw, th) // 7)
                for x in range(0, tw, step):
                    cv2.line(frame, (x, 0), (x, th), (45, 55, 72), 1)
                for y in range(0, th, step):
                    cv2.line(frame, (0, y), (tw, y), (45, 55, 72), 1)

                cv2.putText(
                    frame,
                    f"SIM CAM {current_id}: {tw}x{th}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (56, 189, 248),
                    1,
                    cv2.LINE_AA,
                )
                ts_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-4]
                cv2.putText(
                    frame,
                    ts_str,
                    (12, th - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (148, 163, 184),
                    1,
                    cv2.LINE_AA,
                )

            self.frame_ready.emit(current_id, frame)

            # Benchmark FPS
            fps_counter += 1
            if time.perf_counter() - fps_start >= 1.0:
                measured_fps = fps_counter / (time.perf_counter() - fps_start)
                self.fps_updated.emit(current_id, measured_fps)
                fps_counter = 0
                fps_start = time.perf_counter()

            elapsed = time.perf_counter() - t0
            sleep_time = (1.0 / max(1, self.target_fps)) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._running = False

    def stop(self):
        self._running = False
        self.wait(1500)


class CameraFeedWidget(QWidget):
    """Clean video feed viewport widget displaying camera feed with FPS and status."""

    def __init__(self, title: str, camera_id: int, parent=None):
        super().__init__(parent)
        self.title = title
        self.camera_id = camera_id

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self.lbl_title = QLabel(f"<b>{self.title}</b>")
        self.lbl_title.setStyleSheet("color: #94a3b8; font-size: 11px;")
        header.addWidget(self.lbl_title)

        header.addStretch()

        self.lbl_badge = QLabel("LIVE")
        self.lbl_badge.setStyleSheet("""
            color: #10b981;
            font-size: 9px;
            font-weight: 800;
            padding: 1px 5px;
            background-color: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 3px;
        """)
        header.addWidget(self.lbl_badge)

        self.lbl_fps = QLabel("0.0 FPS")
        self.lbl_fps.setStyleSheet("color: #38bdf8; font-family: monospace; font-size: 10px; font-weight: bold;")
        header.addWidget(self.lbl_fps)

        layout.addLayout(header)

        self.video_canvas = QLabel("Initializing Video Feed...")
        self.video_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_canvas.setStyleSheet("""
            background-color: #050811;
            color: #64748b;
            font-weight: bold;
            font-size: 11px;
            border: 1px solid #1e293b;
            border-radius: 4px;
        """)
        self.video_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_canvas.setMinimumSize(220, 160)
        layout.addWidget(self.video_canvas)

    @pyqtSlot(int, float)
    def update_fps(self, cam_id: int, fps: float):
        if cam_id == self.camera_id:
            self.lbl_fps.setText(f"{fps:4.1f} FPS")

    @pyqtSlot(int, np.ndarray)
    def update_frame(self, cam_id: int, frame_bgr: np.ndarray):
        if cam_id != self.camera_id:
            return

        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

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
# Robot Telemetry & Discrete Motion Worker Thread
# =============================================================================

class RobotRTDEWorker(QThread):
    """
    Dedicated QThread maintaining real-time RTDE communication with UR5e.
    Polls joint angles at 125 Hz and processes discrete motions (Go to Home, Go to Start).
    """

    telemetry_signal = pyqtSignal(dict)              # Joint angles, TCP pose, latency
    connection_state_changed = pyqtSignal(bool, str) # (is_connected, message)
    motion_finished = pyqtSignal(bool, str)          # (success, message)

    def __init__(self, robot_ip: str = DEFAULT_ROBOT_IP, mock: bool = False, parent=None):
        super().__init__(parent)
        self.robot_ip = robot_ip
        self.mock = mock
        self._running = False
        self._robot: Optional[Any] = None

        self._cmd_lock = QMutex()
        self._cmd_disconnect = False
        self._cmd_target_pose: Optional[Dict[str, float]] = None

        # Motion monitoring
        self._motion_in_progress = False
        self._motion_target_q: List[float] = [0.0] * 6
        self._motion_start_time = 0.0
        self._motion_timeout = 8.0

        # Cached telemetry state initialized to default start coordinates
        self._latest_q: Dict[str, float] = {
            j: math.radians(DEFAULT_START_JOINTS_DICT[j]) for j in UR5E_JOINT_NAMES
        }
        self._latest_tcp: List[float] = [-0.1334, -0.0631, 0.5503, 1.0980, -1.8947, 1.3240]
        self._mock_velocity: List[float] = [0.0] * 6

    @property
    def is_motion_in_progress(self) -> bool:
        with QMutexLocker(self._cmd_lock):
            return self._motion_in_progress

    def get_latest_joints(self) -> Dict[str, float]:
        with QMutexLocker(self._cmd_lock):
            return dict(self._latest_q)

    def get_latest_tcp(self) -> List[float]:
        with QMutexLocker(self._cmd_lock):
            return list(self._latest_tcp)

    def request_move_pose(self, pose: Dict[str, float]):
        with QMutexLocker(self._cmd_lock):
            self._cmd_target_pose = dict(pose)

    def request_disconnect(self):
        with QMutexLocker(self._cmd_lock):
            self._cmd_disconnect = True

    def apply_cartesian_velocity(self, v: List[float]):
        with QMutexLocker(self._cmd_lock):
            self._mock_velocity = list(v)

    def run(self):
        self._running = True

        # Connect to physical arm or fallback to mock
        if not self.mock and LEROBOT_UR5E_AVAILABLE and UR5eRobot is not None:
            try:
                config = UR5eRobotConfig(robot_ip=self.robot_ip)
                self._robot = UR5eRobot(config)
                self._robot.connect()
                self.connection_state_changed.emit(True, f"Connected to UR5e at {self.robot_ip}")
            except Exception as e:
                self.connection_state_changed.emit(
                    False, f"Hardware connection failed ({e}). Entering Mock Robot Mode."
                )
                self.mock = True
                self._robot = None
        else:
            self.connection_state_changed.emit(
                True, f"Mock Robot Mode active (Simulated UR5e on {self.robot_ip})"
            )

        emit_interval = 1.0 / 30.0
        last_emit_time = time.perf_counter()

        while self._running:
            t0 = time.perf_counter()

            # 1. Process queued motion commands
            with QMutexLocker(self._cmd_lock):
                if self._cmd_disconnect:
                    break

                if self._cmd_target_pose is not None:
                    target_pose = self._cmd_target_pose
                    self._cmd_target_pose = None
                    self._motion_target_q = [
                        float(target_pose.get(f"{j}.pos", target_pose.get(j, 0.0)))
                        for j in UR5E_JOINT_NAMES
                    ]
                    self._motion_in_progress = True
                    self._motion_start_time = time.perf_counter()

                    max_disp = max(
                        abs(self._latest_q[j] - self._motion_target_q[i])
                        for i, j in enumerate(UR5E_JOINT_NAMES)
                    )
                    self._motion_timeout = max(8.0, (max_disp / 0.3) + 3.0)

                    if self._robot and hasattr(self._robot, "send_action"):
                        try:
                            action_dict = {f"{j}.pos": self._motion_target_q[i] for i, j in enumerate(UR5E_JOINT_NAMES)}
                            self._robot.send_action(action_dict)
                        except Exception as err:
                            self._motion_in_progress = False
                            self.motion_finished.emit(False, f"Motion failed: {err}")

            # 2. Acquire Telemetry
            if not self.mock and self._robot and hasattr(self._robot, "rx") and self._robot.rx:
                try:
                    q_list = self._robot.rx.getActualQ()
                    for idx, j_name in enumerate(UR5E_JOINT_NAMES):
                        self._latest_q[j_name] = float(q_list[idx])
                except Exception:
                    pass

                try:
                    tcp_list = self._robot.rx.getActualTCPPose()
                    self._latest_tcp = [float(v) for v in tcp_list]
                except Exception:
                    pass

                # Monitor discrete motion completion
                if self._motion_in_progress:
                    disp = max(
                        abs(self._latest_q[j] - self._motion_target_q[i])
                        for i, j in enumerate(UR5E_JOINT_NAMES)
                    )
                    elapsed = time.perf_counter() - self._motion_start_time
                    if disp < 0.02:
                        self._motion_in_progress = False
                        self.motion_finished.emit(True, f"Target reached in {elapsed:.2f}s.")
                    elif elapsed >= self._motion_timeout:
                        self._motion_in_progress = False
                        self.motion_finished.emit(False, "Motion timed out.")
            else:
                # Mock simulation kinematics
                dt = 0.008
                with QMutexLocker(self._cmd_lock):
                    if self._motion_in_progress:
                        step_size = 0.8 * dt
                        all_reached = True
                        for idx, j in enumerate(UR5E_JOINT_NAMES):
                            diff = self._motion_target_q[idx] - self._latest_q[j]
                            if abs(diff) > step_size:
                                self._latest_q[j] += math.copysign(step_size, diff)
                                all_reached = False
                            else:
                                self._latest_q[j] = self._motion_target_q[idx]

                        if all_reached or (time.perf_counter() - self._motion_start_time >= self._motion_timeout):
                            self._motion_in_progress = False
                            self.motion_finished.emit(True, "Mock target reached.")
                    else:
                        # Apply mock velocity from gamepad if any: [vx, vy, vz, w_base, w_wrist2, w_wrist1]
                        self._latest_q["shoulder_pan"] += self._mock_velocity[3] * dt
                        self._latest_q["wrist_2"] += self._mock_velocity[4] * dt
                        self._latest_q["wrist_1"] += self._mock_velocity[5] * dt
                        self._latest_tcp[0] += self._mock_velocity[0] * dt
                        self._latest_tcp[1] += self._mock_velocity[1] * dt
                        self._latest_tcp[2] += self._mock_velocity[2] * dt

            # 3. Throttle signal emission for smooth UI fluidity (30 Hz)
            now = time.perf_counter()
            if now - last_emit_time >= emit_interval:
                last_emit_time = now
                with QMutexLocker(self._cmd_lock):
                    payload = {
                        "joints": dict(self._latest_q),
                        "tcp_pose": list(self._latest_tcp),
                        "timestamp": now,
                    }
                self.telemetry_signal.emit(payload)

            elapsed = time.perf_counter() - t0
            sleep_s = 0.008 - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception:
                pass
            self._robot = None
        self._running = False

    def stop(self):
        self._running = False
        self.request_disconnect()
        self.wait(2000)


# =============================================================================
# Main Reference Path Studio Window
# =============================================================================

class ReferencePathStudio(QMainWindow):
    """
    Main application window for UR5e Reference Path recording, saving,
    loading, and deterministic playback with dual live camera operator feeds.
    """

    def __init__(
        self,
        robot_ip: str = DEFAULT_ROBOT_IP,
        gripper_ip: str = DEFAULT_GRIPPER_IP,
        target_fps: int = DEFAULT_FPS,
        camera_1_index: int = DEFAULT_CAMERA_1_INDEX,
        camera_2_index: int = DEFAULT_CAMERA_2_INDEX,
        mock_mode: bool = False,
    ):
        super().__init__()
        self.robot_ip = robot_ip
        self.gripper_host = gripper_ip
        self.target_fps = target_fps
        self._available_cameras = detect_available_cameras()
        self.camera_1_index, self.camera_2_index = resolve_camera_indices(
            self._available_cameras, camera_1_index, camera_2_index
        )
        self.mock_mode = mock_mode

        # Active Start Position initialized to requested defaults:
        # [-80, -45, -100, 150, -85, 60] deg, Gripper: 105.0 mm (wide open)
        self.start_position: Dict[str, float] = dict(DEFAULT_START_JOINTS_DICT)
        self.start_gripper_mm: float = DEFAULT_START_GRIPPER_MM

        # Staged jog values
        self._user_targets: Dict[str, float] = {}
        self._user_modified_joints: set[str] = set()
        self._user_dragging_joint: Optional[str] = None
        self._last_progress_update_time = 0.0

        # Initialize PathManager Backend
        self.path_manager = PathManager(
            robot_ip=self.robot_ip,
            gripper_host=self.gripper_host,
            mock=self.mock_mode,
        )
        self.path_manager.set_start_position(self.start_position, gripper_width=self.start_gripper_mm)

        # Initialize OnRobot Gripper
        self.gripper: Optional[OnRobot2FGModbus] = None
        self._init_gripper()

        # Build User Interface
        self.init_ui()
        self.apply_theme()

        # Start Camera Workers for Dual Live Operator Feeds
        self._init_cameras()

        # Start Robot RTDE Background Worker
        self.robot_worker = RobotRTDEWorker(self.robot_ip, self.mock_mode, parent=self)
        self.robot_worker.telemetry_signal.connect(self.on_robot_telemetry_updated)
        self.robot_worker.connection_state_changed.connect(self.on_robot_connection_status)
        self.robot_worker.motion_finished.connect(self.on_discrete_motion_finished)
        self.robot_worker.start()

        # Connect PathManager to robot worker state provider
        self.path_manager.set_state_provider(self._sample_robot_state_for_path_manager)

        # Initialize Gamepad Teleoperation Worker
        self.gamepad_teleop: Optional[UR5eGamepadTeleop] = None
        self._init_gamepad()

        # High-resolution Recording Stopwatch Timer
        self.recording_timer = QTimer(self)
        self.recording_timer.setInterval(50)  # 20 Hz UI refresh rate
        self.recording_timer.timeout.connect(self.update_recording_stopwatch)
        self._recording_start_time = 0.0

        # Scan initial reference paths
        self.refresh_saved_paths_list()

        self.log(f"UR5e Reference Path Studio initialized. Robot: {self.robot_ip} | Gripper: {self.gripper_host}")
        if self.mock_mode:
            self.log("Running in SIMULATION/MOCK mode. No physical hardware required.")

    # -------------------------------------------------------------------------
    # Styling & Visual Theme
    # -------------------------------------------------------------------------

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0f172a;
                color: #f1f5f9;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            }
            QGroupBox {
                font-size: 11px;
                font-weight: 700;
                color: #38bdf8;
                border: 1px solid #334155;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #1e293b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                background-color: #1e293b;
            }
            QPushButton {
                background-color: #334155;
                color: #f8fafc;
                font-size: 11px;
                font-weight: 600;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 5px 12px;
            }
            QPushButton:hover {
                background-color: #475569;
            }
            QPushButton:pressed {
                background-color: #1e293b;
            }
            QPushButton:disabled {
                background-color: #1e293b;
                color: #64748b;
                border: 1px solid #334155;
            }
            QLineEdit, QComboBox {
                background-color: #0b1120;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 11px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #38bdf8;
            }
            QListWidget {
                background-color: #0b1120;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 4px;
                font-family: monospace;
                font-size: 10px;
            }
            QListWidget::item:selected {
                background-color: #0284c7;
                color: #ffffff;
            }
            QProgressBar {
                background-color: #0b1120;
                border: 1px solid #334155;
                border-radius: 4px;
                text-align: center;
                color: #f8fafc;
                font-size: 10px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #38bdf8;
                border-radius: 3px;
            }
            QScrollBar:vertical {
                border: none;
                background: #0f172a;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 4px;
                min-height: 20px;
            }
        """)

    # -------------------------------------------------------------------------
    # Hardware & Camera Initialization Helpers
    # -------------------------------------------------------------------------

    def _init_cameras(self):
        """Initializes asynchronous camera feeds for visual operator feedback."""
        # Camera 1 (Workspace)
        self.camera_worker_1 = CameraWorker(
            camera_id=self.camera_1_index,
            target_fps=self.target_fps,
            width=320,
            height=240,
            mock=self.mock_mode,
            parent=self,
        )
        self.camera_worker_1.frame_ready.connect(self.cam_widget_1.update_frame)
        self.camera_worker_1.fps_updated.connect(self.cam_widget_1.update_fps)
        self.camera_worker_1.start()

        # Camera 2 (Wrist Flange)
        self.camera_worker_2 = CameraWorker(
            camera_id=self.camera_2_index,
            target_fps=self.target_fps,
            width=320,
            height=240,
            mock=self.mock_mode,
            parent=self,
        )
        self.camera_worker_2.frame_ready.connect(self.cam_widget_2.update_frame)
        self.camera_worker_2.fps_updated.connect(self.cam_widget_2.update_fps)
        self.camera_worker_2.start()

    def _init_gripper(self):
        """Initializes connection to OnRobot 2FG14 gripper."""
        if ONROBOT_GRIPPER_AVAILABLE:
            try:
                if self.mock_mode and OnRobot2FGModbus is not None:
                    self.gripper = OnRobot2FGModbus(host=self.gripper_host, mock=True)
                elif get_shared_gripper is not None:
                    self.gripper = get_shared_gripper(host=self.gripper_host, auto_connect=True)
                elif OnRobot2FGModbus is not None:
                    self.gripper = OnRobot2FGModbus(host=self.gripper_host)
                    self.gripper.connect()
                self.path_manager.set_gripper(self.gripper)
            except Exception as e:
                self.log(f"[GRIPPER] Init notice: {e}")
        else:
            self.gripper = None

    def _init_gamepad(self):
        """Initializes gamepad teleoperation thread."""
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
            self.gamepad_teleop.button_select_pressed.connect(self.on_gamepad_select_clicked)
            self.gamepad_teleop.button_start_pressed.connect(self.on_gamepad_start_clicked)

            # Kinematics providers
            self.gamepad_teleop.set_base_angle_provider(
                lambda: self.robot_worker.get_latest_joints().get("shoulder_pan", 0.0)
            )
            self.gamepad_teleop.set_joint_positions_provider(
                lambda: self.robot_worker.get_latest_joints()
            )
            self.gamepad_teleop.set_tcp_pose_provider(
                lambda: self.robot_worker.get_latest_tcp()
            )

            if hasattr(self, "slider_gamepad_speed"):
                self.gamepad_teleop.set_speed_percentage(self.slider_gamepad_speed.value())
            if hasattr(self, "slider_gamepad_accel"):
                self.gamepad_teleop.set_accel_percentage(self.slider_gamepad_accel.value())

            # Apply initial analog stick inversion settings
            if hasattr(self, "chk_invert_left"):
                self.gamepad_teleop.set_invert_left_horizontal(self.chk_invert_left.isChecked())
            if hasattr(self, "chk_invert_right"):
                self.gamepad_teleop.set_invert_right_horizontal(self.chk_invert_right.isChecked())

            if self.mock_mode:
                self.gamepad_teleop.set_velocity_callback(self.robot_worker.apply_cartesian_velocity)

            self.gamepad_teleop.start()
        else:
            self.gamepad_teleop = None

    def _sample_robot_state_for_path_manager(self) -> Tuple[List[float], List[float], float]:
        """State provider hook given to PathManager."""
        joints_dict = self.robot_worker.get_latest_joints()
        q = [joints_dict.get(j, 0.0) for j in UR5E_JOINT_NAMES]
        tcp = self.robot_worker.get_latest_tcp()
        gripper_w = self.gripper.get_width() if (self.gripper and self.gripper.is_connected) else self.start_gripper_mm
        return q, tcp, gripper_w

    # -------------------------------------------------------------------------
    # UI Layout Construction
    # -------------------------------------------------------------------------

    def init_ui(self):
        self.setWindowTitle("UR5e Reference Path Studio — Phase 1 Deterministic Trajectory Tool")
        self.resize(1440, 920)
        self.setMinimumSize(1100, 720)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # 1. Top Status & System Badges Bar
        top_bar = self._build_top_status_bar()
        root_layout.addLayout(top_bar)

        # 2. Main Horizontal Splitter (Left: Feeds & Workflow, Right: Jog & Teleop)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        left_panel = self._build_left_workflow_panel()
        splitter.addWidget(left_panel)

        right_panel = self._build_right_jog_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        root_layout.addWidget(splitter, stretch=1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background-color: #0b1120; color: #94a3b8; font-size: 11px;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Record golden reference trajectory or select an existing path.")

    # -------------------------------------------------------------------------
    # Top Status Bar
    # -------------------------------------------------------------------------

    def _build_top_status_bar(self) -> QHBoxLayout:
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        # App Title & Phase Badge
        title_box = QHBoxLayout()
        lbl_title = QLabel("🎯 REFERENCE PATH STUDIO")
        lbl_title.setStyleSheet("font-size: 14px; font-weight: 800; color: #38bdf8; letter-spacing: 0.5px;")
        title_box.addWidget(lbl_title)

        lbl_phase = QLabel("PHASE 1: GOLDEN PATH")
        lbl_phase.setStyleSheet("""
            background-color: #0284c7;
            color: #ffffff;
            font-size: 9px;
            font-weight: 800;
            padding: 2px 6px;
            border-radius: 3px;
        """)
        title_box.addWidget(lbl_phase)
        top_bar.addLayout(title_box)

        top_bar.addStretch()

        # UR5e Status Badge
        self.badge_ur5e = QLabel("🤖 UR5e: Connecting...")
        self.badge_ur5e.setStyleSheet("""
            background-color: #1e293b;
            color: #f59e0b;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border: 1px solid #334155;
            border-radius: 4px;
        """)
        top_bar.addWidget(self.badge_ur5e)

        # Gripper Modbus Status Badge
        self.badge_gripper = QLabel("🤏 Gripper: Connecting...")
        self.badge_gripper.setStyleSheet("""
            background-color: #1e293b;
            color: #f59e0b;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border: 1px solid #334155;
            border-radius: 4px;
        """)
        self._update_gripper_badge_status()
        top_bar.addWidget(self.badge_gripper)

        # Gamepad Status Badge
        self.badge_gamepad = QLabel("🎮 Gamepad: No Controller")
        self.badge_gamepad.setStyleSheet("""
            background-color: #1e293b;
            color: #64748b;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border: 1px solid #334155;
            border-radius: 4px;
        """)
        top_bar.addWidget(self.badge_gamepad)

        # Mode Badge (IDLE, RECORDING, PLAYING)
        self.badge_mode = QLabel("MODE: IDLE")
        self.badge_mode.setStyleSheet("""
            background-color: #0b1120;
            color: #94a3b8;
            font-size: 11px;
            font-weight: 800;
            padding: 4px 10px;
            border: 1px solid #334155;
            border-radius: 4px;
        """)
        top_bar.addWidget(self.badge_mode)

        # Gripper Reconnect Button
        self.btn_reconnect_gripper = QPushButton("⟳ Gripper")
        self.btn_reconnect_gripper.setToolTip("Reconnect OnRobot 2FG Compute Box Modbus TCP")
        self.btn_reconnect_gripper.clicked.connect(self.reconnect_gripper)
        top_bar.addWidget(self.btn_reconnect_gripper)

        return top_bar

    # -------------------------------------------------------------------------
    # Left Panel: Cameras, Workflow, HUD, File I/O & Playback
    # -------------------------------------------------------------------------

    def _build_left_workflow_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(8)

        # ---------------------------------------------------------------------
        # Card 1: Dual Camera Live Feed Viewport (Feature 1)
        # ---------------------------------------------------------------------
        box_cam = QGroupBox("VIDEO TELEMETRY VIEWPORT (OPERATOR LIVE FEEDS — DUAL CAMERAS)")
        cam_layout = QVBoxLayout(box_cam)
        cam_layout.setContentsMargins(8, 8, 8, 8)
        cam_layout.setSpacing(6)

        cam_row = QHBoxLayout()
        cam_row.setSpacing(8)

        # Primary Workspace Camera Widget
        self.cam_widget_1 = CameraFeedWidget("Cam 1: Workspace", self.camera_1_index)
        cam_row.addWidget(self.cam_widget_1, stretch=1)

        # Secondary Wrist Flange Camera Widget
        self.cam_widget_2 = CameraFeedWidget("Cam 2: Wrist Flange", self.camera_2_index)
        cam_row.addWidget(self.cam_widget_2, stretch=1)

        cam_layout.addLayout(cam_row)

        # Camera Options Bar (Device Selectors & Resolution)
        cam_ctrl_bar = QHBoxLayout()
        cam_ctrl_bar.setSpacing(8)

        available_cams = getattr(self, "_available_cameras", None) or detect_available_cameras()

        # Cam 1 Selector
        lbl_c1 = QLabel("Cam 1:")
        lbl_c1.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: bold;")
        self.combo_cam1 = QComboBox()
        for idx, label in available_cams:
            self.combo_cam1.addItem(label, idx)
        idx1_pos = self.combo_cam1.findData(self.camera_1_index)
        if idx1_pos >= 0:
            self.combo_cam1.setCurrentIndex(idx1_pos)
        self.combo_cam1.currentIndexChanged.connect(self.on_cam1_selected)
        cam_ctrl_bar.addWidget(lbl_c1)
        cam_ctrl_bar.addWidget(self.combo_cam1)

        # Cam 2 Selector
        lbl_c2 = QLabel("Cam 2:")
        lbl_c2.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: bold;")
        self.combo_cam2 = QComboBox()
        for idx, label in available_cams:
            self.combo_cam2.addItem(label, idx)
        idx2_pos = self.combo_cam2.findData(self.camera_2_index)
        if idx2_pos >= 0:
            self.combo_cam2.setCurrentIndex(idx2_pos)
        self.combo_cam2.currentIndexChanged.connect(self.on_cam2_selected)
        cam_ctrl_bar.addWidget(lbl_c2)
        cam_ctrl_bar.addWidget(self.combo_cam2)

        # Resolution Dropdown
        lbl_res = QLabel("Res:")
        lbl_res.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: bold;")
        self.combo_res = QComboBox()
        self.combo_res.addItem("320 x 240 (Fast)", (320, 240))
        self.combo_res.addItem("640 x 480 (VGA)", (640, 480))
        self.combo_res.addItem("480 x 360 (Medium)", (480, 360))
        self.combo_res.currentIndexChanged.connect(self.on_camera_resolution_changed)
        cam_ctrl_bar.addWidget(lbl_res)
        cam_ctrl_bar.addWidget(self.combo_res)

        cam_ctrl_bar.addStretch()
        cam_hint = QLabel("👁 Live operator views — Not stored in trajectory file")
        cam_hint.setStyleSheet("color: #64748b; font-size: 10px; font-style: italic;")
        cam_ctrl_bar.addWidget(cam_hint)

        cam_layout.addLayout(cam_ctrl_bar)
        layout.addWidget(box_cam)

        # ---------------------------------------------------------------------
        # Card 2: Trajectory Stats & Telemetry HUD (Feature 2: Start Pose Tracking)
        # ---------------------------------------------------------------------
        box_stats = QGroupBox("TRAJECTORY TELEMETRY HUD")
        stats_layout = QVBoxLayout(box_stats)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        stats_layout.setSpacing(6)

        grid_stats = QGridLayout()
        grid_stats.setSpacing(6)

        grid_stats.addWidget(QLabel("Current Trajectory:"), 0, 0)
        self.lbl_path_source = QLabel("No Trajectory Loaded (Buffer Empty)")
        self.lbl_path_source.setStyleSheet("color: #38bdf8; font-weight: bold; font-family: monospace;")
        grid_stats.addWidget(self.lbl_path_source, 0, 1)

        grid_stats.addWidget(QLabel("Waypoints Count:"), 1, 0)
        self.lbl_path_count = QLabel("0 points")
        self.lbl_path_count.setStyleSheet("color: #f1f5f9; font-weight: bold;")
        grid_stats.addWidget(self.lbl_path_count, 1, 1)

        grid_stats.addWidget(QLabel("Total Duration:"), 0, 2)
        self.lbl_path_duration = QLabel("0.00 s")
        self.lbl_path_duration.setStyleSheet("color: #f1f5f9; font-weight: bold;")
        grid_stats.addWidget(self.lbl_path_duration, 0, 3)

        grid_stats.addWidget(QLabel("Sampling Rate:"), 1, 2)
        self.lbl_path_rate = QLabel("30.0 Hz")
        self.lbl_path_rate.setStyleSheet("color: #f1f5f9; font-weight: bold;")
        grid_stats.addWidget(self.lbl_path_rate, 1, 3)

        stats_layout.addLayout(grid_stats)

        # File Start Position Banner (Feature 2)
        pose_init_str = ", ".join(f"{DEFAULT_START_JOINTS_DICT[j]:+.1f}°" for j in UR5E_JOINT_NAMES)
        self.lbl_file_start_pose = QLabel(f"File Start Pose: [{pose_init_str}] | Gripper: {DEFAULT_START_GRIPPER_MM:.1f} mm")
        self.lbl_file_start_pose.setStyleSheet("""
            background-color: #0b1120;
            color: #34d399;
            border: 1px solid #1e293b;
            border-radius: 4px;
            padding: 3px 6px;
            font-family: monospace;
            font-size: 10px;
            font-weight: bold;
        """)
        stats_layout.addWidget(self.lbl_file_start_pose)

        # Live TCP & Gripper Readout Bar
        hud_bar = QHBoxLayout()
        hud_bar.setSpacing(6)

        self.lbl_live_tcp = QLabel("TCP: X=+0.000 Y=+0.000 Z=+0.000 (m)")
        self.lbl_live_tcp.setStyleSheet("""
            background-color: #0b1120;
            color: #38bdf8;
            border: 1px solid #1e293b;
            border-radius: 4px;
            padding: 3px 8px;
            font-family: monospace;
            font-size: 10px;
            font-weight: bold;
        """)
        hud_bar.addWidget(self.lbl_live_tcp, stretch=1)

        self.lbl_live_gripper = QLabel(f"Gripper: {DEFAULT_START_GRIPPER_MM:.1f} mm")
        self.lbl_live_gripper.setStyleSheet("""
            background-color: #0b1120;
            color: #10b981;
            border: 1px solid #1e293b;
            border-radius: 4px;
            padding: 3px 8px;
            font-family: monospace;
            font-size: 10px;
            font-weight: bold;
        """)
        hud_bar.addWidget(self.lbl_live_gripper)
        stats_layout.addLayout(hud_bar)

        # Playback Progress Bar
        self.prog_playback = QProgressBar()
        self.prog_playback.setRange(0, 100)
        self.prog_playback.setValue(0)
        self.prog_playback.setTextVisible(True)
        self.prog_playback.setFormat("Ready for Playback")
        self.prog_playback.setFixedHeight(18)
        stats_layout.addWidget(self.prog_playback)

        layout.addWidget(box_stats)

        # ---------------------------------------------------------------------
        # Card 3: Record Mode (Feature 3: Auto-Return Checkbox)
        # ---------------------------------------------------------------------
        box_rec = QGroupBox("RECORD TRAJECTORY (HUMAN TELEOP VIA GAMEPAD)")
        rec_layout = QVBoxLayout(box_rec)
        rec_layout.setContentsMargins(10, 10, 10, 10)
        rec_layout.setSpacing(8)

        rec_ctrl_row = QHBoxLayout()
        rec_ctrl_row.setSpacing(10)

        # Big Prominent Start/Stop Recording Toggle
        self.btn_record_toggle = QPushButton("● Start Recording Path (Space)")
        self.btn_record_toggle.setFixedHeight(40)
        self.btn_record_toggle.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: #ffffff;
                font-size: 12px;
                font-weight: 800;
                border-radius: 5px;
                border: 1px solid #22c55e;
                padding: 6px 16px;
            }
            QPushButton:hover { background-color: #15803d; }
        """)
        self.btn_record_toggle.clicked.connect(self.toggle_recording)
        rec_ctrl_row.addWidget(self.btn_record_toggle, stretch=2)

        # Recording Rate Dropdown
        col_rate = QVBoxLayout()
        col_rate.setSpacing(2)
        lbl_rate_title = QLabel("Frequency:")
        lbl_rate_title.setStyleSheet("font-size: 10px; color: #94a3b8; font-weight: bold;")
        self.combo_rec_freq = QComboBox()
        self.combo_rec_freq.addItem("30 Hz (Standard)", 30.0)
        self.combo_rec_freq.addItem("50 Hz (High-Res)", 50.0)
        self.combo_rec_freq.addItem("20 Hz (Lightweight)", 20.0)
        self.combo_rec_freq.addItem("10 Hz (Low-Rate)", 10.0)
        col_rate.addWidget(lbl_rate_title)
        col_rate.addWidget(self.combo_rec_freq)
        rec_ctrl_row.addLayout(col_rate)

        # Live Recording Stopwatch
        self.lbl_rec_timer = QLabel("⏱ 00:00.0 (0 pts)")
        self.lbl_rec_timer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_rec_timer.setFixedWidth(140)
        self.lbl_rec_timer.setStyleSheet("""
            background-color: #0b1120;
            color: #94a3b8;
            border: 1px solid #334155;
            border-radius: 4px;
            padding: 6px;
            font-family: monospace;
            font-size: 11px;
            font-weight: bold;
        """)
        rec_ctrl_row.addWidget(self.lbl_rec_timer)
        rec_layout.addLayout(rec_ctrl_row)

        # Workflow Option: Auto-return to Start Position (Feature 3)
        rec_opts_row = QHBoxLayout()
        self.chk_auto_return = QCheckBox("↩ Auto-return to Start Position after recording finishes")
        self.chk_auto_return.setChecked(True)
        self.chk_auto_return.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 11px;")
        self.chk_auto_return.setToolTip("When checked, automatically commands the arm back to the stored Start Position as soon as recording stops.")
        rec_opts_row.addWidget(self.chk_auto_return)
        rec_opts_row.addStretch()
        rec_layout.addLayout(rec_opts_row)

        layout.addWidget(box_rec)

        # ---------------------------------------------------------------------
        # Card 4: Trajectory File I/O & Delete (Feature 4: Delete Path Button)
        # ---------------------------------------------------------------------
        box_io = QGroupBox("TRAJECTORY FILE I/O (reference_paths/)")
        io_layout = QVBoxLayout(box_io)
        io_layout.setContentsMargins(10, 10, 10, 10)
        io_layout.setSpacing(8)

        io_btn_row = QHBoxLayout()
        io_btn_row.setSpacing(8)

        self.btn_save_path = QPushButton("💾 Save Path")
        self.btn_save_path.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #38bdf8;
                padding: 5px 12px;
            }
            QPushButton:hover { background-color: #0369a1; }
        """)
        self.btn_save_path.setToolTip("Saves recorded buffer and start position to 'reference_paths/'")
        self.btn_save_path.clicked.connect(lambda: self.save_recorded_path(show_dialog=False))
        io_btn_row.addWidget(self.btn_save_path)

        self.btn_load_path = QPushButton("📂 Load Path...")
        self.btn_load_path.clicked.connect(self.open_load_path_dialog)
        io_btn_row.addWidget(self.btn_load_path)

        # Delete Selected Path Button (Feature 4)
        self.btn_delete_path = QPushButton("🗑 Delete Selected")
        self.btn_delete_path.setStyleSheet("""
            QPushButton {
                background-color: #7f1d1d;
                color: #fca5a5;
                font-weight: bold;
                border: 1px solid #ef4444;
                padding: 5px 10px;
            }
            QPushButton:hover { background-color: #991b1b; }
        """)
        self.btn_delete_path.setToolTip("Permanently deletes the selected reference path from reference_paths/")
        self.btn_delete_path.clicked.connect(self.delete_selected_path)
        io_btn_row.addWidget(self.btn_delete_path)

        self.btn_refresh_paths = QPushButton("⟳ Refresh")
        self.btn_refresh_paths.clicked.connect(self.refresh_saved_paths_list)
        io_btn_row.addWidget(self.btn_refresh_paths)

        io_layout.addLayout(io_btn_row)

        # Quick Path Selector List
        self.list_saved_paths = QListWidget()
        self.list_saved_paths.setFixedHeight(85)
        self.list_saved_paths.itemDoubleClicked.connect(self.on_path_item_double_clicked)
        self.list_saved_paths.setToolTip("Select a path to load or delete. Double-click to load immediately.")
        io_layout.addWidget(self.list_saved_paths)

        layout.addWidget(box_io)

        # ---------------------------------------------------------------------
        # Card 5: Playback Mode & Emergency Stop (Feature 2: Auto-Align Pre-Check)
        # ---------------------------------------------------------------------
        box_play = QGroupBox("DETERMINISTIC PLAYBACK (servoj / movej)")
        play_layout = QVBoxLayout(box_play)
        play_layout.setContentsMargins(10, 10, 10, 10)
        play_layout.setSpacing(8)

        play_btn_row = QHBoxLayout()
        play_btn_row.setSpacing(10)

        self.btn_play_path = QPushButton("▶ Play Path (Smooth)")
        self.btn_play_path.setFixedHeight(40)
        self.btn_play_path.setStyleSheet("""
            QPushButton {
                background-color: #4f46e5;
                color: #ffffff;
                font-size: 12px;
                font-weight: 800;
                border-radius: 5px;
                border: 1px solid #818cf8;
                padding: 6px 16px;
            }
            QPushButton:hover { background-color: #4338ca; }
            QPushButton:disabled {
                background-color: #1e293b;
                color: #64748b;
                border: 1px solid #334155;
            }
        """)
        self.btn_play_path.clicked.connect(self.start_playback)
        play_btn_row.addWidget(self.btn_play_path, stretch=2)

        # Big Red Safety STOP Button
        self.btn_stop_safety = QPushButton("⏹ STOP (Safety)")
        self.btn_stop_safety.setFixedHeight(40)
        self.btn_stop_safety.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: #ffffff;
                font-size: 12px;
                font-weight: 800;
                border-radius: 5px;
                border: 1px solid #ef4444;
                padding: 6px 16px;
            }
            QPushButton:hover { background-color: #b91c1c; }
        """)
        self.btn_stop_safety.setToolTip("Immediately halts trajectory playback and sends stopj() to UR5e (Esc).")
        self.btn_stop_safety.clicked.connect(self.emergency_stop)
        play_btn_row.addWidget(self.btn_stop_safety, stretch=1)

        play_layout.addLayout(play_btn_row)

        # Playback Settings Row
        play_opt_row = QHBoxLayout()
        play_opt_row.setSpacing(10)

        self.chk_auto_align = QCheckBox("Auto-align to Start Position")
        self.chk_auto_align.setChecked(True)
        self.chk_auto_align.setToolTip(
            "Checks if the robot is currently at the file's start position. If not, smoothly navigates there before waypoint streaming."
        )
        play_opt_row.addWidget(self.chk_auto_align)

        self.chk_smooth_traj = QCheckBox("Smooth Gamepad Tremor")
        self.chk_smooth_traj.setChecked(True)
        self.chk_smooth_traj.setStyleSheet("color: #38bdf8; font-weight: 600;")
        self.chk_smooth_traj.setToolTip(
            "Applies continuous cubic spline interpolation and polynomial noise filtering to eliminate choppy discrete steps."
        )
        play_opt_row.addWidget(self.chk_smooth_traj)

        play_opt_row.addWidget(QLabel("Method:"))
        self.combo_play_method = QComboBox()
        self.combo_play_method.addItem("speedj (Smooth Closed-Loop Velocity) [Recommended]", "speedj")
        self.combo_play_method.addItem("servoj (Spline-Smoothed 100 Hz)", "servoj")
        self.combo_play_method.addItem("movej (Dense Waypoints)", "movej")
        play_opt_row.addWidget(self.combo_play_method)

        play_opt_row.addWidget(QLabel("Speed:"))
        self.slider_play_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_play_speed.setRange(25, 150)
        self.slider_play_speed.setValue(100)
        self.slider_play_speed.setFixedWidth(90)
        self.slider_play_speed.valueChanged.connect(self.on_play_speed_changed)
        play_opt_row.addWidget(self.slider_play_speed)

        self.lbl_play_speed = QLabel("1.0x")
        self.lbl_play_speed.setStyleSheet("color: #38bdf8; font-family: monospace; font-weight: bold;")
        play_opt_row.addWidget(self.lbl_play_speed)

        play_layout.addLayout(play_opt_row)
        layout.addWidget(box_play)

        # ---------------------------------------------------------------------
        # Card 6: Live Activity Log Console
        # ---------------------------------------------------------------------
        box_log = QGroupBox("SYSTEM & MODBUS ACTIVITY CONSOLE")
        log_layout = QVBoxLayout(box_log)
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.setSpacing(4)

        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(400)
        self.txt_log.setFixedHeight(110)
        self.txt_log.setStyleSheet("""
            background-color: #0b1120;
            color: #94a3b8;
            font-family: monospace;
            font-size: 10px;
            border: 1px solid #1e293b;
            border-radius: 4px;
        """)
        log_layout.addWidget(self.txt_log)

        log_ctrl_row = QHBoxLayout()
        log_ctrl_row.addStretch()
        btn_clear_log = QPushButton("Clear Console")
        btn_clear_log.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        btn_clear_log.clicked.connect(self.txt_log.clear)
        log_ctrl_row.addWidget(btn_clear_log)
        log_layout.addLayout(log_ctrl_row)

        layout.addWidget(box_log)

        scroll.setWidget(container)

        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)

        panel = QWidget()
        panel.setLayout(panel_layout)
        return panel

    # -------------------------------------------------------------------------
    # Right Panel: Jogging, Start/Home Poses & Gamepad Controls
    # -------------------------------------------------------------------------

    def _build_right_jog_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)

        # 1. Start & Home Poses Box
        box_poses = QGroupBox("ARM POSES & PRESETS (MATCHING RECORDER)")
        poses_layout = QVBoxLayout(box_poses)
        poses_layout.setContentsMargins(10, 10, 10, 10)
        poses_layout.setSpacing(6)

        header_buttons = QHBoxLayout()
        header_buttons.setSpacing(6)

        # Home Pose Button
        self.btn_move_home = QPushButton("⌂ Home")
        self.btn_move_home.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                font-size: 10px;
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px;
                border: 1px solid #38bdf8;
            }
            QPushButton:hover { background-color: #0369a1; }
        """)
        self.btn_move_home.setToolTip("Command arm to move to default Home Pose [0°, -90°, 90°, -90°, -90°, 0°]")
        self.btn_move_home.clicked.connect(self.move_to_home_pose)
        header_buttons.addWidget(self.btn_move_home)

        # Go to Temp Start Button
        self.btn_go_start = QPushButton("▶ Go to Temp Start")
        self.btn_go_start.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: #ffffff;
                font-size: 10px;
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px;
                border: 1px solid #22c55e;
            }
            QPushButton:hover { background-color: #15803d; }
        """)
        self.btn_go_start.setToolTip("Command arm to move to stored Custom Start Position")
        self.btn_go_start.clicked.connect(self.move_to_start_pose)
        header_buttons.addWidget(self.btn_go_start)

        # Set New Start Button
        self.btn_set_start = QPushButton("💾 Set New Start")
        self.btn_set_start.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #f1f5f9;
                font-size: 10px;
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px;
                border: 1px solid #64748b;
            }
            QPushButton:hover { background-color: #475569; }
        """)
        self.btn_set_start.setToolTip("Save robot's current 6 joint coordinates as the active Start Position")
        self.btn_set_start.clicked.connect(self.set_current_as_start_position)
        header_buttons.addWidget(self.btn_set_start)

        # Move to Target Button
        self.btn_move_target = QPushButton("🎯 Move Target")
        self.btn_move_target.setStyleSheet("""
            QPushButton {
                background-color: #4f46e5;
                color: #ffffff;
                font-size: 10px;
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px;
                border: 1px solid #818cf8;
            }
            QPushButton:hover { background-color: #4338ca; }
        """)
        self.btn_move_target.setToolTip("Command arm to move all joints to the currently specified numeric target values")
        self.btn_move_target.clicked.connect(self.move_to_all_targets)
        header_buttons.addWidget(self.btn_move_target)

        # Sync Button
        self.btn_sync_targets = QPushButton("⟳ Sync")
        self.btn_sync_targets.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #94a3b8;
                font-size: 10px;
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px;
                border: 1px solid #475569;
            }
            QPushButton:hover { background-color: #334155; color: #f1f5f9; }
        """)
        self.btn_sync_targets.setToolTip("Reset all jog input fields to match current robot joint telemetry")
        self.btn_sync_targets.clicked.connect(self.sync_jog_to_robot)
        header_buttons.addWidget(self.btn_sync_targets)

        poses_layout.addLayout(header_buttons)

        # Start Position Display Banner
        self.lbl_start_pose_info = QLabel()
        self.lbl_start_pose_info.setStyleSheet("""
            background-color: #0b1120;
            color: #38bdf8;
            border: 1px solid #1e293b;
            border-radius: 4px;
            padding: 3px 6px;
            font-family: monospace;
            font-size: 10px;
            font-weight: bold;
        """)
        poses_layout.addWidget(self.lbl_start_pose_info)
        self.update_start_pose_display()

        layout.addWidget(box_poses)

        # 2. 6-DoF Joint Jog Sliders & Numeric Inputs
        box_jog = QGroupBox("MANUAL 6-DoF JOINT JOGGING")
        jog_layout = QVBoxLayout(box_jog)
        jog_layout.setContentsMargins(10, 10, 10, 10)
        jog_layout.setSpacing(4)

        grid_jog = QGridLayout()
        grid_jog.setSpacing(4)

        self.jog_sliders: Dict[str, QSlider] = {}
        self.jog_inputs: Dict[str, QLineEdit] = {}
        self.jog_move_buttons: Dict[str, QPushButton] = {}

        for idx, j_name in enumerate(UR5E_JOINT_NAMES):
            name_lbl = QLabel(j_name.replace("_", " ").title())
            name_lbl.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 600;")
            name_lbl.setMinimumWidth(85)

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
                QLineEdit:focus { border: 1px solid #38bdf8; }
            """)

            btn_move = QPushButton("Move")
            btn_move.setFixedWidth(44)
            btn_move.setFixedHeight(22)
            btn_move.setStyleSheet("font-size: 9px; padding: 1px 4px; background-color: #0284c7;")
            btn_move.clicked.connect(lambda _, j=j_name: self.move_single_joint(j))

            self.jog_sliders[j_name] = slider
            self.jog_inputs[j_name] = input_box
            self.jog_move_buttons[j_name] = btn_move

            slider.sliderPressed.connect(lambda j=j_name: self.on_jog_slider_pressed(j))
            slider.valueChanged.connect(lambda val, j=j_name: self.on_jog_slider_value_changed(j, val))
            slider.sliderReleased.connect(lambda j=j_name: self.on_jog_slider_released(j))
            input_box.returnPressed.connect(lambda j=j_name: self.on_jog_input_return(j))

            grid_jog.addWidget(name_lbl, idx, 0)
            grid_jog.addWidget(slider, idx, 1)
            grid_jog.addWidget(input_box, idx, 2)
            grid_jog.addWidget(btn_move, idx, 3)

        jog_layout.addLayout(grid_jog)
        layout.addWidget(box_jog)

        # 3. OnRobot Gripper Aperture Controls (Wide Open = 105.0 mm)
        box_grip = QGroupBox("ONROBOT 2FG14 GRIPPER APERTURE")
        grip_layout = QVBoxLayout(box_grip)
        grip_layout.setContentsMargins(10, 10, 10, 10)
        grip_layout.setSpacing(6)

        grip_row = QHBoxLayout()
        grip_row.setSpacing(6)

        lbl_gw = QLabel("Aperture:")
        lbl_gw.setStyleSheet("color: #10b981; font-weight: bold; font-size: 11px;")
        grip_row.addWidget(lbl_gw)

        self.gripper_slider = QSlider(Qt.Orientation.Horizontal)
        self.gripper_slider.setRange(55, 105)
        self.gripper_slider.setValue(int(round(self.start_gripper_mm)))
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
        self.gripper_slider.sliderReleased.connect(self.on_gripper_slider_released)
        grip_row.addWidget(self.gripper_slider, stretch=1)

        self.gripper_input = QLineEdit(f"{self.start_gripper_mm:.1f}")
        self.gripper_input.setFixedWidth(54)
        self.gripper_input.setFixedHeight(22)
        self.gripper_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.gripper_input.returnPressed.connect(self.on_gripper_input_return)
        grip_row.addWidget(self.gripper_input)

        self.btn_move_gripper = QPushButton("Move")
        self.btn_move_gripper.setFixedWidth(44)
        self.btn_move_gripper.setFixedHeight(22)
        self.btn_move_gripper.setStyleSheet("font-size: 9px; padding: 1px 4px; background-color: #059669;")
        self.btn_move_gripper.clicked.connect(self.on_gripper_input_return)
        grip_row.addWidget(self.btn_move_gripper)

        grip_layout.addLayout(grip_row)

        # Quick Gripper Preset Buttons
        grip_presets = QHBoxLayout()
        grip_presets.setSpacing(6)

        btn_open = QPushButton("Open (105mm)")
        btn_open.clicked.connect(lambda: self.command_gripper_width(105.0))
        btn_close = QPushButton("Close (55mm)")
        btn_close.clicked.connect(lambda: self.command_gripper_width(55.0))
        btn_mid = QPushButton("Mid (80mm)")
        btn_mid.clicked.connect(lambda: self.command_gripper_width(80.0))

        grip_presets.addWidget(btn_close)
        grip_presets.addWidget(btn_mid)
        grip_presets.addWidget(btn_open)
        grip_layout.addLayout(grip_presets)

        layout.addWidget(box_grip)

        # 4. Gamepad Teleoperation Controls (Feature 3: Analog Stick Inversions)
        box_gamepad = QGroupBox("HARDWARE GAMEPAD TELEOPERATION CONTROLS")
        gamepad_layout = QVBoxLayout(box_gamepad)
        gamepad_layout.setContentsMargins(10, 10, 10, 10)
        gamepad_layout.setSpacing(8)

        # Row 1: Enable Toggle, Velocity Readout, Gripper Hint
        gp_row1 = QHBoxLayout()
        gp_row1.setSpacing(8)

        self.btn_toggle_gamepad = QPushButton("🎮 Enable Gamepad Teleop")
        self.btn_toggle_gamepad.setCheckable(True)
        self.btn_toggle_gamepad.setFixedHeight(30)
        self.btn_toggle_gamepad.setToolTip(
            "Toggle real-time Cartesian teleoperation & direct joint control via gamepad.\n"
            "• Left Stick: Tool-Gaze Forward/Backward (gripper pointing direction) & Strafe Left/Right\n"
            "• Right Stick (Y): TCP Vertical Up/Down (+Z/-Z)\n"
            "• Right Stick (X): Direct Base Motor Rotation (shoulder_pan)\n"
            "• Shoulders (LB/RB): Direct Wrist 2 Rotation\n"
            "• Triggers (LT/RT): Direct Wrist 1 Rotation\n"
            "• [A]/[B]: Close/Open Gripper"
        )
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
            QPushButton:checked {
                background-color: #0284c7;
                color: #ffffff;
                border: 1px solid #38bdf8;
            }
        """)
        self.btn_toggle_gamepad.toggled.connect(self.on_gamepad_toggle_clicked)
        gp_row1.addWidget(self.btn_toggle_gamepad)

        self.lbl_gamepad_vel = QLabel("TCP: 0.00 m/s")
        self.lbl_gamepad_vel.setStyleSheet("""
            color: #38bdf8;
            font-family: monospace;
            font-size: 10px;
            font-weight: bold;
            padding: 3px 6px;
            background-color: #0b1120;
            border: 1px solid #1e293b;
            border-radius: 4px;
        """)
        gp_row1.addWidget(self.lbl_gamepad_vel)

        lbl_gp_grip = QLabel("🤏 [A] Close | [B] Open")
        lbl_gp_grip.setStyleSheet("""
            color: #10b981;
            font-family: monospace;
            font-size: 10px;
            font-weight: bold;
            padding: 3px 6px;
            background-color: #064e3b;
            border: 1px solid #059669;
            border-radius: 4px;
        """)
        gp_row1.addWidget(lbl_gp_grip)
        gp_row1.addStretch()
        gamepad_layout.addLayout(gp_row1)

        # Row 2: Inversion Checkboxes for Both Analog Sticks (Feature 3)
        gp_row2 = QHBoxLayout()
        self.chk_invert_left = QCheckBox("⇄ Invert Left Stick (L/R)")
        self.chk_invert_left.setChecked(False)
        self.chk_invert_left.setToolTip("Invert Left Stick lateral translation direction (Left/Right)")
        self.chk_invert_left.toggled.connect(self.on_invert_left_toggled)
        gp_row2.addWidget(self.chk_invert_left)

        self.chk_invert_right = QCheckBox("⇄ Invert Right Stick (Base)")
        self.chk_invert_right.setChecked(False)
        self.chk_invert_right.setToolTip("Invert Right Stick horizontal base rotation direction (shoulder_pan)")
        self.chk_invert_right.toggled.connect(self.on_invert_right_toggled)
        gp_row2.addWidget(self.chk_invert_right)
        gp_row2.addStretch()
        gamepad_layout.addLayout(gp_row2)

        # Row 3: Master Speed and Accel Sliders
        gp_row3 = QHBoxLayout()
        gp_row3.setSpacing(8)

        self.lbl_gp_speed = QLabel("⚡ Speed: 50%")
        self.lbl_gp_speed.setStyleSheet("color: #38bdf8; font-size: 10px; font-weight: bold;")
        self.slider_gamepad_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_gamepad_speed.setRange(10, 100)
        self.slider_gamepad_speed.setValue(50)
        self.slider_gamepad_speed.setFixedWidth(90)
        self.slider_gamepad_speed.valueChanged.connect(self.on_gamepad_speed_changed)
        gp_row3.addWidget(self.lbl_gp_speed)
        gp_row3.addWidget(self.slider_gamepad_speed)

        gp_row3.addSpacing(6)

        self.lbl_gp_accel = QLabel("🚀 Accel: 50%")
        self.lbl_gp_accel.setStyleSheet("color: #4ade80; font-size: 10px; font-weight: bold;")
        self.slider_gamepad_accel = QSlider(Qt.Orientation.Horizontal)
        self.slider_gamepad_accel.setRange(10, 100)
        self.slider_gamepad_accel.setValue(50)
        self.slider_gamepad_accel.setFixedWidth(90)
        self.slider_gamepad_accel.valueChanged.connect(self.on_gamepad_accel_changed)
        gp_row3.addWidget(self.lbl_gp_accel)
        gp_row3.addWidget(self.slider_gamepad_accel)

        gp_row3.addStretch()
        gamepad_layout.addLayout(gp_row3)

        # Row 4: Gamepad Button Shortcuts Hints
        lbl_shortcuts = QLabel("🎮 Shortcuts: [Select] Go to Temp Start | [Start] Toggle Record (Space)")
        lbl_shortcuts.setStyleSheet("""
            color: #e2e8f0;
            background-color: #0b1120;
            border: 1px solid #334155;
            border-radius: 4px;
            padding: 3px 6px;
            font-size: 9px;
            font-weight: 600;
        """)
        gamepad_layout.addWidget(lbl_shortcuts)

        layout.addWidget(box_gamepad)
        layout.addStretch()

        scroll.setWidget(container)

        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)

        panel = QWidget()
        panel.setLayout(panel_layout)
        return panel

    # -------------------------------------------------------------------------
    # Camera Event Slots
    # -------------------------------------------------------------------------

    def on_cam1_selected(self, index: int):
        cam_id = self.combo_cam1.currentData()
        if cam_id is not None and hasattr(self, "camera_worker_1"):
            self.camera_1_index = int(cam_id)
            self.cam_widget_1.camera_id = self.camera_1_index
            self.cam_widget_1.lbl_title.setText(f"<b>Cam 1: /dev/video{self.camera_1_index}</b>")
            self.camera_worker_1.set_camera_id(self.camera_1_index)
            self.log(f"Camera 1 switched to device /dev/video{self.camera_1_index}")

    def on_cam2_selected(self, index: int):
        cam_id = self.combo_cam2.currentData()
        if cam_id is not None and hasattr(self, "camera_worker_2"):
            self.camera_2_index = int(cam_id)
            self.cam_widget_2.camera_id = self.camera_2_index
            self.cam_widget_2.lbl_title.setText(f"<b>Cam 2: /dev/video{self.camera_2_index}</b>")
            self.camera_worker_2.set_camera_id(self.camera_2_index)
            self.log(f"Camera 2 switched to device /dev/video{self.camera_2_index}")

    def on_camera_resolution_changed(self, index: int):
        res = self.combo_res.currentData()
        if res and hasattr(self, "camera_worker_1") and hasattr(self, "camera_worker_2"):
            w, h = res
            self.camera_worker_1.set_resolution(w, h)
            self.camera_worker_2.set_resolution(w, h)
            self.log(f"Camera operator viewports resized to {w}x{h}")

    # -------------------------------------------------------------------------
    # Trajectory Recording Slots & Workflow (With Auto-Return)
    # -------------------------------------------------------------------------

    def toggle_recording(self):
        """Toggles trajectory recording on or off."""
        if self.path_manager.is_playing:
            self.show_warning("Cannot Record", "Cannot record while playback is active. Stop playback first.")
            return

        if not self.path_manager.is_recording:
            # Synchronize start position into path manager before recording
            self.path_manager.set_start_position(self.start_position, gripper_width=self.gripper_slider.value())

            # Start Recording
            freq = float(self.combo_rec_freq.currentData() or 30.0)
            success = self.path_manager.start_recording(
                frequency_hz=freq,
                callback=self._on_point_recorded,
            )
            if success:
                self._recording_start_time = time.perf_counter()
                self.recording_timer.start()

                self.btn_record_toggle.setText("⏹ Stop Recording Path (Space)")
                self.btn_record_toggle.setStyleSheet("""
                    QPushButton {
                        background-color: #dc2626;
                        color: #ffffff;
                        font-size: 12px;
                        font-weight: 800;
                        border-radius: 5px;
                        border: 1px solid #ef4444;
                        padding: 6px 16px;
                    }
                    QPushButton:hover { background-color: #b91c1c; }
                """)
                self.badge_mode.setText(f"MODE: RECORDING ({freq:.0f} Hz)")
                self.badge_mode.setStyleSheet("""
                    background-color: #7f1d1d;
                    color: #fca5a5;
                    font-size: 11px;
                    font-weight: 800;
                    padding: 4px 10px;
                    border: 1px solid #ef4444;
                    border-radius: 4px;
                """)
                self.combo_rec_freq.setEnabled(False)
                self.btn_play_path.setEnabled(False)
                self.btn_save_path.setEnabled(False)
                self.btn_load_path.setEnabled(False)
                self.btn_delete_path.setEnabled(False)

                # Auto-enable Gamepad Teleoperation if available
                if self.gamepad_teleop and not self.gamepad_teleop.is_enabled():
                    self.btn_toggle_gamepad.setChecked(True)

                self.status_bar.showMessage("Recording trajectory... Drive UR5e with gamepad.")
                self.log(f"Started trajectory recording at {freq:.1f} Hz.")
        else:
            # Stop Recording
            points = self.path_manager.stop_recording()
            self.recording_timer.stop()

            self.btn_record_toggle.setText("● Start Recording Path (Space)")
            self.btn_record_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #16a34a;
                    color: #ffffff;
                    font-size: 12px;
                    font-weight: 800;
                    border-radius: 5px;
                    border: 1px solid #22c55e;
                    padding: 6px 16px;
                }
                QPushButton:hover { background-color: #15803d; }
            """)
            self.badge_mode.setText("MODE: IDLE")
            self.badge_mode.setStyleSheet("""
                background-color: #0b1120;
                color: #94a3b8;
                font-size: 11px;
                font-weight: 800;
                padding: 4px 10px;
                border: 1px solid #334155;
                border-radius: 4px;
            """)
            self.combo_rec_freq.setEnabled(True)
            self.btn_save_path.setEnabled(True)
            self.btn_load_path.setEnabled(True)
            self.btn_delete_path.setEnabled(True)
            self.btn_play_path.setEnabled(len(points) > 0)

            dur = self.path_manager.duration_s
            self.lbl_path_source.setText("[New Unsaved Trajectory Buffer]")
            self.lbl_path_count.setText(f"{len(points)} points")
            self.lbl_path_duration.setText(f"{dur:.2f} s")
            self.status_bar.showMessage(f"Recording stopped. Captured {len(points)} waypoints ({dur:.2f}s).", 5000)
            self.log(f"Recording stopped. Total: {len(points)} points ({dur:.2f}s). Ready to save or play.")

            # Feature 3: Auto-return to Start Position if checked
            if self.chk_auto_return.isChecked():
                self.log("[WORKFLOW] Auto-return active: commanding arm back to start position.")
                self.status_bar.showMessage("Auto-returning arm to Start Position...", 3000)
                self.move_to_start_pose()

    def _on_point_recorded(self, point: TrajectoryPoint, count: int):
        pass

    def update_recording_stopwatch(self):
        """Updates the recording stopwatch HUD at 20 Hz."""
        if not self.path_manager.is_recording:
            return
        elapsed = time.perf_counter() - self._recording_start_time
        pts = self.path_manager.point_count
        mins = int(elapsed // 60)
        secs = elapsed % 60
        self.lbl_rec_timer.setText(f"⏱ {mins:02d}:{secs:04.1f} ({pts} pts)")

    # -------------------------------------------------------------------------
    # Save, Load & Delete Slots
    # -------------------------------------------------------------------------

    def save_recorded_path(self, show_dialog: bool = False):
        """Saves active trajectory buffer and start position to reference_paths/."""
        if self.path_manager.point_count == 0:
            self.show_warning("Empty Buffer", "No trajectory points in buffer to save. Record or load a path first.")
            return

        # Ensure start position and start gripper width are embedded
        self.path_manager.set_start_position(self.start_position, gripper_width=self.gripper_slider.value())

        try:
            saved_path = self.path_manager.save_path(file_format="csv")
            self.lbl_path_source.setText(saved_path.name)
            self.refresh_saved_paths_list()
            self.status_bar.showMessage(f"Trajectory saved: {saved_path.name}", 5000)
            self.log(f"Trajectory saved successfully: {saved_path}")
            if show_dialog:
                QMessageBox.information(
                    self,
                    "Trajectory Saved",
                    f"Successfully saved {self.path_manager.point_count} points to:\n\n{saved_path}",
                )
            return saved_path
        except Exception as e:
            self.show_error("Save Error", f"Failed to save reference trajectory:\n{e}")
            self.log(f"[ERROR] Failed to save path: {e}")

    def open_load_path_dialog(self):
        """Opens a file dialog targeting reference_paths/ to load a trajectory."""
        if self.path_manager.is_playing:
            self.show_warning("Playback Active", "Cannot load path while playback is running.")
            return

        target_dir = str(self.path_manager.paths_dir)
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Reference Path",
            target_dir,
            "Trajectory Files (*.csv *.json);;CSV Files (*.csv);;JSON Files (*.json);;All Files (*)",
        )
        if filepath:
            self.load_path_file(Path(filepath))

    def load_path_file(self, path: Path):
        """Loads trajectory from disk into memory and retrieves start position."""
        try:
            points = self.path_manager.load_path(path)
            dur = self.path_manager.duration_s
            rate = len(points) / dur if dur > 0 else 30.0

            self.lbl_path_source.setText(path.name)
            self.lbl_path_count.setText(f"{len(points)} points")
            self.lbl_path_duration.setText(f"{dur:.2f} s")
            self.lbl_path_rate.setText(f"{rate:.1f} Hz")
            self.btn_play_path.setEnabled(True)

            # Update Start Pose readout from file metadata (Feature 2)
            sp_deg = self.path_manager.get_start_position_deg()
            gw = self.path_manager.get_start_gripper_width()
            if sp_deg is not None and len(sp_deg) == 6:
                pose_str = ", ".join(f"{v:+.1f}°" for v in sp_deg)
                self.lbl_file_start_pose.setText(f"File Start Pose: [{pose_str}] | Gripper: {gw:.1f} mm")

            self.prog_playback.setValue(0)
            self.prog_playback.setFormat(f"Loaded {len(points)} pts ({dur:.1f}s) — Ready")

            self.status_bar.showMessage(f"Loaded {path.name} ({len(points)} pts, {dur:.1f}s)", 5000)
            self.log(f"Loaded reference path: {path.name} ({len(points)} points, {dur:.2f}s, {rate:.1f} Hz)")
        except Exception as e:
            self.show_error("Load Error", f"Failed to load trajectory file '{path.name}':\n{e}")
            self.log(f"[ERROR] Failed to load path {path.name}: {e}")

    def delete_selected_path(self):
        """Deletes the selected trajectory file from disk after user confirmation (Feature 4)."""
        item = self.list_saved_paths.currentItem()
        if not item:
            self.show_warning("No Selection", "Please select a reference path file from the list to delete.")
            return

        path_str = item.data(Qt.ItemDataRole.UserRole)
        if not path_str:
            return

        target_path = Path(path_str)
        reply = QMessageBox.question(
            self,
            "Confirm Delete Path",
            f"Are you sure you want to permanently delete reference path:\n\n'{target_path.name}'?\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.path_manager.delete_path(target_path)
                self.refresh_saved_paths_list()

                if self.lbl_path_source.text() == target_path.name:
                    self.lbl_path_source.setText("No Trajectory Loaded (Buffer Empty)")
                    self.lbl_path_count.setText("0 points")
                    self.lbl_path_duration.setText("0.00 s")
                    self.btn_play_path.setEnabled(False)
                    self.prog_playback.setValue(0)
                    self.prog_playback.setFormat("Buffer Cleared")

                self.status_bar.showMessage(f"Deleted {target_path.name}", 4000)
                self.log(f"[FILE] Deleted reference path: {target_path.name}")
            except Exception as e:
                self.show_error("Delete Error", f"Failed to delete path:\n{e}")
                self.log(f"[ERROR] Failed to delete path: {e}")

    def refresh_saved_paths_list(self):
        """Rescans reference_paths/ and populates quick-access list widget."""
        self.list_saved_paths.clear()
        paths = self.path_manager.list_saved_paths()

        if not paths:
            item = QListWidgetItem("No reference paths found in reference_paths/")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list_saved_paths.addItem(item)
            return

        for p in paths:
            sz_kb = p.stat().st_size / 1024.0
            mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            item = QListWidgetItem(f"📄 {p.name:<32} ({sz_kb:5.1f} KB | {mtime})")
            item.setData(Qt.ItemDataRole.UserRole, str(p.resolve()))
            self.list_saved_paths.addItem(item)

    def on_path_item_double_clicked(self, item: QListWidgetItem):
        path_str = item.data(Qt.ItemDataRole.UserRole)
        if path_str:
            self.load_path_file(Path(path_str))

    # -------------------------------------------------------------------------
    # Playback & Emergency Stop Slots (Feature 2: Start Position Pre-Check & Auto-Move)
    # -------------------------------------------------------------------------

    def start_playback(self):
        """Starts asynchronous trajectory playback on the UR5e."""
        if self.path_manager.point_count == 0:
            self.show_warning("No Path Loaded", "Please record or load a reference path before starting playback.")
            return

        if self.path_manager.is_recording:
            self.show_warning("Recording Active", "Cannot start playback while recording.")
            return

        # Inhibit gamepad during playback
        if self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.gamepad_teleop.set_inhibited(True, "Deterministic playback in progress")

        method = str(self.combo_play_method.currentData() or "speedj")
        speed_scale = self.slider_play_speed.value() / 100.0
        auto_align = self.chk_auto_align.isChecked()
        smooth = self.chk_smooth_traj.isChecked()

        self.btn_play_path.setEnabled(False)
        self.btn_record_toggle.setEnabled(False)
        self.btn_save_path.setEnabled(False)
        self.btn_load_path.setEnabled(False)
        self.btn_delete_path.setEnabled(False)

        self.badge_mode.setText(f"MODE: PLAYING ({method})")
        self.badge_mode.setStyleSheet("""
            background-color: #431407;
            color: #fdba74;
            font-size: 11px;
            font-weight: 800;
            padding: 4px 10px;
            border: 1px solid #f97316;
            border-radius: 4px;
        """)

        self.prog_playback.setValue(0)
        self.prog_playback.setFormat("Checking Start Position & Replaying...")
        self.status_bar.showMessage(f"Executing smooth deterministic playback ({method})...")
        self.log(f"Starting playback: method={method}, speed={speed_scale:.2f}x, auto_align={auto_align}, smooth={smooth}")

        self.path_manager.play_path_async(
            method=method,
            speed_scale=speed_scale,
            auto_align_start=auto_align,
            smooth=smooth,
            progress_callback=self._on_playback_progress,
            completion_callback=self._on_playback_completed,
        )

    def emergency_stop(self):
        """Safety halt triggered by red STOP button or Esc key."""
        self.log("[SAFETY] Emergency STOP triggered!")
        self.status_bar.showMessage("EMERGENCY STOP EXECUTED.", 5000)

        # 1. Stop playback
        self.path_manager.stop_playback()

        # 2. Stop robot worker discrete motions
        if hasattr(self, "robot_worker") and self.robot_worker:
            if hasattr(self.robot_worker, "_robot") and self.robot_worker._robot:
                if hasattr(self.robot_worker._robot, "sock") and self.robot_worker._robot.sock:
                    try:
                        self.robot_worker._robot.sock.sendall(b"stopj(2.5)\n")
                    except Exception:
                        pass

        # 3. Stop gamepad teleop motion
        if self.gamepad_teleop:
            self.gamepad_teleop.emergency_stop()

        # 4. Stop gripper
        if self.gripper:
            try:
                self.gripper.stop()
            except Exception:
                pass

        self.badge_mode.setText("MODE: STOPPED")
        self.badge_mode.setStyleSheet("""
            background-color: #7f1d1d;
            color: #fca5a5;
            font-size: 11px;
            font-weight: 800;
            padding: 4px 10px;
            border: 1px solid #ef4444;
            border-radius: 4px;
        """)
        self._reset_playback_ui()

    def _on_playback_progress(self, idx: int, total: int, point: TrajectoryPoint):
        now = time.perf_counter()
        if now - self._last_progress_update_time < 0.05 and idx < total - 1:
            return
        self._last_progress_update_time = now

        pct = int(round((idx / max(1, total - 1)) * 100))
        QTimer.singleShot(0, lambda: self._update_playback_hud(pct, idx, total, point))

    def _update_playback_hud(self, pct: int, idx: int, total: int, point: TrajectoryPoint):
        self.prog_playback.setValue(pct)
        self.prog_playback.setFormat(f"Playing: {pct}% ({idx}/{total} waypoints)")

    def _on_playback_completed(self, success: bool, message: str):
        QTimer.singleShot(0, lambda: self._handle_playback_finished(success, message))

    def _handle_playback_finished(self, success: bool, message: str):
        self._reset_playback_ui()
        if success:
            self.prog_playback.setValue(100)
            self.prog_playback.setFormat("Playback Completed (100%)")
            self.status_bar.showMessage("Trajectory playback finished successfully.", 5000)
            self.log("[PLAYBACK] Trajectory execution finished successfully.")
        else:
            self.prog_playback.setFormat(f"Playback Interrupted: {message}")
            self.status_bar.showMessage(f"Playback stopped: {message}", 5000)
            self.log(f"[PLAYBACK] {message}")

        if self.gamepad_teleop:
            self.gamepad_teleop.set_inhibited(False)

    def _reset_playback_ui(self):
        self.btn_play_path.setEnabled(self.path_manager.point_count > 0)
        self.btn_record_toggle.setEnabled(True)
        self.btn_save_path.setEnabled(True)
        self.btn_load_path.setEnabled(True)
        self.btn_delete_path.setEnabled(True)

        if not self.path_manager.is_recording:
            self.badge_mode.setText("MODE: IDLE")
            self.badge_mode.setStyleSheet("""
                background-color: #0b1120;
                color: #94a3b8;
                font-size: 11px;
                font-weight: 800;
                padding: 4px 10px;
                border: 1px solid #334155;
                border-radius: 4px;
            """)

    def on_play_speed_changed(self, val: int):
        speed = val / 100.0
        self.lbl_play_speed.setText(f"{speed:.2f}x")

    # -------------------------------------------------------------------------
    # Manual Jogging & Pose Presets (Matching gui_recorder.py)
    # -------------------------------------------------------------------------

    def update_start_pose_display(self):
        pose_str = ", ".join(f"{self.start_position.get(j, 0.0):+.1f}°" for j in UR5E_JOINT_NAMES)
        self.lbl_start_pose_info.setText(f"Start Pose: [{pose_str}] | Grip: {self.start_gripper_mm:.1f}mm")

    def set_current_as_start_position(self):
        """Captures current arm position as the active Start Position."""
        current_rads = self.robot_worker.get_latest_joints()
        for j in UR5E_JOINT_NAMES:
            self.start_position[j] = round(math.degrees(current_rads.get(j, 0.0)), 1)
        self.start_gripper_mm = float(self.gripper_slider.value())

        self.update_start_pose_display()
        self.path_manager.set_start_position(self.start_position, gripper_width=self.start_gripper_mm)
        if persist_start_position is not None:
            persist_start_position(self.start_position, self.start_gripper_mm)
            self.log("[CONFIG] Persisted new start position to default_start_position.json")
        pose_str = ", ".join(f"{self.start_position[j]:+.1f}°" for j in UR5E_JOINT_NAMES)
        self.lbl_file_start_pose.setText(f"Active Start: [{pose_str}] | Grip: {self.start_gripper_mm:.1f}mm")
        self.log(f"Active Start Position saved: [{pose_str}] | Gripper: {self.start_gripper_mm:.1f}mm")
        self.status_bar.showMessage(f"Active Start Position saved: [{pose_str}]", 4000)

    def move_to_start_pose(self):
        """Moves arm to stored Custom Start Position and sets gripper to start width."""
        if self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.gamepad_teleop.set_inhibited(True, "Moving to Start Pose")

        action = {f"{j}.pos": math.radians(self.start_position.get(j, 0.0)) for j in UR5E_JOINT_NAMES}
        pose_str = ", ".join(f"{self.start_position.get(j, 0.0):+.1f}°" for j in UR5E_JOINT_NAMES)
        self.log(f"Commanding arm to Start Position [{pose_str}] and Gripper to {self.start_gripper_mm:.1f} mm...")
        self.status_bar.showMessage("Moving arm to Start Position...", 4000)
        self.command_gripper_width(self.start_gripper_mm)
        self.robot_worker.request_move_pose(action)

    def move_to_home_pose(self):
        """Moves arm to canonical Home Pose."""
        if self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.gamepad_teleop.set_inhibited(True, "Moving to Home Pose")

        home_action = {f"{j}.pos": math.radians(HOME_JOINTS_DICT[j]) for j in UR5E_JOINT_NAMES}
        self.log("Commanding arm to Home position (0°, -90°, 90°, -90°, -90°, 0°)...")
        self.status_bar.showMessage("Moving arm to Home position...", 4000)
        self.robot_worker.request_move_pose(home_action)

    def move_to_all_targets(self):
        """Moves arm to all entered jog coordinates."""
        if self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.gamepad_teleop.set_inhibited(True, "Moving to target joints")

        target_rads = {}
        for j in UR5E_JOINT_NAMES:
            val_deg = float(self.jog_sliders[j].value())
            if j in self.jog_inputs:
                try:
                    val_deg = float(self.jog_inputs[j].text().replace("°", "").strip())
                except ValueError:
                    pass
            target_rads[f"{j}.pos"] = math.radians(val_deg)

        self.log("Commanding arm to all specified target joint positions...")
        self.status_bar.showMessage("Moving arm to target positions...", 4000)
        self.robot_worker.request_move_pose(target_rads)

    def move_single_joint(self, j_name: str):
        """Moves a single joint to typed target value."""
        if self.gamepad_teleop and self.gamepad_teleop.is_enabled():
            self.gamepad_teleop.set_inhibited(True, f"Moving {j_name}")

        try:
            val_deg = float(self.jog_inputs[j_name].text().replace("°", "").strip())
        except ValueError:
            val_deg = float(self.jog_sliders[j_name].value())

        current_joints = self.robot_worker.get_latest_joints()
        action = {f"{j}.pos": current_joints.get(j, 0.0) for j in UR5E_JOINT_NAMES}
        action[f"{j_name}.pos"] = math.radians(val_deg)

        self.log(f"Jogging {j_name} to {val_deg:+.1f}°...")
        self.status_bar.showMessage(f"Jogging {j_name} to {val_deg:+.1f}°...", 3000)
        self.robot_worker.request_move_pose(action)

    def sync_jog_to_robot(self):
        """Snaps all jog sliders and input fields to live telemetry."""
        current_rads = self.robot_worker.get_latest_joints()
        for j in UR5E_JOINT_NAMES:
            deg_val = math.degrees(current_rads.get(j, 0.0))
            if j in self.jog_inputs:
                self.jog_inputs[j].setText(f"{deg_val:+.1f}")
                self.jog_inputs[j].clearFocus()
            if j in self.jog_sliders:
                slider = self.jog_sliders[j]
                slider.blockSignals(True)
                slider.setValue(int(round(deg_val)))
                slider.blockSignals(False)
        self.status_bar.showMessage("Jogging inputs synchronized with robot.", 3000)

    def on_jog_slider_pressed(self, j_name: str):
        self._user_dragging_joint = j_name

    def on_jog_slider_value_changed(self, j_name: str, val: int):
        if self._user_dragging_joint == j_name and j_name in self.jog_inputs:
            self.jog_inputs[j_name].setText(f"{val:+.1f}")

    def on_jog_slider_released(self, j_name: str):
        self._user_dragging_joint = None
        self.move_single_joint(j_name)

    def on_jog_input_return(self, j_name: str):
        self.move_single_joint(j_name)

    def on_discrete_motion_finished(self, success: bool, message: str):
        level = "MOTION" if success else "ERROR"
        self.log(f"[{level}] {message}")
        self.status_bar.showMessage(message, 3000)
        if self.gamepad_teleop:
            self.gamepad_teleop.set_inhibited(False)

    # -------------------------------------------------------------------------
    # Gripper Control Slots
    # -------------------------------------------------------------------------

    def command_gripper_width(self, width_mm: float):
        """Commands opening width to OnRobot gripper."""
        width_clamped = max(55.0, min(105.0, float(width_mm)))
        self.gripper_slider.blockSignals(True)
        self.gripper_slider.setValue(int(round(width_clamped)))
        self.gripper_slider.blockSignals(False)
        self.gripper_input.setText(f"{width_clamped:.1f}")

        if self.gripper is not None and self.gripper.is_connected:
            try:
                self.gripper.grip(width_clamped, wait=False)
                self.log(f"[GRIPPER] Commanded width: {width_clamped:.1f} mm")
            except Exception as e:
                self.log(f"[ERROR] Gripper command failed: {e}")
        elif self.mock_mode:
            self.log(f"[MOCK GRIPPER] Set width: {width_clamped:.1f} mm")
            self.lbl_live_gripper.setText(f"Gripper: {width_clamped:.1f} mm")

    def on_gripper_slider_released(self):
        val = float(self.gripper_slider.value())
        self.command_gripper_width(val)

    def on_gripper_input_return(self):
        try:
            val = float(self.gripper_input.text().strip())
            self.command_gripper_width(val)
        except ValueError:
            pass

    def reconnect_gripper(self):
        """Attempts to reconnect OnRobot Compute Box."""
        self.log("Reconnecting OnRobot 2FG Gripper...")
        success = self.path_manager.connect_gripper()
        if success:
            self.gripper = self.path_manager._gripper
            if self.gamepad_teleop:
                self.gamepad_teleop.set_gripper(self.gripper)
            self._update_gripper_badge_status()
            self.log("OnRobot Gripper reconnected successfully.")
        else:
            self.badge_gripper.setText("🤏 Gripper: Offline")
            self.badge_gripper.setStyleSheet("""
                background-color: #1e293b;
                color: #ef4444;
                font-size: 11px;
                font-weight: 700;
                padding: 4px 10px;
                border: 1px solid #dc2626;
                border-radius: 4px;
            """)
            self.log("[ERROR] Gripper reconnection failed.")

    def _update_gripper_badge_status(self):
        if self.gripper and self.gripper.is_connected:
            w = self.gripper.get_width()
            self.badge_gripper.setText(f"🤏 Gripper: {w:.1f}mm")
            self.badge_gripper.setStyleSheet("""
                background-color: #064e3b;
                color: #34d399;
                font-size: 11px;
                font-weight: 700;
                padding: 4px 10px;
                border: 1px solid #059669;
                border-radius: 4px;
            """)
        else:
            self.badge_gripper.setText("🤏 Gripper: Disconnected")
            self.badge_gripper.setStyleSheet("""
                background-color: #1e293b;
                color: #94a3b8;
                font-size: 11px;
                font-weight: 700;
                padding: 4px 10px;
                border: 1px solid #334155;
                border-radius: 4px;
            """)

    # -------------------------------------------------------------------------
    # Gamepad Slots & Integration (Feature 3: Analog Stick Inversions)
    # -------------------------------------------------------------------------

    def on_gamepad_toggle_clicked(self, checked: bool):
        if self.gamepad_teleop:
            self.gamepad_teleop.set_enabled(checked)
            self.log(f"Gamepad teleoperation {'ENABLED' if checked else 'DISABLED'}.")

    def on_gamepad_status_updated(self, is_connected: bool, name: str):
        if is_connected:
            self.badge_gamepad.setText(f"🎮 {name}")
            self.badge_gamepad.setStyleSheet("""
                background-color: #0c4a6e;
                color: #38bdf8;
                font-size: 11px;
                font-weight: 700;
                padding: 4px 10px;
                border: 1px solid #0284c7;
                border-radius: 4px;
            """)
        else:
            self.badge_gamepad.setText("🎮 No Gamepad Detected")
            self.badge_gamepad.setStyleSheet("""
                background-color: #1e293b;
                color: #64748b;
                font-size: 11px;
                font-weight: 700;
                padding: 4px 10px;
                border: 1px solid #334155;
                border-radius: 4px;
            """)

    def on_gamepad_velocity_updated(self, v: List[float]):
        tcp_speed = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        self.lbl_gamepad_vel.setText(f"TCP: {tcp_speed:.2f} m/s")

    def on_gamepad_state_changed(self, is_active: bool, message: str):
        if is_active:
            self.lbl_gamepad_vel.setStyleSheet("""
                color: #10b981;
                font-family: monospace;
                font-size: 10px;
                font-weight: bold;
                padding: 3px 6px;
                background-color: #064e3b;
                border: 1px solid #059669;
                border-radius: 4px;
            """)
        else:
            self.lbl_gamepad_vel.setStyleSheet("""
                color: #38bdf8;
                font-family: monospace;
                font-size: 10px;
                font-weight: bold;
                padding: 3px 6px;
                background-color: #0b1120;
                border: 1px solid #1e293b;
                border-radius: 4px;
            """)

    def on_gamepad_gripper_width_updated(self, width_mm: float):
        self.gripper_slider.blockSignals(True)
        self.gripper_slider.setValue(int(round(width_mm)))
        self.gripper_slider.blockSignals(False)
        self.gripper_input.setText(f"{width_mm:.1f}")
        self.lbl_live_gripper.setText(f"Gripper: {width_mm:.1f} mm")

    def on_gamepad_select_clicked(self):
        """Gamepad Select button pressed -> Go to Temp Start."""
        if self.path_manager.is_recording:
            self.log("[GAMEPAD] Cannot move to Start while trajectory recording is active.")
            return
        self.log("[GAMEPAD] 'Select' pressed -> Triggering Go to Start Pose.")
        self.move_to_start_pose()

    def on_gamepad_start_clicked(self):
        """Gamepad Start button pressed -> Toggle Record (same as Space)."""
        self.log("[GAMEPAD] 'Start' pressed -> Toggling Trajectory Recording.")
        self.toggle_recording()

    def on_invert_left_toggled(self, checked: bool):
        if self.gamepad_teleop:
            self.gamepad_teleop.set_invert_left_horizontal(checked)
            self.log(f"Invert Left Stick (L/R) set to: {checked}")

    def on_invert_right_toggled(self, checked: bool):
        if self.gamepad_teleop:
            self.gamepad_teleop.set_invert_right_horizontal(checked)
            self.log(f"Invert Right Stick (Base) set to: {checked}")

    def on_gamepad_speed_changed(self, val: int):
        self.lbl_gp_speed.setText(f"⚡ Speed: {val}%")
        if self.gamepad_teleop:
            self.gamepad_teleop.set_speed_percentage(val)

    def on_gamepad_accel_changed(self, val: int):
        self.lbl_gp_accel.setText(f"🚀 Accel: {val}%")
        if self.gamepad_teleop:
            self.gamepad_teleop.set_accel_percentage(val)

    # -------------------------------------------------------------------------
    # Telemetry Updates & Keyboard Shortcuts
    # -------------------------------------------------------------------------

    def on_robot_telemetry_updated(self, data: Dict[str, Any]):
        joints = data.get("joints", {})
        tcp = data.get("tcp_pose", [0.0] * 6)

        # Update TCP HUD
        if len(tcp) >= 3:
            self.lbl_live_tcp.setText(f"TCP: X={tcp[0]:+.3f} Y={tcp[1]:+.3f} Z={tcp[2]:+.3f} (m)")

        # Update Live Gripper Badge if connected
        if self.gripper and self.gripper.is_connected:
            w = self.gripper._cached_width_mm
            self.lbl_live_gripper.setText(f"Gripper: {w:.1f} mm")

        # Update jog inputs if user is not currently dragging/typing
        if self._user_dragging_joint is None:
            for j in UR5E_JOINT_NAMES:
                deg_val = math.degrees(joints.get(j, 0.0))
                if j in self.jog_inputs and not self.jog_inputs[j].hasFocus():
                    self.jog_inputs[j].setText(f"{deg_val:+.1f}")
                if j in self.jog_sliders:
                    self.jog_sliders[j].blockSignals(True)
                    self.jog_sliders[j].setValue(int(round(deg_val)))
                    self.jog_sliders[j].blockSignals(False)

    def on_robot_connection_status(self, is_connected: bool, message: str):
        self.log(f"[UR5e] {message}")
        if is_connected:
            if "Mock" in message:
                self.badge_ur5e.setText("🤖 UR5e: Simulated Mock")
                self.badge_ur5e.setStyleSheet("""
                    background-color: #0c4a6e;
                    color: #38bdf8;
                    font-size: 11px;
                    font-weight: 700;
                    padding: 4px 10px;
                    border: 1px solid #0284c7;
                    border-radius: 4px;
                """)
            else:
                self.badge_ur5e.setText(f"🤖 UR5e: Connected ({self.robot_ip})")
                self.badge_ur5e.setStyleSheet("""
                    background-color: #064e3b;
                    color: #34d399;
                    font-size: 11px;
                    font-weight: 700;
                    padding: 4px 10px;
                    border: 1px solid #059669;
                    border-radius: 4px;
                """)
                # Inject shared command socket to gamepad and path manager
                if hasattr(self.robot_worker, "_robot") and self.robot_worker._robot:
                    if hasattr(self.robot_worker._robot, "sock") and self.robot_worker._robot.sock:
                        sock = self.robot_worker._robot.sock
                        rx = getattr(self.robot_worker._robot, "rx", None)
                        self.path_manager.set_robot_connection(rtde_rx=rx, command_sock=sock)
                        if self.gamepad_teleop:
                            self.gamepad_teleop.set_external_socket(sock)
        else:
            self.badge_ur5e.setText("🤖 UR5e: Disconnected")
            self.badge_ur5e.setStyleSheet("""
                background-color: #7f1d1d;
                color: #fca5a5;
                font-size: 11px;
                font-weight: 700;
                padding: 4px 10px;
                border: 1px solid #ef4444;
                border-radius: 4px;
            """)

    def keyPressEvent(self, event):
        """Global window keyboard shortcuts."""
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.toggle_recording()
            event.accept()
        elif key == Qt.Key.Key_Escape:
            self.emergency_stop()
            event.accept()
        elif key == Qt.Key.Key_Delete and self.list_saved_paths.hasFocus():
            self.delete_selected_path()
            event.accept()
        elif key == Qt.Key.Key_H:
            self.move_to_home_pose()
            event.accept()
        elif key == Qt.Key.Key_S and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.move_to_start_pose()
            event.accept()
        else:
            super().keyPressEvent(event)

    def log(self, text: str):
        """Appends timestamped log message to the console."""
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{ts}] {text}"
        self.txt_log.appendPlainText(entry)
        print(entry)

    def show_warning(self, title: str, text: str):
        QMessageBox.warning(self, title, text)

    def show_error(self, title: str, text: str):
        QMessageBox.critical(self, title, text)

    def closeEvent(self, event):
        """Clean shutdown of background threads."""
        self.log("Shutting down UR5e Reference Path Studio...")
        self.path_manager.disconnect()

        if hasattr(self, "camera_worker_1") and self.camera_worker_1:
            self.camera_worker_1.stop()

        if hasattr(self, "camera_worker_2") and self.camera_worker_2:
            self.camera_worker_2.stop()

        if self.gamepad_teleop:
            self.gamepad_teleop.stop()

        if self.robot_worker:
            self.robot_worker.stop()

        event.accept()


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="UR5e Reference Path Studio — Phase 1 Golden Trajectory Tool"
    )
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP, help=f"UR5e arm IP (default: {DEFAULT_ROBOT_IP})")
    parser.add_argument("--gripper-ip", type=str, default=DEFAULT_GRIPPER_IP, help=f"OnRobot Compute Box IP (default: {DEFAULT_GRIPPER_IP})")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"Recording frequency in Hz (default: {DEFAULT_FPS})")
    parser.add_argument("--camera-1", type=int, default=DEFAULT_CAMERA_1_INDEX, help=f"Primary Workspace camera index (default: {DEFAULT_CAMERA_1_INDEX})")
    parser.add_argument("--camera-2", type=int, default=DEFAULT_CAMERA_2_INDEX, help=f"Secondary Wrist Flange camera index (default: {DEFAULT_CAMERA_2_INDEX})")
    parser.add_argument("--mock", action="store_true", help="Launch in simulated mock mode without physical hardware")
    return parser.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("UR5e Reference Path Studio")

    window = ReferencePathStudio(
        robot_ip=args.robot_ip,
        gripper_ip=args.gripper_ip,
        target_fps=args.fps,
        camera_1_index=args.camera_1,
        camera_2_index=args.camera_2,
        mock_mode=args.mock,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
