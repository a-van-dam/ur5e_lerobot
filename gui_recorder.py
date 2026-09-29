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
import datetime
import math
import os
from pathlib import Path
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

    def __init__(self, camera_id: int, target_fps: int = 30, width: int = 320, height: int = 240, mock: bool = False, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.target_width = width
        self.target_height = height
        self.mock = mock
        self._running = False
        self._cap = None
        self._mutex = QMutex()

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

        while self._running:
            t0 = time.perf_counter()

            self._mutex.lock()
            tw, th = self.target_width, self.target_height
            self._mutex.unlock()

            if not self.mock and self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
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

        # Latest joint states (radians)
        self._latest_q = {j: 0.0 for j in UR5E_JOINTS}
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

                    # Dynamic timeout based on maximum angular joint displacement
                    max_disp = max(abs(self._latest_q[j] - target_pose[f"{j}.pos"]) for j in UR5E_JOINTS)
                    self._motion_timeout = max(3.0, (max_disp / 0.5) + 2.0)

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

                # Monitor motion completion on physical arm
                if self._motion_in_progress:
                    current_disp = max(abs(self._latest_q[j] - self._motion_target_q[idx]) for idx, j in enumerate(UR5E_JOINTS))
                    elapsed = time.perf_counter() - self._motion_start_time
                    if current_disp < 0.02 or elapsed >= self._motion_timeout:
                        self._motion_in_progress = False
                        if elapsed >= self._motion_timeout and current_disp >= 0.02:
                            print(f"[WARNING] Motion wait timed out after {elapsed:.1f}s (max joint err: {math.degrees(current_disp):.1f}°).")
                        else:
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
                    if max_err < 0.005 or elapsed >= self._motion_timeout:
                        self._motion_in_progress = False
                        print(f"[MOTION] Mock target motion reached ({elapsed:.2f}s).")
                        if self._motion_restore_freedrive:
                            self._motion_restore_freedrive = False
                            self._freedrive_active = True
                            self.freedrive_state_changed.emit(True)
                            print("[MOTION] Freedrive automatically restored after motion completion.")
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
        self.video_canvas.setMinimumSize(320, 240)
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

    @pyqtSlot(int, np.ndarray)
    def update_frame(self, cam_id: int, frame_bgr: np.ndarray):
        if cam_id != self.camera_id:
            return

        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        # OpenCV BGR to RGB conversion
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

        # Custom Start Position (defaults to canonical HOME_JOINTS)
        self.start_position: Dict[str, float] = dict(HOME_JOINTS_DICT)
        self.jog_sliders: Dict[str, QSlider] = {}
        self.jog_inputs: Dict[str, QLineEdit] = {}
        self.jog_move_buttons: Dict[str, QPushButton] = {}
        self.jog_labels: Dict[str, QLineEdit] = self.jog_inputs
        self._user_modified_joints: set[str] = set()
        self._user_targets: Dict[str, float] = {}

        self.camera_width = 320
        self.camera_height = 240

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

        # Initialize Gamepad Teleoperation Worker
        self.gamepad_teleop_active = False
        if GAMEPAD_AVAILABLE and UR5eGamepadTeleop is not None:
            self.gamepad_teleop = UR5eGamepadTeleop(
                robot_ip=self.robot_ip,
                mock=self.mock_mode,
                parent=self,
            )
            self.gamepad_teleop.status_updated.connect(self.on_gamepad_status_updated)
            self.gamepad_teleop.velocity_updated.connect(self.on_gamepad_velocity_updated)
            self.gamepad_teleop.teleop_state_changed.connect(self.on_gamepad_state_changed)
            # Provide live base motor angle (shoulder_pan) for heading-relative Cartesian control
            self.gamepad_teleop.set_base_angle_provider(
                lambda: self.robot_worker.get_latest_joints().get("shoulder_pan", 0.0)
            )
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

        # Scan initial dataset directory
        self.rescan_dataset_directory()

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
            ("320 x 240 (Fast Training - Recommended)", 320, 240),
            ("256 x 256 (Square - Policy/ACT)", 256, 256),
            ("224 x 224 (Square - ResNet/ViT)", 224, 224),
            ("640 x 480 (Full / High-Res)", 640, 480),
            ("160 x 120 (Ultra Fast)", 160, 120),
        ]
        self.combo_resolution = QComboBox()
        for label, _, _ in self.RESOLUTION_PRESETS:
            self.combo_resolution.addItem(label)
        self.combo_resolution.setCurrentIndex(0)  # Default: 320x240
        self.combo_resolution.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        self.combo_resolution.setToolTip("Select recording resolution for cameras. Smaller resolutions (e.g. 320x240) train faster and save disk space.")
        self.combo_resolution.currentIndexChanged.connect(self.on_resolution_changed)
        cam_ctrl_bar.addWidget(self.combo_resolution)

        self.lbl_dual_badge = QLabel("○ 1-Cam Mode")
        self.lbl_dual_badge.setStyleSheet("color: #64748b; font-size: 10px; margin-left: 6px;")
        cam_ctrl_bar.addWidget(self.lbl_dual_badge)

        self.lbl_cam_info = QLabel("Format: 320x240 RGB @ 30 FPS")
        self.lbl_cam_info.setStyleSheet("color: #64748b; font-size: 10px; margin-left: 4px;")
        cam_ctrl_bar.addWidget(self.lbl_cam_info)
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
        gamepad_layout = QHBoxLayout(gamepad_panel)
        gamepad_layout.setContentsMargins(8, 5, 8, 5)
        gamepad_layout.setSpacing(10)

        # Toggle Button: "Enable Gamepad Teleop"
        self.btn_toggle_gamepad = QPushButton("🎮 ENABLE GAMEPAD TELEOP")
        self.btn_toggle_gamepad.setCheckable(True)
        self.btn_toggle_gamepad.setMinimumHeight(36)
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
            "• Left Stick (XY): TCP Forward/Backward (+X/-X) and Left/Right (+Y/-Y)\n"
            "• Right Stick (Y): TCP Vertical Up/Down (+Z/-Z)\n"
            "• Right Stick (X): Direct Base Motor Rotation (shoulder_pan)\n"
            "• Shoulders (LB/RB / L1/R1): Direct Wrist 2 Rotation\n"
            "• Triggers (LT/RT): Direct Wrist 3 Rotation\n"
            "• Neutral sticks/buttons: Immediate braking (0 drift)"
        )
        self.btn_toggle_gamepad.toggled.connect(self.on_gamepad_toggle_clicked)
        gamepad_layout.addWidget(self.btn_toggle_gamepad)

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
        gamepad_layout.addWidget(self.lbl_gamepad_badge)

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
        gamepad_layout.addWidget(self.lbl_gamepad_vel)

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
        gamepad_layout.addWidget(self.chk_invert_left_lr)

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
        gamepad_layout.addWidget(self.chk_invert_right_lr)

        # Backwards compatibility alias
        self.chk_invert_lr = self.chk_invert_left_lr

        gamepad_layout.addStretch()
        rec_layout.addWidget(gamepad_panel)

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
        self.txt_dataset_name = QLineEdit(DEFAULT_DATASET_NAME)
        self.txt_dataset_name.returnPressed.connect(self.apply_dataset_name)
        ds_name_row.addWidget(self.txt_dataset_name)

        self.btn_apply_ds = QPushButton("Set / Load")
        self.btn_apply_ds.setStyleSheet("""
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
        self.btn_apply_ds.clicked.connect(self.apply_dataset_name)
        ds_name_row.addWidget(self.btn_apply_ds)

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
                    self.camera_2_index, self.target_fps, self.camera_width, self.camera_height, self.mock_mode
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

    def on_cam1_device_changed(self, index: int):
        cam_id = self.combo_cam1.itemData(index)
        if cam_id is not None and cam_id != self.camera_1_index:
            print(f"[CAMERA 1] Switching Workspace camera to /dev/video{cam_id}...")
            self.camera_1_index = cam_id
            self.camera_index = cam_id
            if self.camera_worker_1 is not None:
                self.camera_worker_1.stop()
                self.camera_worker_1 = CameraWorker(
                    self.camera_1_index, self.target_fps, self.camera_width, self.camera_height, self.mock_mode
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
                    self.camera_2_index, self.target_fps, self.camera_width, self.camera_height, self.mock_mode
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
        self.lbl_cam_info.setText(f"Format: {w}x{h} RGB @ {self.target_fps} FPS")
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
        self.lbl_cam_info.setText(f"Format: {w}x{h} RGB @ {self.target_fps} FPS")
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
        cmd = f"lerobot-dataset-viz --repo-id local/{clean_name} --root ./datasets/{clean_name} --episode-index {selected_idx}"
        if hasattr(self, "txt_viz_command"):
            self.txt_viz_command.setText(cmd)

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

        # Sync live base motor angle to gamepad teleop for heading-relative control
        if "shoulder_pan" in joints and hasattr(self, "gamepad_teleop") and self.gamepad_teleop:
            self.gamepad_teleop.set_base_angle(joints["shoulder_pan"])

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
        # Demonstrations can be recorded via Gamepad Cartesian Teleoperation,
        # physical Freedrive hand-guiding, or precision axis controls.
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop and self.gamepad_teleop.is_enabled() and not self.gamepad_teleop.is_inhibited():
            mode_desc = f"Gamepad Teleoperation ({self.gamepad_teleop.get_controller_name()})"
        elif self.freedrive_active:
            mode_desc = "Freedrive Hand-Guiding"
        else:
            mode_desc = "Precision Axis Controls"
        print(f"[RECORD] Starting demonstration episode recording ({mode_desc})...")
        self.status_bar.showMessage(f"Recording episode active ({mode_desc}). Press Space to stop.", 5000)

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
        self.btn_record.setText("⏸  END EPISODE (Space)")
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
            # Stop recording timers
            self.dataset_frame_timer.stop()
            self.recording_timer.stop()
            self.is_recording = False
            duration_s = time.perf_counter() - (self.record_start_time or time.perf_counter())

            self.cam_widget_1.set_recording(False)
            self.cam_widget_2.set_recording(False)

            # Prevent accidental double-clicks: disable button and display saving state
            self.btn_record.setEnabled(False)
            self.btn_record.setText("⏳  SAVING EPISODE (Encoding Video & Parquet)...")
            self.btn_record.setStyleSheet("""
                QPushButton {
                    background-color: #92400e;
                    color: #fef08a;
                    font-size: 13px;
                    font-weight: bold;
                    border-radius: 6px;
                    border: 2px solid #f59e0b;
                }
            """)
            self.btn_cancel.setEnabled(False)
            self.btn_stop_session.setEnabled(False)
            QApplication.processEvents()

            num_frames = len(self.current_episode_buffer)
            ep_idx = self.list_episodes.count()

            print(f"[RECORD] Saving Episode #{ep_idx} ({num_frames} frames, {duration_s:.1f}s)...")
            self.status_bar.showMessage(f"Writing Episode #{ep_idx} to disk...")

            # Write episode to disk / dataset
            self.commit_episode_to_dataset(ep_idx, num_frames, duration_s)

            # Maintain arm in freedrive mode as requested
            print("[INFO] Episode saved! Robot remains in Freedrive mode for the next demo.")
            self.status_bar.showMessage(f"Episode #{ep_idx} saved ({num_frames} frames). Ready for next episode.", 6000)
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed in end_episode_and_save: {e}", file=sys.__stderr__)
            import traceback
            traceback.print_exc(file=sys.__stderr__)
            self.status_bar.showMessage(f"Error saving episode: {e}", 8000)
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving the demonstration:\n\n{e}")
        finally:
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
    def apply_dataset_name(self):
        """Switches or initializes the active dataset when the user clicks 'Set / Load' or presses Enter."""
        clean_name = self.txt_dataset_name.text().strip()
        if not clean_name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a valid dataset name.")
            return

        self.active_dataset = None
        self.dataset_root = DEFAULT_DATASETS_DIR / clean_name
        self.lbl_dataset_path.setText(f"Path: datasets/{clean_name}")
        self.rescan_dataset_directory()

        print(f"[DATASET] Switched active dataset to: '{clean_name}' (Path: datasets/{clean_name})")
        self.status_bar.showMessage(f"Active dataset switched to '{clean_name}'.", 4000)

    def rescan_dataset_directory(self):
        self.list_episodes.clear()

        if not self.dataset_root.exists():
            print(f"[DATASET] Directory '{self.dataset_root.name}' does not exist yet. It will be created when an episode is saved.")
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

    def commit_episode_to_dataset(self, ep_idx: int, num_frames: int, duration_s: float):
        try:
            dataset = self.get_or_create_dataset()

            if dataset is not None and torch is not None:
                task_str = self.txt_task_desc.text().strip() or "Pick and place demonstration"
                for frame in self.current_episode_buffer:
                    t_frame = {
                        "observation.state": torch.from_numpy(frame["observation.state"]),
                        "action": torch.from_numpy(frame["action"]),
                        "observation.images.workspace": torch.from_numpy(
                            np.ascontiguousarray(frame["observation.images.workspace"], dtype=np.uint8)
                        ),
                        "task": task_str,
                    }
                    if "observation.images.wrist" in frame:
                        t_frame["observation.images.wrist"] = torch.from_numpy(
                            np.ascontiguousarray(frame["observation.images.wrist"], dtype=np.uint8)
                        )
                    dataset.add_frame(t_frame)

                dataset.save_episode()
                # Immediately finalize so parquet footers and meta/episodes are completely written to disk
                dataset.finalize()
                total_episodes_saved = dataset.num_episodes
                has_wrist = self.enable_dual_cameras
                self.active_dataset = None  # Reset instance so next episode cleanly resumes via LeRobotDataset.resume()
                print(f"[DATASET] ✔ LeRobot episode #{ep_idx:04d} written and finalized successfully ({num_frames} frames)!")
                print(f"          • Telemetry Parquet: {self.dataset_root / 'data'}")
                print(f"          • Workspace Video:   {self.dataset_root / 'videos' / 'observation.images.workspace'}")
                if has_wrist:
                    print(f"          • Wrist Video:       {self.dataset_root / 'videos' / 'observation.images.wrist'}")
                print(f"          • Metadata Episodes: {self.dataset_root / 'meta' / 'episodes'}")
                print(f"          • Total episodes:    {total_episodes_saved}")
            else:
                print("[WARNING] Saving to fallback manifest because LeRobot is unavailable in this environment.", file=sys.__stderr__)
                self._write_local_manifest_episode(ep_idx, num_frames, duration_s)

            # Update UI List
            item_text = f"Episode #{ep_idx:04d} — {num_frames} frames ({duration_s:.1f}s)"
            self.list_episodes.addItem(QListWidgetItem(item_text))
            self.list_episodes.setCurrentRow(self.list_episodes.count() - 1)
            self.current_episode_buffer.clear()
            self.update_visualizer_command()
        except Exception as err:
            print(f"[ERROR] commit_episode_to_dataset exception: {err}", file=sys.__stderr__)
            import traceback
            traceback.print_exc(file=sys.__stderr__)
            self._write_local_manifest_episode(ep_idx, num_frames, duration_s)
            item_text = f"Episode #{ep_idx:04d} — {num_frames} frames ({duration_s:.1f}s)"
            self.list_episodes.addItem(QListWidgetItem(item_text))
            self.list_episodes.setCurrentRow(self.list_episodes.count() - 1)
            self.current_episode_buffer.clear()
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
        if self.is_recording:
            self.cancel_recording()

        if self.diagnostic_worker and self.diagnostic_worker.isRunning():
            self.diagnostic_worker.terminate_process()

        # Stop workers
        if hasattr(self, "gamepad_teleop") and self.gamepad_teleop is not None:
            try:
                self.gamepad_teleop.stop()
            except Exception:
                pass
        if self.camera_worker_1 is not None:
            self.camera_worker_1.stop()
        if self.camera_worker_2 is not None:
            self.camera_worker_2.stop()
        self.robot_worker.stop()

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

