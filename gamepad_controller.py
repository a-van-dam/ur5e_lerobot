#!/usr/bin/env python3
"""
UR5e Gamepad Teleoperation Controller
=====================================
Dedicated module providing real-time Cartesian (TCP / tool) and direct joint
teleoperation for the Universal Robots UR5e arm using a standard USB/Bluetooth gamepad via Pygame.

Movement Scheme:
- Left Stick (X/Y)       : Tool Translation Forward/Backward (+X/-X) and Left/Right (+Y/-Y) in robot heading frame via `speedl`
- Right Stick (Y)        : Tool Translation Vertical Up/Down (+Z/-Z) via `speedl`
- Right Stick (X)        : DIRECT Base motor rotation (joint 0 / shoulder_pan) via `speedj`
- Shoulder Buttons (LB/RB): DIRECT Wrist 2 rotation (joint 4 / wrist_2) via `speedj`
- Triggers (LT/RT)       : DIRECT Wrist 1 rotation (joint 3 / wrist_1) via `speedj`
- Select / Back Button   : Go to Start / Home Position
- Start / Options Button : Toggle Episode Demonstration Recording (Space)
- Neutral Sticks/Buttons : Smooth software deceleration ramp -> `stopl` / `stopj` (zero drift)

Features:
- Decoupled Wrist 3: Tool positioning in space via sticks keeps Wrist 3 unaffected.
- Smooth acceleration ramping (slew rate limiter) protecting Wrist 1 from protective stops.
- Dynamic user scaling for Speed (10%-100%, default 50%) and Acceleration (10%-100%, default 50%).
- Universal support for Xbox, PlayStation, and generic gamepads.
- Standard default start pose: [-80, -45, -100, 150, -85, 60].
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
import socket
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

# Enable background events in SDL before initializing Pygame so gamepads work without a focused window
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)
        import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from rtde_receive import RTDEReceiveInterface
    RTDE_RX_AVAILABLE = True
except ImportError:
    try:
        from ur_rtde import rtde_receive
        RTDEReceiveInterface = rtde_receive.RTDEReceiveInterface
        RTDE_RX_AVAILABLE = True
    except ImportError:
        RTDEReceiveInterface = None
        RTDE_RX_AVAILABLE = False

# Safe import of PyQt6 components with graceful fallback to standard threading
try:
    from PyQt6.QtCore import QMutex, QMutexLocker, QObject, QThread, pyqtSignal
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False

    class QObject:
        pass

    class QThread:
        def __init__(self, parent=None):
            import threading
            self._thread: Optional[threading.Thread] = None

        def start(self):
            import threading
            self._thread = threading.Thread(target=self.run, daemon=True)
            self._thread.start()

        def isRunning(self):
            return self._thread is not None and self._thread.is_alive()

        def wait(self, msecs=2000):
            if self._thread:
                self._thread.join(timeout=msecs / 1000.0)

    class pyqtSignal:
        def __init__(self, *types):
            self._callbacks = []

        def connect(self, callback):
            self._callbacks.append(callback)

        def emit(self, *args):
            for cb in self._callbacks:
                try:
                    cb(*args)
                except Exception:
                    pass

    class QMutex:
        def __init__(self):
            import threading
            self._lock = threading.Lock()

        def lock(self):
            self._lock.acquire()

        def unlock(self):
            self._lock.release()

    class QMutexLocker:
        def __init__(self, mutex: QMutex):
            self.mutex = mutex

        def __enter__(self):
            self.mutex.lock()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.mutex.unlock()


# Standard default start pose loaded dynamically from default_start_position.json
try:
    from start_position_config import (
        get_default_start_joints_deg,
        get_default_start_joints_rad,
        get_default_start_gripper_mm,
        UR5E_JOINT_NAMES,
    )
    DEFAULT_START_JOINTS_DEG = get_default_start_joints_deg()
    DEFAULT_START_JOINTS_RAD = get_default_start_joints_rad()
    DEFAULT_START_GRIPPER_MM = get_default_start_gripper_mm()
except ImportError:
    DEFAULT_START_JOINTS_DEG = [-90.0, -40.0, -100.0, 120.0, -90.0, 120.0]
    DEFAULT_START_JOINTS_RAD = [math.radians(x) for x in DEFAULT_START_JOINTS_DEG]
    DEFAULT_START_GRIPPER_MM = 105.0
    UR5E_JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]


# =============================================================================
# 1. Modular Gamepad Configuration Mapping
# =============================================================================

@dataclass
class GamepadConfig:
    """
    Configurable hardware mappings, deadzones, and velocity scaling for UR5e teleoperation.
    Standard Xbox / SDL2 layout on Linux:
      - Left Stick X (Axis 0)      : Left / Right translation (-Y / +Y)
      - Left Stick Y (Axis 1)      : Forward / Backward translation (+X / -X)
      - Right Stick X (Axis 3)     : Direct Base motor rotation (-shoulder_pan / +shoulder_pan)
      - Right Stick Y (Axis 4)     : Vertical Up / Down translation (+Z / -Z)
      - Left Trigger (Axis 2)      : Rotate Wrist 1 in negative direction (-wrist_1)
      - Right Trigger (Axis 5)     : Rotate Wrist 1 in positive direction (+wrist_1)
      - Left Shoulder (Button 4)   : Rotate Wrist 2 in negative direction (-wrist_2)
      - Right Shoulder (Button 5)  : Rotate Wrist 2 in positive direction (+wrist_2)
      - Select / Back (Button 6)   : Go to Start / Home Position
      - Start / Options (Button 7) : Toggle Demonstration Recording (Space)
    """
    # Axis indices
    axis_left_x: int = 0
    axis_left_y: int = 1
    axis_right_x: int = 3
    axis_right_y: int = 4
    axis_trigger_left: int = 2
    axis_trigger_right: int = 5

    # Button indices
    button_shoulder_left: int = 4
    button_shoulder_right: int = 5
    button_select: int = 6
    button_start: int = 7

    # Gripper Button Indices (Xbox: A=0 close, B=1 open; PlayStation: Cross=0 close, Circle=1 open)
    button_gripper_close: int = 0
    button_gripper_open: int = 1
    gripper_discrete_mode: bool = True  # Crisp binary snap: [A]=Full Close (55mm), [B]=Full Open (105mm)
    gripper_speed_mms: float = 25.0     # Fallback aperture rate in mm/s if gripper_discrete_mode=False

    # Direction inversions (+1.0 normal, -1.0 inverted)
    invert_x: float = -1.0
    invert_y: float = 1.0
    invert_z: float = -1.0
    invert_base: float = 1.0
    invert_wrist1: float = 1.0
    invert_wrist2: float = 1.0

    # User togglable direction inversions
    invert_left_horizontal: bool = False
    invert_right_horizontal: bool = False

    # Direction & Frame modes
    direction_mode: str = "tool_gaze"        # 'tool_gaze' (moves in gripper pointing direction) or 'base_heading'
    heading_relative: bool = True
    heading_offset_rad: float = 0.0

    # Filtering & Response curve (smooth, responsive, intuitive control)
    deadzone: float = 0.08                     # Deadzone threshold (starts smoothly)
    trigger_deadzone: float = 0.05
    velocity_curve_exponent: float = 1.4       # Responsive curve for fine & brisk control
    left_stick_curve_exponent: float = 1.4     # Left stick responsive curve (smooth near center, brisk when pushed)
    left_stick_speed_mult: float = 0.85        # Active forward/backward reach (moves gripper head clearly in space)

    # Master Scaling Factors (0.10 to 1.00, default 0.50 = 50%)
    speed_scale: float = 0.50
    accel_scale: float = 0.50

    # Baseline Velocities at 100% scale
    base_linear_velocity: float = 0.15         # m/s -> at 50% scale, 0.075 m/s (~6.4 cm/s with left mult)
    base_base_velocity: float = 0.50           # rad/s -> 0.25 rad/s at 50% (Right stick X: unchanged)
    base_wrist1_velocity: float = 0.50         # rad/s -> 0.25 rad/s at 50% (Triggers LT/RT)
    base_wrist2_velocity: float = 0.50         # rad/s -> 0.25 rad/s at 50% (Shoulders LB/RB)

    # Baseline Accelerations at 100% scale (gentle defaults eliminate protective stops)
    base_linear_acceleration: float = 0.40     # m/s^2 -> responsive linear acceleration ramp
    base_joint_acceleration: float = 0.60      # rad/s^2 -> 0.30 rad/s^2 at 50%

    # URScript execution parameters
    time_step: float = 0.08                 # URScript watchdog duration (seconds)
    poll_rate_hz: float = 35.0              # Gamepad poll & command stream frequency (Hz)
    robot_port: int = 30002                 # Secondary client interface port for URScript

    @property
    def max_linear_velocity(self) -> float:
        return self.base_linear_velocity * self.speed_scale

    @max_linear_velocity.setter
    def max_linear_velocity(self, val: float):
        self.base_linear_velocity = val / max(1e-3, self.speed_scale)

    @property
    def max_base_velocity(self) -> float:
        return self.base_base_velocity * self.speed_scale

    @max_base_velocity.setter
    def max_base_velocity(self, val: float):
        self.base_base_velocity = val / max(1e-3, self.speed_scale)

    @property
    def max_wrist1_velocity(self) -> float:
        return self.base_wrist1_velocity * self.speed_scale

    @max_wrist1_velocity.setter
    def max_wrist1_velocity(self, val: float):
        self.base_wrist1_velocity = val / max(1e-3, self.speed_scale)

    @property
    def max_wrist2_velocity(self) -> float:
        return self.base_wrist2_velocity * self.speed_scale

    @max_wrist2_velocity.setter
    def max_wrist2_velocity(self, val: float):
        self.base_wrist2_velocity = val / max(1e-3, self.speed_scale)

    @property
    def linear_acceleration(self) -> float:
        return max(0.05, self.base_linear_acceleration * self.accel_scale)

    @linear_acceleration.setter
    def linear_acceleration(self, val: float):
        self.base_linear_acceleration = val / max(1e-3, self.accel_scale)

    @property
    def joint_acceleration(self) -> float:
        return max(0.05, self.base_joint_acceleration * self.accel_scale)

    @joint_acceleration.setter
    def joint_acceleration(self, val: float):
        self.base_joint_acceleration = val / max(1e-3, self.accel_scale)


# Presets for popular controller families
XBOX_CONFIG = GamepadConfig(
    axis_left_x=0, axis_left_y=1,
    axis_right_x=3, axis_right_y=4,
    axis_trigger_left=2, axis_trigger_right=5,
    button_shoulder_left=4, button_shoulder_right=5,
    button_select=6, button_start=7,
    button_gripper_close=0, button_gripper_open=1,
    invert_x=-1.0, invert_y=1.0, invert_z=-1.0,
    invert_base=1.0, invert_wrist1=1.0, invert_wrist2=1.0,
)

PLAYSTATION_CONFIG = GamepadConfig(
    axis_left_x=0, axis_left_y=1,
    axis_right_x=2, axis_right_y=3,
    axis_trigger_left=4, axis_trigger_right=5,
    button_shoulder_left=4, button_shoulder_right=5,
    button_select=8, button_start=9,
    button_gripper_close=0, button_gripper_open=1,
    invert_x=-1.0, invert_y=1.0, invert_z=-1.0,
    invert_base=1.0, invert_wrist1=1.0, invert_wrist2=1.0,
)


def get_recommended_config_for_name(controller_name: str) -> GamepadConfig:
    """Selects suitable axis/button defaults based on detected controller name."""
    name_lower = controller_name.lower()
    if any(k in name_lower for k in ("playstation", "dualshock", "dualsense", "sony")):
        return PLAYSTATION_CONFIG
    return XBOX_CONFIG


# =============================================================================
# 2. Mathematical Utilities & String Formatting
# =============================================================================

def apply_deadzone(value: float, deadzone: float, exponent: float = 1.0) -> float:
    """Applies linear or polynomial deadzone rescaling starting smoothly at 0.0."""
    abs_val = abs(value)
    if abs_val <= deadzone:
        return 0.0

    scaled = (abs_val - deadzone) / (1.0 - deadzone)
    scaled = min(1.0, max(0.0, scaled))

    if exponent != 1.0:
        scaled = math.pow(scaled, exponent)

    return math.copysign(scaled, value)


def rotvec_to_matrix(rotvec: Union[List[float], Tuple[float, ...]]) -> List[List[float]]:
    """Converts a 3D rotation vector [rx, ry, rz] to a 3x3 rotation matrix using Rodrigues' formula."""
    rx, ry, rz = float(rotvec[0]), float(rotvec[1]), float(rotvec[2])
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-6:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    kx, ky, kz = rx / theta, ry / theta, rz / theta
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    v = 1.0 - cos_t

    return [
        [cos_t + kx * kx * v, kx * ky * v - kz * sin_t, kx * kz * v + ky * sin_t],
        [ky * kx * v + kz * sin_t, cos_t + ky * ky * v, ky * kz * v - kx * sin_t],
        [kz * kx * v - ky * sin_t, kz * ky * v + kx * sin_t, cos_t + kz * kz * v],
    ]


def format_speedl(v: List[float], a: float = 0.35, t: float = 0.08) -> str:
    """Formats a Cartesian TCP velocity command as a standard URScript string ending with newline."""
    v_str = ", ".join(f"{val:.4f}" for val in v)
    return f"speedl([{v_str}], a={a:.2f}, t={t:.2f})\n"


def format_speedj(qd: List[float], a: float = 0.60, t: float = 0.08) -> str:
    """Formats a joint velocity command as a standard URScript string ending with newline."""
    qd_str = ", ".join(f"{val:.4f}" for val in qd)
    return f"speedj([{qd_str}], a={a:.2f}, t={t:.2f})\n"


def format_stopl(a: float = 0.35) -> str:
    """Formats a URScript Cartesian stop command to smoothly decelerate to rest ending with newline."""
    return f"stopl({a:.2f})\n"


def format_stopj(a: float = 0.60) -> str:
    """Formats a URScript joint stop command to smoothly decelerate to rest ending with newline."""
    return f"stopj({a:.2f})\n"


# =============================================================================
# 3. UR5e Gamepad Teleoperation Worker Class
# =============================================================================

class UR5eGamepadTeleop(QThread):
    """
    High-frequency background worker thread for UR5e teleoperation.
    Features:
      - Tool 3D position translation in space (`speedl`) with Wrist 3 strictly decoupled
      - Direct Base pan via Right Stick X (`speedj`) with inversion toggle
      - Direct Wrist 2 rotation via Shoulder buttons (`speedj`)
      - Direct Wrist 1 rotation via Triggers (`speedj`)
      - Select button: Go to Start / Home Position signal
      - Start button: Start / End Episode Recording (Space) signal
      - Acceleration slew-rate limiting preventing Wrist 1 protective stops
      - User-configurable speed and acceleration scaling (10% to 100%)
    """
    # Qt Signals for GUI integration
    status_updated = pyqtSignal(bool, str)        # (is_connected, controller_name)
    velocity_updated = pyqtSignal(list)           # [vx, vy, vz, w_base, w_wrist2, w_wrist1]
    teleop_state_changed = pyqtSignal(bool, str)  # (is_active, status_message)
    command_sent = pyqtSignal(str)                # URScript command string
    speed_scale_changed = pyqtSignal(float)       # speed_scale (0.10 to 1.00)
    accel_scale_changed = pyqtSignal(float)       # accel_scale (0.10 to 1.00)
    button_select_pressed = pyqtSignal()          # Trigger Go to Start position
    button_start_pressed = pyqtSignal()           # Trigger Start/End demonstration recording
    gripper_width_updated = pyqtSignal(float)     # Target gripper width in mm

    def __init__(
        self,
        robot_ip: str = "192.168.1.102",
        config: Optional[GamepadConfig] = None,
        mock: bool = False,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.robot_ip = robot_ip
        self.config = config or GamepadConfig()
        self.mock = mock

        # Thread lifecycle
        self._running = False
        self._lock = QMutex()

        # Teleoperation activation & safety flags
        self._enabled = False          # User-controlled master switch
        self._inhibited = False        # Safety interlock (Freedrive, discrete moves)
        self._inhibition_reason = ""
        self._is_moving = False        # True while non-zero velocity commands are active
        self._active_mode: Optional[str] = None

        # Hardware connection & Gamepad state
        self._socket: Optional[socket.socket] = None
        self._external_socket: Optional[socket.socket] = None
        self._rtde_rx: Optional[Any] = None
        self._velocity_callback: Optional[Callable[[List[float]], None]] = None
        self._joystick: Optional[pygame.joystick.Joystick] = None
        self._controller_name: str = "No Gamepad Detected"
        self._is_controller_connected: bool = False

        # Trigger activation trackers (safeguards against SDL initial 0.0 unpressed value)
        self._lt_activated = False
        self._rt_activated = False

        # Edge detectors for Select and Start buttons
        self._prev_btn_select = False
        self._prev_btn_start = False

        # Commanded velocity state for smooth software slew-rate limiting
        self._current_commanded_cart: List[float] = [0.0] * 6
        self._current_commanded_joint: List[float] = [0.0] * 6
        self._latest_state: List[float] = [0.0] * 6

        # Active robot joint angles (radians)
        self._current_joints: List[float] = list(DEFAULT_START_JOINTS_RAD)
        self._joint_positions_provider: Optional[Callable[[], List[float] | Dict[str, float]]] = None

        # Base orientation tracking for heading-relative Cartesian control (radians)
        self._current_base_angle: float = self._current_joints[0]
        self._base_angle_provider: Optional[Callable[[], float]] = None

        # 6-DoF TCP pose tracking for Tool-Gaze frame directional control [x, y, z, rx, ry, rz]
        self._tcp_pose: List[float] = [-0.1334, -0.0631, 0.5503, 1.098, -1.8947, 1.324]
        self._tcp_pose_provider: Optional[Callable[[], List[float]]] = None

        # Gripper control state
        self.gripper: Optional[Any] = None
        self._gripper_target_width: float = DEFAULT_START_GRIPPER_MM  # Default wide open (105.0 mm)
        self._last_gripper_cmd_time: float = 0.0
        self._gripper_button_active: bool = False
        self._prev_btn_gripper_close: bool = False
        self._prev_btn_gripper_open: bool = False

    # -------------------------------------------------------------------------
    # Public Control & Configuration API
    # -------------------------------------------------------------------------

    def set_speed_scale(self, scale: float):
        """Sets master speed scale (0.10 to 1.00)."""
        with QMutexLocker(self._lock):
            self.config.speed_scale = max(0.05, min(1.0, float(scale)))
        self.speed_scale_changed.emit(self.config.speed_scale)

    def get_speed_scale(self) -> float:
        with QMutexLocker(self._lock):
            return self.config.speed_scale

    def set_speed_percentage(self, pct: int):
        self.set_speed_scale(float(pct) / 100.0)

    def get_speed_percentage(self) -> int:
        return int(round(self.get_speed_scale() * 100.0))

    def set_accel_scale(self, scale: float):
        """Sets master acceleration scale (0.10 to 1.00)."""
        with QMutexLocker(self._lock):
            self.config.accel_scale = max(0.05, min(1.0, float(scale)))
        self.accel_scale_changed.emit(self.config.accel_scale)

    def get_accel_scale(self) -> float:
        with QMutexLocker(self._lock):
            return self.config.accel_scale

    def set_accel_percentage(self, pct: int):
        self.set_accel_scale(float(pct) / 100.0)

    def get_accel_percentage(self) -> int:
        return int(round(self.get_accel_scale() * 100.0))

    def set_joint_positions(self, joints: Dict[str, float] | List[float]):
        """Updates internal joint position state from live robot telemetry."""
        with QMutexLocker(self._lock):
            if isinstance(joints, dict):
                if all(k in joints for k in UR5E_JOINT_NAMES):
                    self._current_joints = [float(joints[k]) for k in UR5E_JOINT_NAMES]
                    self._current_base_angle = self._current_joints[0]
            elif isinstance(joints, (list, tuple)) and len(joints) >= 6:
                self._current_joints = [float(x) for x in joints[:6]]
                self._current_base_angle = self._current_joints[0]

    def set_joint_positions_provider(self, provider_fn: Callable[[], List[float] | Dict[str, float]]):
        with QMutexLocker(self._lock):
            self._joint_positions_provider = provider_fn

    def get_joint_positions(self) -> List[float]:
        with QMutexLocker(self._lock):
            if self._joint_positions_provider is not None:
                try:
                    val = self._joint_positions_provider()
                    if isinstance(val, dict):
                        if all(k in val for k in UR5E_JOINT_NAMES):
                            return [float(val[k]) for k in UR5E_JOINT_NAMES]
                    elif isinstance(val, (list, tuple)) and len(val) >= 6:
                        return [float(x) for x in val[:6]]
                except Exception:
                    pass
            return list(self._current_joints)

    def set_base_angle(self, angle_rad: float):
        with QMutexLocker(self._lock):
            self._current_base_angle = float(angle_rad)
            self._current_joints[0] = float(angle_rad)

    def get_base_angle(self) -> float:
        with QMutexLocker(self._lock):
            if self._base_angle_provider is not None:
                try:
                    return float(self._base_angle_provider())
                except Exception:
                    pass
            return self._current_base_angle

    def set_base_angle_provider(self, provider_fn: Callable[[], float]):
        with QMutexLocker(self._lock):
            self._base_angle_provider = provider_fn

    def set_tcp_pose_provider(self, provider_fn: Callable[[], List[float]]):
        """Attaches an external callable providing live [x, y, z, rx, ry, rz] TCP pose."""
        with QMutexLocker(self._lock):
            self._tcp_pose_provider = provider_fn

    def set_tcp_pose(self, pose: List[float]):
        """Sets cached 6-DoF TCP pose."""
        with QMutexLocker(self._lock):
            if isinstance(pose, (list, tuple)) and len(pose) >= 6:
                self._tcp_pose = [float(x) for x in pose[:6]]

    def get_tcp_pose(self) -> List[float]:
        """Returns the current 6-DoF TCP pose [x, y, z, rx, ry, rz]."""
        with QMutexLocker(self._lock):
            if self._tcp_pose_provider is not None:
                try:
                    p = self._tcp_pose_provider()
                    if isinstance(p, (list, tuple)) and len(p) >= 6:
                        return [float(x) for x in p[:6]]
                except Exception:
                    pass
            return list(self._tcp_pose)

    def set_gripper(self, gripper: Any):
        """Attaches an OnRobot2FGModbus instance for gamepad gripper control."""
        with QMutexLocker(self._lock):
            self.gripper = gripper
            if self.gripper is not None and hasattr(self.gripper, "get_width"):
                try:
                    w = float(self.gripper.get_width())
                    if w > 0:
                        self._gripper_target_width = w
                except Exception:
                    pass

    def set_gripper_target_width(self, width_mm: float):
        """Synchronizes target gripper width from external sources (such as GUI sliders)."""
        with QMutexLocker(self._lock):
            min_w = getattr(self.gripper, "min_width_mm", 55.0) if self.gripper else 55.0
            max_w = getattr(self.gripper, "max_width_mm", 105.0) if self.gripper else 105.0
            self._gripper_target_width = max(min_w, min(max_w, float(width_mm)))

    def get_gripper_target_width(self) -> float:
        with QMutexLocker(self._lock):
            return self._gripper_target_width

    def set_invert_left_horizontal(self, invert: bool):
        with QMutexLocker(self._lock):
            self.config.invert_left_horizontal = bool(invert)

    def is_invert_left_horizontal(self) -> bool:
        with QMutexLocker(self._lock):
            return self.config.invert_left_horizontal

    def set_invert_right_horizontal(self, invert: bool):
        with QMutexLocker(self._lock):
            self.config.invert_right_horizontal = bool(invert)

    def is_invert_right_horizontal(self) -> bool:
        with QMutexLocker(self._lock):
            return self.config.invert_right_horizontal

    def set_invert_horizontal(self, invert: bool):
        with QMutexLocker(self._lock):
            self.config.invert_left_horizontal = bool(invert)
            self.config.invert_right_horizontal = bool(invert)

    def is_invert_horizontal(self) -> bool:
        with QMutexLocker(self._lock):
            return self.config.invert_left_horizontal and self.config.invert_right_horizontal

    def set_enabled(self, enabled: bool):
        with QMutexLocker(self._lock):
            self._enabled = enabled
            if not enabled and self._is_moving:
                self._send_stop()
        self._emit_teleop_state()

    def is_enabled(self) -> bool:
        with QMutexLocker(self._lock):
            return self._enabled

    def set_inhibited(self, inhibited: bool, reason: str = ""):
        with QMutexLocker(self._lock):
            self._inhibited = inhibited
            self._inhibition_reason = reason
            if inhibited and self._is_moving:
                self._send_stop()
        self._emit_teleop_state()

    def is_inhibited(self) -> bool:
        with QMutexLocker(self._lock):
            return self._inhibited

    def get_inhibition_reason(self) -> str:
        with QMutexLocker(self._lock):
            return self._inhibition_reason

    def is_controller_connected(self) -> bool:
        with QMutexLocker(self._lock):
            return self._is_controller_connected

    def has_active_teleop_input(self) -> bool:
        """
        Returns True if the user is actively commanding the robot via gamepad:
        any analog stick deflected beyond deadzone, trigger squeezed, bumper held,
        or gripper button (A/B) pressed.
        """
        with QMutexLocker(self._lock):
            if self.mock:
                return any(abs(x) > 1e-4 for x in self._latest_state)
            if not self._is_controller_connected or self._joystick is None:
                return False
            # Check non-zero analog velocity axes: [vx, vy, vz, w_base, w_wrist2, w_wrist1]
            if any(abs(x) > 1e-4 for x in self._latest_state):
                return True
            # Check gripper buttons (A / B) or shoulder buttons
            try:
                num_buttons = self._joystick.get_numbuttons()
                if self.config.button_gripper_close < num_buttons and bool(self._joystick.get_button(self.config.button_gripper_close)):
                    return True
                if self.config.button_gripper_open < num_buttons and bool(self._joystick.get_button(self.config.button_gripper_open)):
                    return True
                if self.config.button_shoulder_left < num_buttons and bool(self._joystick.get_button(self.config.button_shoulder_left)):
                    return True
                if self.config.button_shoulder_right < num_buttons and bool(self._joystick.get_button(self.config.button_shoulder_right)):
                    return True
            except Exception:
                pass
            return False

    def get_controller_name(self) -> str:
        with QMutexLocker(self._lock):
            return self._controller_name

    def set_velocity_callback(self, cb: Optional[Callable[[List[float]], None]]):
        with QMutexLocker(self._lock):
            self._velocity_callback = cb

    def set_external_socket(self, sock: Optional[socket.socket]):
        with QMutexLocker(self._lock):
            self._external_socket = sock

    def emergency_stop(self):
        with QMutexLocker(self._lock):
            self._send_stop()

    def stop(self):
        self._running = False
        with QMutexLocker(self._lock):
            self._send_stop()
        self.wait(2000)
        self._disconnect_socket()

    # -------------------------------------------------------------------------
    # Socket Connection Management
    # -------------------------------------------------------------------------

    def _ensure_socket_connected(self) -> bool:
        """Maintains an active TCP connection to UR5e port 30002."""
        if self.mock:
            return True

        if self._external_socket is not None:
            return True

        if self._socket is not None:
            return True

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((self.robot_ip, self.config.robot_port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(0.5)
            self._socket = sock
            print(f"[GAMEPAD] Connected to robot command socket at {self.robot_ip}:{self.config.robot_port}")
            return True
        except Exception as e:
            print(f"[GAMEPAD ERROR] Failed to connect to robot at {self.robot_ip}:{self.config.robot_port} - {e}")
            self._socket = None
            return False

    def _disconnect_socket(self):
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def _send_urscript(self, script: str):
        """Sends raw URScript text to the robot command socket."""
        if not script.endswith("\n"):
            script += "\n"

        if self.mock:
            self.command_sent.emit(script.strip())
            return

        sock = self._external_socket if self._external_socket is not None else self._socket
        if sock is None:
            if not self._ensure_socket_connected():
                return
            sock = self._external_socket if self._external_socket is not None else self._socket

        try:
            sock.sendall(script.encode("utf-8"))
            self.command_sent.emit(script.strip())
        except (socket.error, BrokenPipeError, ConnectionResetError) as e:
            print(f"[GAMEPAD ERROR] URScript transmission failed: {e}")
            if self._socket is not None and sock is self._socket:
                self._disconnect_socket()

    def _send_stop(self):
        """Issues smooth stop command based on active mode."""
        if self._active_mode == "joint":
            cmd = format_stopj(self.config.joint_acceleration)
        else:
            cmd = format_stopl(self.config.linear_acceleration)

        self._send_urscript(cmd)
        self._is_moving = False
        self._active_mode = None
        self._latest_state = [0.0] * 6
        self._current_commanded_cart = [0.0] * 6
        self._current_commanded_joint = [0.0] * 6
        self.velocity_updated.emit([0.0] * 6)

    def _emit_teleop_state(self):
        """Emits human-readable state summary for UI badges."""
        with QMutexLocker(self._lock):
            if not self._enabled:
                msg = "Disabled (Standby)"
                active = False
            elif self._inhibited:
                msg = f"Inhibited ({self._inhibition_reason or 'Interlock active'})"
                active = False
            elif not self._is_controller_connected:
                msg = "Waiting for Gamepad"
                active = False
            else:
                msg = f"Active ({self._controller_name})"
                active = True
        self.teleop_state_changed.emit(active, msg)

    # -------------------------------------------------------------------------
    # Gamepad Hardware Discovery & Input Processing
    # -------------------------------------------------------------------------

    def _init_joystick(self) -> bool:
        """Scans and initializes the first available gamepad via Pygame."""
        if not PYGAME_AVAILABLE:
            return False

        if not pygame.joystick.get_init():
            pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count > 0:
            if self._joystick is None or not self._is_controller_connected:
                try:
                    js = pygame.joystick.Joystick(0)
                    js.init()
                    self._joystick = js
                    self._controller_name = js.get_name().strip()
                    self._is_controller_connected = True
                    if self.config == XBOX_CONFIG:
                        self.config = get_recommended_config_for_name(self._controller_name)
                    self.status_updated.emit(True, self._controller_name)
                    self._emit_teleop_state()
                    return True
                except Exception as err:
                    self._joystick = None
                    self._is_controller_connected = False
                    self._controller_name = f"Error: {err}"
                    self.status_updated.emit(False, self._controller_name)
                    self._emit_teleop_state()
                    return False
            return True
        else:
            if self._is_controller_connected or self._joystick is not None:
                self._joystick = None
                self._is_controller_connected = False
                self._controller_name = "No Gamepad Detected"
                self.status_updated.emit(False, self._controller_name)
                self._emit_teleop_state()
            return False

    def _read_trigger(self, axis_idx: int, is_left: bool) -> float:
        if self._joystick is None:
            return 0.0

        num_axes = self._joystick.get_numaxes()
        if axis_idx >= num_axes:
            return 0.0

        raw = float(self._joystick.get_axis(axis_idx))

        was_activated = self._lt_activated if is_left else self._rt_activated
        if not was_activated:
            if abs(raw) < 1e-3:
                return 0.0
            if is_left:
                self._lt_activated = True
            else:
                self._rt_activated = True

        norm = (raw + 1.0) / 2.0
        norm = max(0.0, min(1.0, norm))

        if norm <= self.config.trigger_deadzone:
            return 0.0
        return (norm - self.config.trigger_deadzone) / (1.0 - self.config.trigger_deadzone)

    def _read_all_inputs(self) -> Tuple[List[float], List[float], List[float]]:
        """
        Polls all sticks, bumpers, triggers, and special buttons (Select / Start).
        Returns:
          - cart_vel:  [vx, vy, vz, 0, 0, 0] (Tool 3D linear positioning velocity in space)
          - joint_vel: [w_base, 0, 0, w_wrist1, w_wrist2, 0] (Direct joint commands)
          - combined:  [vx, vy, vz, w_base, w_wrist2, w_wrist1]
        """
        if self._joystick is None:
            return [0.0] * 6, [0.0] * 6, [0.0] * 6

        num_axes = self._joystick.get_numaxes()
        num_buttons = self._joystick.get_numbuttons()

        # ---------------------------------------------------------------------
        # 0. Edge Detection for Select (Go to Start) & Start (Toggle Record)
        # ---------------------------------------------------------------------
        btn_sel_now = (
            (self.config.button_select < num_buttons and bool(self._joystick.get_button(self.config.button_select)))
            or (6 < num_buttons and bool(self._joystick.get_button(6)))
            or (8 < num_buttons and bool(self._joystick.get_button(8)))
        )
        if btn_sel_now and not self._prev_btn_select:
            self.button_select_pressed.emit()
        self._prev_btn_select = btn_sel_now

        btn_start_now = (
            (self.config.button_start < num_buttons and bool(self._joystick.get_button(self.config.button_start)))
            or (7 < num_buttons and bool(self._joystick.get_button(7)))
            or (9 < num_buttons and bool(self._joystick.get_button(9)))
        )
        if btn_start_now and not self._prev_btn_start:
            self.button_start_pressed.emit()
        self._prev_btn_start = btn_start_now

        # ---------------------------------------------------------------------
        # 1. Cartesian Translation: Left Stick (X/Y) & Right Stick (Z)
        # ---------------------------------------------------------------------
        raw_lx = self._joystick.get_axis(self.config.axis_left_x) if self.config.axis_left_x < num_axes else 0.0
        raw_ly = self._joystick.get_axis(self.config.axis_left_y) if self.config.axis_left_y < num_axes else 0.0
        raw_ry = self._joystick.get_axis(self.config.axis_right_y) if self.config.axis_right_y < num_axes else 0.0

        # Dedicated exponential response curve for Left Stick (fine, non-twitchy control near center)
        left_exp = getattr(self.config, "left_stick_curve_exponent", 2.2)
        scale_x = apply_deadzone(raw_ly, self.config.deadzone, left_exp)
        scale_y = apply_deadzone(raw_lx, self.config.deadzone, left_exp)

        # Right stick Z uses standard exponent (preserving user-approved Right Stick feel)
        scale_z = apply_deadzone(raw_ry, self.config.deadzone, self.config.velocity_curve_exponent)

        left_lr_mult = -1.0 if self.config.invert_left_horizontal else 1.0
        right_lr_mult = -1.0 if self.config.invert_right_horizontal else 1.0

        # Left stick translational speed scaled down for smooth, calm, controllable teleoperation
        left_v_max = self.config.max_linear_velocity * getattr(self.config, "left_stick_speed_mult", 0.45)
        vfwd = scale_x * self.config.invert_x * left_v_max
        vlat = scale_y * self.config.invert_y * left_v_max * left_lr_mult
        vz = scale_z * self.config.invert_z * self.config.max_linear_velocity

        # Gripper Tool-Gaze Directional Mapping:
        # Pushing UP moves forward along the actual line of sight / pointing direction of the gripper
        tcp_pose = self.get_tcp_pose()
        has_tool_rot = tcp_pose is not None and len(tcp_pose) >= 6 and any(abs(x) > 1e-4 for x in tcp_pose[3:6])

        if has_tool_rot and getattr(self.config, "direction_mode", "tool_gaze") == "tool_gaze":
            R_mat = rotvec_to_matrix(tcp_pose[3:6])
            # Tool Z axis (column 2) is the pointing / gaze direction of the gripper
            u_fwd = [R_mat[0][2], R_mat[1][2], R_mat[2][2]]

            # Lateral vector: horizontal right relative to gripper gaze (u_fwd x [0,0,1])
            u_right = [
                u_fwd[1] * 1.0 - u_fwd[2] * 0.0,
                u_fwd[2] * 0.0 - u_fwd[0] * 1.0,
                0.0,
            ]
            norm_r = math.sqrt(u_right[0]**2 + u_right[1]**2)
            if norm_r > 1e-3:
                u_lat = [u_right[0] / norm_r, u_right[1] / norm_r, 0.0]
            else:
                u_lat = [R_mat[0][0], R_mat[1][0], R_mat[2][0]]

            vx = vfwd * u_fwd[0] + vlat * u_lat[0]
            vy = vfwd * u_fwd[1] + vlat * u_lat[1]
            vz_total = vz + (vfwd * u_fwd[2] + vlat * u_lat[2])
        else:
            # Fallback to heading-relative cylindrical transformation
            theta = self.get_base_angle() + self.config.heading_offset_rad
            if self.config.heading_relative:
                vx = vfwd * math.cos(theta) - vlat * math.sin(theta)
                vy = vfwd * math.sin(theta) + vlat * math.cos(theta)
            else:
                vx = vfwd
                vy = vlat
            vz_total = vz

        cart_vel = [vx, vy, vz_total, 0.0, 0.0, 0.0]

        # ---------------------------------------------------------------------
        # 2. Direct Base Motor Rotation: Right Stick (X) -> shoulder_pan
        # ---------------------------------------------------------------------
        raw_rx = self._joystick.get_axis(self.config.axis_right_x) if self.config.axis_right_x < num_axes else 0.0
        scale_base = apply_deadzone(raw_rx, self.config.deadzone, self.config.velocity_curve_exponent)
        w_base = scale_base * self.config.invert_base * self.config.max_base_velocity * right_lr_mult

        # ---------------------------------------------------------------------
        # 3. Direct Wrist 2 Rotation: Shoulder Buttons (LB / RB, or L1 / R1)
        # ---------------------------------------------------------------------
        btn_lb = bool(self._joystick.get_button(self.config.button_shoulder_left)) if self.config.button_shoulder_left < num_buttons else False
        btn_rb = bool(self._joystick.get_button(self.config.button_shoulder_right)) if self.config.button_shoulder_right < num_buttons else False

        w_wrist2 = 0.0
        if btn_rb and not btn_lb:
            w_wrist2 = +self.config.max_wrist2_velocity * self.config.invert_wrist2
        elif btn_lb and not btn_rb:
            w_wrist2 = -self.config.max_wrist2_velocity * self.config.invert_wrist2

        # ---------------------------------------------------------------------
        # 4. Direct Wrist 1 Rotation: Triggers (LT / RT)
        # ---------------------------------------------------------------------
        lt_val = self._read_trigger(self.config.axis_trigger_left, is_left=True)
        rt_val = self._read_trigger(self.config.axis_trigger_right, is_left=False)

        trigger_diff = rt_val - lt_val
        w_wrist1 = trigger_diff * self.config.invert_wrist1 * self.config.max_wrist1_velocity

        joint_vel = [w_base, 0.0, 0.0, w_wrist1, w_wrist2, 0.0]
        combined = [vx, vy, vz, w_base, w_wrist2, w_wrist1]

        return cart_vel, joint_vel, combined

    # -------------------------------------------------------------------------
    # Main High-Frequency Execution Loop
    # -------------------------------------------------------------------------

    def run(self):
        """
        Background loop streaming commands at 20-50 Hz (default 35 Hz).
        Features:
          - Tool 3D Cartesian translation in space via `speedl` with Wrist 3 decoupled
          - Direct Joint Rotations (Base, Wrist 1, Wrist 2) via `speedj`
          - Software acceleration slew-rate limiting protecting Wrist 1
        """
        self._running = True

        if PYGAME_AVAILABLE:
            try:
                pygame.init()
                pygame.joystick.init()
            except Exception as e:
                print(f"[GAMEPAD ERROR] Failed to initialize Pygame: {e}", file=sys.__stderr__)

        poll_period = 1.0 / max(10.0, min(100.0, self.config.poll_rate_hz))
        scan_timer = 0.0

        if not self.mock:
            self._ensure_socket_connected()

        while self._running:
            t0 = time.perf_counter()

            # Periodic gamepad discovery / hot-plug check (~every 1.0s)
            scan_timer += poll_period
            if scan_timer >= 1.0:
                scan_timer = 0.0
                self._init_joystick()

            if PYGAME_AVAILABLE:
                try:
                    pygame.event.pump()
                except Exception:
                    pass

            if self._is_controller_connected and self._joystick is not None:
                cart_vel, joint_vel, combined = self._read_all_inputs()
                self._latest_state = combined

                # Integrate base rotation into internal heading tracker if no external provider is set
                if self._base_angle_provider is None and abs(joint_vel[0]) > 1e-4:
                    with QMutexLocker(self._lock):
                        self._current_base_angle += joint_vel[0] * poll_period

                has_cart = any(abs(v) > 1e-4 for v in cart_vel[:3])
                has_joint = any(abs(w) > 1e-4 for w in (joint_vel[0], joint_vel[3], joint_vel[4]))

                # ---------------------------------------------------------------------
                # Gripper Teleoperation: Button A (Close) & Button B (Open)
                # Discrete mode (default) snaps cleanly between min_w (55mm) and max_w (105mm)
                # so imitation learning datasets receive a sharp, unmistakable open/close signal.
                # ---------------------------------------------------------------------
                num_buttons = self._joystick.get_numbuttons()
                btn_close = (
                    self.config.button_gripper_close < num_buttons
                    and bool(self._joystick.get_button(self.config.button_gripper_close))
                )
                btn_open = (
                    self.config.button_gripper_open < num_buttons
                    and bool(self._joystick.get_button(self.config.button_gripper_open))
                )

                if self.config.gripper_discrete_mode:
                    if not self._inhibited:
                        min_w = getattr(self.gripper, "min_width_mm", 55.0) if self.gripper else 55.0
                        max_w = getattr(self.gripper, "max_width_mm", 105.0) if self.gripper else 105.0

                        # Rising-edge discrete CLOSE on Button [A]
                        if btn_close and not self._prev_btn_gripper_close and not btn_open:
                            with QMutexLocker(self._lock):
                                self._gripper_target_width = float(min_w)
                                target_to_send = self._gripper_target_width
                            self._last_gripper_cmd_time = time.perf_counter()
                            if self.gripper is not None and hasattr(self.gripper, "grip"):
                                try:
                                    self.gripper.grip(target_to_send, speed_percent=80, wait=False)
                                except Exception:
                                    pass
                            self.gripper_width_updated.emit(target_to_send)

                        # Rising-edge discrete OPEN on Button [B]
                        elif btn_open and not self._prev_btn_gripper_open and not btn_close:
                            with QMutexLocker(self._lock):
                                self._gripper_target_width = float(max_w)
                                target_to_send = self._gripper_target_width
                            self._last_gripper_cmd_time = time.perf_counter()
                            if self.gripper is not None and hasattr(self.gripper, "grip"):
                                try:
                                    self.gripper.grip(target_to_send, speed_percent=80, wait=False)
                                except Exception:
                                    pass
                            self.gripper_width_updated.emit(target_to_send)

                    self._prev_btn_gripper_close = btn_close
                    self._prev_btn_gripper_open = btn_open
                else:
                    if (btn_close or btn_open) and not self._inhibited:
                        self._gripper_button_active = True
                        min_w = getattr(self.gripper, "min_width_mm", 55.0) if self.gripper else 55.0
                        max_w = getattr(self.gripper, "max_width_mm", 105.0) if self.gripper else 105.0

                        delta = 0.0
                        if btn_close and not btn_open:
                            delta = -self.config.gripper_speed_mms * poll_period
                        elif btn_open and not btn_close:
                            delta = +self.config.gripper_speed_mms * poll_period

                        if abs(delta) > 1e-4:
                            with QMutexLocker(self._lock):
                                self._gripper_target_width = max(min_w, min(max_w, self._gripper_target_width + delta))
                                target_to_send = self._gripper_target_width

                            now = time.perf_counter()
                            if now - self._last_gripper_cmd_time >= 0.065:
                                self._last_gripper_cmd_time = now
                                if self.gripper is not None and hasattr(self.gripper, "grip"):
                                    try:
                                        self.gripper.grip(target_to_send, speed_percent=50, wait=False)
                                    except Exception:
                                        pass
                                self.gripper_width_updated.emit(target_to_send)
                    elif self._gripper_button_active:
                        self._gripper_button_active = False
                        with QMutexLocker(self._lock):
                            final_w = self._gripper_target_width
                        if self.gripper is not None and hasattr(self.gripper, "grip"):
                            try:
                                self.gripper.grip(final_w, speed_percent=50, wait=False)
                            except Exception:
                                pass
                        self.gripper_width_updated.emit(final_w)

                with QMutexLocker(self._lock):
                    can_stream = self._enabled and not self._inhibited

                if can_stream:
                    if has_joint or has_cart:
                        norm_joint = max(
                            abs(joint_vel[0]) / max(1e-4, self.config.max_base_velocity),
                            abs(joint_vel[3]) / max(1e-4, self.config.max_wrist1_velocity),
                            abs(joint_vel[4]) / max(1e-4, self.config.max_wrist2_velocity),
                        )
                        norm_cart = max(
                            abs(cart_vel[0]) / max(1e-4, self.config.max_linear_velocity),
                            abs(cart_vel[1]) / max(1e-4, self.config.max_linear_velocity),
                            abs(cart_vel[2]) / max(1e-4, self.config.max_linear_velocity),
                        )

                        if norm_joint >= norm_cart:
                            next_mode = "joint"
                            if self._active_mode is not None and self._active_mode != next_mode:
                                self._send_stop()

                            # Apply acceleration slew-rate limiting on joint velocities
                            max_dj = self.config.joint_acceleration * poll_period
                            for i in range(6):
                                delta = joint_vel[i] - self._current_commanded_joint[i]
                                delta = max(-max_dj, min(max_dj, delta))
                                self._current_commanded_joint[i] += delta

                            cmd = format_speedj(
                                self._current_commanded_joint,
                                a=self.config.joint_acceleration,
                                t=self.config.time_step,
                            )
                            self._send_urscript(cmd)
                            self._active_mode = "joint"
                            self._is_moving = True
                        else:
                            next_mode = "cartesian"
                            if self._active_mode is not None and self._active_mode != next_mode:
                                self._send_stop()

                            # Apply acceleration slew-rate limiting on Cartesian velocities
                            max_dc = self.config.linear_acceleration * poll_period
                            for i in range(3):
                                delta = cart_vel[i] - self._current_commanded_cart[i]
                                delta = max(-max_dc, min(max_dc, delta))
                                self._current_commanded_cart[i] += delta

                            cmd = format_speedl(
                                self._current_commanded_cart,
                                a=self.config.linear_acceleration,
                                t=self.config.time_step,
                            )
                            self._send_urscript(cmd)
                            self._active_mode = "cartesian"
                            self._is_moving = True

                        self.velocity_updated.emit(combined)
                        if self._velocity_callback is not None:
                            self._velocity_callback(combined)
                    else:
                        # Neutral sticks/buttons: stop smoothly
                        if self._is_moving:
                            self._send_stop()
                else:
                    if self._is_moving:
                        self._send_stop()
            else:
                if self._is_moving:
                    self._send_stop()

            # Accurate rate throttling
            elapsed = time.perf_counter() - t0
            sleep_time = poll_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Cleanup on exit
        self._send_stop()
        self._disconnect_socket()


# =============================================================================
# 4. Standalone CLI Diagnostic Runner
# =============================================================================

def run_standalone():
    """Terminal diagnostic runner to verify gamepad axes, buttons, and URScript streaming."""
    parser = argparse.ArgumentParser(description="UR5e Gamepad Teleoperation Diagnostic Runner")
    parser.add_argument("--robot-ip", type=str, default="192.168.1.102", help="UR5e Robot IP address")
    parser.add_argument("--mock", action="store_true", help="Run in mock simulation mode without hardware")
    parser.add_argument("--speed", type=int, default=50, help="Master speed scale percentage (10 to 100)")
    parser.add_argument("--accel", type=int, default=50, help="Master acceleration scale percentage (10 to 100)")
    parser.add_argument("--invert-left-lr", action="store_true", help="Invert left/right lateral movement on Left Stick")
    parser.add_argument("--invert-right-lr", action="store_true", help="Invert left/right base motor rotation on Right Stick")
    parser.add_argument("--invert-lr", action="store_true", help="Invert left/right movement on BOTH analog sticks")
    args = parser.parse_args()

    inv_left = args.invert_left_lr or args.invert_lr
    inv_right = args.invert_right_lr or args.invert_lr

    config = GamepadConfig(
        speed_scale=max(0.10, min(1.0, args.speed / 100.0)),
        accel_scale=max(0.10, min(1.0, args.accel / 100.0)),
        invert_left_horizontal=inv_left,
        invert_right_horizontal=inv_right,
    )

    print("=" * 80)
    print("🎮  UR5e Real-Time Gamepad Teleoperation Controller")
    print("=" * 80)
    print(f"Target Robot IP : {args.robot_ip} (Mock Mode: {args.mock})")
    print(f"Speed Setting   : {config.speed_scale*100:.0f}% (Max Linear: {config.max_linear_velocity*100:.1f} cm/s)")
    print(f"Accel Setting   : {config.accel_scale*100:.0f}% (Linear Accel: {config.linear_acceleration:.2f} m/s², Joint Accel: {config.joint_acceleration:.2f} rad/s²)")
    print(f"L/R Inversion   : Left Stick={inv_left} | Right Stick={inv_right}")
    print("-" * 80)
    print("Controls:")
    print("  • Left Stick (XY)     : Tool Position Forward/Backward (+X/-X) and Left/Right (+Y/-Y) in arm heading")
    print("  • Right Stick (Y)     : Tool Position Vertical Up/Down (+Z/-Z)")
    print("  • Right Stick (X)     : DIRECT Base Motor Rotation (shoulder_pan)")
    print("  • Shoulders (LB/RB)   : DIRECT Wrist 2 Rotation")
    print("  • Triggers (LT/RT)    : DIRECT Wrist 1 Rotation")
    print("  • Button A (Hold)     : Slowly CLOSE Gripper (OnRobot 2FG)")
    print("  • Button B (Hold)     : Slowly OPEN Gripper (OnRobot 2FG)")
    print("  • Select Button       : Go to Start / Home Position")
    print("  • Start Button        : Start / End Episode Recording (Space)")
    print("  • Note                : Wrist 3 is strictly decoupled from analog stick movement")
    print("  • Neutral Sticks/Btns : Smooth deceleration ramping -> stopl/stopj (0 drift)")
    print("=" * 80)

    teleop = UR5eGamepadTeleop(robot_ip=args.robot_ip, config=config, mock=args.mock)
    teleop.set_enabled(True)

    try:
        from onrobot_gripper import get_shared_gripper
        gripper = get_shared_gripper(auto_connect=not args.mock)
        if gripper.is_connected:
            teleop.set_gripper(gripper)
            print(f"[GAMEPAD] OnRobot 2FG Gripper connected! (Hold [A] to Close, [B] to Open)")
    except Exception as e:
        print(f"[GAMEPAD] Gripper init notice: {e}")

    def on_status(conn: bool, name: str):
        status = "CONNECTED" if conn else "DISCONNECTED"
        print(f"\n[GAMEPAD] Status: {status} -> {name}")

    def on_vel(v: List[float]):
        vx, vy, vz, w_base, w_w2, w_w1 = v
        has_motion = any(abs(x) > 1e-4 for x in v)
        if has_motion:
            heading_deg = math.degrees(teleop.get_base_angle())
            print(
                f"\r\033[K[TELEOP] TCP:[{vx:+5.2f},{vy:+5.2f},{vz:+5.2f}] m/s "
                f"| Base:{w_base:+5.2f} | W1:{w_w1:+5.2f} | W2:{w_w2:+5.2f} rad/s [Heading:{heading_deg:+4.0f}°]",
                end="",
                flush=True,
            )

    def on_select():
        print("\n[GAMEPAD BUTTON] 'Select' pressed -> Trigger Go to Start Pose")

    def on_start_btn():
        print("\n[GAMEPAD BUTTON] 'Start' pressed -> Trigger Episode Recording Toggle")

    teleop.status_updated.connect(on_status)
    teleop.velocity_updated.connect(on_vel)
    teleop.button_select_pressed.connect(on_select)
    teleop.button_start_pressed.connect(on_start_btn)

    teleop.start()

    print("\nTeleoperation loop running. Move sticks / press buttons to test (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nShutting down teleoperation...")
    finally:
        teleop.stop()
        print("Done.")


if __name__ == "__main__":
    run_standalone()
