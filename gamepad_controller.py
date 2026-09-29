#!/usr/bin/env python3
"""
UR5e Gamepad Teleoperation Controller
=====================================
Dedicated module providing real-time Cartesian (TCP / end-effector) and direct joint
teleoperation for the Universal Robots UR5e arm using a standard USB/Bluetooth gamepad via Pygame.

Movement Scheme:
- Left Stick (X/Y)       : TCP Translation Forward/Backward (+X/-X) and Left/Right (+Y/-Y) via `speedl`
- Right Stick (Y)        : TCP Translation Vertical Up/Down (+Z/-Z) via `speedl`
- Right Stick (X)        : DIRECT Base motor rotation (joint 0 / shoulder_pan) via `speedj`
- Shoulder Buttons (LB/RB: DIRECT Wrist 2 rotation (joint 4 / wrist_2) via `speedj`
- Triggers (LT/RT)       : DIRECT Wrist 3 rotation (joint 5 / wrist_3) via `speedj`
- Neutral Sticks/Buttons : Instant `stopl` / `stopj` braking (zero drift)

Features:
- Configurable axis and button mapping dataclass with support for Xbox, PlayStation, and generic pads.
- Analog deadzone filtering (10%) and smooth velocity curve scaling.
- Seamless automatic switching between Cartesian tool translation (`speedl`) and direct joint rotation (`speedj`).
- High-frequency background polling loop running at 20-50 Hz (default 35 Hz).
- Thread-safe safety interlocks for integration into PyQt6 GUI recorder.
- Standalone CLI runner for direct terminal diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
import socket
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

# Enable background events in SDL before initializing Pygame so gamepads work without a focused window
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

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


# =============================================================================
# 1. Modular Gamepad Configuration Mapping
# =============================================================================

@dataclass
class GamepadConfig:
    """
    Configurable hardware mappings, deadzones, and velocity scaling for UR5e teleoperation.
    Standard Xbox / SDL2 layout on Linux:
      - Left Stick X (Axis 0)      : Left / Right TCP translation (-Y / +Y)
      - Left Stick Y (Axis 1)      : Forward / Backward TCP translation (+X / -X)
      - Right Stick X (Axis 3)     : Direct Base motor rotation (-shoulder_pan / +shoulder_pan)
      - Right Stick Y (Axis 4)     : Up / Down vertical TCP translation (+Z / -Z)
      - Shoulder Buttons (Btn 4/5) : Direct Wrist 2 rotation (LB = -wrist_2, RB = +wrist_2)
      - Triggers (Axis 2/5)        : Direct Wrist 3 rotation (LT = -wrist_3, RT = +wrist_3)
    """
    # Analog axis channel assignments
    axis_left_x: int = 0
    axis_left_y: int = 1
    axis_right_x: int = 3
    axis_right_y: int = 4
    axis_trigger_left: int = 2
    axis_trigger_right: int = 5

    # Button channel assignments (Shoulder bumpers / L1 & R1)
    button_shoulder_left: int = 4   # LB / L1: Wrist 2 CCW
    button_shoulder_right: int = 5  # RB / R1: Wrist 2 CW

    # Inversions (1.0 or -1.0)
    # Pygame returns -1.0 for pushing sticks UP/FORWARD; inverted to +1.0 for UR5e base frame
    invert_x: float = -1.0          # Stick forward -> +X (TCP forward in robot base frame)
    invert_y: float = 1.0           # Stick right -> +Y (TCP right in robot base frame)
    invert_z: float = -1.0          # Stick up -> +Z (TCP upward)
    invert_base: float = 1.0        # Right stick right -> +Base rotation
    invert_wrist2: float = 1.0      # RB -> +Wrist 2 rotation, LB -> -Wrist 2 rotation
    invert_wrist3: float = 1.0      # RT -> +Wrist 3 rotation, LT -> -Wrist 3 rotation

    # Deadzones & Scaling
    deadzone: float = 0.10             # Analog stick deadzone (10%)
    trigger_deadzone: float = 0.05     # Trigger deadzone (5%)
    max_linear_velocity: float = 0.15  # Max TCP translational speed (m/s) -> 15 cm/s
    max_base_velocity: float = 0.50    # Max Base angular speed (rad/s) -> ~28.6 deg/s
    max_wrist2_velocity: float = 0.50  # Max Wrist 2 angular speed (rad/s) -> ~28.6 deg/s
    max_wrist3_velocity: float = 0.75  # Max Wrist 3 angular speed (rad/s) -> ~43.0 deg/s
    velocity_curve_exponent: float = 1.0  # 1.0 = linear, 1.5-2.0 = exponential fine control

    # Orientation / Heading Frame settings
    heading_relative: bool = True           # Left Stick moves relative to current arm base heading
    heading_offset_rad: float = 0.0         # Angular calibration offset in radians (default 0.0)
    invert_left_horizontal: bool = False    # Invert Left/Right lateral translation on Left Stick
    invert_right_horizontal: bool = False   # Invert Left/Right base motor rotation on Right Stick

    # URScript execution parameters
    linear_acceleration: float = 0.5   # Tool acceleration for speedl (m/s^2)
    joint_acceleration: float = 1.2    # Joint acceleration for speedj (rad/s^2)
    time_step: float = 0.08            # URScript watchdog duration (seconds)
    poll_rate_hz: float = 35.0         # Gamepad poll & command stream frequency (Hz)
    robot_port: int = 30002            # Secondary client interface port for URScript


# Presets for popular controller families
XBOX_CONFIG = GamepadConfig(
    axis_left_x=0, axis_left_y=1,
    axis_right_x=3, axis_right_y=4,
    axis_trigger_left=2, axis_trigger_right=5,
    button_shoulder_left=4, button_shoulder_right=5,
    invert_x=-1.0, invert_y=1.0, invert_z=-1.0,
    invert_base=1.0, invert_wrist2=1.0, invert_wrist3=1.0,
)

PLAYSTATION_CONFIG = GamepadConfig(
    axis_left_x=0, axis_left_y=1,
    axis_right_x=2, axis_right_y=3,
    axis_trigger_left=4, axis_trigger_right=5,
    button_shoulder_left=4, button_shoulder_right=5,
    invert_x=-1.0, invert_y=1.0, invert_z=-1.0,
    invert_base=1.0, invert_wrist2=1.0, invert_wrist3=1.0,
)


def get_recommended_config_for_name(controller_name: str) -> GamepadConfig:
    """Selects suitable axis/button defaults based on detected controller name."""
    name_lower = controller_name.lower()
    if any(k in name_lower for k in ("playstation", "dualshock", "dualsense", "sony")):
        return PLAYSTATION_CONFIG
    return XBOX_CONFIG


# =============================================================================
# 2. Mathematical Utilities for Filtering & Velocity Computation
# =============================================================================

def apply_deadzone(value: float, deadzone: float, exponent: float = 1.0) -> float:
    """
    Applies linear or polynomial deadzone rescaling.
    Ensures velocity smoothly starts at 0.0 as soon as the deadzone threshold is crossed.
    """
    abs_val = abs(value)
    if abs_val <= deadzone:
        return 0.0

    # Rescale [deadzone, 1.0] -> [0.0, 1.0]
    scaled = (abs_val - deadzone) / (1.0 - deadzone)
    scaled = min(1.0, max(0.0, scaled))

    if exponent != 1.0:
        scaled = math.pow(scaled, exponent)

    return math.copysign(scaled, value)


def format_speedl(v: List[float], a: float = 0.5, t: float = 0.08) -> str:
    """Formats a Cartesian TCP velocity command as a standard URScript string."""
    v_str = ", ".join(f"{val:.4f}" for val in v)
    return f"speedl([{v_str}], a={a:.2f}, t={t:.2f})\n"


def format_speedj(qd: List[float], a: float = 1.2, t: float = 0.08) -> str:
    """Formats a joint velocity command as a standard URScript string."""
    qd_str = ", ".join(f"{val:.4f}" for val in qd)
    return f"speedj([{qd_str}], a={a:.2f}, t={t:.2f})\n"


def format_stopl(a: float = 0.5) -> str:
    """Formats a URScript Cartesian stop command to smoothly decelerate to rest."""
    return f"stopl({a:.2f})\n"


def format_stopj(a: float = 1.5) -> str:
    """Formats a URScript joint stop command to smoothly decelerate to rest."""
    return f"stopj({a:.2f})\n"


# =============================================================================
# 3. UR5e Gamepad Teleoperation Worker Class
# =============================================================================

class UR5eGamepadTeleop(QThread):
    """
    High-frequency background worker thread for UR5e teleoperation.
    Supports Cartesian translation (`speedl`) and direct joint rotations (`speedj`).
    """
    # Qt Signals for GUI integration
    status_updated = pyqtSignal(bool, str)        # (is_connected, controller_name)
    velocity_updated = pyqtSignal(list)           # [vx, vy, vz, w_base, w_wrist2, w_wrist3]
    teleop_state_changed = pyqtSignal(bool, str)  # (is_active, status_message)
    command_sent = pyqtSignal(str)                # URScript command string

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
        self._active_mode: Optional[str] = None  # "cartesian", "joint", or None

        # Hardware connection & Gamepad state
        self._socket: Optional[socket.socket] = None
        self._external_socket: Optional[socket.socket] = None
        self._velocity_callback: Optional[Callable[[List[float]], None]] = None
        self._joystick: Optional[pygame.joystick.Joystick] = None
        self._controller_name: str = "No Gamepad Detected"
        self._is_controller_connected: bool = False

        # Trigger activation trackers (safeguards against SDL initial 0.0 unpressed value)
        self._lt_activated = False
        self._rt_activated = False

        # Latest combined telemetry vector: [vx, vy, vz, w_base, w_wrist2, w_wrist3]
        self._latest_state: List[float] = [0.0] * 6

        # Base orientation tracking for heading-relative Cartesian control (radians)
        self._current_base_angle: float = 0.0
        self._base_angle_provider: Optional[Callable[[], float]] = None

    # -------------------------------------------------------------------------
    # Public Control & Configuration API
    # -------------------------------------------------------------------------

    def set_base_angle(self, angle_rad: float):
        """Updates the current base motor angle (shoulder_pan) in radians."""
        with QMutexLocker(self._lock):
            self._current_base_angle = float(angle_rad)

    def get_base_angle(self) -> float:
        """Returns the active base motor angle (shoulder_pan) in radians."""
        with QMutexLocker(self._lock):
            if self._base_angle_provider is not None:
                try:
                    return float(self._base_angle_provider())
                except Exception:
                    pass
            return self._current_base_angle

    def set_base_angle_provider(self, provider_fn: Callable[[], float]):
        """Registers a callable returning the live robot base joint angle in radians."""
        with QMutexLocker(self._lock):
            self._base_angle_provider = provider_fn

    def set_invert_left_horizontal(self, invert: bool):
        """Toggles direction inversion for Left Stick lateral movement (Left/Right)."""
        with QMutexLocker(self._lock):
            self.config.invert_left_horizontal = bool(invert)

    def is_invert_left_horizontal(self) -> bool:
        """Returns True if Left Stick lateral movement is inverted."""
        with QMutexLocker(self._lock):
            return self.config.invert_left_horizontal

    def set_invert_right_horizontal(self, invert: bool):
        """Toggles direction inversion for Right Stick base motor rotation."""
        with QMutexLocker(self._lock):
            self.config.invert_right_horizontal = bool(invert)

    def is_invert_right_horizontal(self) -> bool:
        """Returns True if Right Stick base rotation is inverted."""
        with QMutexLocker(self._lock):
            return self.config.invert_right_horizontal

    def set_invert_horizontal(self, invert: bool):
        """Toggles direction inversion for left-to-right movement on BOTH analog sticks."""
        with QMutexLocker(self._lock):
            self.config.invert_left_horizontal = bool(invert)
            self.config.invert_right_horizontal = bool(invert)

    def is_invert_horizontal(self) -> bool:
        """Returns True if left-to-right movement is inverted on both sticks."""
        with QMutexLocker(self._lock):
            return self.config.invert_left_horizontal and self.config.invert_right_horizontal

    def set_enabled(self, enabled: bool):
        """Enables or disables teleoperation streaming."""
        with QMutexLocker(self._lock):
            self._enabled = enabled
            if not enabled and self._is_moving:
                self._send_stop()
        self._emit_teleop_state()

    def is_enabled(self) -> bool:
        with QMutexLocker(self._lock):
            return self._enabled

    def set_inhibited(self, inhibited: bool, reason: str = ""):
        """
        Temporarily silences gamepad commands during safety-critical events
        (e.g., active Freedrive mode or discrete Go to Start/Home motions).
        """
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

    def get_controller_name(self) -> str:
        with QMutexLocker(self._lock):
            return self._controller_name

    def set_velocity_callback(self, cb: Optional[Callable[[List[float]], None]]):
        """Optional callback invoked on every velocity update (useful for mock simulation)."""
        with QMutexLocker(self._lock):
            self._velocity_callback = cb

    def set_external_socket(self, sock: Optional[socket.socket]):
        """Allows sharing an existing robot command socket instead of creating a new one."""
        with QMutexLocker(self._lock):
            self._external_socket = sock

    def emergency_stop(self):
        """Immediately halts motion on the robot."""
        with QMutexLocker(self._lock):
            self._send_stop()

    def stop(self):
        """Cleanly halts teleoperation thread and releases network resources."""
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
            sock.setblocking(False)
            self._socket = sock
            return True
        except Exception:
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
        if self.mock:
            self.command_sent.emit(script.strip())
            return

        sock = self._external_socket if self._external_socket is not None else self._socket
        if sock is None:
            if not self._ensure_socket_connected():
                return
            sock = self._socket

        try:
            sock.sendall(script.encode("utf-8"))
            self.command_sent.emit(script.strip())
        except (socket.error, BrokenPipeError, ConnectionResetError):
            self._disconnect_socket()

    def _send_stop(self):
        """Issues appropriate stop command based on active mode."""
        if self._active_mode == "joint":
            cmd = format_stopj(self.config.joint_acceleration)
        else:
            cmd = format_stopl(self.config.linear_acceleration)

        self._send_urscript(cmd)
        self._is_moving = False
        self._active_mode = None
        self._latest_state = [0.0] * 6
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
                    # Auto-pick configuration preset if using generic defaults
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
        """
        Reads and normalizes an analog trigger axis (range 0.0 unpressed to 1.0 fully pressed).
        Safeguards against the SDL quirk where unpressed triggers initially report 0.0 before first touch.
        """
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
        Polls all sticks, bumpers, and triggers.
        Returns:
          - cart_vel:  [vx, vy, vz, 0, 0, 0]
          - joint_vel: [w_base, 0, 0, 0, w_wrist2, w_wrist3]
          - combined:  [vx, vy, vz, w_base, w_wrist2, w_wrist3]
        """
        if self._joystick is None:
            return [0.0] * 6, [0.0] * 6, [0.0] * 6

        num_axes = self._joystick.get_numaxes()
        num_buttons = self._joystick.get_numbuttons()

        # ---------------------------------------------------------------------
        # 1. Cartesian Translation: Left Stick (X/Y) & Right Stick (Z)
        # ---------------------------------------------------------------------
        raw_lx = self._joystick.get_axis(self.config.axis_left_x) if self.config.axis_left_x < num_axes else 0.0
        raw_ly = self._joystick.get_axis(self.config.axis_left_y) if self.config.axis_left_y < num_axes else 0.0
        raw_ry = self._joystick.get_axis(self.config.axis_right_y) if self.config.axis_right_y < num_axes else 0.0

        scale_x = apply_deadzone(raw_ly, self.config.deadzone, self.config.velocity_curve_exponent)
        scale_y = apply_deadzone(raw_lx, self.config.deadzone, self.config.velocity_curve_exponent)
        scale_z = apply_deadzone(raw_ry, self.config.deadzone, self.config.velocity_curve_exponent)

        left_lr_mult = -1.0 if self.config.invert_left_horizontal else 1.0
        right_lr_mult = -1.0 if self.config.invert_right_horizontal else 1.0

        vfwd = scale_x * self.config.invert_x * self.config.max_linear_velocity
        vlat = scale_y * self.config.invert_y * self.config.max_linear_velocity * left_lr_mult
        vz = scale_z * self.config.invert_z * self.config.max_linear_velocity

        # Transform (vfwd, vlat) into robot base frame coordinates based on active arm heading
        theta = self.get_base_angle() + self.config.heading_offset_rad
        if self.config.heading_relative:
            vx = vfwd * math.cos(theta) - vlat * math.sin(theta)
            vy = vfwd * math.sin(theta) + vlat * math.cos(theta)
        else:
            vx = vfwd
            vy = vlat

        cart_vel = [vx, vy, vz, 0.0, 0.0, 0.0]

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
        # 4. Direct Wrist 3 Rotation: Triggers (LT / RT)
        # ---------------------------------------------------------------------
        lt_val = self._read_trigger(self.config.axis_trigger_left, is_left=True)
        rt_val = self._read_trigger(self.config.axis_trigger_right, is_left=False)

        trigger_diff = rt_val - lt_val
        w_wrist3 = trigger_diff * self.config.invert_wrist3 * self.config.max_wrist3_velocity

        joint_vel = [w_base, 0.0, 0.0, 0.0, w_wrist2, w_wrist3]
        combined = [vx, vy, vz, w_base, w_wrist2, w_wrist3]

        return cart_vel, joint_vel, combined

    # -------------------------------------------------------------------------
    # Main High-Frequency Execution Loop
    # -------------------------------------------------------------------------

    def run(self):
        """
        Background loop streaming commands at 20-50 Hz (default 35 Hz).
        Automatically chooses between `speedl` (Cartesian) and `speedj` (direct joint).
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
                has_joint = any(abs(w) > 1e-4 for w in (joint_vel[0], joint_vel[4], joint_vel[5]))

                with QMutexLocker(self._lock):
                    can_stream = self._enabled and not self._inhibited

                if can_stream:
                    if has_joint or has_cart:
                        # Determine dominant motion mode
                        norm_joint = max(
                            abs(joint_vel[0]) / max(1e-4, self.config.max_base_velocity),
                            abs(joint_vel[4]) / max(1e-4, self.config.max_wrist2_velocity),
                            abs(joint_vel[5]) / max(1e-4, self.config.max_wrist3_velocity),
                        )
                        norm_cart = max(
                            abs(cart_vel[0]) / max(1e-4, self.config.max_linear_velocity),
                            abs(cart_vel[1]) / max(1e-4, self.config.max_linear_velocity),
                            abs(cart_vel[2]) / max(1e-4, self.config.max_linear_velocity),
                        )

                        if norm_joint >= norm_cart:
                            # Mode: Direct Joint Rotation (`speedj`)
                            cmd = format_speedj(
                                joint_vel,
                                a=self.config.joint_acceleration,
                                t=self.config.time_step,
                            )
                            self._send_urscript(cmd)
                            self._active_mode = "joint"
                            self._is_moving = True
                        else:
                            # Mode: Cartesian Translation (`speedl`)
                            cmd = format_speedl(
                                cart_vel,
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
    """Terminal diagnostic runner to verify gamepad axes and URScript speedl/speedj streaming."""
    parser = argparse.ArgumentParser(description="UR5e Gamepad Teleoperation Diagnostic Runner")
    parser.add_argument("--robot-ip", type=str, default="192.168.1.102", help="UR5e Robot IP address")
    parser.add_argument("--mock", action="store_true", help="Run in mock simulation mode without hardware")
    parser.add_argument("--max-lin", type=float, default=0.15, help="Max linear speed (m/s)")
    parser.add_argument("--max-base", type=float, default=0.50, help="Max Base angular speed (rad/s)")
    parser.add_argument("--max-w2", type=float, default=0.50, help="Max Wrist 2 angular speed (rad/s)")
    parser.add_argument("--max-w3", type=float, default=0.75, help="Max Wrist 3 angular speed (rad/s)")
    parser.add_argument("--invert-left-lr", action="store_true", help="Invert left/right lateral movement on Left Stick")
    parser.add_argument("--invert-right-lr", action="store_true", help="Invert left/right base motor rotation on Right Stick")
    parser.add_argument("--invert-lr", action="store_true", help="Invert left/right movement on BOTH analog sticks")
    args = parser.parse_args()

    inv_left = args.invert_left_lr or args.invert_lr
    inv_right = args.invert_right_lr or args.invert_lr

    print("=" * 75)
    print("🎮  UR5e Real-Time Teleoperation: Cartesian Translation & Direct Joint Control")
    print("=" * 75)
    print(f"Target Robot IP: {args.robot_ip} (Mock Mode: {args.mock})")
    print(f"Speed Limits: Linear = {args.max_lin} m/s | Base = {args.max_base} rad/s")
    print(f"              Wrist 2 = {args.max_w2} rad/s | Wrist 3 = {args.max_w3} rad/s")
    print(f"L/R Inversion: Left Stick={inv_left} | Right Stick={inv_right}")
    print("-" * 75)
    print("Controls:")
    print("  • Left Stick (XY)       : TCP Forward / Backward (+X/-X) and Left / Right (+Y/-Y)")
    print("  • Right Stick (Y)      : TCP Vertical Up / Down (+Z/-Z)")
    print("  • Right Stick (X)      : DIRECT Base Motor Rotation (shoulder_pan)")
    print("  • Shoulders (LB/RB)    : DIRECT Wrist 2 Rotation")
    print("  • Triggers (LT/RT)     : DIRECT Wrist 3 Rotation")
    print("  • Neutral Sticks/Btns  : Instant Braking (0 drift)")
    print("=" * 75)

    config = GamepadConfig(
        max_linear_velocity=args.max_lin,
        max_base_velocity=args.max_base,
        max_wrist2_velocity=args.max_w2,
        max_wrist3_velocity=args.max_w3,
        invert_left_horizontal=inv_left,
        invert_right_horizontal=inv_right,
    )

    teleop = UR5eGamepadTeleop(robot_ip=args.robot_ip, config=config, mock=args.mock)
    teleop.set_enabled(True)

    def on_status(conn: bool, name: str):
        status = "CONNECTED" if conn else "DISCONNECTED"
        print(f"\n[GAMEPAD] Status: {status} -> {name}")

    def on_vel(v: List[float]):
        vx, vy, vz, w_base, w_w2, w_w3 = v
        has_motion = any(abs(x) > 1e-4 for x in v)
        if has_motion:
            heading_deg = math.degrees(teleop.get_base_angle())
            print(
                f"\r\033[K[TELEOP] TCP:[{vx:+5.2f},{vy:+5.2f},{vz:+5.2f}] m/s "
                f"| Base:{w_base:+5.2f} | W2:{w_w2:+5.2f} | W3:{w_w3:+5.2f} rad/s [Heading:{heading_deg:+4.0f}°]",
                end="",
                flush=True,
            )

    teleop.status_updated.connect(on_status)
    teleop.velocity_updated.connect(on_vel)

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
