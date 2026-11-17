#!/usr/bin/env python3
"""
hybrid_path_runner.py
=====================
Hybrid Control Engine: Pre-Recorded Trajectory Playback with Real-Time
Gamepad Override and Minimum-Jerk Path Recovery for UR5e & OnRobot 2FG14.

State Machine Architecture:
  • STATE 1 (PLAYBACK):
      Streams the pre-recorded reference path (from reference_paths/*.csv)
      using closed-loop velocity tracking (speedj) at 50 Hz.
  • STATE 2 (OVERRIDE):
      The instant non-zero axis or button input is detected from the gamepad,
      playback halts immediately. Manual teleoperation takes over with zero latency.
  • STATE 3 (RECOVERY / INTERPOLATION):
      When gamepad input returns to zero for a user-specified grace period (e.g., 500 ms),
      the script captures the robot's current pose, searches the remaining unexecuted
      portion of the path for the nearest spatial waypoint (Euclidean distance), generates
      a smooth minimum-jerk polynomial trajectory to avoid acceleration spikes, and snaps
      the robot back to that waypoint before seamlessly resuming path playback.

Continuous LeRobot Dataset Recording:
  Throughout States 1, 2, and 3, gui_recorder records camera feeds, robot states,
  and timestamps as a single continuous episode trajectory.
"""

from __future__ import annotations

from enum import Enum
import math
from pathlib import Path
import socket
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from PyQt6.QtCore import QMutex, QMutexLocker, QObject, QThread, pyqtSignal
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    # Mock QThread/pyqtSignal for headless test environments
    class QObject: pass
    class QThread:
        def __init__(self, parent=None): pass
        def start(self): self.run()
        def isRunning(self): return False
        def wait(self, msecs=2000): pass
    class pyqtSignal:
        def __init__(self, *types): self._callbacks = []
        def connect(self, cb): self._callbacks.append(cb)
        def emit(self, *args):
            for cb in self._callbacks:
                try: cb(*args)
                except Exception: pass
    class QMutex:
        def lock(self): pass
        def unlock(self): pass
    class QMutexLocker:
        def __init__(self, mutex): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

from path_manager import TrajectoryPoint, PathManager


# =============================================================================
# 1. Hybrid State Enum Definition
# =============================================================================

class HybridState(Enum):
    IDLE = "IDLE"
    ALIGNING = "ALIGNING"
    PLAYBACK = "STATE 1: PLAYBACK"
    OVERRIDE = "STATE 2: OVERRIDE"
    RECOVERY = "STATE 3: RECOVERY"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"


# =============================================================================
# 2. Minimum-Jerk Polynomial Trajectory Mathematics
# =============================================================================

def min_jerk_step(tau: float) -> Tuple[float, float, float]:
    """
    Evaluates the 5th-order minimum-jerk polynomial:
      s(tau)   = 10*tau^3 - 15*tau^4 + 6*tau^5
      ds(tau)  = 30*tau^2 - 60*tau^3 + 30*tau^4
      dds(tau) = 60*tau - 180*tau^2 + 120*tau^3
    Boundary Conditions:
      s(0)=0, s(1)=1; ds(0)=ds(1)=0; dds(0)=dds(1)=0.
    Guarantees mathematically ZERO acceleration spikes at start and arrival.
    """
    tau = min(1.0, max(0.0, float(tau)))
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    ds = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    dds = 60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3
    return s, ds, dds


def generate_minimum_jerk_trajectory(
    q_start: List[float],
    q_target: List[float],
    duration_s: float,
    dt: float = 0.02,
) -> List[Tuple[float, List[float], List[float]]]:
    """
    Generates a dense waypoint sequence of (time_s, q_desired, qd_desired)
    connecting q_start to q_target with minimum jerk.
    """
    duration_s = max(0.4, float(duration_s))
    steps = max(2, int(math.ceil(duration_s / dt)))
    actual_dt = duration_s / steps

    q0 = np.array(q_start, dtype=np.float64)
    q1 = np.array(q_target, dtype=np.float64)
    delta_q = q1 - q0

    trajectory: List[Tuple[float, List[float], List[float]]] = []
    for step in range(steps + 1):
        t = step * actual_dt
        tau = min(1.0, t / duration_s)
        s, ds, _ = min_jerk_step(tau)
        q_t = q0 + delta_q * s
        qd_t = (delta_q / duration_s) * ds
        trajectory.append((t, q_t.tolist(), qd_t.tolist()))

    return trajectory


# =============================================================================
# 3. Spatial Nearest Waypoint Search
# =============================================================================

def find_nearest_waypoint(
    waypoints: List[TrajectoryPoint],
    current_tcp: List[float],
    current_q: List[float],
    start_idx: int = 0,
    search_ahead_only: bool = True,
    joint_weight: float = 0.06,
) -> Tuple[int, float]:
    """
    Searches remaining unexecuted portion of reference path [start_idx, N-1]
    to find the nearest spatial waypoint.
    
    Metric:
      D = d_cartesian(current_tcp, wp.tcp) + joint_weight * d_joint(current_q, wp.q)
    This prioritizes physical 3D spatial position while avoiding kinematic branch flips.
    
    Returns:
      (nearest_index, distance_meters)
    """
    if not waypoints:
        return 0, 0.0

    N = len(waypoints)
    from_idx = max(0, min(start_idx, N - 1)) if search_ahead_only else 0

    best_idx = from_idx
    best_dist = float("inf")

    has_tcp = (
        current_tcp is not None
        and len(current_tcp) >= 3
        and any(abs(x) > 1e-4 for x in current_tcp[:3])
    )

    p_curr = np.array(current_tcp[:3], dtype=np.float64) if has_tcp else None
    q_curr = np.array(current_q[:6], dtype=np.float64) if len(current_q) >= 6 else None

    for k in range(from_idx, N):
        wp = waypoints[k]
        
        # 1. Cartesian Euclidean distance
        if has_tcp and wp.tcp_pose and len(wp.tcp_pose) >= 3:
            p_wp = np.array(wp.tcp_pose[:3], dtype=np.float64)
            d_cart = float(np.linalg.norm(p_wp - p_curr))
        else:
            d_cart = 0.0

        # 2. Joint space maximum angular error
        if q_curr is not None and wp.q and len(wp.q) >= 6:
            q_wp = np.array(wp.q[:6], dtype=np.float64)
            d_joint = float(np.max(np.abs(q_wp - q_curr)))
        else:
            d_joint = 0.0

        # Composite score
        if has_tcp:
            score = d_cart + joint_weight * d_joint
        else:
            score = d_joint

        if score < best_dist:
            best_dist = score
            best_idx = k

    return best_idx, best_dist


# =============================================================================
# 4. Hybrid Path Runner Worker (QThread)
# =============================================================================

class HybridPathWorker(QThread):
    """
    Dedicated background thread executing the 3-State Hybrid Assist State Machine:
      - State 1: Follow pre-recorded trajectory
      - State 2: Instant teleop override on controller touch
      - State 3: Grace period expiration -> Minimum-jerk snapback -> Resume State 1
    """

    state_changed = pyqtSignal(str, str, str) # (state_name, badge_color_hex, description)
    progress_updated = pyqtSignal(int, int, float, float) # (curr_idx, total_pts, pct, dist_to_path_cm)
    snapback_started = pyqtSignal(int, float, float) # (target_idx, snapback_duration_s, max_dq_deg)
    log_message = pyqtSignal(str, str) # (level, message)
    finished = pyqtSignal(bool, str) # (success, message)

    def __init__(
        self,
        waypoints: List[TrajectoryPoint],
        robot_ip: str = "192.168.1.102",
        gamepad_teleop: Optional[Any] = None,
        robot_worker: Optional[Any] = None,
        gripper: Optional[Any] = None,
        override_delay_s: float = 0.50,
        playback_speed_scale: float = 1.0,
        mock: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.waypoints = list(waypoints)
        self.robot_ip = robot_ip
        self.gamepad_teleop = gamepad_teleop
        self.robot_worker = robot_worker
        self.gripper = gripper
        self.override_delay_s = max(0.0, float(override_delay_s))
        self.playback_speed_scale = max(0.1, min(3.0, float(playback_speed_scale)))
        self.mock = mock

        self._abort_flag = False
        self._lock = QMutex()
        self._current_state = HybridState.IDLE
        self._socket: Optional[socket.socket] = None

    def request_abort(self):
        """Immediately halts hybrid playback and sets abort flag."""
        with QMutexLocker(self._lock):
            self._abort_flag = True
        self._send_stop()

    def set_override_delay_s(self, delay_s: float):
        with QMutexLocker(self._lock):
            self.override_delay_s = max(0.0, float(delay_s))

    def set_playback_speed_scale(self, scale: float):
        with QMutexLocker(self._lock):
            self.playback_speed_scale = max(0.1, min(3.0, float(scale)))

    @property
    def current_state(self) -> HybridState:
        with QMutexLocker(self._lock):
            return self._current_state

    # -------------------------------------------------------------------------
    # Internal Hardware Socket & Command Helpers
    # -------------------------------------------------------------------------

    def _ensure_socket(self) -> bool:
        if self.mock:
            return True
        if self._socket is not None:
            return True
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((self.robot_ip, 30002))
            self._socket = s
            return True
        except Exception as e:
            self.log_message.emit("ERROR", f"[HYBRID] Failed to connect command socket: {e}")
            self._socket = None
            return False

    def _disconnect_socket(self):
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def _send_urscript(self, script: str):
        if not script.endswith("\n"):
            script += "\n"
        if self.mock:
            return
        if not self._ensure_socket():
            return
        try:
            self._socket.sendall(script.encode("utf-8"))
        except Exception as e:
            self.log_message.emit("WARNING", f"[HYBRID] Socket transmission error: {e}")
            self._disconnect_socket()

    def _send_speedj(self, qd: List[float], a: float = 1.4, t: float = 0.06):
        qd_str = ", ".join(f"{val:.4f}" for val in qd)
        cmd = f"speedj([{qd_str}], a={a:.2f}, t={t:.2f})\n"
        self._send_urscript(cmd)
        if self.mock and self.robot_worker and hasattr(self.robot_worker, "stream_speedj"):
            self.robot_worker.stream_speedj(qd, a=a, t=t)

    def _send_servoj(self, q: List[float], dt: float = 0.02):
        q_str = ", ".join(f"{val:.5f}" for val in q)
        cmd = f"servoj([{q_str}], t={dt:.3f}, lookahead_time=0.1, gain=300)\n"
        self._send_urscript(cmd)
        if self.mock and self.robot_worker and hasattr(self.robot_worker, "stream_servoj"):
            self.robot_worker.stream_servoj(q, dt=dt)

    def _send_stop(self):
        self._send_urscript("stopj(2.0)\n")
        if self.mock and self.robot_worker and hasattr(self.robot_worker, "stop_motion"):
            self.robot_worker.stop_motion(2.0)

    def _get_current_joints(self) -> List[float]:
        if self.robot_worker and hasattr(self.robot_worker, "get_latest_joints"):
            jd = self.robot_worker.get_latest_joints()
            if isinstance(jd, dict):
                from start_position_config import UR5E_JOINT_NAMES
                return [float(jd.get(j, 0.0)) for j in UR5E_JOINT_NAMES]
            elif isinstance(jd, (list, tuple)):
                return [float(x) for x in jd[:6]]
        if self.waypoints:
            return list(self.waypoints[0].q)
        return [0.0] * 6

    def _get_current_tcp(self) -> List[float]:
        if self.robot_worker and hasattr(self.robot_worker, "get_latest_tcp"):
            t = self.robot_worker.get_latest_tcp()
            if t and len(t) >= 6:
                return [float(x) for x in t]
        if self.waypoints:
            return list(self.waypoints[0].tcp_pose)
        return [-0.1334, -0.0631, 0.5503, 1.0980, -1.8947, 1.3240]

    def _set_state(self, new_state: HybridState):
        with QMutexLocker(self._lock):
            self._current_state = new_state
        
        style_map = {
            HybridState.IDLE: ("#64748b", "Idle / Ready"),
            HybridState.ALIGNING: ("#38bdf8", "Aligning to Start Pose (Min-Jerk)"),
            HybridState.PLAYBACK: ("#0284c7", "State 1: Reference Path Playback"),
            HybridState.OVERRIDE: ("#f59e0b", "State 2: Teleop Manual Override (Active)"),
            HybridState.RECOVERY: ("#8b5cf6", "State 3: Min-Jerk Snapback Recovery"),
            HybridState.COMPLETED: ("#10b981", "Path Playback Completed"),
            HybridState.STOPPED: ("#ef4444", "Halted / Aborted"),
        }
        color, desc = style_map.get(new_state, ("#64748b", str(new_state.value)))
        self.state_changed.emit(new_state.value, color, desc)

    # -------------------------------------------------------------------------
    # Main State Machine Execution Thread
    # -------------------------------------------------------------------------

    def run(self):
        N = len(self.waypoints)
        if N < 2:
            self.finished.emit(False, "Loaded trajectory contains insufficient points (< 2).")
            return

        self._abort_flag = False
        if not self.mock:
            if not self._ensure_socket():
                self.finished.emit(False, f"Could not open command socket to robot {self.robot_ip}:30002")
                return

        self.log_message.emit("INFO", f"[HYBRID] Initializing Hybrid Runner ({N} waypoints)...")

        # ---------------------------------------------------------------------
        # Pre-flight: Align to Start Pose if needed
        # ---------------------------------------------------------------------
        self._set_state(HybridState.ALIGNING)
        curr_q = self._get_current_joints()
        start_q = self.waypoints[0].q
        disp = max(abs(c - s) for c, s in zip(curr_q, start_q))

        if disp > 0.05:  # > ~2.8 degrees deviation
            deg_disp = math.degrees(disp)
            self.log_message.emit("INFO", f"[HYBRID] Arm is not at path start pose ({deg_disp:.1f}° offset). Aligning smoothly...")
            align_duration = max(1.2, min(5.0, (disp / 0.35) * 1.875))
            align_traj = generate_minimum_jerk_trajectory(curr_q, start_q, align_duration, dt=0.02)
            
            for _, q_des, qd_des in align_traj:
                if self._abort_flag:
                    self._set_state(HybridState.STOPPED)
                    self.finished.emit(False, "Alignment aborted by user.")
                    return
                # Check if user overrides during alignment
                if self.gamepad_teleop and self.gamepad_teleop.has_active_teleop_input():
                    self.log_message.emit("INFO", "[HYBRID] User input detected during alignment. Transitioning to teleop.")
                    break
                self._send_speedj(qd_des, a=1.4, t=0.06)
                time.sleep(0.02)
            self._send_stop()
            time.sleep(0.1)

        # Apply initial gripper width from reference path if connected
        if self.gripper and hasattr(self.gripper, "grip"):
            try:
                w_start = float(self.waypoints[0].gripper_width)
                if w_start > 50.0:
                    self.gripper.grip(w_start, force_n=30.0, speed_percent=50, wait=False)
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # State 1: Reference Path Playback Loop (50 Hz)
        # ---------------------------------------------------------------------
        self._set_state(HybridState.PLAYBACK)
        if self.gamepad_teleop:
            self.gamepad_teleop.set_inhibited(True, "Path playback in progress")

        control_rate_hz = 50.0
        dt = 1.0 / control_rate_hz
        kp = 2.5  # Proportional feedback gain
        max_j_vel = 2.0  # rad/s

        current_idx = 0
        interrupt_idx = 0
        last_gripper_cmd_time = 0.0
        prev_gripper_width = float(self.waypoints[0].gripper_width)

        self.log_message.emit("INFO", "[HYBRID] >>> Commencing State 1: Path Playback. Touch gamepad sticks/buttons to override! <<<")

        while current_idx < N:
            t_loop_start = time.perf_counter()

            if self._abort_flag:
                self._set_state(HybridState.STOPPED)
                self._send_stop()
                if self.gamepad_teleop:
                    self.gamepad_teleop.set_inhibited(False)
                self.finished.emit(False, "Hybrid execution aborted by user.")
                return

            # =================================================================
            # CHECK TRANSITION TO STATE 2 (TELEOP OVERRIDE)
            # =================================================================
            if self.gamepad_teleop and self.gamepad_teleop.has_active_teleop_input():
                interrupt_idx = current_idx
                self.log_message.emit("INFO", f"[HYBRID] 🎮 Non-zero gamepad input detected at waypoint {interrupt_idx}/{N}! HALTING playback -> STATE 2 (OVERRIDE).")
                self._send_stop()
                self._set_state(HybridState.OVERRIDE)
                self.gamepad_teleop.set_inhibited(False)

                last_user_input_time = time.perf_counter()

                # -------------------------------------------------------------
                # STATE 2: ACTIVE TELEOP OVERRIDE LOOP
                # -------------------------------------------------------------
                while True:
                    if self._abort_flag:
                        self._set_state(HybridState.STOPPED)
                        self._send_stop()
                        self.finished.emit(False, "Hybrid execution aborted during override.")
                        return

                    t_ov_start = time.perf_counter()

                    # Check if user is still actively driving the controller
                    is_active = self.gamepad_teleop.has_active_teleop_input()
                    if is_active:
                        last_user_input_time = time.perf_counter()
                    else:
                        # Controller is neutral (inside deadzone)
                        idle_time = time.perf_counter() - last_user_input_time
                        if idle_time >= self.override_delay_s:
                            # Grace period expired! Transition to State 3 Recovery
                            self.log_message.emit("INFO", f"[HYBRID] Gamepad neutral for {idle_time*1000:.0f}ms (>= {self.override_delay_s*1000:.0f}ms grace period). Commencing STATE 3: Recovery.")
                            break

                    # Report spatial distance to the remaining path during override
                    c_tcp = self._get_current_tcp()
                    c_q = self._get_current_joints()
                    n_idx, n_dist = find_nearest_waypoint(self.waypoints, c_tcp, c_q, start_idx=interrupt_idx)
                    pct = (n_idx / max(1, N - 1)) * 100.0
                    self.progress_updated.emit(n_idx, N, pct, n_dist * 100.0)

                    time.sleep(0.02)

                # =============================================================
                # STATE 3: RECOVERY / INTERPOLATION (MINIMUM-JERK SNAPBACK)
                # =============================================================
                self._set_state(HybridState.RECOVERY)
                self.gamepad_teleop.set_inhibited(True, "Recovery snapback in progress")

                # 1. Capture current spatial pose
                c_q = self._get_current_joints()
                c_tcp = self._get_current_tcp()

                # 2. Search nearest spatial waypoint in unexecuted portion [interrupt_idx, N-1]
                target_idx, target_dist = find_nearest_waypoint(
                    self.waypoints, c_tcp, c_q, start_idx=interrupt_idx, search_ahead_only=True
                )
                target_wp = self.waypoints[target_idx]
                target_q = list(target_wp.q)

                max_dq = max(abs(t - c) for t, c in zip(target_q, c_q))
                max_dq_deg = math.degrees(max_dq)
                dist_cm = target_dist * 100.0

                self.log_message.emit("INFO", f"[HYBRID] Nearest waypoint matched: #{target_idx}/{N} (Dist: {dist_cm:.1f} cm, Max joint diff: {max_dq_deg:.1f}°)")

                # 3. Generate smooth Minimum-Jerk polynomial trajectory
                # Duration scaled so max joint velocity never exceeds ~0.30 rad/s
                snapback_duration = max(0.6, min(4.5, (max_dq / 0.30) * 1.875))
                self.snapback_started.emit(target_idx, snapback_duration, max_dq_deg)

                snap_traj = generate_minimum_jerk_trajectory(c_q, target_q, snapback_duration, dt=0.02)
                snapback_interrupted = False

                for step_idx, (t_rel, q_des, qd_des) in enumerate(snap_traj):
                    if self._abort_flag:
                        self._set_state(HybridState.STOPPED)
                        self._send_stop()
                        self.finished.emit(False, "Aborted during recovery snapback.")
                        return

                    # SAFETY INTERLOCK: If user deflects gamepad during snapback, immediately retake control!
                    if self.gamepad_teleop and self.gamepad_teleop.has_active_teleop_input():
                        self.log_message.emit("INFO", "[HYBRID] 🎮 Gamepad touched during snapback! Instant override re-triggered.")
                        snapback_interrupted = True
                        break

                    # Closed-loop tracking for snapback
                    live_q = self._get_current_joints()
                    qd_cmd = [0.0] * 6
                    for j in range(6):
                        err = q_des[j] - live_q[j]
                        qd_cmd[j] = qd_des[j] + 2.0 * err
                        qd_cmd[j] = max(-max_j_vel, min(max_j_vel, qd_cmd[j]))

                    self._send_speedj(qd_cmd, a=1.4, t=0.06)

                    # Update progress UI
                    pct = (target_idx / max(1, N - 1)) * 100.0
                    self.progress_updated.emit(target_idx, N, pct, (1.0 - (step_idx / len(snap_traj))) * dist_cm)
                    time.sleep(0.02)

                if snapback_interrupted:
                    # User interrupted snapback -> jump directly back to State 2 loop
                    continue

                self._send_stop()
                time.sleep(0.05)

                # 4. Resume State 1 from target_idx!
                current_idx = target_idx
                interrupt_idx = target_idx
                self._set_state(HybridState.PLAYBACK)
                self.gamepad_teleop.set_inhibited(True, "Path playback in progress")
                self.log_message.emit("INFO", f"[HYBRID] >>> Resuming STATE 1 (Path Playback) from Waypoint #{current_idx}/{N} <<<")

            # =================================================================
            # NORMAL STATE 1 PLAYBACK STEPPING
            # =================================================================
            wp_curr = self.waypoints[current_idx]
            
            # Feedforward joint velocity from next waypoint delta
            if current_idx + 1 < N:
                wp_next = self.waypoints[current_idx + 1]
                dt_step = max(0.005, wp_next.timestamp - wp_curr.timestamp)
                qd_ff = [(wp_next.q[j] - wp_curr.q[j]) / dt_step for j in range(6)]
            else:
                qd_ff = [0.0] * 6

            # Feedback error correction: qd_cmd = qd_ff + Kp * (q_des - q_act)
            live_q = self._get_current_joints()
            qd_cmd = [0.0] * 6
            for j in range(6):
                pos_err = wp_curr.q[j] - live_q[j]
                v = (qd_ff[j] * self.playback_speed_scale) + (kp * pos_err)
                qd_cmd[j] = max(-max_j_vel, min(max_j_vel, v))

            self._send_speedj(qd_cmd, a=1.4, t=0.06)

            # Gripper control streaming if changed
            now = time.perf_counter()
            if self.gripper and hasattr(self.gripper, "grip") and now - last_gripper_cmd_time >= 0.10:
                target_w = float(wp_curr.gripper_width)
                if abs(target_w - prev_gripper_width) > 1.0 and target_w > 50.0:
                    try:
                        self.gripper.grip(target_w, force_n=30.0, speed_percent=50, wait=False)
                        prev_gripper_width = target_w
                        last_gripper_cmd_time = now
                    except Exception:
                        pass

            # Progress update
            pct = (current_idx / max(1, N - 1)) * 100.0
            self.progress_updated.emit(current_idx, N, pct, 0.0)

            # Advance waypoint index based on scaled playback speed
            # At 50 Hz, default recording is ~30-50 Hz, so step by 1 or speed factor
            step_increment = max(1, int(round(self.playback_speed_scale)))
            current_idx += step_increment

            # Accurate rate throttling at 50 Hz
            elapsed = time.perf_counter() - t_loop_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        # ---------------------------------------------------------------------
        # Playback Completion
        # ---------------------------------------------------------------------
        self._send_stop()
        self._set_state(HybridState.COMPLETED)
        if self.gamepad_teleop:
            self.gamepad_teleop.set_inhibited(False)
        self.progress_updated.emit(N - 1, N, 100.0, 0.0)
        self.log_message.emit("INFO", "[HYBRID] Reference path playback fully completed!")
        self.finished.emit(True, "Reference path playback successfully completed.")
        self._disconnect_socket()
