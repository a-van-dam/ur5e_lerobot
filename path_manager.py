#!/usr/bin/env python3
"""
UR5e Reference Path Manager (Phase 1 "Golden" Deterministic Trajectory Tool)
=============================================================================
Backend module for recording, saving, loading, and playing back golden
reference trajectories on a Universal Robots UR5e arm with an OnRobot 2FG14
Modbus TCP gripper.

Features:
- Precise 30 Hz (or user-configurable) telemetry recording of:
  * Elapsed timestamp & epoch time
  * Joint positions: q0 .. q5 (in radians)
  * Cartesian TCP pose: x, y, z, rx, ry, rz (in meters & radians)
  * OnRobot 2FG gripper width (in mm)
- File I/O:
  * Automatic creation of 'reference_paths/' directory in project root
  * Timestamped CSV and JSON file serialization & deserialization
  * Resilient loading supporting both standard header formats (q0..q5 and joint names)
- High-precision playback:
  * Smooth 'servoj' streaming at recorded frequency
  * Optional 'movej' waypoint execution
  * Safe automatic alignment to trajectory start point before playback starts
  * Non-blocking Modbus TCP gripper command synchronization
  * Immediate safety halt ('stopj' / 'stopl')
- Thread-safe architecture with callbacks for GUI integration & headless CLI
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import datetime
import json
import logging
import math
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Optional UR RTDE and Gripper imports with safe fallbacks
try:
    from rtde_receive import RTDEReceiveInterface
    RTDE_AVAILABLE = True
except ImportError:
    try:
        from ur_rtde import rtde_receive
        RTDEReceiveInterface = rtde_receive.RTDEReceiveInterface
        RTDE_AVAILABLE = True
    except ImportError:
        RTDEReceiveInterface = None
        RTDE_AVAILABLE = False

try:
    from onrobot_gripper import OnRobot2FGModbus, get_shared_gripper
    ONROBOT_AVAILABLE = True
except ImportError:
    OnRobot2FGModbus = None
    get_shared_gripper = None
    ONROBOT_AVAILABLE = False

logger = logging.getLogger("ReferencePathManager")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Centralized start position loader (default_start_position.json)
try:
    from start_position_config import (
        get_default_start_joints_deg,
        get_default_start_joints_rad,
        get_default_start_gripper_mm,
        save_start_position as persist_start_position,
        UR5E_JOINT_NAMES,
    )
    DEFAULT_START_JOINTS_DEG = get_default_start_joints_deg()
    DEFAULT_START_JOINTS_RAD = get_default_start_joints_rad()
    DEFAULT_START_GRIPPER_MM = get_default_start_gripper_mm()
except ImportError:
    UR5E_JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    DEFAULT_START_JOINTS_DEG = [-90.0, -40.0, -100.0, 120.0, -90.0, 120.0]
    DEFAULT_START_JOINTS_RAD = [math.radians(v) for v in DEFAULT_START_JOINTS_DEG]
    DEFAULT_START_GRIPPER_MM = 105.0
    persist_start_position = None

# Scientific computing libraries for trajectory smoothing and spline interpolation
try:
    import numpy as np
    from scipy.interpolate import CubicSpline
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    np = None
    CubicSpline = None
    savgol_filter = None
    SCIPY_AVAILABLE = False

# Canonical CSV Header
CSV_HEADER = [
    "timestamp",
    "q0",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "x",
    "y",
    "z",
    "rx",
    "ry",
    "rz",
    "gripper_width",
]


@dataclass
class TrajectoryPoint:
    """Represents a single discrete state sample of the robot and end-effector."""

    timestamp: float           # Relative time in seconds from start of trajectory
    q: List[float]             # 6-DoF joint angles in radians: [q0, q1, q2, q3, q4, q5]
    tcp_pose: List[float]      # Cartesian TCP pose: [x, y, z, rx, ry, rz] (m, rad)
    gripper_width: float       # Gripper external opening width in millimeters
    epoch_time: float = 0.0    # Absolute UNIX epoch timestamp (s)

    def to_dict(self) -> Dict[str, Any]:
        """Converts point to a serializable dictionary."""
        return {
            "timestamp": round(self.timestamp, 4),
            "epoch_time": round(self.epoch_time, 4),
            "q": [round(float(v), 6) for v in self.q],
            "tcp_pose": [round(float(v), 6) for v in self.tcp_pose],
            "gripper_width": round(float(self.gripper_width), 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrajectoryPoint:
        """Constructs a TrajectoryPoint from dictionary data."""
        # Support joint keys: 'q' list, or named dict
        q_vals = data.get("q")
        if q_vals is None:
            q_vals = [
                float(data.get(f"q{i}", data.get(name, 0.0)))
                for i, name in enumerate(UR5E_JOINT_NAMES)
            ]
        else:
            q_vals = [float(v) for v in q_vals]

        # Support tcp_pose list or individual coords
        tcp_vals = data.get("tcp_pose")
        if tcp_vals is None:
            tcp_vals = [
                float(data.get("x", 0.0)),
                float(data.get("y", 0.0)),
                float(data.get("z", 0.0)),
                float(data.get("rx", 0.0)),
                float(data.get("ry", 0.0)),
                float(data.get("rz", 0.0)),
            ]
        else:
            tcp_vals = [float(v) for v in tcp_vals]

        return cls(
            timestamp=float(data.get("timestamp", 0.0)),
            q=q_vals,
            tcp_pose=tcp_vals,
            gripper_width=float(data.get("gripper_width", 72.0)),
            epoch_time=float(data.get("epoch_time", 0.0)),
        )

    def to_csv_row(self) -> List[str]:
        """Returns string values for CSV serialization matching CSV_HEADER."""
        row = [f"{self.timestamp:.4f}"]
        row.extend(f"{val:.6f}" for val in self.q)
        row.extend(f"{val:.6f}" for val in self.tcp_pose)
        row.append(f"{self.gripper_width:.2f}")
        return row

    @classmethod
    def from_csv_row(cls, row_dict: Dict[str, str]) -> TrajectoryPoint:
        """Parses a CSV dictionary row into a TrajectoryPoint."""
        # Read timestamp
        ts = float(row_dict.get("timestamp", 0.0))

        # Read joints: check q0..q5 first, then joint names
        q = []
        for i in range(6):
            key_q = f"q{i}"
            key_name = UR5E_JOINT_NAMES[i]
            if key_q in row_dict:
                q.append(float(row_dict[key_q]))
            elif key_name in row_dict:
                q.append(float(row_dict[key_name]))
            else:
                q.append(0.0)

        # Read TCP pose
        tcp_keys = ["x", "y", "z", "rx", "ry", "rz"]
        tcp = [float(row_dict.get(k, 0.0)) for k in tcp_keys]

        # Read Gripper width
        gripper_w = float(
            row_dict.get(
                "gripper_width",
                row_dict.get("gripper", row_dict.get("width_mm", 72.0)),
            )
        )

        return cls(
            timestamp=ts,
            q=q,
            tcp_pose=tcp,
            gripper_width=gripper_w,
            epoch_time=float(row_dict.get("epoch_time", 0.0)),
        )


class PathManager:
    """
    Trajectory Path Manager for recording, saving, loading, and playing back
    golden deterministic trajectories on UR5e arms and OnRobot 2FG Modbus grippers.
    """

    def __init__(
        self,
        robot_ip: str = "192.168.1.102",
        gripper_host: str = "192.168.1.1",
        command_port: int = 30002,
        mock: bool = False,
        base_dir: Optional[Union[str, Path]] = None,
    ):
        self.robot_ip = robot_ip
        self.gripper_host = gripper_host
        self.command_port = command_port
        self.mock = mock

        # Resolve project root and reference paths directory
        if base_dir is not None:
            self.project_root = Path(base_dir).resolve()
        else:
            self.project_root = Path(__file__).resolve().parent

        self.paths_dir = self.project_root / "reference_paths"
        self._ensure_paths_directory()

        # In-memory trajectory buffer
        self._trajectory: List[TrajectoryPoint] = []
        self._trajectory_lock = threading.Lock()
        self._loaded_filepath: Optional[Path] = None
        self._start_position: Optional[List[float]] = list(DEFAULT_START_JOINTS_RAD)
        self._start_gripper_width: float = DEFAULT_START_GRIPPER_MM

        # Hardware connection references
        self._rtde_rx: Optional[RTDEReceiveInterface] = None
        self._sock: Optional[socket.socket] = None
        self._gripper: Optional[OnRobot2FGModbus] = None
        self._owns_robot_connection: bool = False
        self._owns_gripper_connection: bool = False

        # Custom state provider hook (callable returning (q, tcp_pose, gripper_width))
        self._state_provider: Optional[
            Callable[[], Tuple[List[float], List[float], float]]
        ] = None

        # State flags
        self._is_recording = False
        self._is_playing = False
        self._stop_playback_flag = threading.Event()
        self._stop_record_flag = threading.Event()

        # Threads
        self._record_thread: Optional[threading.Thread] = None
        self._play_thread: Optional[threading.Thread] = None

        # Simulation/mock state trackers
        self._mock_q: List[float] = list(DEFAULT_START_JOINTS_RAD)
        self._mock_tcp: List[float] = [0.35, -0.20, 0.45, 2.22, -2.22, 0.0]
        self._mock_gripper_width: float = DEFAULT_START_GRIPPER_MM

    # -------------------------------------------------------------------------
    # Properties & Status
    # -------------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def trajectory(self) -> List[TrajectoryPoint]:
        with self._trajectory_lock:
            return list(self._trajectory)

    @property
    def point_count(self) -> int:
        with self._trajectory_lock:
            return len(self._trajectory)

    @property
    def duration_s(self) -> float:
        with self._trajectory_lock:
            if not self._trajectory:
                return 0.0
            return max(0.0, self._trajectory[-1].timestamp - self._trajectory[0].timestamp)

    @property
    def loaded_filepath(self) -> Optional[Path]:
        return self._loaded_filepath

    def set_start_position(self, pos: Union[List[float], Dict[str, float]], gripper_width: Optional[float] = None):
        """Sets the reference start position (radians or dict of radians/degrees)."""
        if isinstance(pos, dict):
            # Check if values are in degrees (> 2*pi or typical UR5e ranges)
            q_rad = []
            for j in UR5E_JOINT_NAMES:
                val = float(pos.get(f"{j}.pos", pos.get(j, 0.0)))
                # If absolute value > 7.0, it's degrees, convert to radians
                if abs(val) > 7.0:
                    val = math.radians(val)
                q_rad.append(val)
            self._start_position = q_rad
            if gripper_width is None and ("gripper" in pos or "gripper_width" in pos):
                self._start_gripper_width = float(pos.get("gripper_width", pos.get("gripper", DEFAULT_START_GRIPPER_MM)))
        else:
            self._start_position = [float(v) for v in pos]

        if gripper_width is not None:
            self._start_gripper_width = float(gripper_width)

    def get_start_position(self) -> Optional[List[float]]:
        """Returns the start position joint angles in radians."""
        if self._start_position is not None:
            return list(self._start_position)
        with self._trajectory_lock:
            if self._trajectory:
                return list(self._trajectory[0].q)
        return None

    def get_start_position_deg(self) -> Optional[List[float]]:
        """Returns the start position joint angles in degrees."""
        sp = self.get_start_position()
        if sp is not None:
            return [round(math.degrees(v), 1) for v in sp]
        return None

    def get_start_gripper_width(self) -> float:
        """Returns the start gripper width in millimeters."""
        return self._start_gripper_width

    def delete_path(self, filepath: Union[str, Path]) -> bool:
        """Deletes a reference path file from disk."""
        path = Path(filepath).resolve()
        if path.is_file():
            try:
                path.unlink()
                logger.info(f"Deleted reference path file: {path.name}")
                if self._loaded_filepath == path:
                    self._loaded_filepath = None
                return True
            except Exception as e:
                logger.error(f"Failed to delete {path}: {e}")
                raise
        return False

    def _ensure_paths_directory(self) -> Path:
        """Ensures the reference_paths/ subfolder exists in the project root."""
        try:
            self.paths_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create reference_paths directory at {self.paths_dir}: {e}")
        return self.paths_dir

    # -------------------------------------------------------------------------
    # Hardware Connection & Dependency Injection
    # -------------------------------------------------------------------------

    def set_robot_connection(
        self,
        rtde_rx: Optional[Any] = None,
        command_sock: Optional[socket.socket] = None,
    ):
        """Allows injecting an existing RTDE receiver and URScript command socket."""
        self._rtde_rx = rtde_rx
        self._sock = command_sock
        self._owns_robot_connection = False
        logger.info("External robot connection injected into PathManager.")

    def set_gripper(self, gripper: Optional[OnRobot2FGModbus]):
        """Allows injecting an existing OnRobot2FGModbus instance."""
        self._gripper = gripper
        self._owns_gripper_connection = False
        logger.info("External OnRobot gripper injected into PathManager.")

    def set_state_provider(
        self,
        provider_fn: Optional[Callable[[], Tuple[List[float], List[float], float]]],
    ):
        """
        Sets a custom function that returns `(q_list, tcp_pose_list, gripper_width_float)`.
        Useful for syncing directly with GUI worker loops.
        """
        self._state_provider = provider_fn

    def connect_robot(self) -> bool:
        """Establishes dedicated RTDE and command socket connections to the UR5e."""
        if self.mock:
            logger.info("Running in mock mode. Robot connection simulated.")
            return True

        success = True
        # 1. Connect RTDE
        if self._rtde_rx is None and RTDE_AVAILABLE and RTDEReceiveInterface is not None:
            try:
                logger.info(f"Connecting RTDE telemetry on {self.robot_ip}...")
                self._rtde_rx = RTDEReceiveInterface(self.robot_ip)
                self._owns_robot_connection = True
            except Exception as e:
                logger.warning(f"Failed to connect RTDE interface to {self.robot_ip}: {e}")
                success = False

        # 2. Connect URScript Command Socket (Port 30002)
        if self._sock is None:
            try:
                logger.info(f"Connecting URScript command socket on {self.robot_ip}:{self.command_port}...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(4.0)
                sock.connect((self.robot_ip, self.command_port))
                self._sock = sock
                self._owns_robot_connection = True
            except Exception as e:
                logger.warning(f"Failed to connect URScript socket on {self.robot_ip}:{self.command_port}: {e}")
                self._sock = None
                success = False

        return success

    def connect_gripper(self) -> bool:
        """Establishes Modbus TCP connection to the OnRobot 2FG compute box."""
        if self.mock:
            logger.info("Running in mock mode. Gripper connection simulated.")
            if self._gripper is None and ONROBOT_AVAILABLE and OnRobot2FGModbus is not None:
                self._gripper = OnRobot2FGModbus(host=self.gripper_host, mock=True)
            return True

        if not ONROBOT_AVAILABLE or OnRobot2FGModbus is None:
            logger.warning("onrobot_gripper module not available.")
            return False

        try:
            if self._gripper is None:
                if get_shared_gripper is not None:
                    self._gripper = get_shared_gripper(host=self.gripper_host, auto_connect=True)
                else:
                    self._gripper = OnRobot2FGModbus(host=self.gripper_host)
                    self._gripper.connect()
                self._owns_gripper_connection = True
            elif not self._gripper.is_connected:
                self._gripper.connect()

            return bool(self._gripper and self._gripper.is_connected)
        except Exception as e:
            logger.error(f"Failed to connect to OnRobot Gripper on {self.gripper_host}: {e}")
            return False

    def disconnect(self):
        """Cleanly releases robot socket and gripper connections."""
        self.stop_recording()
        self.stop_playback()

        if self._owns_robot_connection:
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None

            if self._rtde_rx is not None:
                try:
                    self._rtde_rx.disconnect()
                except Exception:
                    pass
                self._rtde_rx = None

        if self._owns_gripper_connection and self._gripper is not None:
            try:
                self._gripper.disconnect()
            except Exception:
                pass
            self._gripper = None

    # -------------------------------------------------------------------------
    # State Sampling
    # -------------------------------------------------------------------------

    def get_current_state(self) -> Tuple[List[float], List[float], float]:
        """
        Samples the current robot state.
        Returns:
            q: 6 joint angles in radians
            tcp_pose: 6 Cartesian coordinates [x, y, z, rx, ry, rz]
            gripper_width: End-effector opening in mm
        """
        # 1. Use custom provider if defined
        if self._state_provider is not None:
            try:
                return self._state_provider()
            except Exception as e:
                logger.debug(f"Custom state provider error: {e}")

        # 2. Mock mode fallback
        if self.mock:
            return list(self._mock_q), list(self._mock_tcp), float(self._mock_gripper_width)

        # 3. Read directly from RTDE
        q = [0.0] * 6
        tcp = [0.0] * 6
        if self._rtde_rx is not None:
            try:
                q = [float(v) for v in self._rtde_rx.getActualQ()]
            except Exception as e:
                logger.debug(f"Error reading RTDE joint positions: {e}")

            try:
                tcp = [float(v) for v in self._rtde_rx.getActualTCPPose()]
            except Exception as e:
                logger.debug(f"Error reading RTDE TCP pose: {e}")

        # 4. Read Gripper Width
        gripper_w = 72.0
        if self._gripper is not None:
            try:
                gripper_w = float(self._gripper.get_width())
            except Exception as e:
                logger.debug(f"Error reading Gripper width: {e}")

        return q, tcp, gripper_w

    # -------------------------------------------------------------------------
    # Trajectory Recording
    # -------------------------------------------------------------------------

    def start_recording(
        self,
        frequency_hz: float = 30.0,
        callback: Optional[Callable[[TrajectoryPoint, int], None]] = None,
    ) -> bool:
        """
        Begins logging the robot trajectory at the given frequency (default 30 Hz).
        """
        if self._is_recording:
            logger.warning("Recording is already in progress.")
            return False

        if self._is_playing:
            logger.warning("Cannot record while playback is active.")
            return False

        with self._trajectory_lock:
            self._trajectory.clear()

        self._stop_record_flag.clear()
        self._is_recording = True

        self._record_thread = threading.Thread(
            target=self._recording_loop,
            args=(frequency_hz, callback),
            name="PathManager-RecordWorker",
            daemon=True,
        )
        self._record_thread.start()
        logger.info(f"Trajectory recording started at {frequency_hz:.1f} Hz.")
        return True

    def stop_recording(self) -> List[TrajectoryPoint]:
        """Stops the recording loop and returns the captured trajectory."""
        if not self._is_recording:
            return self.trajectory

        self._stop_record_flag.set()
        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=1.5)

        self._is_recording = False
        self._record_thread = None

        with self._trajectory_lock:
            count = len(self._trajectory)
            duration = (
                self._trajectory[-1].timestamp - self._trajectory[0].timestamp
                if count > 0
                else 0.0
            )

        logger.info(f"Trajectory recording stopped. Captured {count} points ({duration:.2f}s).")
        return self.trajectory

    def _recording_loop(
        self,
        frequency_hz: float,
        callback: Optional[Callable[[TrajectoryPoint, int], None]],
    ):
        """Worker loop sampling robot state at a stable fixed interval."""
        interval = 1.0 / max(1.0, float(frequency_hz))
        t_start = time.perf_counter()

        while not self._stop_record_flag.is_set():
            loop_t0 = time.perf_counter()
            elapsed = loop_t0 - t_start

            # Sample robot state
            q, tcp, gripper_w = self.get_current_state()

            pt = TrajectoryPoint(
                timestamp=elapsed,
                q=q,
                tcp_pose=tcp,
                gripper_width=gripper_w,
                epoch_time=time.time(),
            )

            with self._trajectory_lock:
                self._trajectory.append(pt)
                current_count = len(self._trajectory)

            if callback is not None:
                try:
                    callback(pt, current_count)
                except Exception as cb_err:
                    logger.debug(f"Error in record callback: {cb_err}")

            # Precise timing sleep
            loop_dt = time.perf_counter() - loop_t0
            sleep_time = interval - loop_dt
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._is_recording = False

    # -------------------------------------------------------------------------
    # File I/O (CSV & JSON)
    # -------------------------------------------------------------------------

    def save_path(
        self,
        filepath: Optional[Union[str, Path]] = None,
        file_format: str = "csv",
    ) -> Path:
        """
        Saves the recorded trajectory buffer to file.
        If filepath is None, auto-generates a timestamped file inside `reference_paths/`.
        Returns the resolved Path of the saved file.
        """
        with self._trajectory_lock:
            points = list(self._trajectory)

        if not points:
            raise ValueError("Trajectory buffer is empty. Nothing to save.")

        self._ensure_paths_directory()

        if filepath is None:
            # Timestamp format matching requirement: ref_path_YYYYMMDD_HHMM.csv
            now_str = datetime.datetime.now().strftime("ref_path_%Y%m%d_%H%M%S")
            fmt = file_format.lower().strip().replace(".", "")
            if fmt not in ("csv", "json"):
                fmt = "csv"
            save_path = self.paths_dir / f"{now_str}.{fmt}"
        else:
            save_path = Path(filepath).resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fmt = save_path.suffix.lower().replace(".", "")
            if fmt not in ("csv", "json"):
                fmt = file_format.lower().strip().replace(".", "")

        try:
            if fmt == "json":
                self._save_json(save_path, points)
            else:
                self._save_csv(save_path, points)

            self._loaded_filepath = save_path
            logger.info(f"Trajectory ({len(points)} points) successfully saved to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save trajectory to {save_path}: {e}")
            raise

    def _save_csv(self, path: Path, points: List[TrajectoryPoint]):
        """Serializes trajectory to CSV with start position metadata."""
        start_q = self.get_start_position() or (points[0].q if points else None)
        gw = self.get_start_gripper_width()
        with open(path, "w", newline="", encoding="utf-8") as f:
            if start_q is not None:
                start_deg = [math.degrees(v) for v in start_q]
                f.write(f"# start_position_rad: {','.join(f'{v:.6f}' for v in start_q)}\n")
                f.write(f"# start_position_deg: {','.join(f'{v:.1f}' for v in start_deg)}\n")
                f.write(f"# start_gripper_mm: {gw:.1f}\n")
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            for pt in points:
                writer.writerow(pt.to_csv_row())

    def _save_json(self, path: Path, points: List[TrajectoryPoint]):
        """Serializes trajectory to JSON with rich metadata including start position."""
        duration = points[-1].timestamp - points[0].timestamp if points else 0.0
        avg_freq = len(points) / duration if duration > 0 else 0.0
        start_q = self.get_start_position() or (points[0].q if points else None)
        gw = self.get_start_gripper_width()
        payload = {
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "num_points": len(points),
                "duration_s": round(duration, 3),
                "estimated_frequency_hz": round(avg_freq, 1),
                "robot_ip": self.robot_ip,
                "version": "1.0",
            },
            "points": [pt.to_dict() for pt in points],
        }
        if start_q is not None:
            start_deg = [round(math.degrees(v), 2) for v in start_q]
            payload["metadata"]["start_position"] = {
                "joints_rad": [round(float(v), 6) for v in start_q],
                "joints_deg": start_deg,
                "gripper_width": round(gw, 1),
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_path(self, filepath: Union[str, Path]) -> List[TrajectoryPoint]:
        """
        Loads a trajectory file (CSV or JSON) from disk into memory.
        """
        path = Path(filepath).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory file does not exist: {path}")

        loaded_points: List[TrajectoryPoint] = []
        fmt = path.suffix.lower().replace(".", "")

        try:
            if fmt == "json":
                loaded_points = self._load_json(path)
            else:
                loaded_points = self._load_csv(path)

            if not loaded_points:
                raise ValueError(f"File {path.name} contained no valid trajectory points.")

            with self._trajectory_lock:
                self._trajectory = loaded_points
                self._loaded_filepath = path

            dur = loaded_points[-1].timestamp - loaded_points[0].timestamp
            logger.info(
                f"Successfully loaded {len(loaded_points)} waypoints ({dur:.2f}s) from {path.name}."
            )
            return self.trajectory
        except Exception as e:
            logger.error(f"Failed to load trajectory from {path}: {e}")
            raise

    def _load_csv(self, path: Path) -> List[TrajectoryPoint]:
        """Parses CSV trajectory file and extracts start position metadata."""
        points: List[TrajectoryPoint] = []
        parsed_start_q = None

        with open(path, "r", newline="", encoding="utf-8") as f:
            # Check for header metadata comments
            data_lines = []
            for line in f:
                stripped = line.strip()
                if stripped.startswith("#"):
                    if "start_position_rad:" in stripped:
                        try:
                            parts = stripped.split("start_position_rad:")[1].strip().split(",")
                            parsed_start_q = [float(p.strip()) for p in parts if p.strip()]
                        except Exception:
                            pass
                    elif "start_position_deg:" in stripped and parsed_start_q is None:
                        try:
                            parts = stripped.split("start_position_deg:")[1].strip().split(",")
                            parsed_start_q = [math.radians(float(p.strip())) for p in parts if p.strip()]
                        except Exception:
                            pass
                    elif "start_gripper_mm:" in stripped:
                        try:
                            self._start_gripper_width = float(stripped.split("start_gripper_mm:")[1].strip())
                        except Exception:
                            pass
                elif stripped:
                    data_lines.append(line)

            reader = csv.DictReader(data_lines)
            if not reader.fieldnames:
                raise ValueError("CSV file has no header row.")

            for row in reader:
                cleaned_row = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                points.append(TrajectoryPoint.from_csv_row(cleaned_row))

        if points:
            self._start_position = parsed_start_q if parsed_start_q is not None else list(points[0].q)
            if parsed_start_q is None and hasattr(points[0], "gripper_width"):
                self._start_gripper_width = points[0].gripper_width

        return points

    def _load_json(self, path: Path) -> List[TrajectoryPoint]:
        """Parses JSON trajectory file and extracts start position metadata."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        points_data = data.get("points") or data.get("trajectory") or data
        if not isinstance(points_data, list):
            raise ValueError("Invalid JSON format: expected list of points.")

        points: List[TrajectoryPoint] = []
        for item in points_data:
            if isinstance(item, dict):
                points.append(TrajectoryPoint.from_dict(item))

        # Extract start position metadata
        meta = data.get("metadata", {})
        if "start_position" in meta:
            sp_meta = meta["start_position"]
            if isinstance(sp_meta, dict):
                if "joints_rad" in sp_meta:
                    self._start_position = [float(v) for v in sp_meta["joints_rad"]]
                if "gripper_width" in sp_meta:
                    self._start_gripper_width = float(sp_meta["gripper_width"])
            elif isinstance(sp_meta, list):
                self._start_position = [float(v) for v in sp_meta]
        elif points:
            self._start_position = list(points[0].q)
            self._start_gripper_width = points[0].gripper_width

        return points

    def list_saved_paths(self) -> List[Path]:
        """Returns sorted list of saved reference trajectory files in `reference_paths/`."""
        self._ensure_paths_directory()
        files = [
            f for f in self.paths_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".csv", ".json")
        ]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files

    def clear(self):
        """Clears the loaded trajectory from memory."""
        with self._trajectory_lock:
            self._trajectory.clear()
            self._loaded_filepath = None

    # -------------------------------------------------------------------------
    # Trajectory Playback (speedj / servoj / movej with Spline Smoothing)
    # -------------------------------------------------------------------------

    def play_path_async(
        self,
        method: str = "speedj",
        speed_scale: float = 1.0,
        auto_align_start: bool = True,
        smooth: bool = True,
        progress_callback: Optional[Callable[[int, int, TrajectoryPoint], None]] = None,
        completion_callback: Optional[Callable[[bool, str], None]] = None,
    ) -> threading.Thread:
        """
        Executes playback asynchronously in a background thread.
        Methods:
          - 'speedj' (Recommended): Closed-loop continuous velocity streaming with cubic spline.
                                    Provides the same silky smooth motion as gamepad teleoperation.
          - 'servoj': High-rate (100 Hz) position servoing with spline interpolation and tuned damping.
          - 'movej': Waypoint-by-waypoint discrete position steps.
        """
        if self._is_playing:
            raise RuntimeError("Playback is already in progress.")

        self._play_thread = threading.Thread(
            target=self._playback_worker,
            args=(method, speed_scale, auto_align_start, smooth, progress_callback, completion_callback),
            name="PathManager-PlayWorker",
            daemon=True,
        )
        self._play_thread.start()
        return self._play_thread

    def play_path(
        self,
        method: str = "speedj",
        speed_scale: float = 1.0,
        auto_align_start: bool = True,
        smooth: bool = True,
        progress_callback: Optional[Callable[[int, int, TrajectoryPoint], None]] = None,
    ) -> bool:
        """
        Synchronously executes the loaded trajectory on UR5e and OnRobot gripper.
        """
        if self._is_playing:
            logger.warning("Playback is already running.")
            return False

        return self._execute_playback(
            method=method,
            speed_scale=speed_scale,
            auto_align_start=auto_align_start,
            smooth=smooth,
            progress_callback=progress_callback,
        )

    def stop_playback(self):
        """Immediately halts trajectory playback and sends stop commands to hardware."""
        if not self._is_playing:
            return

        logger.warning("Emergency Stop requested for trajectory playback.")
        self._stop_playback_flag.set()

        # Send immediate robot deceleration
        self.send_robot_stop()

        # Stop gripper motion
        if self._gripper is not None:
            try:
                self._gripper.stop()
            except Exception as e:
                logger.debug(f"Error stopping gripper: {e}")

        if self._play_thread and self._play_thread.is_alive():
            if threading.current_thread() != self._play_thread:
                self._play_thread.join(timeout=1.5)

        self._is_playing = False

    def send_robot_stop(self):
        """Sends URScript stop command (stopj / stopl) to the arm."""
        if self._sock is not None:
            try:
                self._sock.sendall(b"stopj(2.0)\n")
            except Exception as e:
                logger.error(f"Failed to transmit stopj URScript: {e}")

    def _playback_worker(
        self,
        method: str,
        speed_scale: float,
        auto_align_start: bool,
        smooth: bool,
        progress_callback: Optional[Callable[[int, int, TrajectoryPoint], None]],
        completion_callback: Optional[Callable[[bool, str], None]],
    ):
        """Worker wrapper managing async playback lifecycle and completion signal."""
        success = False
        message = ""
        try:
            success = self._execute_playback(
                method=method,
                speed_scale=speed_scale,
                auto_align_start=auto_align_start,
                smooth=smooth,
                progress_callback=progress_callback,
            )
            message = "Playback finished successfully." if success else "Playback stopped or interrupted."
        except Exception as e:
            message = f"Playback error: {e}"
            logger.error(message, exc_info=True)
            self.send_robot_stop()
            success = False
        finally:
            self._is_playing = False
            if completion_callback is not None:
                try:
                    completion_callback(success, message)
                except Exception as cb_err:
                    logger.debug(f"Error in playback completion callback: {cb_err}")

    def _execute_playback(
        self,
        method: str,
        speed_scale: float,
        auto_align_start: bool,
        smooth: bool,
        progress_callback: Optional[Callable[[int, int, TrajectoryPoint], None]],
    ) -> bool:
        """Internal core playback logic."""
        with self._trajectory_lock:
            points = list(self._trajectory)

        if not points:
            logger.error("No trajectory loaded for playback.")
            return False

        total_pts = len(points)
        self._stop_playback_flag.clear()
        self._is_playing = True

        scale = max(0.1, min(2.5, float(speed_scale)))
        logger.info(
            f"Starting playback: {total_pts} points, mode='{method}', speed_scale={scale:.2f}, smooth={smooth}"
        )

        # ---------------------------------------------------------------------
        # 1. Step 1: Pre-Playback Start Alignment & Check
        # ---------------------------------------------------------------------
        start_pt = points[0]
        target_start_q = self.get_start_position() or start_pt.q
        if auto_align_start:
            logger.info("Checking if robot is at reference start position...")
            if not self._align_to_start_pose(target_start_q):
                logger.warning("Start position alignment cancelled or timed out.")
                self.send_robot_stop()
                self._is_playing = False
                return False

        # Align initial gripper state (discrete open/close clamping so measured cube widths still clamp firmly)
        initial_gripper = self.get_start_gripper_width() if self.get_start_gripper_width() is not None else start_pt.gripper_width
        init_cmd_w, _ = self._classify_replay_gripper(float(initial_gripper), is_closed=False)
        if self._gripper is not None:
            try:
                self._gripper.grip(init_cmd_w, wait=False)
            except Exception as e:
                logger.warning(f"Failed to command initial gripper width: {e}")

        # Short settling pause before starting waypoint stream
        if not self._wait_or_abort(0.25):
            self.send_robot_stop()
            self._is_playing = False
            return False

        # ---------------------------------------------------------------------
        # 2. Build Continuous Cubic Spline Trajectory (with Gamepad Noise Filtering)
        # ---------------------------------------------------------------------
        t_start = points[0].timestamp
        t_raw = np.array([pt.timestamp - t_start for pt in points], dtype=np.float64) if np is not None else [pt.timestamp - t_start for pt in points]

        # Enforce strictly monotonic increasing timestamps
        for i in range(1, len(t_raw)):
            if t_raw[i] <= t_raw[i - 1]:
                t_raw[i] = t_raw[i - 1] + 0.001
        total_duration = float(t_raw[-1])
        if total_duration <= 0.005:
            total_duration = 0.033 * total_pts

        q_raw = np.array([pt.q for pt in points], dtype=np.float64) if np is not None else [pt.q for pt in points]
        gripper_raw = np.array([pt.gripper_width for pt in points], dtype=np.float64) if np is not None else [pt.gripper_width for pt in points]

        # Filter out minor analog stick jitter using Savitzky-Golay polynomial filter
        if smooth and SCIPY_AVAILABLE and len(points) >= 7 and savgol_filter is not None:
            try:
                window = min(7, len(points) if len(points) % 2 != 0 else len(points) - 1)
                q_filtered = savgol_filter(q_raw, window_length=window, polyorder=2, axis=0)
                # Strictly preserve exact start and end boundaries
                q_filtered[0] = q_raw[0]
                q_filtered[-1] = q_raw[-1]
            except Exception as filt_err:
                logger.debug(f"Savgol filter notice: {filt_err}")
                q_filtered = q_raw
        else:
            q_filtered = q_raw

        # Fit clamped cubic spline for joint trajectories; use oscillation-free linear lookup for gripper width
        if SCIPY_AVAILABLE and CubicSpline is not None and len(points) >= 4:
            cs_q = CubicSpline(t_raw, q_filtered, axis=0, bc_type='clamped')
            cs_gripper = (lambda t_val, _tr=t_raw, _gr=gripper_raw: float(np.interp(t_val, _tr, _gr)))
        else:
            cs_q = None
            cs_gripper = None

        method_norm = method.lower().strip()
        if method_norm in ("speedj", "velocity", "closed_loop") and cs_q is not None:
            return self._execute_speedj_playback(
                cs_q=cs_q,
                cs_gripper=cs_gripper,
                total_duration=total_duration,
                scale=scale,
                total_pts=total_pts,
                points=points,
                progress_callback=progress_callback,
            )
        elif method_norm in ("servoj", "servo") and cs_q is not None:
            return self._execute_servoj_playback(
                cs_q=cs_q,
                cs_gripper=cs_gripper,
                total_duration=total_duration,
                scale=scale,
                total_pts=total_pts,
                points=points,
                progress_callback=progress_callback,
            )
        else:
            return self._execute_movej_playback(
                points=points,
                scale=scale,
                total_pts=total_pts,
                progress_callback=progress_callback,
            )

    def _classify_replay_gripper(self, measured_w: float, is_closed: bool) -> Tuple[float, bool]:
        """
        Maps recorded gripper width (which stops at the object's physical width, e.g. ~80 mm on a cube)
        into firm discrete OPEN / CLOSE commands with hysteresis.

        Commanding min_width_mm (55 mm) instead of the recorded resting width (80.6 mm) ensures
        the OnRobot 2FG actually clamps onto the cube with its configured grip force rather than
        stopping right at the cube's surface with near-zero holding force.
        """
        min_w = float(getattr(self._gripper, "min_width_mm", 55.0)) if self._gripper else 55.0
        max_w = float(getattr(self._gripper, "max_width_mm", 105.0)) if self._gripper else 105.0
        close_thresh = max_w - 12.0  # e.g., < 93.0 mm -> CLOSE (command min_w so jaws clamp firmly)
        open_thresh = max_w - 6.0    # e.g., >= 99.0 mm -> OPEN (command max_w)

        next_closed = is_closed
        if not is_closed and measured_w < close_thresh:
            next_closed = True
        elif is_closed and measured_w >= open_thresh:
            next_closed = False

        cmd_w = min_w if next_closed else max_w
        return cmd_w, next_closed

    def _execute_speedj_playback(
        self,
        cs_q: Any,
        cs_gripper: Any,
        total_duration: float,
        scale: float,
        total_pts: int,
        points: List[TrajectoryPoint],
        progress_callback: Optional[Callable[[int, int, TrajectoryPoint], None]],
    ) -> bool:
        """
        Closed-loop velocity tracking via speedj at 50 Hz.
        Calculates analytical feedforward joint velocity from cubic spline
        plus proportional position feedback: qd_cmd = qd_ff + Kp * (q_des - q_act).
        This matches the silky smooth motion of gamepad teleoperation with millimeter accuracy.
        """
        loop_rate_hz = 50.0
        dt = 1.0 / loop_rate_hz
        kp = 2.5                     # Proportional tracking gain (s^-1)
        max_joint_vel = 2.0          # Max joint velocity (rad/s)
        max_joint_acc = 1.4          # Max joint acceleration (rad/s^2)
        max_vel_step = max_joint_acc * dt

        prev_qd = [0.0] * 6
        prev_gripper_width, prev_gripper_closed = self._classify_replay_gripper(float(points[0].gripper_width), is_closed=False)
        last_gripper_cmd_time = time.perf_counter()

        playback_start_wall = time.perf_counter()
        effective_duration = total_duration / scale

        while True:
            if self._stop_playback_flag.is_set():
                logger.info("Playback aborted by user.")
                self.send_robot_stop()
                self._is_playing = False
                return False

            now = time.perf_counter()
            elapsed_scaled = (now - playback_start_wall) * scale

            if elapsed_scaled >= total_duration:
                break

            # Current desired position & feedforward velocity from spline
            q_des = cs_q(elapsed_scaled)
            qd_ff = cs_q(elapsed_scaled, 1) * scale

            # Feedback: read actual robot joint angles
            curr_q, curr_tcp, _ = self.get_current_state()

            # Commanded velocity with closed-loop proportional error correction
            qd_cmd = [0.0] * 6
            for i in range(6):
                pos_err = q_des[i] - curr_q[i]
                v = qd_ff[i] + kp * pos_err
                # Clamp velocity limits
                v = max(-max_joint_vel, min(max_joint_vel, v))
                # Acceleration slew rate limiter to prevent jerks
                delta_v = v - prev_qd[i]
                delta_v = max(-max_vel_step, min(max_vel_step, delta_v))
                qd_cmd[i] = prev_qd[i] + delta_v

            prev_qd = list(qd_cmd)

            # Send speedj command (watchdog t=0.06s provides smooth buffer for 0.02s cycle)
            self._send_speedj_step(qd_cmd, a=max_joint_acc, t=0.06)

            # Gripper update: firm discrete open/close transitions (prevents dropping cube on measured-width replay)
            if cs_gripper is not None:
                raw_gripper = float(cs_gripper(elapsed_scaled))
                cmd_gripper, next_closed = self._classify_replay_gripper(raw_gripper, prev_gripper_closed)
                if next_closed != prev_gripper_closed and (now - last_gripper_cmd_time) >= 0.12:
                    prev_gripper_closed = next_closed
                    prev_gripper_width = cmd_gripper
                    last_gripper_cmd_time = now
                    if self._gripper is not None:
                        try:
                            self._gripper.grip(cmd_gripper, wait=False)
                        except Exception:
                            pass
                target_gripper = prev_gripper_width
            else:
                target_gripper = prev_gripper_width

            # Mock simulation integration
            if self.mock:
                self._mock_q = [self._mock_q[i] + qd_cmd[i] * dt for i in range(6)]
                self._mock_gripper_width = target_gripper

            # Progress callback update
            if progress_callback is not None:
                pt_idx = min(total_pts - 1, int((elapsed_scaled / total_duration) * total_pts))
                dummy_pt = TrajectoryPoint(
                    timestamp=elapsed_scaled,
                    q=list(q_des),
                    tcp_pose=list(curr_tcp),
                    gripper_width=target_gripper,
                )
                try:
                    progress_callback(pt_idx, total_pts, dummy_pt)
                except Exception:
                    pass

            # Precise cycle timing
            target_next_time = playback_start_wall + ((elapsed_scaled + (dt * scale)) / scale)
            sleep_time = target_next_time - time.perf_counter()
            if sleep_time > 0.0005:
                time.sleep(sleep_time)

        # Smooth deceleration at end of path
        self.send_robot_stop()
        logger.info(f"Trajectory speedj playback complete ({total_pts} points followed smoothly).")
        self._is_playing = False
        return True

    def _execute_servoj_playback(
        self,
        cs_q: Any,
        cs_gripper: Any,
        total_duration: float,
        scale: float,
        total_pts: int,
        points: List[TrajectoryPoint],
        progress_callback: Optional[Callable[[int, int, TrajectoryPoint], None]],
    ) -> bool:
        """
        High-rate (100 Hz) Spline-Upsampled servoj streaming.
        Uses lookahead_time=0.18 and gain=120 for smooth damping without high-frequency chattering.
        """
        loop_rate_hz = 100.0
        dt = 1.0 / loop_rate_hz
        t_param = 0.015
        lookahead = 0.18
        gain = 120

        prev_gripper_width, prev_gripper_closed = self._classify_replay_gripper(float(points[0].gripper_width), is_closed=False)
        last_gripper_cmd_time = time.perf_counter()

        playback_start_wall = time.perf_counter()

        while True:
            if self._stop_playback_flag.is_set():
                logger.info("Playback aborted by user.")
                self.send_robot_stop()
                self._is_playing = False
                return False

            now = time.perf_counter()
            elapsed_scaled = (now - playback_start_wall) * scale

            if elapsed_scaled >= total_duration:
                break

            # Desired position from continuous cubic spline
            q_des = cs_q(elapsed_scaled)

            # Send servoj command with tuned smooth parameters
            q_str = ", ".join(f"{val:.5f}" for val in q_des)
            cmd = f"servoj([{q_str}], t={t_param:.4f}, lookahead_time={lookahead}, gain={gain})\n"
            if self._sock is not None:
                try:
                    self._sock.sendall(cmd.encode("utf-8"))
                except Exception as e:
                    logger.error(f"Failed to transmit servoj command: {e}")

            # Gripper update: firm discrete open/close transitions
            if cs_gripper is not None:
                raw_gripper = float(cs_gripper(elapsed_scaled))
                cmd_gripper, next_closed = self._classify_replay_gripper(raw_gripper, prev_gripper_closed)
                if next_closed != prev_gripper_closed and (now - last_gripper_cmd_time) >= 0.12:
                    prev_gripper_closed = next_closed
                    prev_gripper_width = cmd_gripper
                    last_gripper_cmd_time = now
                    if self._gripper is not None:
                        try:
                            self._gripper.grip(cmd_gripper, wait=False)
                        except Exception:
                            pass
                target_gripper = prev_gripper_width
            else:
                target_gripper = prev_gripper_width

            if self.mock:
                self._mock_q = list(q_des)
                self._mock_gripper_width = target_gripper

            if progress_callback is not None:
                pt_idx = min(total_pts - 1, int((elapsed_scaled / total_duration) * total_pts))
                dummy_pt = TrajectoryPoint(
                    timestamp=elapsed_scaled,
                    q=list(q_des),
                    tcp_pose=[0.0] * 6,
                    gripper_width=target_gripper,
                )
                try:
                    progress_callback(pt_idx, total_pts, dummy_pt)
                except Exception:
                    pass

            target_next_time = playback_start_wall + ((elapsed_scaled + (dt * scale)) / scale)
            sleep_time = target_next_time - time.perf_counter()
            if sleep_time > 0.0005:
                time.sleep(sleep_time)

        self.send_robot_stop()
        logger.info(f"Trajectory servoj playback complete ({total_pts} points executed).")
        self._is_playing = False
        return True

    def _execute_movej_playback(
        self,
        points: List[TrajectoryPoint],
        scale: float,
        total_pts: int,
        progress_callback: Optional[Callable[[int, int, TrajectoryPoint], None]],
    ) -> bool:
        """Fallback waypoint-by-waypoint discrete movej streaming."""
        prev_gripper_width, prev_gripper_closed = self._classify_replay_gripper(float(points[0].gripper_width), is_closed=False)
        last_gripper_cmd_time = time.perf_counter()
        playback_start_wall = time.perf_counter()
        t_traj_start = points[0].timestamp

        for idx, pt in enumerate(points):
            if self._stop_playback_flag.is_set():
                logger.info(f"Playback aborted by user at point {idx}/{total_pts}.")
                self.send_robot_stop()
                self._is_playing = False
                return False

            rel_t = (pt.timestamp - t_traj_start) / scale
            if idx + 1 < total_pts:
                next_pt = points[idx + 1]
                dt = max(0.015, (next_pt.timestamp - pt.timestamp) / scale)
            else:
                dt = 0.033

            self._send_movej_step(pt.q, dt=dt)

            now = time.perf_counter()
            cmd_gripper, next_closed = self._classify_replay_gripper(float(pt.gripper_width), prev_gripper_closed)
            if next_closed != prev_gripper_closed and (now - last_gripper_cmd_time) >= 0.12:
                prev_gripper_closed = next_closed
                prev_gripper_width = cmd_gripper
                last_gripper_cmd_time = now
                if self._gripper is not None:
                    try:
                        self._gripper.grip(cmd_gripper, wait=False)
                    except Exception:
                        pass

            if self.mock:
                self._mock_q = list(pt.q)
                self._mock_tcp = list(pt.tcp_pose)
                self._mock_gripper_width = pt.gripper_width

            if progress_callback is not None:
                try:
                    progress_callback(idx, total_pts, pt)
                except Exception:
                    pass

            target_wall_time = playback_start_wall + rel_t
            sleep_needed = target_wall_time - time.perf_counter()
            if sleep_needed > 0:
                time.sleep(sleep_needed)

        self.send_robot_stop()
        logger.info(f"Trajectory movej playback complete ({total_pts} points executed).")
        self._is_playing = False
        return True

    def _send_speedj_step(self, qd: List[float], a: float = 1.4, t: float = 0.06):
        """Sends a continuous speedj command to the robot command socket."""
        if self._sock is not None:
            qd_str = ", ".join(f"{val:.5f}" for val in qd)
            cmd = f"speedj([{qd_str}], a={a:.2f}, t={t:.3f})\n"
            try:
                self._sock.sendall(cmd.encode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to transmit speedj command: {e}")

    def _send_servoj_step(self, q: List[float], dt: float):
        """Sends a smooth servoj command to the robot command socket."""
        if self._sock is not None:
            lookahead = 0.18
            gain = 120
            t_param = max(0.012, dt * 1.15)
            q_str = ", ".join(f"{val:.5f}" for val in q)
            cmd = f"servoj([{q_str}], t={t_param:.4f}, lookahead_time={lookahead}, gain={gain})\n"
            try:
                self._sock.sendall(cmd.encode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to transmit servoj command: {e}")

    def _send_movej_step(self, q: List[float], dt: float):
        """Sends a movej command for dense waypoints."""
        if self._sock is not None:
            q_str = ", ".join(f"{val:.5f}" for val in q)
            cmd = f"movej([{q_str}], a=1.4, v=0.7, t={dt:.4f})\n"
            try:
                self._sock.sendall(cmd.encode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to transmit movej command: {e}")

    def _align_to_start_pose(self, target_pose: Union[TrajectoryPoint, List[float]], max_wait_s: float = 8.0) -> bool:
        """
        Verifies if the robot is currently at the target start position. If not,
        smoothly commands the arm to move there before high-frequency streaming begins.
        """
        if hasattr(target_pose, "q"):
            goal_q = target_pose.q
        else:
            goal_q = list(target_pose)

        curr_q, _, _ = self.get_current_state()

        # Check maximum joint error
        max_err = max(abs(curr_q[i] - goal_q[i]) for i in range(6))
        if max_err < 0.025:  # Within ~1.4 degrees, robot is already at start position
            logger.info(f"Robot is already at start position (max error: {math.degrees(max_err):.2f}°).")
            return True

        logger.info(
            f"Robot is NOT at start position (deviation: {math.degrees(max_err):.1f}°). "
            f"Automatically moving arm to start position..."
        )

        # Command movej with conservative speed & acceleration
        if self._sock is not None:
            q_str = ", ".join(f"{val:.5f}" for val in goal_q)
            cmd = f"movej([{q_str}], a=0.8, v=0.4)\n"
            try:
                self._sock.sendall(cmd.encode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to command start position alignment: {e}")
                return False

        # Wait until arrival or stop requested
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < max_wait_s:
            if self._stop_playback_flag.is_set():
                return False

            time.sleep(0.05)
            if self.mock:
                self._mock_q = list(goal_q)
                return True

            curr_q, _, _ = self.get_current_state()
            err = max(abs(curr_q[i] - goal_q[i]) for i in range(6))
            if err < 0.035:
                logger.info(f"Arrived at start position in {time.perf_counter() - t0:.2f}s.")
                return True

        logger.warning("Start waypoint alignment reached timeout, proceeding with playback.")
        return True

    def _wait_or_abort(self, duration_s: float) -> bool:
        """Helper that sleeps for duration_s while checking the abort flag."""
        t_end = time.perf_counter() + duration_s
        while time.perf_counter() < t_end:
            if self._stop_playback_flag.is_set():
                return False
            time.sleep(0.02)
        return True


# -----------------------------------------------------------------------------
# Standalone CLI / Self-Test Runner
# -----------------------------------------------------------------------------

def run_self_test():
    """Runs a quick verification test of PathManager recording, saving, and loading."""
    print("=" * 70)
    print("Running PathManager Self-Test (Mock Mode)")
    print("=" * 70)

    pm = PathManager(mock=True)

    # 1. Test Recording
    print("\n1. Testing 30 Hz Trajectory Recording (1.0 second)...")
    pm.start_recording(frequency_hz=30.0)
    time.sleep(1.0)
    pts = pm.stop_recording()
    print(f"Recorded {len(pts)} points. Duration: {pm.duration_s:.3f}s")
    assert len(pts) > 0, "No points recorded!"

    # 2. Test CSV Save & Load
    print("\n2. Testing CSV Save and Reload...")
    saved_csv = pm.save_path(file_format="csv")
    print(f"Saved CSV: {saved_csv}")

    pm.clear()
    assert pm.point_count == 0, "Buffer not cleared!"

    loaded_pts = pm.load_path(saved_csv)
    print(f"Loaded {len(loaded_pts)} points from CSV.")
    assert len(loaded_pts) == len(pts), "Loaded point count mismatch!"
    assert abs(loaded_pts[0].gripper_width - pts[0].gripper_width) < 1e-3

    # 3. Test JSON Save & Load
    print("\n3. Testing JSON Save and Reload...")
    saved_json = pm.save_path(file_format="json")
    print(f"Saved JSON: {saved_json}")

    pm.clear()
    loaded_json = pm.load_path(saved_json)
    print(f"Loaded {len(loaded_json)} points from JSON.")
    assert len(loaded_json) == len(pts), "Loaded JSON point count mismatch!"
    assert pm.get_start_position() is not None, "Start position was not recovered from JSON!"

    # 4. Test Start Position Check & Auto-Move in Playback
    print("\n4. Testing Start Position Check & Playback...")
    # Intentionally disturb mock joints away from start position
    pm._mock_q = [0.5, -1.0, 1.0, -1.0, -1.0, 0.5]
    progress_count = 0

    def on_progress(idx, total, pt):
        nonlocal progress_count
        progress_count += 1

    success = pm.play_path(method="speedj", speed_scale=2.0, auto_align_start=True, progress_callback=on_progress)
    print(f"speedj Playback finished: success={success}, progress updates={progress_count}")
    assert success, "speedj Playback failed!"

    # Test servoj mode
    success_servo = pm.play_path(method="servoj", speed_scale=2.0, auto_align_start=False)
    print(f"servoj Playback finished: success={success_servo}")
    assert success_servo, "servoj Playback failed!"

    # 5. Clean up test files
    pm.delete_path(saved_csv)
    pm.delete_path(saved_json)

    print("\nAll PathManager self-tests PASSED successfully!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UR5e Reference Path Manager")
    parser.add_argument("--test", action="store_true", help="Run internal self-tests in mock mode")
    args = parser.parse_args()

    if args.test or len(sys.argv) == 1:
        run_self_test()
