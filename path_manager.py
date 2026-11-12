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

# Standard kinematic joint order from base to flange
UR5E_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

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
        self._mock_q: List[float] = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]
        self._mock_tcp: List[float] = [0.35, -0.20, 0.45, 2.22, -2.22, 0.0]
        self._mock_gripper_width: float = 72.0

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
        """Serializes trajectory to CSV."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            for pt in points:
                writer.writerow(pt.to_csv_row())

    def _save_json(self, path: Path, points: List[TrajectoryPoint]):
        """Serializes trajectory to JSON with rich metadata."""
        duration = points[-1].timestamp - points[0].timestamp if points else 0.0
        avg_freq = len(points) / duration if duration > 0 else 0.0
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
        """Parses CSV trajectory file."""
        points: List[TrajectoryPoint] = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV file has no header row.")

            # Validate header has minimal trajectory columns
            headers = [h.strip().lower() for h in reader.fieldnames]
            for row in reader:
                # Clean row dictionary keys
                cleaned_row = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                points.append(TrajectoryPoint.from_csv_row(cleaned_row))
        return points

    def _load_json(self, path: Path) -> List[TrajectoryPoint]:
        """Parses JSON trajectory file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        points_data = data.get("points") or data.get("trajectory") or data
        if not isinstance(points_data, list):
            raise ValueError("Invalid JSON format: expected list of points.")

        points: List[TrajectoryPoint] = []
        for item in points_data:
            if isinstance(item, dict):
                points.append(TrajectoryPoint.from_dict(item))
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
    # Trajectory Playback (servoj / movej)
    # -------------------------------------------------------------------------

    def play_path_async(
        self,
        method: str = "servoj",
        speed_scale: float = 1.0,
        auto_align_start: bool = True,
        progress_callback: Optional[Callable[[int, int, TrajectoryPoint], None]] = None,
        completion_callback: Optional[Callable[[bool, str], None]] = None,
    ) -> threading.Thread:
        """
        Executes playback asynchronously in a background thread.
        """
        if self._is_playing:
            raise RuntimeError("Playback is already in progress.")

        self._play_thread = threading.Thread(
            target=self._playback_worker,
            args=(method, speed_scale, auto_align_start, progress_callback, completion_callback),
            name="PathManager-PlayWorker",
            daemon=True,
        )
        self._play_thread.start()
        return self._play_thread

    def play_path(
        self,
        method: str = "servoj",
        speed_scale: float = 1.0,
        auto_align_start: bool = True,
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
                self._sock.sendall(b"stopj(2.5)\n")
            except Exception as e:
                logger.error(f"Failed to transmit stopj URScript: {e}")

    def _playback_worker(
        self,
        method: str,
        speed_scale: float,
        auto_align_start: bool,
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

        scale = max(0.1, min(2.0, float(speed_scale)))
        logger.info(
            f"Starting playback: {total_pts} points, mode='{method}', speed_scale={scale:.2f}"
        )

        # ---------------------------------------------------------------------
        # 1. Step 1: Pre-Playback Start Alignment
        # ---------------------------------------------------------------------
        start_pt = points[0]
        if auto_align_start:
            logger.info("Aligning robot to starting waypoint...")
            if not self._align_to_start_pose(start_pt):
                logger.warning("Start position alignment cancelled or timed out.")
                self.send_robot_stop()
                self._is_playing = False
                return False

        # Align initial gripper width
        if self._gripper is not None:
            try:
                self._gripper.grip(start_pt.gripper_width, wait=False)
            except Exception as e:
                logger.warning(f"Failed to command initial gripper width: {e}")

        # Short settling pause before starting dense waypoint stream
        if not self._wait_or_abort(0.3):
            self.send_robot_stop()
            self._is_playing = False
            return False

        # ---------------------------------------------------------------------
        # 2. Step 2: Dense Waypoint Streaming Loop
        # ---------------------------------------------------------------------
        prev_gripper_width = start_pt.gripper_width
        last_gripper_cmd_time = time.perf_counter()

        playback_start_wall = time.perf_counter()
        t_traj_start = points[0].timestamp

        for idx, pt in enumerate(points):
            if self._stop_playback_flag.is_set():
                logger.info(f"Playback aborted by user at point {idx}/{total_pts}.")
                self.send_robot_stop()
                self._is_playing = False
                return False

            # Calculate target playback time for this waypoint
            rel_t = (pt.timestamp - t_traj_start) / scale

            # Calculate delta time for servoj
            if idx + 1 < total_pts:
                next_pt = points[idx + 1]
                dt = max(0.010, (next_pt.timestamp - pt.timestamp) / scale)
            else:
                dt = 0.033

            # Transmit URScript command to robot
            if method.lower() == "movej":
                self._send_movej_step(pt.q, dt=dt)
            else:
                self._send_servoj_step(pt.q, dt=dt)

            # Command Gripper if width changed meaningfully (> 0.5 mm)
            # Throttle Modbus traffic to max 10 Hz to keep connection healthy
            now = time.perf_counter()
            width_diff = abs(pt.gripper_width - prev_gripper_width)
            if width_diff >= 0.5 and (now - last_gripper_cmd_time) >= 0.08:
                if self._gripper is not None:
                    try:
                        self._gripper.grip(pt.gripper_width, wait=False)
                        prev_gripper_width = pt.gripper_width
                        last_gripper_cmd_time = now
                    except Exception as e:
                        logger.debug(f"Modbus gripper command warning: {e}")

            # Update mock state if in simulation
            if self.mock:
                self._mock_q = list(pt.q)
                self._mock_tcp = list(pt.tcp_pose)
                self._mock_gripper_width = pt.gripper_width

            # Notify progress callback
            if progress_callback is not None:
                try:
                    progress_callback(idx, total_pts, pt)
                except Exception as cb_err:
                    logger.debug(f"Progress callback error: {cb_err}")

            # High-precision time synchronization with recorded trajectory
            target_wall_time = playback_start_wall + rel_t
            sleep_needed = target_wall_time - time.perf_counter()
            if sleep_needed > 0:
                time.sleep(sleep_needed)

        # Smooth deceleration at end of path
        self.send_robot_stop()
        logger.info(f"Trajectory playback complete ({total_pts} points executed).")
        self._is_playing = False
        return True

    def _send_servoj_step(self, q: List[float], dt: float):
        """Sends a smooth servoj command to the robot command socket."""
        if self._sock is not None:
            # Overlap lookahead time slightly beyond dt to avoid buffer starvation
            lookahead = 0.1
            gain = 300
            t_param = max(0.02, dt * 1.15)
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

    def _align_to_start_pose(self, start_pt: TrajectoryPoint, max_wait_s: float = 8.0) -> bool:
        """
        Smoothly positions the arm at the initial trajectory waypoint before
        high-frequency streaming begins, avoiding harsh kinematic jerks.
        """
        curr_q, _, _ = self.get_current_state()

        # Check maximum joint error
        max_err = max(abs(curr_q[i] - start_pt.q[i]) for i in range(6))
        if max_err < 0.025:  # Within ~1.4 degrees, ready to proceed
            logger.info("Arm already within tolerance of start waypoint.")
            return True

        logger.info(f"Moving arm to start waypoint (max joint displacement: {math.degrees(max_err):.1f}°)...")

        # Command movej with conservative speed & acceleration
        if self._sock is not None:
            q_str = ", ".join(f"{val:.5f}" for val in start_pt.q)
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
                self._mock_q = list(start_pt.q)
                return True

            curr_q, _, _ = self.get_current_state()
            err = max(abs(curr_q[i] - start_pt.q[i]) for i in range(6))
            if err < 0.035:
                logger.info(f"Arrived at start waypoint in {time.perf_counter() - t0:.2f}s.")
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

    # 4. Test Playback in Mock Mode
    print("\n4. Testing Mock Playback...")
    progress_count = 0

    def on_progress(idx, total, pt):
        nonlocal progress_count
        progress_count += 1

    success = pm.play_path(method="servoj", speed_scale=2.0, auto_align_start=False, progress_callback=on_progress)
    print(f"Playback finished: success={success}, progress updates={progress_count}")
    assert success, "Playback failed!"

    print("\nAll PathManager self-tests PASSED successfully!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UR5e Reference Path Manager")
    parser.add_argument("--test", action="store_true", help="Run internal self-tests in mock mode")
    args = parser.parse_args()

    if args.test or len(sys.argv) == 1:
        run_self_test()
