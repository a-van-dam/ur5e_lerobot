import socket
import time
from typing import Any

# Safe import of RTDEReceiveInterface with clear error messaging if outside conda env
try:
    from rtde_receive import RTDEReceiveInterface
except ImportError:
    try:
        from ur_rtde import rtde_receive
        RTDEReceiveInterface = rtde_receive.RTDEReceiveInterface
    except ImportError:
        RTDEReceiveInterface = None

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position

from .configuration_ur5e import UR5eRobotConfig

# Standard Universal Robots 6-DoF joint names in kinematic order (base to tool flange)
UR5E_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


class UR5eRobot(Robot):
    """
    Hardware wrapper for the Universal Robots UR5e arm.
    Reads 125 Hz joint telemetry via RTDE (Port 30004) and commands motion via URScript socket (Port 30002).
    Includes Freedrive mode for physical hand-guiding and demonstration recording.
    """

    config_class = UR5eRobotConfig
    name = "ur5e"

    def __init__(self, config: UR5eRobotConfig):
        super().__init__(config)
        self.config: UR5eRobotConfig = config
        self.rx: RTDEReceiveInterface | None = None
        self.sock: socket.socket | None = None
        self._freedrive_active: bool = False

        # Camera management via LeRobot's unified camera subsystem
        self.cameras = make_cameras_from_configs(config.cameras)
        self.logs: dict[str, float] = {}

    @property
    def observation_features(self) -> dict[str, Any]:
        """Defines the structure of observations returned by get_observation()."""
        features: dict[str, Any] = {f"{j}.pos": float for j in UR5E_JOINTS}
        for cam_name, cam in self.cameras.items():
            features[cam_name] = (cam.height, cam.width, 3)
        return features

    @property
    def action_features(self) -> dict[str, type]:
        """Defines the structure of actions expected by send_action()."""
        return {f"{j}.pos": float for j in UR5E_JOINTS}

    @property
    def is_connected(self) -> bool:
        return self.rx is not None and self.sock is not None

    def connect(self, calibrate: bool = False) -> None:
        if self.is_connected:
            return

        if RTDEReceiveInterface is None:
            raise ImportError(
                "rtde_receive is not installed or not found in the current Python environment.\n"
                "Please make sure you are running in the 'lerobot' conda environment:\n"
                "    conda activate lerobot"
            )

        # 1. Connect RTDE telemetry on port 30004
        self.rx = RTDEReceiveInterface(self.config.robot_ip)

        # 2. Connect primary command socket on port 30002
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        self.sock.connect((self.config.robot_ip, self.config.command_port))

        # 3. Connect cameras if configured
        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    def configure(self) -> None:
        """Apply runtime configuration or verify connection health."""
        pass

    @property
    def is_calibrated(self) -> bool:
        # UR5e has factory-calibrated absolute optical encoders on every joint
        return True

    def calibrate(self) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        """Polls current joint angles (in radians) and latest camera frames."""
        if not self.is_connected:
            raise ConnectionError("UR5e is not connected. Call connect() first.")

        t0 = time.perf_counter()
        q = self.rx.getActualQ()
        self.logs["read_pos_dt_s"] = time.perf_counter() - t0

        obs: RobotObservation = {
            f"{name}.pos": float(val) for name, val in zip(UR5E_JOINTS, q)
        }

        # Query frame buffers from attached cameras
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        """Sends a motion command to target joint positions."""
        if not self.is_connected:
            raise ConnectionError("UR5e is not connected. Call connect() first.")

        # Ensure freedrive is deactivated before commanding positions
        if self._freedrive_active:
            self.disable_freedrive()

        t0 = time.perf_counter()

        # Optional safety clamping against current telemetry
        if self.config.max_relative_target is not None:
            current_q = self.rx.getActualQ()
            goal_present_pos = {
                f"{j}.pos": (action[f"{j}.pos"], current_q[i])
                for i, j in enumerate(UR5E_JOINTS)
            }
            action = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        target_q = [action[f"{j}.pos"] for j in UR5E_JOINTS]
        q_str = ", ".join(f"{val:.5f}" for val in target_q)

        cmd = f"movej([{q_str}], a={self.config.acceleration}, v={self.config.velocity})\n"
        self.sock.sendall(cmd.encode("utf-8"))
        self.logs["write_pos_dt_s"] = time.perf_counter() - t0

        return action

    def enable_freedrive(self) -> None:
        """
        Enable freedrive mode so the user can physically guide the arm by hand.
        Sends a URScript program holding freedrive_mode() until deactivated.
        """
        if not self.is_connected:
            raise ConnectionError("UR5e is not connected. Call connect() first.")

        program = (
            "def freedrive_prog():\n"
            "  freedrive_mode()\n"
            "  while True:\n"
            "    sync()\n"
            "  end\n"
            "end\n"
            "freedrive_prog()\n"
        )
        self.sock.sendall(program.encode("utf-8"))
        self._freedrive_active = True

    def disable_freedrive(self) -> None:
        """Exit freedrive mode and re-lock the joints in their current position."""
        if not self.is_connected:
            return

        program = (
            "def stop_freedrive():\n"
            "  end_freedrive_mode()\n"
            "  stopj(2.0)\n"
            "end\n"
            "stop_freedrive()\n"
        )
        self.sock.sendall(program.encode("utf-8"))
        self._freedrive_active = False

    def disconnect(self) -> None:
        """Cleanly closes camera feeds, socket, and RTDE telemetry receiver."""
        if getattr(self, "_freedrive_active", False):
            try:
                self.disable_freedrive()
            except Exception:
                pass

        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass

        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

        if self.rx is not None:
            try:
                self.rx.disconnect()
            except Exception:
                pass
            self.rx = None

