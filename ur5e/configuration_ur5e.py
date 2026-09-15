from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@dataclass
class UR5eRobotConfig(RobotConfig):
    # Network configuration for Universal Robots UR5e
    robot_ip: str = "192.168.1.102"
    command_port: int = 30002
    telemetry_port: int = 30004

    # Default motion limits for safe operation
    velocity: float = 0.2  # rad/s
    acceleration: float = 0.4  # rad/s^2

    # Safety limits: optional clamp on maximum relative joint jump per action
    max_relative_target: float | None = None

    # Cameras attached to workspace or robot (optional)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


# Register with LeRobot's Draccus ChoiceRegistry (safe if already registered in lerobot core)
try:
    RobotConfig.register_subclass("ur5e", UR5eRobotConfig)
except ValueError:
    pass

