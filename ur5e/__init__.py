try:
    # If installed in lerobot core, use the core-registered classes
    from lerobot.robots.ur5e import UR5eRobot, UR5eRobotConfig
except (ImportError, ModuleNotFoundError):
    # Otherwise use the standalone local definitions
    from .configuration_ur5e import UR5eRobotConfig
    from .robot_ur5e import UR5eRobot

__all__ = ["UR5eRobotConfig", "UR5eRobot"]

