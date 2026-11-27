try:
    # If installed in lerobot core, use the core-registered classes
    from lerobot.robots.ur5e import UR5eRobot, UR5eRobotConfig
    from lerobot.robots.ur5e.robot_ur5e import UR5E_JOINTS
except (ImportError, ModuleNotFoundError):
    # Otherwise use the standalone local definitions
    from .configuration_ur5e import UR5eRobotConfig
    from .robot_ur5e import UR5E_JOINTS, UR5eRobot

try:
    from lerobot.robots import make_robot_from_config as _lerobot_make_robot_from_config
except (ImportError, ModuleNotFoundError):
    _lerobot_make_robot_from_config = None


def make_robot_from_config(config):
    """Instantiate UR5eRobot directly or delegate other configs to LeRobot's factory."""
    if isinstance(config, UR5eRobotConfig):
        return UR5eRobot(config)
    if _lerobot_make_robot_from_config is not None:
        return _lerobot_make_robot_from_config(config)
    raise ImportError("lerobot.robots.make_robot_from_config is not available.")


__all__ = ["UR5E_JOINTS", "UR5eRobotConfig", "UR5eRobot", "make_robot_from_config"]
