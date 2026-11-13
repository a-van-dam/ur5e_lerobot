#!/usr/bin/env python3
"""
start_position_config.py
========================
Centralized configuration manager for UR5e and OnRobot 2FG default start position (temp home).

Reads configuration directly from 'default_start_position.json' in the project root.
If the JSON file does not exist, it creates it with standard default coordinates:
  Joints (deg) : [-90.0, -40.0, -100.0, 120.0, -90.0, 120.0]
  Gripper (mm) : 105.0 (wide open)

All project scripts (gui_reference_path.py, gui_recorder.py, gui_tester.py,
gamepad_controller.py, path_manager.py, test_policy.py) import from this module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Canonical UR5e joint names in kinematic order (base to tool flange)
UR5E_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

# Hardcoded fallback values in case the file is missing or corrupted
FALLBACK_START_JOINTS_DEG = [-90.0, -40.0, -100.0, 120.0, -90.0, 120.0]
FALLBACK_START_GRIPPER_MM = 105.0

CONFIG_FILE_PATH = Path(__file__).resolve().parent / "default_start_position.json"


def load_start_position(config_path: Optional[Union[str, Path]] = None) -> Tuple[Dict[str, float], float]:
    """
    Loads default start joints (dict in degrees) and gripper width (mm) from JSON.
    If the file is not found, automatically creates the default JSON file.
    
    Returns:
        (joints_dict, gripper_mm):
          joints_dict: Dict with keys 'shoulder_pan'..'wrist_3' mapped to degrees
          gripper_mm: float aperture width in mm (clamped between 55.0 and 105.0)
    """
    p = Path(config_path) if config_path else CONFIG_FILE_PATH

    joints_dict: Dict[str, float] = {
        name: FALLBACK_START_JOINTS_DEG[i] for i, name in enumerate(UR5E_JOINT_NAMES)
    }
    gripper_mm: float = FALLBACK_START_GRIPPER_MM

    if not p.exists():
        save_start_position(joints=FALLBACK_START_JOINTS_DEG, gripper_mm=FALLBACK_START_GRIPPER_MM, config_path=p)
        return joints_dict, gripper_mm

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Parse joints: check 'joints_deg' dict first, then 'joints_list_deg'
        if "joints_deg" in data and isinstance(data["joints_deg"], dict):
            j_deg = data["joints_deg"]
            for name in UR5E_JOINT_NAMES:
                if name in j_deg:
                    joints_dict[name] = float(j_deg[name])
        elif "joints_list_deg" in data and isinstance(data["joints_list_deg"], list):
            vals = data["joints_list_deg"]
            if len(vals) == 6:
                for i, name in enumerate(UR5E_JOINT_NAMES):
                    joints_dict[name] = float(vals[i])
        elif "joints" in data and isinstance(data["joints"], list):
            vals = data["joints"]
            if len(vals) == 6:
                for i, name in enumerate(UR5E_JOINT_NAMES):
                    joints_dict[name] = float(vals[i])

        # 2. Parse gripper width
        for k in ("gripper_width_mm", "gripper_mm", "gripper_width", "gripper"):
            if k in data:
                gripper_mm = float(data[k])
                break

        # Clamp gripper between OnRobot 2FG physical limits: 55.0mm to 105.0mm
        gripper_mm = max(55.0, min(105.0, gripper_mm))

    except Exception as e:
        print(f"[CONFIG WARNING] Could not parse '{p.name}' ({e}). Using fallback defaults.")

    return joints_dict, gripper_mm


def save_start_position(
    joints: Union[List[float], Dict[str, float]],
    gripper_mm: float = 105.0,
    config_path: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Saves new start joint coordinates and gripper width to default_start_position.json.
    Accepts joints as either a 6-element list or dictionary.
    """
    p = Path(config_path) if config_path else CONFIG_FILE_PATH

    joints_dict: Dict[str, float] = {}
    joints_list: List[float] = []

    if isinstance(joints, dict):
        for name in UR5E_JOINT_NAMES:
            val = float(joints.get(name, 0.0))
            joints_dict[name] = round(val, 2)
            joints_list.append(round(val, 2))
    elif isinstance(joints, (list, tuple)) and len(joints) == 6:
        for i, name in enumerate(UR5E_JOINT_NAMES):
            val = float(joints[i])
            joints_dict[name] = round(val, 2)
            joints_list.append(round(val, 2))
    else:
        raise ValueError(f"Invalid joints parameter: expected 6 joints, got {joints}")

    gripper_mm = max(55.0, min(105.0, float(gripper_mm)))

    payload = {
        "joints_deg": joints_dict,
        "joints_list_deg": joints_list,
        "gripper_width_mm": round(gripper_mm, 1),
        "description": "Default start position (temp home) for UR5e arm and OnRobot 2FG gripper. Edit these values directly in a text editor. All Python scripts read from this file.",
    }

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True
    except Exception as e:
        print(f"[CONFIG ERROR] Failed to write '{p}': {e}")
        return False


# Convenience module-level accessors that read dynamically from default_start_position.json
def get_default_start_joints_dict() -> Dict[str, float]:
    """Returns {joint_name: angle_deg} from default_start_position.json."""
    j_dict, _ = load_start_position()
    return j_dict


def get_default_start_joints_deg() -> List[float]:
    """Returns [q0..q5] in degrees from default_start_position.json."""
    j_dict, _ = load_start_position()
    return [j_dict[name] for name in UR5E_JOINT_NAMES]


def get_default_start_joints_rad() -> List[float]:
    """Returns [q0..q5] in radians from default_start_position.json."""
    j_deg = get_default_start_joints_deg()
    return [math.radians(v) for v in j_deg]


def get_default_start_gripper_mm() -> float:
    """Returns gripper aperture in mm from default_start_position.json."""
    _, gripper_mm = load_start_position()
    return gripper_mm


if __name__ == "__main__":
    print(f"Testing start_position_config loading from: {CONFIG_FILE_PATH}")
    j_dict, g_mm = load_start_position()
    print("Loaded Start Joints (deg):", j_dict)
    print(f"Loaded Start Gripper: {g_mm:.1f} mm")
    print("Joints list in deg:", get_default_start_joints_deg())
    print("Joints list in rad:", [round(x, 4) for x in get_default_start_joints_rad()])
