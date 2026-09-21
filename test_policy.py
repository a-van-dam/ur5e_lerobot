#!/usr/bin/env python3
"""
UR5e LeRobot — Trained Policy Test & Live Verification Script
============================================================
Loads a trained imitation learning checkpoint (e.g. ACT, Diffusion), queries the live
camera and robot telemetry, predicts the target joint action, executes the movement safely
on the UR5e arm, and restores Freedrive mode so the arm can be hand-guided back.

Usage:
    # Run with default checkpoint and live hardware:
    python test_policy.py

    # Or specify custom checkpoint path:
    python test_policy.py --model outputs/train/ur5e_policy/checkpoint

    # Test offline in Simulation/Mock mode (without physical robot arm):
    python test_policy.py --mock
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# LeRobot imports
try:
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy
    from lerobot.robots import UR5eRobotConfig, make_robot_from_config
    from lerobot.robots.ur5e.robot_ur5e import UR5E_JOINTS, UR5eRobot
    LEROBOT_AVAILABLE = True
except ImportError as err:
    LEROBOT_AVAILABLE = False
    LEROBOT_IMPORT_ERR = str(err)


# =============================================================================
# 1. User Configuration & Safety Parameters (Adjust freely here)
# =============================================================================

# Path to trained policy checkpoint directory
DEFAULT_MODEL_PATH = "outputs/train/ur5e_policy/checkpoint"

# UR5e Network Configuration
DEFAULT_ROBOT_IP = "192.168.1.102"

# Camera Hardware Indices (V4L2 Linux device nodes)
DEFAULT_WORKSPACE_CAMERA_INDEX = 4   # HD Webcam C525 (Workspace)
DEFAULT_WRIST_CAMERA_INDEX = 6       # eFace 1325R (Wrist Flange, if 2-camera policy was trained)

# Conservative Safety Motion Limits
# Using gentle velocity & acceleration for the first live physical trials
SAFE_VELOCITY = 0.15      # rad/s (~8.6 deg/s) - slow, predictable, safe
SAFE_ACCELERATION = 0.30  # rad/s^2

# Safety Guard: Maximum allowed change on any single joint in a single test step
# Prevents violent arm motion if an undertrained model produces an erratic prediction
MAX_ALLOWED_JUMP_DEG = 45.0  # degrees

# If True, executes without waiting for terminal [Enter] confirmation
AUTO_EXECUTE = False

# Device configuration (CUDA if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 2. Camera Helpers
# =============================================================================

def capture_single_frame(
    camera_index: int,
    target_width: int,
    target_height: int,
    warmup_frames: int = 8,
) -> np.ndarray:
    """
    Opens the specified USB camera, reads a few warmup frames to allow
    sensor auto-exposure to stabilize, and returns a resized (H, W, 3) BGR image.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device /dev/video{camera_index}.")

    try:
        # Request full resolution from driver
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame = None
        for _ in range(warmup_frames):
            ret, tmp = cap.read()
            if ret and tmp is not None:
                frame = tmp
            time.sleep(0.03)

        if frame is None:
            raise RuntimeError(f"Failed to capture valid image frame from /dev/video{camera_index}.")

        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        return frame
    finally:
        cap.release()


# =============================================================================
# 3. Mock Robot Class (For Offline Testing without Hardware)
# =============================================================================

class MockUR5eRobot:
    """Simulated UR5e arm for testing code logic without physical hardware."""

    def __init__(self, config: Any):
        self.config = config
        self.is_connected = False
        self._freedrive_active = False
        # Starting ready pose: [0, -90, 90, -90, -90, 0] degrees in radians
        self._joints = {
            "shoulder_pan.pos": math.radians(0.0),
            "shoulder_lift.pos": math.radians(-90.0),
            "elbow.pos": math.radians(90.0),
            "wrist_1.pos": math.radians(-90.0),
            "wrist_2.pos": math.radians(-90.0),
            "wrist_3.pos": math.radians(0.0),
        }

    def connect(self):
        self.is_connected = True
        print("[MOCK ROBOT] Connected to simulated UR5e arm.")

    def get_observation(self) -> Dict[str, float]:
        return dict(self._joints)

    def send_action(self, action: Dict[str, float]) -> Dict[str, float]:
        if self._freedrive_active:
            self.disable_freedrive()
        for k, v in action.items():
            if k in self._joints:
                self._joints[k] = v
        time.sleep(1.0)
        return action

    def enable_freedrive(self):
        self._freedrive_active = True
        print("[MOCK ROBOT] Freedrive mode enabled.")

    def disable_freedrive(self):
        self._freedrive_active = False
        print("[MOCK ROBOT] Freedrive mode disabled.")

    def disconnect(self):
        self.is_connected = False
        print("[MOCK ROBOT] Disconnected.")


# =============================================================================
# 4. Main Verification Engine
# =============================================================================

def run_policy_test(
    model_path: str | Path,
    robot_ip: str = DEFAULT_ROBOT_IP,
    camera_index: int = DEFAULT_WORKSPACE_CAMERA_INDEX,
    wrist_camera_index: int = DEFAULT_WRIST_CAMERA_INDEX,
    safe_velocity: float = SAFE_VELOCITY,
    safe_acceleration: float = SAFE_ACCELERATION,
    max_jump_deg: float = MAX_ALLOWED_JUMP_DEG,
    auto_execute: bool = AUTO_EXECUTE,
    mock: bool = False,
    device_name: str = DEVICE,
):
    if not LEROBOT_AVAILABLE:
        print(f"[ERROR] LeRobot is not installed in this environment: {LEROBOT_IMPORT_ERR}", file=sys.stderr)
        return

    model_dir = Path(model_path).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Trained model checkpoint directory not found at: {model_dir}")

    config_json = model_dir / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(f"Missing 'config.json' in {model_dir}. Please ensure training finished.")

    device = torch.device(device_name if (device_name == "cuda" and torch.cuda.is_available()) else "cpu")

    print("\n" + "=" * 75)
    print("      UR5e Trained Policy Live Verification & Test Runner")
    print("=" * 75)
    print(f"  • Model Checkpoint:  {model_dir}")
    print(f"  • Compute Device:    {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  • Target Robot IP:   {robot_ip} {'[SIMULATED MOCK]' if mock else ''}")
    print(f"  • Safe Motion Speed: {safe_velocity} rad/s (~{math.degrees(safe_velocity):.1f}°/s)")
    print(f"  • Max Jump Guard:    ±{max_jump_deg}°")
    print("=" * 75 + "\n")

    # ─────────────────────────────────────────────────────────────────
    # Phase 1: Load Trained Policy & Pre/Post Processors
    # ─────────────────────────────────────────────────────────────────
    print("[1/5] Loading trained policy checkpoint and normalizers...")
    try:
        # Load policy architecture (ACT, Diffusion, etc.)
        policy = ACTPolicy.from_pretrained(model_dir).to(device)
        policy.eval()
        policy.reset()
        print(f"      ✔ Policy loaded successfully: {policy.__class__.__name__}")

        # Load input/output normalizer pipelines
        preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=model_dir)
        print("      ✔ Input/output normalizers loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load policy: {e}", file=sys.stderr)
        raise

    # Inspect model requirements
    img_features = getattr(policy.config, "image_features", {})
    has_workspace_cam = "observation.images.workspace" in img_features
    has_wrist_cam = "observation.images.wrist" in img_features

    # Determine target resolution
    workspace_shape = (240, 320)
    if has_workspace_cam:
        ft = img_features["observation.images.workspace"]
        if hasattr(ft, "shape") and len(ft.shape) >= 2:
            workspace_shape = (ft.shape[-2], ft.shape[-1])  # (H, W)
    target_h, target_w = workspace_shape
    print(f"      • Required Workspace Camera Resolution: {target_w}x{target_h}")
    if has_wrist_cam:
        print("      • Model requires secondary Wrist Camera ('observation.images.wrist').")

    # ─────────────────────────────────────────────────────────────────
    # Phase 2: Connect Robot & Camera Sensors
    # ─────────────────────────────────────────────────────────────────
    print("\n[2/5] Initializing live camera feed and UR5e robot connection...")

    # Connect Robot Arm
    if mock:
        robot = MockUR5eRobot(UR5eRobotConfig(robot_ip=robot_ip, velocity=safe_velocity, acceleration=safe_acceleration))
    else:
        robot_cfg = UR5eRobotConfig(
            robot_ip=robot_ip,
            velocity=safe_velocity,
            acceleration=safe_acceleration,
        )
        robot = make_robot_from_config(robot_cfg)

    print(f"      Connecting to UR5e at {robot_ip}...")
    robot.connect()
    print("      ✔ Robot connected successfully!")

    # Check if arm is in freedrive mode
    if getattr(robot, "_freedrive_active", False):
        print("      ℹ Robot is currently in FREEDRIVE mode.")

    # Acquire Current Joint Observation
    raw_obs = robot.get_observation()
    current_q_rad = [raw_obs[f"{j}.pos"] for j in UR5E_JOINTS]
    current_q_deg = [math.degrees(rad) for rad in current_q_rad]

    # Acquire Camera Frame(s)
    if mock:
        print("      [MOCK] Synthesizing camera observation...")
        raw_cam_frame = np.full((target_h, target_w, 3), (80, 70, 60), dtype=np.uint8)
        cv2.putText(raw_cam_frame, "Mock Camera Frame", (30, target_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        print(f"      Capturing live frame from /dev/video{camera_index}...")
        raw_cam_frame = capture_single_frame(camera_index, target_width=target_w, target_height=target_h)
        print("      ✔ Camera frame acquired.")

    # Save a copy of what the camera saw for inspection
    eval_dir = Path("outputs/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = eval_dir / "test_camera_input.png"
    cv2.imwrite(str(snapshot_path), raw_cam_frame)
    print(f"      • Snapshot saved for visual inspection: {snapshot_path}")

    # Prepare secondary camera if model requires it
    raw_wrist_frame = None
    if has_wrist_cam:
        if mock:
            raw_wrist_frame = np.full((target_h, target_w, 3), (60, 70, 80), dtype=np.uint8)
        else:
            print(f"      Capturing wrist frame from /dev/video{wrist_camera_index}...")
            raw_wrist_frame = capture_single_frame(wrist_camera_index, target_width=target_w, target_height=target_h)

    # ─────────────────────────────────────────────────────────────────
    # Phase 3: Run Policy Inference
    # ─────────────────────────────────────────────────────────────────
    print("\n[3/5] Computing predicted action trajectory via model...")

    # Convert image(s) to RGB, normalized float tensors (1, 3, H, W)
    img_rgb = cv2.cvtColor(raw_cam_frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    state_tensor = torch.from_numpy(np.array(current_q_rad, dtype=np.float32)).unsqueeze(0).to(device)

    batch: Dict[str, torch.Tensor] = {
        "observation.state": state_tensor,
        "observation.images.workspace": img_tensor,
    }
    if has_wrist_cam and raw_wrist_frame is not None:
        wrist_rgb = cv2.cvtColor(raw_wrist_frame, cv2.COLOR_BGR2RGB)
        wrist_tensor = torch.from_numpy(wrist_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        batch["observation.images.wrist"] = wrist_tensor

    with torch.no_grad():
        preprocessed_batch = preprocessor(batch)
        raw_action = policy.select_action(preprocessed_batch)
        postprocessed_action = postprocessor(raw_action)

    # Extract 6-DoF target joint values
    pred_action_np = postprocessed_action.squeeze().detach().cpu().numpy()
    if pred_action_np.ndim > 1:
        # In case action chunk has temporal dimension (T, 6), take the first action waypoint
        target_q_rad = pred_action_np[0].tolist()
    else:
        target_q_rad = pred_action_np.tolist()

    target_q_deg = [math.degrees(rad) for rad in target_q_rad]
    deltas_deg = [tgt - cur for tgt, cur in zip(target_q_deg, current_q_deg)]
    deltas_rad = [tgt - cur for tgt, cur in zip(target_q_rad, current_q_rad)]
    max_delta_deg = max(abs(d) for d in deltas_deg)

    # ─────────────────────────────────────────────────────────────────
    # Phase 4: Safety Review & User Confirmation
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("                    >>> PREDICTED ACTION REVIEW <<<")
    print("=" * 75)
    print(f" {'Joint':<16} | {'Current Pos':<16} | {'Target Pos':<16} | {'Shift (Delta)':<14}")
    print("-" * 75)
    for j_name, c_deg, c_rad, t_deg, t_rad, d_deg in zip(
        UR5E_JOINTS, current_q_deg, current_q_rad, target_q_deg, target_q_rad, deltas_deg
    ):
        status = "WARN" if abs(d_deg) > max_jump_deg else "OK"
        print(f" {j_name:<16} | {c_deg:+7.2f}° ({c_rad:+.3f}) | {t_deg:+7.2f}° ({t_rad:+.3f}) | {d_deg:+6.2f}° [{status}]")
    print("-" * 75)
    print(f" Maximum Joint Angular Shift: {max_delta_deg:.2f}°")
    print(f" Commanded Velocity:          {safe_velocity} rad/s (~{math.degrees(safe_velocity):.1f}°/s)")
    print(f" Commanded Acceleration:      {safe_acceleration} rad/s²")
    print("=" * 75)

    # Safety Limit Verification
    if max_delta_deg > max_jump_deg:
        print(f"\n[SAFETY WARNING] The model predicted an angular jump of {max_delta_deg:.1f}°,")
        print(f"                 which exceeds the safety threshold of ±{max_jump_deg}°!")
        print("                 This can occur if the model has not finished training or lighting is drastically different.")
        resp = input("\nType 'FORCE' to proceed anyway, or press Enter to safely abort: ").strip()
        if resp != "FORCE":
            print("[ABORT] Movement canceled by safety check. Re-enabling freedrive mode...")
            robot.enable_freedrive()
            robot.disconnect()
            return

    # User Confirmation Prompt
    if not auto_execute:
        print("\n>>> Please check that the workspace around the UR5e arm is CLEAR of obstacles.")
        print(">>> Hand on the E-Stop or spacebar recommended.")
        try:
            input("\nPress [ENTER] to execute this movement on the robot arm (or Ctrl+C to abort)... ")
        except KeyboardInterrupt:
            print("\n[ABORT] Movement canceled by user. Restoring freedrive mode...")
            robot.enable_freedrive()
            robot.disconnect()
            return

    # ─────────────────────────────────────────────────────────────────
    # Phase 5: Motion Execution & Freedrive Restoration
    # ─────────────────────────────────────────────────────────────────
    try:
        print("\n[4/5] Executing action on UR5e arm...")

        # 1. Switch out of freedrive mode before commanding position
        if getattr(robot, "_freedrive_active", False):
            print("      • Disabling freedrive mode...")
            robot.disable_freedrive()
            time.sleep(0.5)

        # 2. Construct action dictionary
        action_dict = {f"{j}.pos": float(val) for j, val in zip(UR5E_JOINTS, target_q_rad)}

        # Estimate time required for motion based on angular distance and velocity
        max_dist_rad = max(abs(d) for d in deltas_rad)
        motion_duration = max(1.5, (max_dist_rad / max(1e-4, safe_velocity)) + 1.0)

        print(f"      • Sending movej command (Estimated duration: {motion_duration:.1f}s)...")
        t_move_start = time.perf_counter()
        robot.send_action(action_dict)

        # Monitor motion until complete
        time.sleep(motion_duration)
        elapsed_move = time.perf_counter() - t_move_start

        # 3. Read final position after move
        obs_after = robot.get_observation()
        after_q_deg = [math.degrees(obs_after[f"{j}.pos"]) for j in UR5E_JOINTS]

        print("\n" + "=" * 75)
        print(f"       >>> MOVEMENT COMPLETED in {elapsed_move:.2f} seconds <<<")
        print("=" * 75)
        print(f" {'Joint':<16} | {'Target Pos':<16} | {'Actual Final Pos':<16} | {'Error':<10}")
        print("-" * 75)
        for j_name, t_deg, a_deg in zip(UR5E_JOINTS, target_q_deg, after_q_deg):
            err_deg = abs(a_deg - t_deg)
            print(f" {j_name:<16} | {t_deg:+7.2f}°         | {a_deg:+7.2f}°          | {err_deg:5.2f}°")
        print("=" * 75)

    except KeyboardInterrupt:
        print("\n\n[WARNING] Movement interrupted by user! Halting arm motion...")
    finally:
        # ─────────────────────────────────────────────────────────────
        # Final Step: Restore Freedrive Mode as requested
        # ─────────────────────────────────────────────────────────────
        print("\n[5/5] Re-enabling Freedrive mode for manual hand-guiding...")
        try:
            robot.enable_freedrive()
            print("\n" + "*" * 75)
            print("  >>> FREEDRIVE MODE IS NOW ACTIVE! <<<")
            print("  You can now physically grab the UR5e arm and guide it back")
            print("  to your desired starting/home position by hand.")
            print("*" * 75)
            try:
                input("\nPress [ENTER] when you are done guiding the robot arm to disconnect... ")
            except KeyboardInterrupt:
                pass
        except Exception as e:
            print(f"[WARNING] Could not activate freedrive: {e}")

        print("\nRe-locking joints and disconnecting safely...")
        try:
            robot.disable_freedrive()
        except Exception:
            pass
        robot.disconnect()
        print("✔ Teardown complete. Robot disconnected safely.\n")


# =============================================================================
# 5. CLI Entrypoint
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a trained LeRobot imitation learning model on the live UR5e arm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to checkpoint directory")
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP, help="UR5e arm PolyScope IP address")
    parser.add_argument("--camera", type=int, default=DEFAULT_WORKSPACE_CAMERA_INDEX, help="Workspace OpenCV camera index")
    parser.add_argument("--wrist-camera", type=int, default=DEFAULT_WRIST_CAMERA_INDEX, help="Secondary wrist camera index")
    parser.add_argument("--safe-vel", type=float, default=SAFE_VELOCITY, help="Safe joint velocity in rad/s")
    parser.add_argument("--safe-acc", type=float, default=SAFE_ACCELERATION, help="Safe joint acceleration in rad/s^2")
    parser.add_argument("--max-jump", type=float, default=MAX_ALLOWED_JUMP_DEG, help="Safety limit for max joint delta in degrees")
    parser.add_argument("--yes", action="store_true", help="Auto-execute movement without terminal confirmation prompt")
    parser.add_argument("--mock", action="store_true", help="Run in Simulation/Mock mode without physical robot")
    parser.add_argument("--device", type=str, default=DEVICE, help="Compute device (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_policy_test(
        model_path=args.model,
        robot_ip=args.robot_ip,
        camera_index=args.camera,
        wrist_camera_index=args.wrist_camera,
        safe_velocity=args.safe_vel,
        safe_acceleration=args.safe_acc,
        max_jump_deg=args.max_jump,
        auto_execute=args.yes,
        mock=args.mock,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()

