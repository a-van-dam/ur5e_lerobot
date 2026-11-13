#!/usr/bin/env python3
"""
UR5e LeRobot — Trained Policy Live Verification & Continuous Rollout Runner
===========================================================================
Loads a trained imitation learning checkpoint (e.g. ACT, Diffusion), queries the live
camera feed and UR5e RTDE telemetry at 30 Hz, predicts the continuous action trajectory,
executes smooth multi-second movements on the arm, and immediately re-enables Freedrive mode
upon completion so the user can hand-guide the arm back.

Usage:
    # Run continuous 10-second rollout on live hardware:
    python test_policy.py

    # Run custom rollout duration (e.g. 15 seconds):
    python test_policy.py --duration 15.0

    # Test offline in Simulation/Mock mode (no physical robot needed):
    python test_policy.py --mock --duration 5.0
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

PROJECT_NAME = "object_detection2"

# Path to trained policy checkpoint directory
DEFAULT_MODEL_PATH = "outputs/train/" + PROJECT_NAME + "/checkpoint"
if not Path(DEFAULT_MODEL_PATH).exists() and Path("outputs/train/ur5e_policy/checkpoint").exists():
    DEFAULT_MODEL_PATH = "outputs/train/ur5e_policy/checkpoint"

# UR5e Network Configuration
DEFAULT_ROBOT_IP = "192.168.1.102"

# Camera Hardware Indices (V4L2 Linux device nodes)
DEFAULT_WORKSPACE_CAMERA_INDEX = 4   # HD Webcam C525 (Workspace)
DEFAULT_WRIST_CAMERA_INDEX = 6       # eFace 1325R (Wrist Flange, if 2-camera policy was trained)

# Continuous Rollout Configuration
DEFAULT_ROLLOUT_DURATION_S = 10.0    # Trajectory execution duration in seconds (~300 steps @ 30 Hz)
CONTROL_LOOP_FPS = 30                # Control frequency matching dataset recording

# Conservative Safety Motion Limits
SAFE_VELOCITY = 0.20      # rad/s (~11.5 deg/s) - smooth, controlled, safe
SAFE_ACCELERATION = 0.40  # rad/s^2

# Safety Guard: Maximum allowed change on any single joint between consecutive steps
MAX_ALLOWED_JUMP_DEG = 45.0  # degrees

# If True, executes without waiting for terminal [Enter] confirmation
AUTO_EXECUTE = False

# Device configuration (CUDA if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 2. Camera Capture Helpers
# =============================================================================

class CameraStream:
    """Manages continuous, low-latency OpenCV video capture with auto-exposure stabilization."""

    def __init__(self, camera_index: int, width: int, height: int):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self, warmup_frames: int = 10):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device /dev/video{self.camera_index}.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Warmup frames to let sensor gain/white-balance stabilize
        for _ in range(warmup_frames):
            self.cap.read()
            time.sleep(0.02)

    def read_frame(self) -> np.ndarray:
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera is not opened.")
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read video frame from /dev/video{self.camera_index}.")
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# =============================================================================
# 3. Mock Robot Class (For Offline Testing without Hardware)
# =============================================================================

class MockUR5eRobot:
    """Simulated UR5e arm for testing full trajectory rollout without hardware."""

    def __init__(self, config: Any):
        self.config = config
        self.is_connected = False
        self._freedrive_active = False
        self.sock = None
        try:
            from start_position_config import get_default_start_joints_dict
            _sp_dict = get_default_start_joints_dict()
            self._joints = {f"{k}.pos": math.radians(v) for k, v in _sp_dict.items()}
        except Exception:
            self._joints = {
                "shoulder_pan.pos": math.radians(-90.0),
                "shoulder_lift.pos": math.radians(-40.0),
                "elbow.pos": math.radians(-100.0),
                "wrist_1.pos": math.radians(120.0),
                "wrist_2.pos": math.radians(-90.0),
                "wrist_3.pos": math.radians(120.0),
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
        return action

    def stream_servoj(self, target_q: List[float], dt: float = 0.033):
        for i, j in enumerate(UR5E_JOINTS):
            self._joints[f"{j}.pos"] = target_q[i]

    def stop_motion(self):
        pass

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
# 4. Main Verification & Continuous Rollout Engine
# =============================================================================

def run_policy_rollout(
    model_path: str | Path,
    robot_ip: str = DEFAULT_ROBOT_IP,
    rollout_duration_s: float = DEFAULT_ROLLOUT_DURATION_S,
    camera_index: int = DEFAULT_WORKSPACE_CAMERA_INDEX,
    wrist_camera_index: int = DEFAULT_WRIST_CAMERA_INDEX,
    safe_velocity: float = SAFE_VELOCITY,
    safe_acceleration: float = SAFE_ACCELERATION,
    max_jump_deg: float = MAX_ALLOWED_JUMP_DEG,
    n_action_steps: Optional[int] = None,
    temporal_ensemble: bool = True,
    temporal_ensemble_coeff: Optional[float] = 0.01,
    auto_execute: bool = AUTO_EXECUTE,
    single_step: bool = False,
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
    total_steps = 1 if single_step else max(1, int(rollout_duration_s * CONTROL_LOOP_FPS))

    print("\n" + "=" * 75)
    print("      UR5e Trained Policy Live Rollout & Verification Engine")
    print("=" * 75)
    print(f"  • Model Checkpoint:   {model_dir}")
    print(f"  • Compute Device:     {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  • Target Robot IP:    {robot_ip} {'[SIMULATED MOCK]' if mock else ''}")
    print(f"  • Rollout Duration:   {rollout_duration_s:.1f}s ({total_steps} sequential action steps @ {CONTROL_LOOP_FPS} Hz)")
    print(f"  • Safe Motion Speed:  {safe_velocity} rad/s (~{math.degrees(safe_velocity):.1f}°/s)")
    print(f"  • Max Jump Guard:     ±{max_jump_deg}°")
    print("=" * 75 + "\n")

    # ─────────────────────────────────────────────────────────────────
    # Phase 1: Load Trained Policy & Pre/Post Processors
    # ─────────────────────────────────────────────────────────────────
    print("[1/5] Loading trained policy checkpoint and normalizers...")
    try:
        policy = ACTPolicy.from_pretrained(model_dir).to(device)
        policy.eval()
        policy.reset()
        print(f"      ✔ Policy loaded successfully: {policy.__class__.__name__}")

        preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=model_dir)
        print("      ✔ Input/output normalizers loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load policy: {e}", file=sys.stderr)
        raise

    # Inspect camera requirements
    img_features = getattr(policy.config, "image_features", {})
    has_workspace_cam = "observation.images.workspace" in img_features
    has_wrist_cam = "observation.images.wrist" in img_features

    workspace_shape = (240, 320)
    if has_workspace_cam:
        ft = img_features["observation.images.workspace"]
        if hasattr(ft, "shape") and len(ft.shape) >= 2:
            workspace_shape = (ft.shape[-2], ft.shape[-1])
    target_h, target_w = workspace_shape
    print(f"      • Required Camera Resolution: {target_w}x{target_h}")
    if has_wrist_cam:
        print("      • Model utilizes secondary Wrist Camera ('observation.images.wrist').")

    # Check if model was trained in grayscale mode
    is_grayscale = False
    for info_candidate in [model_dir / "training_info.json", model_dir.parent / "training_info.json"]:
        if info_candidate.exists():
            try:
                with open(info_candidate, "r") as f_info:
                    t_info = json.load(f_info)
                    if t_info.get("grayscale") is True or t_info.get("image_channels", 3) == 1:
                        is_grayscale = True
                        break
            except Exception:
                pass

    if is_grayscale:
        print("      • Vision Input Mode:  Grayscale (1-Ch luminance -> live frames converted)")
    else:
        print("      • Vision Input Mode:  RGB (3-Ch color)")

    # Inspect decision horizon and execution steps
    model_fps = 30
    default_n_steps = getattr(policy.config, "n_action_steps", 10)
    for info_candidate in [model_dir / "training_info.json", model_dir.parent / "training_info.json"]:
        if info_candidate.exists():
            try:
                with open(info_candidate, "r") as f_info:
                    t_info = json.load(f_info)
                    model_fps = int(t_info.get("fps", 30))
                    default_n_steps = int(t_info.get("n_action_steps", default_n_steps))
                    break
            except Exception:
                pass

    # Configure Temporal Ensembling vs Open-Loop Chunks
    use_te = temporal_ensemble
    te_coeff = temporal_ensemble_coeff if temporal_ensemble_coeff is not None else 0.01

    if use_te and hasattr(policy, "temporal_ensembler"):
        try:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
            policy.config.temporal_ensemble_coeff = float(te_coeff)
            policy.config.n_action_steps = 1
            policy.temporal_ensembler = ACTTemporalEnsembler(
                float(te_coeff), getattr(policy.config, "chunk_size", 100)
            )
            policy.reset()
            print(f"      • Replanning Mode:    Temporal Ensembling Active (m={te_coeff:.3f}, 30 Hz continuous smooth blend)")
        except Exception as e:
            print(f"      • Note: Temporal ensembling initialization fallback ({e}). Using open-loop chunks.")
            use_te = False

    if not use_te:
        if hasattr(policy.config, "temporal_ensemble_coeff"):
            policy.config.temporal_ensemble_coeff = None
        exec_steps = n_action_steps if n_action_steps is not None else default_n_steps
        if hasattr(policy.config, "n_action_steps"):
            policy.config.n_action_steps = int(exec_steps)
        if hasattr(policy, "drop_queued_actions"):
            policy.drop_queued_actions()
        policy.reset()
        dt_decision = exec_steps / max(1, model_fps)
        print(f"      • Replanning Mode:    Open-Loop Chunks ({exec_steps} steps, {dt_decision:.2f}s @ {model_fps} FPS)")

    # ─────────────────────────────────────────────────────────────────
    # Phase 2: Connect Robot & Camera Sensors
    # ─────────────────────────────────────────────────────────────────
    print("\n[2/5] Connecting to robot arm and opening camera streams...")

    if mock:
        robot = MockUR5eRobot(UR5eRobotConfig(robot_ip=robot_ip, velocity=safe_velocity, acceleration=safe_acceleration))
    else:
        robot_cfg = UR5eRobotConfig(
            robot_ip=robot_ip,
            velocity=safe_velocity,
            acceleration=safe_acceleration,
        )
        robot = make_robot_from_config(robot_cfg)

    robot.connect()
    print(f"      ✔ Connected to UR5e ({robot_ip}).")

    cam1_stream = None
    cam2_stream = None
    if not mock:
        print(f"      Opening primary camera /dev/video{camera_index}...")
        cam1_stream = CameraStream(camera_index, width=target_w, height=target_h)
        cam1_stream.open(warmup_frames=10)
        print("      ✔ Primary camera active.")

        if has_wrist_cam:
            print(f"      Opening secondary camera /dev/video{wrist_camera_index}...")
            cam2_stream = CameraStream(wrist_camera_index, width=target_w, height=target_h)
            cam2_stream.open(warmup_frames=10)
            print("      ✔ Secondary camera active.")

    # Grab initial snapshot for verification
    if mock:
        initial_cam_frame = np.full((target_h, target_w, 3), (80, 70, 60), dtype=np.uint8)
        cv2.putText(initial_cam_frame, "Mock Camera Frame", (30, target_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        initial_cam_frame = cam1_stream.read_frame()

    if is_grayscale:
        initial_cam_frame = cv2.cvtColor(cv2.cvtColor(initial_cam_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    eval_dir = Path("outputs/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = eval_dir / "test_camera_input.png"
    cv2.imwrite(str(snapshot_path), initial_cam_frame)
    print(f"      • Snapshot saved: {snapshot_path}")

    # ─────────────────────────────────────────────────────────────────
    # Phase 3: Initial Policy Inference & Trajectory Preview
    # ─────────────────────────────────────────────────────────────────
    print("\n[3/5] Sampling initial model prediction for trajectory safety review...")

    obs_start = robot.get_observation()
    start_q_rad = [obs_start[f"{j}.pos"] for j in UR5E_JOINTS]
    start_q_deg = [math.degrees(rad) for rad in start_q_rad]

    # Run 1 inference step to inspect the starting waypoint
    img_rgb = cv2.cvtColor(initial_cam_frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    state_tensor = torch.from_numpy(np.array(start_q_rad, dtype=np.float32)).unsqueeze(0).to(device)

    batch = {
        "observation.state": state_tensor,
        "observation.images.workspace": img_tensor,
    }
    if has_wrist_cam and cam2_stream is not None:
        wrist_rgb = cv2.cvtColor(cam2_stream.read_frame(), cv2.COLOR_BGR2RGB)
        batch["observation.images.wrist"] = torch.from_numpy(wrist_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        pre_batch = preprocessor(batch)
        pred_raw = policy.select_action(pre_batch)
        pred_post = postprocessor(pred_raw)

    first_target_rad = pred_post.squeeze().detach().cpu().numpy()
    if first_target_rad.ndim > 1:
        first_target_rad = first_target_rad[0].tolist()
    else:
        first_target_rad = first_target_rad.tolist()

    first_target_deg = [math.degrees(r) for r in first_target_rad]
    deltas_deg = [t - c for t, c in zip(first_target_deg, start_q_deg)]
    max_delta_deg = max(abs(d) for d in deltas_deg)

    # ─────────────────────────────────────────────────────────────────
    # Phase 4: Safety Review & User Confirmation
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("                    >>> PREDICTED TRAJECTORY PREVIEW <<<")
    print("=" * 75)
    print(f" {'Joint':<16} | {'Current Pos':<16} | {'First Target':<16} | {'Initial Shift':<14}")
    print("-" * 75)
    for j_name, c_deg, c_rad, t_deg, t_rad, d_deg in zip(
        UR5E_JOINTS, start_q_deg, start_q_rad, first_target_deg, first_target_rad, deltas_deg
    ):
        status = "WARN" if abs(d_deg) > max_jump_deg else "OK"
        print(f" {j_name:<16} | {c_deg:+7.2f}° ({c_rad:+.3f}) | {t_deg:+7.2f}° ({t_rad:+.3f}) | {d_deg:+6.2f}° [{status}]")
    print("-" * 75)
    print(f" Initial Angular Shift:       {max_delta_deg:.2f}°")
    print(f" Rollout Duration:            {rollout_duration_s:.1f} seconds ({total_steps} sequential action steps)")
    print(f" Commanded Motion Velocity:   {safe_velocity} rad/s (~{math.degrees(safe_velocity):.1f}°/s)")
    print("=" * 75)

    if max_delta_deg > max_jump_deg:
        print(f"\n[SAFETY WARNING] Initial predicted shift of {max_delta_deg:.1f}° exceeds limit (±{max_jump_deg}°).")
        resp = input("Type 'FORCE' to proceed anyway, or press Enter to safely abort: ").strip()
        if resp != "FORCE":
            print("[ABORT] Rollout canceled by safety check.")
            robot.enable_freedrive()
            robot.disconnect()
            return

    if not auto_execute:
        print("\n>>> Ensure the workspace around the UR5e arm is CLEAR of all obstacles.")
        print(">>> Keep your hand near the PolyScope Emergency Stop or spacebar.")
        try:
            input(f"\nPress [ENTER] to begin {rollout_duration_s:.1f}s live trajectory rollout (or Ctrl+C to abort)... ")
        except KeyboardInterrupt:
            print("\n[ABORT] Rollout canceled by user.")
            robot.enable_freedrive()
            robot.disconnect()
            return

    # ─────────────────────────────────────────────────────────────────
    # Phase 5: Continuous Multi-Step Trajectory Rollout Loop
    # ─────────────────────────────────────────────────────────────────
    step_count = 0
    start_rollout_time = time.perf_counter()
    loop_period = 1.0 / CONTROL_LOOP_FPS

    try:
        print(f"\n[4/5] Commencing live {rollout_duration_s:.1f}s demonstration rollout...")

        # Switch arm out of freedrive mode
        if getattr(robot, "_freedrive_active", False):
            print("      • Disabling freedrive mode for autonomous execution...")
            robot.disable_freedrive()
            time.sleep(0.5)

        # Reset policy action queue to cleanly start chunk trajectory
        policy.reset()

        last_target_q = list(start_q_rad)

        while step_count < total_steps:
            t_tick_start = time.perf_counter()
            step_count += 1

            # 1. Acquire latest camera frame
            if mock:
                cam_frame = initial_cam_frame
            else:
                cam_frame = cam1_stream.read_frame()

            if is_grayscale:
                cam_frame = cv2.cvtColor(cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

            # 2. Acquire latest robot telemetry
            obs = robot.get_observation()
            current_q = [obs[f"{j}.pos"] for j in UR5E_JOINTS]

            # 3. Construct model input tensors
            img_rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            state_tensor = torch.from_numpy(np.array(current_q, dtype=np.float32)).unsqueeze(0).to(device)

            step_batch = {
                "observation.state": state_tensor,
                "observation.images.workspace": img_tensor,
            }
            if has_wrist_cam and cam2_stream is not None:
                wrist_raw = cam2_stream.read_frame()
                if is_grayscale:
                    wrist_raw = cv2.cvtColor(cv2.cvtColor(wrist_raw, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                wrist_rgb = cv2.cvtColor(wrist_raw, cv2.COLOR_BGR2RGB)
                step_batch["observation.images.wrist"] = torch.from_numpy(wrist_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            # 4. Predict next action
            with torch.no_grad():
                pre_batch = preprocessor(step_batch)
                act_raw = policy.select_action(pre_batch)
                act_post = postprocessor(act_raw)

            step_target_rad = act_post.squeeze().detach().cpu().numpy()
            if step_target_rad.ndim > 1:
                step_target_rad = step_target_rad[0].tolist()
            else:
                step_target_rad = step_target_rad.tolist()

            # 5. Safety step delta clamp
            clamped_target = []
            max_jump_rad = math.radians(max_jump_deg)
            for tgt, last in zip(step_target_rad, last_target_q):
                d = tgt - last
                d = max(-max_jump_rad, min(max_jump_rad, d))
                clamped_target.append(last + d)
            last_target_q = list(clamped_target)

            # 6. Command robot motion smoothly
            if mock:
                robot.stream_servoj(clamped_target, dt=loop_period)
            else:
                q_str = ", ".join(f"{val:.5f}" for val in clamped_target)
                servoj_cmd = f"servoj([{q_str}], t={loop_period*1.2:.4f}, lookahead_time=0.1, gain=300)\n"
                if hasattr(robot, "sock") and robot.sock is not None:
                    robot.sock.sendall(servoj_cmd.encode("utf-8"))
                else:
                    action_dict = {f"{j}.pos": float(val) for j, val in zip(UR5E_JOINTS, clamped_target)}
                    robot.send_action(action_dict)

            # 7. Live terminal progress bar
            elapsed = time.perf_counter() - start_rollout_time
            progress = min(1.0, elapsed / max(1e-4, rollout_duration_s))
            bar_len = 20
            filled = int(bar_len * progress)
            bar_str = "█" * filled + "░" * (bar_len - filled)

            q_degs = [math.degrees(r) for r in clamped_target]
            deg_summary = f"P:{q_degs[0]:+5.1f}° L:{q_degs[1]:+5.1f}° E:{q_degs[2]:+5.1f}° W1:{q_degs[3]:+5.1f}°"

            print(
                f"\r[{bar_str}] {elapsed:4.1f}s / {rollout_duration_s:.1f}s "
                f"(Step {step_count:3d}/{total_steps}) | {deg_summary}",
                end="",
                flush=True,
            )

            # Regulate loop rate to 30 Hz
            t_used = time.perf_counter() - t_tick_start
            sleep_time = max(0.0, loop_period - t_used)
            if sleep_time > 0:
                time.sleep(sleep_time)

        total_rollout_time = time.perf_counter() - start_rollout_time
        print(f"\n\n" + "=" * 75)
        print(f"       >>> DEMONSTRATION ROLLOUT COMPLETED in {total_rollout_time:.1f}s <<<")
        print(f"       Executed {step_count} sequential 30 Hz action commands.")
        print("=" * 75)

    except KeyboardInterrupt:
        print("\n\n[WARNING] Rollout interrupted by user! Halting arm motion...")
    finally:
        # Halt physical arm motion smoothly
        if not mock and hasattr(robot, "sock") and robot.sock is not None:
            try:
                robot.sock.sendall(b"stopj(2.0)\n")
                time.sleep(0.5)
            except Exception:
                pass

        if cam1_stream is not None:
            cam1_stream.close()
        if cam2_stream is not None:
            cam2_stream.close()

        # ─────────────────────────────────────────────────────────────
        # Final Step: Restore Freedrive Mode as requested
        # ─────────────────────────────────────────────────────────────
        print("\n[5/5] Re-enabling Freedrive mode for manual hand-guiding...")
        try:
            robot.enable_freedrive()
            print("\n" + "*" * 75)
            print("  >>> FREEDRIVE MODE IS NOW ACTIVE! <<<")
            print("  You can now physically grab the UR5e arm and guide it back")
            print("  to your starting position by hand.")
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
        description="Run continuous trajectory rollout of a trained policy on the live UR5e arm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to checkpoint directory")
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP, help="UR5e arm PolyScope IP address")
    parser.add_argument("--duration", type=float, default=DEFAULT_ROLLOUT_DURATION_S, help="Trajectory rollout duration in seconds")
    parser.add_argument("--camera", type=int, default=DEFAULT_WORKSPACE_CAMERA_INDEX, help="Workspace OpenCV camera index")
    parser.add_argument("--wrist-camera", type=int, default=DEFAULT_WRIST_CAMERA_INDEX, help="Secondary wrist camera index")
    parser.add_argument("--safe-vel", type=float, default=SAFE_VELOCITY, help="Safe joint velocity in rad/s")
    parser.add_argument("--safe-acc", type=float, default=SAFE_ACCELERATION, help="Safe joint acceleration in rad/s^2")
    parser.add_argument("--max-jump", type=float, default=MAX_ALLOWED_JUMP_DEG, help="Safety limit for max joint delta in degrees")
    parser.add_argument("--single-step", action="store_true", help="Execute only a single action step instead of full rollout")
    parser.add_argument("--yes", action="store_true", help="Auto-execute movement without terminal confirmation prompt")
    parser.add_argument("--mock", action="store_true", help="Run in Simulation/Mock mode without physical robot")
    parser.add_argument("--n-action-steps", type=int, default=None, help="Action execution steps before re-querying camera observations")
    parser.add_argument(
        "--temporal-ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ACT Temporal Ensembling (smooth 30 Hz continuous exponential moving average)",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        help="Decay coefficient for temporal ensembling (default: 0.01 from ACT paper)",
    )
    parser.add_argument("--device", type=str, default=DEVICE, help="Compute device (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_policy_rollout(
        model_path=args.model,
        robot_ip=args.robot_ip,
        rollout_duration_s=args.duration,
        camera_index=args.camera,
        wrist_camera_index=args.wrist_camera,
        safe_velocity=args.safe_vel,
        safe_acceleration=args.safe_acc,
        max_jump_deg=args.max_jump,
        n_action_steps=args.n_action_steps,
        temporal_ensemble=args.temporal_ensemble,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        auto_execute=args.yes,
        single_step=args.single_step,
        mock=args.mock,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
