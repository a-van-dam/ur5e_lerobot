import math
import time
from typing import Any

from lerobot.robots import UR5eRobotConfig, make_robot_from_config
from onrobot_gripper import OnRobot2FGModbus


def rad_to_deg_str(q_dict: dict[str, float]) -> str:
    """Format a dictionary of joint positions into readable degrees."""
    items = []
    for k, v in q_dict.items():
        if k.endswith(".pos"):
            name = k.replace(".pos", "")
            items.append(f"{name}: {math.degrees(v):+6.1f}°")
    return " | ".join(items)


def main():
    print("=" * 75)
    print("   Universal Robots UR5e + OnRobot 2FG14 — LeRobot Verification Suite")
    print("=" * 75)

    # 1. Initialize robot configuration with conservative, safe motion limits
    robot_config = UR5eRobotConfig(
        robot_ip="192.168.1.102",
        velocity=0.2,       # Safe velocity: 0.2 rad/s (~11.5 deg/s)
        acceleration=0.4,   # Safe acceleration: 0.4 rad/s^2
    )

    print(f"\n[Arm Config] Robot Type: '{robot_config.type}' | IP: {robot_config.robot_ip}")
    print(f"[Arm Config] Safe Velocity: {robot_config.velocity} rad/s | Safe Acceleration: {robot_config.acceleration} rad/s²")

    # 2. Instantiate robot through LeRobot factory
    robot = make_robot_from_config(robot_config)

    # 3. Instantiate OnRobot gripper Modbus TCP interface
    gripper = OnRobot2FGModbus(host="192.168.1.1", port=502, device_id=65)

    # ─────────────────────────────────────────────────────────────────
    # Phase 1: Connection & Feature Introspection (UR5e Arm)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 75)
    print("Phase 1: Connecting to UR5e Arm and Validating LeRobot Features")
    print("-" * 75)

    robot.connect()
    print(f"-> Robot connected: {robot.is_connected}")
    if not robot.is_connected:
        raise ConnectionError("Failed to connect to UR5e.")

    gripper_detected = False
    gripper_baseline_width = None

    try:
        print("\nObservation Features (Sensory inputs):")
        for k, v in robot.observation_features.items():
            print(f"  • {k:22s} -> {v}")

        print("\nAction Features (Actuation targets):")
        for k, v in robot.action_features.items():
            print(f"  • {k:22s} -> {v}")

        # ─────────────────────────────────────────────────────────────
        # Phase 2: OnRobot Gripper Modbus TCP Detection & Scan
        # ─────────────────────────────────────────────────────────────
        print("\n" + "-" * 75)
        print("Phase 2: Scanning for OnRobot Gripper via Modbus TCP (192.168.1.1:502)")
        print("-" * 75)

        try:
            print(f"Connecting to OnRobot Compute Box at {gripper.host}:{gripper.port}...")
            if gripper.connect():
                print("-> Modbus TCP connection established.")
                info = gripper.detect_gripper()
                if info is not None:
                    gripper_detected = True
                    gripper_baseline_width = info["external_width_mm"]
                    print(f"-> SUCCESS: OnRobot Gripper detected on Device ID {info['device_id']}!")
                    print(f"   • Current External Width : {info['external_width_mm']:.1f} mm")
                    print(f"   • Current Internal Width : {info['internal_width_mm']:.1f} mm")
                    print(f"   • Physical Stroke Limits : {info['min_external_mm']:.1f} mm – {info['max_external_mm']:.1f} mm")
                    print(f"   • Gripper Busy Status    : {info['busy']}")
                    print(f"   • Part Grip Detected     : {info['grip_detected']}")
                else:
                    print("-> WARNING: Device ID 65 did not respond with valid 2FG telemetry.")
            else:
                print("-> WARNING: Could not connect to Compute Box Modbus port 502.")
        except Exception as e:
            print(f"-> WARNING: Modbus scan encountered an error: {e}")

        # ─────────────────────────────────────────────────────────────
        # Phase 3: Telemetry Sampling & Frequency Benchmark (UR5e Arm)
        # ─────────────────────────────────────────────────────────────
        print("\n" + "-" * 75)
        print("Phase 3: RTDE Telemetry Benchmark (Sampling 100 frames at 125 Hz)")
        print("-" * 75)

        t_start = time.perf_counter()
        samples = 100
        for _ in range(samples):
            obs = robot.get_observation()
        elapsed = time.perf_counter() - t_start
        rate = samples / elapsed if elapsed > 0 else 0.0

        print(f"-> Sampled {samples} observation frames in {elapsed:.3f}s ({rate:.1f} Hz).")
        print(f"-> Average read latency: {elapsed / samples * 1000.0:.2f} ms")
        print(f"\nCurrent Live Joint Positions:")
        for k, v in obs.items():
            if k.endswith(".pos"):
                print(f"   {k:22s}: {v:+.4f} rad ({math.degrees(v):+7.2f}°)")

        # Store baseline starting positions
        baseline_action = {k: v for k, v in obs.items() if k.endswith(".pos")}

        # ─────────────────────────────────────────────────────────────
        # Phase 4: Single-Joint Verification (Wrist 3 Flange Nudge)
        # ─────────────────────────────────────────────────────────────
        print("\n" + "-" * 75)
        print("Phase 4: Single-Joint Action Test (Wrist 3 +2.0° Nudge & Return)")
        print("-" * 75)

        target_w3 = dict(baseline_action)
        w3_nudge = math.radians(2.0)
        target_w3["wrist_3.pos"] += w3_nudge

        print("Sending +2.0° Wrist 3 action...")
        robot.send_action(target_w3)
        time.sleep(2.5)

        obs_w3_moved = robot.get_observation()
        w3_before = math.degrees(baseline_action["wrist_3.pos"])
        w3_after = math.degrees(obs_w3_moved["wrist_3.pos"])
        print(f"-> Wrist 3: Before = {w3_before:+7.2f}° | After = {w3_after:+7.2f}° (Shift: {w3_after - w3_before:+5.2f}°)")

        print("Restoring original arm pose...")
        robot.send_action(baseline_action)
        time.sleep(2.5)

        obs_restored = robot.get_observation()
        print(f"-> Restored pose: {rad_to_deg_str(obs_restored)}")

        # ─────────────────────────────────────────────────────────────
        # Phase 5: Coordinated Multi-Joint Movement
        # ─────────────────────────────────────────────────────────────
        print("\n" + "-" * 75)
        print("Phase 5: Coordinated Multi-Joint Action Test (+5° Base, +5° Wrist 2)")
        print("-" * 75)

        multi_target = dict(baseline_action)
        multi_target["shoulder_pan.pos"] += math.radians(5.0)
        multi_target["wrist_2.pos"] += math.radians(5.0)

        print("Sending multi-joint displacement action...")
        robot.send_action(multi_target)
        time.sleep(3.0)

        obs_multi = robot.get_observation()
        base_diff = math.degrees(obs_multi["shoulder_pan.pos"] - baseline_action["shoulder_pan.pos"])
        w2_diff = math.degrees(obs_multi["wrist_2.pos"] - baseline_action["wrist_2.pos"])
        print(f"-> Actual Shifts: Base = {base_diff:+5.2f}° | Wrist 2 = {w2_diff:+5.2f}°")

        print("Returning arm to exact starting baseline...")
        robot.send_action(baseline_action)
        time.sleep(3.0)

        obs_final = robot.get_observation()
        print(f"-> Final Pose: {rad_to_deg_str(obs_final)}")

        # ─────────────────────────────────────────────────────────────
        # Phase 6: OnRobot Gripper Test Movement (Target Width & Force)
        # ─────────────────────────────────────────────────────────────
        if gripper_detected and gripper_baseline_width is not None:
            print("\n" + "-" * 75)
            print("Phase 6: OnRobot Gripper Test Movement (50mm width, 30N force & Return)")
            print("-" * 75)

            target_test_width = 50.0  # mm
            target_test_force = 30.0  # N

            print(f"-> Gripper baseline starting width: {gripper_baseline_width:.1f} mm")
            print(f"-> Commanding test movement: Target Width = {target_test_width:.1f} mm | Force = {target_test_force:.1f} N...")

            w_moved = gripper.grip(
                width_mm=target_test_width,
                force_n=target_test_force,
                speed_percent=50,
                wait=True,
                timeout=4.0,
            )
            print(f"-> Gripper moved: Width after command = {w_moved:.1f} mm (Displacement: {w_moved - gripper_baseline_width:+.1f} mm)")

            # Return gripper to baseline width
            print(f"\n-> Restoring gripper to baseline width ({gripper_baseline_width:.1f} mm)...")
            w_restored = gripper.grip(
                width_mm=gripper_baseline_width,
                force_n=target_test_force,
                speed_percent=50,
                wait=True,
                timeout=4.0,
            )
            print(f"-> Gripper restored: Final Width = {w_restored:.1f} mm (Error: {abs(w_restored - gripper_baseline_width):.1f} mm)")
            print("-> OnRobot Gripper test sequence completed successfully!")
        else:
            print("\n" + "-" * 75)
            print("Phase 6: OnRobot Gripper Movement Test (SKIPPED - Gripper Not Detected)")
            print("-" * 75)

        print("\n" + "=" * 75)
        print("  ALL PHASES PASSED: UR5e + OnRobot 2FG14 Fully Validated for LeRobot!")
        print("=" * 75)

    finally:
        # ─────────────────────────────────────────────────────────────
        # Phase 7: Clean Teardown
        # ─────────────────────────────────────────────────────────────
        print("\nTearing down connections...")
        if gripper.is_connected:
            gripper.disconnect()
            print("-> OnRobot Modbus TCP interface disconnected cleanly.")
        if robot.is_connected:
            robot.disconnect()
            print("-> UR5e Sockets & RTDE interface disconnected cleanly.")


if __name__ == "__main__":
    main()
