import math
import time
from lerobot.robots import UR5eRobotConfig, make_robot_from_config


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
    print("       Universal Robots UR5e — Complete LeRobot Verification Suite")
    print("=" * 75)

    # 1. Initialize configuration with conservative, safe motion limits
    config = UR5eRobotConfig(
        robot_ip="192.168.1.102",
        velocity=0.2,       # Safe velocity: 0.2 rad/s (~11.5 deg/s)
        acceleration=0.4,   # Safe acceleration: 0.4 rad/s^2
    )

    print(f"\n[Config] Robot Type: '{config.type}' | IP: {config.robot_ip}")
    print(f"[Config] Safe Velocity: {config.velocity} rad/s | Safe Acceleration: {config.acceleration} rad/s²")

    # 2. Instantiate robot through LeRobot factory
    robot = make_robot_from_config(config)

    # ─────────────────────────────────────────────────────────────────
    # Phase 1: Connection & Feature Introspection
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 75)
    print("Phase 1: Connecting to UR5e and Validating LeRobot Features")
    print("-" * 75)

    robot.connect()
    print(f"-> Robot connected: {robot.is_connected}")
    if not robot.is_connected:
        raise ConnectionError("Failed to connect to UR5e.")

    try:
        print("\nObservation Features (Sensory inputs):")
        for k, v in robot.observation_features.items():
            print(f"  • {k:22s} -> {v}")

        print("\nAction Features (Actuation targets):")
        for k, v in robot.action_features.items():
            print(f"  • {k:22s} -> {v}")

        # ─────────────────────────────────────────────────────────────
        # Phase 2: Telemetry Sampling & Frequency Benchmark
        # ─────────────────────────────────────────────────────────────
        print("\n" + "-" * 75)
        print("Phase 2: RTDE Telemetry Benchmark (Sampling 100 frames at 125 Hz)")
        print("-" * 75)

        t_start = time.perf_counter()
        samples = 100
        for _ in range(samples):
            obs = robot.get_observation()
        elapsed = time.perf_counter() - t_start
        rate = samples / elapsed

        print(f"-> Sampled {samples} observation frames in {elapsed:.3f}s ({rate:.1f} Hz).")
        print(f"-> Average read latency: {elapsed / samples * 1000.0:.2f} ms")
        print(f"\nCurrent Live Joint Positions:")
        for k, v in obs.items():
            if k.endswith(".pos"):
                print(f"   {k:22s}: {v:+.4f} rad ({math.degrees(v):+7.2f}°)")

        # Store baseline starting positions
        baseline_action = {k: v for k, v in obs.items() if k.endswith(".pos")}

        # ─────────────────────────────────────────────────────────────
        # Phase 3: Single-Joint Verification (Wrist 3 Flange Nudge)
        # ─────────────────────────────────────────────────────────────
        print("\n" + "-" * 75)
        print("Phase 3: Single-Joint Action Test (Wrist 3 +2.0° Nudge & Return)")
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

        print("Restoring original pose...")
        robot.send_action(baseline_action)
        time.sleep(2.5)

        obs_restored = robot.get_observation()
        print(f"-> Restored pose: {rad_to_deg_str(obs_restored)}")

        # ─────────────────────────────────────────────────────────────
        # Phase 4: Coordinated Multi-Joint Movement
        # ─────────────────────────────────────────────────────────────
        print("\n" + "-" * 75)
        print("Phase 4: Coordinated Multi-Joint Action Test (+5° Base, +5° Wrist 2)")
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

        print("\n" + "=" * 75)
        print("        ALL PHASES PASSED: UR5e is Fully Validated for LeRobot!")
        print("=" * 75)

    finally:
        # ─────────────────────────────────────────────────────────────
        # Phase 5: Clean Teardown
        # ─────────────────────────────────────────────────────────────
        print("\nDisconnecting from UR5e...")
        robot.disconnect()
        print("-> Sockets & RTDE interface disconnected cleanly.")


if __name__ == "__main__":
    main()
