import math
import time
from ur5e import UR5eRobotConfig, make_robot_from_config

ROBOT_IP = "192.168.1.102"
FREEDRIVE_DURATION_S = 20.0


def main():
    print("=" * 65)
    print("         UR5e LeRobot — Freedrive Hand-Guiding Test")
    print("=" * 65)

    config = UR5eRobotConfig(robot_ip=ROBOT_IP)
    robot = make_robot_from_config(config)

    print(f"\nConnecting to UR5e at {ROBOT_IP}...")
    robot.connect()
    print("Connected successfully!\n")

    try:
        obs_start = robot.get_observation()
        print("Starting Joint Angles:")
        for k, v in obs_start.items():
            if k.endswith(".pos"):
                print(f"  {k:20s}: {math.degrees(v):+7.2f}°")

        print("\n" + "=" * 65)
        print(">>> ENABLING FREEDRIVE MODE <<<")
        print("You can now grab and physically guide the UR5e arm by hand!")
        print(f"Freedrive will run for {FREEDRIVE_DURATION_S} seconds.")
        print("=" * 65 + "\n")

        robot.enable_freedrive()

        t_start = time.perf_counter()
        while time.perf_counter() - t_start < FREEDRIVE_DURATION_S:
            remaining = FREEDRIVE_DURATION_S - (time.perf_counter() - t_start)
            obs = robot.get_observation()
            q_deg = [math.degrees(obs[f"{j}.pos"]) for j in [
                "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"
            ]]
            deg_str = " | ".join(f"{d:+6.1f}°" for d in q_deg)
            print(f"\r[{remaining:4.1f}s remaining] Live Joints: {deg_str}", end="", flush=True)
            time.sleep(0.1)

        print("\n\n>>> DISABLING FREEDRIVE MODE <<<")
        print("Re-locking joints in current position...")
        robot.disable_freedrive()
        time.sleep(1.0)

        obs_end = robot.get_observation()
        print("\nFinal Position after Freedrive:")
        for k, v in obs_end.items():
            if k.endswith(".pos"):
                print(f"  {k:20s}: {math.degrees(v):+7.2f}°")

        print("\n-> Freedrive test completed successfully!")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user! Stopping freedrive safely...")
        robot.disable_freedrive()

    finally:
        print("\nDisconnecting...")
        robot.disconnect()
        print("Disconnected cleanly.")


if __name__ == "__main__":
    main()

