import math
import time
import numpy as np
import cv2
import torch
from pathlib import Path

from lerobot.robots import UR5eRobotConfig, make_robot_from_config
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROBOT_IP = "192.168.1.102"
CAMERA_INDEX = 4       # USB Camera (KYE Systems Corp. eFace 1325R)
FPS = 30               # Target recording frequency
DATASET_REPO = "local/ur5e_freedrive_demos"
OUTPUT_DIR = Path("./datasets") / "ur5e_freedrive_demos"

UR5E_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


def main():
    print("=" * 70)
    print("      UR5e + USB Camera — Freedrive Dataset Recorder (LeRobot)")
    print("=" * 70)

    # 1. Configure robot with attached USB camera
    camera_cfg = OpenCVCameraConfig(
        index_or_path=CAMERA_INDEX,
        width=640,
        height=480,
        fps=FPS,
    )
    robot_cfg = UR5eRobotConfig(
        robot_ip=ROBOT_IP,
        cameras={"workspace": camera_cfg},
    )

    print(f"\n1. Connecting to UR5e ({ROBOT_IP}) and USB Camera (/dev/video{CAMERA_INDEX})...")
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    print("-> Robot and Camera successfully connected!\n")

    # 2. Define standard LeRobot Dataset features
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"{j}_pos" for j in UR5E_JOINTS],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"{j}_pos" for j in UR5E_JOINTS],
        },
        "observation.images.workspace": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO,
        fps=FPS,
        features=features,
        root=OUTPUT_DIR,
        robot_type="ur5e",
        use_videos=True,
    )
    print(f"Initialized LeRobot dataset at: {OUTPUT_DIR.resolve()}")

    try:
        episode_idx = 0
        while True:
            episode_idx += 1
            print("\n" + "=" * 70)
            print(f"Ready to record Episode #{episode_idx}")
            print("=" * 70)
            task_desc = input("Enter task description (or press Enter for 'Pick and place demo', or 'q' to finish): ").strip()
            if task_desc.lower() == "q":
                break
            if not task_desc:
                task_desc = "Pick and place demo"

            input("\n-> Press [ENTER] to start recording and enable FREEDRIVE...")

            print("\n>>> ENABLING FREEDRIVE — You can move the robot arm now! <<<")
            print(">>> Press [ENTER] in terminal or [q] in video window to STOP recording episode. <<<\n")
            robot.enable_freedrive()

            frame_count = 0
            t_start = time.perf_counter()
            dt = 1.0 / FPS

            recording = True
            while recording:
                t_cycle_start = time.perf_counter()

                obs = robot.get_observation()

                # Extract joint angles as float32 array
                q_vals = np.array(
                    [obs[f"{j}.pos"] for j in UR5E_JOINTS],
                    dtype=np.float32
                )
                img = obs["workspace"]  # (480, 640, 3) uint8 RGB

                # In freedrive demonstration, action is the current joint setpoint
                frame_data = {
                    "observation.state": torch.from_numpy(q_vals),
                    "action": torch.from_numpy(q_vals),
                    "observation.images.workspace": torch.from_numpy(img),
                }
                dataset.add_frame(frame_data)
                frame_count += 1

                # Live preview window with HUD overlay
                bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elapsed_s = time.perf_counter() - t_start
                cv2.putText(
                    bgr_img,
                    f"REC [Ep #{episode_idx}] - {elapsed_s:.1f}s | {frame_count} frames",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("LeRobot Freedrive Recorder (Press 'q' or 'SPACE' to stop)", bgr_img)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord(" "), 27):  # q, Space, or Esc
                    recording = False
                    break

                # Maintain steady 30 Hz recording loop
                t_elapsed = time.perf_counter() - t_cycle_start
                sleep_t = dt - t_elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            cv2.destroyAllWindows()

            print("\n>>> STOPPING FREEDRIVE — Re-locking joints in place... <<<")
            robot.disable_freedrive()

            print(f"Saving Episode #{episode_idx} ({frame_count} frames, {frame_count / FPS:.1f}s)...")
            dataset.save_episode(task=task_desc)
            print(f"-> Episode #{episode_idx} saved successfully!")

            cont = input("\nRecord another episode? [Y/n]: ").strip().lower()
            if cont in ("n", "no", "q"):
                break

        print("\nConsolidating dataset and encoding final MP4 videos...")
        dataset.consolidate()
        print(f"\n-> Dataset recording COMPLETE! Saved to {OUTPUT_DIR.resolve()}")
        print("\nTo visualize the recorded dataset in your browser:")
        print(f"    lerobot-dataset-viz --dataset.repo_id={DATASET_REPO} --dataset.root={OUTPUT_DIR}")

    finally:
        print("\nDisconnecting...")
        robot.disconnect()
        print("Disconnected cleanly.")


if __name__ == "__main__":
    main()

