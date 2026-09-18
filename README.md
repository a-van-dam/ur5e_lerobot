# UR5e Integration for Hugging Face LeRobot

Hardware wrapper and demonstration collection toolkit for controlling a **Universal Robots UR5e** arm, recording synchronized multimodal datasets (6-DoF joint telemetry + USB camera feeds), and training imitation learning policies using [Hugging Face LeRobot](https://github.com/huggingface/lerobot).

---

## Architecture Overview

```
       ┌──────────────────────── LeRobot Framework ────────────────────────┐
       │                                                                   │
       │   lerobot-record / lerobot-dataset-viz / Policy Rollout           │
       │                                │                                  │
       │                     make_robot_from_config()                      │
       │                                │                                  │
       │                     ┌──────────▼──────────┐                       │
       │                     │      UR5eRobot      │                       │
       │                     └────┬───────────┬────┘                       │
       └──────────────────────────┼───────────┼────────────────────────────┘
                                  │           │
                     125 Hz State │           │ Motion & Freedrive
                 (Port 30004 RTDE)│           │ (Port 30002 Socket)
                                  ▼           ▼
                      ┌───────────────────────────┐
                      │  Universal Robots UR5e   │
                      │  (PolyScope 192.168.1.x)  │
                      └───────────────────────────┘
```

* **Telemetry (Port 30004)**: High-frequency (125 Hz) joint state streaming (`q`, `qd`, TCP pose) via `rtde_receive.RTDEReceiveInterface`.
* **Motion & Freedrive (Port 30002)**: Direct URScript socket interface for point-to-point positioning (`movej`) and physical hand-guiding (`freedrive_mode`).
* **Visual Feeds**: Integrated with LeRobot's unified camera subsystem (`OpenCVCameraConfig`) for 30 FPS video synchronization.

---

## Repository Structure

```text
ur5e_lerobot/
├── ur5e/                          # LeRobot hardware wrapper package
│   ├── __init__.py
│   ├── configuration_ur5e.py      # UR5eRobotConfig (Draccus-compatible)
│   └── robot_ur5e.py              # UR5eRobot implementation
├── gui_recorder.py                # Standalone PyQt6 Teleoperation & Dataset Studio
├── test_lerobot_ur5e.py           # 5-phase comprehensive diagnostic suite
├── test_freedrive.py              # 10-second physical hand-guiding test
├── test_webcam.py                 # Live USB camera preview (/dev/video4)
├── record_freedrive_dataset.py    # Demonstration recorder (LeRobot v2.0 format)
├── .gitignore
└── README.md
```

---

## Prerequisites

1. **Conda Environment**:
   ```bash
   conda activate lerobot
   ```

2. **Dependencies**:
   * `PyQt6`
   * `opencv-python`
   * `ur_rtde` / `rtde_receive`
   * `lerobot` (Hugging Face)
   * `torch`

3. **Robot Network Configuration**:
   * Default Robot IP: `192.168.1.102`
   * Primary/Secondary Port: `30002`
   * RTDE Port: `30004`
   * Ensure host PC has a static IP on the same subnet (e.g. `192.168.1.100`, mask `255.255.255.0`).

---

## Quickstart

### 0. Launch the Teleoperation & Dataset Studio (Desktop GUI)
Launch the full multithreaded PyQt6 studio for teleoperation and episodic dataset recording:
```bash
python gui_recorder.py
```
* **Simulation / Mock Mode**: If you are testing the GUI without the physical UR5e arm or USB camera connected, run:
  ```bash
  python gui_recorder.py --mock
  ```
* **Hotkeys**:
  - **`Space`**: Start / End demonstration episode. (Must be in Freedrive mode to start!)
  - **`c`** or **`Left Arrow`**: Cancel and discard the active recording without saving.
  - **`Esc`**: Stop session and safely lock robot joints.
* **Hardware Diagnostics**: Click **"Run Full Hardware Test"** to execute `test_lerobot_ur5e.py` asynchronously inside the GUI console.

### 1. Test Robot Telemetry & Bidirectional Motion (CLI)
* **Phase 1**: Connection & feature introspection (`observation_features` & `action_features`).
* **Phase 2**: 125 Hz RTDE telemetry benchmark (100 frame sampling & latency check).
* **Phase 3**: Single-joint nudge & return (+2° on Wrist 3).
* **Phase 4**: Coordinated multi-joint displacement (+5° on Base, +5° on Wrist 2).
* **Phase 5**: Safe teardown.

### 2. Verify USB Camera Feed
Check camera framing and lighting:
```bash
python test_webcam.py
```
* Opens live feed from `/dev/video4`. Press `q` to quit.

### 3. Test Freedrive Hand-Guiding
Test physical hand-teaching before recording:
```bash
python test_freedrive.py
```
* Enables freedrive mode for 10 seconds. You can grab and move the arm freely.
* Prints real-time joint positions. Re-locks smoothly upon exit.

### 4. Record Demonstrations with Camera Sync
Record demonstration trajectories for policy training:
```bash
python record_freedrive_dataset.py
```
* Press **Enter** to start recording an episode and engage freedrive.
* Perform the task by hand-guiding the arm.
* Press **`q`** or **Space** in the camera window to stop the episode.
* Automatically encodes H.264 MP4 video and Parquet joint metadata into `./datasets/ur5e_freedrive_demos/`.

### 5. Inspect in LeRobot Web GUI
Visualize recorded trajectories, video feeds, and joint plots:
```bash
conda activate lerobot
lerobot-dataset-viz --repo-id local/ur5e_freedrive_demos --root ./datasets/ur5e_freedrive_demos --episode-index 0
```
> **Note**: Visualizing with Rerun requires `rerun-sdk` (`pip install rerun-sdk`).

