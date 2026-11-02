#!/usr/bin/env python3
"""
UR5e LeRobot Isolated Episode Save Worker.

Runs in a clean, isolated Python subprocess separate from the GUI application.
This guarantees that:
1. Video encoding (PyAV / SVT-AV1 / H.264) and dataset serialization run in their own memory space.
2. Even if an unexpected segmentation fault or fatal C-level error occurs, only this worker process
   terminates; the main GUI application remains completely unharmed, responsive, and open.
3. Encoding is executed deterministically with parallel_encoding=False to avoid unsafe ProcessPool
   forking within a multi-threaded application.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
import traceback

import numpy as np

UR5E_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


def save_episode_payload(payload_path: str | Path) -> None:
    payload_file = Path(payload_path).resolve()
    if not payload_file.exists():
        raise FileNotFoundError(f"Payload file not found: {payload_file}")

    with open(payload_file, "rb") as f:
        payload = pickle.load(f)

    dataset_root = Path(payload["dataset_root"]).resolve()
    repo_id = payload.get("repo_id", f"local/{dataset_root.name}")
    fps = int(payload.get("fps", 30))
    camera_width = int(payload.get("camera_width", 320))
    camera_height = int(payload.get("camera_height", 240))
    enable_dual_cameras = bool(payload.get("enable_dual_cameras", False))
    is_grayscale = bool(payload.get("is_grayscale", False))
    ep_idx = int(payload.get("ep_idx", 0))
    num_frames = int(payload.get("num_frames", 0))
    duration_s = float(payload.get("duration_s", 0.0))
    task_str = payload.get("task_str", "Pick and place demonstration")
    frames = payload.get("frames", [])

    print(f"[WORKER] Commencing isolated save for Episode #{ep_idx:04d} ({num_frames} frames, {duration_s:.1f}s)...")
    print(f"         Target Root: {dataset_root}")
    print(f"         Dual Cameras: {enable_dual_cameras} | Grayscale: {is_grayscale} | Resolution: {camera_width}x{camera_height}")

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta_info_file = dataset_root / "meta" / "info.json"

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
            "shape": (camera_height, camera_width, 3),
            "names": ["height", "width", "channels"],
        },
    }
    if enable_dual_cameras:
        features["observation.images.wrist"] = {
            "dtype": "video",
            "shape": (camera_height, camera_width, 3),
            "names": ["height", "width", "channels"],
        }

    is_valid_lerobot = False
    if meta_info_file.exists():
        try:
            with open(meta_info_file, "r") as f:
                meta = json.load(f)
            if "features" in meta and "codebase_version" in meta:
                is_valid_lerobot = True
        except Exception:
            is_valid_lerobot = False

    if is_valid_lerobot:
        print(f"[WORKER] Resuming existing LeRobot dataset at: {dataset_root}")
        dataset = LeRobotDataset.resume(repo_id=repo_id, root=dataset_root)
    else:
        if dataset_root.exists():
            print(f"[WORKER] Cleaning non-LeRobot directory for fresh initialization: {dataset_root}")
            shutil.rmtree(dataset_root)
        print(f"[WORKER] Creating fresh LeRobot dataset at: {dataset_root}")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=dataset_root,
            robot_type="ur5e",
            use_videos=True,
        )

    # Ingest frames into dataset buffer
    for idx, frame in enumerate(frames):
        t_frame = {
            "observation.state": torch.from_numpy(np.ascontiguousarray(frame["observation.state"], dtype=np.float32)),
            "action": torch.from_numpy(np.ascontiguousarray(frame["action"], dtype=np.float32)),
            "observation.images.workspace": torch.from_numpy(
                np.ascontiguousarray(frame["observation.images.workspace"], dtype=np.uint8)
            ),
            "task": str(frame.get("task", task_str)),
        }
        if enable_dual_cameras and "observation.images.wrist" in frame:
            t_frame["observation.images.wrist"] = torch.from_numpy(
                np.ascontiguousarray(frame["observation.images.wrist"], dtype=np.uint8)
            )
        dataset.add_frame(t_frame)

    # Save episode with sequential encoding to eliminate multi-process fork crashes
    print("[WORKER] Encoding episode video frames and persisting Parquet records...")
    dataset.save_episode(parallel_encoding=False)

    # Finalize dataset metadata
    print("[WORKER] Finalizing dataset metadata...")
    dataset.finalize()

    # Update metadata configs
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    rec_cfg = {
        "grayscale": is_grayscale,
        "channels": 1 if is_grayscale else 3,
        "width": camera_width,
        "height": camera_height,
        "fps": fps,
        "dual_cam": enable_dual_cameras,
        "updated_at": datetime.datetime.now().isoformat(),
    }
    with open(meta_dir / "recording_config.json", "w") as f_cfg:
        json.dump(rec_cfg, f_cfg, indent=2)
    with open(dataset_root / "recording_config.json", "w") as f_cfg:
        json.dump(rec_cfg, f_cfg, indent=2)

    total_episodes = dataset.num_episodes
    print(f"[WORKER] SUCCESS: Episode #{ep_idx:04d} committed. Total dataset episodes: {total_episodes}.")

    # Clean up temporary payload file on confirmed success
    try:
        if payload_file.exists():
            payload_file.unlink()
    except Exception as cleanup_err:
        print(f"[WORKER] [NOTE] Could not delete temp payload file: {cleanup_err}", file=sys.stderr)

    result_json = json.dumps({
        "status": "success",
        "ep_idx": ep_idx,
        "num_frames": num_frames,
        "duration_s": duration_s,
        "total_episodes": total_episodes,
    })
    print(f"__WORKER_RESULT_JSON_START__{result_json}__WORKER_RESULT_JSON_END__")


def main():
    parser = argparse.ArgumentParser(description="UR5e LeRobot Isolated Episode Save Worker")
    parser.add_argument("--payload", type=str, required=True, help="Path to pickled episode payload file")
    args = parser.parse_args()

    try:
        save_episode_payload(args.payload)
        sys.exit(0)
    except Exception as exc:
        print(f"[WORKER ERROR] Fatal error while saving episode: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

