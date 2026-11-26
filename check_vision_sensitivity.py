#!/usr/bin/env python3
"""
UR5e LeRobot Offline Vision Sensitivity Diagnostic
==================================================
Tests whether a trained ACT policy genuinely uses the camera images to adapt
to shifted object positions (rather than memorizing a proprioception-only path).

Runs 100% OFFLINE without a physical robot connected:
  1. Loads the checkpoint and its training dataset.
  2. Finds episodes with the most extreme LEFT vs RIGHT and NEAR vs FAR grasp poses.
  3. Holds observation.state IDENTICAL and swaps:
       - Both cameras (LEFT vs RIGHT, NEAR vs FAR)
       - Workspace camera only
       - Wrist camera only
       - Real images vs All-Black zero frames
  4. Reports per-camera joint prediction sensitivity (in degrees) and a clear
     PASS / LOW / BLIND verdict.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import lerobot.utils.import_utils
lerobot.utils.import_utils.get_safe_default_video_backend = lambda: "pyav"

from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy


def resolve_dataset_path(ckpt_dir: Path, explicit_dataset: str | None) -> Path:
    if explicit_dataset:
        return Path(explicit_dataset).resolve()

    repo_root = Path(__file__).resolve().parent
    for info_p in [ckpt_dir / "training_info.json", ckpt_dir.parent / "training_info.json"]:
        if info_p.exists():
            try:
                ds_name = json.loads(info_p.read_text()).get("dataset")
                if ds_name:
                    cand = repo_root / "datasets" / ds_name
                    if cand.exists():
                        return cand.resolve()
            except Exception:
                pass
    raise FileNotFoundError(
        f"Could not auto-locate dataset for checkpoint '{ckpt_dir}'. Pass --dataset <path>."
    )


def run_vision_sensitivity_check(checkpoint_path: str | Path, dataset_path: str | None = None) -> int:
    ckpt_dir = Path(checkpoint_path).resolve()
    if (ckpt_dir / "best_checkpoint" / "config.json").exists():
        ckpt_dir = ckpt_dir / "best_checkpoint"
    elif (ckpt_dir / "checkpoint" / "config.json").exists():
        ckpt_dir = ckpt_dir / "checkpoint"

    if not (ckpt_dir / "config.json").exists():
        print(f"[ERROR] No config.json found in checkpoint: {ckpt_dir}", file=sys.stderr)
        return 1

    ds_dir = resolve_dataset_path(ckpt_dir, dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("        UR5e OFFLINE VISION SENSITIVITY DIAGNOSTIC")
    print("=" * 72)
    print(f"  • Checkpoint: {ckpt_dir}")
    print(f"  • Dataset:    {ds_dir}")
    print(f"  • Device:     {device}")

    policy = ACTPolicy.from_pretrained(ckpt_dir).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=ckpt_dir)

    img_keys = [k for k in (getattr(policy.config, "input_features", {}) or {}) if "observation.images" in k]
    has_ws = "observation.images.workspace" in img_keys
    has_wr = "observation.images.wrist" in img_keys
    print(f"  • Cameras:    {', '.join(img_keys) if img_keys else 'None'}")

    pq_files = sorted(glob.glob(str(ds_dir / "data" / "*" / "*.parquet")))
    if not pq_files:
        print(f"[ERROR] No parquet files found in {ds_dir}", file=sys.stderr)
        return 1

    tbl = pa.concat_tables([pq.read_table(f) for f in pq_files])
    ep_idx = tbl["episode_index"].to_numpy()
    obs = np.stack(tbl["observation.state"].to_pylist())
    act = np.stack(tbl["action"].to_pylist())
    has_gripper = act.shape[1] >= 7

    # Locate grasp frame (or midpoint if no gripper) per episode
    grasps = []
    for ep in np.unique(ep_idx):
        g_indices = np.where(ep_idx == ep)[0]
        if len(g_indices) < 20:
            continue
        o = obs[g_indices]
        a = act[g_indices]
        if has_gripper:
            ci = np.where(a[:, 6] < 80.0)[0]
            if len(ci) == 0:
                continue  # Skip episodes with no grasp/close event
            ref_t = int(ci[0])
        else:
            ref_t = len(g_indices) // 2
        grasps.append((int(ep), int(g_indices[0]), int(g_indices[ref_t]), ref_t, len(g_indices), o[ref_t, :6]))

    if len(grasps) < 2:
        print("[WARNING] Need at least 2 episodes to compare shifted object positions.")
        return 0

    grasps_pan = sorted(grasps, key=lambda x: x[5][0])
    ep_left = grasps_pan[0]
    ep_mid = grasps_pan[len(grasps_pan) // 2]
    ep_right = grasps_pan[-1]

    grasps_elb = sorted(grasps, key=lambda x: x[5][2])
    ep_near = grasps_elb[0]
    ep_far = grasps_elb[-1]

    gt_lr_max = float(np.degrees(np.abs(ep_left[5] - ep_right[5])).max())
    gt_nf_max = float(np.degrees(np.abs(ep_near[5] - ep_far[5])).max())
    print(f"  • Dataset Grasp Spread: LEFT(ep{ep_left[0]}) vs RIGHT(ep{ep_right[0]}) max={gt_lr_max:.1f}° | NEAR(ep{ep_near[0]}) vs FAR(ep{ep_far[0]}) max={gt_nf_max:.1f}°")
    print("=" * 72)

    ds = LeRobotDataset(f"local/{ds_dir.name}", root=ds_dir, video_backend="pyav")

    def predict_chunk(state_t: torch.Tensor, imgs: dict[str, torch.Tensor]) -> np.ndarray:
        batch = {"observation.state": state_t.clone()}
        for k, v in imgs.items():
            batch[k] = v.unsqueeze(0).to(device)
        with torch.no_grad():
            policy.reset()
            pb = preprocessor(batch)
            raw = policy.predict_action_chunk(pb)
            return postprocessor(raw).squeeze(0).cpu().numpy()

    min_grasp = min(g[3] for g in [ep_left, ep_mid, ep_right, ep_near, ep_far])
    test_offsets = [
        ("Start (t=0)", 0),
        ("Mid-Approach", max(1, min_grasp // 2)),
        ("Pre-Grasp", max(1, min_grasp - 15)),
    ]

    best_lr_resp = 0.0
    best_nf_resp = 0.0
    best_ws_resp = 0.0
    best_wr_resp = 0.0

    for label, off in test_offsets:
        ref_state = ds[ep_mid[1] + min(off, ep_mid[4] - 1)]["observation.state"].unsqueeze(0).to(device)

        def get_imgs(ep_info):
            s = ds[ep_info[1] + min(off, ep_info[4] - 1)]
            return {k: s[k] for k in img_keys if k in s}

        imgs_L = get_imgs(ep_left)
        imgs_M = get_imgs(ep_mid)
        imgs_R = get_imgs(ep_right)
        imgs_N = get_imgs(ep_near)
        imgs_F = get_imgs(ep_far)
        imgs_BLK = {k: torch.zeros_like(v) for k, v in imgs_M.items()}

        c_L = predict_chunk(ref_state, imgs_L)
        c_M = predict_chunk(ref_state, imgs_M)
        c_R = predict_chunk(ref_state, imgs_R)
        c_N = predict_chunk(ref_state, imgs_N)
        c_F = predict_chunk(ref_state, imgs_F)
        c_BLK = predict_chunk(ref_state, imgs_BLK)

        lr_deg = np.degrees(np.abs(c_L[:, :6] - c_R[:, :6])).max(axis=0)
        nf_deg = np.degrees(np.abs(c_N[:, :6] - c_F[:, :6])).max(axis=0)
        blk_deg = np.degrees(np.abs(c_M[:, :6] - c_BLK[:, :6])).max(axis=0)

        best_lr_resp = max(best_lr_resp, float(lr_deg.max()))
        best_nf_resp = max(best_nf_resp, float(nf_deg.max()))

        print(f"\n[{label} | frame offset={off}] (holding observation.state CONSTANT)")
        print(f"  LEFT vs RIGHT max joint shift (deg): {np.round(lr_deg, 2)}  (max={lr_deg.max():.2f}°)")
        print(f"  NEAR vs FAR   max joint shift (deg): {np.round(nf_deg, 2)}  (max={nf_deg.max():.2f}°)")
        print(f"  REAL vs BLACK max joint shift (deg): {np.round(blk_deg, 2)}  (max={blk_deg.max():.2f}°)")

        if has_ws and has_wr:
            c_ws_swap = predict_chunk(
                ref_state,
                {"observation.images.workspace": imgs_L["observation.images.workspace"], "observation.images.wrist": imgs_R["observation.images.wrist"]},
            )
            ws_only = float(np.degrees(np.abs(c_ws_swap[:, :6] - c_R[:, :6])).max())
            wr_only = float(np.degrees(np.abs(c_L[:, :6] - c_ws_swap[:, :6])).max())
            best_ws_resp = max(best_ws_resp, ws_only)
            best_wr_resp = max(best_wr_resp, wr_only)
            print(f"  Per-Camera Contribution (L vs R):    Workspace={ws_only:.2f}° | Wrist={wr_only:.2f}°")

    peak_resp = max(best_lr_resp, best_nf_resp)
    print("\n" + "=" * 72)
    if peak_resp >= 5.0:
        verdict = f"PASS ✔ — Strong visual adaptation detected (peak response {peak_resp:.1f}° to shifted cube images)."
    elif peak_resp >= 2.0:
        verdict = f"MODERATE ⚠ — Partial visual adaptation ({peak_resp:.1f}°). Consider cleaner ROI / stronger state noise."
    else:
        verdict = f"LOW / BLIND ✖ — Weak visual response ({peak_resp:.1f}°). Policy is mostly relying on observation.state!"
    print(f"  VERDICT: {verdict}")
    if has_ws and has_wr:
        print(f"  Camera Breakdown: Peak Workspace={best_ws_resp:.2f}° | Peak Wrist={best_wr_resp:.2f}°")
    print("=" * 72)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Offline Vision Sensitivity Check for Trained ACT Policies")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model or checkpoint directory")
    parser.add_argument("--dataset", type=str, default=None, help="Optional explicit path to dataset folder")
    args = parser.parse_args()
    sys.exit(run_vision_sensitivity_check(args.checkpoint, args.dataset))


if __name__ == "__main__":
    main()
