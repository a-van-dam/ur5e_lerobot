#!/usr/bin/env python3
"""
Shared Camera ROI (Region of Interest) Masking Utilities
========================================================
Provides resolution-independent normalized ROI rectangle handling ([x1, y1, x2, y2] in [0.0, 1.0])
so pixels outside the marked workspace/wrist rectangle are set to pure black (0, 0, 0)
identically across:
  - gui_recorder.py (interactive drawing + recording)
  - save_episode_worker.py (dataset metadata persistence)
  - train_robot.py (copying ROI metadata into trained checkpoints)
  - gui_tester.py & test_policy.py (automatic inference-time masking)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

MIN_ROI_SPAN = 0.03  # Minimum 3% width/height to avoid accidental tiny clicks


def normalize_roi(roi: Optional[Sequence[float]]) -> Optional[List[float]]:
    """
    Validates and clamps a normalized ROI rectangle [x1, y1, x2, y2] in [0.0, 1.0].
    Returns None if roi is None, invalid, or smaller than MIN_ROI_SPAN.
    """
    if roi is None:
        return None
    try:
        vals = [float(v) for v in roi]
    except Exception:
        return None
    if len(vals) != 4:
        return None

    x1, y1, x2, y2 = vals
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    if (x2 - x1) < MIN_ROI_SPAN or (y2 - y1) < MIN_ROI_SPAN:
        return None

    # Treat virtually full-frame selections as no mask
    if x1 <= 0.005 and y1 <= 0.005 and x2 >= 0.995 and y2 >= 0.995:
        return None

    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def apply_roi_mask(frame: np.ndarray, roi: Optional[Sequence[float]]) -> np.ndarray:
    """
    Masks pixels outside the normalized ROI rectangle [x1, y1, x2, y2] to pure black (0).
    Returns the original frame untouched if roi is None/invalid.
    Works for both (H, W, C) color frames and (H, W) grayscale frames.
    """
    norm = normalize_roi(roi)
    if norm is None or frame is None:
        return frame

    h, w = frame.shape[:2]
    x1_px = max(0, min(w, int(round(norm[0] * w))))
    y1_px = max(0, min(h, int(round(norm[1] * h))))
    x2_px = max(0, min(w, int(round(norm[2] * w))))
    y2_px = max(0, min(h, int(round(norm[3] * h))))

    if x2_px <= x1_px or y2_px <= y1_px:
        return frame

    masked = np.zeros_like(frame)
    masked[y1_px:y2_px, x1_px:x2_px] = frame[y1_px:y2_px, x1_px:x2_px]
    return masked


def load_camera_roi_from_dir(directory: str | Path) -> Dict[str, Optional[List[float]]]:
    """
    Reads {"workspace": roi_or_None, "wrist": roi_or_None} from a dataset directory
    or trained checkpoint directory (checks training_info.json and recording_config.json).
    """
    p = Path(directory)
    result: Dict[str, Optional[List[float]]] = {"workspace": None, "wrist": None}

    candidates = [
        p / "training_info.json",
        p.parent / "training_info.json",
        p / "meta" / "recording_config.json",
        p / "recording_config.json",
    ]
    for cand in candidates:
        if cand.exists():
            try:
                data = json.loads(cand.read_text())
                roi_dict = data.get("camera_roi")
                if isinstance(roi_dict, dict):
                    result["workspace"] = normalize_roi(roi_dict.get("workspace"))
                    result["wrist"] = normalize_roi(roi_dict.get("wrist"))
                    return result
            except Exception:
                pass
    return result


def format_roi_label(roi: Optional[Sequence[float]]) -> str:
    norm = normalize_roi(roi)
    if norm is None:
        return "Full Frame"
    w_pct = int(round((norm[2] - norm[0]) * 100))
    h_pct = int(round((norm[3] - norm[1]) * 100))
    return f"ROI {w_pct}%×{h_pct}%"
