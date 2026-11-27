"""
Shared Linux V4L2 camera discovery, default-index resolution, and OpenCV capture helpers.

Ensures GUI tools and test scripts automatically adapt when moving between computers
where connected USB cameras receive different /dev/videoN indices (e.g. /dev/video0 and
/dev/video2 instead of /dev/video4 and /dev/video6), and configures V4L2 + MJPG so
MJPG-only and dual USB cameras open reliably.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import cv2
except ImportError:
    cv2 = None


# Linux V4L2 ioctl constants for VIDIOC_QUERYCAP
_VIDIOC_QUERYCAP = 0x80685600
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_DEVICE_CAPS = 0x80000000


def is_v4l2_capture_device(dev_path: str) -> Optional[bool]:
    """
    Returns True if dev_path supports V4L2 video capture, False if it is a metadata/non-capture
    node, or None if the ioctl query cannot be performed (non-Linux or permission error).
    """
    if fcntl is None or not sys.platform.startswith("linux"):
        return None

    fd = -1
    try:
        try:
            fd = os.open(dev_path, os.O_RDWR | os.O_NONBLOCK)
        except PermissionError:
            fd = os.open(dev_path, os.O_RDONLY | os.O_NONBLOCK)
        buf = bytearray(104)
        fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
        caps, dev_caps = struct.unpack_from("<II", buf, 84)
        effective_caps = dev_caps if (caps & _V4L2_CAP_DEVICE_CAPS) else caps
        return bool(effective_caps & _V4L2_CAP_VIDEO_CAPTURE)
    except OSError:
        return None
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def detect_available_cameras() -> List[Tuple[int, str]]:
    """
    Scans Linux v4l2 device nodes and returns a list of (camera_index, user_friendly_label).
    Filters out metadata-only sub-nodes using V4L2 VIDIOC_QUERYCAP (with sysfs 'index' fallback).
    """
    cameras: List[Tuple[int, str]] = []
    v4l_dir = Path("/sys/class/video4linux")
    if v4l_dir.exists():
        devs = sorted(
            v4l_dir.glob("video*"),
            key=lambda p: int(p.name.replace("video", "")) if p.name.replace("video", "").isdigit() else 999,
        )
        for dev in devs:
            suffix = dev.name.replace("video", "")
            if not suffix.isdigit():
                continue
            idx = int(suffix)
            dev_path = f"/dev/video{idx}"

            cap_ok = is_v4l2_capture_device(dev_path)
            if cap_ok is False:
                continue
            if cap_ok is None:
                idx_file = dev / "index"
                if idx_file.exists():
                    try:
                        if idx_file.read_text().strip() != "0":
                            continue
                    except Exception:
                        pass

            name_file = dev / "name"
            dev_name = name_file.read_text().strip() if name_file.exists() else f"Camera {idx}"
            clean_name = dev_name.split(":")[0].strip() or f"Camera {idx}"
            cameras.append((idx, f"/dev/video{idx}: {clean_name}"))

    if not cameras:
        cameras = [(0, "/dev/video0: Default Camera")]
    return cameras


def resolve_camera_indices(
    available_cameras: List[Tuple[int, str]],
    requested_cam1: Optional[int] = None,
    requested_cam2: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Resolves primary and secondary camera indices against actually connected devices.

    - If requested_cam1 exists in available_cameras, keeps it; otherwise picks the first detected camera.
    - If requested_cam2 exists in available_cameras and differs from cam1, keeps it; otherwise picks
      the first detected camera different from cam1.
    """
    avail_ids = [idx for idx, _ in available_cameras]
    if not avail_ids:
        cam1 = requested_cam1 if requested_cam1 is not None else 0
        cam2 = requested_cam2 if (requested_cam2 is not None and requested_cam2 != cam1) else (cam1 + 2)
        return cam1, cam2

    if requested_cam1 is not None and requested_cam1 in avail_ids:
        cam1 = requested_cam1
    else:
        cam1 = avail_ids[0]

    other_ids = [idx for idx in avail_ids if idx != cam1]
    if requested_cam2 is not None and requested_cam2 in other_ids:
        cam2 = requested_cam2
    elif other_ids:
        cam2 = other_ids[0]
    elif requested_cam2 is not None:
        cam2 = requested_cam2
    else:
        cam2 = cam1

    return cam1, cam2


def open_video_capture(
    camera_id: int,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
):
    """
    Opens an OpenCV VideoCapture using V4L2 on Linux and requests MJPG pixel format.
    MJPG is required by MJPG-only USB cameras (e.g. DJI Osmo Action 4, UVC webcams)
    and avoids USB bandwidth saturation when running two cameras simultaneously.
    YUYV-only cameras ignore the MJPG request and keep their native format.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not installed.")

    cap = None
    if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    if cap is None or not cap.isOpened():
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        cap = cv2.VideoCapture(camera_id)

    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
        except Exception:
            pass

    return cap
