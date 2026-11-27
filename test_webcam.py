"""
Quick webcam verification script.

Automatically detects all connected Linux V4L2 capture cameras (filtering out
metadata-only nodes) and opens them simultaneously using V4L2 + MJPG.

Usage:
    python test_webcam.py              # Auto-detect and preview all connected cameras
    python test_webcam.py 0            # Preview only /dev/video0
    python test_webcam.py 0 2          # Preview /dev/video0 and /dev/video2
"""

import sys
import cv2
from camera_utils import detect_available_cameras, open_video_capture


def main():
    detected = detect_available_cameras()
    print("Detected video capture cameras:")
    for idx, label in detected:
        print(f"  - {label}")

    if len(sys.argv) > 1:
        indices = [int(arg) for arg in sys.argv[1:]]
        targets = [(i, f"/dev/video{i}") for i in indices]
    else:
        targets = detected

    caps = []
    for idx, label in targets:
        cap = open_video_capture(idx, width=640, height=480, fps=30)
        if not cap.isOpened():
            print(f"ERROR: Could not open camera at index {idx} ({label})")
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Opened {label} ({w}x{h}).")
        caps.append((idx, label, cap))

    if not caps:
        print("No cameras could be opened.")
        sys.exit(1)

    print("Press 'q' in any preview window to quit.")

    try:
        while True:
            for idx, label, cap in caps:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"Failed to grab frame from {label}.")
                    continue
                if frame.shape[1] > 960:
                    scale = 960.0 / frame.shape[1]
                    frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)))
                cv2.imshow(f"Camera Test - {label} (press q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for _, _, cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        print("Cameras closed.")


if __name__ == "__main__":
    main()
