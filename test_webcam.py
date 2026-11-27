"""
Quick webcam verification script.

Automatically detects all connected Linux V4L2 capture cameras (filtering out
metadata-only nodes) and opens them simultaneously using V4L2 + MJPG.

Uses PyQt6 for display when available (so it works seamlessly with
opencv-python-headless), falling back to OpenCV HighGUI otherwise.

Usage:
    python test_webcam.py              # Auto-detect and preview all connected cameras
    python test_webcam.py 0            # Preview only /dev/video0
    python test_webcam.py 0 2          # Preview /dev/video0 and /dev/video2
"""

import sys
import time
import cv2
from camera_utils import detect_available_cameras, open_video_capture

try:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
    from PyQt6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QVBoxLayout,
        QWidget,
    )
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False


def run_qt_preview(caps):
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("UR5e Camera Test — Press 'q' or Esc to quit")
    window.setStyleSheet("background-color: #0f172a; color: #f1f5f9;")

    central = QWidget()
    row = QHBoxLayout(central)
    row.setContentsMargins(10, 10, 10, 10)
    row.setSpacing(10)

    panels = []
    for idx, label, cap in caps:
        col = QVBoxLayout()
        title = QLabel(label)
        title.setStyleSheet("font-weight: bold; font-size: 13px; color: #38bdf8;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        view = QLabel("Starting stream...")
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        view.setMinimumSize(480, 270)
        view.setStyleSheet("background-color: #020617; border: 1px solid #334155;")

        col.addWidget(title)
        col.addWidget(view, stretch=1)
        row.addLayout(col, stretch=1)
        panels.append({
            "idx": idx,
            "label": label,
            "cap": cap,
            "title": title,
            "view": view,
            "frames": 0,
            "t0": time.perf_counter(),
        })

    window.setCentralWidget(central)
    window.resize(max(640, 560 * len(panels)), 420)

    for key in ("q", "Q", "Escape"):
        sc = QShortcut(QKeySequence(key), window)
        sc.activated.connect(window.close)

    def update_frames():
        now = time.perf_counter()
        for p in panels:
            ret, frame = p["cap"].read()
            if not ret or frame is None:
                continue
            p["frames"] += 1
            elapsed = now - p["t0"]
            if elapsed >= 1.0:
                fps = p["frames"] / elapsed
                h, w = frame.shape[:2]
                p["title"].setText(f"{p['label']}  |  {w}x{h} @ {fps:.1f} FPS")
                p["frames"] = 0
                p["t0"] = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg).scaled(
                p["view"].size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            p["view"].setPixmap(pix)

    timer = QTimer(window)
    timer.timeout.connect(update_frames)
    timer.start(30)

    window.show()
    try:
        app.exec()
    finally:
        timer.stop()


def run_cv2_preview(caps):
    try:
        while True:
            for _, label, cap in caps:
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
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


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

    print("Press 'q' or Esc in the preview window to quit.")

    try:
        if PYQT6_AVAILABLE:
            run_qt_preview(caps)
        else:
            run_cv2_preview(caps)
    finally:
        for _, _, cap in caps:
            cap.release()
        print("Cameras closed.")


if __name__ == "__main__":
    main()
