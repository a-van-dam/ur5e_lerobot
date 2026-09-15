import cv2

# USB camera is /dev/video4 (KYE SYSTEMS eFace 1325R)
# Laptop integrated webcam is /dev/video0 — we skip that one
CAMERA_INDEX = 4

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"ERROR: Could not open camera at index {CAMERA_INDEX}")
    exit(1)

print(f"Camera /dev/video{CAMERA_INDEX} opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("USB Camera Test (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")

