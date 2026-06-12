# ============================================================
# V5-B YOLO Debug Viewer
#
# Run this outside Isaac Sim, e.g. in VSCode terminal:
#   python3 v5b_yolo_viewer.py
#
# It reads:
#   ~/yolo_debug_frames/latest.jpg
#   ~/yolo_debug_frames/status.txt
#
# and displays the latest frame using OpenCV.
# ============================================================

import os
import time
import cv2

DEBUG_DIR = os.path.expanduser("~/yolo_debug_frames")
LATEST_IMAGE_PATH = os.path.join(DEBUG_DIR, "latest.jpg")
STATUS_PATH = os.path.join(DEBUG_DIR, "status.txt")

WINDOW_NAME = "V5-B Isaac Sim YOLO Viewer"

print("[viewer] Waiting for:", LATEST_IMAGE_PATH)
print("[viewer] Press q or ESC to quit.")

last_mtime = 0.0
last_status = ""

while True:
    if os.path.exists(STATUS_PATH):
        try:
            with open(STATUS_PATH, "r", encoding="utf-8") as f:
                status = f.read().strip()
            if status != last_status:
                print("[status]", status)
                last_status = status
        except Exception:
            pass

    if os.path.exists(LATEST_IMAGE_PATH):
        try:
            mtime = os.path.getmtime(LATEST_IMAGE_PATH)
            if mtime != last_mtime:
                img = cv2.imread(LATEST_IMAGE_PATH)
                if img is not None:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.imshow(WINDOW_NAME, img)
                    last_mtime = mtime
        except Exception as e:
            print("[viewer] read/display error:", e)

    key = cv2.waitKey(30) & 0xFF
    if key == 27 or key == ord("q"):
        break

    time.sleep(0.03)

cv2.destroyAllWindows()
print("[viewer] closed.")
