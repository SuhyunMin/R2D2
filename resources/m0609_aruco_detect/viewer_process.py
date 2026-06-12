#!/usr/bin/env python3
"""
Standalone viewer process run by system Python (which has GTK/OpenCV display support).
Reads serialized BGR frames from stdin and displays them in an OpenCV window.
Writes 1-byte key codes back to stdout.

Protocol (stdin):
  [1 byte type] [4 bytes LE size] [data]
  type 0x01 : main BGR frame  — data: H(2LE) W(2LE) + BGR bytes
  type 0x02 : mask frame      — data: H(2LE) W(2LE) + gray bytes
  type 0xFF : shutdown
Protocol (stdout):
  [1 byte] key code from cv2.waitKey (0xFF if no key)
"""
import sys
import struct
import cv2
import numpy as np

MAIN_WIN = "wrist_camera"
MASK_WIN = "mask"

stdin = sys.stdin.buffer
stdout = sys.stdout.buffer


def read_exact(n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = stdin.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(MAIN_WIN, 640, 480)
_mask_win_created = False

while True:
    header = read_exact(5)   # 1 byte type + 4 bytes size
    if header is None:
        break

    msg_type = header[0]
    size = struct.unpack('<I', header[1:5])[0]

    if msg_type == 0xFF:     # shutdown
        break

    data = read_exact(size) if size > 0 else b""
    if data is None:
        break

    if msg_type == 0x01 and len(data) >= 4:
        h, w = struct.unpack('<HH', data[:4])
        frame = np.frombuffer(data[4:], dtype=np.uint8).reshape(h, w, 3).copy()
        
        # 화면 정중앙에 에임용 십자선(Crosshair) 그리기
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1) # 세로선 (흰색)
        cv2.line(frame, (0, cy), (w, cy), (255, 255, 255), 1) # 가로선 (흰색)
        
        # 중심점 강조
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1) # 빨간색 점
        
        cv2.imshow(MAIN_WIN, frame)

    elif msg_type == 0x02 and len(data) >= 4:
        h, w = struct.unpack('<HH', data[:4])
        mask = np.frombuffer(data[4:], dtype=np.uint8).reshape(h, w).copy()
        if not _mask_win_created:
            cv2.namedWindow(MASK_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(MASK_WIN, 320, 240)
            _mask_win_created = True
        cv2.imshow(MASK_WIN, mask)

    key = cv2.waitKey(1) & 0xFF
    stdout.write(bytes([key]))
    stdout.flush()

cv2.destroyAllWindows()
