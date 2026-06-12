import os
import select
import struct
import subprocess
import threading
import queue
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_VIEWER_SCRIPT = Path(__file__).parent / "viewer_process.py"
_SAVE_DIR = "/tmp"


def _clean_env() -> dict:
    """Isaac Sim Python 환경변수를 제거한 시스템 Python 용 최소 환경."""
    keep = {
        'HOME', 'USER', 'USERNAME', 'LOGNAME',
        'DISPLAY', 'XAUTHORITY', 'WAYLAND_DISPLAY',
        'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS',
        'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ', 'TERM',
    }
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.setdefault('PATH', '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')
    return env


class CameraViewer:
    """
    Isaac Sim 과 별개의 OpenCV 윈도우로 wrist 카메라를 표시.

    Isaac Sim 번들 Python 의 cv2 는 GUI 없이 컴파일되어 있으므로,
    imshow 가 불가능할 경우 시스템 python3 (GTK3 지원) 로 별도 프로세스를 띄워
    stdin/stdout 파이프로 프레임을 주고받는다.
    둘 다 실패하면 /tmp/wrist_*.png 에 imwrite 로 fallback.
    """

    def __init__(self,
                 window_name: str = "wrist_camera",
                 mask_window_name: str = "mask",
                 show_mask: bool = True,
                 enabled: bool = True):
        self.window_name = window_name
        self.mask_window_name = mask_window_name
        self.show_mask = show_mask
        self.enabled = enabled

        self._initialized = False
        # 'local' | 'subprocess' | 'imwrite' | None
        self._mode: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._send_queue: Optional[queue.Queue] = None
        self._send_thread: Optional[threading.Thread] = None
        self._frame_count = 0

    # ------------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------------
    def _ensure_init(self):
        if self._initialized or not self.enabled:
            return

        # 1) Isaac Sim 번들 cv2 로 직접 imshow
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 640, 480)
            if self.show_mask:
                cv2.namedWindow(self.mask_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.mask_window_name, 320, 240)
            self._mode = 'local'
            print("[CameraViewer] 로컬 cv2.imshow 모드")
        except cv2.error:
            # 2) 시스템 python3 로 별도 뷰어 프로세스
            try:
                self._proc = subprocess.Popen(
                    ['/usr/bin/python3', str(_VIEWER_SCRIPT)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    env=_clean_env(),
                )
                self._send_queue = queue.Queue(maxsize=2)
                self._send_thread = threading.Thread(
                    target=self._sender_loop, daemon=True)
                self._send_thread.start()
                self._mode = 'subprocess'
                print("[CameraViewer] 시스템 python3 뷰어 프로세스 실행 "
                      f"(PID={self._proc.pid})")
            except Exception as e:
                # 3) imwrite fallback
                self._mode = 'imwrite'
                print(f"[CameraViewer] 뷰어 실행 실패 ({e}). "
                      f"imwrite 모드 → {_SAVE_DIR}/wrist_*.png")

        self._initialized = True

    # ------------------------------------------------------------------
    # subprocess 전송 스레드 (non-blocking send)
    # ------------------------------------------------------------------
    def _sender_loop(self):
        """별도 스레드: queue 에서 메시지를 꺼내 subprocess stdin 에 씀."""
        while True:
            item = self._send_queue.get()
            if item is None:
                break
            if self._proc is None or self._proc.poll() is not None:
                break
            try:
                for raw in item:
                    self._proc.stdin.write(raw)
                self._proc.stdin.flush()
            except BrokenPipeError:
                break

    def _enqueue(self, msg_type: int, data: bytes):
        """프레임 데이터를 전송 큐에 넣는다 (큐가 꽉 찼으면 드롭)."""
        header = bytes([msg_type]) + struct.pack('<I', len(data))
        try:
            self._send_queue.put_nowait([header + data])
        except queue.Full:
            pass

    def _send_shutdown(self):
        header = bytes([0xFF]) + struct.pack('<I', 0)
        try:
            self._send_queue.put_nowait([header])
        except queue.Full:
            pass

    def _read_key_from_proc(self) -> int:
        """subprocess stdout 에서 키 코드를 비동기로 읽는다."""
        if self._proc is None or self._proc.poll() is not None:
            return -1
        try:
            if select.select([self._proc.stdout], [], [], 0)[0]:
                b = self._proc.stdout.read(1)
                return b[0] if b else -1
        except Exception:
            pass
        return -1

    # ------------------------------------------------------------------
    # overlay 그리기
    # ------------------------------------------------------------------
    def _draw_overlay(self, bgr: np.ndarray, detection,
                      state_str: str, extra_lines: Optional[list]) -> np.ndarray:
        H, W = bgr.shape[:2]
        cv2.line(bgr, (W // 2, 0), (W // 2, H), (0, 255, 255), 1)
        cv2.line(bgr, (0, H // 2), (W, H // 2), (0, 255, 255), 1)

        if detection is not None and getattr(detection, "found", False):
            x, y, w, h = detection.bbox
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(bgr, (int(detection.cx), int(detection.cy)),
                       4, (0, 0, 255), -1)
            err_x = detection.cx - W / 2
            err_y = detection.cy - H / 2
            cv2.putText(bgr, f"err=({err_x:+.1f},{err_y:+.1f})px",
                        (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

        if state_str:
            cv2.putText(bgr, f"state: {state_str}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if extra_lines:
            for i, line in enumerate(extra_lines):
                cv2.putText(bgr, str(line), (10, 50 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return bgr

    # ------------------------------------------------------------------
    # 메인 API
    # ------------------------------------------------------------------
    def update(self,
               rgb: Optional[np.ndarray],
               detection=None,
               state_str: str = "",
               extra_lines: Optional[list] = None) -> int:
        """
        rgb: WristCamera.get_rgb() 결과 (RGB) 또는 None.
        반환: 키 코드 (-1 if no input).
        """
        if not self.enabled:
            return -1
        self._ensure_init()

        # ── 프레임 없음 ──────────────────────────────────────────────
        if rgb is None:
            if self._mode == 'local':
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(blank, "no frame", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow(self.window_name, blank)
                return cv2.waitKey(1) & 0xFF
            elif self._mode == 'subprocess':
                return self._read_key_from_proc()
            return -1

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = self._draw_overlay(bgr, detection, state_str, extra_lines)

        # ── local 모드 ───────────────────────────────────────────────
        if self._mode == 'local':
            cv2.imshow(self.window_name, bgr)
            if self.show_mask and detection is not None and detection.mask is not None:
                cv2.imshow(self.mask_window_name, detection.mask)
            return cv2.waitKey(1) & 0xFF

        # ── subprocess 모드 ──────────────────────────────────────────
        if self._mode == 'subprocess':
            h, w = bgr.shape[:2]
            self._enqueue(0x01, struct.pack('<HH', h, w) + bgr.tobytes())
            if self.show_mask and detection is not None and detection.mask is not None:
                mask = detection.mask
                mh, mw = mask.shape[:2]
                self._enqueue(0x02, struct.pack('<HH', mh, mw) + mask.tobytes())
            return self._read_key_from_proc()

        # ── imwrite fallback ─────────────────────────────────────────
        if self._frame_count % 10 == 0:
            cv2.imwrite(f"{_SAVE_DIR}/wrist_{self._frame_count:06d}.png", bgr)
            if self.show_mask and detection is not None and detection.mask is not None:
                cv2.imwrite(f"{_SAVE_DIR}/mask_{self._frame_count:06d}.png",
                            detection.mask)
        self._frame_count += 1
        return -1

    # ------------------------------------------------------------------
    # 종료
    # ------------------------------------------------------------------
    def close(self):
        if not self._initialized:
            return
        if self._mode == 'local':
            try:
                cv2.destroyWindow(self.window_name)
                if self.show_mask:
                    cv2.destroyWindow(self.mask_window_name)
            except cv2.error:
                pass
        elif self._mode == 'subprocess':
            self._send_shutdown()
            if self._send_thread:
                self._send_queue.put(None)   # 스레드 종료
                self._send_thread.join(timeout=2)
            if self._proc:
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        self._initialized = False
