import numpy as np


class VisualServoController:
    """픽셀 에러 → EE XY target 보정 (P 제어)."""

    def __init__(self,
                 image_size,
                 kp=0.0008,
                 max_step=0.02,
                 tolerance_px=8,
                 lock_frames=15,
                 axis_sign=(-1.0, -1.0),
                 pixel_to_world_xy=None):
        self.W, self.H = image_size
        self.kp = kp
        self.max_step = max_step
        self.tolerance_px = tolerance_px
        self.lock_frames = lock_frames
        self.axis_sign = axis_sign
        if pixel_to_world_xy is None:
            sx, sy = axis_sign
            pixel_to_world_xy = np.array([[sx, 0.0], [0.0, sy]])
        self.pixel_to_world_xy = np.asarray(pixel_to_world_xy, dtype=float)
        self._stable_count = 0

    def reset(self):
        self._stable_count = 0

    def is_locked(self) -> bool:
        return self._stable_count >= self.lock_frames

    def update(self, current_ee_xy: np.ndarray, det) -> tuple:
        """
        current_ee_xy: world frame XY (2,)
        det: vision_tracker.Detection
        returns: (target_ee_xy: ndarray(2), error_px: float)
        det.found 가 False 면 현재 위치 유지.
        """
        if not det.found:
            self._stable_count = 0
            return current_ee_xy.copy(), float("inf")

        ex_px = det.cx - self.W / 2.0
        ey_px = det.cy - self.H / 2.0
        err_px = float(np.hypot(ex_px, ey_px))

        if err_px < self.tolerance_px:
            self._stable_count += 1
        else:
            self._stable_count = 0

        dx, dy = self.kp * self.pixel_to_world_xy @ np.array([ex_px, ey_px])

        step = float(np.hypot(dx, dy))
        if step > self.max_step:
            scale = self.max_step / step
            dx *= scale
            dy *= scale

        target = current_ee_xy + np.array([dx, dy])
        return target, err_px
