import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ArucoDetection:
    """VisualServoController 와 호환되는 detection (cx, cy, bbox) 에
    ArUco pose 추정 결과 (corners, rvec, tvec, marker_id) 를 추가한다.
    """
    found: bool
    cx: float = 0.0
    cy: float = 0.0
    area: float = 0.0
    bbox: tuple = (0, 0, 0, 0)
    mask: Optional[np.ndarray] = None
    marker_id: int = -1
    corners: Optional[np.ndarray] = None      # (4, 2) image pixel coords
    rvec: Optional[np.ndarray] = None         # (3,) Rodrigues, marker in camera frame
    tvec: Optional[np.ndarray] = None         # (3,) translation in camera frame (m)


class ArucoTracker:
    """카메라 영상에서 특정 ID 의 ArUco 마커를 검출하고 solvePnP 로 pose 를 추정한다.

    VisualServoController 호환:
      - detect() 가 반환하는 ArucoDetection 은 cx, cy, bbox 를 갖는다.
        ⇒ 기존 servo 코드 (픽셀 중심 기반 P 제어) 와 그대로 호환된다.

    추가 정보:
      - marker_id, corners, rvec, tvec → 카메라 좌표계 pose 사용 가능.
    """

    def __init__(self,
                 marker_length: float,
                 target_id: Optional[int] = None,
                 aruco_dict_id: int = cv2.aruco.DICT_6X6_250,
                 K: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        marker_length: ArUco 검출 코너 (outer black border) 간 한 변 길이 (m).
                       PNG quiet-zone 비율을 곱해 결정. (aruco_multiple_standalone.py 참조)
        target_id    : 검출할 마커 ID. None 이면 검출된 첫 마커 사용.
        K, dist_coeffs: pose 추정을 위한 카메라 intrinsics. 나중에
                       set_intrinsics() 로 주입 가능 (Camera.initialize() 이후).
        """
        self.marker_length = float(marker_length)
        self.target_id = target_id
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self._detector = cv2.aruco.ArucoDetector(
            self._aruco_dict, cv2.aruco.DetectorParameters()
        )
        self.K = None if K is None else np.asarray(K, dtype=np.float64)
        self.dist = (
            np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None
            else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        )

        half = self.marker_length / 2.0
        self._obj_points = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float32)

    def set_intrinsics(self, K: np.ndarray, dist_coeffs: Optional[np.ndarray] = None):
        self.K = np.asarray(K, dtype=np.float64)
        if dist_coeffs is not None:
            self.dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    def detect(self, bgr: np.ndarray) -> ArucoDetection:
        if bgr is None:
            return ArucoDetection(found=False)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self._detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return ArucoDetection(found=False)

        ids_flat = ids.flatten()
        if self.target_id is None:
            idx = 0
        else:
            matches = np.where(ids_flat == int(self.target_id))[0]
            if len(matches) == 0:
                return ArucoDetection(found=False)
            idx = int(matches[0])

        corner = corners_list[idx][0]   # (4, 2)
        cx = float(corner[:, 0].mean())
        cy = float(corner[:, 1].mean())

        x_min, y_min = corner.min(axis=0)
        x_max, y_max = corner.max(axis=0)
        bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        area = float((x_max - x_min) * (y_max - y_min))

        rvec = None
        tvec = None
        if self.K is not None:
            ok, rvec_out, tvec_out = cv2.solvePnP(
                self._obj_points, corner, self.K, self.dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ok:
                rvec = rvec_out.flatten().astype(np.float64)
                tvec = tvec_out.flatten().astype(np.float64)

        return ArucoDetection(
            found=True,
            cx=cx, cy=cy, area=area, bbox=bbox,
            marker_id=int(ids_flat[idx]),
            corners=corner.copy(),
            rvec=rvec, tvec=tvec,
        )

    def annotate(self, bgr: np.ndarray, det: ArucoDetection) -> np.ndarray:
        """디버그용 시각화. detection 이 found 이면 corner/축/ID 를 그린다."""
        out = bgr.copy()
        if not det.found or det.corners is None:
            return out
        corners_for_draw = [det.corners.reshape(1, 4, 2).astype(np.float32)]
        cv2.aruco.drawDetectedMarkers(out, corners_for_draw,
                                      np.array([[det.marker_id]]))
        if self.K is not None and det.rvec is not None and det.tvec is not None:
            cv2.drawFrameAxes(out, self.K, self.dist,
                              det.rvec, det.tvec, self.marker_length * 0.5)
        return out
