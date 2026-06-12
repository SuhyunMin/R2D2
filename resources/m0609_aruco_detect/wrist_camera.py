import numpy as np

try:
    from isaacsim.sensors.camera import Camera
except ImportError:
    try:
        from omni.isaac.sensor import Camera
    except ImportError:
        raise ImportError("Camera class not found — Isaac Sim 버전을 확인하세요.")

try:
    from scipy.spatial.transform import Rotation as R

    def _rpy_to_quat_wxyz(rpy_deg):
        q = R.from_euler("xyz", rpy_deg, degrees=True).as_quat()  # xyzw
        return np.array([q[3], q[0], q[1], q[2]])

except ImportError:
    def _rpy_to_quat_wxyz(rpy_deg):
        rx, ry, rz = [v * np.pi / 180.0 for v in rpy_deg]
        cx, sx = np.cos(rx / 2), np.sin(rx / 2)
        cy, sy = np.cos(ry / 2), np.sin(ry / 2)
        cz, sz = np.cos(rz / 2), np.sin(rz / 2)
        w = cx * cy * cz + sx * sy * sz
        x = sx * cy * cz - cx * sy * sz
        y = cx * sy * cz + sx * cy * sz
        z = cx * cy * sz - sx * sy * cz
        return np.array([w, x, y, z])


class WristCamera:
    """그리퍼에 부착된 wrist RGB 카메라 센서."""

    def __init__(self,
                 parent_prim_path: str,
                 name: str = "wrist_camera",
                 resolution=(640, 480),
                 frequency: int = 30,
                 translation=(0.0, 0.0, 0.0),
                 rpy_deg=(0.0, 0.0, 0.0)):
        self._prim_path = f"{parent_prim_path}/{name}"
        quat_wxyz = _rpy_to_quat_wxyz(rpy_deg)

        self.camera = Camera(
            prim_path=self._prim_path,
            name=name,
            resolution=resolution,
            frequency=frequency,
            translation=np.array(translation),
            orientation=quat_wxyz,
        )
        self.resolution = resolution

    @classmethod
    def from_existing_prim(cls, prim_path: str,
                           resolution=(640, 480),
                           frequency: int = 30):
        """USD 에 이미 존재하는 Camera prim 을 wrapping 한다.
        (예: RealSense D455 USD 내장 Camera_OmniVision_OV9782_Color)
        translation/orientation 은 prim 이 갖고 있는 값을 그대로 사용한다.
        """
        obj = cls.__new__(cls)
        obj._prim_path = prim_path
        obj.resolution = resolution
        name = prim_path.rsplit('/', 1)[-1]
        obj.camera = Camera(
            prim_path=prim_path,
            name=name,
            resolution=resolution,
            frequency=frequency,
        )
        return obj

    def initialize(self):
        """World.reset() 이후 호출."""
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()

    def get_rgb(self):
        """(H, W, 3) uint8 RGB ndarray 반환. 프레임이 없으면 None."""
        rgba = self.camera.get_rgba()
        if rgba is None or rgba.size == 0:
            return None
        return rgba[..., :3].copy()

    def get_depth(self):
        """(H, W) float32 Depth ndarray 반환 (미터 단위). 프레임이 없으면 None."""
        depth = self.camera.get_current_frame().get("distance_to_image_plane")
        if depth is None or depth.size == 0:
            # 기본적으로 get_distance_to_image_plane_to_frame()이 실패하거나 첫 프레임이 없을 수 있음
            try:
                depth = self.camera.get_depth()
            except AttributeError:
                pass
        if depth is None or depth.size == 0:
            return None
        return depth.copy()

    @property
    def width(self):
        return self.resolution[0]

    @property
    def height(self):
        return self.resolution[1]
