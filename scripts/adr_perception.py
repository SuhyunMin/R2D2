# adr_perception.py
# ============================================================
# ADR perception 함수 모음 (Isaac 비의존: numpy/cv2/torch/ultralytics만)
#   - adr_integrated.py 가 import해서 사용
#   - 6a: YOLO 탐지 + cv2 오버레이
#   - (예정) 6b: KeypointNet + PnP / 6c: ICP
# ============================================================
import os
import json
import numpy as np

try:
    import cv2
except Exception as _e:
    cv2 = None
    print("[PERC] cv2 import 실패:", _e)

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    from torchvision.models import resnet18
except Exception:
    resnet18 = None
    print("[PERC] torchvision import 실패 - KeypointNet 비활성")

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
    print("[PERC] ultralytics import 실패 - YOLO 비활성")

# ---- 기본 상수 (통합 쪽에서 override 가능) ----
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
YOLO_IMGSZ = 640
YOLO_CONF_THRES = 0.20

# ---- 6b: KeypointNet + PnP 상수 (v11 학습값) ----
KEYPOINT_BBOX_PAD = 0.25                 # 학습 시 pad (v9k의 0.15 아님)
KEYPOINT_CONF_THRES = 0.2                # 히트맵 conf 컷
KEYPOINT_MIN_BBOX_AREA = 900.0           # 이보다 작은 bbox는 pose 스킵
KEYPOINT_MIN_YOLO_CONF_FOR_POSE = 0.35   # YOLO conf 게이트
PNP_MIN_MARKERS = 4                      # PnP 최소 대응점
PNP_ACCEPT_REPROJ_PX = 16.0              # 재투영 오차 허용 상한
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 디버그 이미지 저장 경로 (v11과 동일: ~/yolo_debug_frames/latest.jpg)
#   Isaac 프로세스 내 cv2.imshow는 headless opencv라 죽으므로, 파일로 저장하고
#   별도 이미지 뷰어(자동 새로고침)로 본다.
YOLO_DEBUG_SAVE_DIR = os.path.expanduser("~/yolo_debug_frames")
YOLO_LATEST_IMAGE_PATH = os.path.join(YOLO_DEBUG_SAVE_DIR, "latest.jpg")


def ensure_debug_dir():
    try:
        os.makedirs(YOLO_DEBUG_SAVE_DIR, exist_ok=True)
    except Exception:
        pass


def save_debug_image(rgb_image, detection, status_text,
                     detected_keypoints=None, projected_points=None, projected_lidar_points=None):
    """오버레이 BGR을 latest.jpg에 원자적으로 저장. 경로 반환(실패 시 '')."""
    if cv2 is None:
        return ""
    try:
        ensure_debug_dir()
        bgr = make_yolo_debug_image(rgb_image, detection, status_text,
                                    detected_keypoints, projected_points, projected_lidar_points)
        if bgr is None:
            return ""
        tmp = YOLO_LATEST_IMAGE_PATH + ".tmp.jpg"
        cv2.imwrite(tmp, bgr)
        os.replace(tmp, YOLO_LATEST_IMAGE_PATH)
        return YOLO_LATEST_IMAGE_PATH
    except Exception as e:
        print("[PERC] 디버그 이미지 저장 실패:", e)
        return ""


def load_yolo_model(model_path):
    if YOLO is None:
        print("[PERC] YOLO 사용 불가(ultralytics 없음)")
        return None
    if not os.path.exists(model_path):
        print("[PERC] YOLO 모델 파일 없음:", model_path)
        return None
    try:
        model = YOLO(model_path)
        print("[PERC] YOLO 로드:", model_path)
        try:
            print("[PERC] YOLO names:", model.names)
        except Exception:
            pass
        return model
    except Exception as e:
        print("[PERC] YOLO 로드 실패:", e)
        return None


def select_best_yolo_box(results):
    """최고 conf 박스 1개 -> {'xyxy':np(4), 'conf':float, 'cls_id':int} 또는 None."""
    if results is None or len(results) == 0:
        return None
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    best = None
    best_conf = -1.0
    for box in result.boxes:
        conf = float(box.conf[0].detach().cpu().item())
        if conf > best_conf:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
            cls_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else -1
            best = {"xyxy": xyxy, "conf": conf, "cls_id": cls_id}
            best_conf = conf
    return best


def make_yolo_debug_image(rgb_image, detection, status_text,
                          detected_keypoints=None, projected_points=None, projected_lidar_points=None):
    """RGB(np HxWx3) -> BGR 오버레이(bbox/키포인트/투영점/상태문구). cv2 없으면 None."""
    if cv2 is None:
        return None
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    if detection is not None:
        x1, y1, x2, y2 = detection["xyxy"].astype(int)
        conf = detection["conf"]
        cls_id = detection["cls_id"]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(bgr, f"YOLO cls={cls_id} conf={conf:.2f}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    if detected_keypoints is not None:
        for item in detected_keypoints:
            u, v = item["pixel"].astype(int)
            cv2.circle(bgr, (int(u), int(v)), 6, (0, 255, 255), -1)
            cv2.putText(bgr, item["name"][:4], (int(u) + 6, int(v) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    if projected_points is not None:
        for pt in projected_points:
            if np.any(np.isnan(pt)) or np.any(np.isinf(pt)):
                continue
            u, v = int(pt[0]), int(pt[1])
            if -10000 < u < 10000 and -10000 < v < 10000:
                cv2.drawMarker(bgr, (u, v), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                               markerSize=12, thickness=2)
    if projected_lidar_points is not None:
        pts = np.asarray(projected_lidar_points)
        if pts.ndim == 2 and pts.shape[1] >= 2:
            for pt in pts:
                if np.any(np.isnan(pt)) or np.any(np.isinf(pt)):
                    continue
                u, v = int(pt[0]), int(pt[1])
                if -10000 < u < 10000 and -10000 < v < 10000:
                    cv2.circle(bgr, (u, v), 2, (255, 255, 0), -1)
    yy = 28
    for _chunk in status_text.split(" | "):
        cv2.putText(bgr, _chunk, (16, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        yy += 24
    return bgr


# ============================================================
# 6b: KeypointNet + PnP 6D pose
#   - v11_v8track_cbf.py 에서 이식
#   - estimate_pose_solvepnp 의 K(intrinsic)는 인자로 받음(모듈 독립)
#   - make_camera_rotation_from_forward 는 _set_chaser의 SetLookAt(up=(0,0,1))과 동일 컨벤션
# ============================================================

def _normalize_vec(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def make_camera_rotation_from_forward(forward):
    """forward(월드; 카메라 -Z가 향하는 방향) -> (R_usd_wc, R_cv_wc).
       _set_chaser의 Gf.Matrix4d().SetLookAt(eye,center,up=(0,0,1))과 동일한 카메라 자세."""
    forward = _normalize_vec(forward)
    up_world = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(forward, up_world))) > 0.98:
        up_world = np.array([0.0, 1.0, 0.0], dtype=float)
    right = _normalize_vec(np.cross(forward, up_world))
    up = _normalize_vec(np.cross(right, forward))
    R_usd_wc = np.column_stack((right, up, -forward))   # USD 카메라(+Y up, -Z forward)
    R_cv_wc = np.column_stack((right, -up, forward))     # OpenCV 카메라(+Y down, +Z forward)
    return R_usd_wc, R_cv_wc


class KeypointNet(nn.Module if nn is not None else object):
    def __init__(self, num_kp):
        if nn is None or resnet18 is None:
            raise RuntimeError("torch/torchvision is not available")
        super().__init__()
        bb = resnet18(weights=None)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                                  bb.layer1, bb.layer2, bb.layer3, bb.layer4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(256, num_kp, 1)

    def forward(self, x):
        return self.head(self.deconv(self.stem(x)))


def decode_heatmaps_np(hm):
    K, H, W = hm.shape
    flat = hm.reshape(K, -1)
    idx = flat.argmax(axis=1)
    conf = flat.max(axis=1)
    xs = (idx % W).astype(np.float32)
    ys = (idx // W).astype(np.float32)
    return np.stack([xs, ys], axis=1), conf


def square_crop_params(bbox, pad, W, H):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * pad)
    side = max(side, 8.0)
    x0, y0 = cx - side * 0.5, cy - side * 0.5
    return float(x0), float(y0), float(side)


def load_keypoint_model(model_path, kp3d_path):
    """체크포인트(best.pt) + keypoints_3d.json 로드 -> (model, num_kp, status).
       model에 _crop/_hm/_kps_obj/_device 부착. CAD 3D 키포인트 = kp3d['keypoints_obj']."""
    if torch is None or nn is None or resnet18 is None:
        print("[KPNET] disabled: torch/torchvision unavailable.")
        return None, 0, "torch_unavailable"
    if not os.path.exists(model_path):
        print("[KPNET] checkpoint missing:", model_path)
        return None, 0, "checkpoint_missing"
    if not os.path.exists(kp3d_path):
        print("[KPNET] keypoints_3d.json missing:", kp3d_path)
        return None, 0, "keypoints3d_missing"
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        num_kp = int(ckpt["num_kp"]) if isinstance(ckpt, dict) and "num_kp" in ckpt else None
        crop = int(ckpt.get("crop", 256)) if isinstance(ckpt, dict) else 256
        hm = int(ckpt.get("hm", 64)) if isinstance(ckpt, dict) else 64
        kp3d = json.load(open(kp3d_path))
        kps_obj = np.array(kp3d["keypoints_obj"], dtype=np.float64)
        if num_kp is None:
            num_kp = len(kps_obj)
        if len(kps_obj) != num_kp:
            print(f"[KPNET] WARNING num_kp({num_kp}) != keypoints_3d({len(kps_obj)})")
        model = KeypointNet(num_kp).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model._crop = crop
        model._hm = hm
        model._kps_obj = kps_obj
        model._device = device
        print(f"[KPNET] loaded strict=True num_kp={num_kp} crop={crop} hm={hm} device={device}")
        return model, num_kp, "loaded"
    except Exception as e:
        print("[KPNET] load FAILED:", type(e).__name__, e)
        return None, 0, f"load_failed_{type(e).__name__}"


def predict_keypoints_from_yolo_bbox(rgb_image, yolo_detection, keypoint_model):
    """YOLO bbox crop -> KeypointNet -> 히트맵 -> 풀이미지 픽셀 키포인트.
       반환: (image_points Nx2, object_points Nx3, detected_info[list], crop_box, status)."""
    if keypoint_model is None:
        return None, None, [], None, "keypoint_model_none"
    if torch is None or cv2 is None:
        return None, None, [], None, "torch_or_cv2_none"
    if rgb_image is None or yolo_detection is None:
        return None, None, [], None, "no_rgb_or_yolo"
    if yolo_detection["conf"] < KEYPOINT_MIN_YOLO_CONF_FOR_POSE:
        return None, None, [], None, f"low_yolo_conf_{yolo_detection['conf']:.2f}"
    x1, y1, x2, y2 = yolo_detection["xyxy"].astype(float)
    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if bbox_area < KEYPOINT_MIN_BBOX_AREA:
        return None, None, [], None, f"bbox_too_small_{bbox_area:.0f}px2"
    crop = keypoint_model._crop
    hm = keypoint_model._hm
    kps_obj = keypoint_model._kps_obj
    device = keypoint_model._device
    H_img, W_img = rgb_image.shape[:2]
    x0, y0, side = square_crop_params(yolo_detection["xyxy"], KEYPOINT_BBOX_PAD, W_img, H_img)
    M = np.array([[crop / side, 0, -x0 * crop / side],
                  [0, crop / side, -y0 * crop / side]], dtype=np.float32)
    crop_img = cv2.warpAffine(rgb_image, M, (crop, crop), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    cropf = crop_img.astype(np.float32) / 255.0
    cropf = (cropf - IMAGENET_MEAN) / IMAGENET_STD
    inp = torch.from_numpy(cropf.transpose(2, 0, 1).copy())[None].to(device)
    try:
        with torch.no_grad():
            pred = keypoint_model(inp)[0].detach().cpu().numpy()
    except Exception as e:
        return None, None, [], None, f"forward_exception_{type(e).__name__}"
    coords, confs = decode_heatmaps_np(pred)
    full = coords / hm * side + np.array([x0, y0], dtype=np.float32)
    detected_info = []
    image_points = []
    object_points = []
    for k in range(len(coords)):
        u, v = float(full[k, 0]), float(full[k, 1])
        cc = float(confs[k])
        detected_info.append({"name": f"kp{k:02d}", "pixel": np.array([u, v], dtype=float),
                              "conf": cc, "index": k, "source": "keypoint_net"})
        if cc > KEYPOINT_CONF_THRES:
            image_points.append([u, v])
            object_points.append(kps_obj[k].tolist())
    crop_box = np.array([x0, y0, x0 + side, y0 + side], dtype=float)
    if len(image_points) < PNP_MIN_MARKERS:
        return None, None, detected_info, crop_box, f"not_enough_conf_kp_{len(image_points)}"
    return (np.array(image_points, dtype=np.float32), np.array(object_points, dtype=np.float32),
            detected_info, crop_box, f"keypoint_net_ok_{len(image_points)}kp")


def estimate_pose_solvepnp(image_points, object_points, camera_pos, R_cv_wc, K):
    """2D-3D 대응 -> solvePnPRansac(EPNP)+RefineLM -> 월드 6D pose.
       K: 3x3 intrinsic(통합에서 FX/FY/CX/CY로 구성해 넘김).
       반환: (ok, pos_world(3), R_world_obj(3x3), reproj_err, projected(Nx2), method)."""
    if cv2 is None:
        return False, None, None, -1.0, None, "cv2_none"
    if image_points is None or object_points is None:
        return False, None, None, -1.0, None, "no_points"
    n = len(image_points)
    if n < PNP_MIN_MARKERS:
        return False, None, None, -1.0, None, f"not_enough_markers_{n}"
    K = np.asarray(K, dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    obj = object_points.astype(np.float64)
    img = image_points.astype(np.float64)
    try:
        rvec = None
        tvec = None
        method_used = "none"
        try:
            ok, rvec_r, tvec_r, inliers = cv2.solvePnPRansac(
                obj, img, K, dist_coeffs, iterationsCount=100, reprojectionError=12.0,
                confidence=0.99, flags=cv2.SOLVEPNP_EPNP)
            if ok:
                rvec = rvec_r; tvec = tvec_r; method_used = "ransac_epnp"
        except Exception:
            ok = False
        if rvec is None:
            ok, rvec_e, tvec_e = cv2.solvePnP(obj, img, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
            if not ok:
                return False, None, None, -1.0, None, "epnp_failed"
            rvec = rvec_e; tvec = tvec_e; method_used = "epnp"
        try:
            if hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist_coeffs, rvec, tvec)
                method_used += "+refineLM"
        except Exception:
            pass
        R_cam_obj, _ = cv2.Rodrigues(rvec)
        t_cam_obj = tvec.reshape(3)
        if t_cam_obj[2] <= 0.05:
            return False, None, None, -1.0, None, f"{method_used}_behind_camera"
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist_coeffs)
        projected = projected.reshape(-1, 2)
        reproj_err = float(np.mean(np.linalg.norm(projected - image_points, axis=1)))
        if reproj_err > PNP_ACCEPT_REPROJ_PX:
            return False, None, None, reproj_err, projected, f"{method_used}_reject_reproj_{reproj_err:.1f}"
        R_world_obj = R_cv_wc @ R_cam_obj
        pos_world_obj = np.asarray(camera_pos, dtype=float) + R_cv_wc @ t_cam_obj
        return True, pos_world_obj, R_world_obj, reproj_err, projected, method_used
    except Exception as e:
        print("[PNP WARNING] solvePnP failed.", e)
        return False, None, None, -1.0, None, f"exception_{type(e).__name__}"


# ============================================================
# V8 추적 / 페이즈 머신 / KF / 텀블링 추정 (v11_v8track_cbf.py 이식)
#   - Isaac 비의존 (numpy/math만). 상수·로직 v11 그대로.
#   - FX/CX 등 intrinsic, keypoints_3d 경로는 인자로 받음(모듈 독립).
# ============================================================
import math

# --- 제어/접근 상수 (v11) ---
CAPTURE_DISTANCE = 6.0
MAX_APPROACH_SPEED = 18.0
SLOWDOWN_RADIUS = 60.0
STOP_RADIUS = 0.20
CATALOG_PREDICT_HORIZON_MAX = 22.0
ACQUISITION_START_RANGE = 80.0
ACQUISITION_STANDOFF_DISTANCE = 38.0

# --- FOV 스캔/락 (v11) ---
FOV_LOCK_ANGLE_DEG = 8.0
SEARCH_CONE_DEG = 7.0
SEARCH_AZ_SPEED = 0.55
SEARCH_EL_SPEED = 0.37
VISION_LOCK_HOLD_SEC = 1.5
FOV_MARGIN_PX = 25

# --- 카탈로그/KF 노이즈 (v11) ---
CATALOG_UPDATE_PERIOD = 0.20
CATALOG_POS_NOISE_STD = 0.45
CATALOG_VEL_NOISE_STD = 0.20
PROCESS_ACCEL_STD = 1.2
MEAS_POS_STD = CATALOG_POS_NOISE_STD
MEAS_VEL_STD = CATALOG_VEL_NOISE_STD
DETECTOR_POS_MEAS_STD = 0.35

# --- YOLO/PnP 제어 필터 (v11) ---
YOLO_UPDATE_PERIOD = 0.30
PNP_ACCEPT_KF_POS_DIFF_M = 3.5
PNP_ACCEPT_CAPTURE_JUMP_M = 8.0
PNP_ACCEPT_LAST_GOOD_ANGLE_JUMP_DEG = 45.0
PNP_HOLD_TIME_SEC = 2.0
PNP_CONTROL_MIN_STREAK = 10
USE_PNP_FOR_CONTROL = False
USE_HYBRID_KF_POSITION_PNP_ORIENTATION = True
LOCAL_CAPTURE_OFFSET = np.array([0.0, -6.0, 0.0], dtype=float)

# --- V8 추적 (v11) ---
V8_TRACK_MIN_GOOD_STREAK = 3
V8_BODY_GAP = 0.30
V8_GRASP_INSET = 0.20
V8_LEAD_SEC = 0.02
V8_KP = 1.5
V8_APPROACH_SPEED = 3.0
V8_MAX_CHASER_SPEED = 8.0
V8_MIN_GAP = 0.15
V8_EST_WINDOW = 4
V8_TRIGGER_RANGE = 10.0
V8_TRIGGER_HOLD_SEC = 3.0
V8_WAYPOINT_RANGE = 5.0
V8_WAYPOINT_REACH = 0.6
V8_CREEP_SPEED = 0.6
V8_BODY_HALF = 0.8 / 2.0          # 체이서 본체(0.8m) 반치수

# --- 페이즈 라벨 (v11) ---
PHASE_CATALOG_GUIDANCE = "CATALOG_GUIDANCE"
PHASE_FOV_ACQUISITION = "FOV_ACQUISITION"
PHASE_YOLO_TRACKING = "YOLO_TRACKING"
PHASE_POSE_ESTIMATION = "POSE_ESTIMATION"


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vec / norm


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def limit_vector(vec, max_norm):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    if norm > max_norm:
        return vec / norm * max_norm
    return vec


def euler_xyz_to_rotmat(roll, pitch, yaw):
    cr = math.cos(roll); sr = math.sin(roll)
    cp = math.cos(pitch); sp = math.sin(pitch)
    cy = math.cos(yaw); sy = math.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return Rz @ Ry @ Rx


def rotmat_to_quat_xyzw(R):
    tr = np.trace(R)
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=float)


def forward_to_rotmat_chaser(forward):
    x_axis = normalize(forward)
    up_hint = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(x_axis, up_hint))) > 0.98:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=float)
    y_axis = normalize(np.cross(up_hint, x_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def so3_log(R):
    c = (np.trace(R) - 1.0) * 0.5
    c = max(-1.0, min(1.0, float(c)))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    if abs(ang - math.pi) < 1e-5:
        A = (R + np.eye(3)) * 0.5
        d = np.sqrt(np.clip(np.diag(A), 0.0, 1.0))
        k = int(np.argmax(d))
        axis = A[:, k] / (d[k] + 1e-12)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * ang
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(ang))
    return w * ang


def so3_exp(w):
    ang = float(np.linalg.norm(w))
    if ang < 1e-9:
        return np.eye(3)
    k = w / ang
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def rotation_angle_between_deg(R_a, R_b):
    R_rel = R_a.T @ R_b
    tr = float(np.trace(R_rel))
    cos_angle = (tr - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def angle_error_deg(R_est, R_true):
    R_delta = R_est @ R_true.T
    value = (np.trace(R_delta) - 1.0) / 2.0
    value = max(-1.0, min(1.0, float(value)))
    return math.degrees(math.acos(value))


def angle_between_vectors_deg(a, b):
    a_n = normalize(a); b_n = normalize(b)
    dot = max(-1.0, min(1.0, float(np.dot(a_n, b_n))))
    return math.degrees(math.acos(dot))


def transform_local_to_world(pos, R, local_point):
    return pos + R @ local_point


def compute_slowdown_speed(error_dist):
    if error_dist <= STOP_RADIUS:
        return 0.0
    if error_dist >= SLOWDOWN_RADIUS:
        return MAX_APPROACH_SPEED
    ratio = error_dist / SLOWDOWN_RADIUS
    return MAX_APPROACH_SPEED * ratio


def compute_capture_from_pos_vel(pos, vel):
    return pos - normalize(vel) * CAPTURE_DISTANCE


def predict_catalog_target_position(kf_pos, kf_vel, chaser_pos):
    range_est = float(np.linalg.norm(kf_pos - chaser_pos))
    lead_t = clamp(range_est / max(MAX_APPROACH_SPEED, 1e-6), 0.0, CATALOG_PREDICT_HORIZON_MAX)
    return kf_pos + kf_vel * lead_t, lead_t


def compute_standoff_point_from_predicted(pred_pos, pred_vel, distance):
    return pred_pos - normalize(pred_vel) * distance


def choose_mission_phase(kf_pos, chaser_pos, vision_recent, pose_recent):
    estimated_range = float(np.linalg.norm(kf_pos - chaser_pos))
    if pose_recent:
        return PHASE_POSE_ESTIMATION
    if vision_recent:
        return PHASE_YOLO_TRACKING
    if estimated_range <= ACQUISITION_START_RANGE:
        return PHASE_FOV_ACQUISITION
    return PHASE_CATALOG_GUIDANCE


def make_F(dt):
    F = np.eye(6)
    F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
    return F


def make_Q(dt, accel_std):
    q = accel_std ** 2
    Q = np.zeros((6, 6))
    dt2 = dt * dt; dt3 = dt2 * dt; dt4 = dt2 * dt2
    for i in range(3):
        Q[i, i] = 0.25 * dt4 * q
        Q[i, i + 3] = 0.5 * dt3 * q
        Q[i + 3, i] = 0.5 * dt3 * q
        Q[i + 3, i + 3] = dt2 * q
    return Q


def kalman_predict(x, P, dt):
    F = make_F(dt)
    Q = make_Q(dt, PROCESS_ACCEL_STD)
    return F @ x, F @ P @ F.T + Q


def kalman_update_catalog(x, P, z):
    H = np.eye(6)
    R = np.diag([MEAS_POS_STD ** 2, MEAS_POS_STD ** 2, MEAS_POS_STD ** 2,
                 MEAS_VEL_STD ** 2, MEAS_VEL_STD ** 2, MEAS_VEL_STD ** 2])
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    return x + K @ y, (np.eye(6) - K @ H) @ P


def kalman_update_detector_position(x, P, z_pos):
    H = np.zeros((3, 6)); H[0, 0] = 1.0; H[1, 1] = 1.0; H[2, 2] = 1.0
    R = np.diag([DETECTOR_POS_MEAS_STD ** 2, DETECTOR_POS_MEAS_STD ** 2, DETECTOR_POS_MEAS_STD ** 2])
    y = z_pos - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    return x + K @ y, (np.eye(6) - K @ H) @ P


class TumblingEstimator:
    def __init__(self, window=8):
        self.window = window
        self.hist = []

    def update(self, t, R, p):
        self.hist.append((float(t), np.asarray(R, float).reshape(3, 3), np.asarray(p, float).reshape(3)))
        if len(self.hist) > self.window:
            self.hist.pop(0)

    def ready(self):
        return len(self.hist) >= 2

    def omega(self):
        if len(self.hist) < 2:
            return np.zeros(3)
        ws = []
        for i in range(1, len(self.hist)):
            t0, R0, _ = self.hist[i - 1]
            t1, R1, _ = self.hist[i]
            dt = t1 - t0
            if dt <= 1e-6:
                continue
            ws.append(so3_log(R1 @ R0.T) / dt)
        return np.mean(ws, axis=0) if ws else np.zeros(3)

    def vel(self):
        if len(self.hist) < 2:
            return np.zeros(3)
        t0, _, p0 = self.hist[0]
        t1, _, p1 = self.hist[-1]
        dt = t1 - t0
        return (p1 - p0) / dt if dt > 1e-6 else np.zeros(3)

    def predict(self, h):
        _, R, p = self.hist[-1]
        return so3_exp(self.omega() * h) @ R, p + self.vel() * h

    def current_point(self, pb):
        _, R, p = self.hist[-1]
        return R @ np.asarray(pb, float) + p

    def predict_point(self, pb, h):
        R, p = self.predict(h)
        return R @ np.asarray(pb, float) + p

    def point_velocity(self, pb):
        if not self.ready():
            return np.zeros(3)
        _, R, _ = self.hist[-1]
        return self.vel() + np.cross(self.omega(), R @ np.asarray(pb, float))


def point_to_segment(p, a, b):
    ab = b - a
    tt = float(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12))
    tt = max(0.0, min(1.0, tt))
    return float(np.linalg.norm(p - (a + tt * ab)))


def load_v8_grasp_geometry(kp3d_path):
    """keypoints_3d.json의 CAD 키포인트로 사다리 장축/tip/grasp(body frame) 계산.
       반환: (grasp, centroid, tip_pos, tip_neg). v11 동일."""
    try:
        kp = json.load(open(kp3d_path, "r", encoding="utf-8"))
        K = np.array(kp["keypoints_obj"], dtype=float)
        c = K.mean(axis=0)
        w, v = np.linalg.eigh(np.cov((K - c).T))
        axis = v[:, int(np.argmax(w))]
        proj = (K - c) @ axis
        tip_pos = c + axis * proj.max()
        tip_neg = c + axis * proj.min()
        grasp = c + axis * (proj.max() - V8_GRASP_INSET)
        print(f"[V8] grasp={grasp.round(3)} tip={tip_pos.round(3)} center={c.round(3)}")
        return grasp, c, tip_pos, tip_neg
    except Exception as e:
        print(f"[V8 WARNING] keypoints_3d load fail({type(e).__name__}: {e}). default axis.")
        return (np.array([0.0, 0.0, 2.4 - V8_GRASP_INSET], dtype=float),
                np.zeros(3, dtype=float),
                np.array([0.0, 0.0, 2.4], dtype=float),
                np.array([0.0, 0.0, -2.4], dtype=float))


def bbox_center_to_world_position(bbox_xyxy, camera_pos, R_cv_wc, range_est, fx, fy, cx, cy):
    x1, y1, x2, y2 = bbox_xyxy
    u = 0.5 * (x1 + x2); v = 0.5 * (y1 + y2)
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
    ray_cam = normalize(ray_cam)
    ray_world = normalize(R_cv_wc @ ray_cam)
    return camera_pos + ray_world * range_est


def project_world_point_to_camera_pixel(world_point, camera_pos, R_cv_wc, fx, fy, cx, cy):
    R_cw = R_cv_wc.T
    p_cam = R_cw @ (world_point - camera_pos)
    if p_cam[2] <= 1e-6:
        return None, p_cam
    u = fx * (p_cam[0] / p_cam[2]) + cx
    v = fy * (p_cam[1] / p_cam[2]) + cy
    return np.array([u, v], dtype=float), p_cam


def is_pixel_inside_fov(pixel, img_w, img_h):
    if pixel is None:
        return False
    u, v = pixel
    return (FOV_MARGIN_PX <= u <= img_w - FOV_MARGIN_PX
            and FOV_MARGIN_PX <= v <= img_h - FOV_MARGIN_PX)


def compute_camera_search_offset(R_chaser_wc, range_est, t):
    y_axis = R_chaser_wc[:, 1]
    z_axis = R_chaser_wc[:, 2]
    az = math.radians(SEARCH_CONE_DEG) * math.sin(SEARCH_AZ_SPEED * t)
    el = 0.75 * math.radians(SEARCH_CONE_DEG) * math.sin(SEARCH_EL_SPEED * t + 1.2)
    return (math.tan(az) * range_est * y_axis + math.tan(el) * range_est * z_axis,
            math.degrees(az), math.degrees(el))