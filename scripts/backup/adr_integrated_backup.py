# scripts/adr_integrated.py
# ============================================================
# ADR 통합 진입점 (v10_leo 맵 + v11 추적/회피 에이전트)
# ============================================================
# 구조:
#   - 이 파일이 유일한 standalone 진입점. SimulationApp을 여기서 한 번만 생성.
#   - space_environment_v10_leo 를 "씬 모듈"로 import (맵: 텀블링 사다리/공전체/지구/배경/카메라).
#   - (이후 단계) v11_v8track_cbf 의 에이전트 로직(perception/추적/회피/제어)을 얹는다.
#
# 실행:
#   cd ~/isaacsim && ./python.sh ~/space_debris_ai/scripts/adr_integrated.py
#
# 진행 단계(설계):
#   [1] (현재) 씬 머지 골격 - 맵만 띄워 공전/텀블링 확인. 체이서 없음.
#   [2] 타깃 swap        - v11 true_target_state -> 맵 DebrisLadder 월드 transform 읽기
#   [3] 체이서 생성/시작점 - 사다리 t=0 위치 +400m, 회피 끄고 직선 접근 확인
#   [4] 회피 연결        - CBF 장애물 소스=맵 공전체, 지구를 구형 정적 장애물로 추가
#   [5] 원거리 카탈로그   - 맵 사다리 궤도로 KF 초기화 + capture standoff 미리 마킹
#   [6] perception 재확인 - 맵 조명/별 배경/큰 지구에서 YOLO/PnP 정확도 점검
# ============================================================

# --- SimulationApp은 어떤 omni/pxr import보다 먼저, 그리고 씬 모듈 import보다 먼저 ---
import argparse
parser = argparse.ArgumentParser(description='Integrated Space Debris Recovery')
parser.add_argument('--enable-adr', action='store_true', help='Enable Cube tracking ADR')
parser.add_argument('--enable-berthing', action='store_true', help='Enable Robot arm berthing')
args, _ = parser.parse_known_args()
if not args.enable_adr and not args.enable_berthing:
    args.enable_adr = True
    args.enable_berthing = True
config = args

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# asset 변환(GLB->USD) 확장 (맵 빌드 시 필요) 및 ROS 2 브릿지 활성화
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.kit.asset_converter")
    enable_extension("omni.isaac.ros2_bridge")
    print("[INTEGRATED] ROS 2 브릿지 활성화 성공")
except Exception as _e:
    print("[INTEGRATED WARN] 확장 활성화 실패:", _e)

import omni.timeline
import omni.usd
import numpy as np
from pxr import Usd, UsdGeom, Gf, UsdLux

# SimulationApp 생성 후라야 씬 모듈 내부의 omni/pxr import가 성공한다
import space_environment_v10_leo as scene
import sys, os, cv2, time, math, random
import numpy as np
from scipy.spatial.transform import Rotation
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.join(THIS_DIR, '..', 'resources', 'm0609_aruco_detect') not in sys.path:
    sys.path.append(os.path.join(THIS_DIR, '..', 'resources', 'm0609_aruco_detect'))
if os.path.join(THIS_DIR, '..', 'resources', 'robots') not in sys.path:
    sys.path.append(os.path.join(THIS_DIR, '..', 'resources', 'robots'))
from wrist_camera import WristCamera
from visual_servo_controller import VisualServoController
from m0609_rmpflow_controller import RMPFlowController
from realsense_mount import attach_realsense_d455
from camera_viewer import CameraViewer
from omni.isaac.core.articulations import ArticulationView, Articulation
from omni.isaac.dynamic_control import _dynamic_control
from pxr import UsdPhysics, PhysxSchema, Gf, UsdGeom, UsdLux, Sdf, Usd, UsdShade
class Det: pass
def apply_high_friction(stage, prim_path, mu=1.8):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid(): return
    mat_path = f'{prim_path}/HighFrictionMat'
    mat = UsdShade.Material.Define(stage, mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys_mat.CreateStaticFrictionAttr().Set(mu)
    phys_mat.CreateDynamicFrictionAttr().Set(mu)
    phys_mat.CreateRestitutionAttr().Set(0.0)
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(mat, materialPurpose='physics')
def find_prim_path_by_name(root_path, link_name):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid(): return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == link_name: return str(prim.GetPath())
    return None
    
# ============================================================
# [1] 맵 빌드
# ============================================================
handles = scene.build_scene(simulation_app)
ladder_path = handles["ladder_path"]
orbit_step = handles["orbit_step"]
kinematic_bodies = handles["kinematic_bodies"]

print("\n[INTEGRATED] 맵 빌드 완료")
print("  타깃 사다리 prim :", ladder_path)
print("  공전체 수        :", len(kinematic_bodies))
print("  지구 prim        :", handles["earth_path"])

import iss_berthing

# ROS 2 셋업
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CommandListener(Node):
    def __init__(self):
        super().__init__('chaser_command_listener')
        self.subscription = self.create_subscription(
            String,
            '/chaser_command',
            self.listener_callback,
            10
        )
        print("[INTEGRATED] ROS 2 노드 생성됨. 토픽 /chaser_command 대기 중...")
        
    def listener_callback(self, msg):
        cmd = msg.data
        print(f"[ROS 2] 명령 수신됨: {cmd}")
        iss_berthing.external_command = cmd

rclpy.init()
ros_node = CommandListener()

if config.enable_berthing:
    print("[INTEGRATED] Berthing Setup 시작...")
    stage = omni.usd.get_context().get_stage()
    iss_berthing.setup_berthing(stage, None, None, None, simulation_app, scene.SPACE_STATION_PATH)

# ------------------------------------------------------------
# [2+3] 체이서: 맵 사다리(GT) 기준 ~400m 밖에 생성 -> GT로 직선 접근(임시)
#   * 지금은 perception/회피 OFF. "체이서가 사다리 따라가 standoff까지 가는지"만 확인.
#   * 다음 단계에서 이 chaser_step을 v11 on_update(perception->KF->V8->CBF->제어)로 교체.
# ------------------------------------------------------------
chaser_pos = np.zeros(3)
obstacle_list = []
stage = omni.usd.get_context().get_stage()

def _world_pos(path):
    prim = stage.GetPrimAtPath(path)
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=float)

if config.enable_adr:
    
    # 체이서 시작점: 사다리 t=0 위치에서 반경 바깥 방향으로 400m + 약간 위 (정밀 시작점은 나중에)
    _ladder0 = _world_pos(ladder_path)
    _radial = _ladder0 / (float(np.linalg.norm(_ladder0)) + 1e-9)
    chaser_pos = _ladder0 + _radial * 400.0 + np.array([0.0, 0.0, 5.0])
    
    # 체이서 prim: 본체 큐브(0.8m) + 전방 카메라(-Z 전방). v11 perception 붙일 때 이 카메라 사용.
    CHASER_PATH = "/World/Chaser"
    _croot = UsdGeom.Xform.Define(stage, CHASER_PATH)
    _cxf = UsdGeom.Xformable(_croot.GetPrim())
    _cxf.ClearXformOpOrder()
    # 물리(RigidPrim)와 호환되는 translate+orient op 스택 (v11 검증 패턴). 매트릭스 TransformOp은 RigidPrim과 충돌.
    _chaser_translate_op = _cxf.AddTranslateOp()
    _chaser_orient_op = _cxf.AddOrientOp()
    _cbody = UsdGeom.Cube.Define(stage, CHASER_PATH + "/Body")
    _cbody.GetSizeAttr().Set(0.8)
    _ccam = UsdGeom.Camera.Define(stage, CHASER_PATH + "/FrontCam")
    _ccam.CreateFocalLengthAttr(24.0)
    _ccam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1.0e6))
    # 카메라를 큐브(0.8m) 밖 전방(-Z=조준방향)으로 빼낸다. 큐브 반치수 0.4 + 여유 0.1.
    #   creep 막바지 비충돌 클램프(중심이 사다리끝 0.55m)에서 카메라가 사다리를 뚫지 않도록 0.5로 제한.
    CAM_FWD_OFFSET = 0.5
    _ccam_xf = UsdGeom.Xformable(_ccam.GetPrim())
    _ccam_xf.ClearXformOpOrder()
    _ccam_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -CAM_FWD_OFFSET))
    # 카메라 헤드라이트(전방 -Z 평행광): 카메라 자식이라 카메라 따라 이동.
    #   -> 체이서캠 화면이 밝아지고 RGB(YOLO) 입력도 같이 밝아져 인식에 유리.
    _headlight = UsdLux.DistantLight.Define(stage, CHASER_PATH + "/FrontCam/Headlight")
    _headlight.CreateIntensityAttr(3000.0)
    _headlight.CreateAngleAttr(1.5)
    
    # ----- 3인칭 관전 카메라(월드 고정 prim): 매 프레임 체이서 뒤+위 대각선으로 옮겨 체이서·사다리 둘 다 담음 -----
    THIRD_PERSON_CAM_PATH = "/World/ThirdPersonCam"
    _tpcam = UsdGeom.Camera.Define(stage, THIRD_PERSON_CAM_PATH)
    _tpcam.CreateFocalLengthAttr(18.0)                      # FrontCam(24)보다 살짝 광각 -> 둘 다 들어오게
    _tpcam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1.0e6))
    _tpcam_xf = UsdGeom.Xformable(_tpcam.GetPrim())
    _tpcam_xf.ClearXformOpOrder()
    _tp_cam_top = _tpcam_xf.AddTransformOp()                # 매 프레임 look-at 4x4로 세팅
    
    
    def _set_chaser(eye, center):
        """체이서를 eye에 두고 center(사다리)를 바라보게(-Z 전방) 세팅. 카메라도 따라옴.
           translate+orient op로 세팅(기존 SetLookAt 매트릭스를 분해 -> 자세 동일, 물리 RigidPrim 호환)."""
        _m = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])),
            Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])),
            Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
        _chaser_translate_op.Set(_m.ExtractTranslation())
        _q = _m.ExtractRotationQuat()                       # Gf.Quatd (w + xyz)
        _im = _q.GetImaginary()
        _chaser_orient_op.Set(Gf.Quatf(float(_q.GetReal()),
                                       float(_im[0]), float(_im[1]), float(_im[2])))
    
    
    # ===== [물리] 리지드바디 + 콜라이더 + 중력0 + 가속제한 속도제어 + 토크 자세제어 (v11 검증) =====
    ENABLE_PHYSICS = True          # False면 기존 키네마틱 경로로 즉시 폴백
    CHASER_MASS = 500.0            # kg (동적 바디)
    CHASER_BODY_SIZE = 0.8         # m (관성 추정/표시용; Body Cube와 동일)
    PHYS_MAX_ACCEL = 4.0           # m/s^2 (추력/질량 -> 속도명령 변화율 상한, 관성 느낌)
    PHYS_ATT_KP = 6.0              # (토크 OFF일 때) 각속도 P
    PHYS_MAX_ANGVEL = 5.0          # rad/s
    ENABLE_TORQUE_ATTITUDE = True  # 토크 PD 자세제어 (v11 검증). 흔들리면 False -> 각속도 직접.
    ATT_KP_TORQUE = 300.0          # 자세 P (쿼터니언 오차 -> 토크)
    ATT_KD_TORQUE = 200.0          # 자세 D (각속도 감쇠)
    ATT_MAX_TORQUE = 400.0         # N·m 상한
    
    # ===== [추력기 플룸 시각화 — RCS jet 파티클] =====
    ENABLE_THRUSTER_FX = True      # 동적 추력/토크 방향으로 파티클 플룸 분사 (어디서 추진하는지 시각화)
    THR_PARTICLES_PER_NOZZLE = 6   # 노즐당 파티클 수 (밀도 ↓: 시야 가림 완화)
    THR_PLUME_LEN = 2.4            # 플룸 최대 길이(m) (길이 ↓: 카메라 시야 침범 완화)
    THR_PARTICLE_R = 0.06          # 파티클 반경(m) (굵기 ↓)
    THR_FORCE_REF = 2000.0         # 이 추력(N)에서 플룸 최대 (낮출수록 작은 추력에도 크게 보임)
    THR_TORQUE_REF = 150.0         # 이 토크(N·m)에서 플룸 최대
    
    
    def quat_xyzw_to_rotmat(q):
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]], dtype=float)
    
    
    def _quat_mul_xyzw(a, b):
        ax, ay, az, aw = a; bx, by, bz, bw = b
        return np.array([
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz], dtype=float)
    
    
    def quat_align_angvel(q_cur_xyzw, q_des_xyzw, k):
        """현재->목표 자세로 정렬하는 각속도(소각 근사). 토크 PD의 오차벡터로도 사용."""
        qc = np.asarray(q_cur_xyzw, float)
        qd = np.asarray(q_des_xyzw, float)
        q_err = _quat_mul_xyzw(qd, np.array([-qc[0], -qc[1], -qc[2], qc[3]]))   # qd * conj(qc)
        if q_err[3] < 0.0:
            q_err = -q_err                                                      # 최단경로
        return 2.0 * float(k) * q_err[:3]
    
    
    CHASER_APPROACH_SPEED = 18.0   # v11과 동일
    STANDOFF_DISTANCE = 10.0       # 6b PnP 검증용: PnP가 잡히는 거리까지 접근 (v9j 12m 검증보다 가까이). 근접 V8 추적은 6d
    CATALOG_PREDICT_HORIZON_MAX = 22.0   # 미래 예측 최대 lead (s)
    ENABLE_AVOIDANCE = True
    _sim_t = 0.0                   # orbit_step과 동일 dt로 누적 -> 궤도 예측에 사용
    
    # 타깃 사다리 궤도 def(케플러). "지구가 알려주는 궤도" 역할 -> 미래위치 예측에 사용.
    ladder_def = None
    for _b in kinematic_bodies:
        if _b["name"] == "DebrisLadder":
            ladder_def = _b
            break
    if ladder_def is None:
        print("[INTEGRATED][WARN] DebrisLadder 궤도 def를 못 찾음 -> 예측 불가")
    
    
    def _normalize(v):
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else np.array(v, dtype=float)
    
    
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))
    
    
    # capture(standoff) 마커: 카메라 시야를 가려서 기본 OFF (필요하면 True로)
    ENABLE_CAPTURE_MARKER = False
    MARKER_PATH = "/World/CaptureMarker"
    _marker_t = None
    if ENABLE_CAPTURE_MARKER:
        _msphere = UsdGeom.Sphere.Define(stage, MARKER_PATH)
        _msphere.GetRadiusAttr().Set(2.5)
        _msphere.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.9, 0.1)])
        _mxf = UsdGeom.Xformable(_msphere.GetPrim())
        _mxf.ClearXformOpOrder()
        _marker_t = _mxf.AddTranslateOp()
    
    
    def _update_marker(p):
        if _marker_t is not None:
            _marker_t.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
    
    
    def _predict_capture():
        """맵 사다리 궤도(케플러)로 미래 위치 예측 -> 진행방향 뒤 standoff 지점.
        체이서 도달시간 lead_t = range/speed 만큼 앞선 위치를 인터셉트하도록."""
        cur_pos, _cur_vel = map_obstacle_state(_sim_t, ladder_def)
        range_est = float(np.linalg.norm(cur_pos - chaser_pos))
        lead_t = _clamp(range_est / max(CHASER_APPROACH_SPEED, 1e-6), 0.0, CATALOG_PREDICT_HORIZON_MAX)
        pred_pos, pred_vel = map_obstacle_state(_sim_t + lead_t, ladder_def)   # 케플러 정확 예측
        standoff_pt = pred_pos - _normalize(pred_vel) * STANDOFF_DISTANCE      # 진행방향 뒤
        return standoff_pt, pred_pos, lead_t, range_est
    
    
    def chaser_step(dt):
        """원거리 예측 인터셉트: 미래 capture(standoff) 지점으로 접근 + CBF 회피.
        카메라는 '실제' 사다리를 조준(perception 대비). (perception/토크 제어는 6단계)"""
        global chaser_pos, _sim_t
        _sim_t += dt
        ladder_now = _world_pos(ladder_path)                 # 실제 사다리(GT prim): 카메라 조준/회피판정
        standoff_pt, pred_pos, lead_t, range_est = _predict_capture()
        _update_marker(standoff_pt)                          # 잡을 위치 마커 갱신
    
        to = standoff_pt - chaser_pos                        # 현재 사다리가 아니라 '예측 standoff'로 향함
        d = float(np.linalg.norm(to))
        true_range = float(np.linalg.norm(ladder_now - chaser_pos))
        n_intr = 0; min_d = 1e9; dv = 0.0
        if d > 0.5:
            spd = min(CHASER_APPROACH_SPEED, d * 0.5 + 1.0)
            v_nom = (to / d) * spd
            if ENABLE_AVOIDANCE and obstacle_list and true_range > AVOID_DISABLE_RANGE:
                v_safe, diag = cbf_avoidance_velocity(chaser_pos, v_nom, obstacle_list, _sim_t, CBF_ALPHA)
                n_intr = sum(1 for (_g, _dm, _ts, _in) in diag if _in)
                min_d = min((_dm for (_g, _dm, _ts, _in) in diag), default=1e9)
                dv = float(np.linalg.norm(v_safe - v_nom))
            else:
                v_safe = v_nom
            chaser_pos = chaser_pos + v_safe * dt
        _set_chaser(chaser_pos, ladder_now)                  # 카메라는 실제 사다리 조준
        if int(_sim_t * 2.0) != int((_sim_t - dt) * 2.0):
            print(f"[CHASE] t={_sim_t:6.1f}s range={true_range:7.1f}m lead={lead_t:4.1f}s "
                  f"standoff_d={d:6.1f}m intr={n_intr} minD={min_d:6.1f}m dv={dv:4.1f}")
    
    
    _set_chaser(chaser_pos, _ladder0)
    print(f"[INTEGRATED] 체이서 생성: 사다리에서 ~400m 지점 {chaser_pos.round(1)}")
    
    # ------------------------------------------------------------
    # [4] CBF 회피: 장애물 = 맵 공전체(케플러 궤도) + 지구(정적 구)
    #   v11의 solve_cbf_qp / cbf_avoidance_velocity 를 그대로 이식하되,
    #   장애물 미래위치 함수만 맵 공전체 케플러로 교체(map_obstacle_state).
    # ------------------------------------------------------------
    import math
    import random
    
    CBF_ALPHA = 1.5
    CBF_BODY_HALF = 0.8 / 2.0           # 체이서 반체(0.8m)
    CBF_SAFE_EXTRA = 1.0                # 공전체 안전여유(m)
    EARTH_SAFE_EXTRA = 10.0             # 지구는 크게 여유
    AVOID_LOOKAHEAD_SEC = 8.0
    AVOID_PRED_SAMPLES = 32
    AVOID_INTRUSION_MARGIN = 2.0
    AVOID_DISABLE_RANGE = 5.0           # 타깃 이 거리 안이면 회피 OFF
    OBSTACLE_FALLBACK_RADIUS = 4.0      # bbox 못 구하면 쓸 보수적 반경(m)
    
    
    def rot_orbit_np(p, tilt_x, tilt_z):
        """맵의 rot_orbit(rot_z∘rot_x)을 numpy 벡터에 적용."""
        out = scene.rot_orbit((float(p[0]), float(p[1]), float(p[2])), tilt_x, tilt_z)
        return np.array(out, dtype=float)
    
    
    def map_obstacle_state(t, d):
        """공전체(또는 정적 지구)의 (월드위치, 월드속도) at 시각 t.
        d static=True -> 고정 중심/영속도. 아니면 케플러 phi(t)=phi0+omega*t."""
        if d.get("static"):
            return d["center"], np.zeros(3, dtype=float)
        r = float(d["r"]); w = float(d["omega"]); phi = d["phi0"] + w * t
        p = np.array([r * math.cos(phi), r * math.sin(phi), 0.0], dtype=float)
        v = np.array([-r * w * math.sin(phi), r * w * math.cos(phi), 0.0], dtype=float)
        p = rot_orbit_np(p, d["tilt"], d["tilt_z"])
        v = rot_orbit_np(v, d["tilt"], d["tilt_z"])
        return p, v
    
    
    def solve_cbf_qp(v_nom, A, b):
        """min 0.5||v-v_nom||^2 s.t. A v >= b. active-set 전수조사(2^m). numpy만."""
        v_nom = np.asarray(v_nom, dtype=float)
        m = A.shape[0]
        if m == 0:
            return v_nom.copy()
        if np.all(A @ v_nom - b >= -1e-9):
            return v_nom.copy()
        best_v = None
        best_cost = 1e18
        for mask in range(1, 1 << m):
            idx = [i for i in range(m) if (mask >> i) & 1]
            As = A[idx]; bs = b[idx]
            G = As @ As.T
            try:
                lam = np.linalg.solve(G, bs - As @ v_nom)
            except np.linalg.LinAlgError:
                continue
            if np.any(lam < -1e-9):
                continue
            v = v_nom + As.T @ lam
            if np.all(A @ v - b >= -1e-6):
                cost = float(np.dot(v - v_nom, v - v_nom))
                if cost < best_cost:
                    best_cost = cost
                    best_v = v
        return best_v if best_v is not None else v_nom.copy()
    
    
    def cbf_avoidance_velocity(p_c, v_nom, obstacle_list, t_now, alpha):
        """예측형(spacetime) CBF 안전필터. 반환 (v_safe, diag)."""
        A_rows = []; b_rows = []; diag = []
        taus = np.linspace(0.0, AVOID_LOOKAHEAD_SEC, AVOID_PRED_SAMPLES)
        for item in obstacle_list:
            d = item["def"]; d_safe = item["d_safe"]
            best_d = 1e18; tau_s = 0.0; p_o_s = None
            for tau in taus:
                p_o, _v_o = map_obstacle_state(t_now + tau, d)
                p_r = p_c + v_nom * tau
                dd = float(np.linalg.norm(p_r - p_o))
                if dd < best_d:
                    best_d = dd; tau_s = float(tau); p_o_s = p_o
            intruding = best_d < d_safe + AVOID_INTRUSION_MARGIN
            rel0 = p_c - p_o_s
            r_pred = rel0 + v_nom * tau_s
            g = float(np.dot(r_pred, r_pred)) - d_safe * d_safe
            if tau_s < 1e-3:
                p_o0, v_o0 = map_obstacle_state(t_now, d)
                rel = p_c - p_o0
                a = 2.0 * rel
                bb = 2.0 * float(np.dot(rel, v_o0)) - alpha * (float(np.dot(rel, rel)) - d_safe * d_safe)
            else:
                grad = 2.0 * tau_s * r_pred
                a = grad
                bb = float(np.dot(grad, v_nom)) - alpha * g
            na = float(np.linalg.norm(a))
            if na > 1e-9:
                a = a / na; bb = bb / na
            A_rows.append(a); b_rows.append(bb)
            diag.append((g, best_d, tau_s, intruding))
        A = np.array(A_rows, dtype=float); b = np.array(b_rows, dtype=float)
        v_safe = solve_cbf_qp(v_nom, A, b)
        sp_nom = float(np.linalg.norm(v_nom)); nv = float(np.linalg.norm(v_safe))
        if nv > sp_nom + 5.0:
            v_safe = v_safe / nv * (sp_nom + 5.0)
        return v_safe, diag
    
    
    def _world_radius(path):
        """prim 월드 bbox로 대략 반경(최대 반치수). 실패 시 폴백."""
        try:
            prim = stage.GetPrimAtPath(path)
            bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
            rng = bbox.ComputeAlignedRange()
            s = rng.GetSize()
            rad = 0.5 * max(float(s[0]), float(s[1]), float(s[2]))
            if rad > 1e-3 and rad < 1e7:
                return rad
        except Exception as e:
            print("[INTEGRATED][bbox] 반경 계산 실패:", path, e)
        return OBSTACLE_FALLBACK_RADIUS
    
    
    # --- 회피 장애물 리스트 구성: 타깃 사다리는 제외, 나머지 공전체 + 지구 ---
    obstacle_list = []
    for b in kinematic_bodies:
        if b["name"] == "DebrisLadder":
            continue  # 타깃은 회피 대상 아님
        obj_path = b.get("path")
        r_obs = _world_radius(obj_path) if obj_path else OBSTACLE_FALLBACK_RADIUS
        obstacle_list.append({"def": b, "d_safe": r_obs + CBF_BODY_HALF + CBF_SAFE_EXTRA, "name": b["name"]})
    
    _earth_center = _world_pos(handles["earth_path"])
    _earth_r = _world_radius(handles["earth_path"])
    obstacle_list.append({
        "def": {"static": True, "center": _earth_center},
        "d_safe": _earth_r + CBF_BODY_HALF + EARTH_SAFE_EXTRA,
        "name": "Earth",
    })
    print(f"[INTEGRATED] CBF 장애물 {len(obstacle_list)}개 (공전체 {len(obstacle_list)-1} + 지구). "
          f"지구중심 {_earth_center.round(1)} 반경~{_earth_r:.1f}m, alpha={CBF_ALPHA}")
    for it in obstacle_list:
        print(f"    - {it['name']}: d_safe={it['d_safe']:.2f}m")
    
    
    
    
    # ============================================================
    # 메인 루프
    # ============================================================
    # ------------------------------------------------------------
    # [6a] perception: 체이서 카메라 -> YOLO 탐지 + cv2 오버레이
    #   perception 함수는 adr_perception 모듈에서 import. (6b/6c에서 KeypointNet/PnP/ICP 추가)
    # ------------------------------------------------------------
import adr_perception as perc
try:
    import omni.replicator.core as rep
except Exception as _e:
    rep = None
    print("[INTEGRATED] replicator import 실패 - perception 비활성:", _e)

import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(THIS_DIR, "..", "resources", "models", "yolo_best.pt")
KEYPOINT_MODEL_PATH = os.path.join(THIS_DIR, "..", "resources", "models", "keypoint_best.pt")
KEYPOINTS_3D_PATH = os.path.join(THIS_DIR, "..", "resources", "datasets", "keypoints_3d.json")
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# CHASER_PATH는 enable_adr일 때만 정의됨 → 기본값 설정
if not config.enable_adr:
    CHASER_PATH = "/World/Chaser"  # 더미 (berthing-only 모드)
    THIRD_PERSON_CAM_PATH = "/World/ThirdPersonCam"
    ENABLE_PHYSICS = True

CHASER_CAM_PATH = CHASER_PATH + "/FrontCam"

# intrinsic / perception 초기화 (ADR 모드 전용)
_cam = None
FX = FY = CX = CY = 0.0
K_MATRIX = np.eye(3)
rgb_annotator = None
yolo_model = None
keypoint_model = None
keypoint_num_kp = 0
keypoint_status = "disabled"

if config.enable_adr:
    _cam = UsdGeom.Camera(stage.GetPrimAtPath(CHASER_CAM_PATH))
    _cam.CreateHorizontalApertureAttr(20.955)

    def _compute_intrinsics():
        focal = float(_cam.GetFocalLengthAttr().Get() or 24.0)
        h_ap = float(_cam.GetHorizontalApertureAttr().Get() or 20.955)
        fx = focal / h_ap * IMAGE_WIDTH
        cx = IMAGE_WIDTH / 2.0
        cy = IMAGE_HEIGHT / 2.0
        print(f"[INTRINSICS] focal={focal} h_ap={h_ap:.3f} fx=fy={fx:.1f} cx={cx} cy={cy}")
        return fx, fx, cx, cy

    FX, FY, CX, CY = _compute_intrinsics()
    K_MATRIX = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float64)

    # render product + rgb annotator (체이서 전방 카메라)
    if rep is not None:
        try:
            _rgb_rp = rep.create.render_product(CHASER_CAM_PATH, (IMAGE_WIDTH, IMAGE_HEIGHT))
            rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb_annotator.attach([_rgb_rp])
            print("[INTEGRATED] RGB render product 생성:", CHASER_CAM_PATH)
        except Exception as _e:
            print("[INTEGRATED] RGB render product 실패:", _e)

    yolo_model = perc.load_yolo_model(YOLO_MODEL_PATH)
    keypoint_model, keypoint_num_kp, keypoint_status = perc.load_keypoint_model(KEYPOINT_MODEL_PATH, KEYPOINTS_3D_PATH)
    print(f"[INTEGRATED] KeypointNet status={keypoint_status} num_kp={keypoint_num_kp}")
    perc.ensure_debug_dir()
    print(f"[INTEGRATED] YOLO 디버그 이미지: {perc.YOLO_LATEST_IMAGE_PATH} (이미지 뷰어로 열어두면 실시간 갱신)")

# ============================================================
# [6/V8] v11 on_update 이식: 페이즈머신 + KF + V8 텀블링추적 + PnP필터 + CBF
#   - 타깃 = 맵 DebrisLadder GT(위치 _world_pos + 자세 _world_R + 속도 프레임차분)
#   - 물리 OFF(키네마틱). LiDAR/ICP는 센서 미생성 -> 자동 비활성(6c에서 추가).
#   - V8 추적 입력 R/pos = GT (v11 동일). 추정(PnP) 전환은 후속 단계.
# ============================================================

def _world_R(path):
    """prim 월드 회전행렬(scale 제거 -> orthonormal). column = body축의 world방향(body->world).
       Gf.Matrix4d는 row-vector: 로컬축 i의 world방향 = row i 상위3성분."""
    prim = stage.GetPrimAtPath(path)
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    cols = []
    for i in range(3):
        r = m.GetRow(i)
        row = np.array([r[0], r[1], r[2]], dtype=float)
        n = float(np.linalg.norm(row))
        cols.append(row / n if n > 1e-9 else row)
    return np.column_stack(cols)


# V8 grasp geometry: 월드 사다리 실제 mesh 형상에서 장축(레일 길이방향)/끝/grasp를 직접 측정.
#   keypoints_3d는 v11 사다리 frame 기준이라, 월드 DebrisLadder의 _world_R(텀블+base_rot+궤도가
#   섞인 자세)로 변환하면 "장축 끝"이 사다리 진짜 끝이 아니라 측면을 가리킴(omega 측정이 v11과
#   다른 게 그 증거). -> 사다리 mesh 점을 _world_R 기준 로컬로 환산해 PCA로 끝을 직접 잡으면
#   R에 뭐가 섞이든 항상 레일 끝 중앙에 떨어진다.
def _measure_ladder_body_geometry(lpath, grasp_inset):
    try:
        R0 = _world_R(lpath)                 # 측정 시점(t≈0) 사다리 자세
        p0 = _world_pos(lpath)
        pts = []
        for pr in Usd.PrimRange(stage.GetPrimAtPath(lpath)):
            if not pr.IsA(UsdGeom.Mesh):
                continue
            pa = UsdGeom.Mesh(pr).GetPointsAttr().Get()
            if not pa:
                continue
            xf = UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for q in pa:
                wp = xf.Transform(Gf.Vec3d(float(q[0]), float(q[1]), float(q[2])))
                w = np.array([wp[0], wp[1], wp[2]], dtype=float)
                pts.append(R0.T @ (w - p0))   # 사다리 로컬(텀블/base 제거) — current_point와 동일 frame
        if len(pts) < 12:
            return None
        P = np.asarray(pts, dtype=float)
        if len(P) > 20000:                    # 점 너무 많으면 다운샘플(PCA용)
            P = P[np.random.default_rng(0).choice(len(P), 20000, replace=False)]
        c = P.mean(axis=0)
        ew, ev = np.linalg.eigh(np.cov((P - c).T))
        axis = ev[:, int(np.argmax(ew))]      # 최대분산축 = 사다리 길이방향(레일)
        proj = (P - c) @ axis
        tip_pos = c + axis * float(proj.max())
        tip_neg = c + axis * float(proj.min())
        grasp = c + axis * (float(proj.max()) - grasp_inset)
        return grasp, c, tip_pos, tip_neg, int(len(P))
    except Exception as _e:
        print("[INTEGRATED/V8 WARNING] 사다리 mesh 측정 실패:", _e)
        return None

if config.enable_adr:
    _meas = _measure_ladder_body_geometry(ladder_path, perc.V8_GRASP_INSET)
    if _meas is not None:
        V8_GRASP_BODY, V8_CENTROID_BODY, V8_TIP_POS_BODY, V8_TIP_NEG_BODY, _npts = _meas
        print(f"[INTEGRATED/V8] mesh측정 OK (npts={_npts}) — 사다리 실제 형상 기준")
    else:
        print("[INTEGRATED/V8] mesh측정 실패 -> keypoints_3d fallback (frame 어긋날 수 있음)")
        V8_GRASP_BODY, V8_CENTROID_BODY, V8_TIP_POS_BODY, V8_TIP_NEG_BODY = perc.load_v8_grasp_geometry(KEYPOINTS_3D_PATH)
    V8_APPROACH_BODY = perc.normalize(V8_TIP_POS_BODY - V8_CENTROID_BODY)
    V8_STANDOFF_BODY = V8_TIP_POS_BODY + V8_APPROACH_BODY * (perc.V8_BODY_GAP + perc.V8_BODY_HALF)
    _v8_ladder_long = float(np.linalg.norm(V8_TIP_POS_BODY - V8_TIP_NEG_BODY))
    print(f"[INTEGRATED/V8] body_gap={perc.V8_BODY_GAP:.2f} ladder_long={_v8_ladder_long:.3f}m "
          f"tip={V8_TIP_POS_BODY.round(3)} grasp={V8_GRASP_BODY.round(3)} standoff_body={V8_STANDOFF_BODY.round(3)}")

ENABLE_PERCEPTION = config.enable_adr
_PERC_EVERY = 2
_perc_frame = 0
state = {}  # ADR이 꺼져 있으면 빈 dict

if config.enable_adr:
    # 초기 KF: 맵 사다리 GT + 카탈로그 노이즈로 시드 (v11)
    _rng = np.random.default_rng(7)
    _ladder_now0 = _world_pos(ladder_path)
    _init_cat_pos = _ladder_now0 + _rng.normal(0.0, perc.CATALOG_POS_NOISE_STD, 3)
    _init_cat_vel = _rng.normal(0.0, perc.CATALOG_VEL_NOISE_STD, 3)
    _init_x = np.zeros(6); _init_x[0:3] = _init_cat_pos; _init_x[3:6] = _init_cat_vel
    _init_P = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

    state = {
        "prev_ladder_pos": _ladder_now0.copy(),
        "catalog_pos": _init_cat_pos.copy(),
        "catalog_vel": _init_cat_vel.copy(),
        "time_since_catalog_update": 0.0,
        "time_since_yolo_update": 0.0,
        "kf_x": _init_x.copy(),
        "kf_P": _init_P.copy(),
        "phase": perc.PHASE_CATALOG_GUIDANCE,
        "_phase_prev": perc.PHASE_CATALOG_GUIDANCE,
        "last_yolo_lock_valid": False, "last_yolo_lock_time": -999.0, "vision_lock_age": 999.0,
        "yolo_detected": False, "yolo_conf": 0.0, "yolo_cls_id": -1,
        "pnp_detected": False, "pnp_accepted": False, "pnp_reproj_err": -1.0,
        "pnp_pose_pos_error": -1.0, "pnp_pose_ang_error": -1.0, "pnp_reject_reason": "init",
        "pnp_keypoint_count": 0, "pnp_method": "init", "pnp_R": np.eye(3),
        "pnp_good_streak": 0, "pnp_miss_streak": 0, "pnp_hold_active": False,
        "last_good_pnp_valid": False, "last_good_pnp_time": -999.0,
        "last_good_pnp_pos": _ladder_now0.copy(),
        "last_good_pnp_R": np.eye(3), "last_good_pnp_capture_pos": _ladder_now0.copy(),
        "detector_pos": _ladder_now0.copy(), "lidar_projected_points": None,
        "control_source": "KF",
        "v8_estimator": perc.TumblingEstimator(window=perc.V8_EST_WINDOW),
        "v8_engaged": False, "v8_trigger_hold": 0.0, "v8_substage": "approach5",
        "v8_tracking_active": False, "v8_body_gap": -1.0, "v8_rel_speed": -1.0,
        "v8_grasp_now": np.zeros(3),
        "avoid_active": False, "avoid_intruders": 0, "avoid_dv": 0.0, "avoid_min_h": 999.0,
        "last_yolo_status": "init", "_dbg_R_logged": 0,
        # --- 물리(동적바디+토크자세) 런타임 상태 ---
        "chaser_vel_cmd": np.zeros(3),                       # 가속제한 속도명령(이전값)
        "chaser_vel_actual": np.zeros(3),                    # 물리 실제 선속도
        "chaser_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),  # 물리 실제 자세(매 프레임 phys_read로 갱신)
        "prev_omega": np.zeros(3),                           # 이전 각속도(토크 D항)
        "cmd_force": np.zeros(3), "cmd_torque": np.zeros(3), # 표시/플룸용 명령
        "active_nozzles": 0,                                 # 플룸: 이번 프레임 점화 노즐 수
    }


def _run_perception(camera_pos, R_cv_wc, predicted_kf_pos, true_pose_pos, true_pose_R,
                    kf_pos, kf_capture_pos, expected_camera_range):
    """v11 on_update의 YOLO->KeypointNet->PnP + accept/reject 필터 블록 그대로.
       상태 오버레이(status_text)는 v11과 동일: phase/yolo/est_range/v8/avoid (+v8_in).
       last_yolo_status는 화면에 안 넣고 state에 저장만(콘솔 로그에 출력) — v11 동일."""
    global _perc_frame
    if not (ENABLE_PERCEPTION and rgb_annotator is not None and yolo_model is not None and perc.cv2 is not None):
        state["last_yolo_status"] = "YOLO disabled | model/cv2 None"
        return
    _perc_frame += 1
    if _perc_frame % _PERC_EVERY != 0:
        return
    try:
        data = rgb_annotator.get_data()
    except Exception:
        data = None
    if data is None or len(data) == 0:
        state["last_yolo_status"] = "YOLO waiting | viewport capture not ready"
        return
    rgb_image = np.asarray(data)
    if rgb_image.ndim == 3 and rgb_image.shape[2] == 4:
        rgb_image = rgb_image[:, :, :3]
    if rgb_image.dtype != np.uint8:
        rgb_image = rgb_image.astype(np.uint8)
    rgb_image = np.ascontiguousarray(rgb_image)
    try:
        results = yolo_model.predict(source=rgb_image, imgsz=perc.YOLO_IMGSZ, conf=perc.YOLO_CONF_THRES, verbose=False)
        detection = perc.select_best_yolo_box(results)
        detected_keypoints = None
        projected_points = None
        state["yolo_detected"] = detection is not None
        state["pnp_accepted"] = False

        if detection is not None:
            state["yolo_conf"] = detection["conf"]
            state["yolo_cls_id"] = detection["cls_id"]
            state["last_yolo_lock_valid"] = True
            state["last_yolo_lock_time"] = _sim_t
            state["vision_lock_age"] = 0.0
            range_est = float(np.linalg.norm(kf_pos - camera_pos))
            detector_pos = perc.bbox_center_to_world_position(detection["xyxy"], camera_pos, R_cv_wc, range_est, FX, FY, CX, CY)
            state["detector_pos"] = detector_pos
            state["kf_x"], state["kf_P"] = perc.kalman_update_detector_position(state["kf_x"], state["kf_P"], detector_pos)

            image_points, object_points, detected_keypoints, keypoint_crop_box, keypoint_status = \
                perc.predict_keypoints_from_yolo_bbox(rgb_image, detection, keypoint_model)
            pnp_ok, pnp_pos, pnp_R, reproj_err, projected_points, pnp_method = \
                perc.estimate_pose_solvepnp(image_points, object_points, camera_pos, R_cv_wc, K_MATRIX)
            state["pnp_keypoint_count"] = len(detected_keypoints)
            state["pnp_method"] = f"{pnp_method}|{keypoint_status}"
            state["pnp_accepted"] = False

            if pnp_ok:
                pose_pos_error = float(np.linalg.norm(pnp_pos - true_pose_pos))
                pose_ang_error = perc.angle_error_deg(pnp_R, true_pose_R)
                raw_pnp_pos = pnp_pos.copy()
                raw_kf_pose_diff = float(np.linalg.norm(raw_pnp_pos - kf_pos))
                if perc.USE_HYBRID_KF_POSITION_PNP_ORIENTATION:
                    fused_pnp_pos = kf_pos.copy()
                    fused_pnp_capture_pos = perc.transform_local_to_world(fused_pnp_pos, pnp_R, perc.LOCAL_CAPTURE_OFFSET)
                    fused_pose_pos_error = float(np.linalg.norm(fused_pnp_pos - true_pose_pos))
                    fused_pose_ang_error = pose_ang_error
                else:
                    fused_pnp_pos = pnp_pos.copy()
                    fused_pnp_capture_pos = perc.transform_local_to_world(pnp_pos, pnp_R, perc.LOCAL_CAPTURE_OFFSET)
                    fused_pose_pos_error = pose_pos_error
                    fused_pose_ang_error = pose_ang_error
                capture_jump = float(np.linalg.norm(fused_pnp_capture_pos - kf_capture_pos))
                accepted = True
                reject_reason = "accepted_hybrid" if perc.USE_HYBRID_KF_POSITION_PNP_ORIENTATION else "accepted"
                if (not perc.USE_HYBRID_KF_POSITION_PNP_ORIENTATION) and raw_kf_pose_diff > perc.PNP_ACCEPT_KF_POS_DIFF_M:
                    accepted = False
                    reject_reason = f"reject_kf_diff_{raw_kf_pose_diff:.2f}m"
                elif capture_jump > perc.PNP_ACCEPT_CAPTURE_JUMP_M:
                    accepted = False
                    reject_reason = f"reject_capture_jump_{capture_jump:.2f}m"
                elif state["last_good_pnp_valid"]:
                    last_good_angle_jump = perc.rotation_angle_between_deg(state["last_good_pnp_R"], pnp_R)
                    if last_good_angle_jump > perc.PNP_ACCEPT_LAST_GOOD_ANGLE_JUMP_DEG:
                        accepted = False
                        reject_reason = f"reject_last_angle_jump_{last_good_angle_jump:.1f}deg"
                state["pnp_detected"] = bool(accepted)
                state["pnp_accepted"] = bool(accepted)
                state["pnp_R"] = pnp_R.copy()
                state["pnp_reproj_err"] = reproj_err
                state["pnp_pose_pos_error"] = fused_pose_pos_error
                state["pnp_pose_ang_error"] = fused_pose_ang_error
                state["pnp_reject_reason"] = reject_reason
                if accepted:
                    state["last_good_pnp_valid"] = True
                    state["last_good_pnp_time"] = _sim_t
                    state["last_good_pnp_pos"] = fused_pnp_pos.copy()
                    state["last_good_pnp_R"] = pnp_R.copy()
                    state["last_good_pnp_capture_pos"] = fused_pnp_capture_pos.copy()
                    state["pnp_good_streak"] += 1
                    state["pnp_miss_streak"] = 0
                    state["pnp_hold_active"] = True
                    state["last_yolo_status"] = (
                        f"YOLO+PnP accepted | conf={state['yolo_conf']:.2f} | reproj={reproj_err:.1f}px | "
                        f"ang_err={fused_pose_ang_error:.1f}deg | streak={state['pnp_good_streak']}")
                else:
                    state["pnp_miss_streak"] += 1
                    state["pnp_good_streak"] = 0
                    state["last_yolo_status"] = (
                        f"YOLO+PnP rejected | conf={state['yolo_conf']:.2f} | reproj={reproj_err:.1f}px | reason={reject_reason}")
            else:
                state["pnp_detected"] = False
                state["pnp_miss_streak"] += 1
                state["pnp_good_streak"] = 0
                state["last_yolo_status"] = f"YOLO ok but PnP not ready | {pnp_method} | {keypoint_status}"
        else:
            state["yolo_conf"] = 0.0
            state["pnp_detected"] = False
            state["pnp_keypoint_count"] = 0
            state["last_yolo_status"] = (
                f"YOLO running | no detection | phase={state['phase']} | expected_range={expected_camera_range:.1f}m")

        # ----- 상태 오버레이 (v11 status_lines 그대로: last_yolo_status는 화면에 안 넣음) -----
        status_lines = [
            f"phase={state['phase']}",
            f"yolo={state['yolo_detected']} conf={state['yolo_conf']:.2f}",
            f"est_range={expected_camera_range:.1f}m",
            f"v8={state['v8_tracking_active']} gap={state['v8_body_gap']:.2f} rel={state['v8_rel_speed']:.2f}",
            f"avoid={state['avoid_active']} intr={state['avoid_intruders']} dv={state['avoid_dv']:.2f} h={state['avoid_min_h']:.1f}",
        ]
        if state["v8_trigger_hold"] > 0.0 and not state["v8_engaged"]:
            _remain = max(0.0, perc.V8_TRIGGER_HOLD_SEC - state["v8_trigger_hold"])
            status_lines.append(f"v8_in={_remain:.1f}s")
        if ENABLE_PHYSICS and _chaser_rb is not None:
            _F = state["cmd_force"]; _T = state["cmd_torque"]; _w = state["prev_omega"]
            _att = "TORQUE" if ENABLE_TORQUE_ATTITUDE else "rate"
            status_lines.append(f"phys=ON att={_att} nozzles={state['active_nozzles']}/{len(_thruster_nozzles)}")
            status_lines.append(f"thrust|F|={np.linalg.norm(_F):.0f}N")
            status_lines.append(f"tau=({_T[0]:+.0f},{_T[1]:+.0f},{_T[2]:+.0f})Nm |T|={np.linalg.norm(_T):.0f}")
            status_lines.append(f"|w|={np.linalg.norm(_w):.2f}rad/s")
        status_text = " | ".join(status_lines)
        perc.save_debug_image(rgb_image, detection, status_text,
                              detected_keypoints, projected_points, state.get("lidar_projected_points"))
    except Exception as e:
        state["last_yolo_status"] = f"YOLO exception | {type(e).__name__}: {e}"
        print("[INTEGRATED WARNING] YOLO update failed.", e)

def on_update(dt):
    global chaser_pos, _sim_t
    # _sim_t is now incremented in the main loop
    t = _sim_t
    state["time_since_catalog_update"] += dt
    state["time_since_yolo_update"] += dt

    # ----- [물리] 체이서 실제 포즈를 물리에서 읽어 chaser_pos/자세 동기화 (v11). 이후 전 구간이 실제값 사용 -----
    if ENABLE_PHYSICS and _chaser_rb is not None:
        try:
            _cp, _cq, _cv = phys_read_chaser()
            chaser_pos = _cp
            state["chaser_quat_xyzw"] = _cq
            state["chaser_vel_actual"] = _cv
        except Exception as _rde:
            if int(t * 2.0) != int((t - dt) * 2.0):
                print("[INTEGRATED/PHYS] chaser read 실패:", _rde)

    # ----- PnP hold / vision lock 타이머 (v11) -----
    if state["last_good_pnp_valid"]:
        age = t - state["last_good_pnp_time"]
        state["pnp_hold_active"] = age <= perc.PNP_HOLD_TIME_SEC
        if not state["pnp_hold_active"]:
            state["last_good_pnp_valid"] = False
            state["pnp_good_streak"] = 0
    else:
        state["pnp_hold_active"] = False
    if state["last_yolo_lock_valid"]:
        state["vision_lock_age"] = t - state["last_yolo_lock_time"]
        if state["vision_lock_age"] > perc.VISION_LOCK_HOLD_SEC:
            state["last_yolo_lock_valid"] = False
    else:
        state["vision_lock_age"] = 999.0

    # ----- 타깃 GT: 맵 DebrisLadder 위치/자세/속도 -----
    true_pos = _world_pos(ladder_path)
    true_pose_R = _world_R(ladder_path)
    true_vel = (true_pos - state["prev_ladder_pos"]) / dt if dt > 1e-6 else np.zeros(3)
    state["prev_ladder_pos"] = true_pos.copy()

    # ----- KF predict + 카탈로그(GT+노이즈) update (v11) -----
    state["kf_x"], state["kf_P"] = perc.kalman_predict(state["kf_x"], state["kf_P"], dt)
    if state["time_since_catalog_update"] >= perc.CATALOG_UPDATE_PERIOD:
        noisy_pos = true_pos + _rng.normal(0.0, perc.CATALOG_POS_NOISE_STD, 3)
        noisy_vel = true_vel + _rng.normal(0.0, perc.CATALOG_VEL_NOISE_STD, 3)
        state["catalog_pos"] = noisy_pos
        state["catalog_vel"] = noisy_vel
        state["time_since_catalog_update"] = 0.0
        z = np.zeros(6); z[0:3] = noisy_pos; z[3:6] = noisy_vel
        state["kf_x"], state["kf_P"] = perc.kalman_update_catalog(state["kf_x"], state["kf_P"], z)
    else:
        state["catalog_pos"] = state["catalog_pos"] + state["catalog_vel"] * dt

    kf_pos = state["kf_x"][0:3]
    kf_vel = state["kf_x"][3:6]
    predicted_kf_pos, lead_t = perc.predict_catalog_target_position(kf_pos, kf_vel, chaser_pos)
    acquisition_point = perc.compute_standoff_point_from_predicted(predicted_kf_pos, kf_vel, perc.ACQUISITION_STANDOFF_DISTANCE)
    kf_capture_pos = perc.compute_standoff_point_from_predicted(predicted_kf_pos, kf_vel, perc.CAPTURE_DISTANCE)

    # ----- 페이즈 (v11) -----
    vision_recent = state["last_yolo_lock_valid"]
    pose_recent = state["last_good_pnp_valid"] and state["pnp_hold_active"]
    state["phase"] = perc.choose_mission_phase(predicted_kf_pos, chaser_pos, vision_recent, pose_recent)
    if state["phase"] != state["_phase_prev"]:
        print(f"[PHASE] t={t:6.1f}s {state['_phase_prev']:16s} -> {state['phase']:16s} "
              f"(vision={vision_recent} pose={pose_recent} yolo_conf={state['yolo_conf']:.2f})")
        state["_phase_prev"] = state["phase"]
    pnp_control_ready = (perc.USE_PNP_FOR_CONTROL and state["last_good_pnp_valid"]
                         and state["pnp_hold_active"] and state["pnp_good_streak"] >= perc.PNP_CONTROL_MIN_STREAK)

    # ----- V8 트리거: true_range<=10m가 3초 -> engage. estimator에 GT 자세 먹임 (v11) -----
    true_range = float(np.linalg.norm(true_pos - chaser_pos))
    if true_range <= perc.V8_TRIGGER_RANGE:
        state["v8_trigger_hold"] += dt
    else:
        state["v8_trigger_hold"] = 0.0
    if (not state["v8_engaged"]) and state["v8_trigger_hold"] >= perc.V8_TRIGGER_HOLD_SEC:
        state["v8_engaged"] = True
        state["v8_substage"] = "approach5"
        print(f"[INTEGRATED] >>> V8 ENGAGED (range={true_range:.2f}m stable {state['v8_trigger_hold']:.1f}s) -> 5m 웨이포인트 <<<")
    if state["v8_engaged"]:
        state["v8_estimator"].update(t, true_pose_R, true_pos)
    v8_track_ready = state["v8_engaged"] and state["v8_estimator"].ready()

    # 사다리 R 검증 로그 (V8 시작 직후 몇 번): orthonormal + 각속도=(0.225,0.16,0.275) 확인
    if state["_dbg_R_logged"] < 3 and v8_track_ready:
        om = state["v8_estimator"].omega()
        orth = float(np.max(np.abs(true_pose_R @ true_pose_R.T - np.eye(3))))
        print(f"[DBG/R] |R@Rt - I|max={orth:.2e}  omega(rad/s)={np.round(om,4)}  (v11=0.225,0.16,0.275)")
        state["_dbg_R_logged"] += 1

    # ----- selected_capture_pos 결정 (v11) -----
    v_ff = np.zeros(3)
    tip_pos_now = tip_neg_now = None
    if v8_track_ready:
        est = state["v8_estimator"]
        grasp_now = est.current_point(V8_GRASP_BODY)
        tip_pos_now = est.current_point(V8_TIP_POS_BODY)
        tip_neg_now = est.current_point(V8_TIP_NEG_BODY)
        standoff_tar = est.predict_point(V8_STANDOFF_BODY, perc.V8_LEAD_SEC)
        _wp_body = V8_TIP_POS_BODY + V8_APPROACH_BODY * perc.V8_WAYPOINT_RANGE
        waypoint5 = est.predict_point(_wp_body, perc.V8_LEAD_SEC)
        if state["v8_substage"] == "approach5":
            if float(np.linalg.norm(chaser_pos - waypoint5)) < perc.V8_WAYPOINT_REACH:
                state["v8_substage"] = "creep"
                print("[INTEGRATED] V8: 5m 웨이포인트 도달 -> 레일끝으로 creep")
            selected_capture_pos = waypoint5
            v_ff = est.point_velocity(_wp_body)
        else:
            selected_capture_pos = standoff_tar
            v_ff = est.point_velocity(V8_STANDOFF_BODY)
        state["control_source"] = "V8_PNP_STANDOFF_TRACK"
        state["v8_tracking_active"] = True
        state["v8_grasp_now"] = grasp_now.copy()
    elif state["phase"] in (perc.PHASE_CATALOG_GUIDANCE, perc.PHASE_FOV_ACQUISITION):
        selected_capture_pos = acquisition_point
        state["control_source"] = state["phase"]
        state["v8_tracking_active"] = False
    elif pnp_control_ready:
        selected_capture_pos = state["last_good_pnp_capture_pos"]
        state["control_source"] = "PNP_HELD"
        state["v8_tracking_active"] = False
    else:
        selected_capture_pos = kf_capture_pos
        state["control_source"] = "KF_VISION_TRACK"
        state["v8_tracking_active"] = False

    # ----- chaser_vel (v11) -----
    to_capture = selected_capture_pos - chaser_pos
    selected_capture_error = float(np.linalg.norm(to_capture))
    if v8_track_ready:
        _cap = perc.V8_APPROACH_SPEED if state["v8_substage"] == "approach5" else perc.V8_CREEP_SPEED
        corr = perc.limit_vector(perc.V8_KP * (selected_capture_pos - chaser_pos), _cap)
        chaser_vel = corr + v_ff
        chaser_vel = perc.limit_vector(chaser_vel, perc.V8_MAX_CHASER_SPEED)
        clearance = perc.point_to_segment(chaser_pos, tip_pos_now, tip_neg_now)
        state["v8_body_gap"] = float(clearance - perc.V8_BODY_HALF)
        state["v8_rel_speed"] = float(np.linalg.norm(chaser_vel - v_ff))
    else:
        # 상대속도 매칭: kf_vel(타깃 공전속도 보상 = feedforward) + glideslope 접근(거리비례).
        #   타깃 기준 상대 접근속도 = approach_speed만 남아 9.3m 평형이 사라짐.
        #   멀리선 approach_speed가 18로 포화돼 기존처럼 빠르고, 가까이선 감속하되
        #   kf_vel이 공전을 상쇄해 standoff까지 끝까지 접근 -> true_range 10m 통과 -> V8 점화.
        approach_speed = perc.compute_slowdown_speed(selected_capture_error)
        chaser_vel = kf_vel + perc.normalize(to_capture) * approach_speed
        chaser_vel = perc.limit_vector(chaser_vel, perc.MAX_APPROACH_SPEED)
        state["v8_body_gap"] = -1.0
        state["v8_rel_speed"] = float(approach_speed)   # 계측: 상대 접근속도(>0=접근, 0=평형)

    # ----- CBF 회피 (맵 케플러 장애물). 타깃 AVOID_DISABLE_RANGE 안이면 OFF (v11) -----
    state["avoid_active"] = False; state["avoid_intruders"] = 0
    state["avoid_dv"] = 0.0; state["avoid_min_h"] = 999.0
    if ENABLE_AVOIDANCE and obstacle_list and true_range > AVOID_DISABLE_RANGE:
        v_safe, diag = cbf_avoidance_velocity(chaser_pos, chaser_vel, obstacle_list, t, CBF_ALPHA)
        state["avoid_dv"] = float(np.linalg.norm(v_safe - chaser_vel))
        state["avoid_min_h"] = min((d[0] for d in diag), default=999.0)
        state["avoid_intruders"] = int(sum(1 for d in diag if d[3]))
        state["avoid_active"] = state["avoid_dv"] > 1e-4
        chaser_vel = v_safe

    # ----- 제어: 물리(가속제한 속도 + 토크 자세) or 키네마틱(폴백) -----
    if ENABLE_PHYSICS and _chaser_rb is not None:
        # v8/KF가 낸 desired velocity를 가속도 상한 안에서 추종(오버슈트 X). 실제 이동은 다음 sim_ctx.step.
        v_prev = state["chaser_vel_cmd"]
        v_cmd = v_prev + perc.limit_vector(chaser_vel - v_prev, PHYS_MAX_ACCEL * dt)
        state["chaser_vel_cmd"] = v_cmd
        state["cmd_force"] = CHASER_MASS * (v_cmd - v_prev) / max(dt, 1e-3)
        try:
            phys_set_linvel(v_cmd)
        except Exception as _le:
            if int(t * 2.0) != int((t - dt) * 2.0):
                print("[INTEGRATED/PHYS] linvel 실패:", _le)
        # 자세: 카메라(-Z)가 pointing_target 보게. make_camera_rotation_from_forward = -Z 전방(=_set_chaser와 동일 자세).
        _pt = state["v8_grasp_now"] if v8_track_ready else predicted_kf_pos
        q_des = perc.rotmat_to_quat_xyzw(perc.make_camera_rotation_from_forward(_pt - chaser_pos)[0])
        q_cur = state["chaser_quat_xyzw"]
        if ENABLE_TORQUE_ATTITUDE:
            q_errv = quat_align_angvel(q_cur, q_des, 0.5)              # 쿼터니언 오차 벡터부
            try:
                omega_act = phys_get_angvel()
            except Exception:
                omega_act = state["prev_omega"]
            tau = perc.limit_vector(ATT_KP_TORQUE * q_errv - ATT_KD_TORQUE * omega_act, ATT_MAX_TORQUE)
            try:
                phys_apply_torque(tau)
            except Exception as _te:
                if int(t * 2.0) != int((t - dt) * 2.0):
                    print("[INTEGRATED/PHYS] torque 실패:", _te)
            state["cmd_torque"] = tau
            state["prev_omega"] = omega_act
        else:
            omega = perc.limit_vector(quat_align_angvel(q_cur, q_des, PHYS_ATT_KP), PHYS_MAX_ANGVEL)
            try:
                phys_set_angvel(omega)
            except Exception:
                pass
            _I_est = (1.0 / 6.0) * CHASER_MASS * (CHASER_BODY_SIZE ** 2)        # 암묵 토크 추정(표시/플룸용)
            state["cmd_torque"] = _I_est * (omega - state["prev_omega"]) / max(dt, 1e-3)
            state["prev_omega"] = omega
        # 추력기 플룸: 동적 추력(cmd_force)·토크(cmd_torque) 방향으로 분사 (실제 바디 자세 기준)
        if ENABLE_THRUSTER_FX:
            _R_now = quat_xyzw_to_rotmat(state["chaser_quat_xyzw"])
            state["active_nozzles"] = update_thruster_fx(chaser_pos, _R_now,
                                                         state["cmd_force"], state["cmd_torque"], dt)
        # 비관통은 콜라이더가 막음. v8_body_gap은 위 v8 분기에서 측정값이 들어가 있음.
        new_chaser_pos = chaser_pos                       # 실제 이동은 물리(다음 스텝). 마커/카메라는 현재값.
    else:
        # ----- 키네마틱(폴백): 위치 적분 + V8 비충돌 클램프 -----
        new_chaser_pos = chaser_pos + chaser_vel * dt
        if v8_track_ready:
            ab = tip_neg_now - tip_pos_now
            tt = float(np.clip(np.dot(new_chaser_pos - tip_pos_now, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0))
            closest = tip_pos_now + tt * ab
            out = new_chaser_pos - closest
            d_seg = float(np.linalg.norm(out))
            min_d = perc.V8_MIN_GAP + perc.V8_BODY_HALF
            if d_seg < min_d:
                out_dir = out / d_seg if d_seg > 1e-6 else perc.normalize(chaser_pos - closest)
                new_chaser_pos = closest + out_dir * min_d
                state["v8_body_gap"] = float(min_d - perc.V8_BODY_HALF)
    chaser_pos = new_chaser_pos

    # ----- 3인칭 관전 카메라: 체이서 뒤(사다리 반대)+위 대각선에서 둘 사이(mid) 바라봄 -----
    _tp_tgt = state["v8_grasp_now"] if v8_track_ready else true_pos    # 추적 중이면 grasp, 아니면 사다리 중심
    _tp_mid = 0.5 * (chaser_pos + _tp_tgt)
    _tp_back = perc.normalize(chaser_pos - _tp_tgt)                    # 사다리 반대(체이서 뒤) 방향
    _tp_span = max(float(np.linalg.norm(chaser_pos - _tp_tgt)), _v8_ladder_long, 5.0)
    _tp_eye = _tp_mid + _tp_back * (_tp_span * 0.9) + np.array([0.0, 0.0, _tp_span * 0.7], dtype=float)
    _tp_view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(float(_tp_eye[0]), float(_tp_eye[1]), float(_tp_eye[2])),
        Gf.Vec3d(float(_tp_mid[0]), float(_tp_mid[1]), float(_tp_mid[2])),
        Gf.Vec3d(0.0, 0.0, 1.0))
    _tp_cam_top.Set(_tp_view.GetInverse())
    # ----- 카메라: 위치=체이서+전방offset, 조준=predicted_kf_pos(+FOV 스캔). GT 아님(자로 조준) (v11) -----
    pointing_target = state["v8_grasp_now"] if v8_track_ready else predicted_kf_pos
    R_chaser_wc = perc.forward_to_rotmat_chaser(pointing_target - chaser_pos)
    expected_camera_range = float(np.linalg.norm(predicted_kf_pos - chaser_pos))
    if state["phase"] == perc.PHASE_FOV_ACQUISITION and not vision_recent:
        search_offset, _az, _el = perc.compute_camera_search_offset(R_chaser_wc, max(expected_camera_range, 1.0), t)
        camera_look_target = predicted_kf_pos + search_offset
    else:
        camera_look_target = predicted_kf_pos
    # FrontCam은 체이서(eye=chaser_pos)의 자식이고 -Z(전방)로 CAM_FWD_OFFSET 이동돼 있음.
    if ENABLE_PHYSICS and _chaser_rb is not None:
        # 물리: 자세는 토크가 소유 -> _set_chaser 호출 안 함(호출 시 물리 자세를 덮어씀).
        #   카메라/PnP는 물리 실제 자세에서 유도 -> 토크 슬루 중에도 렌더 FrontCam과 PnP 수식이 일치.
        R_usd_wc = quat_xyzw_to_rotmat(state["chaser_quat_xyzw"])   # 실제 바디(=FrontCam) 자세
        camera_forward = perc.normalize(-R_usd_wc[:, 2])           # 카메라 로컬 -Z의 월드 방향
        camera_pos = chaser_pos + camera_forward * CAM_FWD_OFFSET
        R_cv_wc = R_usd_wc @ np.diag([1.0, -1.0, -1.0])            # USD(-Z fwd,+Y up) -> CV(+Z fwd,+Y down)
    else:
        # 키네마틱: 즉시 look-at으로 바디 자세 세팅(카메라도 따라옴), commanded=actual.
        _set_chaser(chaser_pos, camera_look_target)
        camera_forward = perc.normalize(camera_look_target - chaser_pos)
        camera_pos = chaser_pos + camera_forward * CAM_FWD_OFFSET
        _R_usd, R_cv_wc = perc.make_camera_rotation_from_forward(camera_forward)

    # ----- perception: YOLO -> KeypointNet -> PnP -> 필터 + 오버레이 -----
    if state["time_since_yolo_update"] >= perc.YOLO_UPDATE_PERIOD:
        state["time_since_yolo_update"] = 0.0
        _run_perception(camera_pos, R_cv_wc, predicted_kf_pos, true_pos, true_pose_R, kf_pos, kf_capture_pos, expected_camera_range)

    # ----- 콘솔 로그 (0.5s 간격) -----
    if int(t * 2.0) != int((t - dt) * 2.0):
        _kf_range = float(np.linalg.norm(kf_pos - chaser_pos))
        print(f"[V8] t={t:6.1f}s phase={state['phase']:16s} ctrl={state['control_source']:20s} "
              f"true={true_range:6.1f}m kf={_kf_range:6.1f}m sel={selected_capture_error:6.1f}m "
              f"sub={state['v8_substage']:8s} v8={state['v8_tracking_active']} "
              f"gap={state['v8_body_gap']:5.2f} rel={state['v8_rel_speed']:5.2f} "
              f"yolo={state['yolo_detected']} conf={state['yolo_conf']:.2f} vage={state['vision_lock_age']:5.1f}s "
              f"pnp={state['pnp_accepted']} avoid={state['avoid_active']} dv={state['avoid_dv']:.2f}\n"
              f"     status={state['last_yolo_status']}")


# ============================================================
# [물리 B] USD 스키마: 체이서=동적+질량+박스콜라이더, 사다리=키네마틱+메시콜라이더
#   space_env가 사다리 ROOT(ladder_path)를 스케일 없는 깨끗한 xform으로 만들어둠 -> 리지드바디 OK.
#   스케일/메시는 ladder_path/Model 아래. 공전+텀블은 orbit_step이 set_transform으로 계속 구동(키네마틱).
# ============================================================
if ENABLE_PHYSICS:
    try:
        from pxr import Sdf, UsdPhysics, PhysxSchema
        # --- 체이서: 동적 리지드바디 + 500kg + 박스 콜라이더 ---
        _chaser_prim_p = stage.GetPrimAtPath(Sdf.Path(CHASER_PATH))
        _rb = UsdPhysics.RigidBodyAPI.Apply(_chaser_prim_p)
        _rb.CreateRigidBodyEnabledAttr(True)
        _rb.CreateKinematicEnabledAttr(False)
        _mass = UsdPhysics.MassAPI.Apply(_chaser_prim_p)
        _mass.CreateMassAttr(float(CHASER_MASS))
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(Sdf.Path(CHASER_PATH + "/Body")))   # Cube -> 박스
        PhysxSchema.PhysxRigidBodyAPI.Apply(_chaser_prim_p)
        print(f"[INTEGRATED/PHYS] 체이서 동적바디 질량={CHASER_MASS}kg + 박스콜라이더")
        # --- 사다리: 키네마틱 리지드바디 (set_transform 공전/텀블 유지 + 충돌 참여, 안 밀림) ---
        _ladder_prim_p = stage.GetPrimAtPath(Sdf.Path(ladder_path))
        _lrb = UsdPhysics.RigidBodyAPI.Apply(_ladder_prim_p)
        _lrb.CreateRigidBodyEnabledAttr(True)
        _lrb.CreateKinematicEnabledAttr(True)
        _ncol = 0
        for _p in Usd.PrimRange(_ladder_prim_p):
            if _p.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(_p)
                UsdPhysics.MeshCollisionAPI.Apply(_p).CreateApproximationAttr().Set("none")   # 삼각메시(키네마틱이라 OK)
                _ncol += 1
        print(f"[INTEGRATED/PHYS] 사다리 키네마틱바디 + 메시콜라이더 {_ncol}개")
    except Exception as _pe:
        print(f"[INTEGRATED/PHYS WARNING] USD 물리 스키마 실패 -> 키네마틱 폴백: {type(_pe).__name__}: {_pe}")
        ENABLE_PHYSICS = False


# ============================================================
# [물리 F] 추력기 플룸: 노즐 배치 + 파티클 풀 (절차적 jet, 월드 프림) — v11 그대로
#   병진 추력기 6면 + 자세(요) 접선 제트 4개. 반작용력 = -배기방향(-e_b).
# ============================================================
_thruster_nozzles = []      # [(r_b[3], e_b[3])]  e_b = 배기방향, 반작용력 = -e_b
_thruster_particles = []    # 노즐별 파티클 dict 리스트
if ENABLE_PHYSICS and ENABLE_THRUSTER_FX:
    try:
        _h = perc.V8_BODY_HALF
        _nozzle_defs = [
            # 병진(face) 추력기: 배기 = 바깥 -> 반작용으로 반대로 가속
            ((+_h * 1.1, 0, 0), (+1, 0, 0)), ((-_h * 1.1, 0, 0), (-1, 0, 0)),
            ((0, +_h * 1.1, 0), (0, +1, 0)), ((0, -_h * 1.1, 0), (0, -1, 0)),
            ((0, 0, +_h * 1.1), (0, 0, +1)), ((0, 0, -_h * 1.1), (0, 0, -1)),
            # 자세(요) 커플용 접선 제트 (회전 시각화)
            ((+_h * 1.1, +_h * 0.7, 0), (0, +1, 0)), ((-_h * 1.1, -_h * 0.7, 0), (0, -1, 0)),
            ((+_h * 1.1, -_h * 0.7, 0), (0, -1, 0)), ((-_h * 1.1, +_h * 0.7, 0), (0, +1, 0)),
        ]
        UsdGeom.Xform.Define(stage, "/World/chaser_thruster_fx")
        for _ni, (_rb, _eb) in enumerate(_nozzle_defs):
            _thruster_nozzles.append((np.array(_rb, float), perc.normalize(np.array(_eb, float))))
            _plist = []
            for _pj in range(THR_PARTICLES_PER_NOZZLE):
                _sph = UsdGeom.Sphere.Define(stage, f"/World/chaser_thruster_fx/n{_ni}_p{_pj}")
                _sph.CreateRadiusAttr(THR_PARTICLE_R)
                _top = _sph.AddTranslateOp()
                _sop = _sph.AddScaleOp()
                _imgp = UsdGeom.Imageable(_sph.GetPrim())
                _imgp.MakeInvisible()
                _plist.append({"img": _imgp, "t": _top, "s": _sop,
                               "c": _sph.GetDisplayColorAttr(),
                               "life": random.uniform(0.0, 1.0), "vis": False})
            _thruster_particles.append(_plist)
        print(f"[INTEGRATED/FX] 추력기 플룸 생성: 노즐 {len(_thruster_nozzles)} x 파티클 {THR_PARTICLES_PER_NOZZLE}")
    except Exception as _fe:
        print(f"[INTEGRATED/FX WARNING] 추력기 파티클 생성 실패: {type(_fe).__name__}: {_fe}")
        _thruster_nozzles = []


def update_thruster_fx(chaser_pos, R, F_world, T_world, dt):
    """동적 추력(F_world)·토크(T_world) 방향에 기여하는 노즐에서 플룸 분사. 점화 노즐 수 반환."""
    if not _thruster_nozzles:
        return 0
    _n_active = 0
    for _ni, (r_b, e_b) in enumerate(_thruster_nozzles):
        e_w = R @ e_b                                   # 배기 방향(월드)
        r_w = R @ r_b                                   # 레버암(월드)
        a_f = max(0.0, float(np.dot(-e_w, F_world))) / THR_FORCE_REF
        a_t = max(0.0, float(np.dot(np.cross(r_w, -e_w), T_world))) / THR_TORQUE_REF
        act = min(1.0, a_f + a_t)                       # 이 노즐 점화 강도(원하는 힘/토크에 기여할 때만 >0)
        plist = _thruster_particles[_ni]
        if act < 0.05:
            for p in plist:
                if p["vis"]:
                    p["img"].MakeInvisible(); p["vis"] = False
            continue
        _n_active += 1
        nozzle_w = chaser_pos + r_w
        plume = THR_PLUME_LEN * act
        for p in plist:
            p["life"] += dt * random.uniform(2.5, 4.5)
            if p["life"] > 1.0:
                p["life"] = random.uniform(0.0, 0.15)
            tt = p["life"]
            jit = np.array([random.uniform(-0.03, 0.03) for _ in range(3)])
            pos = nozzle_w + e_w * (tt * plume) + jit    # 배기 방향으로 솟아나감
            p["t"].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
            sc = max(0.35, 1.0 - tt * 0.6) * (0.75 + 0.25 * act)
            p["s"].Set(Gf.Vec3f(sc, sc, sc))
            g = max(0.0, 0.9 - tt * 1.2)                 # 노랑->주황->빨강
            p["c"].Set([(1.0, g, 0.0)])
            if not p["vis"]:
                p["img"].MakeVisible(); p["vis"] = True
    return _n_active


# ============================================================
# [물리 C] 런타임: SimulationContext(gravity=0) + RigidPrim(체이서) + 초기포즈 + phys 헬퍼
#   ★ Isaac 5.1 물리 API 지점 (v11에서 검증된 호출 그대로). 실패 시 키네마틱 폴백.
# ============================================================
_sim_ctx = None
_chaser_rb = None
if ENABLE_PHYSICS:
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim
        _sim_ctx = World.instance()
        if _sim_ctx is None:
            _sim_ctx = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, stage_units_in_meters=1.0)
        _sim_ctx.get_physics_context().set_gravity(0.0)                      # 중력 0 (우주)
        
        if config.enable_adr:
            _chaser_rb = RigidPrim(prim_paths_expr=CHASER_PATH, name="adr_chaser_rb")
            
        _sim_ctx.reset()

        if config.enable_adr:
            # 체이서 초기 포즈: chaser_pos에 두고 _ladder0 바라보게 (부팅 _set_chaser와 동일 자세 -> -Z 전방)
            _q0 = perc.rotmat_to_quat_xyzw(perc.make_camera_rotation_from_forward(_ladder0 - chaser_pos)[0])
            _q0_wxyz = np.array([[_q0[3], _q0[0], _q0[1], _q0[2]]], dtype=float)   # xyzw -> wxyz
            _chaser_rb.set_world_poses(positions=np.array([chaser_pos], dtype=float), orientations=_q0_wxyz)
            print(f"[INTEGRATED/PHYS] SimulationContext + RigidPrim 준비. gravity=0, 체이서 시작={np.round(chaser_pos,1)}")
        else:
            print("[INTEGRATED/PHYS] SimulationContext 준비. gravity=0 (Berthing Only)")
    except Exception as _re:
        print(f"[INTEGRATED/PHYS WARNING] 물리 런타임 초기화 실패 -> 키네마틱 폴백: {type(_re).__name__}: {_re}")
        _sim_ctx = None
        _chaser_rb = None
        ENABLE_PHYSICS = False


def phys_read_chaser():
    pos, quat_wxyz = _chaser_rb.get_world_poses()
    vel = _chaser_rb.get_linear_velocities()
    p = np.asarray(pos[0], float)
    qw = np.asarray(quat_wxyz[0], float)                       # wxyz (Isaac core 관례)
    q_xyzw = np.array([qw[1], qw[2], qw[3], qw[0]], float)
    return p, q_xyzw, np.asarray(vel[0], float)


def phys_set_linvel(vel_world):
    _chaser_rb.set_linear_velocities(np.asarray(vel_world, float).reshape(1, 3))


def phys_set_angvel(omega_world):
    _chaser_rb.set_angular_velocities(np.asarray(omega_world, float).reshape(1, 3))


def phys_get_angvel():
    return np.asarray(_chaser_rb.get_angular_velocities()[0], float)


def phys_apply_torque(torque_world):
    _chaser_rb.apply_forces_and_torques_at_pos(torques=np.asarray(torque_world, float).reshape(1, 3), is_global=True)


# ============================================================
# 메인 루프
# ============================================================
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(30):
    if ENABLE_PHYSICS and _sim_ctx is not None:
        _sim_ctx.step(render=True)
    else:
        simulation_app.update()

# ------------------------------------------------------------
# 체이서캠 전용 뷰포트(2번째). 기존(왼쪽) 뷰포트는 Perspective 유지.
#   -> 퍼스펙티브를 마음껏 돌려도 YOLO 입력은 FrontCam render product에 고정이라 안 흔들림.
#   -> 카메라 화면 밝기는 FrontCam에 단 헤드라이트(DistantLight)가 담당(RGB에도 반영).
#   (play+update 후 생성: boot 직후 뷰포트 생성 시 크래시 회피)
# ------------------------------------------------------------
try:
    from omni.kit.viewport.utility import create_viewport_window
    _cam_vp_win = create_viewport_window("Chaser FrontCam", width=720, height=405)
    try:
        _cam_vp_win.viewport_api.set_active_camera(CHASER_CAM_PATH)
    except Exception:
        _cam_vp_win.viewport_api.camera_path = CHASER_CAM_PATH
    print("[INTEGRATED] 체이서캠 뷰포트 생성:", CHASER_CAM_PATH, "(퍼스펙티브와 분리)")
except Exception as _ve:
    print("[INTEGRATED] 체이서캠 뷰포트 자동생성 실패 -> 수동으로 뷰포트 추가 후 카메라 선택:", CHASER_CAM_PATH, "| 사유:", _ve)

# 3인칭 관전 뷰포트(3번째): 체이서·사다리가 어떻게 붙어 도는지 대각선 위에서.
try:
    from omni.kit.viewport.utility import create_viewport_window as _cvw3
    _tp_vp_win = _cvw3("3rd Person (Chaser + Ladder)", width=720, height=405)
    try:
        _tp_vp_win.viewport_api.set_active_camera(THIRD_PERSON_CAM_PATH)
    except Exception:
        _tp_vp_win.viewport_api.camera_path = THIRD_PERSON_CAM_PATH
    print("[INTEGRATED] 3인칭 관전 뷰포트 생성:", THIRD_PERSON_CAM_PATH)
except Exception as _ve2:
    print("[INTEGRATED] 3인칭 뷰포트 자동생성 실패 -> 수동으로 뷰포트 추가 후 카메라 선택:", THIRD_PERSON_CAM_PATH, "| 사유:", _ve2)

print(f"\n>>> [INTEGRATED V8] 페이즈머신 + KF + V8 텀블링추적 + PnP필터 + CBF "
      f"({'물리 ON (동적바디+토크자세)' if (ENABLE_PHYSICS and _sim_ctx is not None) else '물리 OFF/키네마틱'}) <<<")
print(f">>> YOLO 오버레이: {perc.YOLO_LATEST_IMAGE_PATH} 를 이미지 뷰어로 열어두면 실시간 갱신 <<<")
print(">>> [V8]: phase/ctrl/range/substage/gap/rel/yolo/pnp/avoid <<<")

_orbit_rings_hidden = False
HIDE_ORBIT_RINGS_AFTER = 10.0

if '_sim_t' not in locals():
    _sim_t = 0.0

try:
    while simulation_app.is_running():
        try:
            dt = 1.0 / 60.0
            
            # _sim_t is updated globally here so both ADR and Berthing can use it
            _sim_t += dt
            
            if orbit_step is not None:
                orbit_step(dt)
            if config.enable_adr:
                on_update(dt)

            if config.enable_berthing:
                iss_berthing.step_berthing(dt)
                
            if (not _orbit_rings_hidden) and _sim_t > HIDE_ORBIT_RINGS_AFTER:
                _rings = stage.GetPrimAtPath(scene.ORBIT_PATHS_PATH)
                if _rings and _rings.IsValid():
                    UsdGeom.Imageable(_rings).MakeInvisible()
                    _orbit_rings_hidden = True
                    print("[INTEGRATED] 궤도 링(점들) 숨김 (10s 경과)")
            if ENABLE_PHYSICS and _sim_ctx is not None:
                _sim_ctx.step(render=True)         # 물리 스텝 + 렌더 (체이서 동역학 + 키네마틱 사다리)
            else:
                simulation_app.update()
                
            # ROS 2 Spin
            if rclpy.ok():
                rclpy.spin_once(ros_node, timeout_sec=0.0)
        except Exception as e:
            import traceback
            print(f"\n[INTEGRATED] Exception in main loop: {e}")
            traceback.print_exc()
            break
except KeyboardInterrupt:
    print("[INTEGRATED] 사용자 중단")
finally:
    simulation_app.close()