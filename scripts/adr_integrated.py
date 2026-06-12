# scripts/adr_integrated.py
# ============================================================
# ADR 통합 진입점 (R2D2 블랙홀 회오리 흡입 + 구깃구깃 애니메이션)
# ============================================================
# [ROS 통합] SimulationApp 로드 전에 ROS 2 환경 충돌 방지 + C++ 바인딩 경로 주입
import sys, os
# 터미널에 꼬여있는 시스템 ROS(Humble/Jazzy) 경로 제거 → Isaac 전용 rclpy 보호
sys.path = [p for p in sys.path if '/opt/ros' not in p and 'jazzy_ws' not in p and 'humble_ws' not in p]
if 'PYTHONPATH' in os.environ:
    os.environ['PYTHONPATH'] = ':'.join([p for p in os.environ['PYTHONPATH'].split(':') if '/opt/ros' not in p and 'jazzy_ws' not in p and 'humble_ws' not in p])
# mission_interfaces C++ .so 로드 위해 LD_LIBRARY_PATH 추가 후 즉시 재시작(execv)
WORKSPACE_DIR_TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_DIR = os.path.join(WORKSPACE_DIR_TOP, "install", "mission_interfaces", "lib")
_cur_ld = os.environ.get('LD_LIBRARY_PATH', '')
if _LIB_DIR not in _cur_ld:
    os.environ['LD_LIBRARY_PATH'] = _LIB_DIR + ":" + _cur_ld
    os.execv(sys.executable, [sys.executable] + sys.argv)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.kit.asset_converter")
except Exception as _e:
    print("[INTEGRATED WARN] asset_converter 확장 활성화 실패:", _e)

# ============================================================
# [ROS 통합] Isaac 내장 ROS2 bridge 활성화 + rclpy 경로 주입 (뒤의 import rclpy 전에 필수)
try:
    from isaacsim.core.utils.extensions import enable_extension as _enable_ext
    try:
        _enable_ext("isaacsim.ros2.bridge")          # Isaac 5.x 네이밍
    except Exception:
        try: _enable_ext("omni.isaac.ros2_bridge")    # 구버전 별칭 (옛 메인이 쓰던 것)
        except Exception as _e2: print("[ROS WARN] ros2 bridge 확장 활성화 실패:", _e2)
except Exception as _e:
    print("[ROS WARN] enable_extension import 실패:", _e)
# rclpy 가 확장 활성화로 안 잡힐 때를 대비해 경로 직접 주입 (mission_controller 와 동일)
_ISAAC_RCLPY_DIR = "/home/rokey/dev_ws/venv/isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/rclpy"
if os.path.isdir(_ISAAC_RCLPY_DIR) and _ISAAC_RCLPY_DIR not in sys.path:
    sys.path.append(_ISAAC_RCLPY_DIR)
    print(f"[ROS] rclpy 경로 주입: {_ISAAC_RCLPY_DIR}")
# ============================================================

import omni.timeline
import omni.usd
import numpy as np
from pxr import Usd, UsdGeom, Gf, UsdLux, UsdPhysics

import space_environment_v10_leo as scene
import iss_berthing   # [2단계] 로봇 init 의 setup_berthing 보다 먼저 import 되어야 함

import os
import omni.kit.commands
import omni.kit.app
import random

_ext_mgr = omni.kit.app.get_app().get_extension_manager()
_ext_mgr.set_extension_enabled_immediate("isaacsim.robot_setup.assembler", True)

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # m0609_pick_place_controller.py 가 이 scripts/ 폴더에 있음
from pathlib import Path
from isaacsim.core.utils.types import ArticulationAction
from m0609_pick_place_controller import PickPlaceController
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.robot_setup.assembler import RobotAssembler
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.rotations import euler_angles_to_quat

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).resolve()   # IK YAML(m0609_rg2_description/rmpflow_common) = scripts/
M0609_URDF_PATH   = "/home/rokey/dev_ws/isaac_sim/src/doosan-robot2/urdf/m0609_isaac_sim.urdf"
ONROBOT_URDF_PATH = "/home/rokey/dev_ws/isaac_sim/src/onrobot_rg2/urdf/onrobot_rg2.urdf"
R2D2_USD_PATH     = "/home/rokey/dev_ws/isaac_sim/IsaacLab/space_debris/space_debris_integrate_claude/resources/assets/usd/R2D2.usd"
EE_LINK_NAME      = "link_6"
GRIPPER_BASE_LINK = "angle_bracket"
BASE_LINK_NAME    = "base_link"

ARM_GRASP_DISTANCE = 0.12   
ARM_TRACK_STEP     = 0.008
ARM_GRASP_HOLD     = 20    
ARM_REACH_LIMIT = 1.05      
ARM_IK_JUMP_LIMIT = 0.10    # rad (약 6°): 한 스텝에 이보다 큰 관절 점프 = 거부

def import_urdf(urdf_path, fix_base=True):
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF 파일이 존재하지 않습니다: {urdf_path}")
    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = True
    import_config.import_inertia_tensor = True
    import_config.fix_base = fix_base
    import_config.distance_scale = 1.0
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.default_drive_strength = 5e3
    import_config.default_position_drive_damping = 5e2
    _, artic_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path, import_config=import_config, get_articulation_root=True,
    )
    if artic_path is None: return None, None
    robot_root = artic_path.rsplit("/", 1)[0] or artic_path
    return robot_root, artic_path

def find_prim_path_by_name(root_path, link_name):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid(): return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == link_name: return str(prim.GetPath())
    return None

def find_articulation_root(search_root):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(search_root)
    if not root_prim.IsValid(): return None
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI): return str(prim.GetPath())
    return None

def assemble_robot(stage, robot_base, robot_base_mount, robot_attach, robot_attach_mount, assembly_namespace, variant_name):
    assembler = RobotAssembler()
    assembler.begin_assembly(stage, robot_base, robot_base_mount, robot_attach, robot_attach_mount, assembly_namespace, variant_name)
    assembler.assemble()
    assembler.finish_assemble()

# ============================================================
# [1] 맵 빌드
# ============================================================
handles = scene.build_scene(simulation_app) 
ladder_path = handles["ladder_path"]
orbit_step = handles["orbit_step"]
kinematic_bodies = handles["kinematic_bodies"]

print("\n[INTEGRATED] 맵 빌드 완료")
stage = omni.usd.get_context().get_stage()

map_r2d2 = stage.GetPrimAtPath("/World/SpaceCleanupOrbitWorldV7/OrbitSystem/OrbitObjects/DebrisR2D2OrbitPivot/DebrisR2D2")
if map_r2d2.IsValid():
    UsdGeom.Imageable(map_r2d2).MakeInvisible()

def _world_pos(path):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return state.get("prev_ladder_pos", _ladder0).copy()   # invalid면 마지막 위치
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=float)

# ============================================================
# [★ 애니메이션 강화된 R2D2 매니저]
# ============================================================
class R2D2Manager:
    def __init__(self, stage, usd_path, start_pos):
        self.stage = stage
        self.r2d2_path = "/World/R2D2_Delivery_Receiver"
        self.usd_path = usd_path
        self.state = "WAIT" 
        self.ladder_scale = 1.0
        self.spin_angle = 0.0 # 회오리 애니메이션용 각도 변수
        self.r2d2_pos = start_pos
        self._load_r2d2_usd()

    def _load_r2d2_usd(self):
        self.r2d2_root = UsdGeom.Xform.Define(self.stage, self.r2d2_path)
        self.r2d2_root.GetPrim().GetReferences().AddReference(self.usd_path)
        self.r2d2_xf = UsdGeom.Xformable(self.r2d2_root.GetPrim())
        self.r2d2_xf.ClearXformOpOrder()
        self.translate_op = self.r2d2_xf.AddTranslateOp()
        self.orient_op = self.r2d2_xf.AddOrientOp()
        
        self.translate_op.Set(Gf.Vec3d(float(self.r2d2_pos[0]), float(self.r2d2_pos[1]), float(self.r2d2_pos[2])))
        q = euler_angles_to_quat(np.array([90.0, 0.0, 270.0]), degrees=True)
        self.orient_op.Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        
        self.scale_op = self.r2d2_xf.AddScaleOp()
        self.scale_op.Set(Gf.Vec3d(0.01, 0.01, 0.01))

    def update(self, dt, chaser_pos, ladder_path, ladder_handed_over):
        ladder_prim = self.stage.GetPrimAtPath(ladder_path)
        
        if self.state == "WAIT":
            if ladder_handed_over:
                self.state = "CONSUME"

        elif self.state == "CONSUME":
            # ★ 손으로 쥐어짜 으스러뜨리며 먹기 (회오리 X)
            self.ladder_scale -= 0.55 * dt
            crush = self.ladder_scale            # 1.0 -> 0.0 (남은 부피)
            progress = 1.0 - crush               # 0.0 -> 1.0 (으스러진 정도)

            if self.ladder_scale <= 0.05:
                self.ladder_scale = 0.001
                self.state = "DONE"
                if ladder_prim.IsValid():
                    UsdGeom.Imageable(ladder_prim).MakeInvisible()
                print("\n🎉 [R2D2] 끄억! (사다리를 손으로 구깃구깃 구겨서 꿀꺽 삼켰습니다) 🎉")

            if ladder_prim.IsValid():
                ladder_xf = UsdGeom.Xformable(ladder_prim)

                # ★ [1] 구기기 (Crush): 손에 쥐어짜이듯 축마다 불규칙하게 으스러짐
                #     긴 축(Z)이 가장 빨리 접히고, 단면(X/Y)은 들쭉날쭉 찌그러짐
                jx = random.uniform(0.55, 1.35)
                jy = random.uniform(0.55, 1.35)
                jz = random.uniform(0.20, 0.65) * (1.0 - 0.5 * progress)  # 진행될수록 더 납작하게 접힘
                s_x = max(0.01, crush * jx)
                s_y = max(0.01, crush * jy)
                s_z = max(0.01, crush * jz)

                scale_ops = [op for op in ladder_xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
                if not scale_ops: scale_op = ladder_xf.AddScaleOp()
                else: scale_op = scale_ops[0]
                scale_op.Set(Gf.Vec3d(s_x, s_y, s_z))

                # ★ [2] 쥐어짜는 꿈틀거림 (Crumple Jitter): 매 프레임 작은 각도로 마구 흔들림
                #     일정한 방향으로 도는 회오리가 아니라, 손아귀에서 버둥대는 느낌
                shake = 35.0 * (0.25 + crush)    # 클 땐 크게, 작아질수록 잔떨림
                rx = random.uniform(-shake, shake)
                ry = random.uniform(-shake, shake)
                rz = random.uniform(-shake, shake)
                rot_ops = [op for op in ladder_xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
                if not rot_ops: rot_op = ladder_xf.AddRotateXYZOp()
                else: rot_op = rot_ops[0]
                rot_op.Set(Gf.Vec3d(rx, ry, rz))

                # ★ [3] 입으로 가져가 삼키기 (Swallow): 으스러질수록 빠르게 당겨짐
                trans_ops = [op for op in ladder_xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
                if trans_ops:
                    trans_op = trans_ops[0]
                    curr_pos = np.array(trans_op.Get())
                    dir_to_r2d2 = self.r2d2_pos - curr_pos
                    pull_accel = 1.5 + progress * 12.0
                    curr_pos += dir_to_r2d2 * pull_accel * dt
                    trans_op.Set(Gf.Vec3d(float(curr_pos[0]), float(curr_pos[1]), float(curr_pos[2])))

_ladder0 = _world_pos(ladder_path)
_radial = _ladder0 / (float(np.linalg.norm(_ladder0)) + 1e-9)

# [2단계] 정거장 동반공전: 정거장을 사다리 바로 바깥(같은 궤도면)에 두어 접근선이 지구를 안 지나게
SPACE_STATION_PATH = "/World/SpaceCleanupOrbitWorldV7/OrbitSystem/OrbitObjects/StationOrbitPivot/SpaceStation"
try:
    _lad_e = next(b for b in kinematic_bodies if b.get("name") == "DebrisLadder")
    _sta_e = next(b for b in kinematic_bodies if b.get("kind") == "station")
    _STATION_OUT = 300.0
    _sta_e["r"]      = float(_lad_e["r"]) + _STATION_OUT
    _sta_e["phi0"]   = float(_lad_e["phi0"])
    _sta_e["omega"]  = float(_lad_e["omega"])
    _sta_e["tilt"]   = float(_lad_e.get("tilt", 0.0))
    _sta_e["tilt_z"] = float(_lad_e.get("tilt_z", 0.0))
    if orbit_step is not None:
        orbit_step(0.0)
    print("[2단계] 정거장 동반공전 설정 완료")
except Exception as _e:
    print(f"[2단계] 정거장 동반공전 설정 실패: {_e}")

# [2단계] berthing 으로 시작 → 체이서를 '핸들 접근축' 위에 띄워 스폰
#  berthing(라인264~268)은 outward = (handle - station_center) 방향으로 접근하게 돼 있음.
#  그 동일 축 위 _CHASER_STANDOFF m 밖에 스폰해야 핸들로 똑바로 접근함 (반경방향은 틀린 축).
_CHASER_STANDOFF = 15.0  # [튜닝] 핸들에서 접근축 따라 띄울 거리(m)
_CUSTOM_HANDLE_POS = np.array([-0.00475, 0.20282, 0.34249]) * 2.5414  # berthing 라인211과 동일
try:
    _sta_prim = stage.GetPrimAtPath(SPACE_STATION_PATH)
    _sta_l2w = UsdGeom.Xformable(_sta_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    _hw = _sta_l2w.Transform(Gf.Vec3d(float(_CUSTOM_HANDLE_POS[0]), float(_CUSTOM_HANDLE_POS[1]), float(_CUSTOM_HANDLE_POS[2])))
    _handle_world = np.array([_hw[0], _hw[1], _hw[2]])
    _station_world = _world_pos(SPACE_STATION_PATH)
    _outward = _handle_world - _station_world
    _outward_dir = _outward / (float(np.linalg.norm(_outward)) + 1e-9)
    # [통합] 실린더 핸들(로컬 Y축, 회전 [180,0,90])을 옆에서 잡도록 축에 수직으로 투영
    #  → berthing Phase1 의 _perp_approach_dir 과 동일한 접근축에 스폰 (end-on 방지)
    from scipy.spatial.transform import Rotation as _Rsp
    _cyl_local = _Rsp.from_euler('xyz', [180.0, 0.0, 90.0], degrees=True).apply([0.0, 1.0, 0.0])
    _cw = _sta_l2w.TransformDir(Gf.Vec3d(float(_cyl_local[0]), float(_cyl_local[1]), float(_cyl_local[2])))
    _cyl_world = np.array([_cw[0], _cw[1], _cw[2]])
    _cyl_world = _cyl_world / (float(np.linalg.norm(_cyl_world)) + 1e-9)
    _perp = _outward_dir - np.dot(_outward_dir, _cyl_world) * _cyl_world
    _pn = float(np.linalg.norm(_perp))
    if _pn < 1e-6:
        _tmp = np.array([0.0, 0.0, 1.0]) if abs(_cyl_world[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        _perp = _tmp - np.dot(_tmp, _cyl_world) * _cyl_world
        _pn = float(np.linalg.norm(_perp))
    _approach_dir = _perp / (_pn + 1e-9)
    chaser_pos = _handle_world + _approach_dir * _CHASER_STANDOFF
    print(f"[2단계] 체이서 스폰: 핸들 수직접근축 {_CHASER_STANDOFF:.0f}m 밖 → {chaser_pos}")
    print(f"        핸들={_handle_world} 실린더축={np.round(_cyl_world,2)} 접근방향={np.round(_approach_dir,2)}")
except Exception as _e:
    print(f"[2단계] 핸들 접근축 스폰 계산 실패({_e}) → 정거장 위치로 폴백")
    chaser_pos = _world_pos(SPACE_STATION_PATH).copy()

CHASER_PATH = "/World/Chaser"
_croot = UsdGeom.Xform.Define(stage, CHASER_PATH)
_cxf = UsdGeom.Xformable(_croot.GetPrim())
_cxf.ClearXformOpOrder()
_chaser_translate_op = _cxf.AddTranslateOp()
_chaser_orient_op = _cxf.AddOrientOp()

_cbody = UsdGeom.Cube.Define(stage, CHASER_PATH + "/Body")
_cbody.GetSizeAttr().Set(0.5)
_cbody_xf = UsdGeom.Xformable(_cbody.GetPrim())
_cbody_xf.ClearXformOpOrder()
_cbody_xf.AddScaleOp().Set(Gf.Vec3f(1.0, 0.5, 1.0))

_ccam = UsdGeom.Camera.Define(stage, CHASER_PATH + "/FrontCam")
_ccam.CreateFocalLengthAttr(24.0)
_ccam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1.0e6))
CAM_FWD_OFFSET = 0.5
_ccam_xf = UsdGeom.Xformable(_ccam.GetPrim())
_ccam_xf.ClearXformOpOrder()
_ccam_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -CAM_FWD_OFFSET))

_headlight = UsdLux.DistantLight.Define(stage, CHASER_PATH + "/FrontCam/Headlight")
_headlight.CreateIntensityAttr(3000.0)
_headlight.CreateAngleAttr(1.5)

THIRD_PERSON_CAM_PATH = "/World/ThirdPersonCam"
_tpcam = UsdGeom.Camera.Define(stage, THIRD_PERSON_CAM_PATH)
_tpcam.CreateFocalLengthAttr(18.0)
_tpcam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1.0e6))
_tpcam_xf = UsdGeom.Xformable(_tpcam.GetPrim())
_tpcam_xf.ClearXformOpOrder()
_tp_cam_top = _tpcam_xf.AddTransformOp()

ROBOT_CAM_PATH = "/World/RobotLookingCam"
_rcam = UsdGeom.Camera.Define(stage, ROBOT_CAM_PATH)
_rcam.CreateFocalLengthAttr(18.0)
_rcam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1.0e6))
_rcam_xf = UsdGeom.Xformable(_rcam.GetPrim())
_rcam_xf.ClearXformOpOrder()
_rcam_top = _rcam_xf.AddTransformOp()

def _set_chaser(eye, center):
    _m = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])),
        Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])),
        Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
    _chaser_translate_op.Set(_m.ExtractTranslation())
    _q = _m.ExtractRotationQuat()
    _im = _q.GetImaginary()
    _chaser_orient_op.Set(Gf.Quatf(float(_q.GetReal()), float(_im[0]), float(_im[1]), float(_im[2])))

DEBUG_LOG = False   # ★ True면 매 프레임 미션 대시보드 출력 (기본 off → 콘솔 깔끔)
ENABLE_PHYSICS = True
CHASER_MASS = 500.0
CHASER_BODY_SIZE = 0.5
PHYS_MAX_ACCEL = 4.0
PHYS_ATT_KP = 6.0
PHYS_MAX_ANGVEL = 5.0
ENABLE_TORQUE_ATTITUDE = True
ATT_KP_TORQUE = 300.0
ATT_KD_TORQUE = 200.0
ATT_MAX_TORQUE = 400.0

ENABLE_THRUSTER_FX = True
THR_PARTICLES_PER_NOZZLE = 6
THR_PLUME_LEN = 2.4
THR_PARTICLE_R = 0.06
THR_FORCE_REF = 2000.0
THR_TORQUE_REF = 150.0

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
    qc = np.asarray(q_cur_xyzw, float)
    qd = np.asarray(q_des_xyzw, float)
    q_err = _quat_mul_xyzw(qd, np.array([-qc[0], -qc[1], -qc[2], qc[3]]))
    if q_err[3] < 0.0: q_err = -q_err
    return 2.0 * float(k) * q_err[:3]

CHASER_APPROACH_SPEED = 18.0
STANDOFF_DISTANCE = 10.0
CATALOG_PREDICT_HORIZON_MAX = 22.0
ENABLE_AVOIDANCE = True
_sim_t = 0.0

ladder_def = None
for _b in kinematic_bodies:
    if _b["name"] == "DebrisLadder":
        ladder_def = _b
        break

def _normalize(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=float)

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

_set_chaser(chaser_pos, _ladder0)

# ============================================================
# [4] CBF 회피
# ============================================================
import math

CBF_ALPHA = 1.5
CBF_BODY_HALF = 0.5 / 2.0
CBF_SAFE_EXTRA = 1.0
EARTH_SAFE_EXTRA = 10.0
AVOID_LOOKAHEAD_SEC = 60.0
AVOID_PRED_SAMPLES = 60
AVOID_INTRUSION_MARGIN = 2.0
AVOID_DISABLE_RANGE = 50.0
OBSTACLE_FALLBACK_RADIUS = 4.0

def rot_orbit_np(p, tilt_x, tilt_z):
    out = scene.rot_orbit((float(p[0]), float(p[1]), float(p[2])), tilt_x, tilt_z)
    return np.array(out, dtype=float)

def map_obstacle_state(t, d):
    if d.get("static"):
        return d["center"], np.zeros(3, dtype=float)
    r = float(d["r"]); w = float(d["omega"]); phi = d["phi0"] + w * t
    p = np.array([r * math.cos(phi), r * math.sin(phi), 0.0], dtype=float)
    v = np.array([-r * w * math.sin(phi), r * w * math.cos(phi), 0.0], dtype=float)
    p = rot_orbit_np(p, d["tilt"], d["tilt_z"])
    v = rot_orbit_np(v, d["tilt"], d["tilt_z"])
    return p, v

def solve_cbf_qp(v_nom, A, b):
    v_nom = np.asarray(v_nom, dtype=float)
    m = A.shape[0]
    if m == 0: return v_nom.copy()
    if np.all(A @ v_nom - b >= -1e-9): return v_nom.copy()
    
    best_v = None; best_cost = 1e18
    for mask in range(1, 1 << m):
        idx = [i for i in range(m) if (mask >> i) & 1]
        if len(idx) > 3: continue
        As = A[idx]; bs = b[idx]
        G = As @ As.T
        try:
            lam = np.linalg.solve(G, bs - As @ v_nom)
        except np.linalg.LinAlgError:
            continue
        if np.any(lam < -1e-9): continue
        v = v_nom + As.T @ lam
        if np.all(A @ v - b >= -1e-6):
            cost = float(np.dot(v - v_nom, v - v_nom))
            if cost < best_cost:
                best_cost = cost; best_v = v
    
    if best_v is not None:
        return best_v
        
    v_repulse = v_nom.copy()
    violations = A @ v_nom - b
    for i in range(m):
        if violations[i] < 0.0:
            repulse_mag = min(-violations[i] + 0.3, 6.0)
            v_repulse = v_repulse + A[i] * repulse_mag
    nrep = float(np.linalg.norm(v_repulse))
    if nrep > CHASER_APPROACH_SPEED:
        v_repulse = v_repulse / nrep * CHASER_APPROACH_SPEED
    return v_repulse

def cbf_avoidance_velocity(p_c, v_nom, obstacle_list, t_now, alpha):
    A_rows = []; b_rows = []; diag = []
    taus = np.linspace(0.0, AVOID_LOOKAHEAD_SEC, AVOID_PRED_SAMPLES)
    
    for item in obstacle_list:
        d_def = item["def"]; d_safe = item["d_safe"]
        best_d = 1e18; tau_s = 0.0; p_o_s = None; v_o_s = None
        
        for tau in taus:
            p_o, v_o = map_obstacle_state(t_now + tau, d_def)
            p_r = p_c + v_nom * tau
            dd = float(np.linalg.norm(p_r - p_o))
            if dd < best_d:
                best_d = dd; tau_s = float(tau); p_o_s = p_o; v_o_s = v_o

        intruding = best_d < d_safe + AVOID_INTRUSION_MARGIN
        if not intruding:
            diag.append((best_d**2 - d_safe**2, best_d, tau_s, False))
            continue
            
        p_o_now, v_o_now = map_obstacle_state(t_now, d_def)
        rel_now = p_c - p_o_now
        h_now = float(np.dot(rel_now, rel_now)) - d_safe * d_safe

        if tau_s < 1e-3:
            a = 2.0 * rel_now
            bb = -2.0 * float(np.dot(rel_now, v_o_now)) - alpha * h_now
        else:
            rel_pred = (p_c + v_nom * tau_s) - p_o_s
            h_pred = float(np.dot(rel_pred, rel_pred)) - d_safe * d_safe
            a = 2.0 * tau_s * rel_pred
            bb = -alpha * h_pred - 2.0 * float(np.dot(rel_pred, v_o_s))

        na = float(np.linalg.norm(a))
        if na < 1e-9:
            diag.append((h_now, best_d, tau_s, intruding))
            continue
            
        a = a / na
        bb = bb / na
        
        a[2] = 0.0
        na2 = float(np.linalg.norm(a))
        if na2 < 1e-9:
            diag.append((h_now, best_d, tau_s, intruding))
            continue
        a = a / na2

        A_rows.append(a); b_rows.append(bb)
        diag.append((h_now, best_d, tau_s, intruding))

    if not A_rows:
        return v_nom.copy(), diag

    A = np.array(A_rows, dtype=float); b = np.array(b_rows, dtype=float)
    v_safe = solve_cbf_qp(v_nom, A, b)
    
    nv = float(np.linalg.norm(v_safe))
    if nv > CHASER_APPROACH_SPEED + 5.0:
        v_safe = v_safe / nv * (CHASER_APPROACH_SPEED + 5.0)
    return v_safe, diag

def _world_radius(path):
    try:
        prim = stage.GetPrimAtPath(path)
        bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
        rng = bbox.ComputeAlignedRange()
        s = rng.GetSize()
        rad = 0.5 * max(float(s[0]), float(s[1]), float(s[2]))
        if rad > 1e-3 and rad < 1e7:
            return rad
    except Exception as e:
        pass
    return OBSTACLE_FALLBACK_RADIUS

obstacle_list = []
for b in kinematic_bodies:
    if b["name"] == "DebrisLadder": continue
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

to_target_dir = _normalize(_ladder0 - chaser_pos)
dist_total = float(np.linalg.norm(_ladder0 - chaser_pos))

test_obs_pos = chaser_pos + to_target_dir * (dist_total * 0.5)

TEST_OBS_PATH = "/World/ForcedTestObstacle"
_tobs = UsdGeom.Sphere.Define(stage, TEST_OBS_PATH)
_tobs_radius = 30.0
_tobs.GetRadiusAttr().Set(_tobs_radius)
_tobs.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)]) 
_tobs_xf = UsdGeom.Xformable(_tobs.GetPrim())
_tobs_xf.ClearXformOpOrder()
_tobs_xf.AddTranslateOp().Set(Gf.Vec3d(float(test_obs_pos[0]), float(test_obs_pos[1]), float(test_obs_pos[2])))

obstacle_list.append({
    "def": {"static": True, "center": test_obs_pos},
    "d_safe": _tobs_radius + CBF_BODY_HALF + CBF_SAFE_EXTRA,
    "name": "ForcedTestObstacle",
})

# ============================================================
# [6] 메인 루프 
# ============================================================
import adr_perception as perc
try:
    import omni.replicator.core as rep
except Exception as _e:
    rep = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(_THIS_DIR, "..", "resources", "models", "yolo_best.pt")
KEYPOINT_MODEL_PATH = os.path.join(_THIS_DIR, "..", "resources", "models", "keypoint_best.pt")
KEYPOINTS_3D_PATH = os.path.join(_THIS_DIR, "..", "resources", "datasets", "keypoints_3d.json")
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CHASER_CAM_PATH = CHASER_PATH + "/FrontCam"

_cam = UsdGeom.Camera(stage.GetPrimAtPath(CHASER_CAM_PATH))
_cam.CreateHorizontalApertureAttr(20.955)

def _compute_intrinsics():
    focal = float(_cam.GetFocalLengthAttr().Get() or 24.0)
    h_ap = float(_cam.GetHorizontalApertureAttr().Get() or 20.955)
    fx = focal / h_ap * IMAGE_WIDTH
    return fx, fx, IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0

FX, FY, CX, CY = _compute_intrinsics()
K_MATRIX = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float64)

rgb_annotator = None
if rep is not None:
    try:
        _rgb_rp = rep.create.render_product(CHASER_CAM_PATH, (IMAGE_WIDTH, IMAGE_HEIGHT))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([_rgb_rp])
    except Exception as _e:
        pass

yolo_model = perc.load_yolo_model(YOLO_MODEL_PATH)
keypoint_model, keypoint_num_kp, keypoint_status = perc.load_keypoint_model(KEYPOINT_MODEL_PATH, KEYPOINTS_3D_PATH)
perc.ensure_debug_dir()

def _world_R(path):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return state.get("prev_ladder_R", np.eye(3))   # invalid면 마지막 R (None 금지)
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    cols = []
    for i in range(3):
        r = m.GetRow(i)
        row = np.array([r[0], r[1], r[2]], dtype=float)
        n = float(np.linalg.norm(row))
        cols.append(row / n if n > 1e-9 else row)
    return np.column_stack(cols)

def _measure_ladder_body_geometry(lpath, grasp_inset):
    try:
        R0 = _world_R(lpath)
        p0 = _world_pos(lpath)
        pts = []
        for pr in Usd.PrimRange(stage.GetPrimAtPath(lpath)):
            if not pr.IsA(UsdGeom.Mesh): continue
            pa = UsdGeom.Mesh(pr).GetPointsAttr().Get()
            if not pa: continue
            xf = UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for q in pa:
                wp = xf.Transform(Gf.Vec3d(float(q[0]), float(q[1]), float(q[2])))
                w = np.array([wp[0], wp[1], wp[2]], dtype=float)
                pts.append(R0.T @ (w - p0))
        if len(pts) < 12: return None
        P = np.asarray(pts, dtype=float)
        if len(P) > 20000:
            P = P[np.random.default_rng(0).choice(len(P), 20000, replace=False)]
        c = P.mean(axis=0)
        ew, ev = np.linalg.eigh(np.cov((P - c).T))
        axis = ev[:, int(np.argmax(ew))]
        proj = (P - c) @ axis
        tip_pos = c + axis * float(proj.max())
        tip_neg = c + axis * float(proj.min())
        grasp = c + axis * (float(proj.max()) - grasp_inset)
        return grasp, c, tip_pos, tip_neg, int(len(P))
    except Exception:
        return None

_meas = _measure_ladder_body_geometry(ladder_path, perc.V8_GRASP_INSET)
if _meas is not None:
    V8_GRASP_BODY, V8_CENTROID_BODY, V8_TIP_POS_BODY, V8_TIP_NEG_BODY, _npts = _meas
else:
    V8_GRASP_BODY, V8_CENTROID_BODY, V8_TIP_POS_BODY, V8_TIP_NEG_BODY = perc.load_v8_grasp_geometry(KEYPOINTS_3D_PATH)

V8_APPROACH_BODY = perc.normalize(V8_TIP_POS_BODY - V8_CENTROID_BODY)
V8_STANDOFF_BODY = V8_TIP_POS_BODY + V8_APPROACH_BODY * (perc.V8_BODY_GAP + perc.V8_BODY_HALF)
_v8_ladder_long = float(np.linalg.norm(V8_TIP_POS_BODY - V8_TIP_NEG_BODY))

ENABLE_PERCEPTION = True
_PERC_EVERY = 2
_perc_frame = 0

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
    "_arm_grasp_q": None,
    "arm_handoff": False, "arm_state": "WAIT",
    "avoid_active": False, "avoid_intruders": 0, "avoid_dv": 0.0, "avoid_min_h": 999.0,
    "last_yolo_status": "init", "_dbg_R_logged": 0,
    "chaser_vel_cmd": np.zeros(3),
    "chaser_vel_actual": np.zeros(3),
    "chaser_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
    "prev_omega": np.zeros(3),
    "cmd_force": np.zeros(3), "cmd_torque": np.zeros(3),
    "active_nozzles": 0,
    "ladder_grabbed": False,
    "ladder_handed_over": False, 
}

R2D2_START_POS = _ladder0 + np.array([-7.0, 7.0, 0.0], dtype=float)
# R2D2_START_POS[1] = 270.0   # ★ R2D2 y좌표 절대값 고정 (270)
r2d2_manager = R2D2Manager(stage, R2D2_USD_PATH, R2D2_START_POS)

def _run_perception(camera_pos, R_cv_wc, predicted_kf_pos, true_pose_pos, true_pose_R,
                    kf_pos, kf_capture_pos, expected_camera_range, chaser_pos, chaser_vel, obstacle_list, true_range):
    global _perc_frame
    if not (ENABLE_PERCEPTION and rgb_annotator is not None and yolo_model is not None and perc.cv2 is not None):
        return
    _perc_frame += 1
    if _perc_frame % _PERC_EVERY != 0: return
    
    try: data = rgb_annotator.get_data()
    except Exception: data = None
    if data is None or len(data) == 0: return
    
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

        if detection is not None:
            state["yolo_conf"] = detection["conf"]
            state["yolo_cls_id"] = detection["cls_id"]
            state["last_yolo_lock_valid"] = True
            state["last_yolo_lock_time"] = _sim_t
            state["vision_lock_age"] = 0.0
            range_est = float(np.linalg.norm(kf_pos - camera_pos))
            detector_pos = perc.bbox_center_to_world_position(detection["xyxy"], camera_pos, R_cv_wc, range_est, FX, FY, CX, CY)
            state["kf_x"], state["kf_P"] = perc.kalman_update_detector_position(state["kf_x"], state["kf_P"], detector_pos)

            image_points, object_points, detected_keypoints, keypoint_crop_box, keypoint_status = \
                perc.predict_keypoints_from_yolo_bbox(rgb_image, detection, keypoint_model)
            pnp_ok, pnp_pos, pnp_R, reproj_err, projected_points, pnp_method = \
                perc.estimate_pose_solvepnp(image_points, object_points, camera_pos, R_cv_wc, K_MATRIX)

            if pnp_ok:
                state["last_good_pnp_valid"] = True
                state["last_good_pnp_time"] = _sim_t
                state["last_good_pnp_pos"] = pnp_pos.copy()
                state["last_good_pnp_R"] = pnp_R.copy()
                state["pnp_good_streak"] += 1
                state["pnp_hold_active"] = True
        else:
            state["yolo_conf"] = 0.0

    except Exception as e:
        pass

_arm_dbg = 0
_arm_last_good_q = None
_arm_grasp_step = None

def arm_step():
    global _arm_dbg, _arm_last_good_q, _arm_grasp_step, fixed_align_quat
    if not state.get("arm_handoff", False): return
    try:
        cur_q = robot.get_joint_positions()
        if _arm_last_good_q is None: _arm_last_good_q = np.copy(cur_q)
        target_valid = bool(state.get("v8_tracking_active", False))
        st = state["arm_state"]

        if (not target_valid) and st == "TRACK":
            state["arm_state"] = "WAIT_HOME"; st = "WAIT_HOME"
        if st == "WAIT_HOME" and target_valid:
            state["arm_state"] = "TRACK"; st = "TRACK"

        _ee_w = UsdGeom.Xformable(omni.usd.get_context().get_stage().GetPrimAtPath(robot_ee_path)).ComputeLocalToWorldTransform(0)
        ee_pos = np.array(_ee_w.ExtractTranslation(), dtype=float)
        grasp_pos = np.asarray(state["v8_grasp_now"], dtype=float)
        dist = float(np.linalg.norm(grasp_pos - ee_pos))

        cspace = ik_controller._cspace_controller
        base_p, base_q_xyzw, _ = phys_read_chaser()
        _cq = np.array([base_q_xyzw[3], base_q_xyzw[0], base_q_xyzw[1], base_q_xyzw[2]])  # 체이서 (w,x,y,z)
        MOUNT_Q = np.array([0.5, -0.5, 0.5, 0.5])     # 1205줄 LocalRot0 = 팔 마운트 회전
        w0,x0,y0,z0 = _cq; w1,x1,y1,z1 = MOUNT_Q
        base_q_wxyz = np.array([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1])
        cspace._motion_policy.set_robot_base_pose(robot_position=base_p, robot_orientation=base_q_wxyz)

        if st == "WAIT_HOME":
            cur_arm = np.array(cur_q[:6], dtype=float)
            step = np.clip(init_arm - cur_arm, -0.01, 0.01)
            tgt = np.copy(cur_q); tgt[:6] = cur_arm + step
            robot.apply_action(ArticulationAction(joint_positions=tgt))
        elif st == "TRACK":
            base_to_grasp = float(np.linalg.norm(grasp_pos - base_p))
            if base_to_grasp > ARM_REACH_LIMIT:
                robot.apply_action(ArticulationAction(joint_positions=_arm_last_good_q))
            else:
                direction = grasp_pos - ee_pos
                d = np.linalg.norm(direction)
                if d > ARM_TRACK_STEP: direction = direction / d * ARM_TRACK_STEP
                # --- 손목 재조준(look-at): FPV카메라 -Z를 grasp 쪽으로 점진 정렬 ---
                AIM_STEP = 0.06
                _stg = omni.usd.get_context().get_stage()
                _cam_xf = UsdGeom.Xformable(_stg.GetPrimAtPath(f"{robot_ee_path}/FPV_Camera")).ComputeLocalToWorldTransform(0)
                cam_pos = np.array(_cam_xf.ExtractTranslation(), dtype=float)
                _fwd_gf = _cam_xf.TransformDir(Gf.Vec3d(0, 0, -1))
                cam_fwd = np.array([_fwd_gf[0], _fwd_gf[1], _fwd_gf[2]], dtype=float)
                cam_fwd /= (np.linalg.norm(cam_fwd) + 1e-9)
                _f = grasp_pos - cam_pos; _f /= (np.linalg.norm(_f) + 1e-9)
                _axis = np.cross(cam_fwd, _f); _s = np.linalg.norm(_axis)
                _ang = np.arctan2(_s, float(np.dot(cam_fwd, _f)))
                _q = _ee_w.ExtractRotationQuat()
                cur_quat = np.array([_q.GetReal(), *_q.GetImaginary()], dtype=float)
                if _s > 1e-6:
                    _axis /= _s
                    _h = np.clip(_ang, 0.0, AIM_STEP) / 2.0
                    qi = np.array([np.cos(_h), *(np.sin(_h) * _axis)])
                    w0,x0,y0,z0 = qi; w1,x1,y1,z1 = cur_quat
                    aim_quat = np.array([
                        w0*w1 - x0*x1 - y0*y1 - z0*z1,
                        w0*x1 + x0*w1 + y0*z1 - z0*y1,
                        w0*y1 - x0*z1 + y0*w1 + z0*x1,
                        w0*z1 + x0*y1 - y0*x1 + z0*w1])
                else:
                    aim_quat = cur_quat
                if _arm_dbg % 30 == 0:
                    print(f"[AIM] cam→grasp_ang={np.degrees(_ang):5.1f}deg ee→grasp={dist:.2f}", flush=True)
                actions = cspace.forward(target_end_effector_position=ee_pos + direction,
                                         target_end_effector_orientation=aim_quat)
                # IK 해 검증: NaN/과도점프면 적용 안 하고 직전자세 유지 (죽음 방지)
                ok = (actions is not None and actions.joint_positions is not None
                      and np.all(np.isfinite(actions.joint_positions))
                      and np.max(np.abs(np.asarray(actions.joint_positions)[:6] - np.asarray(cur_q)[:6])) < ARM_IK_JUMP_LIMIT)
                if ok:
                    robot.apply_action(actions)
                    _arm_last_good_q = np.copy(actions.joint_positions)
                else:
                    robot.apply_action(ArticulationAction(joint_positions=_arm_last_good_q))
                    if _arm_dbg % 60 == 0:
                        print("[INTEGRATED/ARM] IK 해 불량(NaN/점프) → 직전자세 유지", flush=True)
                if dist <= ARM_GRASP_DISTANCE:
                    state["arm_state"] = "GRASP"
                    print(f"[INTEGRATED/ARM] TRACK->GRASP (dist={dist:.3f}m)", flush=True)
        
        elif st == "GRASP":
            if _arm_grasp_step is None:
                _arm_grasp_step = _arm_dbg; state["_arm_grasp_q"] = np.copy(cur_q)
                print(f"[GRASP] 진입 dist={dist:.3f}", flush=True)
            tgt = np.copy(state["_arm_grasp_q"]); tgt[6], tgt[9] = 0.9, -0.9
            robot.apply_action(ArticulationAction(joint_positions=tgt))
            if _arm_dbg - _arm_grasp_step >= ARM_GRASP_HOLD:
                print("[GRASP] HOLD 끝 → 사다리 분리/RETURN 시작", flush=True)
                state["arm_state"] = "RETURN"
                state["ladder_grabbed"] = True
                print("[DBG1] state 전환 완료", flush=True)

                l_prim = omni.usd.get_context().get_stage().GetPrimAtPath(ladder_path)

                # 공전 시스템이 못 건드리게 버블 등록
                if not hasattr(scene, "_PHYSICS_BUBBLE_PATHS"):
                    scene._PHYSICS_BUBBLE_PATHS = set()
                scene._PHYSICS_BUBBLE_PATHS.add(ladder_path)
                print("[DBG3] bubble 등록 완료", flush=True)

                try:
                    _stg2 = omni.usd.get_context().get_stage()
                    # (a) 리지드바디 끄기 — reparent 전 물리 분리
                    _rb = UsdPhysics.RigidBodyAPI(l_prim)
                    if _rb:
                        _rb.GetRigidBodyEnabledAttr().Set(False)
                    # (b) 콜라이더 비활성화 (본체+프록시 전부)
                    for p in Usd.PrimRange(l_prim):
                        if p.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
                    print("[DBG4] 물리 OFF 완료", flush=True)

                    # (c) reparent: 사다리를 EE(link_6) 자식으로 → 잡은 위치 그대로 따라옴
                    _ee_path_now = robot_ee_path
                    _new_ladder_path = f"{_ee_path_now}/GrabbedLadder"
                    # 잡은 순간의 사다리 월드 트랜스폼을 기억(reparent 후 로컬로 환산해 위치 유지)
                    _l_world = UsdGeom.Xformable(l_prim).ComputeLocalToWorldTransform(0)
                    _ee_world = UsdGeom.Xformable(_stg2.GetPrimAtPath(_ee_path_now)).ComputeLocalToWorldTransform(0)
                    _ee_world_inv = _ee_world.GetInverse()
                    _l_local = _l_world * _ee_world_inv     # 사다리의 EE-로컬 트랜스폼
                    from pxr import Sdf
                    _edit = Sdf.BatchNamespaceEdit()
                    _edit.Add(Sdf.Path(ladder_path), Sdf.Path(_new_ladder_path))
                    if _stg2.GetEditTarget().GetLayer().Apply(_edit):
                        state["ladder_path_grabbed"] = _new_ladder_path
                        # reparent 후 로컬 트랜스폼을 잡은 그 자세로 고정
                        _lp2 = _stg2.GetPrimAtPath(_new_ladder_path)
                        _lx2 = UsdGeom.Xformable(_lp2)
                        _lx2.ClearXformOpOrder()
                        _lx2.AddTransformOp().Set(_l_local)
                        print(f"[DBG5] reparent 완료 → {_new_ladder_path}", flush=True)
                    else:
                        print("[DBG5] reparent 실패 — trans_op 폴백 사용", flush=True)
                        state["ladder_path_grabbed"] = None
                except Exception as _e:
                    import traceback
                    print("[DBG_GRAB] 예외:", _e, flush=True); traceback.print_exc()
                
        elif st == "RETURN":
            if _arm_dbg % 30 == 0:
                _to_r = r2d2_manager.r2d2_pos - chaser_pos
                print(f"[RETURN] handed={state.get('ladder_handed_over')} R2D2거리={np.linalg.norm(_to_r):.1f}", flush=True)
            # 잡은 자세 그대로 freeze (시작자세 복귀 X) — 사다리 매달림 흐물거림 방지
            tgt = np.copy(state.get("_arm_grasp_q", cur_q))
            tgt[6], tgt[9] = 0.9, -0.9   # 그리퍼 닫은 채 유지
            robot.apply_action(ArticulationAction(joint_positions=tgt))
                    
        elif st == "RELEASE":
            # (1) 사다리를 그리퍼에서 떼어내 원래 경로로 되돌림 (R2D2가 삼킬 수 있게)
            if not state.get("ladder_released_done"):
                try:
                    _stg3 = omni.usd.get_context().get_stage()
                    _grabbed = state.get("ladder_path_grabbed")
                    if _grabbed and _stg3.GetPrimAtPath(_grabbed).IsValid():
                        # 현재 사다리 월드 트랜스폼 기억
                        _lp = _stg3.GetPrimAtPath(_grabbed)
                        _l_world = UsdGeom.Xformable(_lp).ComputeLocalToWorldTransform(0)
                        # 원래 경로로 reparent 복원
                        from pxr import Sdf
                        _edit = Sdf.BatchNamespaceEdit()
                        _edit.Add(Sdf.Path(_grabbed), Sdf.Path(ladder_path))
                        if _stg3.GetEditTarget().GetLayer().Apply(_edit):
                            # 떼어낸 자리(R2D2 근처 월드위치)에 고정
                            _lp2 = _stg3.GetPrimAtPath(ladder_path)
                            _lx2 = UsdGeom.Xformable(_lp2)
                            _lx2.ClearXformOpOrder()
                            _t = _l_world.ExtractTranslation()
                            _lx2.AddTranslateOp().Set(Gf.Vec3d(_t[0], _t[1], _t[2]))
                            _lx2.AddScaleOp().Set(Gf.Vec3d(1,1,1))   # R2D2Manager가 scale로 삼킴
                            print(f"[RELEASE] 사다리 그리퍼에서 분리 → {ladder_path}", flush=True)
                    state["ladder_released_done"] = True
                except Exception as _e:
                    import traceback
                    print("[RELEASE] 분리 예외:", _e, flush=True); traceback.print_exc()
                    state["ladder_released_done"] = True
            # (2) 그리퍼 열기
            cur_arm = np.array(cur_q[:6], dtype=float)
            tgt = np.copy(cur_q); tgt[:6] = cur_arm
            tgt[6], tgt[9] = 0.0, 0.0
            robot.apply_action(ArticulationAction(joint_positions=tgt))

        _arm_dbg += 1
    except Exception:
        try: robot.apply_action(ArticulationAction(joint_positions=(_arm_last_good_q if _arm_last_good_q is not None else robot.get_joint_positions())))
        except Exception: pass
        _arm_dbg += 1

def on_update(dt):
    global chaser_pos, _sim_t
    _sim_t += dt
    t = _sim_t
    state["time_since_catalog_update"] += dt
    state["time_since_yolo_update"] += dt

    if ENABLE_PHYSICS and _chaser_rb is not None:
        try:
            _cp, _cq, _cv = phys_read_chaser()
            chaser_pos = _cp
            state["chaser_quat_xyzw"] = _cq
            state["chaser_vel_actual"] = _cv
        except Exception: pass

    if state["last_good_pnp_valid"]:
        age = t - state["last_good_pnp_time"]
        state["pnp_hold_active"] = age <= perc.PNP_HOLD_TIME_SEC
        if not state["pnp_hold_active"]:
            state["last_good_pnp_valid"] = False
            state["pnp_good_streak"] = 0
    else: state["pnp_hold_active"] = False

    if state["last_yolo_lock_valid"]:
        state["vision_lock_age"] = t - state["last_yolo_lock_time"]
        if state["vision_lock_age"] > perc.VISION_LOCK_HOLD_SEC:
            state["last_yolo_lock_valid"] = False
    else: state["vision_lock_age"] = 999.0

    if state.get("ladder_grabbed"):
        # 잡은 뒤엔 사다리가 link_6 자식으로 옮겨가 옛 경로가 없음 → 마지막 값 재사용
        true_pos = state.get("prev_ladder_pos", _ladder0).copy()
        true_pose_R = state.get("prev_ladder_R", np.eye(3))
        true_vel = np.zeros(3)
    else:
        true_pos = _world_pos(ladder_path)
        true_pose_R = _world_R(ladder_path)
        true_vel = (true_pos - state["prev_ladder_pos"]) / dt if dt > 1e-6 else np.zeros(3)
        state["prev_ladder_pos"] = true_pos.copy()
        state["prev_ladder_R"] = true_pose_R          # R도 기억해두기

    true_pose_R = _world_R(ladder_path)
    state["prev_ladder_pos"] = true_pos.copy()

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

    vision_recent = state["last_yolo_lock_valid"]
    pose_recent = state["last_good_pnp_valid"] and state["pnp_hold_active"]
    state["phase"] = perc.choose_mission_phase(predicted_kf_pos, chaser_pos, vision_recent, pose_recent)

    true_range = float(np.linalg.norm(true_pos - chaser_pos))
    if true_range <= perc.V8_TRIGGER_RANGE: state["v8_trigger_hold"] += dt
    else: state["v8_trigger_hold"] = 0.0
    if (not state["v8_engaged"]) and state["v8_trigger_hold"] >= perc.V8_TRIGGER_HOLD_SEC:
        state["v8_engaged"] = True
        state["v8_substage"] = "approach5"
    if state["v8_engaged"]:
        state["v8_estimator"].update(t, true_pose_R, true_pos)
    v8_track_ready = state["v8_engaged"] and state["v8_estimator"].ready()

    v_ff = np.zeros(3)
    tip_pos_now = tip_neg_now = None
    
    if v8_track_ready:
        est = state["v8_estimator"]
        grasp_now = est.current_point(V8_TIP_POS_BODY) 
        tip_pos_now = est.current_point(V8_TIP_POS_BODY)
        tip_neg_now = est.current_point(V8_TIP_NEG_BODY)
        standoff_tar = est.predict_point(V8_STANDOFF_BODY, perc.V8_LEAD_SEC)
        _wp_body = V8_TIP_POS_BODY + V8_APPROACH_BODY * perc.V8_WAYPOINT_RANGE
        waypoint5 = est.predict_point(_wp_body, perc.V8_LEAD_SEC)
        if state["v8_substage"] == "approach5":
            if float(np.linalg.norm(chaser_pos - waypoint5)) < perc.V8_WAYPOINT_REACH:
                state["v8_substage"] = "creep"
            selected_capture_pos = waypoint5
            v_ff = est.point_velocity(_wp_body)
        else:
            selected_capture_pos = standoff_tar
            v_ff = est.point_velocity(V8_STANDOFF_BODY)
        state["control_source"] = "V8_PNP_STANDOFF_TRACK"
        state["v8_tracking_active"] = True
        state["v8_grasp_now"] = grasp_now.copy()
        
        HANDOFF_REL_SPEED = 1.5   
        HANDOFF_MAX_GAP   = 8.0   
        if (not state["arm_handoff"]) and state["v8_substage"] == "creep" and 0.0 <= state["v8_rel_speed"] < HANDOFF_REL_SPEED and 0.0 <= state["v8_body_gap"] < HANDOFF_MAX_GAP:
            state["arm_handoff"] = True
            state["arm_state"] = "TRACK"
    else:
        selected_capture_pos = kf_capture_pos
        state["control_source"] = "KF_VISION_TRACK"
        state["v8_tracking_active"] = False

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
        approach_speed = perc.compute_slowdown_speed(selected_capture_error)
        chaser_vel = kf_vel + perc.normalize(to_capture) * approach_speed
        chaser_vel = perc.limit_vector(chaser_vel, perc.MAX_APPROACH_SPEED)
        state["v8_body_gap"] = -1.0
        state["v8_rel_speed"] = float(approach_speed)

    if DEBUG_LOG and int(t * 2.0) != int((t - dt) * 2.0):
        print(f"\n--- [{t:.1f}s] MISSION DASHBOARD ---")
        print(f" 🎯 Phase: {state['phase']} | 🦾 Arm: {state['arm_state']}")
        
        if not state.get("ladder_grabbed"):
            if state["v8_engaged"]:
                print(f" 🚀 Approach: 속도차={state.get('v8_rel_speed',0):.2f}m/s | 거리차={state.get('v8_body_gap',0):.2f}m")
            else:
                print(f" 🔭 Searching... 사다리까지 남은 거리: {true_range:.1f}m")
        else:
            to_r2d2 = r2d2_manager.r2d2_pos - chaser_pos
            dist_to_r2d2 = float(np.linalg.norm(to_r2d2))
            print(f" 📦 Delivery: R2D2까지 남은 거리: {dist_to_r2d2:.1f}m | 건네주기 완료: {state.get('ladder_handed_over')}")

    if state.get("ladder_grabbed"):
        state["v8_tracking_active"] = False 
        state["avoid_active"] = False       
        
        to_r2d2 = r2d2_manager.r2d2_pos - chaser_pos
        dist_to_r2d2 = float(np.linalg.norm(to_r2d2))
        
        if not state.get("ladder_handed_over"):
            if dist_to_r2d2 > 5.5: 
                approach_speed = perc.compute_slowdown_speed(dist_to_r2d2 - 4.0) 
                v_mag = max(approach_speed, 1.0) 
                chaser_vel = perc.normalize(to_r2d2) * min(v_mag, 8.0) 
            else:
                chaser_vel = np.zeros(3) 
                state["ladder_handed_over"] = True
                state["arm_state"] = "RELEASE"
        else:
            chaser_vel = np.zeros(3)

    if ENABLE_AVOIDANCE and obstacle_list and true_range > AVOID_DISABLE_RANGE and not state.get("ladder_grabbed"):
        v_safe, avoid_diag = cbf_avoidance_velocity(chaser_pos, chaser_vel, obstacle_list, _sim_t, CBF_ALPHA)
        avoid_delta = v_safe - chaser_vel
        dv = float(np.linalg.norm(avoid_delta))
        if dv > 1e-4:
            state["avoid_active"] = True
            state["avoid_dv"] = dv
            state["avoid_min_h"] = min([d[0] for d in avoid_diag] if avoid_diag else [999.0])
            state["avoid_intruders"] = int(sum(1 for d in avoid_diag if d[3]))
            
            chaser_vel = v_safe
            _z_now = float(chaser_pos[2])
            chaser_vel[2] = chaser_vel[2] * 0.5 - 3.0 * _z_now
            chaser_vel[2] = float(np.clip(chaser_vel[2], -4.0, 4.0))
            chaser_vel = perc.limit_vector(chaser_vel, perc.MAX_APPROACH_SPEED)
        else:
            state["avoid_active"] = False
            state["avoid_dv"] = 0.0
    else:
        state["avoid_active"] = False
        state["avoid_dv"] = 0.0

    if ENABLE_PHYSICS and _chaser_rb is not None:
        v_prev = state["chaser_vel_cmd"]
        accel_limit = PHYS_MAX_ACCEL * dt * (3.0 if state["avoid_active"] else 1.0)
        v_cmd = v_prev + perc.limit_vector(chaser_vel - v_prev, accel_limit)
        
        if state.get("ladder_handed_over"):
            v_cmd = np.zeros(3)
            
        state["chaser_vel_cmd"] = v_cmd
        try:
            _, _, v_now = phys_read_chaser()
            v_now = np.asarray(v_now, float).reshape(3)
        except Exception:
            v_now = v_prev
        K_VEL_FORCE = CHASER_MASS / 0.3
        f_cmd = K_VEL_FORCE * (v_cmd - v_now)
        f_cmd = perc.limit_vector(f_cmd, CHASER_MASS * PHYS_MAX_ACCEL * 2.0)
        state["cmd_force"] = f_cmd
        try: phys_apply_force(f_cmd)
        except Exception: pass

        _pt = state["v8_grasp_now"] if v8_track_ready else predicted_kf_pos
        
        if state.get("ladder_grabbed"):
            _pt = r2d2_manager.r2d2_pos
            
        q_des = perc.rotmat_to_quat_xyzw(perc.make_camera_rotation_from_forward(_pt - chaser_pos)[0])
        q_cur = state["chaser_quat_xyzw"]
        if ENABLE_TORQUE_ATTITUDE:
            q_errv = quat_align_angvel(q_cur, q_des, 0.5)
            try: omega_act = phys_get_angvel()
            except Exception: omega_act = state["prev_omega"]
            tau = perc.limit_vector(ATT_KP_TORQUE * q_errv - ATT_KD_TORQUE * omega_act, ATT_MAX_TORQUE)
            try: phys_apply_torque(tau)
            except Exception: pass
            state["cmd_torque"] = tau
            state["prev_omega"] = omega_act
            
        if ENABLE_THRUSTER_FX:
            _R_now = quat_xyzw_to_rotmat(state["chaser_quat_xyzw"])
            state["active_nozzles"] = update_thruster_fx(chaser_pos, _R_now, state["cmd_force"], state["cmd_torque"], dt)
        new_chaser_pos = chaser_pos
    else:
        new_chaser_pos = chaser_pos + chaser_vel * dt
    chaser_pos = new_chaser_pos

    _tp_tgt = state["v8_grasp_now"] if v8_track_ready else true_pos
    _tp_mid = 0.5 * (chaser_pos + _tp_tgt)
    _tp_back = perc.normalize(chaser_pos - _tp_tgt)
    _tp_span = max(float(np.linalg.norm(chaser_pos - _tp_tgt)), _v8_ladder_long, 5.0)
    _tp_eye = _tp_mid + _tp_back * (_tp_span * 0.9) + np.array([0.0, 0.0, _tp_span * 0.7], dtype=float)
    _tp_view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(float(_tp_eye[0]), float(_tp_eye[1]), float(_tp_eye[2])),
        Gf.Vec3d(float(_tp_mid[0]), float(_tp_mid[1]), float(_tp_mid[2])),
        Gf.Vec3d(0.0, 0.0, 1.0))
    _tp_cam_top.Set(_tp_view.GetInverse())
    
    rcam_eye = chaser_pos + np.array([-4.0, -4.0, 3.0])
    rcam_view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(float(rcam_eye[0]), float(rcam_eye[1]), float(rcam_eye[2])),
        Gf.Vec3d(float(chaser_pos[0]), float(chaser_pos[1]), float(chaser_pos[2])),
        Gf.Vec3d(0.0, 0.0, 1.0))
    _rcam_top.Set(rcam_view.GetInverse())

    pointing_target = state["v8_grasp_now"] if v8_track_ready else predicted_kf_pos
    R_chaser_wc = perc.forward_to_rotmat_chaser(pointing_target - chaser_pos)
    expected_camera_range = float(np.linalg.norm(predicted_kf_pos - chaser_pos))
    camera_look_target = predicted_kf_pos

    if ENABLE_PHYSICS and _chaser_rb is not None:
        R_usd_wc = quat_xyzw_to_rotmat(state["chaser_quat_xyzw"])
        camera_forward = perc.normalize(-R_usd_wc[:, 2])
        camera_pos = chaser_pos + camera_forward * CAM_FWD_OFFSET
        R_cv_wc = R_usd_wc @ np.diag([1.0, -1.0, -1.0])
    else:
        _set_chaser(chaser_pos, camera_look_target)
        camera_forward = perc.normalize(camera_look_target - chaser_pos)
        camera_pos = chaser_pos + camera_forward * CAM_FWD_OFFSET
        _R_usd, R_cv_wc = perc.make_camera_rotation_from_forward(camera_forward)

    if state["time_since_yolo_update"] >= perc.YOLO_UPDATE_PERIOD:
        state["time_since_yolo_update"] = 0.0
        _run_perception(camera_pos, R_cv_wc, predicted_kf_pos, true_pos, true_pose_R, kf_pos, kf_capture_pos, expected_camera_range, chaser_pos, chaser_vel, obstacle_list, true_range)


if ENABLE_PHYSICS:
    try:
        from pxr import Sdf, UsdPhysics, PhysxSchema
        _chaser_prim_p = stage.GetPrimAtPath(Sdf.Path(CHASER_PATH))
        _rb = UsdPhysics.RigidBodyAPI.Apply(_chaser_prim_p)
        _rb.CreateRigidBodyEnabledAttr(True)
        _rb.CreateKinematicEnabledAttr(False)
        _mass = UsdPhysics.MassAPI.Apply(_chaser_prim_p)
        _mass.CreateMassAttr(float(CHASER_MASS))
        
        chaser_body_prim = stage.GetPrimAtPath(Sdf.Path(CHASER_PATH + "/Body"))
        if chaser_body_prim.IsValid():
            UsdPhysics.CollisionAPI.Apply(chaser_body_prim)
        PhysxSchema.PhysxRigidBodyAPI.Apply(_chaser_prim_p)
        
        _ladder_prim_p = stage.GetPrimAtPath(Sdf.Path(ladder_path))
        _lrb = UsdPhysics.RigidBodyAPI.Apply(_ladder_prim_p)
        _lrb.CreateRigidBodyEnabledAttr(True)
        _lrb.CreateKinematicEnabledAttr(True)

        # ★ 콜리전 전용 프록시 메시 (GeomSubset/머티리얼 없는 깨끗한 복제)
        #   크래시 원인 = 원본 메시의 면별 머티리얼(GeomSubset)을 쿠킹이 읽다 세그폴트.
        #   approximation 종류로는 못 피함(다 트라이앵귤레이션 단계에서 읽음).
        #   → 시각 메시엔 콜라이더 X(룩 그대로), 서브셋 없는 프록시에만 콜라이더 → fillFaceMaterials 회피.
        _proxy_n = 0
        for _p in list(Usd.PrimRange(_ladder_prim_p)):
            if (not _p.IsA(UsdGeom.Mesh)) or _p.GetName() == "collision_proxy":
                continue
            _src = UsdGeom.Mesh(_p)
            _pts = _src.GetPointsAttr().Get()
            _fvc = _src.GetFaceVertexCountsAttr().Get()
            _fvi = _src.GetFaceVertexIndicesAttr().Get()
            if _pts is None or _fvc is None or _fvi is None:
                continue
            _proxy = UsdGeom.Mesh.Define(stage, _p.GetPath().AppendChild("collision_proxy"))
            _proxy.CreatePointsAttr(_pts)
            _proxy.CreateFaceVertexCountsAttr(_fvc)
            _proxy.CreateFaceVertexIndicesAttr(_fvi)
            UsdGeom.Imageable(_proxy.GetPrim()).MakeInvisible()  # 렌더 X, 물리 콜리전 전용
            UsdPhysics.CollisionAPI.Apply(_proxy.GetPrim())
            # 빈틈(오목) 유지 + 실제 접촉 + 동적 가능 → convexDecomposition
            UsdPhysics.MeshCollisionAPI.Apply(_proxy.GetPrim()).CreateApproximationAttr().Set("convexDecomposition")
            try:
                _cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(_proxy.GetPrim())
                _cd.CreateMaxConvexHullsAttr(64)        # 칸/레일을 충분히 쪼개 빈틈 보존
                _cd.CreateHullVertexLimitAttr(64)
                _cd.CreateVoxelResolutionAttr(500000)   # 해상도↑ → 얇은 레일/칸 더 잘 잡음
                _cd.CreateErrorPercentageAttr(0.5)
            except Exception as _cde:
                print(f"[PHYSICS] convex decomp 튜닝 스킵(기본값 사용): {_cde}")
            _proxy_n += 1
        print(f"[PHYSICS] 사다리 콜리전 프록시 {_proxy_n}개 생성 (GeomSubset 우회)")
    except Exception as _pe:
        import traceback; traceback.print_exc()
        ENABLE_PHYSICS = False

_thruster_nozzles = []
_thruster_particles = []
if ENABLE_PHYSICS and ENABLE_THRUSTER_FX:
    try:
        _h = perc.V8_BODY_HALF
        _nozzle_defs = [
            ((+_h * 1.1, 0, 0), (+1, 0, 0)), ((-_h * 1.1, 0, 0), (-1, 0, 0)),
            ((0, +_h * 1.1, 0), (0, +1, 0)), ((0, -_h * 1.1, 0), (0, -1, 0)),
            ((0, 0, +_h * 1.1), (0, 0, +1)), ((0, 0, -_h * 1.1), (0, 0, -1)),
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
                _plist.append({"img": _imgp, "t": _top, "s": _sop, "c": _sph.GetDisplayColorAttr(), "life": random.uniform(0.0, 1.0), "vis": False})
            _thruster_particles.append(_plist)
    except Exception:
        _thruster_nozzles = []

def update_thruster_fx(chaser_pos, R, F_world, T_world, dt):
    if not _thruster_nozzles: return 0
    _n_active = 0
    for _ni, (r_b, e_b) in enumerate(_thruster_nozzles):
        e_w = R @ e_b
        r_w = R @ r_b
        a_f = max(0.0, float(np.dot(-e_w, F_world))) / THR_FORCE_REF
        a_t = max(0.0, float(np.dot(np.cross(r_w, -e_w), T_world))) / THR_TORQUE_REF
        act = min(1.0, a_f + a_t)
        plist = _thruster_particles[_ni]
        if act < 0.05:
            for p in plist:
                if p["vis"]: p["img"].MakeInvisible(); p["vis"] = False
            continue
        _n_active += 1
        nozzle_w = chaser_pos + r_w
        plume = THR_PLUME_LEN * act
        for p in plist:
            p["life"] += dt * random.uniform(2.5, 4.5)
            if p["life"] > 1.0: p["life"] = random.uniform(0.0, 0.15)
            tt = p["life"]
            jit = np.array([random.uniform(-0.03, 0.03) for _ in range(3)])
            pos = nozzle_w + e_w * (tt * plume) + jit
            p["t"].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
            sc = max(0.35, 1.0 - tt * 0.6) * (0.75 + 0.25 * act)
            p["s"].Set(Gf.Vec3f(sc, sc, sc))
            p["c"].Set([(1.0, max(0.0, 0.9 - tt * 1.2), 0.0)])
            if not p["vis"]: p["img"].MakeVisible(); p["vis"] = True
    return _n_active

_sim_ctx = None
_chaser_rb = None
robot = None

if ENABLE_PHYSICS:
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim
        _sim_ctx = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, stage_units_in_meters=1.0)
        _sim_ctx.get_physics_context().set_gravity(0.0)
        
        robot_root, _   = import_urdf(M0609_URDF_PATH, fix_base=False)
        gripper_root, _ = import_urdf(ONROBOT_URDF_PATH, fix_base=False)
        robot_ee_path     = find_prim_path_by_name(robot_root, EE_LINK_NAME) or f"{robot_root}/{EE_LINK_NAME}"
        gripper_base_path = find_prim_path_by_name(gripper_root, GRIPPER_BASE_LINK) or f"{gripper_root}/{GRIPPER_BASE_LINK}"
        assemble_robot(stage, robot_root, robot_ee_path, gripper_root, gripper_base_path, "Gripper", "m0609_rg2")
        for _ in range(10): simulation_app.update()

        articulation_prim = (find_articulation_root(robot_root) or find_articulation_root("/World") or find_articulation_root("/"))
        manipulator_prim_path = articulation_prim.rsplit("/", 1)[0] if articulation_prim.rsplit("/", 1)[-1] in ("root_joint", "rootJoint") else articulation_prim
        robot_ee_path  = find_prim_path_by_name(manipulator_prim_path, EE_LINK_NAME) or robot_ee_path
        base_link_path = find_prim_path_by_name(manipulator_prim_path, BASE_LINK_NAME) or f"{robot_root}/{BASE_LINK_NAME}"

        # [2단계] iss_berthing 손목캠 마운트: _final 조립 로봇은 그리퍼 angle_bracket 이
        # manipulator_prim_path 밖(gripper_root)에 있어 iss_berthing 이 못 찾음 → EE(link_6)
        # 자식으로 angle_bracket 프록시 Xform 생성(손목 위치, manipulator 아래라 검색됨).
        try:
            _ab_proxy_path = robot_ee_path + "/angle_bracket"
            if not stage.GetPrimAtPath(_ab_proxy_path).IsValid():
                _ab_xf = UsdGeom.Xform.Define(stage, _ab_proxy_path)
                UsdGeom.XformCommonAPI(_ab_xf.GetPrim()).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))
                print(f"[2단계] angle_bracket 프록시 생성: {_ab_proxy_path}")
        except Exception as _abe:
            print(f"[2단계] angle_bracket 프록시 생성 실패: {_abe}")

        _arm_joint = UsdPhysics.FixedJoint.Define(stage, CHASER_PATH + "/ArmMount")
        _arm_joint.GetBody0Rel().SetTargets([CHASER_PATH])
        _arm_joint.GetBody1Rel().SetTargets([base_link_path])
        _arm_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))      
        _arm_joint.GetLocalRot0Attr().Set(Gf.Quatf(0.5, -0.5, 0.5, 0.5))

        gripper = ParallelGripper(end_effector_prim_path=robot_ee_path, joint_prim_names=["finger_joint", "right_inner_knuckle_joint"],
                                  joint_opened_positions=np.array([0.0, 0.0]), joint_closed_positions=np.array([0.9, -0.9]), action_deltas=np.array([-0.9, 0.9]))
        robot = SingleManipulator(prim_path=manipulator_prim_path, name="m0609_robot", end_effector_prim_path=robot_ee_path, gripper=gripper)
        _sim_ctx.scene.add(robot)

        _sim_ctx.reset()

        # [2단계] iss_berthing 셋업 — _final 큐브(CHASER_PATH)+팔(robot)에 배선, 기존 정거장 사용
        # robot_prim_path 는 manipulator_prim_path 가 아니라 robot_root(/m0609) 를 넘긴다:
        #  iss_berthing 은 robot_prim_path 아래에서 angle_bracket/base_link/tool0 를 찾는데,
        #  _final 평탄계층(/m0609/base_link, /m0609/link_6 ...)이라 robot_root 여야 전부 잡힘.
        # robot_prim_path 는 manipulator_prim_path 가 아니라 robot_root(/m0609) 를 넘긴다:
        #  iss_berthing 은 robot_prim_path 아래에서 angle_bracket/base_link/tool0 를 찾는데,
        #  _final 평탄계층(/m0609/base_link, /m0609/link_6 ...)이라 robot_root 여야 전부 잡힘.
        _berth_robot_root = robot_root
        try:
            _ab_found = find_prim_path_by_name(_berth_robot_root, "angle_bracket")
            print(f"[2단계/진단] robot_root={robot_root} | manip={manipulator_prim_path} | ee={robot_ee_path}")
            print(f"[2단계/진단] berthing robot_prim_path={_berth_robot_root} | angle_bracket 검색결과={_ab_found}")
        except Exception as _de:
            print(f"[2단계/진단] 경로 진단 실패: {_de}")
        try:
            iss_berthing.setup_berthing(stage, _sim_ctx, None, None, simulation_app,
                                        SPACE_STATION_PATH, spawn_robot=False,
                                        robot_prim_path=_berth_robot_root,
                                        chaser_base_path=CHASER_PATH, passed_robot_art=robot)
            print("[2단계] iss_berthing 셋업 완료")
        except Exception as _be:
            print(f"[2단계] iss_berthing 셋업 실패 (ADR 전용으로 계속): {_be}")

        # [2단계] RealSense(angle_bracket 아래)의 물리 비활성화 — 관절 링크(link_6) 아래 nested
        #  rigid body 가 PhysX broadphase 를 깨는 것 방지 (셋업 직후 선제 처리)
        try:
            _rs_root = stage.GetPrimAtPath((_ab_found or "/__none__") + "/realsense_d455")
            if _rs_root and _rs_root.IsValid():
                _ndis = 0
                for _p in Usd.PrimRange(_rs_root):
                    if _p.HasAPI(UsdPhysics.RigidBodyAPI):
                        UsdPhysics.RigidBodyAPI(_p).GetRigidBodyEnabledAttr().Set(False); _ndis += 1
                    if _p.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(_p).GetCollisionEnabledAttr().Set(False)
                print(f"[2단계] RealSense 물리 비활성화 완료 (rigidbody {_ndis}개)")
            else:
                print("[2단계] RealSense prim 못 찾음 — 물리 비활성화 생략")
        except Exception as _rse:
            print(f"[2단계] RealSense 물리 비활성화 실패: {_rse}")

        _chaser_rb = RigidPrim(prim_paths_expr=CHASER_PATH, name="adr_chaser_rb")
        # [2단계] berthing 이 /World/Chaser(non-root articulation link)를 dc 대신 RigidPrim 으로
        #  힘/토크 제어하도록 핸들 공유 (Step1 ADR 에서 검증된 방식)
        iss_berthing.chaser_rb_external = _chaser_rb

        from pxr import Sdf as _Sdf2, UsdPhysics as _UP2
        _cp2 = stage.GetPrimAtPath(_Sdf2.Path(CHASER_PATH))
        _m2  = _UP2.MassAPI.Apply(_cp2)
        _m2.CreateMassAttr(float(CHASER_MASS))
        _m2.CreateDiagonalInertiaAttr().Set(
            Gf.Vec3f(
                CHASER_MASS * CHASER_BODY_SIZE**2 / 6.0,
                CHASER_MASS * CHASER_BODY_SIZE**2 / 6.0,
                CHASER_MASS * CHASER_BODY_SIZE**2 / 6.0,
            )
        )

        _q0 = perc.rotmat_to_quat_xyzw(perc.make_camera_rotation_from_forward(_ladder0 - chaser_pos)[0])
        _q0_wxyz = np.array([[_q0[3], _q0[0], _q0[1], _q0[2]]], dtype=float)
        _chaser_rb.set_world_poses(positions=np.array([chaser_pos], dtype=float), orientations=_q0_wxyz)

        robot.initialize()
        _init_arm_pos = chaser_pos + np.array([0.0, 0.35, 0.0], dtype=float)
        robot.set_world_pose(position=_init_arm_pos, orientation=_q0_wxyz[0])
        robot.gripper.initialize(physics_sim_view=_sim_ctx.physics_sim_view, articulation_apply_action_func=robot.apply_action,
                                 get_joint_positions_func=robot.get_joint_positions, set_joint_positions_func=robot.set_joint_positions, dof_names=robot.dof_names)
        init_arm = np.deg2rad(np.array([0.0, -45.0, 90.0, 0.0, 80.0, 90.0]))
        init_q = np.zeros(robot.num_dof); init_q[:6] = init_arm
        robot.set_joint_positions(init_q)
        _ac = robot.get_articulation_controller()
        _kps, _kds = _ac.get_gains()
        if _kps is not None and _kds is not None:
            for _i in range(6): _kps[_i] = 600.0; _kds[_i] = 600.0
            _kps[5] = 50.0; _kds[5] = 150.0
            _ac.set_gains(kps=_kps, kds=_kds)
        for _ in range(5): _sim_ctx.step(render=True)
        robot.set_world_pose(position=_init_arm_pos, orientation=_q0_wxyz[0])

        from isaacsim.core.utils.rotations import euler_angles_to_quat
        fpv_camera = Camera(prim_path=f"{robot_ee_path}/FPV_Camera", translation=np.array([0.0, 0.0, -0.1]),
                            orientation=euler_angles_to_quat(np.array([0.0, -90.0, -90.0]), degrees=True), resolution=(640, 480))
        fpv_camera.initialize(); fpv_camera.add_distance_to_image_plane_to_frame()
        
        ik_controller = PickPlaceController(name="ik_controller", gripper=robot.gripper, robot_articulation=robot, urdf_path=M0609_URDF_PATH,
                                            robot_description_path=str(BASE_DIR / "m0609_rg2_description.yaml"), rmpflow_config_path=str(BASE_DIR / "m0609_rmpflow_common.yaml"), end_effector_frame_name=EE_LINK_NAME)
        ik_controller.reset()
        _ee_w0 = UsdGeom.Xformable(stage.GetPrimAtPath(robot_ee_path)).ComputeLocalToWorldTransform(0)
        _q0_gf = _ee_w0.ExtractRotationQuat()
        _im0 = _q0_gf.GetImaginary()
        fixed_align_quat = np.array([_q0_gf.GetReal(), _im0[0], _im0[1], _im0[2]], dtype=float)
    except Exception as _re:
        import traceback
        print(f"[PHYSICS INIT ERROR] {_re}")
        traceback.print_exc()
        _sim_ctx = None; _chaser_rb = None; ENABLE_PHYSICS = False

def phys_read_chaser():
    pos, quat_wxyz = _chaser_rb.get_world_poses()
    vel = _chaser_rb.get_linear_velocities()
    p = np.asarray(pos[0], float)
    qw = np.asarray(quat_wxyz[0], float)
    q_xyzw = np.array([qw[1], qw[2], qw[3], qw[0]], float)
    return p, q_xyzw, np.asarray(vel[0], float)

def phys_set_linvel(vel_world): _chaser_rb.set_linear_velocities(np.asarray(vel_world, float).reshape(1, 3))
def phys_set_angvel(omega_world): _chaser_rb.set_angular_velocities(np.asarray(omega_world, float).reshape(1, 3))
def phys_get_angvel(): return np.asarray(_chaser_rb.get_angular_velocities()[0], float)
def phys_apply_torque(torque_world): _chaser_rb.apply_forces_and_torques_at_pos(torques=np.asarray(torque_world, float).reshape(1, 3), is_global=True)
def phys_apply_force(force_world): _chaser_rb.apply_forces_and_torques_at_pos(forces=np.asarray(force_world, float).reshape(1, 3), is_global=True)

timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(30):
    if ENABLE_PHYSICS and _sim_ctx is not None: _sim_ctx.step(render=True)
    else: simulation_app.update()

try:
    from omni.kit.viewport.utility import create_viewport_window
    _cam_vp_win = create_viewport_window("Chaser FrontCam", width=720, height=405)
    try: _cam_vp_win.viewport_api.set_active_camera(CHASER_CAM_PATH)
    except Exception: _cam_vp_win.viewport_api.camera_path = CHASER_CAM_PATH
except Exception as _ve: pass

try:
    from omni.kit.viewport.utility import create_viewport_window as _cvw3
    _tp_vp_win = _cvw3("3rd Person (Chaser + Ladder)", width=720, height=405)
    try: _tp_vp_win.viewport_api.set_active_camera(THIRD_PERSON_CAM_PATH)
    except Exception: _tp_vp_win.viewport_api.camera_path = THIRD_PERSON_CAM_PATH
except Exception as _ve2: pass

try:
    from omni.kit.viewport.utility import create_viewport_window as _cvw4
    _rcam_vp_win = _cvw4("Robot View", width=720, height=405)
    try: _rcam_vp_win.viewport_api.set_active_camera(ROBOT_CAM_PATH)
    except Exception: _rcam_vp_win.viewport_api.camera_path = ROBOT_CAM_PATH
except Exception as _ve3: pass

_orbit_rings_hidden = False
HIDE_ORBIT_RINGS_AFTER = 10.0

PATH_DOT_INTERVAL = 10.0   
PATH_DOT_RADIUS   = 0.15   
PATH_MAX_DOTS     = 400    
_pdot_last_pos    = None
_pdot_n_idx       = 0
_pdot_a_idx       = 0
UsdGeom.Xform.Define(stage, "/World/PathDots")

def _stamp_dot(color, pos):
    global _pdot_n_idx, _pdot_a_idx
    if color == "normal":
        pp = f"/World/PathDots/n{_pdot_n_idx % PATH_MAX_DOTS}"
        _pdot_n_idx += 1
        rgb = Gf.Vec3f(0.55, 0.0, 1.0)
    else:
        pp = f"/World/PathDots/a{_pdot_a_idx % PATH_MAX_DOTS}"
        _pdot_a_idx += 1
        rgb = Gf.Vec3f(1.0, 0.15, 0.15)
    prim = stage.GetPrimAtPath(pp)
    if not prim.IsValid():
        sph = UsdGeom.Sphere.Define(stage, pp)
        sph.GetRadiusAttr().Set(PATH_DOT_RADIUS)
        sph.GetDisplayColorAttr().Set([rgb])
        xf = UsdGeom.Xformable(sph.GetPrim())
        xf.ClearXformOpOrder()
        _t_op = xf.AddTranslateOp()
    else:
        sph = UsdGeom.Sphere(prim)
        sph.GetDisplayColorAttr().Set([rgb])
        _t_op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
    _t_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

# ============================================================
# [ROS 통합 — 1단계] mission_control 액션 서버
#   - "ladder capture" → IS_ADR_MODE=True (final 자율 ADR 가동)
#   - "grasp"/"deliver" → HOOK_PHASE (final 자율 진행을 모니터링)
#   - berthing 명령(start_approach/docking/undocking) → 2단계 예정(현재 통과)
#   handshake: iss_berthing.external_command / .command_completed / .current_feedback_status
# ============================================================
import iss_berthing  # (위 씬 import 직후에서 이미 로드됨 — 재import는 무해)
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
import threading, time

_INSTALL_DIR = os.path.join(WORKSPACE_DIR_TOP, "install", "mission_interfaces", "lib", "python3.11", "site-packages")
if _INSTALL_DIR not in sys.path:
    sys.path.append(_INSTALL_DIR)
try:
    from mission_interfaces.action import MissionControl
except ImportError as _mi_err:
    print(f"[ERROR] mission_interfaces import 실패: {_mi_err}")
    print(f"        탐색 경로 INSTALL_DIR = {_INSTALL_DIR}")
    print(f"        LD_LIBRARY_PATH(앞부분) = {os.environ.get('LD_LIBRARY_PATH','')[:160]}")
    print("        → Python 3.11용 mission_interfaces 빌드(install/)가 있는지 확인하세요.")
    simulation_app.close()   # ★ Isaac atexit 세그폴트 방지: 반드시 close 후 종료
    sys.exit(1)

IS_ADR_MODE = False
HOOK_PHASE = None           # None | "grasp" | "deliver"
_adr_command_completed = False

class MissionActionServer(Node):
    def __init__(self):
        super().__init__('mission_action_server')
        self._action_server = ActionServer(
            self, MissionControl, 'mission_control',
            execute_callback=self.execute_callback, cancel_callback=self.cancel_callback
        )
        self.current_cmd_id = 0
        print("[INTEGRATED] ROS 2 Action Server 'mission_control' 생성 완료. 클라이언트 명령 대기 중...")

    def cancel_callback(self, cancel_request):
        self.get_logger().info('⚠️ [ROS 2] 명령 취소 요청 수신!')
        return rclpy.action.CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        global IS_ADR_MODE, HOOK_PHASE, _adr_command_completed, ladder_path, state
        cmd = goal_handle.request.command
        self.current_cmd_id += 1
        my_cmd_id = self.current_cmd_id
        print(f"[ROS 2] 액션 목표 수신: {cmd} (ID: {my_cmd_id})")

        cmd_lower = cmd.lower()
        if "capture" in cmd_lower:
            IS_ADR_MODE = True
            iss_berthing.external_command = None
            iss_berthing.command_completed = False
            _adr_command_completed = False

            target_key = cmd_lower.split("capture")[0].strip().capitalize()
            if not target_key: target_key = "Ladder"
            adr_target_name = f"Debris{target_key}"

            state["ladder_grabbed"] = False
            state["ladder_handed_over"] = False
            state["delivering"] = False
            state["returning"] = False
            state["return_arrived"] = False
            try: state["dock_return_pos"] = _world_pos(CHASER_PATH).copy()
            except Exception: pass
            state["_phys_resync"] = True
            try: state["_phys_resync_pos"] = _world_pos(CHASER_PATH).copy()
            except Exception: state["_phys_resync_pos"] = None
            try: state["_phys_resync_R"] = _world_R(CHASER_PATH).copy()
            except Exception: state["_phys_resync_R"] = None

            ladder_path = f"/World/SpaceCleanupOrbitWorldV7/OrbitSystem/OrbitObjects/{adr_target_name}OrbitPivot/{adr_target_name}"
            iss_berthing.current_feedback_status = f"{adr_target_name} 랑데뷰 시작!"
            print(f"📡 [ROS] ADR 모드 활성화: {adr_target_name} 랑데뷰 시작!")
        elif cmd_lower in ("grasp", "deliver"):
            HOOK_PHASE = cmd_lower
            iss_berthing.command_completed = False
            iss_berthing.current_feedback_status = f"{'집기' if cmd_lower=='grasp' else '전달'} 단계 시작..."
            print(f"📡 [ROS] {cmd_lower} 훅 시작 (final 자율 진행 모니터링)")
        elif cmd_lower == "cancel":
            IS_ADR_MODE = False
            HOOK_PHASE = None
            try: update_thruster_fx(np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3), 0.01)
            except Exception: pass
            iss_berthing.external_command = "cancel"
            iss_berthing.current_feedback_status = "명령 취소됨. 대기 중..."
            iss_berthing.command_completed = True
        else:
            # berthing 명령 (start_approach/docking/undocking) — 2단계에서 구현, 지금은 메인루프 else에서 통과 처리
            IS_ADR_MODE = False
            iss_berthing.external_command = cmd.replace(" ", "_")
            iss_berthing.command_completed = False

        feedback_msg = MissionControl.Feedback()
        start_time = time.time()
        while True:
            if self.current_cmd_id != my_cmd_id:
                goal_handle.abort()
                result = MissionControl.Result(); result.success = False
                result.message = "새 명령 수신으로 기존 명령 취소됨."
                return result
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = MissionControl.Result(); result.success = False
                result.message = "사용자에 의해 취소됨."
                if IS_ADR_MODE:
                    IS_ADR_MODE = False
                    try: update_thruster_fx(np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3), 0.01)
                    except Exception: pass
                iss_berthing.external_command = "cancel"
                iss_berthing.current_feedback_status = "명령 취소됨. 대기 중..."
                iss_berthing.command_completed = True
                return result
            feedback_msg.current_status = iss_berthing.current_feedback_status
            goal_handle.publish_feedback(feedback_msg)
            if iss_berthing.command_completed:
                iss_berthing.command_completed = False
                break
            time.sleep(0.1)
            if time.time() - start_time > 300.0:
                goal_handle.abort()
                result = MissionControl.Result(); result.success = False
                result.message = "Timeout."
                return result

        goal_handle.succeed()
        result = MissionControl.Result(); result.success = True
        result.message = f"'{cmd}' 명령 수행 성공!"
        return result

rclpy.init()
ros_node = MissionActionServer()
executor = MultiThreadedExecutor()
executor.add_node(ros_node)
ros_thread = threading.Thread(target=executor.spin, daemon=True)
ros_thread.start()

try:
    while simulation_app.is_running():
        dt = 1.0 / 60.0
        if orbit_step is not None: orbit_step(dt)

        if IS_ADR_MODE:
            on_update(dt)
            arm_step()
            r2d2_manager.update(dt, chaser_pos, ladder_path, state.get("ladder_handed_over"))

            # ── ROS 명령 완료 모니터 (final 자율 진행을 감시해 command_completed 세팅) ──
            if HOOK_PHASE == "grasp":
                iss_berthing.current_feedback_status = "[ADR] 집기 진행중..."
                if state.get("ladder_grabbed"):
                    iss_berthing.current_feedback_status = "✅ 집기 완료!"
                    iss_berthing.command_completed = True
                    HOOK_PHASE = None
            elif HOOK_PHASE == "deliver":
                iss_berthing.current_feedback_status = "[ADR] 전달 진행중..."
                if state.get("ladder_handed_over"):
                    iss_berthing.current_feedback_status = "✅ 전달 완료!"
                    iss_berthing.command_completed = True
                    HOOK_PHASE = None
            else:
                # capture 완료 판정 = 랑데뷰(creep + gap 작음)
                if state.get("v8_substage") == "creep" and state.get("v8_body_gap", 999.0) < perc.V8_MIN_GAP + 0.1:
                    if not _adr_command_completed:
                        iss_berthing.current_feedback_status = "✅ 랑데뷰 완료! 로봇팔 제어 대기 중..."
                        iss_berthing.command_completed = True
                        _adr_command_completed = True
                elif state.get("v8_engaged", False):
                    iss_berthing.current_feedback_status = f"[ADR] 근접 | 남은 틈새: {state.get('v8_body_gap', 0.0):.1f}m"
                else:
                    _d = float(np.linalg.norm(_world_pos(ladder_path) - chaser_pos))
                    iss_berthing.current_feedback_status = f"[ADR] 궤도 접근중 | 남은 거리: {_d:.1f}m"

            _dot_pos = chaser_pos.copy()
            if _pdot_last_pos is None or float(np.linalg.norm(_dot_pos - _pdot_last_pos)) >= PATH_DOT_INTERVAL:
                _stamp_dot("avoid" if state.get("avoid_active") else "normal", _dot_pos)
                _pdot_last_pos = _dot_pos.copy()
        else:
            # ── berthing 모드 (도킹/언도킹) — iss_berthing 가 dc 힘제어로 체이서 큐브 구동 ──
            iss_berthing.step_berthing(dt)
            # 대기 중에도 별도 제어 안 함: /World/Chaser 는 ArmMount 로 팔과 한 articulation 의
            #  non-root 링크라 set_*velocities/transform 이 안 먹음(힘만 가능). zero-g 라 그냥 떠 있음.

        if (not _orbit_rings_hidden) and _sim_t > HIDE_ORBIT_RINGS_AFTER:
            _rings = stage.GetPrimAtPath(scene.ORBIT_PATHS_PATH)
            if _rings and _rings.IsValid(): UsdGeom.Imageable(_rings).MakeInvisible(); _orbit_rings_hidden = True

        if ENABLE_PHYSICS and _sim_ctx is not None: _sim_ctx.step(render=True)
        else: simulation_app.update()

except KeyboardInterrupt:
    print("\n[INFO] 사용자가 강제 종료(Ctrl+C)했습니다.")
except Exception as e:
    print(f"\n[ERROR] 메인 루프 실행 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n[SHUTDOWN] 안전 종료 시퀀 가동...")
    # ROS 먼저 정리 (executor 스레드/rclpy 를 SimulationApp close 전에 내려야 종료 세그폴트 방지)
    try:
        executor.shutdown()
        ros_node.destroy_node()
    except Exception: pass
    try:
        if rclpy.ok(): rclpy.shutdown()
    except Exception: pass
    try:
        omni.timeline.get_timeline_interface().stop()
        if rep is not None:
            if rgb_annotator is not None:
                try: rgb_annotator.detach([_rgb_rp])
                except: pass
            if '_rgb_rp' in globals() and _rgb_rp is not None:
                try: 
                    _rgb_rp.destroy() 
                    print(" -> 렌더 프로덕트 안전 파괴 완료")
                except: pass
        for _ in range(5):
            simulation_app.update()
    except Exception as e:
        print(f" -> 해제 중 경고 (무시 가능): {e}")

    print(" -> Isaac Sim 닫기 진행 중...")
    simulation_app.close()