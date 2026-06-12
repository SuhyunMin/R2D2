# scripts/space_environment_v10_leo.py
#
# Purpose:
#   v10 = v9 physics-gravity world, re-expressed in SCALED LEO units.
#   Kepler/Newton relations are exact; only the unit scale is shrunk so the
#   orbit is visible and float precision stays sane (path 2, agreed).
#
#   Conversion table (see LEO block below):
#     1 sim unit = 287.12 km   (real 7178 km orbit -> sim 25 units)
#     1 sim sec  = 60 real sec (60x time acceleration -> ~101 s per orbit)
#     GM_sim     = 60.62       (derived so v=sqrt(GM/r), T=2*pi*sqrt(r^3/GM) hold)
#   Orbit lanes are now defined by REAL ALTITUDE (km); radius/speed/period follow
#   Kepler automatically. Earth is a perfect sphere; no J2/drag/SRP (by choice).
#
#   What changed vs space_environment_v9_assets.py:
#     - Each orbiting object (5 debris + 5 satellites + asteroid + station) is now
#       a UsdPhysics RigidBody with explicit mass.
#     - NO collision geometry is attached on purpose -> bodies behave like point
#       masses and never bump/scatter each other (clean, stable orbits).
#     - The kinematic "rotate the pivot every frame" animation is GONE. Instead:
#         * each body is placed at its real world position (radius + phase + tilt)
#         * each body gets a tangential initial velocity v = sqrt(GM / r)
#         * every physics step a central gravity force F = GM*m / r^2 toward Earth
#           (origin) is applied via the PhysX simulation interface
#     - Earth stays a pure visual anchor at the origin (no rigid body), so it acts
#       as a fixed gravity source (one-way gravity, like the validated test code).
#
# Classes (semantics unchanged):
#     collectable_debris : rocket_stage, ladder, helmet, walle, sat_body (5)
#     asteroid           : meteorite (1)
#     protected_satellite: space_satellite x5
#     station            : NOT a detection class (base/landmark only)
#
# HOW TO RUN (Isaac Sim Script Editor):
#   exec(open("/home/rokey/space_debris_ai/scripts/space_environment_v9_physics.py").read())
#   ...then press the ▶ PLAY button. Physics (and the orbit) only runs while playing.
#   To stop the gravity loop:  _SPACE_GRAVITY_SUB = None
#
# NOTE: GM / radii / velocities will almost certainly need tuning to taste.
#       The single most important knob is ORBIT_GM below.

import os
import math
import time
import asyncio
import zipfile
import shutil
import random

# ===== SimulationApp: 이 파일을 '직접 실행'할 때만 여기서 생성한다. =====
#   다른 스크립트(통합 진입점)가 이 파일을 import할 때는, 그쪽에서 SimulationApp을
#   먼저 만든 뒤 import해야 한다(SimulationApp은 프로세스당 1개만 가능).
if __name__ == "__main__":
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})
    try:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.kit.asset_converter")
    except Exception as _e:
        print("[STANDALONE WARN] asset_converter 확장 활성화 실패:", _e)
        print("  -> 변환이 안 되면 USD를 한 번 만든 뒤 FORCE_RECONVERT_ASSETS=False 로 실행하세요.")

# omni/pxr import: 이 시점에 SimulationApp이 이미 생성돼 있어야 한다.
#   - 직접 실행: 바로 위 블록에서 생성됨
#   - import 되는 경우: 호출자가 import 전에 SimulationApp을 만들어 둔다
import omni.usd
import omni.kit.app
import omni.timeline
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdPhysics, PhysxSchema, Semantics


# ============================================================
# PATH CONFIG
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(THIS_DIR, "..", "resources")
RAW_ASSET_DIR = os.path.join(PROJECT_DIR, "assets", "raw")
USD_ASSET_DIR = os.path.join(PROJECT_DIR, "assets", "usd")
OUTPUT_STAGE_PATH = os.path.join(PROJECT_DIR, "outputs", "space_environment_v15_satellite_meteor_axis_only.usda")

EARTH_GLB = os.path.join(RAW_ASSET_DIR, "earth.glb")
STATION_GLB = os.path.join(RAW_ASSET_DIR, "space_station.glb")
SATELLITE_GLB = os.path.join(RAW_ASSET_DIR, "space_satellite.glb")

EARTH_USD = os.path.join(USD_ASSET_DIR, "earth.usd")
STATION_USD = os.path.join(USD_ASSET_DIR, "space_station.usd")
SATELLITE_USD = os.path.join(USD_ASSET_DIR, "space_satellite.usd")

# New USDZ assets requested in v14.
# Put these files under ~/space_debris_ai/assets/usd/ before running:
#   - Meteor-M2_No.usdz
#   - Satellite_lnb_Building_roof_top.usdz
METEOR_M2_USDZ = os.path.join(USD_ASSET_DIR, "Meteor-M2_No.usdz")
ROOFTOP_USDZ = os.path.join(USD_ASSET_DIR, "Satellite_lnb_Building_roof_top.usdz")
METEOR_M2_USD = os.path.join(USD_ASSET_DIR, "Meteor-M2_No.usd")
ROOFTOP_USD = os.path.join(USD_ASSET_DIR, "Satellite_lnb_Building_roof_top.usd")

DEBRIS_GLBS = {
    "rocket_stage": os.path.join(RAW_ASSET_DIR, "nexus_1st_stage.glb"),
    "ladder":       os.path.join(RAW_ASSET_DIR, "ladder_metallic_tool.glb"),
    "helmet":       os.path.join(RAW_ASSET_DIR, "nasa_astronaut_helmet.glb"),
    "walle":        os.path.join(RAW_ASSET_DIR, "wall-e.glb"),
    "sat_body":     os.path.join(RAW_ASSET_DIR, "sat02_body_satellite.glb"),
}
ASTEROID_GLB = os.path.join(RAW_ASSET_DIR, "meteorite.glb")

# USDZ assets that Isaac Sim may fail to Add Reference directly.
# 이 파일들은 /assets/usd 안에 둔 상태에서 wrapper USD로 변환해서 참조한다.
UFO_USDZ = os.path.join(USD_ASSET_DIR, "UFO.usdz")
R2D2_USDZ = os.path.join(USD_ASSET_DIR, "R2D2.usdz")
UFO_USD = os.path.join(USD_ASSET_DIR, "UFO.usd")
R2D2_USD = os.path.join(USD_ASSET_DIR, "R2D2.usd")

DEBRIS_USDS = {k: os.path.join(USD_ASSET_DIR, f"debris_{k}.usd") for k in DEBRIS_GLBS}
ASTEROID_USD = os.path.join(USD_ASSET_DIR, "meteorite.usd")

ROOT_PATH = "/World/SpaceCleanupOrbitWorldV7"
ORBIT_SYSTEM_PATH = f"{ROOT_PATH}/OrbitSystem"
EARTH_PATH = f"{ORBIT_SYSTEM_PATH}/Earth"
ORBIT_PATHS_PATH = f"{ORBIT_SYSTEM_PATH}/OrbitPaths"
ORBIT_OBJECTS_PATH = f"{ORBIT_SYSTEM_PATH}/OrbitObjects"
STATION_PIVOT_PATH = f"{ORBIT_OBJECTS_PATH}/StationOrbitPivot"
SPACE_STATION_PATH = f"{STATION_PIVOT_PATH}/SpaceStation"
ROBOT_PATH = f"{SPACE_STATION_PATH}/FreeFlyingRobot_Docked"
ROBOT_CAMERA_PATH = f"{ROBOT_PATH}/FrontCamera"
PERCEPTION_PATH = f"{ROOT_PATH}/PerceptionLayer"
MISSION_PATH = f"{ROOT_PATH}/MissionLogicLayer"
DEBUG_PATH = f"{ROOT_PATH}/DebugLayer"
LIGHTS_PATH = f"{ROOT_PATH}/Lights"
PHYSICS_SCENE_PATH = "/World/PhysicsScene_ZeroG_OrbitCleanupV7"
OVERVIEW_CAMERA_PATH = f"{ROOT_PATH}/OverviewCamera"


# ============================================================
# SCALE / ORBIT CONTROL TABLE   <-- 여기 숫자만 바꾸면 됨
# ============================================================
#
# ABSOLUTE scale: GLB 원본 크기에 그대로 곱해짐. 정규화 없음.

FORCE_RECONVERT_ASSETS = True   # merge 적용 위해 재변환 (안정화되면 False로)

# ============================================================
# LEO 스케일 환산표 (경로2: 물리 법칙은 실제, 스케일만 축소)
# ============================================================
#   실제 저궤도(LEO)를 케플러/뉴턴 관계를 정확히 보존한 채 보기 좋은 크기로 축소.
#   "1 sim unit = ? km", "1 sim sec = ? real sec" 를 명시 -> 모든 거동이 실제 LEO의
#   정확한 축소판이 된다 (포트폴리오에서 환산 관계로 정당화 가능).
#
#   실제 기준값 (SI):
#     GM_real      = 3.986e14 m^3/s^2   (지구 표준중력파라미터)
#     R_earth_real = 6378 km            (완벽한 구 가정, 섭동 없음)
#     기준 고도     = 800 km  -> r_real = 7178 km
#     실제 속도     ~ 7.452 km/s,  실제 주기 ~ 100.9 분
#
#   축척 선택:
#     길이: 실제 7178 km  ->  시뮬 25.0 units   => 1 unit = 287.12 km
#     시간: 실제 60 초     ->  시뮬 1.0 초       => 60x 가속 (1바퀴 ~101초로 관측 가능)
#
#   환산식: v_sim = v_real * TIME_ACCEL / LEN_SCALE,  GM_sim = v_sim^2 * r_sim
#   -> GM_sim = 60.62 (아래). 주기 검증: 2*pi*sqrt(r^3/GM_sim) = 100.9s = 실제/60. OK.

import math as _math

# --- 실제 LEO 기준 (SI, 고정) ---
GM_REAL = 3.986e14          # m^3/s^2
R_EARTH_REAL_M = 6378e3     # m (완벽한 구)
REF_ALT_KM = 800.0          # 기준 고도 (km)

# --- 축척 정의 (이 두 줄이 환산의 전부) ---
# v11: 궤도를 실사 비례로 크게 (지구 시각앵커보다 넉넉히 멀게). 키네마틱이라
#      커져도 부동소수점 문제 없음.
# [B안] 800km 기준궤도를 500 units로 -> 궤도 482~615m. ORBIT_GM/omega가 아래 환산식으로
#       자동 재계산되어 주기는 100.9초 그대로 유지됨(반지름만 10배). 1 unit = 14.36 km.
SIM_R_AT_REF = 400.0        # 기준 고도(800km) 궤도를 시뮬 몇 units로 둘지 (-> 레인 386~492m)
# [속도 맞춤] 선속도 ∝ SIM_R_AT_REF*TIME_ACCEL. 반지름 400급에서도 사다리 2.8m/s 유지하려고
#   TIME_ACCEL=6.75 (=5.4*500/400). 전 레인 2.5~2.85m/s -> v11 사다리(2.8)와 일치. 케플러 유지.
#   주기 14~20분. 더 느리게: TIME_ACCEL 낮춤(예 5.4면 2.0~2.3m/s).
TIME_ACCEL = 6.75           # 시간 가속: 실제 N초 = 시뮬 1초

# --- 환산 계수 (자동 유도, 직접 수정 금지) ---
_r_ref_real_m = R_EARTH_REAL_M + REF_ALT_KM * 1000.0     # 기준 궤도 반지름 (m)
LEN_SCALE_M_PER_UNIT = _r_ref_real_m / SIM_R_AT_REF      # 1 sim unit = ? m  (= 287120 m)

def alt_km_to_sim_radius(alt_km):
    """실제 고도(km) -> 시뮬 궤도 반지름(units). 환산표 그대로 적용."""
    r_real_m = R_EARTH_REAL_M + alt_km * 1000.0
    return r_real_m / LEN_SCALE_M_PER_UNIT

# --- 시뮬 단위 GM (케플러 관계 보존하도록 유도) ---
# 기준 고도에서 실제 속도를 시뮬 단위로 환산 -> GM_sim = v_sim^2 * r_sim
_v_ref_real = _math.sqrt(GM_REAL / _r_ref_real_m)                  # 실제 속도 m/s
_v_ref_sim = _v_ref_real * TIME_ACCEL / LEN_SCALE_M_PER_UNIT       # 시뮬 속도 units/s
ORBIT_GM = (_v_ref_sim ** 2) * SIM_R_AT_REF                        # = 60.62

FORCE_RECONVERT_ASSETS = True   # merge 적용 위해 재변환 (안정화되면 False로)

# 각 물체 질량(kg). 궤도 자체는 질량과 무관(a = GM/r^2)하지만 힘 계산에 쓰임.
BODY_MASS = 1.0
# 자전: 지구만 자전한다. 나머지 물체(쓰레기/위성/소행성/정거장)는 자전 없음.
# 실제 지구 자전 주기 24h 를 시간가속(60x) 적용하면 비현실적으로 느리므로,
# 시각적 효과만을 위한 임의 값으로 둔다 (자전은 궤도 물리와 무관).
EARTH_SPIN_DEG_PER_SEC = 5.0

# ---- 자전(tumbling) / 조석 고정(tidal lock) 제어 ----
# 쓰레기·운석: 느리게 텀블링. self_spin(deg/궤도?)이 아니라 절대 deg/sim-sec로 제어.
# TUMBLE_SPEED_SCALE 로 전체 텀블링 속도를 한 번에 조절 (작을수록 느림).
TUMBLE_SPEED_SCALE = 0.1      # 텀블링 전체 속도 배율 (0.15 = 꽤 느리게)
TUMBLE_DEG_PER_SEC = 8.0        # 기준 텀블링 속도(deg/sim-sec). 실제속도 = 이값 * scale
# v11 타깃 사다리와 동일한 비협조 텀블링(DebrisLadder 전용). roll/pitch/yaw 절대 deg/s.
# = (0.225, 0.16, 0.275) rad/s. TUMBLE_SPEED_SCALE 안 곱함(v11과 1:1 일치 위해).
LADDER_TUMBLE_DEG_PER_SEC = (12.892, 9.167, 15.756)
# 위성·정거장: 조석 고정(공전하며 항상 같은 면이 지구를 향함). 자전=공전 각속도.

# 지구 / 정거장 / 보호위성 절대 스케일 (새 기준: 두산 팔=1단위 ≈ 1m, 즉 1 unit ≈ 1 m)
EARTH_SCALE = 0.96        # 지구는 시각 앵커 (물리 무관). 0.08 -> 0.32 -> 0.96 (추가 3배)
STATION_SCALE = 2.5414    # 정거장 ~30m (실제 100m+이나 화면 위해 축소)
SATELLITE_SCALE = 0.7383  # 보호위성 본체 ~3m

# v14 USDZ replacement assets.
# Meteor-M2 replaces ProtectedSatellite_01~05 visuals.
# Rooftop antenna is added as an extra collectable debris object.
# If the imported USDZ looks too big/small, tune only these two values.
METEOR_M2_SCALE = 0.008
ROOFTOP_DEBRIS_SCALE = 0.001

# 쓰레기 5종 절대 스케일 (새 기준 1 unit ≈ 1m, 두산 팔 reach ~1m로 잡기 가능하게)
DEBRIS_SCALES = {
    "rocket_stage": 0.0656,   # ~4 m (제일 큼, 팔로 한 지점 잡기)
    "ladder":       0.015,    # v11 타깃과 동일 크기(~4.8m) — keypoints_3d(0.015 기준)와 정합
    "helmet":       1.6260,   # ~0.6 m (제일 작음)
    "walle":        0.8547,   # ~1 m
    "sat_body":     0.1964,   # ~2 m
}
ASTEROID_SCALE = 0.4170       # ~3 m

# USDZ로 받은 장난감/캐릭터성 오브젝트.
# 원본마다 크기가 제각각이라 여기서만 조절하면 됨.
UFO_SCALE = 0.005
R2D2_SCALE = 0.005
UFO_ELLIPSE_MAJOR_SCALE = 3.0   # 긴 방향 반지름: 기존 UFO 궤도 반지름의 3배
UFO_ELLIPSE_MINOR_SCALE = 0.18  # 짧은 방향 반지름: 작을수록 직선처럼 보임

# 궤도 레인: 이제 '실제 고도(km)'로 정의하고, 환산표로 시뮬 반지름을 자동 계산.
# 고도가 다르면 케플러 법칙대로 속도/주기가 자동으로 달라진다 (v=sqrt(GM/r)).
ORBIT_ALTITUDES_KM = {
    "inner_debris":   550.0,   # rocket  (낮은 고도 = 빠름)
    "mid_debris":     780.0,   # ladder
    "outer_debris":  1050.0,   # helmet
    "extra_debris_1": 1450.0,  # walle
    "extra_debris_2": 1700.0,  # sat_body
    "sat_1":          900.0,
    "sat_2":         1180.0,
    "sat_3":         1330.0,
    "sat_4":         1580.0,
    "sat_5":         2050.0,

    # special assets: 기존 애들과 반지름이 안 겹치고, 궤도면도 크게 기울일 예정
    "r2d2_special":   665.0,   # rocket(550)과 ladder(780) 사이
    "rooftop_debris": 1900.0,  # rooftop antenna debris: sat_4/sat_5 사이
    "ufo_special":    2250.0,  # sat_5(2050)와 station(2450) 사이
    "station":        2450.0,  # station (가장 바깥)
}
# 고도 -> 시뮬 반지름으로 변환 (기존 코드가 ORBIT_LANES[...]를 그대로 쓰므로 호환 유지)
ORBIT_LANES = {k: alt_km_to_sim_radius(v) for k, v in ORBIT_ALTITUDES_KM.items()}

ORBIT_DOT_COUNT = 260
CAMERA_TRANSLATE = (0.0, -240.0, 120.0)   # 맵 400m급에 맞춰 (rotate 유지)
CAMERA_ROTATE = (57.0, 0.0, 0.0)
ROBOT_DOCK_LOCAL_POS = (0.0, -0.62, 0.05)


# ============================================================
# BASIC HELPERS
# ============================================================

def get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is currently open.")
    return stage


def remove_if_exists(stage, path):
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        stage.RemovePrim(path)


def create_xform(stage, path):
    return UsdGeom.Xform.Define(stage, path).GetPrim()


def set_transform(prim, translate=(0, 0, 0), rotate=(0, 0, 0), scale=(1, 1, 1)):
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*translate))
    api.SetRotate(Gf.Vec3f(*rotate), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    api.SetScale(Gf.Vec3f(*scale))


def set_rotation(prim, rotate):
    UsdGeom.XformCommonAPI(prim).SetRotate(Gf.Vec3f(*rotate), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def rot_x(vec, angle_rad):
    """Rotate a 3-vector about the X axis."""
    x, y, z = vec
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return (x, y * c - z * s, y * s + z * c)


def rot_z(vec, angle_rad):
    """Rotate a 3-vector about the Z axis."""
    x, y, z = vec
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return (x * c - y * s, x * s + y * c, z)


def rot_orbit(vec, tilt_x_rad, tilt_z_rad=0.0):
    """Rotate an orbit plane in two directions.

    Old code only used rot_x(), so all inclined orbits shared the same node line.
    v14 first tilts the XY orbit around X, then spins that tilted plane around Z.
    This makes satellites/meteors move upward/downward/diagonal in visibly
    different directions.
    """
    return rot_z(rot_x(vec, tilt_x_rad), tilt_z_rad)


def make_material(stage, path, color, metallic=0.0, roughness=0.35):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def bind_material(prim, mat):
    UsdShade.MaterialBindingAPI(prim).Bind(mat)


def add_semantic(prim, class_name):
    sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    sem.CreateSemanticTypeAttr()
    sem.CreateSemanticDataAttr()
    sem.GetSemanticTypeAttr().Set("class")
    sem.GetSemanticDataAttr().Set(class_name)


def make_rigid_body(prim, mass=BODY_MASS, lin_vel=(0, 0, 0), ang_vel=(0, 0, 0)):
    """Turn a prim into a dynamic rigid body with explicit mass and an initial
    velocity. NO collider is added on purpose: the body responds only to the
    gravity force we apply, and never collides with anything (no scatter).

    Rotation is fully LOCKED so the body cannot self-rotate. The central gravity
    force is applied at the prim origin, which may not match the imported mesh's
    center of mass; that offset produces a small torque every step and, with no
    angular damping in space, the spin runs away (the '10 spins/sec' bug). We
    (a) pin the center of mass to the body origin and (b) lock all rotational
    DOFs, so the bodies only orbit (translate) and keep a fixed orientation."""
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateVelocityAttr(Gf.Vec3f(*lin_vel))          # m/s (initial linear velocity)
    rb.CreateAngularVelocityAttr(Gf.Vec3f(*ang_vel))   # deg/s (kept 0 for orbiters)

    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))  # COM at body origin -> no torque

    # 회전 자유도 완전 잠금 (X|Y|Z = 1|2|4 = 7) -> 토크가 들어와도 자전 안 함
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    px.CreateLockedRotAxisAttr(7)
    return rb


# ============================================================
# SELECTIVE PHYSICS / COLLIDER BUBBLE
#   - Station: Kinematic RigidBody + Collider
#   - Robot docked to station: collider included in station compound collider
#   - Collectable debris: Kinematic RigidBody + Collider at first
#   - Released debris: Dynamic RigidBody, removed from kinematic orbit update
#   - Protected satellites / asteroid / visual-only objects: no collider
# ============================================================

def _create_or_get_attr(prim, name, type_name, value):
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, type_name, custom=False)
    attr.Set(value)
    return attr


def create_bouncy_physics_material(stage):
    """Low-friction, high-restitution material so released debris visibly bounces."""
    mat_path = f"{ROOT_PATH}/Materials/BouncyPhysics"
    mat = UsdShade.Material.Define(stage, mat_path)

    try:
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        phys_mat.CreateStaticFrictionAttr(0.0)
        phys_mat.CreateDynamicFrictionAttr(0.0)
        phys_mat.CreateRestitutionAttr(0.85)
        print("[PHYSICS] Bouncy physics material created.")
    except Exception as e:
        print("[WARN] Could not create UsdPhysics.MaterialAPI:", e)

    return mat


def apply_collision_recursive(prim, physics_material=None):
    """Apply CollisionAPI to all geometry under prim.

    The rigid body is applied at the clean object root, while collision is applied
    to child meshes/proxies. This is the correct pattern for referenced GLB/USD
    assets whose visible geometry lives under /Model.
    """
    if not prim or not prim.IsValid():
        return 0

    count = 0

    def visit(p):
        nonlocal count

        try:
            if p.IsA(UsdGeom.Gprim):
                UsdPhysics.CollisionAPI.Apply(p)
                count += 1

                if physics_material is not None:
                    try:
                        UsdShade.MaterialBindingAPI(p).Bind(physics_material, UsdShade.Tokens.physics)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[WARN] CollisionAPI failed on {p.GetPath()}: {e}")

        for child in p.GetChildren():
            visit(child)

    visit(prim)

    # Fallback: if no Gprim was found, apply CollisionAPI to root once.
    if count == 0:
        try:
            UsdPhysics.CollisionAPI.Apply(prim)
            count = 1
        except Exception as e:
            print(f"[WARN] CollisionAPI root fallback failed on {prim.GetPath()}: {e}")

    return count


def apply_selective_rigid_body(
    prim,
    *,
    kinematic=True,
    mass=1.0,
    disable_gravity=True,
    lock_rotation=False,
    physics_material=None
):
    """Apply rigid body + recursive collider.

    kinematic=True  : scripted orbit object that can collide with dynamic debris.
    kinematic=False : free physics object.
    """
    if not prim or not prim.IsValid():
        return

    collider_count = apply_collision_recursive(prim, physics_material=physics_material)

    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    rb.CreateAngularVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))

    # Kinematic flag is important: station/debris keep their scripted orbit until released.
    _create_or_get_attr(
        prim,
        "physics:kinematicEnabled",
        Sdf.ValueTypeNames.Bool,
        bool(kinematic)
    )

    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))

    try:
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        px.CreateDisableGravityAttr(bool(disable_gravity))
        px.CreateLinearDampingAttr(0.0)
        px.CreateAngularDampingAttr(0.0)

        if lock_rotation:
            px.CreateLockedRotAxisAttr(7)
        else:
            px.CreateLockedRotAxisAttr(0)
    except Exception as e:
        print(f"[WARN] PhysxRigidBodyAPI attrs failed on {prim.GetPath()}: {e}")

    print(
        f"[PHYSICS] {prim.GetPath()} | "
        f"kinematic={kinematic}, mass={mass}, colliders={collider_count}"
    )


def set_body_kinematic(path, enabled):
    stage = get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        print("[WARN] set_body_kinematic failed. Missing prim:", path)
        return

    _create_or_get_attr(
        prim,
        "physics:kinematicEnabled",
        Sdf.ValueTypeNames.Bool,
        bool(enabled)
    )


def set_body_velocity(path, linear_velocity=(0, 0, 0), angular_velocity=(0, 0, 0)):
    stage = get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        print("[WARN] set_body_velocity failed. Missing prim:", path)
        return

    lv = Gf.Vec3f(float(linear_velocity[0]), float(linear_velocity[1]), float(linear_velocity[2]))
    av = Gf.Vec3f(float(angular_velocity[0]), float(angular_velocity[1]), float(angular_velocity[2]))

    _create_or_get_attr(prim, "physics:velocity", Sdf.ValueTypeNames.Vector3f, lv)
    _create_or_get_attr(prim, "physics:angularVelocity", Sdf.ValueTypeNames.Vector3f, av)


def _current_world_position(path):
    stage = get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None

    try:
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = m.ExtractTranslation()
        return (float(t[0]), float(t[1]), float(t[2]))
    except Exception:
        return None


def _norm3(v):
    x, y, z = v
    n = math.sqrt(x*x + y*y + z*z)
    if n < 1e-8:
        return (1.0, 0.0, 0.0)
    return (x/n, y/n, z/n)


def estimate_kinematic_orbit_velocity(body_path):
    """Estimate current scripted-orbit velocity from the stored Kepler params."""
    kin = globals().get("_KINEMATIC_BODIES", [])
    t = globals().get("_KINEMATIC_TIME", 0.0)

    for b in kin:
        if b["path"] != body_path:
            continue

        r = b["r"]
        phi = b["phi0"] + b["omega"] * t
        omega = b["omega"]

        # d/dt of (r*cos(phi), r*sin(phi), 0)
        v = (-r * omega * math.sin(phi), r * omega * math.cos(phi), 0.0)
        v = rot_x(v, b["tilt"])
        return v

    return (0.0, 0.0, 0.0)


def release_collectable_debris(
    name_or_path,
    *,
    push_speed=3.0,
    toward_station=False,
    angular_speed=1.5
):
    """Switch one collectable debris from scripted kinematic orbit to dynamic physics.

    Usage examples in Script Editor after PLAY:
        release_collectable_debris("DebrisHelmet")
        release_collectable_debris("DebrisHelmet", toward_station=True, push_speed=5.0)

    If it hits the kinematic station collider after release, it should bounce/deflect.
    """
    stage = get_stage()

    by_name = globals().get("_COLLECTABLE_DEBRIS_BY_NAME", {})
    path = by_name.get(name_or_path, name_or_path)

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        print("[WARN] release_collectable_debris failed. Unknown debris:", name_or_path)
        print("[WARN] Available:", sorted(by_name.keys()))
        return

    # Stop scripted orbit update for this debris.
    if "_PHYSICS_BUBBLE_PATHS" not in globals():
        globals()["_PHYSICS_BUBBLE_PATHS"] = set()
    globals()["_PHYSICS_BUBBLE_PATHS"].add(path)

    # Change kinematic -> dynamic.
    set_body_kinematic(path, False)

    pos = _current_world_position(path)
    orbit_v = estimate_kinematic_orbit_velocity(path)

    if toward_station and pos is not None:
        station_pos = _current_world_position(SPACE_STATION_PATH)
        if station_pos is not None:
            direction = _norm3((
                station_pos[0] - pos[0],
                station_pos[1] - pos[1],
                station_pos[2] - pos[2],
            ))
        else:
            direction = _norm3(orbit_v)
    else:
        direction = _norm3(orbit_v)

    linear_velocity = (
        orbit_v[0] + direction[0] * push_speed,
        orbit_v[1] + direction[1] * push_speed,
        orbit_v[2] + direction[2] * push_speed,
    )

    angular_velocity = (
        random.uniform(-angular_speed, angular_speed),
        random.uniform(-angular_speed, angular_speed),
        random.uniform(-angular_speed, angular_speed),
    )

    set_body_velocity(path, linear_velocity, angular_velocity)

    print("===================================================")
    print("[EVENT] Debris released to Dynamic physics:", name_or_path)
    print("[EVENT] path:", path)
    print("[EVENT] linear_velocity:", linear_velocity)
    print("[EVENT] angular_velocity:", angular_velocity)
    print("[EVENT] It will no longer be updated by the kinematic orbit loop.")
    print("===================================================")


def apply_selective_physics(stage):
    """Apply the exact physics policy requested for this scene."""
    physics_material = create_bouncy_physics_material(stage)

    collectable_by_name = {}

    # 1) Station: Kinematic + Collider.
    # Robot is docked under the station, so its proxy parts become part of the
    # station compound collider. Do not put a second nested rigid body on robot.
    station_prim = stage.GetPrimAtPath(SPACE_STATION_PATH)
    apply_selective_rigid_body(
        station_prim,
        kinematic=True,
        mass=10000.0,
        disable_gravity=True,
        lock_rotation=True,
        physics_material=physics_material
    )

    # 2) Collectable debris: Kinematic + Collider at first.
    # They can later be switched to Dynamic by release_collectable_debris().
    for spec in ORBIT_SPECS:
        if spec["kind"] not in ("debris", "special_usdz"):
            continue

        prim = stage.GetPrimAtPath(spec["object_path"])
        if not prim or not prim.IsValid():
            continue

        collectable_by_name[spec["name"]] = spec["object_path"]

        apply_selective_rigid_body(
            prim,
            kinematic=True,
            mass=BODY_MASS,
            disable_gravity=True,
            lock_rotation=False,
            physics_material=physics_material
        )

    globals()["_COLLECTABLE_DEBRIS_BY_NAME"] = collectable_by_name
    print("[PHYSICS] Collectable debris registered:", sorted(collectable_by_name.keys()))
    print("[PHYSICS] Protected satellites / asteroid remain Visual only, no collider.")



def create_sphere(stage, path, translate=(0, 0, 0), scale=(1, 1, 1), material=None, radius=1.0):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    set_transform(sphere.GetPrim(), translate=translate, scale=scale)
    if material is not None:
        bind_material(sphere.GetPrim(), material)
    return sphere.GetPrim()


def create_cube(stage, path, translate=(0, 0, 0), rotate=(0, 0, 0), scale=(1, 1, 1), material=None):
    cube = UsdGeom.Cube.Define(stage, path)
    set_transform(cube.GetPrim(), translate, rotate, scale)
    if material is not None:
        bind_material(cube.GetPrim(), material)
    return cube.GetPrim()


def create_cylinder(stage, path, translate=(0, 0, 0), rotate=(0, 0, 0), scale=(1, 1, 1), material=None, radius=0.5, height=1.0):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    set_transform(cyl.GetPrim(), translate, rotate, scale)
    if material is not None:
        bind_material(cyl.GetPrim(), material)
    return cyl.GetPrim()


# ============================================================
# GLB -> USD CONVERSION
# ============================================================

def assert_asset_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing asset file:\n{path}\n\nPut GLB files under:\n{RAW_ASSET_DIR}")


def safe_set_attr(obj, name, value):
    try:
        setattr(obj, name, value)
    except Exception:
        pass


async def convert_glb_to_usd(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # GLB가 없을 때: 이미 변환된 USD가 있으면 그걸 쓰고, USD도 없으면 이 자산은 생략(None).
    if not os.path.exists(src_path):
        if os.path.exists(dst_path):
            print(f"[ASSET] GLB 없음 -> 기존 USD 사용: {os.path.basename(dst_path)}")
            return dst_path
        print(f"[ASSET][SKIP] GLB도 USD도 없음 -> 생략: {os.path.basename(src_path)}")
        return None

    if FORCE_RECONVERT_ASSETS and os.path.exists(dst_path):
        os.remove(dst_path)
    if os.path.exists(dst_path):
        print("[ASSET] USD already exists:", os.path.basename(dst_path))
        return dst_path

    print("[ASSET] Converting GLB -> USD:", os.path.basename(src_path))
    try:
        import omni.kit.asset_converter as asset_converter
    except Exception as e:
        raise RuntimeError("omni.kit.asset_converter import failed. Enable Asset Converter extension.") from e

    context = asset_converter.AssetConverterContext()
    safe_set_attr(context, "ignore_materials", False)
    safe_set_attr(context, "ignore_animation", False)
    safe_set_attr(context, "merge_all_meshes", False)   # False로 해야 다중 재질이 유지됨 (정거장 하얗게 나오는 문제 해결)
    safe_set_attr(context, "use_meter_as_world_unit", True)
    safe_set_attr(context, "create_world_as_default_root_prim", False)
    safe_set_attr(context, "embed_textures", True) # True로 해야 GLB 내부의 텍스처가 정상 추출 및 연동됨

    converter = asset_converter.get_instance()
    try:
        task = converter.create_converter_task(src_path, dst_path, None, context)
    except TypeError:
        task = converter.create_converter_task(src_path, dst_path, context)

    if not await task.wait_until_finished():
        raise RuntimeError(f"Asset conversion failed: {src_path}")
    print("[ASSET] Conversion complete:", os.path.basename(dst_path))
    return dst_path


def convert_usdz_to_wrapper_usd(src_usdz_path, dst_usd_path, force=False):
    """Unpack a USDZ into assets/usd/<name>_usdz_unpacked and create a tiny
    wrapper USD that references the internal scene.usdc.

    Isaac Sim에서 .usdz를 Add Reference 했을 때 안 들어오는 경우가 많아서,
    직접 USDZ를 참조하지 않고 내부 scene.usdc를 감싼 wrapper .usd를 만든다.
    텍스처 폴더도 같은 unpack 폴더에 같이 풀리므로 material 경로가 유지된다.
    """
    os.makedirs(os.path.dirname(dst_usd_path), exist_ok=True)

    # USDZ가 없을 때: 이미 만든 wrapper USD가 있으면 그걸 쓰고, 없으면 생략(None).
    if not os.path.exists(src_usdz_path):
        if os.path.exists(dst_usd_path):
            print(f"[ASSET] USDZ 없음 -> 기존 wrapper 사용: {os.path.basename(dst_usd_path)}")
            return dst_usd_path
        print(f"[ASSET][SKIP] USDZ도 wrapper도 없음 -> 생략: {os.path.basename(src_usdz_path)}")
        return None

    asset_name = os.path.splitext(os.path.basename(src_usdz_path))[0]
    unpack_dir = os.path.join(os.path.dirname(dst_usd_path), f"{asset_name}_usdz_unpacked")
    scene_usdc = os.path.join(unpack_dir, "scene.usdc")

    if force and os.path.exists(dst_usd_path):
        os.remove(dst_usd_path)
    if force and os.path.isdir(unpack_dir):
        shutil.rmtree(unpack_dir)

    if not os.path.exists(scene_usdc):
        if os.path.isdir(unpack_dir):
            shutil.rmtree(unpack_dir)
        os.makedirs(unpack_dir, exist_ok=True)
        print("[ASSET] Unpacking USDZ ->", unpack_dir)
        with zipfile.ZipFile(src_usdz_path, "r") as zf:
            zf.extractall(unpack_dir)

    # 대부분의 USDZ는 scene.usdc가 루트다. 혹시 이름이 다르면 첫 usdc/usd/usda를 잡는다.
    if not os.path.exists(scene_usdc):
        candidates = []
        for root, _, files in os.walk(unpack_dir):
            for fn in files:
                if fn.lower().endswith((".usdc", ".usd", ".usda")):
                    candidates.append(os.path.join(root, fn))
        if not candidates:
            raise RuntimeError(f"No USD scene found inside USDZ: {src_usdz_path}")
        scene_usdc = candidates[0]

    rel_scene = os.path.relpath(scene_usdc, os.path.dirname(dst_usd_path)).replace(os.sep, "/")

    # .usd 확장자지만 ASCII USD 문법으로 wrapper를 만든다. Isaac/Omniverse가 정상 로드한다.
    wrapper_text = (
        "#usda 1.0\n"
        "(\n"
        "    defaultPrim = \"Model\"\n"
        "    metersPerUnit = 1\n"
        "    upAxis = \"Z\"\n"
        ")\n\n"
        "def Xform \"Model\" (\n"
        f"    references = @{rel_scene}@\n"
        ")\n"
        "{\n"
        "}\n"
    )
    with open(dst_usd_path, "w", encoding="utf-8") as f:
        f.write(wrapper_text)

    print("[ASSET] USDZ wrapper ready:", os.path.basename(dst_usd_path), "->", rel_scene)
    return dst_usd_path


async def convert_assets():
    # v15 수정: GLB를 USD로 변환할 때 텍스처 매핑이 계속 유실되는 문제를 방지하기 위해,
    # 변환 과정을 생략하고 원본 GLB 경로를 그대로 반환하여 옴니버스 네이티브 임포터가 텍스처를 유지하도록 합니다.
    earth_usd = EARTH_GLB
    station_usd = STATION_GLB

    satellite_usd = None
    if os.path.exists(SATELLITE_GLB):
        satellite_usd = SATELLITE_GLB

    debris_usds = {k: glb for k, glb in DEBRIS_GLBS.items()}
    asteroid_usd = ASTEROID_GLB

    # USDZ는 Add Reference가 바로 실패할 수 있으므로 내부 scene.usdc를 wrapper USD로 감싼다.
    ufo_usd = convert_usdz_to_wrapper_usd(UFO_USDZ, UFO_USD, force=FORCE_RECONVERT_ASSETS)
    r2d2_usd = convert_usdz_to_wrapper_usd(R2D2_USDZ, R2D2_USD, force=FORCE_RECONVERT_ASSETS)
    meteor_m2_usd = convert_usdz_to_wrapper_usd(METEOR_M2_USDZ, METEOR_M2_USD, force=FORCE_RECONVERT_ASSETS)
    rooftop_usd = convert_usdz_to_wrapper_usd(ROOFTOP_USDZ, ROOFTOP_USD, force=FORCE_RECONVERT_ASSETS)

    return {
        "earth": earth_usd, "station": station_usd, "satellite": satellite_usd,
        "debris": debris_usds, "asteroid": asteroid_usd,
        "ufo": ufo_usd, "r2d2": r2d2_usd,
        "meteor_m2": meteor_m2_usd,
        "rooftop_debris": rooftop_usd,
    }


# ============================================================
# MODEL REFERENCE  (ABSOLUTE scale -- no normalization)
# ============================================================

def add_model_reference(stage, root_path, usd_path, abs_scale=1.0, semantic_class=None):
    """Reference a USD model under root_path/Model and apply an ABSOLUTE scale.

    Scale lives ONLY on /Model so the object ROOT stays a clean xform we can
    turn into a rigid body (PhysX dislikes scaled rigid-body roots). Position /
    base orientation / velocity live on the ROOT.
    """
    root = create_xform(stage, root_path)
    if semantic_class is not None:
        add_semantic(root, semantic_class)

    model_path = f"{root_path}/Model"
    model_prim = create_xform(stage, model_path)
    model_prim.GetReferences().AddReference(usd_path)

    xf = UsdGeom.Xformable(model_prim)
    xf.ClearXformOpOrder()
    xf.AddScaleOp().Set(Gf.Vec3f(abs_scale, abs_scale, abs_scale))
    return root


# ============================================================
# MATERIALS (robot / proxies / orbit dots)
# ============================================================

def create_materials(stage):
    mat_root = f"{ROOT_PATH}/Materials"
    create_xform(stage, mat_root)
    return {
        "orbit_dot":      make_material(stage, f"{mat_root}/orbit_dot", (0.58, 0.72, 1.00), 0.0, 0.42),
        "robot_body":     make_material(stage, f"{mat_root}/robot_body", (0.90, 0.88, 0.78), 0.18, 0.32),
        "robot_blue":     make_material(stage, f"{mat_root}/robot_blue", (0.10, 0.36, 1.00), 0.05, 0.28),
        "thruster_dark":  make_material(stage, f"{mat_root}/thruster_dark", (0.05, 0.06, 0.07), 0.25, 0.38),
        "capture_yellow": make_material(stage, f"{mat_root}/capture_yellow", (1.00, 0.72, 0.10), 0.0, 0.38),
    }


# ============================================================
# ORBIT RINGS / ROBOT / STATION PROXY
# ============================================================

def build_orbit_paths(stage, mats):
    # v15: draw object-specific orbit rings using each spec's real orbit plane.
    # Flat debris rings remain XY-plane; only Meteor-M2 satellites/asteroid rings are inclined.
    create_xform(stage, ORBIT_PATHS_PATH)

    specs = globals().get("ORBIT_SPECS", [])
    if not specs:
        specs = [{"name": name, "radius": radius, "tilt": 0.0, "tilt_z": 0.0} for name, radius in ORBIT_LANES.items()]

    for spec in specs:
        radius = float(spec["radius"])
        tilt_x = math.radians(float(spec.get("tilt", 0.0)))
        tilt_z = math.radians(float(spec.get("tilt_z", 0.0)))
        ring_path = f"{ORBIT_PATHS_PATH}/Orbit_{spec['name']}"
        create_xform(stage, ring_path)

        for i in range(ORBIT_DOT_COUNT):
            theta = 2.0 * math.pi * i / ORBIT_DOT_COUNT

            if spec.get("name") == "DebrisUFO":
                p = (
                    radius * UFO_ELLIPSE_MAJOR_SCALE * math.cos(theta),
                    radius * UFO_ELLIPSE_MINOR_SCALE * math.sin(theta),
                    0.0,
                )
            else:
                p = (radius * math.cos(theta), radius * math.sin(theta), 0.0)

            p = rot_orbit(p, tilt_x, tilt_z)
            create_sphere(
                stage,
                f"{ring_path}/dot_{i:03d}",
                translate=p,
                scale=(0.5, 0.5, 0.5),   # [B안] 500m 맵에서 점 안 보임 -> 키움 (취향껏 0.3~1.0)
                material=mats["orbit_dot"],
            )


def build_docked_robot(stage, mats, root_path, s=0.52):
    create_xform(stage, root_path)
    create_cube(stage, f"{root_path}/RobotBody", (0, 0, 0), scale=(0.145 * s, 0.110 * s, 0.100 * s), material=mats["robot_body"])
    create_cube(stage, f"{root_path}/FrontCameraVisual", (0.0, -0.13 * s, 0.03 * s), scale=(0.046 * s, 0.022 * s, 0.030 * s), material=mats["robot_blue"])
    create_cube(stage, f"{root_path}/CaptureClaw_Base", (0.0, -0.205 * s, -0.02 * s), scale=(0.036 * s, 0.070 * s, 0.022 * s), material=mats["robot_body"])
    create_cube(stage, f"{root_path}/CaptureClaw_Left", (-0.048 * s, -0.280 * s, -0.02 * s), rotate=(0, 0, -18), scale=(0.016 * s, 0.063 * s, 0.016 * s), material=mats["robot_blue"])
    create_cube(stage, f"{root_path}/CaptureClaw_Right", (0.048 * s, -0.280 * s, -0.02 * s), rotate=(0, 0, 18), scale=(0.016 * s, 0.063 * s, 0.016 * s), material=mats["robot_blue"])
    for name, pos in [("ThrusterLeft", (-0.180 * s, 0.04 * s, 0.0)), ("ThrusterRight", (0.180 * s, 0.04 * s, 0.0)),
                      ("ThrusterTop", (0.0, 0.04 * s, 0.135 * s)), ("ThrusterBottom", (0.0, 0.04 * s, -0.135 * s))]:
        create_cube(stage, f"{root_path}/{name}", pos, scale=(0.036 * s, 0.036 * s, 0.036 * s), material=mats["thruster_dark"])

    robot_cam = UsdGeom.Camera.Define(stage, ROBOT_CAMERA_PATH)
    robot_cam.CreateFocalLengthAttr(28.0)
    set_transform(robot_cam.GetPrim(), translate=(0.0, -0.55, 0.09), rotate=(68.0, 0.0, 0.0))


def add_station_proxy_parts(stage, mats, station_path):
    create_cylinder(stage, f"{station_path}/DockingPort_Proxy", (0.0, -0.48, 0.0), rotate=(90, 0, 0), scale=(0.080, 0.080, 0.055), material=mats["robot_body"])
    create_cube(stage, f"{station_path}/CollectionBay_Proxy", (0.34, -0.42, -0.22), scale=(0.16, 0.11, 0.04), material=mats["capture_yellow"])


# ============================================================
# ORBIT SPECS
#   (speed 필드는 옛 키네마틱 잔재 -- 이제 안 씀. 속도는 v=sqrt(GM/r)로 계산.)
# ============================================================

def debris_spec(name, lane, glb_key, speed, phase, tilt, base_rot, self_spin):
    return {
        "name": name, "kind": "debris", "glb_key": glb_key,
        "pivot_path": f"{ORBIT_OBJECTS_PATH}/{name}OrbitPivot",
        "object_path": f"{ORBIT_OBJECTS_PATH}/{name}OrbitPivot/{name}",
        "radius": ORBIT_LANES[lane], "speed": speed, "phase": phase, "tilt": tilt,
        "scale": DEBRIS_SCALES[glb_key], "base_rot": base_rot, "self_spin": self_spin,
    }


def satellite_spec(idx, lane, speed, phase, tilt, base_rot, self_spin, scale, asset_key="meteor_m2"):
    name = f"ProtectedSatellite_{idx:02d}"
    return {
        "name": name, "kind": "satellite", "asset_key": asset_key,
        "pivot_path": f"{ORBIT_OBJECTS_PATH}/SatelliteOrbitPivot_{idx:02d}",
        "object_path": f"{ORBIT_OBJECTS_PATH}/SatelliteOrbitPivot_{idx:02d}/{name}",
        "radius": ORBIT_LANES[lane], "speed": speed, "phase": phase, "tilt": tilt,
        "scale": scale, "base_rot": base_rot, "self_spin": self_spin,
    }


def special_usdz_spec(name, asset_key, lane, phase, tilt, base_rot, self_spin, scale):
    return {
        "name": name, "kind": "special_usdz", "asset_key": asset_key,
        "pivot_path": f"{ORBIT_OBJECTS_PATH}/{name}OrbitPivot",
        "object_path": f"{ORBIT_OBJECTS_PATH}/{name}OrbitPivot/{name}",
        "radius": ORBIT_LANES[lane], "speed": 0.0, "phase": phase, "tilt": tilt,
        "scale": scale, "base_rot": base_rot, "self_spin": self_spin,
    }


ORBIT_SPECS = [
    {
        "name": "Station", "kind": "station",
        "pivot_path": STATION_PIVOT_PATH, "object_path": SPACE_STATION_PATH,
        "radius": ORBIT_LANES["station"], "speed": 2.60, "phase": 0.0, "tilt": 0.0,
        "scale": STATION_SCALE, "base_rot": (-90.0, 0.0, -90.0), "self_spin": 0.0,
    },

    # ---- 5 debris, each on its OWN lane ----
    debris_spec("DebrisRocket",  "inner_debris",   "rocket_stage", 5.80, 62.0,  -4.5, (0, 0, 28), 18.0),
    debris_spec("DebrisLadder",  "mid_debris",     "ladder",       4.80, 188.0,  5.5, (0, 0, -22), -15.0),
    debris_spec("DebrisHelmet",  "outer_debris",   "helmet",       3.40, 305.0, -7.0, (0, 0, 54), 26.0),
    debris_spec("DebrisWalle",   "extra_debris_1", "walle",        4.20, 150.0,  6.5, (0, 0, 0), 30.0),
    debris_spec("DebrisSatBody", "extra_debris_2", "sat_body",     3.70, 250.0, -3.0, (0, 0, 12), 20.0),

    {
        "name": "Asteroid_01", "kind": "asteroid", "glb_key": None,
        "pivot_path": f"{ORBIT_OBJECTS_PATH}/AsteroidOrbitPivot_01",
        "object_path": f"{ORBIT_OBJECTS_PATH}/AsteroidOrbitPivot_01/Asteroid_01",
        "radius": ORBIT_LANES["outer_debris"], "speed": 3.10, "phase": 110.0, "tilt": 9.0,
        "scale": ASTEROID_SCALE, "base_rot": (0, 0, 0), "self_spin": 12.0,
    },

    # ---- USDZ special objects: 궤도면을 크게 기울여 3D 입체감 살림 ----
    # R2D2: 낮은 고도, 거의 세로로 도는 느낌
    special_usdz_spec("DebrisR2D2", "r2d2", "r2d2_special", 25.0, 68.0, (90.0, 0.0, 270.0), 0.0, R2D2_SCALE),
    # UFO: 바깥쪽 고도, 반대 방향으로 기울어진 궤도면
    special_usdz_spec("DebrisUFO", "ufo", "ufo_special", 215.0, -54.0, (0.0, 0.0, 0.0), 18.0, UFO_SCALE),

    # Rooftop antenna USDZ: extra collectable debris object.
    special_usdz_spec("DebrisRooftop", "rooftop_debris", "rooftop_debris", 332.0, 41.0, (0.0, 0.0, 0.0), 14.0, ROOFTOP_DEBRIS_SCALE),

    # ---- 5 protected satellite orbit pivots, now using Meteor-M2 USDZ visuals ----
    satellite_spec(1, "sat_1", 4.20,  38.0,  28.0, (-90.0, 0.0,   0.0),  1.5, METEOR_M2_SCALE),
    satellite_spec(2, "sat_2", 2.90, 246.0, -36.0, (-90.0, 0.0, -28.0), -1.0, METEOR_M2_SCALE * 0.92),
    satellite_spec(3, "sat_3", 3.60, 120.0,  52.0, (-90.0, 0.0,  45.0),  1.2, METEOR_M2_SCALE * 1.08),
    satellite_spec(4, "sat_4", 2.40, 300.0, -61.0, (-90.0, 0.0,  15.0), -0.8, METEOR_M2_SCALE * 0.85),
    satellite_spec(5, "sat_5", 4.60, 200.0,  74.0, (-90.0, 0.0, -60.0),  1.8, METEOR_M2_SCALE * 1.15),
]

# v15 orbit-plane policy:
#   - collectable debris stay on the flat XY orbit plane.
#     DebrisRocket / Ladder / Helmet / Walle / SatBody / R2D2 / UFO / Rooftop = flat.
#   - only the Meteor-M2 objects that replaced SatelliteOrbitPivot_01~05
#     and the asteroid/meteorite get inclined multi-axis orbits.
#
# Existing ``tilt`` tilts the orbit around X.
# ``tilt_z`` spins that tilted plane around Z so each inclined orbit cuts through
# the scene in a different up/down/diagonal direction.
INCLINED_ORBIT_AXES_DEG = {
    # meteor/comet/asteroid
    "Asteroid_01": (48.0, 128.0),

    # SatelliteOrbitPivot_01~05 now use Meteor-M2 visuals.
    "ProtectedSatellite_01": (28.0, 22.0),
    "ProtectedSatellite_02": (-36.0, -38.0),
    "ProtectedSatellite_03": (52.0, 84.0),
    "ProtectedSatellite_04": (-61.0, -126.0),
    "ProtectedSatellite_05": (74.0, 166.0),
}

for _spec in ORBIT_SPECS:
    if _spec["name"] in INCLINED_ORBIT_AXES_DEG:
        _tilt_x, _tilt_z = INCLINED_ORBIT_AXES_DEG[_spec["name"]]
        _spec["tilt"] = _tilt_x
        _spec["tilt_z"] = _tilt_z
    else:
        # Keep all collectable debris and station flat.
        _spec["tilt"] = 0.0
        _spec["tilt_z"] = 0.0


# ============================================================
# WORLD BUILD
# ============================================================

def build_orbit_system(stage, mats, assets):
    create_xform(stage, ORBIT_SYSTEM_PATH)
    create_xform(stage, ORBIT_OBJECTS_PATH)

    # 지구: 물리 없는 순수 비주얼 앵커 (원점 고정 중력원)
    if assets.get("earth"):
        add_model_reference(stage, EARTH_PATH, assets["earth"], EARTH_SCALE, semantic_class=None)
    else:
        print("[BUILD][SKIP] Earth 자산 없음 -> 지구 비주얼 생략(공전 원점은 그대로 동작)")
    build_orbit_paths(stage, mats)

    skipped = []
    for spec in ORBIT_SPECS:
        create_xform(stage, spec["pivot_path"])   # 피벗은 이제 회전 안 함 (정지 컨테이너)
        kind = spec["kind"]
        if kind == "station":
            usd, sem = assets.get("station"), None
        elif kind == "satellite":
            usd, sem = assets.get(spec.get("asset_key", "meteor_m2")), "protected_satellite"
        elif kind == "debris":
            usd, sem = assets["debris"].get(spec["glb_key"]), "collectable_debris"
        elif kind == "asteroid":
            usd, sem = assets.get("asteroid"), "asteroid"
        elif kind == "special_usdz":
            usd, sem = assets.get(spec["asset_key"]), "collectable_debris"
        else:
            usd, sem = None, None

        if usd is None:
            skipped.append(spec["name"])
            print(f"[BUILD][SKIP] {spec['name']}: 자산 파일 없음 -> 이 오브젝트 생략")
            continue
        add_model_reference(stage, spec["object_path"], usd, spec["scale"], semantic_class=sem)
    if skipped:
        print(f"[BUILD] 생략된 오브젝트 {len(skipped)}개: {skipped}  (GLB/USDZ를 raw 폴더에 넣으면 살아남)")
    print("[BUILD] Orbit asset roots created (flat collectable debris + 5 inclined Meteor-M2 satellites + inclined asteroid + station).")


def apply_initial_transforms(stage, mats):
    # 지구 원점 고정 (X축 90도 회전 적용)
    earth = stage.GetPrimAtPath(EARTH_PATH)
    if earth and earth.IsValid():
        set_transform(earth, translate=(0, 0, 0), rotate=(90, 0, 0), scale=(1, 1, 1))

    # v11 키네마틱: 리지드바디를 만들지 않는다. 대신 각 물체의 궤도 파라미터를
    # 전역(_KINEMATIC_BODIES)에 저장 -> 매 프레임 루프가 위치를 직접 계산해 세팅.
    # 케플러 각속도 omega = sqrt(GM / r^3) (rad/sim-sec). 부동소수점 누적 없음.
    kin = []
    for spec in ORBIT_SPECS:
        obj = stage.GetPrimAtPath(spec["object_path"])
        if not obj or not obj.IsValid():
            continue

        r = float(spec["radius"])
        phi0 = math.radians(spec["phase"])
        tilt = math.radians(spec["tilt"])
        tilt_z = math.radians(spec.get("tilt_z", 0.0))
        omega = math.sqrt(ORBIT_GM / (r ** 3))   # 케플러 각속도

        # 초기 위치 (위상 phi0)
        if spec["name"] == "DebrisUFO":
            pos = (
                r * UFO_ELLIPSE_MAJOR_SCALE * math.cos(phi0),
                r * UFO_ELLIPSE_MINOR_SCALE * math.sin(phi0),
                0.0
            )
        else:
            pos = (
                r * math.cos(phi0),
                r * math.sin(phi0),
                0.0
            )

        pos = rot_orbit(pos, tilt, tilt_z)
        set_transform(obj, translate=pos, rotate=spec["base_rot"], scale=(1, 1, 1))
        
        kin.append({
            "path": spec["object_path"], "kind": spec["kind"],
            "r": r, "phi0": phi0, "omega": omega, "tilt": tilt, "tilt_z": tilt_z,
            "base_rot": spec["base_rot"], "name": spec["name"],
            # 자전 모드: debris/asteroid = 텀블링, satellite/station = 조석 고정
            "spin_mode": ("tidal" if spec["kind"] in ("satellite", "station") else "tumble"),
            # 텀블링 축별 속도(deg/sim-sec). 물체마다 살짝 다르게 -> 자연스러움.
            # self_spin을 기준 삼아 3축에 서로 다른 비율로 분배.
            "tumble_rate": (
                spec.get("self_spin", 8.0) * 0.6,
                spec.get("self_spin", 8.0) * 0.35,
                spec.get("self_spin", 8.0) * 0.8,
            ),
            # DebrisLadder는 v11 타깃과 동일 텀블링(절대 deg/s, scale 무관). 나머지는 None.
            "tumble_abs": (LADDER_TUMBLE_DEG_PER_SEC if spec["name"] == "DebrisLadder" else None),
        })

        # 정거장 프록시 + 도킹 로봇 부착
        if spec["kind"] == "station":
            add_station_proxy_parts(stage, mats, spec["object_path"])

            # 기존 코드에는 build_docked_robot() 함수만 있고 실제 호출이 없었음.
            # 여기서 정거장 자식으로 도킹 로봇을 실제 생성한다.
            build_docked_robot(stage, mats, ROBOT_PATH, s=0.52)
            robot_prim = stage.GetPrimAtPath(ROBOT_PATH)
            if robot_prim and robot_prim.IsValid():
                set_transform(robot_prim, translate=ROBOT_DOCK_LOCAL_POS, rotate=(0, 0, 0), scale=(1, 1, 1))

    globals()["_KINEMATIC_BODIES"] = kin
    print(f"[BUILD] Kinematic orbit state stored for {len(kin)} bodies (selective physics added later).")


def build_logic_layers(stage):
    for p in (PERCEPTION_PATH, f"{PERCEPTION_PATH}/DetectionBBoxes",
              MISSION_PATH, f"{MISSION_PATH}/TargetManager", f"{MISSION_PATH}/RobotStateMachine",
              DEBUG_PATH, f"{DEBUG_PATH}/VelocityArrows", f"{DEBUG_PATH}/TextStatus"):
        create_xform(stage, p)


def build_zero_g_physics_scene(stage):
    remove_if_exists(stage, PHYSICS_SCENE_PATH)
    scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
    # 전역 중력 0 -> 우리가 매 스텝 인가하는 중심 중력만 작용
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(0.0)
    print("[BUILD] Zero-g PhysicsScene created (only our central gravity acts).")


def build_cameras(stage):
    overview = UsdGeom.Camera.Define(stage, OVERVIEW_CAMERA_PATH)
    overview.CreateFocalLengthAttr(30.0)
    set_transform(overview.GetPrim(), translate=CAMERA_TRANSLATE, rotate=CAMERA_ROTATE)
    print("[BUILD] Overview camera created.")


# ============================================================
# 오브젝트별 모니터 카메라 (각 공전체를 따라다니며 비춤)
# ============================================================
MONITOR_CAM_ROOT = f"{ROOT_PATH}/MonitorCameras"
CAM_FOLLOW_DIST = 12.0    # 오브젝트에서 카메라까지(반경 바깥 방향) 거리(m)
CAM_UP_OFFSET = 4.0       # 살짝 위에서 내려보기(m)


def build_monitor_cameras(stage):
    """ORBIT_SPECS의 각 공전체마다 전용 카메라를 만든다.
    위치/시선은 매 프레임 공전 루프(start_gravity_physics)에서 오브젝트를 따라 갱신된다.
    뷰포트 카메라를 /World/.../MonitorCameras/Cam_<이름> 으로 바꾸면 그 오브젝트를 클로즈업."""
    create_xform(stage, MONITOR_CAM_ROOT)
    cams = []
    for spec in ORBIT_SPECS:
        cam_path = f"{MONITOR_CAM_ROOT}/Cam_{spec['name']}"
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cam.CreateFocalLengthAttr(22.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1.0e6))
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.ClearXformOpOrder()
        top = xf.AddTransformOp()   # 매 프레임 4x4(look-at)로 직접 세팅
        cams.append({"name": spec["name"], "obj_path": spec["object_path"], "xform_op": top})
    globals()["_MONITOR_CAMERAS"] = cams
    print(f"[CAM] {len(cams)} monitor cameras under {MONITOR_CAM_ROOT} (Cam_<name>)")
    return cams


def aim_monitor_camera(top_op, eye, center, up=(0.0, 0.0, 1.0)):
    """카메라를 eye에 두고 center를 바라보게(look-at) 4x4 변환을 세팅."""
    view = Gf.Matrix4d()
    view.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*center), Gf.Vec3d(*up))
    top_op.Set(view.GetInverse())


def build_lights(stage):
    create_xform(stage, LIGHTS_PATH)
    sun = UsdLux.DistantLight.Define(stage, f"{LIGHTS_PATH}/SunKeyLight")
    sun.CreateIntensityAttr(5600.0)
    set_transform(sun.GetPrim(), rotate=(315.0, 0.0, 25.0))
    fill = UsdLux.DistantLight.Define(stage, f"{LIGHTS_PATH}/FillLight")
    fill.CreateIntensityAttr(1600.0)
    set_transform(fill.GetPrim(), rotate=(45.0, 0.0, -20.0))
    front = UsdLux.SphereLight.Define(stage, f"{LIGHTS_PATH}/FrontSoftLight")
    front.CreateIntensityAttr(11000.0)
    front.CreateRadiusAttr(7.0)
    set_transform(front.GetPrim(), translate=(0.0, -10.0, 7.0))
    print("[BUILD] Lights created.")


def build_star_background(stage):
    """스타필드 배경(Dome Light + equirectangular 파노라마). assets/raw에서 자동 탐색.
    파노라마 이미지가 없으면 그냥 스킵(에러 없음). (add_space_background.py 통합본)"""
    import glob
    DOME_PATH = "/World/SpaceBackground/StarDome"
    DOME_INTENSITY = 3000.0   # 1000->3000: 별 배경이 안 보이면 더 올려도 됨(EXR/HDR도)
    prefer = ["space_hdri", "milkyway", "milky_way", "starmap", "star_map",
              "deep_star_map", "stars", "nightsky", "space"]
    exts = [".exr", ".hdr", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    tex = None
    if os.path.isdir(RAW_ASSET_DIR):
        files = []
        for ext in exts:
            files += glob.glob(os.path.join(RAW_ASSET_DIR, f"*{ext}"))
        files = [f for f in files if os.path.splitext(f)[1].lower() in exts]

        def _rank(p):
            name = os.path.basename(p).lower()
            pref = next((i for i, x in enumerate(prefer) if x in name), len(prefer))
            ext = os.path.splitext(name)[1]
            er = exts.index(ext) if ext in exts else len(exts)
            return (pref, er, name)

        if files:
            files.sort(key=_rank)
            tex = files[0]

    if tex is None:
        print("[BG][SKIP] assets/raw에 파노라마(별 배경) 이미지 없음 -> 배경 생략")
        return

    parent = stage.GetPrimAtPath("/World/SpaceBackground")
    if parent and parent.IsValid():
        stage.RemovePrim("/World/SpaceBackground")
    UsdGeom.Xform.Define(stage, "/World/SpaceBackground")

    dome = UsdLux.DomeLight.Define(stage, DOME_PATH)
    dome.CreateIntensityAttr(DOME_INTENSITY)
    try:
        dome.CreateExposureAttr(0.0)
    except Exception:
        pass
    dome.CreateTextureFileAttr(tex)
    try:
        dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)   # equirectangular
    except Exception:
        pass
    _a = dome.GetPrim().GetAttribute("inputs:intensity")
    if _a and _a.IsValid():
        _a.Set(DOME_INTENSITY)

    is_ldr = os.path.splitext(tex)[1].lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    print(f"[BG] Star dome: {os.path.basename(tex)} (intensity {DOME_INTENSITY}, latlong)")
    if is_ldr:
        print("[BG] LDR(.png/.jpg)라 별이 어두우면 코드의 DOME_INTENSITY를 2000~4000으로 올려라.")


# ============================================================
# KINEMATIC ORBIT  (실사 궤도: 위치를 직접 계산해 세팅 -> 부동소수점 누적 없음)
# ============================================================

def start_gravity_physics():
    """v11: 물리 적분 대신 케플러 공식으로 매 프레임 위치를 직접 계산해 세팅한다.
    각 물체 위상각 phi(t) = phi0 + omega * t. 누적 시간 t는 dt를 더해 관리.
    부동소수점 누적 폭주가 없어 궤도를 실제 km 비례로 키워도 안정적이다.

    로컬 물리 버블(stage0 add-on)이 어떤 물체를 물리로 전환하면, 그 경로를
    globals()['_PHYSICS_BUBBLE_PATHS'] (set)에 넣는다. 여기서는 그 물체의
    키네마틱 갱신을 건너뛴다 -> 물리(리지드바디)가 위치를 담당."""

    if globals().get("_SPACE_GRAVITY_SUB") is not None:
        globals()["_SPACE_GRAVITY_SUB"] = None

    from omni.physx import get_physx_interface

    stage = omni.usd.get_context().get_stage()
    kin = globals().get("_KINEMATIC_BODIES", [])
    earth_prim = stage.GetPrimAtPath(EARTH_PATH)
    _state = {"t": 0.0, "earth_angle": 0.0}

    # 오브젝트별 모니터 카메라 (이름 -> 카메라). 매 프레임 오브젝트 따라 위치/시선 갱신.
    monitor_cams = globals().get("_MONITOR_CAMERAS", [])
    cams_by_name = {c["name"]: c for c in monitor_cams}

    # 물리 버블 경로 집합 (stage0가 채움). 없으면 빈 집합.
    if "_PHYSICS_BUBBLE_PATHS" not in globals():
        globals()["_PHYSICS_BUBBLE_PATHS"] = set()

    def on_physics_step(dt):
        _state["t"] += dt
        t = _state["t"]
        globals()["_KINEMATIC_TIME"] = t

        # ----- 지구 자전 (지구만, X90 기울기 유지) -----
        if earth_prim and earth_prim.IsValid() and EARTH_SPIN_DEG_PER_SEC != 0.0:
            _state["earth_angle"] = (_state["earth_angle"] + dt * EARTH_SPIN_DEG_PER_SEC) % 360.0
            set_rotation(earth_prim, (90.0, 0.0, _state["earth_angle"]))

        bubble = globals().get("_PHYSICS_BUBBLE_PATHS", set())

        # ----- 각 물체: 키네마틱 위치 + 자전/조석 자세 갱신 (버블 안 물체는 건너뜀) -----
        for b in kin:
            if b["path"] in bubble:
                continue   # 물리 버블이 담당 -> 키네마틱 갱신 스킵
            prim = stage.GetPrimAtPath(b["path"])
            if not prim or not prim.IsValid():
                continue
            phi = b["phi0"] + b["omega"] * t
            r = b["r"]

            if b["name"] == "DebrisUFO":
                pos = (
                    r * UFO_ELLIPSE_MAJOR_SCALE * math.cos(phi),
                    r * UFO_ELLIPSE_MINOR_SCALE * math.sin(phi),
                    0.0
                )
            else:
                pos = (
                    r * math.cos(phi),
                    r * math.sin(phi),
                    0.0
                )

            pos = rot_orbit(pos, b["tilt"], b.get("tilt_z", 0.0))

            bx, by, bz = b["base_rot"]
            if b["spin_mode"] == "tidal":
                # 조석 고정: 공전한 각도(phi)만큼 자세도 돌려 항상 같은 면이 중심을 향함.
                # base_rot의 Z(yaw)에 공전 각도를 더한다 (달처럼).
                phi_deg = math.degrees(phi)
                rot = (bx, by, bz + phi_deg)
            else:
                # 텀블링: 시간 비례 3축 누적 회전.
                tab = b.get("tumble_abs")
                if tab is not None:
                    # DebrisLadder = v11 타깃과 동일 각속도(절대 deg/s, scale 무관)
                    rot = (bx + tab[0] * t, by + tab[1] * t, bz + tab[2] * t)
                else:
                    sx, sy, sz = b["tumble_rate"]
                    k = TUMBLE_SPEED_SCALE
                    rot = (bx + sx * k * t, by + sy * k * t, bz + sz * k * t)

            set_transform(prim, translate=pos, rotate=rot, scale=(1, 1, 1))

            # ----- 이 오브젝트 전용 카메라: 반경 바깥에서 안쪽(오브젝트)을 향해 따라감 -----
            cam = cams_by_name.get(b["name"])
            if cam is not None:
                px, py, pz = pos
                rr = math.sqrt(px * px + py * py + pz * pz)
                if rr > 1e-6:
                    ux, uy, uz = px / rr, py / rr, pz / rr
                else:
                    ux, uy, uz = 0.0, 0.0, 1.0
                eye = (px + ux * CAM_FOLLOW_DIST,
                       py + uy * CAM_FOLLOW_DIST,
                       pz + uz * CAM_FOLLOW_DIST + CAM_UP_OFFSET)
                aim_monitor_camera(cam["xform_op"], eye, pos)

    # standalone: 메인 루프에서 매 프레임 직접 호출(v11과 동일 패턴). dt는 1/60 고정.
    globals()["_ORBIT_STEP_FN"] = on_physics_step
    # Script Editor에서 쓸 땐 아래 두 줄 주석 해제(PLAY 시 물리스텝 콜백으로 구동):
    # globals()["_SPACE_GRAVITY_SUB"] = \
    #     get_physx_interface().subscribe_physics_step_events(on_physics_step)
    print("[KINEMATIC] Orbit step armed (standalone: 메인 루프가 매 프레임 호출).")


# ============================================================
# MAIN
# ============================================================

def clean_previous_worlds(stage):
    for path in [ROOT_PATH, "/World/SpaceCleanupOrbitWorldV6", "/World/SpaceCleanupOrbitWorldV5",
                 "/World/SpaceCleanupOrbitWorld", "/World/SpaceCleanupWorld",
                 "/World/SpaceEnvironmentV2", "/World/SpaceEnvironmentV1"]:
        remove_if_exists(stage, path)


async def build_world_async():
    os.makedirs(os.path.dirname(OUTPUT_STAGE_PATH), exist_ok=True)
    print("### SPACE ENVIRONMENT V15 - METEOR-M2 + ROOFTOP + MULTI-AXIS ORBIT ###")

    assets = await convert_assets()
    stage = get_stage()
    clean_previous_worlds(stage)

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    create_xform(stage, ROOT_PATH)

    mats = create_materials(stage)
    build_orbit_system(stage, mats, assets)
    apply_initial_transforms(stage, mats)
    build_logic_layers(stage)
    build_zero_g_physics_scene(stage)

    # 선택적 물리 적용:
    #   Station: Kinematic + Collider
    #   Collectable debris: Kinematic + Collider -> release 시 Dynamic
    #   Protected satellites / asteroid / show objects: Visual only
    apply_selective_physics(stage)

    build_cameras(stage)
    build_monitor_cameras(stage)
    build_lights(stage)
    build_star_background(stage)

    stage.SetDefaultPrim(stage.GetPrimAtPath(ROOT_PATH))
    stage.GetRootLayer().Export(OUTPUT_STAGE_PATH)

    print("\n[DONE] Space environment v15 (flat debris + inclined Meteor-M2/asteroid orbits) created.")
    print("[SAVED]", OUTPUT_STAGE_PATH)
    print("  classes: collectable_debris(8 incl. R2D2/UFO/Rooftop) | asteroid(1) | protected_satellite(5 Meteor-M2) | station(none)")
    print(f"  [LEO 환산] 1 unit = {LEN_SCALE_M_PER_UNIT/1000:.1f} km | 시간 {TIME_ACCEL:.0f}x | 기준궤도 {SIM_R_AT_REF:.0f} units")
    print(f"  [궤도 방식] 키네마틱(위치 직접 계산) + 선택적 물리 버블")

    start_gravity_physics()
    print("\n>>> 이제 뷰포트에서 PLAY(▶) 를 누르면 중력 공전이 시작됩니다. <<<")


# ============================================================
# 통합용 진입 함수 (다른 스크립트가 import해서 호출)
# ============================================================
def build_scene(sim_app):
    """월드(맵)를 동기로 빌드하고 씬 핸들을 반환한다.
    asset 변환(async)은 sim_app.update()로 이벤트 루프를 펌핑하며 완료를 기다린다.
    반환 핸들:
      ladder_path      : 타깃 사다리 prim 경로(텀블링 공전체)
      kinematic_bodies : 전체 공전체 궤도 파라미터 리스트(_KINEMATIC_BODIES)
      orbit_step       : 매 프레임 호출할 공전/텀블/카메라 갱신 함수 fn(dt)
      orbit_gm         : 시뮬 GM (필요시 궤도 예측용)
    """
    task = asyncio.ensure_future(build_world_async())
    while not task.done():
        sim_app.update()
    exc = task.exception()
    if exc is not None:
        raise exc
    return {
        "ladder_path": f"{ORBIT_OBJECTS_PATH}/DebrisLadderOrbitPivot/DebrisLadder",
        "kinematic_bodies": globals().get("_KINEMATIC_BODIES", []),
        "orbit_step": globals().get("_ORBIT_STEP_FN"),
        "orbit_gm": ORBIT_GM,
        "orbit_objects_path": ORBIT_OBJECTS_PATH,
        "earth_path": EARTH_PATH,
    }


# ============================================================
# 직접 실행(standalone): ./python.sh space_environment_v10_leo.py
# ============================================================
if __name__ == "__main__":
    _handles = build_scene(simulation_app)
    _orbit_step = _handles["orbit_step"]

    _timeline = omni.timeline.get_timeline_interface()
    _timeline.play()
    print("\n>>> [STANDALONE] 공전 시작. 창을 닫거나 Ctrl+C로 종료. <<<")
    print(">>> 뷰포트 카메라를 /World/SpaceCleanupOrbitWorldV7/MonitorCameras/Cam_<오브젝트> 로 전환해 모니터링 <<<")
    print(">>> 오브젝트 목록:", [s["name"] for s in ORBIT_SPECS])

    for _ in range(30):
        simulation_app.update()

    try:
        while simulation_app.is_running():
            if _orbit_step is not None:
                _orbit_step(1.0 / 60.0)
            simulation_app.update()
    except KeyboardInterrupt:
        print("\n[STANDALONE] KeyboardInterrupt: 종료.")
    finally:
        try:
            _timeline.stop()
        except Exception:
            pass
        simulation_app.close()