import sys
sys.path = [p for p in sys.path if '/opt/ros' not in p]

import numpy as np
import math
import time
import os
import cv2
from scipy.spatial.transform import Rotation

class Det:
    pass

external_command = None



THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "..", "resources", "m0609_aruco_detect"))
from wrist_camera import WristCamera
from visual_servo_controller import VisualServoController
from m0609_rmpflow_controller import RMPFlowController
from realsense_mount import attach_realsense_d455
from camera_viewer import CameraViewer

from isaacsim.core.api import World
from omni.isaac.core.articulations import ArticulationView, Articulation
from omni.isaac.dynamic_control import _dynamic_control
from pxr import UsdPhysics, PhysxSchema, Gf, UsdGeom, UsdLux, Sdf, Usd, UsdShade
import omni.usd
import omni.timeline
import carb

def apply_high_friction(stage, prim_path, mu=1.8):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid(): return
    mat_path = f"{prim_path}/HighFrictionMat"
    mat = UsdShade.Material.Define(stage, mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys_mat.CreateStaticFrictionAttr().Set(mu)
    phys_mat.CreateDynamicFrictionAttr().Set(mu)
    phys_mat.CreateRestitutionAttr().Set(0.0)
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(mat, materialPurpose="physics")



def quat_to_euler(w, x, y, z):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.sign(sinp) * np.pi / 2 if abs(sinp) >= 1 else np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def find_prim_path_by_name(root_path, link_name):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == link_name:
            return str(prim.GetPath())
    return None

def get_stable_depth(depth_map, cx, cy, window=5):
    """5x5 주변 픽셀의 평균을 통해 안정적인 거리값을 구합니다."""
    if depth_map is None:
        return 999.0
    cy_int, cx_int = int(cy), int(cx)
    h, w = depth_map.shape
    half = window // 2
    
    y_min, y_max = max(0, cy_int - half), min(h, cy_int + half + 1)
    x_min, x_max = max(0, cx_int - half), min(w, cx_int + half + 1)
    
    patch = depth_map[y_min:y_max, x_min:x_max]
    valid_pixels = patch[patch > 0.0]
    if valid_pixels.size == 0:
        return 999.0
    return float(np.median(valid_pixels))

from pxr import Sdf

def get_rotation_quat(v1, v2):
    # Returns [w, x, y, z] quaternion rotating v1 to v2
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    axis = np.cross(v1, v2)
    axis_len = np.linalg.norm(axis)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    if axis_len < 1e-6:
        if dot > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            return np.array([0.0, 0.0, 1.0, 0.0])
    axis = axis / axis_len
    angle = np.arccos(dot)
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis[0]*s, axis[1]*s, axis[2]*s])

def _attach_cube_to_link(stage, joint_path, link_path, cube_path):
    """Phase 4 진입 시 타겟을 그리퍼 링크에 FixedJoint 로 결속."""
    if stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)

    link_prim = stage.GetPrimAtPath(link_path)
    cube_prim = stage.GetPrimAtPath(cube_path)
    if not link_prim.IsValid() or not cube_prim.IsValid():
        print(f"[grip_joint] invalid prim — link={link_path} cube={cube_path}")
        return False

    link_xf = UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    cube_xf = UsdGeom.Xformable(cube_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rel = cube_xf * link_xf.GetInverse()
    rel_pos = rel.ExtractTranslation()
    rel_rot = rel.ExtractRotationQuat()
    rot_imag = rel_rot.GetImaginary()

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(link_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(cube_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(rel_pos))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(
        rel_rot.GetReal(),
        float(rot_imag[0]), float(rot_imag[1]), float(rot_imag[2]),
    ))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    print(f"[grip_joint] attached: {cube_path} → {link_path}")
    return True

def _detach_grip_joint(stage, joint_path):
    if stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)
        print("[grip_joint] detached")

def _berthing_generator(stage, world, dc, timeline, simulation_app, station_path=None):
    if world is None:
        from isaacsim.core.api import World
        world = World(stage_units_in_meters=1.0)
        physics_ctx = world.get_physics_context()
        physics_ctx.set_gravity(0.0)

    if stage is None:
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World")

    # 1. 조명 및 우주 배경 설정
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(10.0)
    
    sun_light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    sun_light.CreateIntensityAttr(4000.0) 
    sun_light.CreateAngleAttr(0.5)
    sun_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))
    sun_light.AddRotateXYZOp().Set(Gf.Vec3f(-30, 60, 0))

    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.core.utils.viewports import set_camera_view
    
    # 2. 커스텀 도킹 스테이션 생성 (ISS 대체)
    if station_path is None:
        station_path = "/World/StationBase"
        iss_usd_path = os.path.join(THIS_DIR, "..", "resources", "assets", "space_station.glb")
        station_prim = stage.DefinePrim(station_path, "Xform")
        station_prim.GetReferences().AddReference(iss_usd_path)
    else:
        station_prim = stage.GetPrimAtPath(station_path)
    
    # 기존 XformOpOrder를 지우면 space_environment_v10_leo.py가 설정한 궤도 위치(X=424.5)가 초기화되므로 삭제 금지!
    station_xform = UsdGeom.XformCommonAPI(station_prim)
    
    # 우주 정거장 크기 증가 (Scale 2.0 적용) - 이미 로드된 경우 생략 가능
    if not station_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        station_xform.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0)) # 원점 배치
        station_xform.SetRotate(Gf.Vec3f(0.0, 90.0, 0.0))
        station_xform.SetScale(Gf.Vec3f(2.0, 2.0, 2.0))
        UsdPhysics.RigidBodyAPI.Apply(station_prim)
        UsdPhysics.RigidBodyAPI(station_prim).CreateKinematicEnabledAttr(True)
    
    # 정거장 내부 메시에 단순한 큐브 충돌체 적용
    for prim in Usd.PrimRange(station_prim):
        if prim.IsA(UsdGeom.Mesh):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(True)
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr().Set("none")
    # ====================================================================
    # [사용자 설정] 도킹 손잡이 및 마커 위치 설정
    # 정거장 로컬 좌표계 기준입니다. 원하는 위치로 자유롭게 변경하세요!
    # ====================================================================
    # 우주 정거장의 시각적 모델(Model)이 2.5414배 스케일업 되어 있으므로,
    # 동일한 표면 위치에 부착하려면 핸들의 로컬 좌표 오프셋에도 2.5414를 곱해주어야 합니다.
    CUSTOM_HANDLE_POS = np.array([-0.00475, 0.20282, 0.34249]) * 2.5414
    CUSTOM_HANDLE_ROT = np.array([180.0, 0.0, 90.0]) # Euler XYZ (도) - 수평으로 90도 회전
    
    # 도킹 손잡이를 우주 정거장의 자식(Child)으로 생성하여 정거장에 완벽히 종속시킴
    docking_handle_path = f"{station_path}/DockingHandle"
    docking_handle = UsdGeom.Cylinder.Define(stage, docking_handle_path)
    
    # 우주 공간에서 잘 보이도록 두께 6cm, 길이 80cm로 설정 (스케일 1.0 기준)
    docking_handle.CreateRadiusAttr(0.06)
    docking_handle.CreateHeightAttr(0.8)
    docking_handle.GetDisplayColorAttr().Set([(0.0, 1.0, 0.0)]) # 초록색
    
    docking_handle.CreateAxisAttr("Y")
    
    handle_xform = UsdGeom.XformCommonAPI(docking_handle)
    handle_xform.SetTranslate(Gf.Vec3d(*CUSTOM_HANDLE_POS))
    handle_xform.SetRotate(Gf.Vec3f(*CUSTOM_HANDLE_ROT))
    
    # 월드 좌표 업데이트 대기
    for _ in range(3):
        simulation_app.update()
    
    # DockingHandle은 부모(StationBase)가 이미 RigidBody이므로 추가로 RigidBodyAPI를 적용하면 충돌 오류가 납니다.
    # CollisionAPI만 적용하여 부모의 물리 시스템에 종속된 물리 충돌체로만 사용합니다.
    UsdPhysics.CollisionAPI.Apply(docking_handle.GetPrim())
    
    # 3. 로봇(Chaser) 로드 (통합된 m0609_rg2_space.urdf 사용)
    robot_prim_path = "/World/Robot"
    chaser_base_path = f"{robot_prim_path}/chaser_base"
    
    print("[Setup] doosan_loader.py의 설정을 사용하여 URDF 임포트 중...")
    
    resources_dir = os.path.join(THIS_DIR, "..", "resources", "robots")
    if resources_dir not in sys.path:
        sys.path.append(resources_dir)
        
    from doosan_loader import spawn_doosan_rg2
    
    # 정거장의 실제 월드 좌표를 읽어서 기준점으로 삼음 (v15 맵 연동 시 정거장이 멀리 있을 수 있음)
    m = UsdGeom.Xformable(station_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    station_world_pos = np.array([t[0], t[1], t[2]])
    
    hm = UsdGeom.Xformable(docking_handle.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    ht = hm.ExtractTranslation()
    handle_world_pos = np.array([ht[0], ht[1], ht[2]])
    
    # 유저 요청: 멀리서 날아오는 씬 (태양 전지판 충돌을 피하기 위해 손잡이 정면 방향으로 150m 떨어진 곳에 스폰)
    outward_vec = handle_world_pos - station_world_pos
    outward_dir = outward_vec / (np.linalg.norm(outward_vec) + 1e-6)
    robot_spawn_pos = handle_world_pos + outward_dir * 150.0
    spawn_doosan_rg2(robot_prim_path, translation=tuple(robot_spawn_pos))
    
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    
    def disable_fix_base(prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            fix_base_attr = prim.GetAttribute("physxArticulation:fixBase")
            if not fix_base_attr:
                prim.CreateAttribute("physxArticulation:fixBase", Sdf.ValueTypeNames.Bool, False).Set(False)
            else:
                fix_base_attr.Set(False)
        for child in prim.GetChildren():
            disable_fix_base(child)
            
    disable_fix_base(robot_prim)

    # 비콘 라이트 부착
    beacon_light = UsdLux.SphereLight.Define(stage, f"{chaser_base_path}/BeaconLight")
    beacon_light.CreateRadiusAttr(2.0)
    beacon_light.CreateIntensityAttr(5000.0)
    beacon_light.GetColorAttr().Set(Gf.Vec3f(0.0, 0.8, 1.0))
    
    # Base 색상을 회색으로 변경 (재질 바인딩 해제 후 색상 적용)
    base_visual = stage.GetPrimAtPath(f"{chaser_base_path}/visuals")
    if base_visual.IsValid():
        for prim in Usd.PrimRange(base_visual):
            if prim.IsA(UsdGeom.Mesh):
                if prim.HasAPI(UsdShade.MaterialBindingAPI):
                    UsdShade.MaterialBindingAPI(prim).UnbindAllBindings()
                UsdGeom.Mesh(prim).GetDisplayColorAttr().Set([(0.5, 0.5, 0.5)])
    
    for name in ("left_inner_finger", "right_inner_finger"):
        for p in Usd.PrimRange(robot_prim):
            if p.GetName() == name:
                for child in p.GetChildren():
                    if child.HasAPI(UsdPhysics.CollisionAPI):
                        apply_high_friction(stage, child.GetPath().pathString)

    # 4. 제어기 및 뷰 초기화
    art_view = ArticulationView(prim_paths_expr=robot_prim_path, name="robot_view")
    robot_art = Articulation(prim_path=robot_prim_path, name="m0609_robot")
    world.scene.add(art_view)
    world.scene.add(robot_art)

    dc = _dynamic_control.acquire_dynamic_control_interface()
    # world.reset() is omitted because integrated environment resets it.
    
    gains_set = False
    compliance_active = False
    
    # 로봇 팔의 처음 조인트각 설정: 기본 비행 자세 (0, 0, 90, 0, 90, 0)
    init_joints = np.zeros(8)
    init_joints[:6] = np.deg2rad([0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
    
    if timeline is None:
        timeline = omni.timeline.get_timeline_interface()
    # timeline.play() is handled by caller (e.g., _sim_ctx.reset() or play())
    
    # 파티클 불꽃 설정
    num_particles = 100
    particles = []
    fire_root_path = f"{chaser_base_path}/ThrusterExhaust"
    UsdGeom.Xform.Define(stage, fire_root_path)
    
    import random
    for i in range(num_particles):
        p_path = f"{fire_root_path}/p_{i}"
        sphere = UsdGeom.Sphere.Define(stage, p_path)
        sphere.CreateRadiusAttr(0.04)
        translate_op = sphere.AddTranslateOp()
        scale_op = sphere.AddScaleOp()
        color_attr = sphere.GetDisplayColorAttr()
        imageable = UsdGeom.Imageable(sphere.GetPrim())
        imageable.MakeInvisible()
        particles.append({
            "imageable": imageable, "translate_op": translate_op, "scale_op": scale_op, "color_attr": color_attr,
            "life": random.uniform(0.0, 1.0), "speed": random.uniform(0.5, 2.0),
            "offset_y": random.uniform(-0.1, 0.1), "offset_z": random.uniform(-0.1, 0.1)
        })

    fire_light_path = f"{chaser_base_path}/ThrusterLight"
    fire_light = UsdLux.SphereLight.Define(stage, fire_light_path)
    fire_light.CreateRadiusAttr(0.2)
    fire_light.CreateIntensityAttr(80000.0)
    fire_light.CreateColorAttr(Gf.Vec3f(1.0, 0.4, 0.0))
    fire_light.AddTranslateOp().Set(Gf.Vec3d(-0.4, 0.0, 0.0))
    
    fire_light_prim = UsdGeom.Imageable(fire_light.GetPrim())
    if fire_light_prim:
        fire_light_prim.MakeInvisible()

    fire_is_on = False
    prev_time = world.current_time
    station_handle = _dynamic_control.INVALID_HANDLE
    robot_handle = _dynamic_control.INVALID_HANDLE

    print("==================================================")
    print(" 🚀 단방향 접근 방식의 커스텀 도킹 시뮬레이션 가동!")
    print("==================================================")

    # RMPFlow 제어기 초기화
    resources_dir = os.path.join(THIS_DIR, "..", "resources")
    M0609_URDF_PATH = os.path.join(resources_dir, "m0609_aruco_detect", "doosan-robot2", "urdf", "m0609_isaac_sim.urdf")
    M0609_DESCRIPTION_PATH = os.path.join(resources_dir, "m0609_aruco_detect", "m0609_rg2_description.yaml")
    M0609_RMPFLOW_CONFIG_PATH = os.path.join(resources_dir, "m0609_aruco_detect", "m0609_rmpflow_common.yaml")

    cspace_controller = RMPFlowController(
        name="m0609_aruco_servo_rmpflow_controller",
        robot_articulation=robot_art,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name="tool0",
    )
    cspace_controller.reset()

    # 비전 서보잉 초기화
    # 손목 카메라 부착 (tool0 기준)
    gripper_camera_parent = find_prim_path_by_name(robot_prim_path, "angle_bracket")
    if gripper_camera_parent is None:
        raise RuntimeError("angle_bracket을 찾을 수 없습니다.")

    realsense_prim_path = attach_realsense_d455(
        parent_prim_path=gripper_camera_parent,
        child_name="realsense_d455",
        translation=(0.0, 0.045, 0.05),
        rpy_deg=(0.0, -90.0, 90.0),
    )
    
    # USD reference 해결 대기
    for _ in range(5):
        simulation_app.update()
        
    _stage = omni.usd.get_context().get_stage()
    
    # RealSense USD 내장 OmniVision 카메라 탐색 후 래핑 및 회전 적용
    ov_cam_path = find_prim_path_by_name(realsense_prim_path, "Camera_OmniVision_OV9782_Color")
    if not ov_cam_path:
        raise RuntimeError("RealSense USD 내에서 Camera_OmniVision_OV9782_Color를 찾을 수 없습니다.")
        
    _cam_prim = _stage.GetPrimAtPath(ov_cam_path)
    from pxr import Vt
    _xf = UsdGeom.Xformable(_cam_prim)
    _existing = [op.GetOpName() for op in _xf.GetOrderedXformOps()]
    _rot_op = _xf.AddRotateZOp(UsdGeom.XformOp.PrecisionFloat, opSuffix="extra")
    _rot_op.Set(90.0) # 카메라의 추가적인 축 틀어짐(90도) 적용 (Master Prompt)
    _cam_prim.GetAttribute("xformOpOrder").Set(Vt.TokenArray(_existing + [_rot_op.GetOpName()]))
    
    wrist_cam = WristCamera.from_existing_prim(
        prim_path=ov_cam_path,
        resolution=(640, 480),
    )
    wrist_cam.initialize()
    wrist_cam.camera.set_clipping_range(0.01, 1000.0)
    
    # 핀홀 카메라 모델 파라미터 (640x480 기준 대략적인 값)
    cam_fx, cam_fy = 500.0, 500.0
    cam_cx, cam_cy = 320.0, 240.0
    wrist_cam.camera.set_opencv_pinhole_properties(
        cx=cam_cx, cy=cam_cy, fx=cam_fx, fy=cam_fy, pinhole=[0.0]*12
    )
    K_cam = np.array([
        [cam_fx,    0.0, cam_cx],
        [   0.0, cam_fy, cam_cy],
        [   0.0,    0.0,    1.0]
    ])
    
    # (ArUco Tracker는 완전히 제거됨)
    # 카메라가 -X 방향을 볼 때 직관적인 이미지 평면 이동
    servo = VisualServoController(
        image_size=(640, 480),
        kp=0.001,
        max_step=0.05,
        tolerance_px=10,
        pixel_to_world_xy=np.array([[1.0, 0.0], [0.0, -1.0]]) # 디버그 로그 기반 최종 완벽 매핑 (X 반전, Y 정방향)
    )
    viewer = CameraViewer(enabled=True)

    # 게인 설정
    Kp_pos = 5000.0 # 베이스 좌표계 정렬 속도를 높이기 위해 강성 상향 조정
    Kp_rot, Kd_rot = 8000.0, 3000.0 # 부드러운 회전을 위해 토크 하향 조정
    
    PHASE = 0 # 0: 대기 (ROS 명령 대기)
    approach_dist = 2.0
    phase_2_start = 0.0
    arm_reaching = False # Phase 3에서 로봇팔을 뻗고 있는지 여부
    reach_start_pos = None # 로봇팔 뻗기 시작점
    arm_extension_dist = 0.0 # 로봇팔 전진 거리
    visual_offset_right = 0.0 # 시각 서보잉 좌우(Right) 보정량
    visual_offset_up = 0.0 # 시각 서보잉 상하(Up) 보정량
    vision_lost_frames = 0 # 타겟 시야 상실 프레임 카운터
    
    # 도킹 포트 글로벌 좌표 (사용자가 손잡이 위치를 마음대로 바꿔도 자동으로 월드 좌표를 추적하도록 복구!)
    handle_prim = stage.GetPrimAtPath(docking_handle_path)
    if handle_prim.IsValid():
        transform_matrix = UsdGeom.Xformable(handle_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = transform_matrix.ExtractTranslation()
        port_world_pos = np.array([translation[0], translation[1], translation[2]])
    else:
        port_world_pos = np.array([3.0, 0.0, 0.35]) # 예외 발생 시 기본값
    print(f"🎯 [Setup] 도킹 손잡이 월드 좌표 (자동 추적): {port_world_pos}")
    
    R_target = np.eye(3)
    phase_2_ee_quat = np.array([0.5, 0.5, 0.5, 0.5]) # Base 프레임 기준: 카메라가 바닥이 아닌 '전방(스테이션)'을 똑바로 바라보도록 자세 전면 수정
    phase_4_ee_quat = np.array([0.5, 0.5, 0.5, 0.5])
    
    # IK 및 조인트 제어 변수
    dof_targets = np.zeros(8)
    ee_link_idx = -1
    ee_handle = _dynamic_control.INVALID_HANDLE
    
    # === 동적 우주 정거장 설정 ===
    # 통합 맵(v15)에서는 우주 정거장의 위치가 space_environment_v10_leo.py의 orbit_step에 의해 KINEMATIC하게 공전 궤도를 돕니다.
    # 따라서 iss_berthing.py에서 수동으로 직선/회전 이동을 누적시키면 두 코드가 충돌(오버라이드)하여 정거장이 궤도를 이탈합니다.
    # 통합 시에는 이 값을 0으로 설정하여, 순수하게 맵의 공전만 반영되도록 합니다.
    station_linear_vel = np.array([0.0, 0.0, 0.0])
    station_angular_vel = np.array([0.0, 0.0, 0.0])
    
    prev_port_world_pos = port_world_pos.copy()
    port_world_vel = np.zeros(3)
    
    # 사용자의 요청에 따라 카메라 시점 강제 변경을 제거했습니다.
    start_time = time.time()
    print_timer = 0.0
    
    class EMA:
        def __init__(self, alpha=0.1):
            self.alpha = alpha
            self.val = None
        def update(self, new_val):
            if self.val is None:
                self.val = new_val
            else:
                self.val = self.alpha * new_val + (1.0 - self.alpha) * self.val
            return self.val

    class QuatEMA:
        def __init__(self, alpha=0.1):
            self.alpha = alpha
            self.val = None
        def update(self, new_quat):
            if self.val is None:
                self.val = new_quat
            else:
                if np.dot(self.val, new_quat) < 0:
                    new_quat = -new_quat
                q = (1.0 - self.alpha) * self.val + self.alpha * new_quat
                self.val = q / np.linalg.norm(q)
            return self.val


    visual_R_target = None
    visual_up_hint = None
    visual_up_hint_alt = None
    det = None  # ArUco 탐지 결과 초기화

    print("=== 우주 쓰레기 수거 시뮬레이션 시작 ===")
    try:
        while True:
            sim_dt_in = yield
            if sim_dt_in is not None:
                dt = sim_dt_in
            else:
                dt = 1.0/60.0
                
            # world.step(render=True) is handled by the caller (adr_integrated.py or standalone)
            
            if timeline is not None and not timeline.is_playing():
                continue
                
            if not getattr(art_view, "_initialized", False):
                art_view.initialize()
                robot_art.initialize()
                setattr(art_view, "_initialized", True)
                setattr(robot_art, "_initialized", True)
                print("[Setup] ArticulationView 런타임 초기화 완료.")
            
            if not gains_set and getattr(art_view, "num_dof", 0) > 0:
                art_view.set_gains(kps=np.ones(art_view.num_dof)*10000.0, kds=np.ones(art_view.num_dof)*1000.0)
                
                actual_init_joints = np.zeros(art_view.num_dof)
                actual_init_joints[:6] = init_joints[:6]
                art_view.set_joint_positions(np.array([actual_init_joints]))
                gains_set = True
                
            if timeline is not None and not timeline.is_playing():
                continue
                
            current_time = time.time() - start_time
            
            # 파티클 및 비전 제어를 위한 sim_dt 계산
            sim_dt = dt
            
            # --- 1. 우주 정거장 동적 이동 및 회전 ---
            if sim_dt > 0:
                translation, rotation, scale, pivot, rotOrder = station_xform.GetXformVectors(Usd.TimeCode.Default())
                curr_trans = translation
                curr_rot = rotation
                
                # Numpy 배열 대신 직접 계산하여 적용 (XformCommonAPI 활용)
                new_trans = Gf.Vec3d(
                    curr_trans[0] + station_linear_vel[0] * sim_dt,
                    curr_trans[1] + station_linear_vel[1] * sim_dt,
                    curr_trans[2] + station_linear_vel[2] * sim_dt
                )
                new_rot = Gf.Vec3f(
                    curr_rot[0] + station_angular_vel[0] * sim_dt,
                    curr_rot[1] + station_angular_vel[1] * sim_dt,
                    curr_rot[2] + station_angular_vel[2] * sim_dt
                )
                
                station_xform.SetTranslate(new_trans)
                station_xform.SetRotate(new_rot)
            
            # --- 2. 이동하는 도킹 손잡이 위치 실시간 추적 ---
            if handle_prim.IsValid():
                transform_matrix = UsdGeom.Xformable(handle_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                translation = transform_matrix.ExtractTranslation()
                port_world_pos = np.array([translation[0], translation[1], translation[2]])
                
                # 핸들의 현재 월드 회전 추출
                handle_rot = transform_matrix.ExtractRotationQuat() # Gf.Quatd
                handle_curr_quat = [handle_rot.imaginary[0], handle_rot.imaginary[1], handle_rot.imaginary[2], handle_rot.real] # x,y,z,w
            
            # 도킹 손잡이 실시간 이동 속도 계산 (수치 미분)
            if sim_dt > 0:
                port_world_vel = (port_world_pos - prev_port_world_pos) / sim_dt
            prev_port_world_pos = port_world_pos.copy()
            
            # 접근 벡터 (우주 정거장 중심 -> 도킹 손잡이를 향하는 밖으로 뻗어나가는 벡터)
            if 'curr_trans' not in locals():
                translation, rotation, scale, pivot, rotOrder = station_xform.GetXformVectors(Usd.TimeCode.Default())
                curr_trans = translation
            
            station_center = np.array([curr_trans[0], curr_trans[1], curr_trans[2]])
            outward_vec = port_world_pos - station_center
            outward_dir = outward_vec / (np.linalg.norm(outward_vec) + 1e-6)
            
            # 대기 지점을 아예 삭제하고 바로 접근 목표 지점(approach_dist 앞)으로 설정
            approach_target = port_world_pos + outward_dir * approach_dist
            
            prev_time = current_time
            
            if current_time - print_timer > 1.0:
                dt_print = current_time - print_timer
            else:
                dt_print = 0.0
            
            if robot_handle == _dynamic_control.INVALID_HANDLE:
                robot_handle = dc.get_rigid_body(chaser_base_path)
                
                # 시뮬레이션 시작 시 한 번만 RealSense Mesh의 물리 비활성화 처리 (오류 방지)
                if robot_handle != _dynamic_control.INVALID_HANDLE:
                    for prim in Usd.PrimRange(_stage.GetPrimAtPath(realsense_prim_path)):
                        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
                        if prim.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                continue
                
            if station_handle == _dynamic_control.INVALID_HANDLE:
                station_handle = dc.get_rigid_body(station_path)
                if station_handle == _dynamic_control.INVALID_HANDLE:
                    continue
            
            # 스테이션은 이제 Kinematic이므로 속도를 설정할 필요 없음 (오류 방지)
            
            # 로봇 위치/자세
            robot_pose = dc.get_rigid_body_pose(robot_handle)
            robot_vel = dc.get_rigid_body_linear_velocity(robot_handle)
            robot_ang_vel = dc.get_rigid_body_angular_velocity(robot_handle)
            
            curr_pos = np.array([robot_pose.p.x, robot_pose.p.y, robot_pose.p.z])
            curr_vel = np.array([robot_vel.x, robot_vel.y, robot_vel.z])
            
            # --- 정거장 분석적 동역학 정보 계산 (피드포워드용) ---
            station_angular_vel_rad = np.deg2rad(station_angular_vel)
            r_offset = curr_pos - station_center
            v_coorbit = station_linear_vel + np.cross(station_angular_vel_rad, r_offset)
            dist_to_approach_target = np.linalg.norm(approach_target - curr_pos)
            
            look_dir = -outward_dir
            
            # --- 목표 및 자세 덮어쓰기 ---
            approach_target = port_world_pos + outward_dir * approach_dist
            
            # 로봇의 베이스 방향 정렬
            station_pose = dc.get_rigid_body_pose(station_handle)
            R_station = Rotation.from_quat([station_pose.r.x, station_pose.r.y, station_pose.r.z, station_pose.r.w]).as_matrix()
            
            # 윗면(로봇팔 부착면, Z축)이 정거장을 향하도록 설정
            target_Z = -outward_dir
            
            # 위성의 X축(앞면)이 스테이션의 로컬 Z축을 향하도록 설정
            target_X = R_station @ np.array([0.0, 0.0, 1.0])
            
            # 수직화를 위해 외적 수행
            target_Y = np.cross(target_Z, target_X)
            target_Y = target_Y / (np.linalg.norm(target_Y) + 1e-6)
            
            target_X = np.cross(target_Y, target_Z) # 완벽한 직교화
            
            # 최종 베이스 목표 회전 (Z축이 밖을 향함)
            R_target = np.column_stack([target_X, target_Y, target_Z])
            
            # 완벽한 동적 3D 조준 및 파지 각도 동기화 (M0609 + RG2 구조 반영)
            # 1. 그리퍼의 전방(Z축)은 항상 손잡이에 수직이 되도록 (-outward_dir) 설정하여 손목 꺾임 방지
            target_Z = -outward_dir
            
            # 2. 정거장 회전 및 ArUco 마커 방향에 맞춘 그리퍼 자전(Twist) 동기화
            station_pose = dc.get_rigid_body_pose(station_handle)
            R_station = Rotation.from_quat([station_pose.r.x, station_pose.r.y, station_pose.r.z, station_pose.r.w]).as_matrix()
            
            # 우주 정거장 손잡이(초록색 바)의 실제 월드 방향 계산
            R_handle_station = Rotation.from_euler('xyz', CUSTOM_HANDLE_ROT, degrees=True).as_matrix()
            R_handle_world = R_station @ R_handle_station
            handle_dir_world = R_handle_world[:, 1] # 손잡이는 Y축으로 생성됨
            
            # 사용자의 요청: 로봇 6번 조인트 각도를 초록색 손잡이 각도에 완벽하게 맞추기
            # up_hint를 손잡이 방향으로 강제하여 그리퍼의 축이 손잡이와 나란해지도록 정렬
            up_hint = handle_dir_world
            if abs(np.dot(target_Z, up_hint)) > 0.98:
                up_hint = R_handle_world[:, 0]

            target_X = np.cross(up_hint, target_Z)
            target_X = target_X / (np.linalg.norm(target_X) + 1e-6)
            
            # 3. Y축은 Z와 X의 외적으로 생성
            target_Y = np.cross(target_Z, target_X)
            
            # 회전 행렬 구성
            R_ee_base = np.column_stack((target_X, target_Y, target_Z))
            
            # (사용자 요청에 따라 초록색 손잡이 자체가 가로로 회전되었으므로 그리퍼의 추가 90도 회전을 생략합니다)
            # --- 로봇팔 고정 자세 및 위성 주도 카메라 조준 ---
            # 준비 자세: 로컬 베이스(+Z축) 방향으로 곧게 0.5m 뻗은 상태
            # 위성 본체가 이 고정된 카메라 오프셋을 역산하여, 카메라가 손잡이를 정확히 바라보도록 스스로 이동합니다.
            
            fingertip_local = np.array([0.02705, -0.00953, 0.15537])
            cam_offset_local = np.array([0.0, 0.045, 0.05])
            
            fingertip_world = R_ee_base @ fingertip_local
            cam_offset_world = R_ee_base @ cam_offset_local
            
            # 1. 로봇팔의 능동적 카메라 조준 (월드 좌표 기준)
            # 위성이 흔들리더라도 로봇팔이 즉각적으로 관절을 움직여 카메라를 손잡이에 완벽히 고정시킵니다.
            ik_target_ready = port_world_pos + outward_dir * approach_dist - cam_offset_world
            
            # 2. 실제 잡을 때의 타겟 (그리퍼 끝이 손잡이에 닿는 위치)
            ik_target_grip = port_world_pos - fingertip_world
            
            # 평상시에는 로봇팔이 고정 자세를 유지합니다.
            ik_target = ik_target_ready
            
            # 그리퍼의 손가락은 Y축 방향으로 벌어지므로, 손잡이(수직)를 잡으려면 Y축이 좌우여야 합니다.
            # 위에서 90도 회전을 적용했으므로 R_ee_base를 그대로 타겟으로 사용합니다.
            R_ee_target = Rotation.from_matrix(R_ee_base)
            
            dynamic_ee_quat = np.array([R_ee_target.as_quat()[3], R_ee_target.as_quat()[0], R_ee_target.as_quat()[1], R_ee_target.as_quat()[2]]) # w,x,y,z
            
            # 카메라 프레임 및 상태 오버레이 갱신 (모든 Phase 공통)
            rgb = wrist_cam.get_rgb()
            depth_map = wrist_cam.get_depth()
            det = None
            green_found = False
            green_depth = None
            green_cx, green_cy = None, None
            if rgb is not None:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                # --- 초록색 핸들 감지 및 시각화 로직 추가 ---
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                lower_green = np.array([40, 100, 50])
                upper_green = np.array([80, 255, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                green_found = False
                green_handle_dir_cv = None
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 50:
                        green_found = True
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        green_cx = x + w / 2.0
                        green_cy = y + h / 2.0
                        
                        if depth_map is not None:
                            roi_depth = depth_map[y:y+h, x:x+w]
                            valid_depth = roi_depth[np.isfinite(roi_depth)]
                            if len(valid_depth) > 0:
                                green_depth = np.mean(valid_depth)
                                
                        # 손잡이의 실제 2D 기울기(Angle) 추정
                        rect = cv2.minAreaRect(largest_contour)
                        box = cv2.boxPoints(rect)
                        box = np.intp(box) # np.int0 is deprecated in NumPy 1.24
                        cv2.drawContours(bgr, [box], 0, (0, 255, 0), 2)
                        
                        depth_str = f"Depth: {green_depth:.2f}m" if green_depth is not None else "Depth: N/A"
                        cv2.putText(bgr, f"GREEN HANDLE ({depth_str})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # (이전의 시각적 롤 각도 계산 코드 삭제됨: GT 정렬로 대체)
                        
                        # 오버레이 (오렌지/노랑: 목표 타겟 중심 십자선)
                        cx, cy = int(green_cx), int(green_cy)
                        cv2.line(bgr, (cx - 50, cy), (cx + 50, cy), (255, 255, 0), 1) # 가로선
                        cv2.line(bgr, (cx, cy - 50), (cx, cy + 50), (255, 255, 0), 1) # 세로선
                
                # camera_viewer.py 렌더링 버그(bbox unpacking) 방지
                if det is not None and det.found:
                    if green_found:
                        det.bbox = (x, y, w, h)
                        det.cx = x + w / 2
                        det.cy = y + h / 2
                    elif det.bbox is None:
                        det.bbox = (0, 0, 0, 0)
                        det.cx = 0
                        det.cy = 0
                
                if (det is not None and det.found) or green_found:
                    vision_lost_frames = 0
                else:
                    vision_lost_frames += 1
                

                # 시각화(색상 + 마커)가 적용된 BGR을 다시 RGB로 변환하여 뷰어에 전달
                rgb_display = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                
                state_str = f"Phase {PHASE}"
                dist_str = f"{np.linalg.norm(curr_pos - port_world_pos):.2f}m"
                extra_lines = [
                    f"base_x: {curr_pos[0]:.2f}",
                    f"base_yz: {curr_pos[1]:.2f}, {curr_pos[2]:.2f}",
                    f"dist: {dist_str}"
                ]
                
                key = viewer.update(rgb_display, det, state_str=state_str, extra_lines=extra_lines)
                if key == ord('q'):
                    break
            

            # 로봇팔 부착면(Z축)이 손잡이를 향하므로, 위성 본체는 손잡이 정면 직선상에 위치하도록 합니다.
            # Z축 방향으로 로봇팔이 자연스럽게 뻗을 수 있도록 0.35m 정도 거리를 두고 대기합니다.
            approach_target = port_world_pos + outward_dir * (approach_dist + 0.35)
            dist_to_approach_target = np.linalg.norm(approach_target - curr_pos)
            
            # [피드포워드 추종] 대기 지점(Standoff Point)의 실시간 월드 궤도 속도 정밀 계산 (수치 미분)
            # station_linear_vel 등 물리 기반 속도가 제대로 읽히지 않는 문제를 해결하여 완벽하게 궤도를 따라가도록 보장합니다.
            if sim_dt > 0:
                if 'prev_approach_target' not in locals():
                    prev_approach_target = approach_target.copy()
                port_world_vel_feedforward = (approach_target - prev_approach_target) / sim_dt
                prev_approach_target = approach_target.copy()
            else:
                port_world_vel_feedforward = np.zeros(3)
            
            R_target_rot = Rotation.from_matrix(R_target)
            
            # [자세 및 각속도 동기화 제어] (Euler 각도 짐벌락 방지를 위해 v11_v8track 쿼터니언 기반 오차 사용)
            R_base = Rotation.from_quat([robot_pose.r.x, robot_pose.r.y, robot_pose.r.z, robot_pose.r.w])
            
            # 현재 자세에서 목표 자세로 가는 회전 (World Frame)
            R_err = R_target_rot * R_base.inv()
            q_err = R_err.as_quat() # [x, y, z, w]
            
            # 최단 경로 선택 (w가 음수이면 뒤집기)
            if q_err[3] < 0.0:
                q_err = -q_err
            
            # 허수부 벡터 (회전축 * sin(theta/2))
            q_errv = q_err[:3] 
            
            # 완벽한 마주보기 유지를 위해 각속도(angular velocity) 항상 100% 동기화
            err_ang_vel_world = station_angular_vel_rad - np.array([robot_ang_vel.x, robot_ang_vel.y, robot_ang_vel.z])
            
            # v11_v8track 방식의 안정적인 쿼터니언 기반 토크 제어 (오차 증폭을 위해 2.0 곱함)
            torque_world = (Kp_rot * q_errv * 2.0) + (Kd_rot * err_ang_vel_world)
            
            # v11_v8track처럼 월드 좌표계 기준으로 직접 토크를 가함
            dc.apply_body_torque(robot_handle, carb.Float3(torque_world[0], torque_world[1], torque_world[2]), True)
            
            force_x, force_y, force_z = 0.0, 0.0, 0.0
            
            if PHASE == -1:
                # 10m 밖으로 고속 후퇴 (궤도 재진입 준비)
                pos_error = approach_target - curr_pos
                distance = np.linalg.norm(pos_error)
                rel_vel = np.clip(pos_error * 5.0, -20.0, 20.0) # 고속 복귀를 위해 속도 증가
                target_vel = port_world_vel_feedforward + rel_vel
                vel_error = target_vel - curr_vel
                force_np = Kp_pos * vel_error
                force_x, force_y, force_z = force_np[0], force_np[1], force_np[2]
                
                # 로봇팔은 기본 대기 자세 유지
                dof_targets[:6] = init_joints[:6]
                art_view.set_joint_position_targets(np.array([dof_targets]))
                
                if dt_print > 0:
                    print(f"[Phase -1] 궤도 밖 10m로 후퇴 중... 남은 거리: {distance:.1f}m")
                    print_timer = current_time
                    
                if distance < 1.0:
                    print("✅ [Phase -1 -> 1] 10m 후퇴 완료! 다시 궤도 접근을 시작합니다.")
                    PHASE = 1
                    approach_dist = 2.0
                    
            elif PHASE == 0:
                global external_command
                if dt_print > 0:
                    print("⏳ [Phase 0] 대기 중... 외부에서 'start_approach' 명령을 기다립니다.")
                    print_timer = current_time
                    
                if external_command == 'start_approach':
                    print("🚀 [Phase 0 -> 1] 접근 시작 명령 수신! 목표물로 전진합니다.")
                    PHASE = 1
                    external_command = None
                    
            elif PHASE == 1:
                # 안전 거리 접근
                pos_error = approach_target - curr_pos
                distance = np.linalg.norm(pos_error)
                true_dist_to_target = distance 
                
                # --- 우주 정거장 회피 및 궤도(Orbit) 비행 로직 ---
                station_center = np.array([curr_trans[0], curr_trans[1], curr_trans[2]])
                vec_from_center = curr_pos - station_center
                dist_from_center = np.linalg.norm(vec_from_center)
                center_to_robot = vec_from_center / (dist_from_center + 1e-6)
                
                # 로봇과 손잡이의 방향 일치도 (-1: 완전 반대편, 1: 완벽히 같은 방향)
                alignment = np.dot(center_to_robot, outward_dir)
                
                # [수정] Phase 1a (장거리)에서는 20m/s 고속 유지, 5m 이내(Phase 1b 진입 즈음)에서 감속
                if true_dist_to_target > 5.0:
                    max_speed = 20.0
                    target_speed = max_speed
                else:
                    max_speed = 8.0
                    target_speed = max(0.5, max_speed * (true_dist_to_target / 5.0))
                
                # 1. 목표 지점(손잡이)으로 향하는 기본 P 제어 속도
                rel_vel = (pos_error / distance) * target_speed if distance > 0 else np.zeros(3)
                
                # 근접 거리(4m 이내)에서는 능동적이고 정밀한 P 제어로 전환 (오버슈트 방지)
                if true_dist_to_target < 4.0:
                    rel_vel = np.clip(pos_error * 3.0, -3.0, 3.0)
                
                # 2. v11_v8track 참조: 목표의 완전한 궤도 이동 속도(회전 포함)를 100% 피드포워드로 반영
                target_vel = port_world_vel_feedforward + rel_vel
                
                # 3. 동적 우주 정거장 충돌 회피 (Repulsion) - 정렬 조건 완화 및 반발력 축소
                if dist_from_center < 20.0 and alignment < 0.8:
                    repulsion_strength = (20.0 - dist_from_center) / 20.0
                    repulsive_vel = center_to_robot * (max_speed * repulsion_strength * 0.3)
                    target_vel += repulsive_vel
                
                # [추가] 회피 벡터 합산으로 인해 속도가 폭주(급발진)하는 것을 막기 위한 안전장치
                speed_norm = np.linalg.norm(target_vel)
                if speed_norm > max_speed + 2.0:
                    target_vel = (target_vel / speed_norm) * (max_speed + 2.0)
                
                vel_error = target_vel - curr_vel
                force_np = Kp_pos * vel_error
                force_x, force_y, force_z = force_np[0], force_np[1], force_np[2]
                
                # RMPFlow 베이스 포즈 지속 업데이트
                base_link_handle = dc.get_rigid_body(f"{robot_prim_path}/base_link")
                if base_link_handle != _dynamic_control.INVALID_HANDLE:
                    robot_base_pose = dc.get_rigid_body_pose(base_link_handle)
                    cspace_controller._motion_policy.set_robot_base_pose(
                        robot_position=np.array([robot_base_pose.p.x, robot_base_pose.p.y, robot_base_pose.p.z]),
                        robot_orientation=np.array([robot_base_pose.r.w, robot_base_pose.r.x, robot_base_pose.r.y, robot_base_pose.r.z])
                    )
                
                # J1~J6 홈 포즈로 강체 고정 (거리가 멀 때는 기본 자세, 궤도 진입시 0도로 위를 봄)
                if true_dist_to_target > 1.5:
                    dof_targets[:6] = init_joints[:6]
                else:
                    dof_targets[:6] = np.zeros(6)
                art_view.set_joint_position_targets(np.array([dof_targets]))
                
                # 정렬 완료 및 근접 거리 확인
                if true_dist_to_target < 0.2 and alignment > 0.95:
                    print("🎯 [Phase 1 -> 2] 완벽하게 마주보는 상태(정렬 완료, 속도 동기화). 비전 서보잉 시작!")
                    PHASE = 2
                    phase_2_start = current_time
                    target_pos = approach_target.copy() # Phase 2부터는 target_pos를 직접 업데이트
                    
                elif dt_print > 0:
                    if true_dist_to_target > 1.5:
                        print(f"[Phase 1a] 손잡이로 직선 접근 중... Distance: {true_dist_to_target:.1f}m | Target Speed: {target_speed:.1f}m/s")
                    else:
                        print(f"[Phase 1b] 궤도 동기화 및 정밀 접근 중... Distance: {true_dist_to_target:.1f}m | Target Speed: {target_speed:.1f}m/s")
                    print_timer = current_time
                    
            elif PHASE == 2:
                # 베이스 절대 좌표 정렬 및 속도 매칭
                target_pos = approach_target.copy()
                pos_error = target_pos - curr_pos
                distance_error = np.linalg.norm(pos_error)
                # 안정적인 좌표계 정렬을 위해 속도 증폭을 2.0으로 낮춤 (과도한 진동 방지)
                target_vel = port_world_vel_feedforward + np.clip(pos_error * 2.0, -3.0, 3.0)
                vel_error = target_vel - curr_vel
                force_np = Kp_pos * vel_error
                force_x, force_y, force_z = force_np[0], force_np[1], force_np[2]
                
                base_link_handle = dc.get_rigid_body(f"{robot_prim_path}/base_link")
                robot_base_pose = dc.get_rigid_body_pose(base_link_handle)
                cspace_controller._motion_policy.set_robot_base_pose(
                    robot_position=np.array([robot_base_pose.p.x, robot_base_pose.p.y, robot_base_pose.p.z]),
                    robot_orientation=np.array([robot_base_pose.r.w, robot_base_pose.r.x, robot_base_pose.r.y, robot_base_pose.r.z])
                )
                
                if dt_print > 0:
                    print(f"[Phase 2] 베이스 좌표계 정렬 중... 거리 에러: {distance_error:.2f}m")
                    print_timer = current_time
                    
                # 피드포워드 상대 속도 오차 확인
                rel_speed = np.linalg.norm(curr_vel - port_world_vel_feedforward)
                
                # 정렬 속도 향상을 위해 조건 완화 (0.4m, 0.5m/s 이내면 통과)
                if distance_error < 0.4 and rel_speed < 0.5:
                    if dt_print > 0:
                        print("✅ [Phase 2] 베이스 Standoff 완벽 정지 및 정렬 완료! 'start_docking' 명령 대기 중...")
                        
                    if external_command == 'start_docking':
                        print("🚀 [Phase 2 -> 3] 도킹 명령 수신! 로봇팔 비전 서보잉 & Creep 시작.")
                        PHASE = 3
                        phase_3_start = current_time
                        external_command = None

                # 궤도 진입 및 주차 중에는 로봇팔이 흔들리지 않도록 (사용자 요청에 따라) 곧게 뻗은 0도 자세 고정
                dof_targets[:6] = np.zeros(6)
                art_view.set_joint_position_targets(np.array([dof_targets]))
                
            elif PHASE == 3:
                # [안전 로직] 초록색 손잡이(또는 마커)가 시야에서 완전히 사라질 경우, 1m보다 멀 때만 후퇴합니다.
                # (1m 이내로 초근접했을 때는 그리퍼가 시야를 가릴 수 있으므로 맹목적으로 계속 전진합니다.)
                if vision_lost_frames > 15 and approach_dist > 1.0:
                    print("⚠️ [VISION LOSS] 타겟을 놓쳤습니다! 시각 정보를 초기화하고 궤도 밖 10m에서 다시 접근합니다.")
                    PHASE = -1
                    approach_dist = 10.0 # 10m로 후퇴
                    arm_reaching = False
                    arm_extension_dist = 0.0
                    visual_offset_right = 0.0
                    visual_offset_up = 0.0
                    visual_offset_roll = 0.0
                    
                    # 시각 잠금을 해제하여 옛날 시각 좌표가 아닌 실시간 예측 궤도(GT)를 추종하도록 복구
                    vision_locked = False
                    visual_port_world_pos = None
                    
                    if compliance_active:
                        stiff_kps = np.ones(art_view.num_dof) * 10000.0
                        stiff_kds = np.ones(art_view.num_dof) * 1000.0
                        art_view.set_gains(kps=stiff_kps, kds=stiff_kds)
                        compliance_active = False
                    continue # 이번 프레임 건너뛰고 다음 루프에서 Phase -1 로직 진입
                    
                # 베이스 능동 제어 및 속도 매칭
                target_pos = approach_target.copy()
                pos_error = target_pos - curr_pos
                target_vel = port_world_vel_feedforward + np.clip(pos_error * 5.0, -3.0, 3.0)
                vel_error = target_vel - curr_vel
                force_np = Kp_pos * vel_error
                force_x, force_y, force_z = force_np[0], force_np[1], force_np[2]
                
                if ee_handle == _dynamic_control.INVALID_HANDLE:
                    try:
                        ee_link_idx = art_view.get_link_index("tool0")
                        ee_handle = dc.get_rigid_body(f"{robot_prim_path}/tool0")
                    except KeyError:
                        ee_link_idx = art_view.get_link_index(art_view.body_names[-1])
                        ee_handle = dc.get_rigid_body(f"{robot_prim_path}/{art_view.body_names[-1]}")
                        
                    if ee_handle == _dynamic_control.INVALID_HANDLE:
                        ee_handle = dc.get_rigid_body(f"{robot_prim_path}/{art_view.body_names[-1]}")
                        
                ee_pose = dc.get_rigid_body_pose(ee_handle)
                ee_pos = np.array([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z])
                
                base_link_handle = dc.get_rigid_body(f"{robot_prim_path}/base_link")
                robot_base_pose = dc.get_rigid_body_pose(base_link_handle)
                cspace_controller._motion_policy.set_robot_base_pose(
                    robot_position=np.array([robot_base_pose.p.x, robot_base_pose.p.y, robot_base_pose.p.z]),
                    robot_orientation=np.array([robot_base_pose.r.w, robot_base_pose.r.x, robot_base_pose.r.y, robot_base_pose.r.z])
                )
                
                if rgb is not None and det is not None:
                    curr_yz = np.array([ee_pose.p.y, ee_pose.p.z])
                    target_yz, err_px = servo.update(curr_yz, det) # 로깅 및 오버레이용으로만 유지
                
                # Ground Truth 기준 EE와 손잡이 간의 절대 거리
                dist_to_handle_gt = np.linalg.norm(port_world_pos - ee_pos)
                
                # [Phase 3] 시각 서보잉 정렬 및 베이스 추력 접근 (Creep & Dock)
                err_px = 999.0
                if green_found and green_cx is not None:
                    err_x = green_cx - 320.0
                    err_y = green_cy - 240.0
                    err_px = np.sqrt(err_x**2 + err_y**2)
                    
                    # R_ee_base[:, 0]은 UP, R_ee_base[:, 1]은 RIGHT 방향을 향합니다.
                    visual_offset_right += (err_x * 0.001)  # err_x(오른쪽) -> RIGHT 축(+) 이동
                    visual_offset_up -= (err_y * 0.001)     # err_y(아래쪽) -> UP 축(-) 이동
                
                # ik_target 계산: 베이스 전진(approach_dist)에 연동되도록 ik_target_ready를 실시간으로 사용
                ik_target = ik_target_ready.copy()
                ik_target += R_ee_base[:, 1] * visual_offset_right  # RIGHT 축 이동
                ik_target += R_ee_base[:, 0] * visual_offset_up     # UP 축 이동
                
                # 실제 로봇팔의 물리적 도달 거리를 측정
                fingertip_pos = ee_pos + R_ee_base @ fingertip_local
                dist_fingertip_to_handle = np.linalg.norm(port_world_pos - fingertip_pos)
                
                # 시각 정렬 오차 허용 범위를 다시 조임 (에러 80px 이하일 때만 접근)
                if err_px < 80.0:
                    # 정렬 완료! 위성 추력으로 천천히 다가감 (접근 속도 10cm/s)
                    creep_speed = 0.10
                    approach_dist -= creep_speed * sim_dt
                    
                    if dt_print > 0:
                        depth_str = f"{green_depth:.2f}m" if green_depth is not None else "N/A"
                        print(f"🚀 [Phase 3] 정렬 완료. 베이스 접근 중... 에러: {err_px:.1f}px | 깊이: {depth_str} | 물리적 거리: {dist_fingertip_to_handle:.2f}m")
                        print_timer = current_time
                else:
                    # 정렬 중 (전진 멈춤)
                    if dt_print > 0:
                        print(f"🦾 [Phase 3] 타겟 중앙 정렬 중... 에러: {err_px:.1f}px")
                        print_timer = current_time
                        
                # 파지 실패(발산) 또는 초록 손잡이 시야 상실 시 로봇팔 접고 10m 밖으로 후퇴
                # (단, 거리가 50cm 이내로 가까워진 경우는 시야각 문제일 수 있으므로 후퇴하지 않음)
                if err_px > 400.0 and dist_fingertip_to_handle > 0.50:
                    visual_offset_right = 0.0
                    visual_offset_up = 0.0
                    PHASE = -1
                    approach_dist = 10.0 # 10m로 완전 후퇴
                    print("⚠️ [Phase 3] 초록색 손잡이를 놓쳤습니다! 10m 밖으로 고속 후퇴하여 재접근합니다.")
                    continue
                    
                # 파지 직전(카메라 기준 약 30cm 이내)에 진입하면 조인트 강성을 낮춰 충격 완화 (Compliance Control)
                if green_depth is not None and green_depth <= 0.30 and not compliance_active:
                    compliance_kps = np.ones(art_view.num_dof) * 10000.0
                    compliance_kds = np.ones(art_view.num_dof) * 1000.0
                    for idx in range(6):
                        compliance_kps[idx] = 200.0  # 강성 대폭 인하 (원활한 정렬과 맞물림 유도)
                        compliance_kds[idx] = 20.0   # 댐핑 인하
                    art_view.set_gains(kps=compliance_kps, kds=compliance_kds)
                    compliance_active = True
                    print("🎛️ [Compliance Control] 그리퍼 접촉 전 관절 강성 인하 완료.")
                    
                # 물리적으로 손잡이에 닿을 거리(그리퍼 길이만큼)가 카메라 깊이(depth)로 감지되었을 때 결속
                # 그리퍼 길이를 고려해 카메라 깊이를 0.20m 임계값으로 설정합니다.
                is_distance_reached = (green_depth is not None and green_depth <= 0.20)
                if is_distance_reached:
                    # 실제 물리 기반 파지 (그리퍼 닫기)
                    depth_info = f"{green_depth:.2f}m"
                    print(f"✅ [Phase 3 -> 4] 정밀 정렬 및 삽입 완료 (에러: {err_px:.1f}px, 거리: {depth_info})! 그리퍼를 닫아 구조물을 결속합니다.")
                    PHASE = 4
                    phase_4_start = current_time
                    # FixedJoint 용접 제거: 마찰력과 실제 그리퍼 조인트 힘으로만 유지

                actions = cspace_controller.forward(
                    target_end_effector_position=ik_target,
                    target_end_effector_orientation=dynamic_ee_quat
                )
                
                if actions is not None and actions.joint_positions is not None:
                    dof_targets[:6] = actions.joint_positions[:6]
                
                art_view.set_joint_position_targets(np.array([dof_targets]))
                
            elif PHASE == 4:
                # 성공 상태 유지(HOLD): 베이스는 계속 손잡이(위성)를 추적하며, 그리퍼를 강하게 닫음
                target_pos = approach_target.copy()
                pos_error = target_pos - curr_pos
                target_vel = port_world_vel_feedforward + np.clip(pos_error * 5.0, -3.0, 3.0)
                vel_error = target_vel - curr_vel
                force_np = Kp_pos * vel_error
                force_x, force_y, force_z = force_np[0], force_np[1], force_np[2]
                
                # 그리퍼 조인트(6번, 7번)를 닫힌 위치(0.8)로 목표 설정하여 꽉 잡음
                dof_targets[6] = 0.8
                dof_targets[7] = 0.8
                
                art_view.set_joint_position_targets(np.array([dof_targets]))
                
                # 도킹 성공(파지 완료) 여부 확인: 속도가 거의 0인 상태로 중간에 멈춰있어야 진짜 성공
                gripper_pos = art_view.get_joint_positions()[0][6]
                gripper_vel = abs(art_view.get_joint_velocities()[0][6])
                if dt_print > 0:
                    print(f"✅ 파지 유지 중 (HOLD)... (그리퍼 관절 위치: {gripper_pos:.2f} rad, 속도: {gripper_vel:.3f})")
                    if 0.15 < gripper_pos < 0.75 and gripper_vel < 0.05 and (current_time - phase_4_start) > 2.0:
                        print(f"🎉 성공적으로 구조물을 파지했습니다! (목표 0.8 rad, 현재 {gripper_pos:.2f} rad에서 마찰력으로 홀드됨)")
                    
                    if external_command == 'start_undocking':
                        print("📡 [ROS 명령 수신] 도킹 해제 명령! 그리퍼를 열고 뒤로 후퇴합니다.")
                        PHASE = 5
                        phase_5_start = current_time
                        external_command = None
                    
                    print_timer = current_time

            elif PHASE == 5:
                # 1. 그리퍼 완전 개방 (-0.84 rad)
                dof_targets[6] = -0.84
                dof_targets[7] = -0.84
                
                # 팔은 기존 뻗은 상태 유지
                ik_target = ik_target_ready.copy()
                actions = cspace_controller.forward(
                    target_end_effector_position=ik_target,
                    target_end_effector_orientation=dynamic_ee_quat
                )
                if actions is not None and actions.joint_positions is not None:
                    dof_targets[:6] = actions.joint_positions[:6]
                art_view.set_joint_position_targets(np.array([dof_targets]))
                
                # 2. 그리퍼가 충분히 열리도록 1초 대기 후 베이스 후퇴
                if (current_time - phase_5_start) > 1.0:
                    approach_dist += 0.5 * sim_dt # 초당 0.5m씩 후퇴
                    if approach_dist > 5.0:
                        approach_dist = 5.0 # 최대 5m까지만 후퇴
                
                # 베이스 추력 제어 (위성의 이동 속도를 계속 동기화하며 안전하게 분리)
                target_pos = port_world_pos + outward_dir * approach_dist
                pos_error = target_pos - curr_pos
                target_vel = port_world_vel_feedforward + np.clip(pos_error * 5.0, -3.0, 3.0)
                vel_error = target_vel - curr_vel
                force_np = Kp_pos * vel_error
                force_x, force_y, force_z = force_np[0], force_np[1], force_np[2]
                
                if dt_print > 0:
                    if approach_dist >= 5.0:
                        print("✅ [Phase 5] 안전 거리(5m) 확보. 도킹 해제 시퀀스 완료 및 대기 중.")
                    else:
                        print(f"👋 [Phase 5] 도킹 해제 및 안전하게 후퇴 중... (현재 목표 거리: {approach_dist:.2f}m)")
                    print_timer = current_time

            # 컴플라이언스 비활성화 시 관절 강성 복구
            if compliance_active and (PHASE != 3 or ('dist_to_handle_gt' in locals() and dist_to_handle_gt > 0.40)):
                art_view.set_gains(kps=np.ones(art_view.num_dof)*10000.0, kds=np.ones(art_view.num_dof)*1000.0)
                compliance_active = False
                print("🎛️ [Compliance Control] 관절 강성 원상 복구 완료.")

            # 추력 인가
            thrust_force = np.array([force_x, force_y, force_z])
            if np.linalg.norm(thrust_force) > 0:
                R_base = Rotation.from_quat([robot_pose.r.x, robot_pose.r.y, robot_pose.r.z, robot_pose.r.w])
                force_local = R_base.inv().apply(thrust_force)
                dc.apply_body_force(robot_handle, carb.Float3(force_local[0], force_local[1], force_local[2]), carb.Float3(0.0, 0.0, 0.0), False)
                
            # 파티클 업데이트 로직
            
            thrust_mag = np.sqrt(force_x**2 + force_y**2 + force_z**2)
            # 로봇이 멀리 있을 때 시야에 잘 보이도록 접근 중(Phase 0, 1)에는 불꽃을 항상 켭니다.
            if (thrust_mag > 10.0 or PHASE < 2) and PHASE < 3:
                if not fire_is_on:
                    if fire_light_prim: fire_light_prim.MakeVisible()
                    for p in particles: p["imageable"].MakeVisible()
                    fire_is_on = True
                    
                flicker = random.uniform(30000.0, 100000.0)
                fire_light.GetIntensityAttr().Set(flicker)
                
                # --- 다방향 스러스터 (RCS) 로직 ---
                # 가해지는 힘(thrust)의 반대 방향으로 불꽃 발사 (작용-반작용)
                thrust_world = np.array([force_x, force_y, force_z])
                if thrust_mag > 1e-3:
                    thrust_dir_world = thrust_world / thrust_mag
                    R_base = Rotation.from_quat([robot_pose.r.x, robot_pose.r.y, robot_pose.r.z, robot_pose.r.w])
                    R_inv = R_base.inv()
                    fire_dir_local = R_inv.apply(-thrust_dir_world) # 로컬 좌표계 기준 반사 방향
                else:
                    fire_dir_local = np.array([1.0, 0.0, 0.0]) # 기본값
                    
                # 조명 위치도 불꽃이 나오는 쪽으로 이동
                fire_light.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(*(fire_dir_local * 0.5)))
                
                for p in particles:
                    if "offset_x" not in p: p["offset_x"] = random.uniform(-0.15, 0.15)
                    
                    p["life"] += sim_dt * p["speed"] * 2.0
                    if p["life"] > 1.0:
                        p["life"] = 0.0
                        p["offset_x"] = random.uniform(-0.15, 0.15)
                        p["offset_y"] = random.uniform(-0.15, 0.15)
                        p["offset_z"] = random.uniform(-0.15, 0.15)
                        p["speed"] = random.uniform(0.5, 2.0)
                    
                    t = p["life"]
                    
                    # 불꽃의 기본 중심축 이동 경로
                    dist = 0.5 + (t * 2.5)
                    base_pos = fire_dir_local * dist
                    
                    # 퍼짐(노이즈) 효과
                    spread = 1.0 - t
                    nx = p["offset_x"] * spread
                    ny = p["offset_y"] * spread
                    nz = p["offset_z"] * spread
                    
                    p["translate_op"].Set(Gf.Vec3d(base_pos[0] + nx, base_pos[1] + ny, base_pos[2] + nz))
                    scale = 1.0 - (t * 0.9)
                    p["scale_op"].Set(Gf.Vec3f(scale, scale, scale))
                    
                    r = 1.0 - (t * 0.2)
                    g = max(0.0, 1.0 - (t * 2.5))
                    b = 0.0
                    p["color_attr"].Set([(r, g, b)])
            else:
                if fire_is_on:
                    if fire_light_prim: fire_light_prim.MakeInvisible()
                    for p in particles: p["imageable"].MakeInvisible()
                    fire_is_on = False

    except KeyboardInterrupt:
        print("시뮬레이션 종료.")
    except Exception as e:
        import traceback
        print(f"\\n[CRITICAL ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        if 'viewer' in locals():
            viewer.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

_berthing_coro = None

def setup_berthing(stage, world, dc, timeline, simulation_app, station_path=None):
    global _berthing_coro
    _berthing_coro = _berthing_generator(stage, world, dc, timeline, simulation_app, station_path)
    next(_berthing_coro) # Run until the first yield (setup complete)

def step_berthing(dt):
    global _berthing_coro
    if _berthing_coro is not None:
        try:
            _berthing_coro.send(dt)
        except StopIteration:
            pass

def main():
    from isaacsim.simulation_app import SimulationApp
    simulation_app = SimulationApp({"headless": False})
    # Standalone execution
    from isaacsim.core.api import World
    from omni.isaac.dynamic_control import _dynamic_control
    import omni.timeline
    
    world = World(stage_units_in_meters=1.0)
    world.get_physics_context().set_gravity(0.0)
    dc = _dynamic_control.acquire_dynamic_control_interface()
    timeline = omni.timeline.get_timeline_interface()
    
    setup_berthing(None, world, dc, timeline, simulation_app, None)
    
    world.reset()
    while simulation_app.is_running():
        world.step(render=True)
        step_berthing(1.0/60.0)
        
    simulation_app.close()

if __name__ == "__main__":
    main()
