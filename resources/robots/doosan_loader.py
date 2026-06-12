import os
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverterCfg

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = f"{THIS_DIR}/m0609_rg2_space.urdf"

def spawn_doosan_rg2(prim_path: str, translation: tuple[float, float, float] = (0.0, 0.0, 0.0), orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)):
    """
    안전하게 Doosan M0609 + RG2 로봇을 시뮬레이션에 소환하는 범용 유틸리티 함수입니다.
    다른 커스텀 스크립트(예: 02_iss_berthing.py 등)에서도 이 함수만 호출하면 
    강화학습 환경과 100% 동일한 설정(마찰력, 센서, 드라이브 등)으로 로봇이 안전하게 로드됩니다.
    
    사용 예시:
    from doosan_loader import spawn_doosan_rg2
    spawn_doosan_rg2("/World/Robot", translation=(15.0, 0.0, 0.0))
    """
    spawn_cfg = sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        activate_contact_sensors=True,  # 가장 중요한 센서 활성화!
        fix_base=False,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=200.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
            fix_root_link=False,
        ),
        collider_type="convex_decomposition",
    )
    
    # IsaacLab의 스포너를 사용하여 로드 및 변환 수행
    prim = spawn_cfg.func(
        prim_path=prim_path,
        cfg=spawn_cfg,
        translation=translation,
        orientation=orientation
    )
    print(f"✅ [doosan_loader] 성공적으로 로봇을 소환했습니다: {prim_path}")
    return prim
