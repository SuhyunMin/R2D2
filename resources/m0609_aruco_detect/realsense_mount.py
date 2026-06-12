from pxr import Usd, UsdGeom, UsdPhysics, Gf
import omni.usd

REALSENSE_D455_USD_REL = "/Isaac/Sensors/Intel/RealSense/rsd455.usd"


def _get_assets_root():
    try:
        from isaacsim.storage.native import get_assets_root_path
        return get_assets_root_path()
    except ImportError:
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        return get_assets_root_path()


def attach_realsense_d455(parent_prim_path: str,
                          child_name: str = "realsense_d455",
                          translation=(0.05, 0.0, 0.02),
                          rpy_deg=(180.0, 0.0, 0.0),
                          usd_path: str | None = None) -> str:
    """
    parent_prim_path 아래에 RealSense D455 mesh 를 reference 로 부착.
    반환: 생성된 RealSense Xform prim path (wrist_camera.py 에서 sensor 부모로 사용)
    """
    stage = omni.usd.get_context().get_stage()

    if usd_path is None:
        root = _get_assets_root()
        if root is None:
            raise RuntimeError(
                "Isaac Sim assets root 를 찾을 수 없습니다. "
                "Nucleus 연결을 확인하거나 usd_path 를 직접 지정하세요."
            )
        usd_path = root + REALSENSE_D455_USD_REL

    rs_prim_path = f"{parent_prim_path}/{child_name}"

    existing = stage.GetPrimAtPath(rs_prim_path)
    if existing and existing.IsValid():
        stage.RemovePrim(rs_prim_path)

    rs_prim = stage.DefinePrim(rs_prim_path, "Xform")
    rs_prim.GetReferences().AddReference(usd_path)

    xform = UsdGeom.Xformable(rs_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*rpy_deg))

    # RealSense USD 내 RigidBodyAPI/CollisionAPI 비활성화:
    # angle_bracket(이미 rigid body)의 자식으로 두면 PhysX 계층 충돌이 발생하므로
    # 물리 시뮬레이션이 진행된 후 prim 이 생성되는 시점에 처리한다.
    # (world.reset() 이전에는 reference 내부 prim 이 아직 로드 안 됨)
    # → post_reset 에서 _disable_physics_on_realsense() 를 호출하거나,
    #   아래와 같이 Xform 자체에 physics:rigidBodyEnabled = false 오버라이드 적용.
    # 현재 단계에서 할 수 있는 것: Xform prim 에 RigidBodyAPI 가 있으면 비활성화
    for prim in Usd.PrimRange(stage.GetPrimAtPath(rs_prim_path)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
            print(f"[realsense_mount] disabled RigidBodyAPI on {prim.GetPath()}")

    print(f"[realsense_mount] D455 attached at {rs_prim_path}")
    print(f"                  source USD = {usd_path}")
    return rs_prim_path
