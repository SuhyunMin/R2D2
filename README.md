# ADR 인지·추적 시스템 (Isaac Sim 5.1.0)

LEO 체이서 위성이 텀블링하는 비협조 표적(사다리 형상 잔해)에 대해 자율
랑데부·근접운용(RPO)을 수행하는 시뮬레이션. 원거리(지상 카탈로그 EKF) →
중거리(Mono RGB YOLO + KeypointNet + PnP 6D pose) → 근거리(LiDAR + ICP)로
이어지는 인지 파이프라인과 체이서 추적·자세제어를 다룬다.

---

## ⚠️ 실행 전 반드시 할 것

업로드된 `adr_perception__1_.py` 파일명을 **`adr_perception.py`로 바꿔야** 한다.
`adr_integrated.py`가 504번째 줄에서 `import adr_perception as perc`로 가져오기
때문에, 이름이 안 맞으면 `ModuleNotFoundError`로 즉시 죽는다.

```bash
mv adr_perception__1_.py adr_perception.py
```

---

## 파일 구성

| 파일 | 역할 | 진입점? |
|------|------|---------|
| `adr_integrated.py` | **메인 통합 진입점.** SimulationApp 생성 → 씬 빌드 → YOLO/KeypointNet 로드 → 인지·추적·제어 메인 루프 | ✅ 직접 실행 |
| `space_environment_v10_leo.py` | LEO 궤도 환경 빌더. 스케일된 Kepler 물리(1 unit≈287km, GM_sim=60.62). `build_scene(sim_app)` 제공 | ✅ standalone 또는 import |
| `adr_perception__1_.py` → `adr_perception.py` | 인지 함수 모음(Isaac 비의존: numpy/cv2/torch/ultralytics). YOLO 탐지 + KeypointNet + PnP + EKF | ❌ import 전용 |
| `iss_berthing.py` | ISS 버싱/비주얼 서보 데모(별도 시나리오). RMPFlow + ArUco 기반 | ✅ 직접 실행 |

### 의존 관계
```
adr_integrated.py
 ├─ import space_environment_v10_leo as scene   (씬 빌드)
 ├─ import adr_perception as perc               (인지)
 └─ import m0609_pick_place_controller          (외부 모듈, 아래 참고)

iss_berthing.py
 └─ wrist_camera / visual_servo_controller / m0609_rmpflow_controller
    / realsense_mount / camera_viewer / doosan_loader  (외부 모듈, 아래 참고)
```

`adr_integrated.py`와 `iss_berthing.py`는 서로 독립적인 시나리오다.
한 번에 하나만 실행한다(프로세스당 SimulationApp 1개 제약).

---

## 실행 환경

- **Isaac Sim 5.1.0** — 반드시 번들 파이썬(`./python.sh`)으로 실행. 시스템
  python으로 돌리면 `isaacsim` / `omni` / `pxr` import에서 실패한다.
- 런타임 라이브러리(확인된 조합):
  - torch 2.7 + cu128 (CUDA True)
  - numpy **1.26.4** (2.x는 `omni.syntheticdata`를 깨뜨림)
  - opencv-python 4.8.x
  - ultralytics 8.4.x
  - open3d (근거리 ICP/LiDAR)
  - scipy (`iss_berthing.py`의 `scipy.spatial.transform`)
- GPU: RTX 계열(개발은 RTX 4060)

---

## 필요한 모델·데이터·에셋 (업로드에 미포함)

이 4개 스크립트만으로는 안 돌아간다. 아래가 지정된 경로에 있어야 한다.
모두 `adr_integrated.py` 상단/중단에 **하드코딩**돼 있다.

**모델/데이터:**
- YOLO 가중치 — `/home/rokey/space_debris_ai/runs/detect/space_debris-2/weights/best.pt`
- KeypointNet 체크포인트 — `/home/rokey/space_debris_ai/scripts/checkpoints/best.pt`
- 3D 키포인트 — `/home/rokey/space_debris_ai/datasets/ladder_pose_v2/keypoints_3d.json`

**에셋(GLB/USDZ):** `~/space_debris_ai/assets/raw`, `~/space_debris_ai/assets/usd`
- earth.glb, sci-fi_space_station.glb, space_satellite.glb
- 잔해 GLB: nexus_1st_stage / ladder_metallic_tool / nasa_astronaut_helmet / wall-e / sat02_body_satellite
- meteorite.glb, UFO.usdz, R2D2.usdz, Meteor-M2_No.usdz, Satellite_lnb_Building_roof_top.usdz

**로봇/외부 모듈:**
- m0609 URDF — `/home/rokey/dev_ws/isaac_sim/src/doosan-robot2/urdf/m0609_isaac_sim.urdf`
- OnRobot RG2 URDF — `/home/rokey/dev_ws/isaac_sim/src/onrobot_rg2/urdf/onrobot_rg2.urdf`
- R2D2.usd — `/home/rokey/dev_ws/isaac_sim/src/r2d2/R2D2.usd`
- `m0609_pick_place_controller.py` — `/home/rokey/dev_ws/isaac_sim/IsaacLab/scripts/fly_move/`
- `iss_berthing.py` 전용: `wrist_camera`, `visual_servo_controller`,
  `m0609_rmpflow_controller`, `realsense_mount`, `camera_viewer`, `doosan_loader`
  (`../resources/m0609_aruco_detect/`에 있어야 함) + `m0609_rg2_description.yaml`,
  `m0609_rmpflow_common.yaml`

> 경로가 본인 환경과 다르면 스크립트 상단의 상수들을 직접 고쳐야 한다.
> 핵심: `adr_integrated.py`의 510~512줄(모델 경로), 30·41~44줄(외부 모듈/URDF 경로).

---

## 실행 방법

Isaac Sim 설치 디렉터리에서 번들 파이썬으로 실행한다.

### 1) 메인 통합 시나리오
```bash
# 0. 파일명 정리
mv adr_perception__1_.py adr_perception.py

# 1. 두 파일을 같은 폴더(또는 PYTHONPATH 상)에 둔다
#    adr_integrated.py, space_environment_v10_leo.py, adr_perception.py

# 2. Isaac 번들 파이썬으로 실행
cd ~/isaacsim          # Isaac Sim 설치 위치
./python.sh /path/to/adr_integrated.py
```
- GUI가 뜨면 ▶ **PLAY**를 눌러야 물리/궤도 루프가 돈다.
- 첫 실행은 `FORCE_RECONVERT_ASSETS = True`라 GLB→USD 변환 때문에 느리다.
  변환이 안정되면 `space_environment_v10_leo.py`의 `FORCE_RECONVERT_ASSETS`를
  `False`로 바꿔서 재변환을 건너뛴다.
- 인지 디버그 이미지는 파일로 저장된다(headless OpenCV라 `imshow` 불가):
  `~/yolo_debug_frames/latest.jpg`. 자동 새로고침되는 이미지 뷰어로 본다.
- LiDAR/포인트클라우드 디버그: `~/lidar_debug/` (`target_cloud_view.jpg`,
  `target_cloud_world.ply` 등).

### 2) 씬만 단독 확인
```bash
./python.sh /path/to/space_environment_v10_leo.py
```
`__main__`일 때만 자체 SimulationApp을 만든다. 통합 스크립트가 import할 때는
호출자(`adr_integrated.py`)가 만든 걸 쓰므로 중복 생성되지 않는다.

### 3) ISS 버싱 데모
```bash
./python.sh /path/to/iss_berthing.py
```
위에 적은 `m0609_aruco_detect` 리소스 모듈들이 갖춰져 있어야 한다.
(주의: 이 파일은 구버전 `omni.isaac.core` API를 사용한다.)

---

## 카메라/좌표계 메모

- 카메라 내부 파라미터는 **하드코딩 금지**. `_compute_intrinsics()`가
  `focalLength=24 / horizontalAperture=20.955` + 1280×720에서
  FX=FY≈1466, CX=640, CY=360을 계산한다(과거 77px 버그 방지).
- GLB→USD는 Y-up→Z-up 변환 → 긴 축이 로컬 Z. identity=broadside,
  +90X=end-on(축퇴).
- PnP·ICP 모두 trimesh CAD 프레임 사용. `T_pnp(cad→world)`를 ICP 초기값으로
  그대로 쓴다(오프셋 불필요).
- YOLO는 사다리를 `collectable_debris`(conf 0.7~0.8)로 탐지.
  KeypointNet ckpt 형식: `{model, num_kp=9, crop=256, hm=64}`.

---

## 트러블슈팅

| 증상 | 원인/조치 |
|------|-----------|
| `ModuleNotFoundError: adr_perception` | 파일명을 `adr_perception.py`로 리네임 |
| `isaacsim`/`omni`/`pxr` import 실패 | 시스템 python으로 실행함. `./python.sh` 사용 |
| `omni.syntheticdata` 깨짐 | numpy 2.x 설치됨. 1.26.4로 다운그레이드 |
| 모델/에셋 파일 없음 경고 | 위 "필요한 모델·데이터·에셋" 경로에 파일 배치 또는 상수 수정 |
| 궤도가 안 움직임 | GUI에서 ▶ PLAY 안 누름 |
| 첫 실행이 너무 느림 | GLB→USD 재변환. 이후 `FORCE_RECONVERT_ASSETS=False` |
| `m0609_pick_place_controller` 못 찾음 | `sys.path.append` 경로(30줄)와 실제 위치 확인 |

---

## 알려진 한계 (정직하게)

- **단안(monocular) 회전 추정 한계**: 사다리 대칭성 때문에 회전 추정이
  ~40° 부근에서 plateau. 이게 LiDAR 근거리 단계를 두는 이유다. depth/RGB-D는
  "Isaac에서만 되는 가짜 feature"라 의도적으로 배제했다.
- **물리 핸드오버(그래스핑) 미구현**: 현재 단계까지는 설계만 되어 있고
  실제 결속·포획 물리는 데모 시나리오(iss_berthing)에서 별도로 다룬다.
- **하드코딩 경로**: `/home/rokey/...`, `/home/rokey/dev_ws/...`가 곳곳에
  박혀 있다. 다른 환경에선 직접 고쳐야 한다.
- **수치 결과는 실측 기준**: 재투영 ~5~6px, translation error ~0.45m,
  카메라-표적 거리 ~12m 등은 실제 측정값. 과장된 통합 성능 주장은 없음.
