import sys
import os
import time

# [환경 충돌 방지] 사용자의 터미널에 꼬여있는 ROS 2 시스템 경로(Humble/Jazzy)를 강제 제거하여 Isaac Sim 전용 rclpy를 보호합니다.
sys.path = [p for p in sys.path if '/opt/ros' not in p and 'jazzy_ws' not in p and 'humble_ws' not in p]
if 'PYTHONPATH' in os.environ:
    os.environ['PYTHONPATH'] = ':'.join([p for p in os.environ['PYTHONPATH'].split(':') if '/opt/ros' not in p and 'jazzy_ws' not in p and 'humble_ws' not in p])

# Isaac Sim 내장 rclpy (Python 3.11 컴파일 버전) 경로 주입
ISAAC_RCLPY_DIR = "/home/rokey/dev_ws/venv/isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/rclpy"
if ISAAC_RCLPY_DIR not in sys.path:
    sys.path.append(ISAAC_RCLPY_DIR)

# Isaac Sim Python 3.11 환경에 맞춘 로컬 빌드 경로 추가
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTALL_DIR = os.path.join(WORKSPACE_DIR, "install", "mission_interfaces", "lib", "python3.11", "site-packages")
if INSTALL_DIR not in sys.path:
    sys.path.append(INSTALL_DIR)

# [LD_LIBRARY_PATH 우회] C++ 바인딩(.so) 로드를 위해 동적 라이브러리 경로 추가
LIB_DIR = os.path.join(WORKSPACE_DIR, "install", "mission_interfaces", "lib")
ISAAC_LIB_DIR = "/home/rokey/dev_ws/venv/isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib"
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')

needs_restart = False
if LIB_DIR not in current_ld_path:
    current_ld_path = LIB_DIR + ":" + current_ld_path
    needs_restart = True
if ISAAC_LIB_DIR not in current_ld_path:
    current_ld_path = ISAAC_LIB_DIR + ":" + current_ld_path
    needs_restart = True

if needs_restart:
    os.environ['LD_LIBRARY_PATH'] = current_ld_path
    # 리눅스는 파이썬 실행 중 환경변수를 바꿔도 C++ dlopen이 인식하지 못하므로, 새 환경변수로 파이썬을 즉시 재시작합니다.
    os.execv(sys.executable, [sys.executable] + sys.argv)

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

try:
    from mission_interfaces.action import MissionControl
except ImportError:
    print("[ERROR] mission_interfaces 패키지를 찾을 수 없습니다. Python 3.11용 빌드를 먼저 진행해주세요.")
    sys.exit(1)

import threading

# ============================================================
# [미션 체인] 전체 파이프라인 단계 순서.
#   각 단계가 success 결과를 반환하면 다음 명령을 자동 발행한다.
#   스폰(자동) -> 도킹 -> 언도킹 -> 랑데뷰 -> 집기(hook) -> 전달(hook) -> 정거장 복귀 -> 재도킹
# ============================================================
MISSION_SEQUENCE = [
    "start_approach",   # 1) 정거장 접근 (스폰 직후)
    "start_docking",    # 2) 도킹
    "start_undocking",  # 3) 언도킹
    "ladder capture",   # 4) 랑데뷰 (ADR)
    "grasp",            # 5) 집기 (best-effort hook)
    "deliver",          # 6) 전달 (placeholder hook)
    "start_approach",   # 7) 정거장 복귀 (타겟 자동으로 정거장 리셋)
    "start_docking",    # 8) 재도킹
]


class MissionActionClient(Node):
    def __init__(self):
        super().__init__('mission_controller_action_client')
        self._action_client = ActionClient(self, MissionControl, 'mission_control')
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._mission_active = False   # 미션 체인 실행 중 여부
        self._mission_idx = 0          # 현재 단계 인덱스
        
    def start_input_thread(self):
        self._input_thread.start()

    def start_mission(self):
        """전체 파이프라인 체인 시작 (첫 명령 발행 → 이후 결과 콜백이 자동 진행)."""
        self._mission_active = True
        self._mission_idx = 0
        print("\n=============================================")
        print(f"🚀 [미션 체인 시작] 총 {len(MISSION_SEQUENCE)}단계")
        print("   " + " → ".join(MISSION_SEQUENCE))
        print("=============================================")
        print(f"\n➡️  [1/{len(MISSION_SEQUENCE)}] {MISSION_SEQUENCE[0]}")
        self.send_goal(MISSION_SEQUENCE[0])

    def _input_loop(self):
        print("\n=============================================")
        print("🚀 미션 컨트롤 센터 활성화 🚀")
        print("지원하는 기본 명령어 예시:")
        print("  - start_approach  : 원거리에서 정거장 접근 시작 (Phase 0 -> 1)")
        print("  - start_docking   : 베이스 정렬 후 로봇팔 도킹 시작 (Phase 2 -> 3)")
        print("  - start_undocking : 도킹 해제 및 후퇴 (Phase 4 -> 5)")
        print("  - ladder capture  : 사다리(DebrisLadder) 좌표로 동적 타겟 변경 후 랑데뷰 시작")
        print("  - walle capture   : 윌리(DebrisWalle) 좌표로 동적 타겟 변경 후 랑데뷰 시작")
        print("  - grasp           : 집기 단계 실행 (best-effort hook)")
        print("  - deliver         : 전달 단계 실행 (placeholder hook)")
        print("  - start_mission   : ⭐ 전체 파이프라인 자동 체인 실행 (스폰~재도킹)")
        print("  - c               : 현재 실행 중인 명령 취소")
        print("=============================================\n")
        
        while rclpy.ok():
            try:
                cmd = input("\n명령을 입력하세요 (종료하려면 q 입력) > ").strip()
                if not cmd:
                    continue
                if cmd.lower() == 'q' or cmd.lower() == 'quit':
                    print("미션 컨트롤러를 종료합니다.")
                    rclpy.shutdown()
                    break
                
                if cmd.lower() == 'c' or cmd.lower() == 'cancel':
                    self._mission_active = False   # 체인 중단
                    self.cancel_goal()
                    continue

                if cmd.lower() in ('start_mission', 'mission', 'm'):
                    self.start_mission()
                    continue
                
                # 'docking'이나 'undocking'만 입력해도 자동으로 start_를 붙여주는 편의 기능
                if cmd.lower() == 'docking':
                    cmd = 'start_docking'
                elif cmd.lower() == 'undocking':
                    cmd = 'start_undocking'
                elif cmd.lower() == 'approach':
                    cmd = 'start_approach'

                self.send_goal(cmd)
            except EOFError:
                break
            except Exception as e:
                print(f"입력 오류: {e}")

    def send_goal(self, command):
        self._action_client.wait_for_server()
        
        goal_msg = MissionControl.Goal()
        goal_msg.command = command

        self.get_logger().info(f'---------------------------------------------')
        self.get_logger().info(f'📤 [미션 전송] "{command}"')
        self.get_logger().info(f'---------------------------------------------')

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        status = feedback_msg.feedback.current_status
        # 상태가 변경되었을 때만 출력하여 화면 도배 방지
        if getattr(self, '_last_feedback', None) != status:
            print(f"\r   [피드백] {status}{' ' * 20}")
            self._last_feedback = status

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ 목표가 서버에서 거부되었습니다.')
            return

        self._current_goal_handle = goal_handle

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def cancel_goal(self):
        if hasattr(self, '_current_goal_handle') and self._current_goal_handle is not None:
            self.get_logger().info('🛑 명령 취소 요청 중...')
            cancel_future = self._current_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done_callback)
        else:
            print("현재 실행 중인 명령이 없습니다.")

    def cancel_done_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('✅ 명령이 성공적으로 취소되었습니다.')
            self._current_goal_handle = None
        else:
            self.get_logger().info('⚠️ 취소할 수 없는 상태이거나 서버에서 거부되었습니다.')

    def get_result_callback(self, future):
        print() # 줄바꿈
        result = future.result().result
        if result.success:
            self.get_logger().info(f'✅ [명령 완료] 성공! 메시지: {result.message}')
        else:
            self.get_logger().error(f'⚠️ [명령 실패/취소] 에러 메시지: {result.message}')
            
        self._current_goal_handle = None

        # ── 미션 체인 자동 진행 ──────────────────────────────────────
        if self._mission_active:
            if result.success:
                self._mission_idx += 1
                if self._mission_idx < len(MISSION_SEQUENCE):
                    nxt = MISSION_SEQUENCE[self._mission_idx]
                    print(f"\n➡️  [{self._mission_idx+1}/{len(MISSION_SEQUENCE)}] 다음 명령 발행: {nxt}")
                    self.send_goal(nxt)
                else:
                    print("\n🎉 [미션 체인 완료] 스폰~재도킹 전체 파이프라인 종료!")
                    self._mission_active = False
            else:
                print("\n⛔ [미션 체인 중단] 단계 실패/취소로 체인을 멈춥니다.")
                self._mission_active = False
            return
        # ─────────────────────────────────────────────────────────────

        # 백그라운드 스레드에서 완료 로그가 뜬 후, 유저가 다시 칠 수 있음을 시각적으로 보여줌
        print("\n명령을 입력하세요 (종료하려면 q 입력, 취소하려면 c 입력) > ", end='', flush=True)

def main(args=None):
    rclpy.init(args=args)
    action_client = MissionActionClient()
    
    # 별도 스레드에서 터미널 입력 받기 시작
    action_client.start_input_thread()
    
    try:
        # ROS 2 이벤트 루프(콜백 등)는 메인 스레드에서 실행
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()