import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from enum import Enum

class MissionState(Enum):
    UNDOCKING = 1
    RECEIVE_COORDINATES = 2
    DEPARTURE = 3
    TARGET_ACQUISITION = 4
    RENDEZVOUS = 5
    DEBRIS_CAPTURE = 6
    DEBRIS_DISPOSAL = 7
    FINAL_UNDOCKING = 8
    FINISHED = 9

class MissionController(Node):
    def __init__(self):
        super().__init__('mission_controller')
        
        # /chaser_command 토픽으로 명령을 퍼블리시
        self.publisher_ = self.create_publisher(String, '/chaser_command', 10)
        
        self.state = MissionState.UNDOCKING
        
        # 간단한 구현을 위해 5초 단위로 다음 상태로 넘어가는 타이머 사용
        self.timer = self.create_timer(5.0, self.timer_callback)
        self.get_logger().info("Mission Controller 시작됨. (업데이트된 8단계 파이프라인)")

    def timer_callback(self):
        msg = String()
        
        if self.state == MissionState.UNDOCKING:
            self.get_logger().info("[State 1/8] 언도킹: 우주 정거장에서 로봇 분리")
            msg.data = "start_undocking"
            self.publisher_.publish(msg)
            self.state = MissionState.RECEIVE_COORDINATES
            
        elif self.state == MissionState.RECEIVE_COORDINATES:
            self.get_logger().info("[State 2/8] 좌표 수신: 월드 좌표계 기준 타겟 쓰레기의 예상 위치 수신 대기")
            msg.data = "receive_target_coords" 
            self.publisher_.publish(msg)
            self.state = MissionState.DEPARTURE
            
        elif self.state == MissionState.DEPARTURE:
            self.get_logger().info("[State 3/8] 출발: 수신된 좌표를 향해 이동 시작")
            msg.data = "depart"
            self.publisher_.publish(msg)
            self.state = MissionState.TARGET_ACQUISITION
            
        elif self.state == MissionState.TARGET_ACQUISITION:
            self.get_logger().info("[State 4/8] 목표물 포착: 비전(YOLO/PnP)을 이용하여 쓰레기 실제 궤도 포착")
            msg.data = "acquire_target"
            self.publisher_.publish(msg)
            self.state = MissionState.RENDEZVOUS
            
        elif self.state == MissionState.RENDEZVOUS:
            self.get_logger().info("[State 5/8] 랑데뷰: 목표물에 접근하여 상대 속도 및 위치 동기화")
            msg.data = "start_approach" 
            self.publisher_.publish(msg)
            self.state = MissionState.DEBRIS_CAPTURE
            
        elif self.state == MissionState.DEBRIS_CAPTURE:
            self.get_logger().info("[State 6/8] 쓰레기 잡기: 로봇 팔을 이용한 쓰레기 포획")
            msg.data = "capture_debris"
            self.publisher_.publish(msg)
            self.state = MissionState.DEBRIS_DISPOSAL
            
        elif self.state == MissionState.DEBRIS_DISPOSAL:
            self.get_logger().info("[State 7/8] 쓰레기 처리: (임시 대기) 쓰레기 처리 시퀀스 수행")
            msg.data = "dispose_debris"
            self.publisher_.publish(msg)
            self.state = MissionState.FINAL_UNDOCKING
            
        elif self.state == MissionState.FINAL_UNDOCKING:
            self.get_logger().info("[State 8/8] 언도킹: 작업 완료 후 쓰레기/장치 등과 최종 분리 (언도킹)")
            msg.data = "final_undock" 
            self.publisher_.publish(msg)
            self.state = MissionState.FINISHED
            
        elif self.state == MissionState.FINISHED:
            self.get_logger().info("모든 미션 파이프라인 시퀀스 완료.")
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = MissionController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
