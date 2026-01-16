import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')
        
        # 1. Publisher 생성
        # 토픽 이름: /camera/image_raw (Subscriber와 맞춰야 함)
        # 큐 사이즈: 10
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        
        # 2. 타이머 설정 (30 FPS 목표 -> 0.033초 주기)
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        # 3. OpenCV 설정 (웹캠 열기)
        self.cap = cv2.VideoCapture(0) # 0번이 보통 노트북 내장 웹캠
        self.bridge = CvBridge()

        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다! (/dev/video0 확인 필요)")

    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if ret:
            # 4. OpenCV(BGR) -> ROS Msg 변환
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_frame"
            
            # 5. 메시지 발행
            self.publisher_.publish(msg)
            # self.get_logger().info('Publishing video frame') # 로그 너무 많이 뜨면 주석 처리

def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 자원 해제
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()