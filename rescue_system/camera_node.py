import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class JetsonCameraNode(Node):
    def __init__(self):
        super().__init__('jetson_camera_node')
        
        # 퍼블리셔 생성 (토픽 이름: /image_raw)
        self.publisher_ = self.create_publisher(Image, '/image_raw', 10)
        self.timer = self.create_timer(1.0/30.0, self.timer_callback) # 30 FPS
        self.bridge = CvBridge()

        # Jetson Orin NX용 GStreamer 파이프라인 (매우 중요)
        # ISP를 통과시켜 RG10 -> BGR로 변환
        self.pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1640, height=1232, format=NV12, framerate=30/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1"
        )

        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            self.get_logger().error("GStreamer pipeline을 열 수 없습니다. 케이블 연결을 확인하세요.")
            exit()
        else:
            self.get_logger().info("Jetson Camera Node가 시작되었습니다! (/image_raw)")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # OpenCV 이미지를 ROS 메시지로 변환하여 발행
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_optical_frame"
            self.publisher_.publish(msg)
        else:
            self.get_logger().warn("프레임을 읽을 수 없습니다.")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = JetsonCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()