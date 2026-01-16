import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

# YOLO 모듈 임포트
from .modules.yolo_wrapper import YoloTRT

ENGINE_FILE_PATH = './models/yolo11s.engine'

class MainPC(Node):
    def __init__(self):
        super().__init__('est_node')
        
        # --- [Parameter] ---
        # 간단한 테스트를 위해 YOLO, POS 기능 활성화 여부만 유지
        self.declare_parameter('enable_yolo', True)
        self.declare_parameter('enable_pos', True)

        self.run_yolo = self.get_parameter('enable_yolo').value
        self.run_pos = self.get_parameter('enable_pos').value
        
        # --- [Camera Intrinsic Parameters (IMX219)] ---
        self.IMG_WIDTH = 1640
        self.IMG_HEIGHT = 1232
        
        # Focal Length 계산 (FOV 77도 기준)
        # f = diagonal_pixels / (2 * tan(FOV/2))
        self.FOCAL_LENGTH = 1321.4
        self.CX = self.IMG_WIDTH / 2.0
        self.CY = self.IMG_HEIGHT / 2.0
        
        # 타겟 실제 크기 (1m x 1m)
        self.TARGET_REAL_SIZE = 1.0 

        # --- [Publisher & Subscriber] ---
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
            
        # [수정] 카메라 기준 좌표를 발행 (토픽 이름 변경: cam_pose)
        self.cam_pose_publisher = self.create_publisher(PointStamped, '/rescue/target_pose_cam', 10)
        
        self.bridge = CvBridge()
        self.yolo_model = None

        if self.run_yolo:
            self.get_logger().info("Loading YOLO Engine...")
            self.yolo_model = YoloTRT(ENGINE_FILE_PATH)
        
        self.get_logger().info("Test Node Ready: Publishing Camera Coordinates Only")

    def image_callback(self, msg):
        try:
            full_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            detections = []

            # 1. YOLO 추론
            if self.run_yolo:
                detections = self.yolo_model.inference(full_image) 

            # 2. 위치 추정 (Camera Frame)
            if self.run_pos and self.run_yolo:
                if not detections:
                    return

                # 가장 신뢰도 높은 객체 선정
                target_obj = None
                max_conf = 0.0
                for obj in detections:
                    if obj['conf'] > 0.5:
                        if obj['conf'] > max_conf:
                            max_conf = obj['conf']
                            target_obj = obj
                
                if target_obj:
                    x, y, w, h = target_obj['bbox']
                    cx, cy = int(x + w/2), int(y + h/2)

                    # --- [Distance Estimation] ---
                    # Pinhole Camera Model
                    # 픽셀 크기 (가로/세로 중 큰 값 사용)
                    pixel_size = max(w, h)
                    
                    if pixel_size > 0:
                        # Z: 깊이 (카메라 렌즈에서 물체까지의 거리)
                        z_cam = (self.FOCAL_LENGTH * self.TARGET_REAL_SIZE) / pixel_size
                        
                        # X: 영상의 좌우 (오른쪽이 +)
                        x_cam = (cx - self.CX) * z_cam / self.FOCAL_LENGTH
                        
                        # Y: 영상의 상하 (아래쪽이 +)
                        y_cam = (cy - self.CY) * z_cam / self.FOCAL_LENGTH
                        
                        # --- [Publish] ---
                        pos_msg = PointStamped()
                        pos_msg.header = msg.header
                        
                        # 중요: 프레임 ID를 카메라 광학 프레임으로 설정
                        pos_msg.header.frame_id = "camera_optical_frame" 
                        
                        pos_msg.point.x = float(x_cam)
                        pos_msg.point.y = float(y_cam)
                        pos_msg.point.z = float(z_cam)
                        
                        self.cam_pose_publisher.publish(pos_msg)
                        
                        # 로그 출력: 카메라 기준 좌표 (드론 기울기/위치 무시)
                        self.get_logger().info(
                            f"Cam Pos -> X: {x_cam:.2f}m, Y: {y_cam:.2f}m, Depth(Z): {z_cam:.2f}m"
                        )

        except Exception as e:
            self.get_logger().error(f"Processing Error: {e}")
    
    def destroy_resources(self):
        self.get_logger().info("Cleaning up resources...")
        if self.yolo_model:
            self.yolo_model = None

def main(args=None):
    rclpy.init(args=args)
    node = MainPC()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_resources()
        node.destroy_node()
        rclpy.shutdown()