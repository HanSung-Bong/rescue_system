import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys

# 모듈 임포트
from .modules.yolo_wrapper import YoloTRT
from .modules.vpi_wrapper import VPIStereoDepth

class MainPC(Node):
    def __init__(self):
        super().__init__('pc_rescue_node')
        
        # --- [Parameter 설정] --
        #사용법 ex. ros2 run rescue_system main_pc --ros-args -p enable_depth:=False -p enable_pos:=False
        self.declare_parameter('enable_yolo', True)
        self.declare_parameter('enable_depth', True)
        self.declare_parameter('enable_pos', True) 

        # 파라미터 값 읽기
        self.run_yolo = self.get_parameter('enable_yolo').value
        self.run_depth = self.get_parameter('enable_depth').value
        self.run_pos = self.get_parameter('enable_pos').value

        # 상태 로그 출력
        self.get_logger().info(f"Pipeline Config -> YOLO: {self.run_yolo}, Depth: {self.run_depth}, Pos: {self.run_pos}")

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.pos_publisher = self.create_publisher(PointStamped, '/rescue/target_pose', 10)
        self.bridge = CvBridge()
        
        # --- [Conditional Model Loading] ---
        self.yolo_model = None
        self.vpi_stereo = None

        if self.run_yolo:
            self.get_logger().info("Loading YOLO Engine...")
            self.yolo_model = YoloTRT("./models/yolo11n.engine")
        
        if self.run_depth:
            self.get_logger().info("Loading VPI Context...")
            # 예: 1280x720 SBS -> 한쪽 640x720
            self.vpi_stereo = VPIStereoDepth(640, 720) 
        
        self.get_logger().info("All Systems Ready!")

    def image_callback(self, msg):
        try:
            full_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # [Step 1] 전처리 (항상 실행)
            
            height, width, _ = full_image.shape
            half_width = width // 2
            left_img = full_image[:, :half_width]
            right_img = full_image[:, half_width:]
            
            detections = []
            depth_map = None

            # [Step 2] VPI Depth (옵션)
            if self.run_depth:
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
                depth_map = self.vpi_stereo.estimate(left_gray, right_gray)

            # [Step 3] YOLO (옵션)
            if self.run_yolo:
                detections = self.yolo_model.inference(left_img) 
                print(detections)

            # [Step 4] Position Estimation 
            if self.run_pos and self.run_yolo and self.run_depth:
                if not detections or depth_map is None:
                    return

                for obj in detections:
                    if obj['conf'] < 0.5: continue 
                    
                    x, y, w, h = obj['bbox']
                    cx, cy = int(x + w/2), int(y + h/2)
                    
                    if 0 <= cx < half_width and 0 <= cy < height:
                        # ROI Median Depth
                        roi = depth_map[max(0, cy-2):min(height, cy+3), 
                                        max(0, cx-2):min(half_width, cx+3)]
                        
                        if roi.size > 0:
                            z_dist = np.median(roi)
                            
                            if 0.2 < z_dist < 10.0:
                                fx = self.vpi_stereo.focal_length 
                                px = self.vpi_stereo.width / 2 
                                py = self.vpi_stereo.height / 2
                                
                                real_x = (cx - px) * z_dist / fx
                                real_y = (cy - py) * z_dist / fx
                                real_z = z_dist

                                #################추후 드론 좌표계로 변환 필요!################
                                
                                pos_msg = PointStamped()
                                pos_msg.header = msg.header
                                pos_msg.point.x = float(real_x)
                                pos_msg.point.y = float(real_y)
                                pos_msg.point.z = float(real_z)
                                self.pos_publisher.publish(pos_msg)
                                
                                self.get_logger().info(f"Target: ({real_x:.2f}, {real_y:.2f}, {real_z:.2f})")

        except Exception as e:
            self.get_logger().error(f"Processing Error: {e}")
    def destroy_resources(self):
        self.get_logger().info("Cleaning up resources...")
        
        # YOLO 메모리 해제
        if self.yolo_model:
            self.yolo_model = None 
            
        # VPI 자원 해제
        if self.vpi_stereo:
            self.vpi_stereo = None

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