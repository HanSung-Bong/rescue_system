import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool  # [수정] Float32 대신 Bool 사용
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

from .modules.yolo_wrapper import YoloTRT

ENGINE_FILE_PATH = './models/yolo11s.engine'

class MainPC(Node):
    def __init__(self):
        super().__init__('pc_rescue_node')
        
        # --- [Parameter] ---
        self.declare_parameter('enable_yolo', True)
        self.declare_parameter('enable_pos', True)
        self.declare_parameter('camera_offset_x', 0.1)
        self.declare_parameter('camera_offset_y', 0.0)
        self.declare_parameter('camera_offset_z', -0.05)

        self.run_yolo = self.get_parameter('enable_yolo').value
        self.run_pos = self.get_parameter('enable_pos').value
        
        self.offset_x = self.get_parameter('camera_offset_x').value
        self.offset_y = self.get_parameter('camera_offset_y').value
        self.offset_z = self.get_parameter('camera_offset_z').value

        # --- [각도 설정] ---
        self.ANGLE_FAR = 30.0   # False
        self.ANGLE_CLOSE = 70.0 # True
        
        self.current_tilt_deg = self.ANGLE_FAR 
        self.update_trig_values(self.current_tilt_deg)

        # 카메라 스펙 (IMX219)
        self.IMG_WIDTH = 1640
        self.IMG_HEIGHT = 1232
        self.FOCAL_LENGTH = 1321.4
        self.V_FOV_HALF_RAD = math.atan((self.IMG_HEIGHT / 2.0) / self.FOCAL_LENGTH)

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.drone_pose_publisher = self.create_publisher(PointStamped, '/rescue/target_pose_drone', 10)
        
        # [수정] 서보 명령 발행기 (Bool 타입)
        self.gimbal_cmd_publisher = self.create_publisher(Bool, '/camera/gimbal_cmd', 10)
        
        self.bridge = CvBridge()
        self.yolo_model = None
        
        self.CX = self.IMG_WIDTH / 2.0
        self.CY = self.IMG_HEIGHT / 2.0
        self.TARGET_REAL_SIZE = 1.0 

        if self.run_yolo:
            self.get_logger().info("Loading YOLO Engine...")
            self.yolo_model = YoloTRT(ENGINE_FILE_PATH)
        
        # 초기 명령 전송
        self.publish_gimbal_cmd(self.current_tilt_deg)
        self.get_logger().info(f"System Ready! Mode: {self.ANGLE_FAR}(False)/{self.ANGLE_CLOSE}(True)")

    def update_trig_values(self, degree):
        rad = math.radians(degree)
        self.sin_tilt = math.sin(rad)
        self.cos_tilt = math.cos(rad)

    def publish_gimbal_cmd(self, angle):
        """
        [수정됨] 각도(float)를 받아서 Bool 명령으로 변환 후 발행
        30도 근처 -> False
        70도 근처 -> True
        """
        msg = Bool()
        # 앵글이 CLOSE(70도)에 가까우면 True, 아니면 False
        if abs(angle - self.ANGLE_CLOSE) < 1.0:
            msg.data = True
        else:
            msg.data = False
            
        self.gimbal_cmd_publisher.publish(msg)

    def check_safety_and_switch(self, cy):
        """
        예측 투영 알고리즘 (Projection Check)
        """
        dy = cy - self.CY
        obj_pixel_angle_deg = math.degrees(math.atan2(dy, self.FOCAL_LENGTH))
        abs_obj_angle_deg = self.current_tilt_deg + obj_pixel_angle_deg

        target_angle = None
        
        # 30도(False) -> 70도(True) 전환 시도
        if abs(self.current_tilt_deg - self.ANGLE_FAR) < 1.0:
            if cy > self.IMG_HEIGHT * 0.8: # 화면 하단 80%
                target_angle = self.ANGLE_CLOSE

        # 70도(True) -> 30도(False) 전환 시도
        elif abs(self.current_tilt_deg - self.ANGLE_CLOSE) < 1.0:
            if cy < self.IMG_HEIGHT * 0.2: # 화면 상단 20%
                target_angle = self.ANGLE_FAR

        self.current_tilt_deg = target_angle
        self.update_trig_values(self.current_tilt_deg)
        self.publish_gimbal_cmd(self.current_tilt_deg)

        # if target_angle is not None:
        #     # 시뮬레이션: 바꾸면 화면 어디에 뜰까?
        #     predicted_pixel_angle_deg = abs_obj_angle_deg - target_angle
        #     predicted_pixel_angle_rad = math.radians(predicted_pixel_angle_deg)

        #     # 안전 마진 (화각의 85% 안쪽)
        #     safety_margin_rad = self.V_FOV_HALF_RAD * 0.85

        #     if abs(predicted_pixel_angle_rad) < safety_margin_rad:
        #         self.get_logger().warn(f"Safe Switch! New Gimbal: {target_angle}° ({'True' if target_angle==70 else 'False'})")

    def image_callback(self, msg):
        try:
            full_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            detections = []
            if self.run_yolo:
                detections = self.yolo_model.inference(full_image)

            if self.run_pos and self.run_yolo:
                if not detections: return

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

                    # [Smart Switching]
                    self.check_safety_and_switch(cy)

                    # [Coordinate Transform]
                    pixel_size = max(w, h)
                    if pixel_size > 0:
                        z_cam = (self.FOCAL_LENGTH * self.TARGET_REAL_SIZE) / pixel_size
                        x_cam = (cx - self.CX) * z_cam / self.FOCAL_LENGTH
                        y_cam = (cy - self.CY) * z_cam / self.FOCAL_LENGTH
                        
                        x_body = (z_cam * self.cos_tilt) - (y_cam * self.sin_tilt) + self.offset_x
                        y_body = -x_cam + self.offset_y
                        z_body = -(z_cam * self.sin_tilt) - (y_cam * self.cos_tilt) + self.offset_z

                        pos_msg = PointStamped()
                        pos_msg.header = msg.header
                        pos_msg.header.frame_id = "base_link" 
                        pos_msg.point.x, pos_msg.point.y, pos_msg.point.z = float(x_body), float(y_body), float(z_body)
                        self.drone_pose_publisher.publish(pos_msg)

        except Exception as e:
            self.get_logger().error(f"Processing Error: {e}")
    
    def destroy_resources(self):
        if self.yolo_model: self.yolo_model = None

def main(args=None):
    rclpy.init(args=args)
    node = MainPC()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_resources()
        node.destroy_node()
        rclpy.shutdown()