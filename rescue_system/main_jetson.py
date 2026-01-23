import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

# TF 관련 라이브러리 추가
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

from .modules.yolo_wrapper import YoloTRT

ENGINE_FILE_PATH = './models/best_half_amrl.engine'

class MainPC_GlobalSimple(Node):
    def __init__(self):
        super().__init__('pc_rescue_node_global')
        
        # --- [TF Buffer 설정] ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # --- [Parameter] ---
        self.declare_parameter('enable_yolo', True)
        self.declare_parameter('enable_pos', True)
        self.declare_parameter('target_frame', 'odom') # 변환할 절대 좌표계 이름 (map 또는 odom)
        
        # 기존 파라미터들...
        self.declare_parameter('camera_offset_x', 0.0)
        self.declare_parameter('camera_offset_y', 0.0)
        self.declare_parameter('camera_offset_z', 0.0)

        self.run_yolo = self.get_parameter('enable_yolo').value
        self.run_pos = self.get_parameter('enable_pos').value
        self.target_frame = self.get_parameter('target_frame').value
        
        self.offset_x = self.get_parameter('camera_offset_x').value
        self.offset_y = self.get_parameter('camera_offset_y').value
        self.offset_z = self.get_parameter('camera_offset_z').value

        # --- [각도 설정] ---
        self.ANGLE_FAR = 30.0   
        self.ANGLE_CLOSE = 70.0 
        self.current_tilt_deg = self.ANGLE_FAR 
        self.update_trig_values(self.current_tilt_deg)

        # 카메라 스펙 (IMX219)
        self.IMG_WIDTH = 1640
        self.IMG_HEIGHT = 1232
        self.FOCAL_LENGTH = 1321.4
        self.V_FOV_HALF_RAD = math.atan((self.IMG_HEIGHT / 2.0) / self.FOCAL_LENGTH)

        self.subscription = self.create_subscription(
            Image, '/camera/cam70/image', self.image_callback, 10)
        
        # [수정] 토픽 이름 변경 (Global)
        self.global_pose_publisher = self.create_publisher(PointStamped, '/rescue/target_pose_global', 10)
        self.gimbal_cmd_publisher = self.create_publisher(Bool, '/camera/gimbal_cmd', 10)
        
        self.bridge = CvBridge()
        self.yolo_model = None
        
        self.CX = self.IMG_WIDTH / 2.0
        self.CY = self.IMG_HEIGHT / 2.0
        self.TARGET_REAL_SIZE = 1.0 

        if self.run_yolo:
            self.get_logger().info("Loading YOLO Engine...")
            self.yolo_model = YoloTRT(ENGINE_FILE_PATH)
        
        self.publish_gimbal_cmd(self.current_tilt_deg)
        self.get_logger().info(f"Simple Global Node Ready! Target Frame: {self.target_frame}")

    def update_trig_values(self, degree):
        rad = math.radians(degree)
        self.sin_tilt = math.sin(rad)
        self.cos_tilt = math.cos(rad)

    def publish_gimbal_cmd(self, angle):
        msg = Bool()
        if abs(angle - self.ANGLE_CLOSE) < 1.0:
            msg.data = True
        else:
            msg.data = False
        self.gimbal_cmd_publisher.publish(msg)

    def check_safety_and_switch(self, cy):
        dy = cy - self.CY
        #obj_pixel_angle_deg = math.degrees(math.atan2(dy, self.FOCAL_LENGTH))
        target_angle = None
        
        if abs(self.current_tilt_deg - self.ANGLE_FAR) < 1.0:
            if cy > self.IMG_HEIGHT * 0.8: 
                target_angle = self.ANGLE_CLOSE
        elif abs(self.current_tilt_deg - self.ANGLE_CLOSE) < 1.0:
            if cy < self.IMG_HEIGHT * 0.2: 
                target_angle = self.ANGLE_FAR

        if target_angle is not None:
            self.current_tilt_deg = target_angle
            self.update_trig_values(self.current_tilt_deg)
            self.publish_gimbal_cmd(self.current_tilt_deg)

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
                    if obj['conf'] > 0.5 and obj['conf'] > max_conf:
                        max_conf = obj['conf']
                        target_obj = obj
                
                if target_obj:
                    x, y, w, h = target_obj['bbox']
                    cx, cy = int(x + w/2), int(y + h/2)

                    self.check_safety_and_switch(cy)

                    # 1. Body Frame 좌표 계산
                    pixel_size = max(w, h)
                    if pixel_size > 0:
                        z_cam = (self.FOCAL_LENGTH * self.TARGET_REAL_SIZE) / pixel_size
                        x_cam = (cx - self.CX) * z_cam / self.FOCAL_LENGTH
                        y_cam = (cy - self.CY) * z_cam / self.FOCAL_LENGTH
                        
                        x_body = (z_cam * self.cos_tilt) - (y_cam * self.sin_tilt) + self.offset_x
                        y_body = -x_cam + self.offset_y
                        z_body = -(z_cam * self.sin_tilt) - (y_cam * self.cos_tilt) + self.offset_z

                        # 2. PointStamped 생성 (base_link 기준)
                        point_body = PointStamped()
                        point_body.header = msg.header
                        point_body.header.frame_id = "base_link"
                        point_body.point.x = float(x_body)
                        point_body.point.y = float(y_body)
                        point_body.point.z = float(z_body)

                        self.get_logger().info(
                            f"Cam Pos -> X: {x_cam:.2f}m, Y: {y_cam:.2f}m, Depth(Z): {z_cam:.2f}m"
                        )

                        # 3. TF 변환 (base_link -> map)
                        try:
                            image_time = rclpy.time.Time.from_msg(msg.header.stamp)
                            transform = self.tf_buffer.lookup_transform(
                                self.target_frame, 
                                "base_link",             
                                rclpy.time.Time(),
                                timeout=rclpy.duration.Duration(seconds=0.1)
                            )
                            
                            point_global = do_transform_point(point_body, transform)
                            
                            # Global 좌표 발행
                            self.global_pose_publisher.publish(point_global)

                        except Exception as tf_e:
                            self.get_logger().warn(f"TF Error: {tf_e}")

        except Exception as e:
            self.get_logger().error(f"Processing Error: {e}")
    
    def destroy_resources(self):
        if self.yolo_model: self.yolo_model = None

def main(args=None):
    rclpy.init(args=args)
    node = MainPC_GlobalSimple()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_resources()
        node.destroy_node()
        rclpy.shutdown()