import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

# TF 관련
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

from .modules.yolo_wrapper import YoloTRT

ENGINE_FILE_PATH = './models/best_half.engine'

# --- [Kalman Filter Class] ---
class StaticKalmanFilter:
    def __init__(self):
        # 1. 상태 벡터: [x, y, z] (3x1) - 속도 제거
        self.x = np.zeros((3, 1))
        
        # 2. 공분산 행렬 (P)
        self.P = np.eye(3) * 1.0
        
        # 3. 전이 행렬 (F): 위치는 변하지 않는다고 가정 (x_new = x_old)
        self.F = np.eye(3)
        
        # 4. 측정 행렬 (H): 상태 변수 그대로 관측함
        self.H = np.eye(3)
        
        # 5. 노이즈 행렬
        # Q (Process Noise): 실제 객체가 움직일 가능성. 
        # 고정된 물체라면 아주 작게(1e-4) 잡아서 "절대 안 움직임"을 표현
        self.Q = np.eye(3) * 0.0001  
        
        # R (Measurement Noise): 센서(YOLO+거리계산)의 오차. 
        # 값이 클수록 관측값보다 기존 추정값을 더 믿음
        self.R = np.eye(3) * 0.1 

        self.initialized = False

    def process(self, measurement):
        # measurement shape: (3, 1) -> [[x], [y], [z]]
        
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x.flatten()

        # --- Predict (예측) ---
        # 위치 고정 모델이므로 x = Fx 에서 F가 단위행렬이라 값 변화 없음
        # 하지만 공분산 P는 Q만큼 증가 (시간이 지날수록 불확실성 약간 증가)
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # --- Update (보정) ---
        z = measurement
        y = z - np.dot(self.H, self.x) # 잔차(Residual)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # 칼만 이득
        
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

        return self.x.flatten()
    
class MainPC_GlobalKF(Node):
    def __init__(self):
        super().__init__('pc_rescue_node_kf')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.declare_parameter('enable_yolo', True)
        self.declare_parameter('enable_pos', True)
        self.declare_parameter('target_frame', 'odom')
        
        self.declare_parameter('camera_offset_x', 0.1)
        self.declare_parameter('camera_offset_y', 0.0)
        self.declare_parameter('camera_offset_z', -0.05)

        self.run_yolo = self.get_parameter('enable_yolo').value
        self.run_pos = self.get_parameter('enable_pos').value
        self.target_frame = self.get_parameter('target_frame').value
        
        self.offset_x = self.get_parameter('camera_offset_x').value
        self.offset_y = self.get_parameter('camera_offset_y').value
        self.offset_z = self.get_parameter('camera_offset_z').value

        self.ANGLE_FAR = 30.0   
        self.ANGLE_CLOSE = 70.0 
        self.current_tilt_deg = self.ANGLE_FAR 
        self.update_trig_values(self.current_tilt_deg)

        self.IMG_WIDTH = 1640
        self.IMG_HEIGHT = 1232
        self.FOCAL_LENGTH = 1321.4
        
        self.subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        
        # [KF] 필터링된 좌표 발행
        self.kf_pose_publisher = self.create_publisher(PointStamped, '/rescue/target_pose_global_kf', 10)
        self.gimbal_cmd_publisher = self.create_publisher(Bool, '/camera/gimbal_cmd', 10)
        
        self.bridge = CvBridge()
        self.yolo_model = None
        self.kf = StaticKalmanFilter() # 칼만 필터 인스턴스 생성
        
        self.CX = self.IMG_WIDTH / 2.0
        self.CY = self.IMG_HEIGHT / 2.0
        self.TARGET_REAL_SIZE = 1.0 

        if self.run_yolo:
            self.get_logger().info("Loading YOLO Engine...")
            self.yolo_model = YoloTRT(ENGINE_FILE_PATH)
            
        self.publish_gimbal_cmd(self.current_tilt_deg)
        self.get_logger().info(f"Kalman Filter Node Ready! Target Frame: {self.target_frame}")

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
        target_angle = None
        if abs(self.current_tilt_deg - self.ANGLE_FAR) < 1.0:
            if cy > self.IMG_HEIGHT * 0.8: target_angle = self.ANGLE_CLOSE
        elif abs(self.current_tilt_deg - self.ANGLE_CLOSE) < 1.0:
            if cy < self.IMG_HEIGHT * 0.2: target_angle = self.ANGLE_FAR

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

                    pixel_size = max(w, h)
                    if pixel_size > 0:
                        z_cam = (self.FOCAL_LENGTH * self.TARGET_REAL_SIZE) / pixel_size
                        x_cam = (cx - self.CX) * z_cam / self.FOCAL_LENGTH
                        y_cam = (cy - self.CY) * z_cam / self.FOCAL_LENGTH
                        
                        x_body = (z_cam * self.cos_tilt) - (y_cam * self.sin_tilt) + self.offset_x
                        y_body = -x_cam + self.offset_y
                        z_body = -(z_cam * self.sin_tilt) - (y_cam * self.cos_tilt) + self.offset_z

                        point_body = PointStamped()
                        point_body.header = msg.header
                        point_body.header.frame_id = "base_link"
                        point_body.point.x, point_body.point.y, point_body.point.z = float(x_body), float(y_body), float(z_body)

                        # --- [TF & Kalman Filter] ---
                        try:
                            image_time = rclpy.time.Time.from_msg(msg.header.stamp)
                            # 1. 절대 좌표로 변환 (Measurement)
                            transform = self.tf_buffer.lookup_transform(
                                self.target_frame, "base_link", image_time, timeout=rclpy.duration.Duration(seconds=0.1))
                            point_global = do_transform_point(point_body, transform)
                            
                            # 2. 칼만 필터 업데이트
                            measurement = np.array([[point_global.point.x], 
                                                    [point_global.point.y], 
                                                    [point_global.point.z]])
                            
                            # 현재 시간 (초 단위)
                            filtered_pos = self.kf.process(measurement)
                            
                            # 3. 결과 발행
                            kf_msg = PointStamped()
                            kf_msg.header.stamp = msg.header.stamp
                            kf_msg.header.frame_id = self.target_frame
                            kf_msg.point.x = filtered_pos[0]
                            kf_msg.point.y = filtered_pos[1]
                            kf_msg.point.z = filtered_pos[2]
                            
                            self.kf_pose_publisher.publish(kf_msg)

                        except Exception as tf_e:
                            self.get_logger().warn(f"TF/KF Error: {tf_e}")

        except Exception as e:
            self.get_logger().error(f"Processing Error: {e}")
    
    def destroy_resources(self):
        if self.yolo_model: self.yolo_model = None

def main(args=None):
    rclpy.init(args=args)
    node = MainPC_GlobalKF()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_resources()
        node.destroy_node()
        rclpy.shutdown()