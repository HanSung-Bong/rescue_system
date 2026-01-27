import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import math

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

from .modules.yolo_wrapper import YoloTRT

ENGINE_FILE_PATH = './models/best_half_amrl.engine'


class MainPC_GlobalSimple(Node):
    def __init__(self):
        super().__init__('pc_rescue_node_global')

        # ---------------- TF ----------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------- Parameters ----------------
        self.declare_parameter('enable_yolo', True)
        self.declare_parameter('enable_pos', True)
        self.declare_parameter('target_frame', 'odom')

        self.declare_parameter('camera_offset_x', 0.0)
        self.declare_parameter('camera_offset_y', 0.0)
        self.declare_parameter('camera_offset_z', 0.0)

        self.run_yolo = self.get_parameter('enable_yolo').value
        self.run_pos = self.get_parameter('enable_pos').value
        self.target_frame = self.get_parameter('target_frame').value

        self.offset_x = self.get_parameter('camera_offset_x').value
        self.offset_y = self.get_parameter('camera_offset_y').value
        self.offset_z = self.get_parameter('camera_offset_z').value

        # ---------------- Gimbal ----------------
        self.ANGLE_FAR = 70.0
        self.ANGLE_CLOSE = 30.0
        self.current_tilt_deg = self.ANGLE_CLOSE  # 초기 상태 30도

        # ---------------- Camera spec ----------------
        self.IMG_WIDTH = 1640
        self.IMG_HEIGHT = 1232
        self.FOCAL_LENGTH = 1321.4

        self.CX = self.IMG_WIDTH / 2.0
        self.CY = self.IMG_HEIGHT / 2.0
        self.TARGET_REAL_SIZE = 1.0

        # ---------------- ROS ----------------
        self.bridge = CvBridge()

        # [수정됨] 콜백 함수는 들어오는대로 연결하되, 내부에서 처리 여부를 결정합니다.
        self.sub_cam30 = self.create_subscription(
            Image, '/camera/cam30/image', self.image_callback_30, 10)

        self.sub_cam70 = self.create_subscription(
            Image, '/camera/cam70/image', self.image_callback_70, 10)

        self.global_pose_publisher = self.create_publisher(
            PointStamped, '/rescue/target_pose_global', 10)

        self.gimbal_cmd_publisher = self.create_publisher(
            Bool, '/camera/gimbal_cmd', 10)

        # ---------------- YOLO ----------------
        self.yolo_model = None
        if self.run_yolo:
            self.get_logger().info("Loading YOLO Engine...")
            self.yolo_model = YoloTRT(ENGINE_FILE_PATH)

        self.publish_gimbal_cmd(self.current_tilt_deg)
        self.get_logger().info("Global Simple Node Ready")

    # =========================================================
    # [수정됨] Camera callbacks
    # 각 콜백이 process_image를 호출할 때, "이 이미지는 몇 도 카메라의 것인지"를 명시합니다.
    # =========================================================
    def image_callback_30(self, msg):
        # 30도 이미지는 source_angle=30으로 전달
        self.process_image(msg, source_angle=self.ANGLE_CLOSE)

    def image_callback_70(self, msg):
        # 70도 이미지는 source_angle=70으로 전달
        self.process_image(msg, source_angle=self.ANGLE_FAR)

    # =========================================================
    # Main processing
    # =========================================================
    def process_image(self, msg, source_angle):
            try:
                # 1. 카메라 소스 일치 여부 확인 (Gatekeeper)
                if abs(self.current_tilt_deg - source_angle) > 1.0:
                    return

                # TF 변환 준비
                tilt_used = source_angle
                rad = math.radians(tilt_used)
                sin_tilt = math.sin(rad)
                cos_tilt = math.cos(rad)

                # YOLO 추론
                image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                detections = []
                if self.run_yolo:
                    detections = self.yolo_model.inference(image)

                if not detections:
                    return

                target = max(detections, key=lambda x: x['conf'])
                if target['conf'] < 0.5:
                    return

                x, y, w, h = target['bbox']
                cx, cy = int(x + w / 2), int(y + h / 2)

                # 2. 짐벌 전환 로직 (먼저 수행해야 추적이 끊기지 않음)
                self.check_safety_and_switch(cy)

                # [추가된 부분] -----------------------------------------------------
                # 화면 중앙 영역에 있을 때만 Publish 허용 (가장자리 노이즈 방지)
                # margin_ratio = 0.2 (상하좌우 20% 영역 제외)
                margin = 0.2 
                x_min = self.IMG_WIDTH * margin
                x_max = self.IMG_WIDTH * (1.0 - margin)
                y_min = self.IMG_HEIGHT * margin
                y_max = self.IMG_HEIGHT * (1.0 - margin)

                # cx, cy가 중앙 박스를 벗어나면 여기서 리턴 (좌표 발행 X)
                if not (x_min < cx < x_max and y_min < cy < y_max):
                    return
                # ------------------------------------------------------------------

                pixel_size = max(w, h)
                if pixel_size <= 0:
                    return

                # 3. 좌표 계산 및 TF 변환
                z_cam = (self.FOCAL_LENGTH * self.TARGET_REAL_SIZE) / pixel_size
                x_cam = (cx - self.CX) * z_cam / self.FOCAL_LENGTH
                y_cam = (cy - self.CY) * z_cam / self.FOCAL_LENGTH

                x_body = (z_cam * cos_tilt) - (y_cam * sin_tilt) + self.offset_x
                y_body = x_cam + self.offset_y
                z_body = (z_cam * sin_tilt) + (y_cam * cos_tilt) + self.offset_z

                point_body = PointStamped()
                point_body.header = msg.header
                point_body.header.frame_id = "base_link"
                point_body.point.x = float(x_body)
                point_body.point.y = float(y_body)
                point_body.point.z = float(z_body)

                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    "base_link",
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )

                point_global = do_transform_point(point_body, transform)
                
                # 4. 최종 발행
                self.global_pose_publisher.publish(point_global)

            except Exception as e:
                self.get_logger().error(f"Processing Error: {e}")

    # =========================================================
    # Gimbal logic (다음 프레임부터 적용)
    # =========================================================
    def check_safety_and_switch(self, cy):
        target_angle = None
        print(cy)

        # 현재 70도(FAR) 상태라면 -> 30도(CLOSE)로 전환할지 검사
        if abs(self.current_tilt_deg - self.ANGLE_FAR) < 1.0:
            if cy < self.IMG_HEIGHT * 0.2: # 너무 아래로 내려가면 가까운 곳(30도)으로 전환
                target_angle = self.ANGLE_CLOSE

        # 현재 30도(CLOSE) 상태라면 -> 70도(FAR)로 전환할지 검사
        elif abs(self.current_tilt_deg - self.ANGLE_CLOSE) < 1.0:
            if cy > self.IMG_HEIGHT * 0.8: # 예: 너무 위로 올라가면 먼 곳(70도)으로 전환
                target_angle = self.ANGLE_FAR

        if target_angle is not None:
            self.current_tilt_deg = target_angle
            self.publish_gimbal_cmd(target_angle)

    def publish_gimbal_cmd(self, angle):
        msg = Bool()
        msg.data = abs(angle - self.ANGLE_FAR) < 1.0
        self.gimbal_cmd_publisher.publish(msg)

    # =========================================================
    def destroy_resources(self):
        self.yolo_model = None


def main(args=None):
    rclpy.init(args=args)
    node = MainPC_GlobalSimple()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_resources()
        node.destroy_node()
        rclpy.shutdown()