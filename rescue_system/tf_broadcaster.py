import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np

class PX4TFBroadcaster(Node):
    def __init__(self):
        super().__init__('px4_tf_broadcaster')

        # --- [QoS 설정: 매우 중요] ---
        # PX4는 Best Effort로 보내므로, 받는 쪽도 Best Effort여야 데이터가 들어옵니다.
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # PX4 Odometry 구독 (토픽 이름 확인 필요: /fmu/out/vehicle_odometry)
        self.subscription = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.listener_callback,
            qos_profile
        )

        # TF 방송기 생성
        self.tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info("PX4 TF Broadcaster Started (NED -> ENU)")

    def listener_callback(self, msg):
        t = TransformStamped()

        # 1. 헤더 설정
        # timestamp는 PX4 시간을 ROS 시간으로 변환해서 써야 하지만, 
        # 간단하게 현재 ROS 시간으로 동기화하거나 msg.timestamp를 변환해야 함.
        # 여기서는 시스템의 현재 시간을 사용하여 TF 트리의 끊김을 방지합니다.
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'       # 부모 프레임 (절대 좌표)
        t.child_frame_id = 'base_link'   # 자식 프레임 (드론 몸체)

        # 2. 좌표 변환 (PX4 NED -> ROS ENU)
        # Position:
        # ROS X (East)  = PX4 Y (East)
        # ROS Y (North) = PX4 X (North)
        # ROS Z (Up)    = -PX4 Z (Down)
        
        t.transform.translation.x = float(msg.position[1]) 
        t.transform.translation.y = float(msg.position[0])
        t.transform.translation.z = -float(msg.position[2])

        # Rotation (Quaternion):
        # 쿼터니언 변환도 필요합니다. (NED -> ENU)
        # q_ned = [w, x, y, z] (PX4 순서 주의: 보통 [w, x, y, z] 혹은 [x, y, z, w])
        # px4_msgs의 VehicleOdometry q는 float32[4] q (usually w, x, y, z format in PX4 internal, but check msg definition)
        # PX4 1.14 기준 q는 [w, x, y, z] 순서입니다.
        
        # 간단한 변환 공식:
        # x_enu = y_ned
        # y_enu = x_ned
        # z_enu = -z_ned
        # w_enu = w_ned
        
        t.transform.rotation.x = float(msg.q[2])
        t.transform.rotation.y = float(msg.q[1])
        t.transform.rotation.z = -float(msg.q[3])
        t.transform.rotation.w = float(msg.q[0])

        # 3. 방송 (Broadcast)
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = PX4TFBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()