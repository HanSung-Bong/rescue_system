import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class PX4TFBroadcaster(Node):
    def __init__(self):
        super().__init__('px4_tf_broadcaster')

        # [중요] QoS 설정 (PX4와 통신하기 위한 필수 설정)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 토픽 구독
        self.subscription = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.listener_callback,
            qos_profile
        )
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info("PX4 TF Broadcaster Started... Waiting for Odometry data...")

    def listener_callback(self, msg):
        t = TransformStamped()

        # --- [핵심 해결책: 시간 동기화] ---
        # 시스템 시간(get_clock)을 쓰지 않고, PX4 메시지의 시간을 그대로 사용합니다.
        # PX4 msg.timestamp는 마이크로초(us) 단위입니다.
        # 이를 ROS Time(초, 나노초)으로 변환합니다.
        
        # Gazebo 환경에서는 PX4 시간과 ROS Sim Time이 거의 일치합니다.
        timestamp_sec = int(msg.timestamp / 1000000)
        timestamp_nanosec = int((msg.timestamp % 1000000) * 1000)

        t.header.stamp.sec = timestamp_sec
        t.header.stamp.nanosec = timestamp_nanosec
        
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        # 1. Position (NED -> ENU)
        # PX4 (X:North, Y:East, Z:Down) -> ROS (X:East, Y:North, Z:Up)
        # NED -> NED (변환 없이 그대로 매핑)
        t.transform.translation.x = float(msg.position[0]) # North -> X
        t.transform.translation.y = float(msg.position[1]) # East -> Y
        t.transform.translation.z = float(msg.position[2]) # Down -> Z

        # Rotation (NED 그대로)
        t.transform.rotation.x = float(msg.q[1])
        t.transform.rotation.y = float(msg.q[2])
        t.transform.rotation.z = float(msg.q[3])
        t.transform.rotation.w = float(msg.q[0])

        # TF 발행
        self.tf_broadcaster.sendTransform(t)
        
        # [디버깅용] 데이터가 잘 나가는지 1초에 한 번 정도만 로그 출력 (선택 사항)
        # self.get_logger().info(f"Published TF at time: {timestamp_sec}.{timestamp_nanosec}", throttle_duration_sec=1.0)

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