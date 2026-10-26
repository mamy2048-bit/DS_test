"""ROS2 node for desk testing without the robot: /cmd_vel -> VelocityPlant -> /odom at 10 Hz.

Publishes the plant velocity as twist and its integral (x, y, yaw) as pose and as the TF
odom -> base_link, like m2_driver.

Mimics m2_driver_node timing (apply latest command, then publish odom). Not a research simulator.
Parameters: config_path (model used as the 'true' plant), dist_v, dist_w (constant disturbance).
"""
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from dsc_control.core.config import load_config
from dsc_control.core.sim import VelocityPlant


class SimRobot(Node):
    def __init__(self):
        super().__init__("dsc_sim_robot")
        default_cfg = os.path.join(get_package_share_directory("dsc_control"), "config", "default.yaml")
        self.declare_parameter("config_path", default_cfg)
        self.declare_parameter("dist_v", 0.0)
        self.declare_parameter("dist_w", 0.0)
        cfg = load_config(self.get_parameter("config_path").value)
        self.plant = VelocityPlant(cfg.model, disturbance=(self.get_parameter("dist_v").value,
                                                           self.get_parameter("dist_w").value))
        self.cmd = np.zeros(2)
        self.dt = cfg.model.dt
        self.x = self.y = self.yaw = 0.0
        self.pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        self.create_timer(cfg.model.dt, self._tick)

    def _on_cmd(self, msg: Twist) -> None:
        self.cmd = np.array([msg.linear.x, msg.angular.z])

    def _tick(self) -> None:
        y = self.plant.step(self.cmd)
        self.x += y[0] * np.cos(self.yaw) * self.dt
        self.y += y[0] * np.sin(self.yaw) * self.dt
        self.yaw += y[1] * self.dt
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(self.x)
        msg.pose.pose.position.y = float(self.y)
        msg.pose.pose.orientation.z = float(np.sin(self.yaw / 2))
        msg.pose.pose.orientation.w = float(np.cos(self.yaw / 2))
        msg.twist.twist.linear.x = float(y[0])
        msg.twist.twist.angular.z = float(y[1])
        self.pub.publish(msg)
        tf = TransformStamped()
        tf.header = msg.header
        tf.child_frame_id = msg.child_frame_id
        tf.transform.translation.x, tf.transform.translation.y = msg.pose.pose.position.x, msg.pose.pose.position.y
        tf.transform.rotation = msg.pose.pose.orientation
        self.tf.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = SimRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
