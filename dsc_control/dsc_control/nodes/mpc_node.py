"""ROS2 node: /cmd_vel_ref + /odom -> VelocityTrackingController -> /cmd_vel (+ /dsc/debug).

[인수인계] Nav2가 낸 속도(/cmd_vel_ref)를 실제 로봇이 그 속도로 움직이도록 가공해 /cmd_vel로 내보내는 ROS 껍데기.
제어 계산 자체는 core/controller.py 에 있고, 이 파일은 "언제 계산하고 언제 멈추는가"만 담당한다.
  - 제어 주기: /odom 콜백 1회 = 제어 1스텝(0.1 s). m2 드라이버가 /odom을 10 Hz로 내므로 별도 타이머를 쓰지 않는다.
    /odom 주기가 10 Hz가 아니면 모델(dt=0.1)이 틀어지므로 드라이버를 바꿀 때 먼저 확인할 것.
  - 기준값 0(도착, collision_monitor 정지, Nav2 종료) -> MPC를 거치지 않고 즉시 0 출력 + 제어기 리셋(stop_on_zero_ref).
  - /cmd_vel_ref 또는 /odom이 timeout_s(0.3 s) 넘게 끊기면 0을 idle_zero_count번 보내고 조용해진다(_go_idle).
  - 종료(Ctrl-C, SIGTERM) 시 주행 중이었으면 0을 5번 보낸다. m2 드라이버는 cmd_vel 타임아웃이 없어 마지막 명령을
    계속 실행하므로 이 정지 처리가 안전상 중요하다. kill -9 / 전원 차단은 막지 못한다.
  - require_gate, /dsc/gate_state, /cmd_vel_dock 은 별도 도킹 패키지(dsc_docking, 이 저장소에 없음)용이다.
    Nav2 주행(nav2_mpc.launch.py)에서는 require_gate=false 이므로 무시해도 된다.

One control step runs per /odom message (the driver publishes /odom at 10 Hz right after it
sends the previous command), so the controller is synchronised to the driver loop.

The node only owns /cmd_vel while /cmd_vel_ref is fresh (i.e. during an approach/docking). When the
reference (or /odom) goes stale it sends idle_zero_count zero commands to stop the robot and then
stays silent. It starts silent and sends a final zero on shutdown only if it was driving, so it does not fight Nav2's controller -> collision_monitor -> /cmd_vel chain while
the robot drives to the staging pose.
With require_gate (set by dsc_docking's docking.launch) it also stays idle unless /dsc/gate_state
is dock or release, so a reference that arrives while Nav2 owns /cmd_vel cannot wind it up.

/dsc/debug layout (Float64MultiArray):
  [0] t_odom  [1:3] ref  [3:5] meas  [5:7] u  [7:9] u_bar  [9:11] y_pred_1
  [11] solve_ms  [12] status_ok (1/0)  [13] mode_id  [14:16] d_hat (ESO estimate, logged in every mode)
"""
import os
from dataclasses import replace

import rclpy
from rclpy.signals import SignalHandlerOptions
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from dsc_control.core.config import load_config
from dsc_control.core.controller import MODE_ID, VelocityTrackingController
from dsc_control.nodes.stop import raise_on_sigterm, send_stop


class MpcNode(Node):
    def __init__(self):
        super().__init__("dsc_mpc_node")
        default_cfg = os.path.join(get_package_share_directory("dsc_control"), "config", "default.yaml")
        self.declare_parameter("config_path", default_cfg)
        self.declare_parameter("mode", "passthrough")
        self.declare_parameter("enabled", True)
        self.declare_parameter("timeout_s", 0.3)
        self.declare_parameter("idle_zero_count", 5)  # zero commands sent after going idle
        # /cmd_vel when driving the robot directly; /cmd_vel_dock behind dsc_docking's cmd_vel_gate
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        # Behind cmd_vel_gate: compute only while the gate forwards our output (dock/release). Otherwise
        # the MPC would assume its dropped commands were applied and wind up (u_prev, ESO d_hat), then
        # start the approach from that wound-up command when the gate switches to dock.
        self.declare_parameter("require_gate", False)
        # A reference of exactly zero (arrival, collision_monitor stop, Nav2 shutdown) goes to the driver
        # at once and resets the MPC. Tracking 0 through the MPC can only be as fast or slower: the input-
        # rate weight spreads the drop over steps and the ESO term can keep pushing. The MPC's part in a
        # precise stop is tracking the deceleration before the 0. (Active braking would be a separate option.)
        self.declare_parameter("stop_on_zero_ref", True)

        # > 0 replaces limits.v_max from the config file. nav2_mpc.launch passes MPPI's vx_max here, so the
        # forward cap is set in one place (collision_monitor is upstream: never go faster than MPPI may).
        self.declare_parameter("v_max", 0.0)

        cfg_path = self.get_parameter("config_path").value
        cfg = load_config(cfg_path)
        v_max = float(self.get_parameter("v_max").value)
        if v_max > 0.0:
            cfg = replace(cfg, limits=replace(cfg.limits, v_max=v_max))
        self.ctrl = VelocityTrackingController(cfg, mode=self.get_parameter("mode").value)
        self.enabled = bool(self.get_parameter("enabled").value)
        self.timeout_s = float(self.get_parameter("timeout_s").value)
        self.idle_zero_count = int(self.get_parameter("idle_zero_count").value)
        # Start silent: the node must never publish /cmd_vel unless it has been driving (Nav2 may own it)
        self.idle_zeros_sent = self.idle_zero_count
        self.active = False
        self.require_gate = bool(self.get_parameter("require_gate").value)
        self.stop_on_zero_ref = bool(self.get_parameter("stop_on_zero_ref").value)
        self.gate_state = None  # unknown counts as not owning /cmd_vel
        self.add_on_set_parameters_callback(self._on_params)

        self.ref = (0.0, 0.0)
        self.last_ref_time = None
        self.last_odom_time = None
        self.pub_cmd = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)
        self.pub_dbg = self.create_publisher(Float64MultiArray, "/dsc/debug", 10)
        self.create_subscription(Twist, "/cmd_vel_ref", self._on_ref, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        if self.require_gate:
            self.create_subscription(String, "/dsc/gate_state", self._on_gate, 10)
        self.create_timer(0.1, self._watchdog)
        self.get_logger().info(f"config={cfg_path} mode={self.ctrl.mode} v_max={cfg.limits.v_max} enabled={self.enabled} "
                               f"require_gate={self.require_gate} stop_on_zero_ref={self.stop_on_zero_ref}")

    def _on_params(self, params):
        for p in params:
            if p.name == "mode":
                if self.active:  # a reset mid-run would drop u_prev to 0 and mix two conditions in one run
                    return SetParametersResult(successful=False, reason="stop the robot before changing mode")
                try:
                    self.ctrl.set_mode(p.value)
                    self.ctrl.reset()
                except ValueError as e:
                    return SetParametersResult(successful=False, reason=str(e))
            elif p.name == "enabled":
                self.enabled = bool(p.value)
                self.ctrl.reset()
            elif p.name == "timeout_s":
                self.timeout_s = float(p.value)
        return SetParametersResult(successful=True)

    def _on_gate(self, msg: String) -> None:
        if msg.data != self.gate_state:
            self.get_logger().info(f"gate: {self.gate_state} -> {msg.data}")
        self.gate_state = msg.data

    def _owns_cmd_vel(self) -> bool:
        return not self.require_gate or self.gate_state in ("dock", "release")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _stale(self, t) -> bool:
        return t is None or self._now() - t > self.timeout_s

    def _publish_cmd(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub_cmd.publish(msg)

    def _on_ref(self, msg: Twist) -> None:
        prev, self.ref = self.ref, (msg.linear.x, msg.angular.z)
        self.last_ref_time = self._now()
        if self._is_stop_ref() and prev != self.ref and self.active and self._owns_cmd_vel():
            self._publish_cmd(0.0, 0.0)  # don't wait for the next /odom

    def _is_stop_ref(self) -> bool:
        return self.stop_on_zero_ref and self.ref == (0.0, 0.0)

    def _on_odom(self, msg: Odometry) -> None:
        # 제어 1스텝. 순서: (1) 계산할 상황인지 판정 -> (2) 정지 기준값이면 즉시 0 -> (3) MPC/ESO 1스텝 -> 발행
        self.last_odom_time = self._now()
        if not self.enabled or self._stale(self.last_ref_time) or not self._owns_cmd_vel():
            self._go_idle()  # resets every cycle, so the MPC starts clean (u_prev = 0) when it gets /cmd_vel
            return
        self.idle_zeros_sent = 0
        self.active = True
        # 측정값은 드라이버 휠 오도메트리의 body-frame 속도 (v [m/s], w [rad/s])
        meas = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)
        if self._is_stop_ref():
            self.ctrl.reset()  # restart from u_prev = 0, matching the stopped command
            self._publish_cmd(0.0, 0.0)
            return
        res = self.ctrl.step(self.ref, meas)
        self._publish_cmd(res.u[0], res.u[1])
        # 분석용 로그(/dsc/debug). 배열 순서는 파일 맨 위 docstring 참고. 순서를 바꾸면 분석 코드도 같이 고칠 것.
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        ok = 0.0 if res.status.startswith("fallback") else 1.0
        dbg = Float64MultiArray()
        dbg.data = [stamp, *self.ref, *meas, *res.u.tolist(), *res.u_bar.tolist(),
                    *res.y_pred_1.tolist(), res.solve_ms, ok, float(MODE_ID[res.mode]),
                    *res.d_hat.tolist()]
        self.pub_dbg.publish(dbg)

    def _go_idle(self) -> None:
        """Stop the robot with a few zero commands, then release /cmd_vel to other publishers."""
        self.ctrl.reset()
        self.active = False
        if self.idle_zeros_sent < self.idle_zero_count:
            self._publish_cmd(0.0, 0.0)
            self.idle_zeros_sent += 1

    def _watchdog(self) -> None:
        if self._stale(self.last_odom_time):
            self._go_idle()


def main(args=None):
    # rclpy's own SIGINT handler would invalidate the context before we can send a final stop.
    # m2_driver_node keeps executing the last /cmd_vel forever, so the zero command matters.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    raise_on_sigterm()
    node = MpcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.active or node.idle_zeros_sent < node.idle_zero_count:  # only if we were driving
            send_stop(lambda: node._publish_cmd(0.0, 0.0))
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
