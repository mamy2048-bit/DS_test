"""Nav2 driving experiment: MPPI's command goes through MPC + ESO before reaching the driver. No docking, no gate.

[인수인계] Nav2 + MPC/ESO 주행의 진입점. 이 launch 하나가 Nav2와 dsc_mpc_node를 함께 띄운다(Nav2를 따로 켜지 말 것).
  1) DS_navigation(isr_navigation)의 nav2_launch.py를 그대로 include 한다. params.yaml 원본은 건드리지 않고
     RewrittenYaml로 아래 값만 실행 시점에 바꾼다.
       - collision_monitor.cmd_vel_out_topic: cmd_vel -> /cmd_vel_ref   (Nav2 최종 출력을 MPC 노드 입력으로 돌림)
       - goal checker xy/yaw_goal_tolerance, MPPI GoalAngleCritic.threshold_to_consider, MPPI vx_max
  2) mpc.launch.py로 dsc_mpc_node를 띄운다: /cmd_vel_ref + /odom -> MPC/ESO -> /cmd_vel (드라이버).
     따라서 /cmd_vel 발행자는 dsc_mpc_node 하나뿐이어야 한다(수동조종 manual mode, 다른 컨트롤러와 동시 실행 금지).
  3) vx_max 인자 하나가 MPPI 전진 상한과 MPC 노드 v_max를 동시에 정한다. 두 값이 다르면 MPC가 MPPI 명령을 자르거나,
     collision_monitor가 검사하지 않은 속도를 낼 수 있다.
  MPC 설정 파일은 config/nav.yaml(Nav2 주행용으로 식별한 모델)이다. mpc.launch.py 단독 실행 시 기본값은 default.yaml
  (저속 도킹용 옛 모델)이므로 Nav2 아래에서는 반드시 이 launch를 쓰거나 config_path:=.../nav.yaml을 준다.

  MPPI -> /cmd_vel_nav -> velocity_smoother -> collision_monitor -> /cmd_vel_ref -> dsc_mpc_node -> /cmd_vel

  ros2 launch dsc_control nav2_mpc.launch.py mode:=offset_free     # passthrough | nominal | offset_free
mode:=passthrough is the MPPI-only baseline with the same topic path (the MPC node only clips to limits).

xy_goal_tolerance (default 0.08 m) narrows Nav2's arrival test: the goal checker stops the robot the
moment it is inside the tolerance, so a wide tolerance decides the final error by itself. The robot
still coasts a little after that test passes; 0.08 is a setting to drive with, not a guarantee of 80 mm.

MPPI's GoalAngleCritic (align to the goal yaw) normally takes over 0.5 m out. If it straightens the
heading there while the robot is still off to the side, a differential drive cannot fix that lateral
offset afterwards. goal_angle_threshold (default 0.2 m) lets MPPI keep steering at the goal point and
turn to the goal yaw only at the end; the stateful goal checker keeps the xy result once inside the
tolerance, so that final turn cannot lose it.

vx_max (default 0.35 m/s; params.yaml 0.5) is the forward cruise speed. It rewrites MPPI's vx_max and is
passed to the MPC node as v_max, so both stay equal. velocity_smoother keeps max_velocity 0.5: it only clips,
so it never binds below MPPI's cap (and Humble's RewrittenYaml cannot rewrite one element of that list).
Reverse (MPPI vx_min -0.35) and rotation (wz_max 1.0) are unchanged.

The source params file is not modified: only collision_monitor's `cmd_vel_out_topic`, the goal
checker tolerances, GoalAngleCritic's threshold_to_consider and the forward speed are rewritten here.
Do not run this together with another Nav2 launch or controller that also drives /cmd_vel.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    isr = FindPackageShare("isr_navigation")
    control = FindPackageShare("dsc_control")
    params = RewrittenYaml(source_file=LaunchConfiguration("params_file"),
                           param_rewrites={"cmd_vel_out_topic": "/cmd_vel_ref",
                                           "xy_goal_tolerance": LaunchConfiguration("xy_goal_tolerance"),
                                           "yaw_goal_tolerance": LaunchConfiguration("yaw_goal_tolerance"),
                                           # full path: threshold_to_consider also exists in other critics
                                           "controller_server.ros__parameters.FollowPath.GoalAngleCritic."
                                           "threshold_to_consider": LaunchConfiguration("goal_angle_threshold"),
                                           "controller_server.ros__parameters.FollowPath.vx_max":
                                               LaunchConfiguration("vx_max")},
                           convert_types=True)
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=PathJoinSubstitution([isr, "config", "params.yaml"])),
        DeclareLaunchArgument("map", default_value=PathJoinSubstitution([isr, "map_2D", "map.yaml"])),
        DeclareLaunchArgument("mode", default_value="passthrough"),
        DeclareLaunchArgument("xy_goal_tolerance", default_value="0.08",
                              description="Nav2 arrival radius [m] (params.yaml: 0.10)"),
        DeclareLaunchArgument("yaw_goal_tolerance", default_value="0.10",
                              description="Nav2 arrival heading tolerance [rad] (0.10 rad = 5.7 deg)"),
        DeclareLaunchArgument("goal_angle_threshold", default_value="0.2",
                              description="MPPI GoalAngleCritic threshold_to_consider [m] (params.yaml: 0.5)"),
        DeclareLaunchArgument("vx_max", default_value="0.35",
                              description="forward speed cap [m/s] for MPPI (FollowPath.vx_max) and the MPC (params.yaml: 0.5)"),
        DeclareLaunchArgument("mpc_config", default_value=PathJoinSubstitution([control, "config", "nav.yaml"])),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([isr, "launch", "nav2_launch.py"])),
            launch_arguments={"params_file": params, "map": LaunchConfiguration("map")}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([control, "launch", "mpc.launch.py"])),
            launch_arguments={"mode": LaunchConfiguration("mode"),
                              "config_path": LaunchConfiguration("mpc_config"),
                              "cmd_vel_topic": "/cmd_vel",
                              "require_gate": "false",
                              "v_max": LaunchConfiguration("vx_max")}.items()),
    ])
