"""Launch the MPC node. Example:
  ros2 launch dsc_control mpc.launch.py mode:=nominal
  ros2 launch dsc_control mpc.launch.py mode:=passthrough sim:=true   # desk test without robot
  ros2 launch dsc_control mpc.launch.py mode:=offset_free sim:=true dist_v:=-0.01   # sim with disturbance
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_cfg = PathJoinSubstitution([FindPackageShare("dsc_control"), "config", "default.yaml"])
    mode = LaunchConfiguration("mode")
    cfg = LaunchConfiguration("config_path")
    sim = LaunchConfiguration("sim")
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="passthrough"),
        DeclareLaunchArgument("config_path", default_value=default_cfg),
        DeclareLaunchArgument("sim", default_value="false"),
        DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel",
                              description="MPC output; /cmd_vel_dock when dsc_docking's cmd_vel_gate owns /cmd_vel"),
        DeclareLaunchArgument("require_gate", default_value="false",
                              description="true behind cmd_vel_gate: compute only while /dsc/gate_state is dock/release"),
        DeclareLaunchArgument("stop_on_zero_ref", default_value="true",
                              description="a zero reference goes straight to the driver (no MPC tracking of 0)"),
        DeclareLaunchArgument("v_max", default_value="0.0",
                              description="> 0 overrides limits.v_max of config_path [m/s]"),
        DeclareLaunchArgument("dist_v", default_value="0.0"),  # sim_robot only: constant disturbance
        DeclareLaunchArgument("dist_w", default_value="0.0"),
        Node(package="dsc_control", executable="mpc_node", name="dsc_mpc_node", output="screen",
             parameters=[{"mode": mode, "config_path": cfg,
                          "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                          "require_gate": ParameterValue(LaunchConfiguration("require_gate"), value_type=bool),
                          "stop_on_zero_ref": ParameterValue(LaunchConfiguration("stop_on_zero_ref"),
                                                             value_type=bool),
                          "v_max": ParameterValue(LaunchConfiguration("v_max"), value_type=float)}]),
        Node(package="dsc_control", executable="sim_robot", name="dsc_sim_robot", output="screen",
             parameters=[{"config_path": cfg,
                          "dist_v": ParameterValue(LaunchConfiguration("dist_v"), value_type=float),
                          "dist_w": ParameterValue(LaunchConfiguration("dist_w"), value_type=float)}],
             condition=IfCondition(sim)),
    ])
