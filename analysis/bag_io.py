"""Read ROS2 bags (no ROS install needed) into pandas DataFrames, using the `rosbags` library."""
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

_TS = get_typestore(Stores.ROS2_HUMBLE)


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _row(msgtype: str, t_recv: float, msg) -> dict:
    if msgtype == "geometry_msgs/msg/Twist":
        return {"t": t_recv, "v": msg.linear.x, "w": msg.angular.z}
    if msgtype == "nav_msgs/msg/Odometry":
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return {"t": _stamp(msg), "t_recv": t_recv, "x": p.x, "y": p.y, "yaw": yaw,
                "v": msg.twist.twist.linear.x, "w": msg.twist.twist.angular.z}
    if msgtype == "sensor_msgs/msg/Imu":
        return {"t": _stamp(msg), "t_recv": t_recv, "wz": msg.angular_velocity.z}
    if msgtype == "std_msgs/msg/Float64MultiArray":
        return {"t": t_recv, **{f"c{i}": x for i, x in enumerate(msg.data)}}
    raise ValueError(f"unsupported message type {msgtype}")


def load_topics(bag: str, topics: Iterable[str]) -> Dict[str, pd.DataFrame]:
    """Return {topic: DataFrame}. Twist rows use receive time; stamped messages use header time."""
    topics = list(topics)
    rows = {t: [] for t in topics}
    with AnyReader([Path(bag)], default_typestore=_TS) as reader:
        conns = [c for c in reader.connections if c.topic in rows]
        for conn, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            rows[conn.topic].append(_row(conn.msgtype, t_ns * 1e-9, msg))
    return {t: pd.DataFrame(r) for t, r in rows.items()}


def align_to_odom(cmd: pd.DataFrame, odom: pd.DataFrame) -> pd.DataFrame:
    """Sample the command (zero-order hold, by receive time) at each odom receive time.

    Returns columns: t, u_v, u_w, y_v, y_w. Rows before the first command get u = 0.
    """
    t = odom["t_recv"].to_numpy()
    idx = np.searchsorted(cmd["t"].to_numpy(), t, side="right") - 1
    valid = idx >= 0
    idx = np.clip(idx, 0, None)
    u_v = np.where(valid, cmd["v"].to_numpy()[idx], 0.0)
    u_w = np.where(valid, cmd["w"].to_numpy()[idx], 0.0)
    return pd.DataFrame({"t": odom["t"].to_numpy(), "u_v": u_v, "u_w": u_w,
                         "y_v": odom["v"].to_numpy(), "y_w": odom["w"].to_numpy()})
