"""Steady-state input target: ubar = ((1 - a) * r - d) / b per axis (d = 0 for the nominal model).

[인수인계] u_bar = "모델상 이 명령을 계속 주면 속도가 ref에 머문다"는 입력. limit_command는 최종 출력의 마지막 안전장치로,
축별 한계(v_min..v_max, |w|<=w_max)를 먼저 자르고 바퀴 속도 한계를 넘으면 v, w를 같은 비율로 줄여 곡률을 유지한다.
"""
from typing import Optional, Tuple

import numpy as np

from dsc_control.core.config import Limits
from dsc_control.core.model import VelocityModel


def limit_command(u: np.ndarray, lim: Limits) -> np.ndarray:
    """Clip to per-axis limits, then scale both axes down if the wheel-speed limit is exceeded."""
    v = float(np.clip(u[0], lim.v_lo, lim.v_max))
    w = float(np.clip(u[1], -lim.w_max, lim.w_max))
    wheel = abs(v) + 0.5 * lim.track_width * abs(w)
    if wheel > lim.wheel_v_max:
        scale = lim.wheel_v_max / wheel
        v, w = v * scale, w * scale
    return np.array([v, w])


def steady_state_input(model: VelocityModel, ref: np.ndarray, lim: Limits,
                       d: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
    d = np.zeros(2) if d is None else np.asarray(d, dtype=float)
    raw = np.array([((1.0 - m.a) * r - dj) / m.b for m, r, dj in zip(model.axes, ref, d)])
    ubar = limit_command(raw, lim)
    return ubar, not np.allclose(ubar, raw)
