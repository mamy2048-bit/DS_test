import os
from dataclasses import replace

import numpy as np
import pytest

from dsc_control.core.config import Limits, load_config
from dsc_control.core.model import AxisModel, VelocityModel
from dsc_control.core.target import limit_command, steady_state_input

CFG = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")
LIM = Limits(v_max=0.3, w_max=0.5, wheel_v_max=0.5, dv_max=0.02, dw_max=0.1, track_width=0.531)


def test_load_default_config():
    cfg = load_config(CFG)
    assert cfg.model.dt == 0.1
    for axis in cfg.model.axes:
        assert 0.0 <= axis.a < 1.0 and axis.b > 0.0 and axis.delay >= 0
    assert cfg.mpc.horizon == 20
    assert cfg.limits.track_width == pytest.approx(0.531)


def test_steady_state_input_formula():
    model = VelocityModel(v=AxisModel(0.6, 0.4, 3), w=AxisModel(0.5, 0.5, 0))
    ubar, sat = steady_state_input(model, np.array([0.05, 0.1]), LIM)
    assert np.allclose(ubar, [0.05 * 0.4 / 0.4, 0.1 * 0.5 / 0.5])
    assert not sat


def test_limit_command_scales_for_wheel_limit():
    lim = Limits(v_max=0.3, w_max=0.5, wheel_v_max=0.4, dv_max=0.02, dw_max=0.1, track_width=0.531)
    u = limit_command(np.array([0.3, 0.5]), lim)  # 0.3 + 0.2655 * 0.5 = 0.433 > 0.4
    assert abs(u[0]) + 0.5 * lim.track_width * abs(u[1]) == pytest.approx(0.4)
    assert u[0] / u[1] == pytest.approx(0.3 / 0.5)


def test_steady_state_input_subtracts_disturbance():
    model = VelocityModel(v=AxisModel(0.6, 0.4, 3), w=AxisModel(0.5, 0.5, 0))
    ubar, _ = steady_state_input(model, np.array([0.05, 0.1]), LIM, d=np.array([0.004, -0.01]))
    assert np.allclose(ubar, [(0.4 * 0.05 - 0.004) / 0.4, (0.5 * 0.1 + 0.01) / 0.5])


def test_default_config_has_observer_bandwidth():
    cfg = load_config(CFG)
    assert len(cfg.observer.bandwidth) == 2 and all(w > 0 for w in cfg.observer.bandwidth)


def test_limit_command_reverse_limit():
    lim = replace(LIM, v_min=-0.1)
    assert limit_command(np.array([-0.3, 0.0]), lim)[0] == pytest.approx(-0.1)
    assert limit_command(np.array([0.3, 0.0]), lim)[0] == pytest.approx(0.3)
    assert limit_command(np.array([-0.3, 0.0]), LIM)[0] == pytest.approx(-0.3)  # unset: symmetric


def test_reverse_limit_must_not_be_positive():
    with pytest.raises(ValueError):
        replace(LIM, v_min=0.1)
    lim = replace(LIM, v_min=-0.4)  # reverse may exceed the forward cap (forward lowered by the v_max override)
    assert limit_command(np.array([-0.5, 0.0]), lim)[0] == pytest.approx(-0.4)
    assert limit_command(np.array([0.5, 0.0]), lim)[0] == pytest.approx(0.3)


def test_load_nav_config():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "nav.yaml"))
    assert cfg.limits.v_lo == pytest.approx(-0.35)  # MPPI vx_min
    assert cfg.observer.max_correction is not None and len(cfg.observer.max_correction) == 2
    assert all(m.delay == 1 for m in cfg.model.axes)
