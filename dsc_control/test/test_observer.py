import numpy as np
import pytest

from dsc_control.core.model import AxisModel, VelocityModel, build_state_space, make_state
from dsc_control.core.observer import DisturbanceObserver, eso_gains
from dsc_control.core.sim import VelocityPlant

MODEL = VelocityModel(v=AxisModel(0.6, 0.4, 3), w=AxisModel(0.5, 0.5, 0))


def test_eso_gains_place_both_poles_at_p():
    a, wo, dt = 0.757, 5.0, 0.1
    l_y, l_d = eso_gains(a, wo, dt)
    # error dynamics of (y, d): [[a - l_y, 1], [-l_d, 1]]
    poles = np.linalg.eigvals(np.array([[a - l_y, 1.0], [-l_d, 1.0]]))
    assert np.allclose(poles, np.exp(-wo * dt), atol=1e-6)


@pytest.mark.parametrize("model", [MODEL, VelocityModel(v=AxisModel(0.757, 0.242, 0),
                                                        w=AxisModel(0.558, 0.429, 0))],
                         ids=["delayed", "no_delay"])
def test_estimates_constant_disturbance_open_loop(model):
    d_true = np.array([-0.004, 0.01])
    plant = VelocityPlant(model, disturbance=d_true)
    ss = build_state_space(model)
    obs = DisturbanceObserver(model, (5.0, 5.0))
    hist = [np.zeros(m.delay) for m in model.axes]
    u = np.array([0.05, 0.1])
    y = plant.output()
    for _ in range(80):
        d_hat = obs.update(y)
        x_hat = make_state(ss, obs.y_hat, tuple(hist))
        obs.propagate(ss.C @ (ss.A @ x_hat + ss.B @ u))
        for j, m in enumerate(model.axes):
            if m.delay:
                hist[j] = np.concatenate(([u[j]], hist[j][:-1]))
        y = plant.step(u)
    assert np.allclose(d_hat, d_true, atol=1e-6)


def test_reset_clears_estimate():
    obs = DisturbanceObserver(MODEL, (5.0, 5.0))
    obs.update([0.0, 0.0]); obs.propagate([0.0, 0.0]); obs.update([0.1, 0.1])
    assert np.any(obs.d_hat != 0)
    obs.reset()
    assert obs.y_hat is None and np.all(obs.d_hat == 0)
