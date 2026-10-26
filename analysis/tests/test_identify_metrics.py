import numpy as np
import pandas as pd

from analysis.bag_io import align_to_odom
from analysis.identify import fit_axis, simulate_axis
from analysis.metrics import integrated_error, stop_residuals, steady_errors


def test_fit_recovers_known_axis():
    rng = np.random.default_rng(1)
    u = np.repeat(rng.uniform(-0.1, 0.1, 40), 15)  # piecewise-constant excitation
    y = simulate_axis(0.7, 0.3, 3, u, 0.0) + rng.normal(0, 1e-4, len(u))
    best = fit_axis(u, y, max_delay=6)[0]
    assert best.delay == 3
    assert abs(best.a - 0.7) < 0.02 and abs(best.b - 0.3) < 0.02


def test_align_to_odom_zero_order_hold():
    cmd = pd.DataFrame({"t": [0.0, 1.0], "v": [0.1, 0.2], "w": [0.0, 0.5]})
    odom = pd.DataFrame({"t": [0.5, 1.5, -1.0], "t_recv": [0.5, 1.5, -1.0], "v": [0, 0, 0], "w": [0, 0, 0]})
    df = align_to_odom(cmd, odom)
    assert list(df["u_v"]) == [0.1, 0.2, 0.0]
    assert list(df["u_w"]) == [0.0, 0.5, 0.0]


def test_stop_residual_and_integrated_error():
    t = np.arange(0, 3.0, 0.1)
    ref = np.where(t < 1.0, 0.1, 0.0)
    y = np.where(t < 1.0, 0.1, np.where(t < 1.3, 0.05, 0.0))  # coasts after the stop command
    res = stop_residuals(t, ref, y)
    assert res.shape == (1,)
    assert 0.01 < res[0] < 0.03
    assert integrated_error(t, ref, y) < 0  # robot went further than commanded


def test_steady_errors_per_reference():
    ref = np.r_[np.full(30, 0.02), np.zeros(5), np.full(30, 0.05)]
    y = ref * 1.03
    se = steady_errors(ref, y)
    assert set(se) == {0.02, 0.05}
    assert np.isclose(se[0.02], -0.0006) and np.isclose(se[0.05], -0.0015)
