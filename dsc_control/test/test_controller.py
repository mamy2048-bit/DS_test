import os
from dataclasses import replace

import numpy as np
import pytest

from dsc_control.core.config import load_config
from dsc_control.core.controller import VelocityTrackingController
from dsc_control.core.model import AxisModel
from dsc_control.core.sim import VelocityPlant

CFG = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml"))
# Same limits/weights, but with input delay on both axes to exercise the delay chain.
CFG_DELAY = replace(CFG, model=replace(CFG.model, v=AxisModel(0.6, 0.4, 3), w=AxisModel(0.5, 0.5, 1)))


def _closed_loop(ctrl, plant, ref, steps=100):
    y = plant.output()
    out = []
    for _ in range(steps):
        res = ctrl.step(ref, y)
        y = plant.step(res.u)
        out.append((res, y.copy()))
    return out


def test_passthrough_outputs_limited_reference():
    ctrl = VelocityTrackingController(CFG, mode="passthrough")
    res = ctrl.step([0.05, 0.1], [0.0, 0.0])
    assert np.allclose(res.u, [0.05, 0.1])
    assert res.status == "passthrough"


@pytest.mark.parametrize("cfg", [CFG, CFG_DELAY], ids=["default", "delayed"])
def test_nominal_tracks_on_matching_plant(cfg):
    ctrl = VelocityTrackingController(cfg, mode="nominal")
    out = _closed_loop(ctrl, VelocityPlant(cfg.model), np.array([0.05, 0.1]))
    assert np.allclose(out[-1][1], [0.05, 0.1], atol=1e-4)
    assert all(r.solve_ms < 50 for r, _ in out)


@pytest.mark.parametrize("cfg", [CFG, CFG_DELAY], ids=["default", "delayed"])
def test_y_pred_1_matches_plant_when_model_exact(cfg):
    ctrl = VelocityTrackingController(cfg, mode="nominal")
    plant = VelocityPlant(cfg.model)
    y = plant.output()
    for _ in range(30):
        res = ctrl.step([0.05, 0.1], y)
        y = plant.step(res.u)
        assert np.allclose(res.y_pred_1, y, atol=1e-9)


@pytest.mark.parametrize("cfg", [CFG, CFG_DELAY], ids=["default", "delayed"])
def test_offset_free_removes_offset_that_nominal_leaves(cfg):
    ref = np.array([0.05, 0.1])
    d_true = (-0.004, 0.01)  # constant disturbance the model does not know about
    final = {}
    for mode in ("nominal", "offset_free"):
        ctrl = VelocityTrackingController(cfg, mode=mode)
        out = _closed_loop(ctrl, VelocityPlant(cfg.model, disturbance=d_true), ref, steps=150)
        final[mode] = out[-1][1]
        d_hat = out[-1][0].d_hat
        assert np.allclose(d_hat, d_true, atol=1e-4)  # observer runs (and is logged) in both modes
    assert np.max(np.abs(final["nominal"] - ref)) > 1e-3
    assert np.allclose(final["offset_free"], ref, atol=1e-5)


def test_offset_free_removes_offset_from_gain_mismatch():
    ref = np.array([0.05, 0.1])
    true_model = replace(CFG.model, v=AxisModel(CFG.model.v.a, CFG.model.v.b * 0.9, 0))
    ctrl = VelocityTrackingController(CFG, mode="offset_free")
    out = _closed_loop(ctrl, VelocityPlant(true_model), ref, steps=150)
    assert np.allclose(out[-1][1], ref, atol=1e-5)


def test_passthrough_ignores_observer():
    ctrl = VelocityTrackingController(CFG, mode="passthrough")
    out = _closed_loop(ctrl, VelocityPlant(CFG.model, disturbance=(-0.004, 0.0)),
                       np.array([0.05, 0.0]), steps=50)
    assert np.allclose(out[-1][0].u, [0.05, 0.0])


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        VelocityTrackingController(CFG, mode="bogus")


def test_mpc_respects_reverse_limit():
    cfg = replace(CFG, limits=replace(CFG.limits, v_min=-0.05))
    ctrl = VelocityTrackingController(cfg, mode="nominal")
    for _ in range(20):
        res = ctrl.step([-0.3, 0.0], [0.0, 0.0])
        assert res.u[0] >= -0.05 - 1e-6


def test_eso_correction_capped_for_control_only():
    cap = 0.02
    cfg = replace(CFG, observer=replace(CFG.observer, max_correction=(cap, cap)))
    ctrl = VelocityTrackingController(cfg, mode="offset_free")
    for _ in range(60):  # robot "keeps moving" at 0.03 against a zero reference: ESO winds up
        res = ctrl.step([0.0, 0.0], [0.03, 0.0])
    b = cfg.model.v.b
    assert abs(res.d_hat[0] / b) > cap           # the observer keeps (and logs) its full estimate
    assert abs(res.u_bar[0]) <= cap + 1e-9       # only the capped part reaches the target
