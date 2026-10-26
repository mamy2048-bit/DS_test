import numpy as np

from dsc_control.core.model import AxisModel, VelocityModel, build_state_space, make_state
from dsc_control.core.mpc import LinearMPC
from dsc_control.core.sim import VelocityPlant

MODEL = VelocityModel(v=AxisModel(0.6, 0.4, 3), w=AxisModel(0.5, 0.5, 0))
TRACK = 0.531
G = [[1, TRACK / 2], [1, -TRACK / 2], [-1, TRACK / 2], [-1, -TRACK / 2]]


def _mpc(ss, du_max=(0.02, 0.1), u_max=(0.3, 0.5), wheel=0.5):
    return LinearMPC(ss.A, ss.B, ss.C, horizon=20, Q=[100.0, 10.0], R=[1.0, 1.0], S=[10.0, 1.0],
                     u_min=[-u_max[0], -u_max[1]], u_max=list(u_max), du_max=list(du_max),
                     G=G, h=[wheel] * 4)


def _run(plant, mpc, ss, ref, ubar, steps=120):
    u_prev = np.zeros(2)
    hist = [np.zeros(m.delay) for m in MODEL.axes]
    y = plant.output()
    us, ys = [], []
    for _ in range(steps):
        x0 = make_state(ss, y, tuple(hist))
        sol = mpc.solve(x0, np.tile(ref, (mpc.N, 1)), np.tile(ubar, (mpc.N, 1)), u_prev)
        assert sol.ok, sol.status
        u = sol.u_seq[0]
        for j, m in enumerate(MODEL.axes):
            if m.delay:
                hist[j] = np.concatenate(([u[j]], hist[j][:-1]))
        y = plant.step(u)
        us.append(u.copy()); ys.append(y.copy()); u_prev = u
    return np.array(us), np.array(ys)


def _ubar(ref):
    return np.array([(1 - m.a) * r / m.b for m, r in zip(MODEL.axes, ref)])


def test_tracks_step_without_offset_on_exact_model():
    ss = build_state_space(MODEL)
    ref = np.array([0.05, 0.1])
    _, ys = _run(VelocityPlant(MODEL), _mpc(ss), ss, ref, _ubar(ref))
    assert np.allclose(ys[-1], ref, atol=1e-4)


def test_constant_disturbance_leaves_offset():
    ss = build_state_space(MODEL)
    ref = np.array([0.05, 0.0])
    plant = VelocityPlant(MODEL, disturbance=(-0.004, 0.0))  # e.g. caster drag
    _, ys = _run(plant, _mpc(ss), ss, ref, _ubar(ref))
    assert abs(ys[-1, 0] - ref[0]) > 1e-3  # nominal MPC cannot remove it


def test_constraints_hold():
    ss = build_state_space(MODEL)
    ref = np.array([0.3, 0.5])  # asks for more than wheel limit allows
    us, _ = _run(VelocityPlant(MODEL), _mpc(ss), ss, ref, _ubar(ref), steps=60)
    tol = 1e-4
    assert np.all(np.abs(us[:, 0]) <= 0.3 + tol)
    assert np.all(np.abs(us[:, 1]) <= 0.5 + tol)
    assert np.all(np.abs(us[:, 0]) + TRACK / 2 * np.abs(us[:, 1]) <= 0.5 + tol)
    du = np.diff(np.vstack([np.zeros(2), us]), axis=0)
    assert np.all(np.abs(du[:, 0]) <= 0.02 + tol)
    assert np.all(np.abs(du[:, 1]) <= 0.1 + tol)


def test_disturbance_term_matches_simulation():
    ss = build_state_space(MODEL)
    mpc = LinearMPC(ss.A, ss.B, ss.C, horizon=20, Q=[100.0, 10.0], R=[1.0, 1.0], S=[10.0, 1.0],
                    u_min=[-0.3, -0.5], u_max=[0.3, 0.5], du_max=[0.02, 0.1], G=G, h=[0.5] * 4,
                    Bd=ss.Bd)
    d = np.array([-0.004, 0.01])
    x0 = make_state(ss, np.array([0.02, 0.05]), tuple(np.full(m.delay, 0.03) for m in MODEL.axes))
    sol = mpc.solve(x0, np.tile([0.05, 0.1], (20, 1)), np.tile([0.05, 0.1], (20, 1)), np.zeros(2), d=d)
    assert sol.ok
    x = x0.copy()
    for i in range(20):
        x = ss.A @ x + ss.B @ sol.u_seq[i] + ss.Bd @ d
        assert np.allclose(ss.C @ x, sol.y_pred[i], atol=1e-9)


def test_solve_with_d_requires_bd():
    ss = build_state_space(MODEL)
    import pytest
    with pytest.raises(ValueError):
        _mpc(ss).solve(np.zeros(ss.A.shape[0]), np.zeros((20, 2)), np.zeros((20, 2)), np.zeros(2),
                       d=np.zeros(2))
