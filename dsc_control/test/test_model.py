import numpy as np

from dsc_control.core.model import AxisModel, VelocityModel, build_state_space, make_state
from dsc_control.core.sim import VelocityPlant


def _impulse_response(delay):
    model = VelocityModel(v=AxisModel(0.5, 1.0, delay), w=AxisModel(0.5, 1.0, 0))
    plant = VelocityPlant(model)
    ys = [plant.step(np.array([1.0, 0.0]))[0]]
    for _ in range(6):
        ys.append(plant.step(np.zeros(2))[0])
    return ys  # ys[k] is y at step k+1


def test_input_reaches_output_after_delay_plus_one():
    for delay in (0, 1, 3):
        ys = _impulse_response(delay)
        first = next(i for i, y in enumerate(ys) if abs(y) > 0)
        assert first == delay  # y index delay+1 -> list index delay
        assert ys[delay] == 1.0


def test_steady_gain_matches_formula():
    m = AxisModel(a=0.6, b=0.3, delay=2)
    plant = VelocityPlant(VelocityModel(v=m, w=m))
    for _ in range(200):
        y = plant.step(np.array([0.1, 0.0]))
    assert np.isclose(y[0], 0.1 * m.steady_gain(), rtol=1e-6)


def test_make_state_places_y_and_history():
    model = VelocityModel(v=AxisModel(0.5, 1.0, 2), w=AxisModel(0.5, 1.0, 0))
    ss = build_state_space(model)
    x = make_state(ss, np.array([0.3, -0.1]), (np.array([0.2, 0.1]), np.array([])))
    assert x.shape == (4,)
    assert np.allclose(x, [0.3, 0.2, 0.1, -0.1])
