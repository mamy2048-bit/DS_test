"""Extended state observer (ESO) for offset-free MPC: one constant disturbance per axis.

[인수인계] 모델과 실제 로봇의 차이(마찰, 경사, 배터리 전압, 모델 오차)를 축마다 "일정 외란 d" 하나로 뭉쳐 추정한다.
  - 조정값은 축별 대역폭 w_o(nav.yaml observer.bandwidth) 하나뿐. 크면 빨리 따라가지만 노이즈와 진동까지 외란으로
    오인해 되먹인다. 모델이 정확하지 않을수록 낮게 둔다.
  - d_hat 단위는 "속도/스텝"이다. 명령 단위로 보려면 d_hat/b (controller._d_for_control, max_correction 참고).
  - 호출 순서: update(측정) 로 d_hat을 얻어 제어에 쓰고, 명령 u가 정해지면 propagate(...) 로 다음 스텝 예측을 만든다.

Augmented axis model (see model.py):
    y[k+1] = a * y[k] + b * u[k - delay] + d[k],    d[k+1] = d[k]
The delay chain holds past inputs that are known exactly, so only (y, d) are estimated.

Predictor-form observer, run once per control step:
    e        = y_meas[k] - y_hat[k]                       (innovation)
    d_hat'   = d_hat + l_d * e                            -> used by target and MPC at step k
    y_hat'   = y_nom[k+1] + d_hat + l_y * e               (y_nom: nominal one-step prediction from y_hat)
Both error poles sit at p = exp(-w_o * dt), which gives
    l_y = a + 1 - 2p,    l_d = (1 - p)^2
so w_o [rad/s] is the only tuning knob per axis (larger = faster, noisier estimate).
"""
from typing import Optional, Sequence, Tuple

import numpy as np

from dsc_control.core.model import VelocityModel


def eso_gains(a: float, bandwidth: float, dt: float) -> Tuple[float, float]:
    p = float(np.exp(-bandwidth * dt))
    return a + 1.0 - 2.0 * p, (1.0 - p) ** 2


class DisturbanceObserver:
    def __init__(self, model: VelocityModel, bandwidth: Sequence[float]):
        gains = [eso_gains(m.a, w, model.dt) for m, w in zip(model.axes, bandwidth)]
        self.l_y = np.array([g[0] for g in gains])
        self.l_d = np.array([g[1] for g in gains])
        self.reset()

    def reset(self) -> None:
        self.y_hat: Optional[np.ndarray] = None
        self.d_hat = np.zeros(2)
        self._e = np.zeros(2)
        self._d_prev = np.zeros(2)

    def update(self, y_meas) -> np.ndarray:
        """Correct with the new measurement; returns d_hat for this control step."""
        y_meas = np.asarray(y_meas, dtype=float)
        if self.y_hat is None:  # first sample after reset: trust the measurement
            self.y_hat = y_meas.copy()
        self._e = y_meas - self.y_hat
        self._d_prev = self.d_hat.copy()
        self.d_hat = self.d_hat + self.l_d * self._e
        return self.d_hat.copy()

    def propagate(self, y_nom_next) -> None:
        """Advance y_hat with the nominal one-step prediction (computed from y_hat and the applied u)."""
        self.y_hat = np.asarray(y_nom_next, dtype=float) + self._d_prev + self.l_y * self._e
