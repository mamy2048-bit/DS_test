"""Per-axis first-order velocity model with input delay, and its state-space form.

[인수인계] 로봇 응답 모델. 축(v, w)마다 a, b, delay 세 값이며 config/*.yaml 의 model 항목이다.
  - 시정수 tau = -dt / ln(a), 정상상태 이득 = b / (1 - a), delay = 명령이 반영되기까지 밀리는 스텝 수.
  - 값은 실주행 bag의 /cmd_vel -> /odom 으로 식별한다:  python3 -m analysis.identify <bag>
  - 이 모델이 실제와 많이 다르면(특히 delay를 빠뜨리면) MPC+ESO가 오히려 진동을 만든다.
    로봇, 드라이버, 바닥, 하중이 바뀌면 가장 먼저 다시 식별할 것.

Axis model (sample time dt):
    y[k+1] = a * y[k] + b * u[k - delay]
so an input applied at step k first shows up in y at step k + delay + 1.

State of one axis: [y, u[k-1], ..., u[k-delay]]  (length delay + 1).
Two axes (v, w) are stacked block-diagonally: x = [x_v; x_w], u = [u_v, u_w], y = [y_v, y_w].

Offset-free extension: a constant disturbance d (one per axis) enters each y equation,
    y[k+1] = a * y[k] + b * u[k - delay] + d,   i.e. x[k+1] = A x + B u + Bd d.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class AxisModel:
    a: float
    b: float
    delay: int

    def steady_gain(self) -> float:
        return self.b / (1.0 - self.a)


@dataclass(frozen=True)
class VelocityModel:
    v: AxisModel
    w: AxisModel
    dt: float = 0.1

    @property
    def axes(self) -> Tuple[AxisModel, AxisModel]:
        return (self.v, self.w)


@dataclass(frozen=True)
class StateSpace:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    y_index: Tuple[int, ...]      # position of each axis' y inside x
    delay_slices: Tuple[slice, ...]  # position of each axis' delay chain inside x
    Bd: np.ndarray                # (nx, n_axes) disturbance input: d_j adds to axis j's y


def _axis_blocks(m: AxisModel):
    n = m.delay
    nx = n + 1
    A = np.zeros((nx, nx))
    B = np.zeros((nx, 1))
    A[0, 0] = m.a
    if n == 0:
        B[0, 0] = m.b
    else:
        A[0, n] = m.b          # y[k+1] uses u[k-n], stored at index n
        B[1, 0] = 1.0          # newest stored input u[k-1] <- u[k]
        for i in range(2, nx):
            A[i, i - 1] = 1.0  # shift the chain
    return A, B


def build_state_space(model: VelocityModel) -> StateSpace:
    blocks = [_axis_blocks(m) for m in model.axes]
    sizes = [blk[0].shape[0] for blk in blocks]
    nx = sum(sizes)
    nu = len(blocks)
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))
    C = np.zeros((nu, nx))
    Bd = np.zeros((nx, nu))
    y_index = []
    delay_slices = []
    off = 0
    for j, ((Aj, Bj), size) in enumerate(zip(blocks, sizes)):
        A[off:off + size, off:off + size] = Aj
        B[off:off + size, j:j + 1] = Bj
        C[j, off] = 1.0
        Bd[off, j] = 1.0
        y_index.append(off)
        delay_slices.append(slice(off + 1, off + size))
        off += size
    return StateSpace(A, B, C, tuple(y_index), tuple(delay_slices), Bd)


def make_state(ss: StateSpace, y_meas: np.ndarray, u_hist: Tuple[np.ndarray, ...]) -> np.ndarray:
    """Assemble x from measured y (per axis) and past inputs.

    u_hist[j] holds axis j's past inputs, most recent first: [u[k-1], ..., u[k-delay_j]].
    """
    x = np.zeros(ss.A.shape[0])
    for j, (iy, sl) in enumerate(zip(ss.y_index, ss.delay_slices)):
        x[iy] = y_meas[j]
        x[sl] = u_hist[j]
    return x
