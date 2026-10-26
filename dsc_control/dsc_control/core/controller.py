"""Velocity-tracking controller: glues model, observer, target and MPC, and keeps the input history.

[인수인계] 제어 알고리즘의 중심. ROS를 모르므로 단위 테스트(test/test_controller.py)와 오프라인 재현이 가능하다.
mode 3가지는 같은 토픽·코드 경로를 지나므로 지연 조건이 같다.
  passthrough : MPPI 명령을 한계값으로만 자른다(기준선).
  nominal     : MPC만. 모델 y[k+1] = a*y + b*u[k-delay] 로 "ref가 실제로 나오게" 하는 명령을 계산.
  offset_free : MPC + ESO. ESO가 추정한 일정 외란 d(마찰, 경사, 모델 오차)를 목표 입력과 예측에 반영해 정상상태 오차를 없앤다.
step() 한 번의 순서와 각 단계 담당 파일:
  측정+입력이력 -> 상태 x0 (model.py) -> ESO 갱신 d_hat (observer.py) -> d 상한 적용(offset_free만)
  -> 정상상태 목표입력 u_bar (target.py) -> QP 풀이 (mpc.py) / 실패 시 직전 계획 이어쓰기 -> 한계 적용
  -> ESO 예측 전진 -> 입력 이력 저장

The disturbance observer runs in every mode so d_hat is always logged; only offset_free feeds it
into the target and the MPC prediction. passthrough and nominal therefore stay unchanged baselines.
"""
from dataclasses import dataclass

import numpy as np

from dsc_control.core.config import ControllerConfig
from dsc_control.core.model import build_state_space, make_state
from dsc_control.core.mpc import LinearMPC
from dsc_control.core.observer import DisturbanceObserver
from dsc_control.core.target import limit_command, steady_state_input

MODES = ("passthrough", "nominal", "offset_free")
MODE_ID = {m: i for i, m in enumerate(MODES)}


@dataclass
class StepResult:
    u: np.ndarray         # command to publish (v, w)
    u_bar: np.ndarray     # steady-state target input
    d_hat: np.ndarray     # observer disturbance estimate (v, w); used only in offset_free
    y_pred: np.ndarray    # (N, 2) MPC prediction; zeros in passthrough
    y_pred_1: np.ndarray  # model prediction of the next measurement given u
    solve_ms: float
    status: str
    mode: str


class VelocityTrackingController:
    def __init__(self, cfg: ControllerConfig, mode: str = "passthrough"):
        self.cfg = cfg
        self.ss = build_state_space(cfg.model)
        lim = cfg.limits
        g = 0.5 * lim.track_width
        self.mpc = LinearMPC(
            self.ss.A, self.ss.B, self.ss.C, cfg.mpc.horizon,
            Q=cfg.mpc.q, R=cfg.mpc.r, S=cfg.mpc.s,
            u_min=[lim.v_lo, -lim.w_max], u_max=[lim.v_max, lim.w_max],
            du_max=[lim.dv_max, lim.dw_max],
            G=[[1, g], [1, -g], [-1, g], [-1, -g]], h=[lim.wheel_v_max] * 4, Bd=self.ss.Bd)
        self.observer = DisturbanceObserver(cfg.model, cfg.observer.bandwidth)
        self.mode = "passthrough"
        self.set_mode(mode)
        self.reset()

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode '{mode}', expected one of {MODES}")
        self.mode = mode

    def reset(self) -> None:
        self.u_prev = np.zeros(2)
        self.u_hist = tuple(np.zeros(m.delay) for m in self.cfg.model.axes)
        self.plan = None       # last successful (N, 2) input sequence
        self.plan_age = 0
        self.failures = 0
        self.observer.reset()

    def _push_history(self, u: np.ndarray) -> None:
        hist = []
        for j, h in enumerate(self.u_hist):
            hist.append(np.concatenate(([u[j]], h[:-1])) if h.size else h)
        self.u_hist = tuple(hist)
        self.u_prev = u.copy()

    def _d_for_control(self, d_hat: np.ndarray) -> np.ndarray:
        """d_hat clipped so that its command-unit correction |d/b| stays within observer.max_correction."""
        cap = self.cfg.observer.max_correction
        if cap is None:
            return d_hat
        lim = np.asarray(cap) * np.array([m.b for m in self.cfg.model.axes])
        return np.clip(d_hat, -lim, lim)

    def step(self, ref, meas) -> StepResult:
        ref = np.asarray(ref, dtype=float)
        meas = np.asarray(meas, dtype=float)
        # MPC 초기상태: 측정 속도 + 아직 로봇에 반영되지 않은 과거 명령(delay 스텝분)
        x0 = make_state(self.ss, meas, self.u_hist)
        lim = self.cfg.limits
        # ESO는 모든 mode에서 돌려 로그를 남기고, 제어에는 offset_free일 때만(상한을 씌워) 쓴다
        d_hat = self.observer.update(meas)
        d_used = self._d_for_control(d_hat) if self.mode == "offset_free" else np.zeros(2)
        # ref를 정상상태에서 내기 위한 입력. MPC 비용의 R항이 이 값 근처로 입력을 끌어당긴다
        u_bar, _ = steady_state_input(self.cfg.model, ref, lim, d_used)
        N = self.cfg.mpc.horizon
        y_pred = np.zeros((N, 2))
        solve_ms = 0.0

        if self.mode == "passthrough":
            u = limit_command(ref, lim)
            status = "passthrough"
        else:
            sol = self.mpc.solve(x0, np.tile(ref, (N, 1)), np.tile(u_bar, (N, 1)), self.u_prev, d=d_used)
            solve_ms = sol.solve_ms
            if sol.ok:
                u, y_pred = sol.u_seq[0].copy(), sol.y_pred
                self.plan, self.plan_age, self.failures = sol.u_seq, 0, 0
                status = sol.status
            else:
                # QP 실패(시간초과 등): max_failures회까지는 직전 성공 계획의 다음 입력을 쓰고, 그 뒤에는 0
                self.failures += 1
                self.plan_age += 1
                if self.plan is not None and self.failures < self.cfg.max_failures \
                        and self.plan_age < self.plan.shape[0]:
                    u = self.plan[self.plan_age].copy()
                    status = f"fallback_shift({sol.status})"
                else:
                    u = np.zeros(2)
                    status = f"fallback_zero({sol.status})"
            u = limit_command(u, lim)

        y_pred_1 = self.ss.C @ (self.ss.A @ x0 + self.ss.B @ u + self.ss.Bd @ d_used)
        # ESO 전진은 반드시 "실제로 내보낼 u"(한계 적용 후)로 한다. 그래야 포화 중에도 d_hat이 와인드업되지 않는다.
        # 순서 주의: update(측정) -> u 계산 -> propagate(u) -> _push_history(u)
        x_hat = make_state(self.ss, self.observer.y_hat, self.u_hist)
        self.observer.propagate(self.ss.C @ (self.ss.A @ x_hat + self.ss.B @ u))
        self._push_history(u)
        return StepResult(u=u, u_bar=u_bar, d_hat=d_hat, y_pred=y_pred, y_pred_1=y_pred_1,
                          solve_ms=solve_ms, status=status, mode=self.mode)
