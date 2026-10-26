"""Generic condensed linear MPC solved with OSQP.

[인수인계] 상태 x를 지우고 입력열 U = [u0..u_{N-1}](N=horizon, 축 2개)만 변수로 둔 QP(condensed)를 푼다.
  - Q: 속도 추종, R: 정상상태 입력 u_bar에서 벗어남, S: 명령 변화량(클수록 부드럽고 느림)에 대한 가중치.
  - 제약: 축별 속도 한계, 스텝당 변화량 du_max, 바퀴 속도 |v| + (track/2)|w| <= wheel_v_max.
  - P, A 행렬은 생성자에서 한 번만 만들고 매 스텝 q, l, u만 바꾼다(0.1 s 주기 안에 수 ms로 풀림).
  - OSQP는 0.6.7을 쓴다(README 설치). 1.x는 API가 달라 확인이 필요하다.

Prediction:  x[i+1] = A x[i] + B u[i] + Bd d,  y[i] = C x[i]   (d constant over the horizon)
Cost:        sum_{i=1..N} |y[i]-r[i]|^2_Q + sum_{i=0..N-1} |u[i]-ubar[i]|^2_R + |u[i]-u[i-1]|^2_S
Constraints: u_min <= u[i] <= u_max,  |u[i]-u[i-1]| <= du_max,  G u[i] <= h
The QP matrices (P, A_c) are built once; each solve only updates q, l, u.
"""
import time
from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp


@dataclass
class MpcSolution:
    ok: bool
    status: str
    u_seq: np.ndarray   # (N, nu)
    y_pred: np.ndarray  # (N, ny), y[1..N]
    solve_ms: float


class LinearMPC:
    def __init__(self, A, B, C, horizon, Q, R, S, u_min, u_max, du_max, G=None, h=None, Bd=None):
        self.A, self.B, self.C = (np.asarray(m, dtype=float) for m in (A, B, C))
        self.Bd = None if Bd is None else np.asarray(Bd, dtype=float)
        self.N = int(horizon)
        nx, nu = self.B.shape
        ny = self.C.shape[0]
        self.nx, self.nu, self.ny = nx, nu, ny
        N = self.N

        # Y = Psi x0 + Theta U
        Psi = np.zeros((N * ny, nx))
        Theta = np.zeros((N * ny, N * nu))
        Ai = np.eye(nx)
        powers = [np.eye(nx)]
        for i in range(1, N + 1):
            Ai = self.A @ Ai
            powers.append(Ai)
            Psi[(i - 1) * ny:i * ny, :] = self.C @ Ai
        for i in range(1, N + 1):
            for j in range(i):
                Theta[(i - 1) * ny:i * ny, j * nu:(j + 1) * nu] = self.C @ powers[i - 1 - j] @ self.B
        self.Psi, self.Theta = Psi, Theta
        # Y += Gamma d, with y[i] picking up C (A^0 + ... + A^(i-1)) Bd d
        self.Gamma = None
        if self.Bd is not None:
            nd = self.Bd.shape[1]
            Gamma = np.zeros((N * ny, nd))
            acc = np.zeros((nx, nx))
            for i in range(1, N + 1):
                acc = acc + powers[i - 1]
                Gamma[(i - 1) * ny:i * ny, :] = self.C @ acc @ self.Bd
            self.Gamma = Gamma

        Qb = np.kron(np.eye(N), np.diag(Q))
        Rb = np.kron(np.eye(N), np.diag(R))
        Sb = np.kron(np.eye(N), np.diag(S))
        D = np.eye(N * nu) - np.eye(N * nu, k=-nu)  # first differences
        E = np.zeros((N * nu, nu))
        E[:nu, :] = np.eye(nu)                      # u[-1] enters the first difference
        self.Qb, self.Rb, self.Sb, self.D, self.E = Qb, Rb, Sb, D, E

        H = Theta.T @ Qb @ Theta + Rb + D.T @ Sb @ D
        H = 0.5 * (H + H.T)

        rows = [np.eye(N * nu), D]
        self.u_min = np.asarray(u_min, dtype=float)
        self.u_max = np.asarray(u_max, dtype=float)
        self.du_max = np.asarray(du_max, dtype=float)
        if G is not None:
            G = np.atleast_2d(np.asarray(G, dtype=float))
            rows.append(np.kron(np.eye(N), G))
            self.h = np.asarray(h, dtype=float)
        else:
            self.h = None
        self.Ac = np.vstack(rows)

        self.solver = osqp.OSQP()
        self.solver.setup(P=sp.csc_matrix(2.0 * H), q=np.zeros(N * nu), A=sp.csc_matrix(self.Ac),
                          l=-np.inf * np.ones(self.Ac.shape[0]), u=np.inf * np.ones(self.Ac.shape[0]),
                          verbose=False, warm_start=True, polish=True,
                          eps_abs=1e-6, eps_rel=1e-6, max_iter=4000, time_limit=0.05)

    def _bounds(self, u_prev):
        N, nu = self.N, self.nu
        l_box = np.tile(self.u_min, N)
        u_box = np.tile(self.u_max, N)
        shift = self.E @ u_prev
        l_du = shift - np.tile(self.du_max, N)
        u_du = shift + np.tile(self.du_max, N)
        l = [l_box, l_du]
        u = [u_box, u_du]
        if self.h is not None:
            l.append(-np.inf * np.ones(N * len(self.h)))
            u.append(np.tile(self.h, N))
        return np.concatenate(l), np.concatenate(u)

    def solve(self, x0, r_seq, ubar_seq, u_prev, d=None) -> MpcSolution:
        """r_seq: (N, ny) references for y[1..N]; ubar_seq: (N, nu); u_prev: (nu,); d: (nd,) or None."""
        x0 = np.asarray(x0, dtype=float)
        y_free = self.Psi @ x0
        if d is not None:
            if self.Gamma is None:
                raise ValueError("solve() got d but the MPC was built without Bd")
            y_free = y_free + self.Gamma @ np.asarray(d, dtype=float)
        r = np.asarray(r_seq, dtype=float).reshape(-1)
        ubar = np.asarray(ubar_seq, dtype=float).reshape(-1)
        u_prev = np.asarray(u_prev, dtype=float)
        f = (self.Theta.T @ self.Qb @ (y_free - r)
             - self.Rb @ ubar
             - self.D.T @ self.Sb @ (self.E @ u_prev))
        l, u = self._bounds(u_prev)
        self.solver.update(q=2.0 * f, l=l, u=u)
        t0 = time.perf_counter()
        res = self.solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1e3
        status = res.info.status
        ok = res.info.status_val in (1, 2)  # solved, solved inaccurate
        if not ok:
            return MpcSolution(False, status, np.zeros((self.N, self.nu)),
                               np.zeros((self.N, self.ny)), solve_ms)
        U = res.x
        Y = y_free + self.Theta @ U
        return MpcSolution(True, status, U.reshape(self.N, self.nu), Y.reshape(self.N, self.ny), solve_ms)
