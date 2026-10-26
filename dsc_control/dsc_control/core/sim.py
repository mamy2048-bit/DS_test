"""Test-only discrete plant built from the same model structure. Not a research simulator."""
from typing import Optional, Sequence

import numpy as np

from dsc_control.core.model import VelocityModel, build_state_space


class VelocityPlant:
    def __init__(self, model: VelocityModel, disturbance: Sequence[float] = (0.0, 0.0),
                 noise_std: float = 0.0, seed: Optional[int] = 0):
        self.ss = build_state_space(model)
        self.d = np.asarray(disturbance, dtype=float)  # added to y each step
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros(self.ss.A.shape[0])

    def output(self) -> np.ndarray:
        y = self.ss.C @ self.x
        if self.noise_std > 0:
            y = y + self.rng.normal(0.0, self.noise_std, size=y.shape)
        return y

    def step(self, u: np.ndarray) -> np.ndarray:
        self.x = self.ss.A @ self.x + self.ss.B @ np.asarray(u, dtype=float)
        self.x[list(self.ss.y_index)] += self.d
        return self.output()
