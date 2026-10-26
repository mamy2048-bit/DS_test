"""Identify per-axis (a, b, delay) from aligned command/response samples.

For each candidate delay n, fit y[k+1] = a*y[k] + b*u[k-n] by least squares, then pick the n
with the lowest free-run (multi-step) simulation RMSE.

CLI:  python3 -m analysis.identify <bag> [--cmd /cmd_vel] [--odom /odom] [--max-delay 8]
"""
import argparse
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class AxisFit:
    a: float
    b: float
    delay: int
    sim_rmse: float


def simulate_axis(a: float, b: float, delay: int, u: np.ndarray, y0: float) -> np.ndarray:
    y = np.zeros(len(u))
    y[0] = y0
    for k in range(len(u) - 1):
        uk = u[k - delay] if k - delay >= 0 else 0.0
        y[k + 1] = a * y[k] + b * uk
    return y


def fit_axis(u: np.ndarray, y: np.ndarray, max_delay: int = 8) -> List[AxisFit]:
    """Return fits for every delay 0..max_delay, best (lowest sim_rmse) first."""
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    fits = []
    for n in range(max_delay + 1):
        k = np.arange(n, len(y) - 1)
        X = np.column_stack([y[k], u[k - n]])
        (a, b), *_ = np.linalg.lstsq(X, y[k + 1], rcond=None)
        if not (0.0 <= a < 1.0) or b <= 0.0:
            continue
        sim = simulate_axis(a, b, n, u, y[0])
        fits.append(AxisFit(float(a), float(b), n, float(np.sqrt(np.mean((sim - y) ** 2)))))
    return sorted(fits, key=lambda f: f.sim_rmse)


def main() -> None:
    from analysis.bag_io import align_to_odom, load_topics

    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--cmd", default="/cmd_vel")
    ap.add_argument("--odom", default="/odom")
    ap.add_argument("--max-delay", type=int, default=8)
    args = ap.parse_args()
    data = load_topics(args.bag, [args.cmd, args.odom])
    df = align_to_odom(data[args.cmd], data[args.odom])
    print("# paste into config/nav.yaml -> model:")
    for axis in ("v", "w"):
        fits = fit_axis(df[f"u_{axis}"].to_numpy(), df[f"y_{axis}"].to_numpy(), args.max_delay)
        if not fits:
            print(f"#  {axis}: no stable fit")
            continue
        best = fits[0]
        print(f"  {axis}: {{a: {best.a:.4f}, b: {best.b:.4f}, delay: {best.delay}}}"
              f"   # sim_rmse={best.sim_rmse:.5f}, steady_gain={best.b / (1 - best.a):.4f}")
        for f in fits[1:4]:
            print(f"  #   alt delay={f.delay}: a={f.a:.4f} b={f.b:.4f} sim_rmse={f.sim_rmse:.5f}")


if __name__ == "__main__":
    main()
