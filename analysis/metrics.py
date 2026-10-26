"""Tracking and stopping metrics on aligned samples (t, ref, y) for one axis.

CLI:  python3 -m analysis.metrics <bag> [--ref /cmd_vel_ref] [--odom /odom]
"""
import argparse
from typing import Dict

import numpy as np


def rmse(ref: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(ref) - np.asarray(y)) ** 2)))


def integrated_error(t: np.ndarray, ref: np.ndarray, y: np.ndarray) -> float:
    """∫(ref - y) dt over the whole record (trapezoid): displacement (v) or heading (w) error."""
    return float(np.trapz(np.asarray(ref) - np.asarray(y), np.asarray(t)))


def stop_residuals(t: np.ndarray, ref: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """For every transition ref != 0 -> ref == 0, the distance travelled while ref stays 0."""
    t, ref, y = map(np.asarray, (t, ref, y))
    zero = np.abs(ref) <= eps
    out = []
    k = 1
    while k < len(ref):
        if zero[k] and not zero[k - 1]:
            j = k
            while j + 1 < len(ref) and zero[j + 1]:
                j += 1
            seg = slice(k - 1, j + 1)
            out.append(abs(float(np.trapz(y[seg], t[seg]))))
            k = j + 1
        else:
            k += 1
    return np.array(out)


def steady_errors(ref: np.ndarray, y: np.ndarray, settle: int = 10, min_len: int = 20,
                  eps: float = 1e-6) -> Dict[float, float]:
    """Mean (ref - y) over constant, nonzero ref segments, skipping the first `settle` samples."""
    ref, y = np.asarray(ref), np.asarray(y)
    out: Dict[float, list] = {}
    i = 0
    while i < len(ref):
        j = i
        while j + 1 < len(ref) and abs(ref[j + 1] - ref[i]) <= eps:
            j += 1
        if j - i + 1 >= min_len and abs(ref[i]) > eps:
            out.setdefault(round(float(ref[i]), 4), []).append(float(np.mean(ref[i + settle:j + 1] - y[i + settle:j + 1])))
        i = j + 1
    return {r: float(np.mean(v)) for r, v in out.items()}


def summarize(t, ref_v, y_v, ref_w, y_w) -> Dict[str, float]:
    res = stop_residuals(t, ref_v, y_v)
    return {
        "rmse_v": rmse(ref_v, y_v),
        "rmse_w": rmse(ref_w, y_w),
        "int_err_v_m": integrated_error(t, ref_v, y_v),
        "int_err_w_rad": integrated_error(t, ref_w, y_w),
        "stop_residual_mean_m": float(res.mean()) if res.size else float("nan"),
        "stop_residual_max_m": float(res.max()) if res.size else float("nan"),
        "n_stops": int(res.size),
    }


def main() -> None:
    from analysis.bag_io import align_to_odom, load_topics

    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--ref", default="/cmd_vel_ref")
    ap.add_argument("--odom", default="/odom")
    args = ap.parse_args()
    data = load_topics(args.bag, [args.ref, args.odom])
    df = align_to_odom(data[args.ref], data[args.odom])
    t = df["t"].to_numpy()
    for k, v in summarize(t, df["u_v"], df["y_v"], df["u_w"], df["y_w"]).items():
        print(f"{k:22s} {v: .5f}" if isinstance(v, float) else f"{k:22s} {v}")
    print("steady error v (ref -> mean ref-y):", steady_errors(df["u_v"], df["y_v"]))
    print("steady error w (ref -> mean ref-y):", steady_errors(df["u_w"], df["y_w"]))


if __name__ == "__main__":
    main()
