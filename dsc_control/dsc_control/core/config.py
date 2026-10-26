"""Controller configuration loaded from YAML (see config/default.yaml).

[인수인계] 설정 파일 두 개:
  config/nav.yaml     : Nav2(MPPI) 주행용. nav2_mpc.launch.py가 쓴다.
  config/default.yaml : 초기 저속 도킹·책상 시뮬레이션용(옛 모델). mpc.launch.py 단독 실행 시 기본값.
limits.v_max는 노드 파라미터 v_max(> 0)로 덮어쓸 수 있다(nav2_mpc.launch.py가 MPPI vx_max를 넘겨줌).
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import yaml

from dsc_control.core.model import AxisModel, VelocityModel


@dataclass(frozen=True)
class Limits:
    v_max: float        # |v| command limit [m/s]
    w_max: float        # |w| command limit [rad/s]
    wheel_v_max: float  # |v| + track/2*|w| limit [m/s], keeps motor RPM below driver MAX_RPM
    dv_max: float       # per-step change of v command [m/s]
    dw_max: float       # per-step change of w command [rad/s]
    track_width: float  # wheel base [m]
    # Reverse limit [m/s] (<= 0). None = -v_max. Under Nav2 set it to MPPI's vx_min: collision_monitor sits
    # upstream of the MPC, so anything faster than what MPPI may ask for is never collision-checked.
    v_min: Optional[float] = None

    def __post_init__(self):
        # Reverse may be allowed faster than forward (e.g. forward capped at 0.35 by the v_max override)
        if self.v_min is not None and self.v_min > 0.0:
            raise ValueError(f"limits.v_min must be <= 0, got {self.v_min}")

    @property
    def v_lo(self) -> float:
        return -self.v_max if self.v_min is None else self.v_min


@dataclass(frozen=True)
class MpcWeights:
    horizon: int
    q: Tuple[float, float]  # output tracking (v, w)
    r: Tuple[float, float]  # input deviation from steady-state target
    s: Tuple[float, float]  # input rate


@dataclass(frozen=True)
class ObserverConfig:
    bandwidth: Tuple[float, float] = (5.0, 5.0)  # ESO bandwidth w_o per axis (v, w) [rad/s]
    # Cap on the ESO correction used for control, in command units |d_hat / b| per axis (v [m/s], w [rad/s]).
    # None = no cap. The observer keeps its full estimate (and logs it); only the part fed to the target
    # and the MPC prediction is clipped, so a transient model mismatch cannot drive a large opposite command.
    max_correction: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class ControllerConfig:
    model: VelocityModel
    limits: Limits
    mpc: MpcWeights
    max_failures: int = 3
    observer: ObserverConfig = ObserverConfig()


def _axis(d) -> AxisModel:
    return AxisModel(a=float(d["a"]), b=float(d["b"]), delay=int(d["delay"]))


def config_from_dict(d: dict) -> ControllerConfig:
    m = d["model"]
    model = VelocityModel(v=_axis(m["v"]), w=_axis(m["w"]), dt=float(m["dt"]))
    lim = Limits(**{k: float(v) for k, v in d["limits"].items()})
    w = d["mpc"]
    weights = MpcWeights(horizon=int(w["horizon"]), q=tuple(map(float, w["q"])),
                         r=tuple(map(float, w["r"])), s=tuple(map(float, w["s"])))
    obs = d.get("observer", {})
    cap = obs.get("max_correction")
    observer = ObserverConfig(bandwidth=tuple(map(float, obs.get("bandwidth", (5.0, 5.0)))),
                              max_correction=None if cap is None else tuple(map(float, cap)))
    return ControllerConfig(model=model, limits=lim, mpc=weights,
                            max_failures=int(d.get("max_failures", 3)), observer=observer)


def load_config(path: str) -> ControllerConfig:
    with open(path, "r", encoding="utf-8") as f:
        return config_from_dict(yaml.safe_load(f))
