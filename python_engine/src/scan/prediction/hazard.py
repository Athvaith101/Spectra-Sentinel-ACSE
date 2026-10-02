from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class HazardResult:
    hazard: np.ndarray
    survival: np.ndarray
    expected_time: float


def hazard_from_event_times(
    event_times: Sequence[int],
    max_horizon: int,
) -> HazardResult:
    """
    Estimate discrete hazard:

        h(t) = P(T=t | T>=t)

    and survival:

        S(t) = product_{u<=t} (1-h(u))

    using empirical event times.
    """

    if max_horizon < 1:
        raise ValueError("max_horizon must be >= 1.")

    events = np.asarray(event_times, dtype=np.int64)

    if events.size == 0:
        hazard = np.zeros(max_horizon, dtype=np.float64)
        survival = np.ones(max_horizon, dtype=np.float64)

        return HazardResult(
            hazard=hazard,
            survival=survival,
            expected_time=float(max_horizon),
        )

    events = np.clip(events, 0, max_horizon - 1)

    hazard = np.zeros(max_horizon, dtype=np.float64)
    survival = np.ones(max_horizon, dtype=np.float64)

    for t in range(max_horizon):
        at_risk = np.sum(events >= t)
        failures = np.sum(events == t)

        if at_risk > 0:
            hazard[t] = failures / at_risk

        if t == 0:
            survival[t] = max(1.0 - hazard[t], 0.0)
        else:
            survival[t] = survival[t - 1] * max(
                1.0 - hazard[t],
                0.0,
            )

    # E[T] ˜ sum_t S(t)
    expected_time = float(np.sum(survival))

    return HazardResult(
        hazard=hazard,
        survival=survival,
        expected_time=expected_time,
    )


def exponential_hazard(
    age: float,
    scale: float,
) -> float:
    """
    Continuous-time exponential hazard proxy.
    """

    scale = max(float(scale), 1e-9)
    age = max(float(age), 0.0)

    # Constant hazard under exponential assumption.
    return float(1.0 / scale)


def survival_from_hazard(
    hazard: Iterable[float],
) -> np.ndarray:
    hazard = np.asarray(list(hazard), dtype=np.float64)

    hazard = np.clip(hazard, 0.0, 1.0)

    survival = np.cumprod(1.0 - hazard)

    return survival
