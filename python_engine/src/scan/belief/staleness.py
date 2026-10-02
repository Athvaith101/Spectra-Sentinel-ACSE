from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StalenessState:
    last_observed: np.ndarray
    current_time: float


class StalenessModel:
    """
    Staleness:

        s_i = t - t_last_observed(i)

    Freshness:

        v_i = exp(-s_i / tau)
    """

    def __init__(
        self,
        n_channels: int,
        tau: float = 1.0,
    ):
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1.")

        self.n_channels = int(n_channels)
        self.tau = max(float(tau), 1e-9)

        self.last_observed = np.full(
            self.n_channels,
            -np.inf,
            dtype=np.float64,
        )

        self.current_time = 0.0

    def reset(self, time: float = 0.0) -> None:
        self.last_observed.fill(-np.inf)
        self.current_time = float(time)

    def observe(
        self,
        channel: int,
        time: float,
    ) -> None:
        channel = int(channel)

        if not 0 <= channel < self.n_channels:
            raise IndexError("channel out of range")

        self.current_time = max(
            self.current_time,
            float(time),
        )

        self.last_observed[channel] = float(time)

    def update_time(self, time: float) -> None:
        self.current_time = max(
            self.current_time,
            float(time),
        )

    def staleness(self) -> np.ndarray:
        values = self.current_time - self.last_observed

        # Never-observed channels remain maximally stale.
        values = np.where(
            np.isfinite(values),
            np.maximum(values, 0.0),
            np.inf,
        )

        return values

    def freshness(self) -> np.ndarray:
        s = self.staleness()

        freshness = np.exp(
            -np.minimum(s, 1e12) / self.tau
        )

        freshness[~np.isfinite(s)] = 0.0

        return freshness.astype(np.float32)
