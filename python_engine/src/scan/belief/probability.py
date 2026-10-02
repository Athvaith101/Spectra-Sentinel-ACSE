from __future__ import annotations

import numpy as np


class ProbabilityBelief:
    """
    Probability distribution over channel activity.
    """

    def __init__(
        self,
        n_channels: int,
        prior: float = 0.5,
    ):
        self.n_channels = int(n_channels)

        if self.n_channels < 1:
            raise ValueError("n_channels must be >= 1.")

        self.belief = np.full(
            self.n_channels,
            float(prior),
            dtype=np.float64,
        )

    def reset(self, prior: float = 0.5) -> None:
        self.belief.fill(float(prior))

    def update(
        self,
        channel: int,
        detected: bool,
        likelihood_hit: float = 0.9,
        likelihood_miss: float = 0.1,
    ) -> np.ndarray:
        channel = int(channel)

        if not 0 <= channel < self.n_channels:
            raise IndexError("channel out of range")

        prior = float(np.clip(self.belief[channel], 1e-6, 1 - 1e-6))

        if detected:
            p_obs_given_active = float(likelihood_hit)
            p_obs_given_inactive = 1.0 - p_obs_given_active
        else:
            p_obs_given_active = float(likelihood_miss)
            p_obs_given_inactive = 1.0 - p_obs_given_active

        numerator = p_obs_given_active * prior

        denominator = (
            numerator
            + p_obs_given_inactive * (1.0 - prior)
        )

        posterior = numerator / max(denominator, 1e-12)

        self.belief[channel] = np.clip(
            posterior,
            1e-6,
            1.0 - 1e-6,
        )

        return self.belief.copy()

    def entropy(self) -> float:
        p = np.clip(self.belief, 1e-12, 1 - 1e-12)

        return float(
            -np.sum(
                p * np.log(p)
                + (1.0 - p) * np.log(1.0 - p)
            )
        )
