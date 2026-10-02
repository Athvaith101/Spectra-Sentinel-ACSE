from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class ActivityPrediction:
    probability: float
    horizon: int
    active: bool


class ActivityRatePredictor:
    """
    Estimates probability that an emitter/channel will be active
    within a future horizon using exponentially weighted history.
    """

    def __init__(
        self,
        decay: float = 0.90,
        threshold: float = 0.5,
    ):
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1].")

        self.decay = float(decay)
        self.threshold = float(threshold)

    def predict(
        self,
        activity_history: Sequence[float],
        horizon: int = 1,
    ) -> ActivityPrediction:
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")

        if not activity_history:
            probability = 0.0
        else:
            weighted_sum = 0.0
            weight_total = 0.0
            weight = 1.0

            for value in reversed(activity_history):
                weighted_sum += weight * float(value)
                weight_total += weight
                weight *= self.decay

            probability = weighted_sum / max(weight_total, 1e-12)

        # Discount confidence as the prediction horizon grows.
        probability = float(
            np.clip(probability * (self.decay ** max(horizon - 1, 0)), 0.0, 1.0)
        )

        return ActivityPrediction(
            probability=probability,
            horizon=int(horizon),
            active=probability >= self.threshold,
        )


class TimeSinceActivityPredictor:
    """Simple persistence/hazard proxy based on time since the last activity."""

    def __init__(
        self,
        scale: float = 10.0,
    ):
        self.scale = max(float(scale), 1e-6)

    def predict(
        self,
        time_since_activity: float,
        horizon: float = 1.0,
    ) -> float:
        age = max(float(time_since_activity), 0.0)
        horizon = max(float(horizon), 0.0)

        # Exponential survival model.
        probability = 1.0 - np.exp(-(age + horizon) / self.scale)

        return float(np.clip(probability, 0.0, 1.0))
