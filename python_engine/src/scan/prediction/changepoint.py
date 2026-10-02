from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ChangePointResult:
    detected: bool
    score: float
    location: int


class CUSUMChangePointDetector:
    """
    Lightweight sequential change detector.

    The detector compares the recent mean against a reference mean
    and accumulates evidence using a two-sided CUSUM statistic.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        drift: float = 0.25,
        min_history: int = 8,
    ):
        self.threshold = float(threshold)
        self.drift = float(drift)
        self.min_history = int(min_history)

        self.last_detected = False
        self.last_score = 0.0

    def detect(
        self,
        values: Sequence[float],
    ) -> ChangePointResult:
        x = np.asarray(values, dtype=np.float64)

        self.last_detected = False
        self.last_score = 0.0

        if x.size < 2 * self.min_history:
            return ChangePointResult(False, 0.0, -1)

        reference = x[:self.min_history]
        recent = x[-self.min_history:]

        mu = float(np.mean(reference))
        sigma = float(np.std(reference) + 1e-6)

        normalized = (recent - mu) / sigma

        positive = 0.0
        negative = 0.0
        best_score = 0.0

        for value in normalized:
            positive = max(0.0, positive + float(value) - self.drift)
            negative = min(0.0, negative + float(value) + self.drift)

            best_score = max(
                best_score,
                positive,
                abs(negative),
            )

        detected = best_score >= self.threshold

        self.last_detected = bool(detected)
        self.last_score = float(best_score)

        return ChangePointResult(
            detected=detected,
            score=float(best_score),
            location=max(x.size - self.min_history, 0) if detected else -1,
        )
