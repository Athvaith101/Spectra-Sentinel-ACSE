from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalResult:
    prediction_set: np.ndarray
    threshold: float


class SplitConformalClassifier:
    """
    Split-conformal prediction using nonconformity:

        score = 1 - p_true
    """

    def __init__(self, alpha: float = 0.1):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")

        self.alpha = float(alpha)
        self.threshold: float | None = None

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> "SplitConformalClassifier":
        p = np.asarray(probabilities, dtype=np.float64)
        y = np.asarray(labels).astype(np.int64)

        if p.ndim != 2:
            raise ValueError("probabilities must be [N, C].")

        if len(p) != len(y):
            raise ValueError("probabilities and labels mismatch.")

        valid = (
            (y >= 0)
            & (y < p.shape[1])
        )

        scores = 1.0 - p[
            np.arange(len(y))[valid],
            y[valid],
        ]

        if scores.size == 0:
            raise ValueError("No valid calibration examples.")

        n = scores.size

        # Conservative finite-sample quantile.
        rank = int(
            np.ceil(
                (n + 1) * (1.0 - self.alpha)
            )
        )

        rank = min(
            max(rank, 1),
            n,
        )

        self.threshold = float(
            np.sort(scores)[rank - 1]
        )

        return self

    def predict(
        self,
        probabilities: np.ndarray,
    ) -> ConformalResult:
        if self.threshold is None:
            raise RuntimeError("Fit conformal model first.")

        p = np.asarray(probabilities, dtype=np.float64)

        if p.ndim != 1:
            raise ValueError(
                "predict expects one probability vector."
            )

        prediction_set = (
            1.0 - p <= self.threshold
        )

        return ConformalResult(
            prediction_set=prediction_set,
            threshold=self.threshold,
        )
