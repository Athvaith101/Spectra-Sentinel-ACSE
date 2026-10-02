from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OODResult:
    score: float
    is_ood: bool
    distance: float


class MahalanobisOOD:
    """
    Feature-space OOD detector.

    Distance:

        d(x) = sqrt((x-mu)^T Sigma^-1 (x-mu))
    """

    def __init__(
        self,
        threshold: float = 5.0,
        regularization: float = 1e-4,
    ):
        self.threshold = float(threshold)
        self.regularization = float(regularization)

        self.mean: np.ndarray | None = None
        self.precision: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "MahalanobisOOD":
        x = np.asarray(features, dtype=np.float64)

        if x.ndim != 2:
            raise ValueError("features must be [N, D].")

        self.mean = np.mean(x, axis=0)

        centered = x - self.mean

        covariance = (
            centered.T @ centered
        ) / max(len(x) - 1, 1)

        covariance += (
            self.regularization
            * np.eye(covariance.shape[0])
        )

        self.precision = np.linalg.pinv(
            covariance
        )

        return self

    def score(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        if self.mean is None or self.precision is None:
            raise RuntimeError("Fit OOD detector first.")

        x = np.asarray(features, dtype=np.float64)

        if x.ndim == 1:
            x = x[None, :]

        delta = x - self.mean

        distances = np.sqrt(
            np.maximum(
                np.einsum(
                    "bi,ij,bj->b",
                    delta,
                    self.precision,
                    delta,
                ),
                0.0,
            )
        )

        return distances

    def predict(
        self,
        features: np.ndarray,
    ) -> OODResult | list[OODResult]:
        distances = self.score(features)

        results = [
            OODResult(
                score=float(d),
                is_ood=bool(d > self.threshold),
                distance=float(d),
            )
            for d in distances
        ]

        return results[0] if len(results) == 1 else results
