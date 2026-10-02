from __future__ import annotations

import numpy as np


class BayesianBelief:
    """
    Lightweight Bayesian channel-state filter.

    Prediction:

        P(S_t | O_1:t-1)
            = sum_j P(S_t | S_{t-1}=j) P(S_{t-1}=j | O_1:t-1)

    Update:

        P(S_t | O_1:t)
            proportional to
            P(O_t | S_t) P(S_t | O_1:t-1)
    """

    def __init__(
        self,
        transition: np.ndarray,
        prior: np.ndarray | None = None,
    ):
        transition = np.asarray(
            transition,
            dtype=np.float64,
        )

        if transition.ndim != 2:
            raise ValueError("transition must be 2-D.")

        if transition.shape[0] != transition.shape[1]:
            raise ValueError("transition must be square.")

        self.transition = self._normalize_rows(transition)

        n = transition.shape[0]

        if prior is None:
            prior = np.ones(n, dtype=np.float64) / n

        prior = np.asarray(prior, dtype=np.float64)

        if prior.shape != (n,):
            raise ValueError("prior has incorrect shape.")

        self.belief = self._normalize(prior)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, 0.0, None)

        total = values.sum()

        if total <= 0:
            return np.ones_like(values) / len(values)

        return values / total

    @classmethod
    def _normalize_rows(cls, matrix: np.ndarray) -> np.ndarray:
        output = np.zeros_like(matrix)

        for i in range(matrix.shape[0]):
            output[i] = cls._normalize(matrix[i])

        return output

    def predict(self) -> np.ndarray:
        self.belief = self.belief @ self.transition
        return self.belief.copy()

    def update(
        self,
        likelihood: np.ndarray,
    ) -> np.ndarray:
        likelihood = np.asarray(
            likelihood,
            dtype=np.float64,
        )

        if likelihood.shape != self.belief.shape:
            raise ValueError("likelihood has incorrect shape.")

        posterior = self.belief * np.clip(
            likelihood,
            1e-12,
            None,
        )

        self.belief = self._normalize(posterior)

        return self.belief.copy()

    def step(
        self,
        likelihood: np.ndarray,
    ) -> np.ndarray:
        self.predict()
        return self.update(likelihood)
