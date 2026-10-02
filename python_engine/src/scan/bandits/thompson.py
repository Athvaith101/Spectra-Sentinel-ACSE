from __future__ import annotations

import numpy as np


class GaussianThompsonSampling:
    """
    Contextual Gaussian Thompson Sampling.

    Samples:

        theta_a ~ N(mu_a, Sigma_a)

    and selects:

        argmax x^T theta_a
    """

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        prior_precision: float = 1.0,
        noise_variance: float = 1.0,
        seed: int = 42,
    ):
        self.n_actions = int(n_actions)
        self.context_dim = int(context_dim)

        self.noise_variance = float(
            noise_variance
        )

        self.A = np.array(
            [
                np.eye(self.context_dim)
                * prior_precision
                for _ in range(self.n_actions)
            ],
            dtype=np.float64,
        )

        self.b = np.zeros(
            (
                self.n_actions,
                self.context_dim,
            ),
            dtype=np.float64,
        )

        self.rng = np.random.default_rng(seed)

    def select(
        self,
        context: np.ndarray,
    ) -> int:
        x = np.asarray(
            context,
            dtype=np.float64,
        ).reshape(-1)

        samples = np.zeros(
            self.n_actions,
            dtype=np.float64,
        )

        for action in range(self.n_actions):
            A_inv = np.linalg.pinv(
                self.A[action]
            )

            mean = A_inv @ self.b[action]

            covariance = (
                self.noise_variance
                * A_inv
            )

            sample = self.rng.multivariate_normal(
                mean,
                covariance + 1e-8 * np.eye(self.context_dim),
            )

            samples[action] = x @ sample

        return int(np.argmax(samples))

    def update(
        self,
        action: int,
        context: np.ndarray,
        reward: float,
    ) -> None:
        action = int(action)

        x = np.asarray(
            context,
            dtype=np.float64,
        ).reshape(-1)

        self.A[action] += np.outer(x, x)

        self.b[action] += (
            float(reward) * x
        )
