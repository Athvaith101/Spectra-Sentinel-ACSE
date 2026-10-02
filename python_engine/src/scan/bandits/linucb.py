from __future__ import annotations

import numpy as np


class LinUCB:
    """
    Contextual LinUCB.

        theta_a = A_a^-1 b_a

        UCB =
            x^T theta_a
            + alpha sqrt(x^T A_a^-1 x)
    """

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        alpha: float = 1.0,
        regularization: float = 1.0,
    ):
        self.n_actions = int(n_actions)
        self.context_dim = int(context_dim)
        self.alpha = float(alpha)

        self.A = np.array(
            [
                np.eye(self.context_dim)
                * regularization
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

    def select(
        self,
        context: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        x = np.asarray(
            context,
            dtype=np.float64,
        ).reshape(-1)

        if x.size != self.context_dim:
            raise ValueError("context dimension mismatch.")

        scores = np.zeros(
            self.n_actions,
            dtype=np.float64,
        )

        for action in range(self.n_actions):
            A_inv = np.linalg.pinv(
                self.A[action]
            )

            theta = A_inv @ self.b[action]

            mean = float(x @ theta)

            uncertainty = float(
                np.sqrt(
                    max(
                        x @ A_inv @ x,
                        0.0,
                    )
                )
            )

            scores[action] = (
                mean
                + self.alpha * uncertainty
            )

        action = int(np.argmax(scores))

        return action, scores

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
