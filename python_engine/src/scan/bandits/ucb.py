from __future__ import annotations

import math

import numpy as np


class UCB1:
    """
    Upper Confidence Bound:

        value_a + alpha * sqrt(log(t) / N_a)
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 2.0,
    ):
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)

        self.values = np.zeros(
            self.n_actions,
            dtype=np.float64,
        )

        self.counts = np.zeros(
            self.n_actions,
            dtype=np.int64,
        )

        self.total_steps = 0

    def select(self) -> int:
        self.total_steps += 1

        unexplored = np.where(
            self.counts == 0
        )[0]

        if len(unexplored):
            return int(unexplored[0])

        confidence = self.alpha * np.sqrt(
            np.log(max(self.total_steps, 2))
            / self.counts
        )

        return int(
            np.argmax(
                self.values + confidence
            )
        )

    def update(
        self,
        action: int,
        reward: float,
    ) -> None:
        action = int(action)

        self.counts[action] += 1

        n = self.counts[action]

        self.values[action] += (
            float(reward) - self.values[action]
        ) / n
