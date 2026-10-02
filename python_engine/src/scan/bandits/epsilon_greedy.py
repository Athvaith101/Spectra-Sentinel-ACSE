from __future__ import annotations

import numpy as np


class EpsilonGreedy:
    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        seed: int = 42,
    ):
        self.n_actions = int(n_actions)
        self.epsilon = float(epsilon)

        self.values = np.zeros(
            self.n_actions,
            dtype=np.float64,
        )

        self.counts = np.zeros(
            self.n_actions,
            dtype=np.int64,
        )

        self.rng = np.random.default_rng(seed)

    def select(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(
                self.rng.integers(
                    0,
                    self.n_actions,
                )
            )

        return int(np.argmax(self.values))

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
