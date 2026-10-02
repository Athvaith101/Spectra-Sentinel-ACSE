from __future__ import annotations

import numpy as np


class PrioritizedReplayBuffer:
    """
    Prioritized replay:

        p_i = (|delta_i| + epsilon)^alpha

        P(i) = p_i / sum_j p_j

        w_i = (N P(i))^-beta
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        seed: int = 42,
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)

        self.buffer = []
        self.priorities = np.zeros(
            self.capacity,
            dtype=np.float64,
        )

        self.position = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        transition,
        priority: float | None = None,
    ) -> None:
        if priority is None:
            priority = (
                self.priorities.max()
                if self.buffer
                else 1.0
            )

        priority = max(
            float(priority),
            self.epsilon,
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = priority

        self.position = (
            self.position + 1
        ) % self.capacity

    def sample(
        self,
        batch_size: int,
    ):
        n = len(self.buffer)

        if n < batch_size:
            raise ValueError("Not enough replay samples.")

        priorities = self.priorities[:n]

        probabilities = (
            priorities ** self.alpha
        )

        probabilities /= probabilities.sum()

        indices = self.rng.choice(
            n,
            size=batch_size,
            replace=False,
            p=probabilities,
        )

        weights = (
            n * probabilities[indices]
        ) ** (-self.beta)

        weights /= weights.max()

        samples = [
            self.buffer[int(i)]
            for i in indices
        ]

        return (
            samples,
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(
        self,
        indices,
        priorities,
    ) -> None:
        for index, priority in zip(
            indices,
            priorities,
        ):
            self.priorities[int(index)] = max(
                float(priority),
                self.epsilon,
            )

    def __len__(self):
        return len(self.buffer)
