from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature(state)

        value = self.value(features)
        advantage = self.advantage(features)

        return value + (
            advantage
            - advantage.mean(
                dim=-1,
                keepdim=True,
            )
        )


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 100_000,
    ):
        self.capacity = int(capacity)

        self.buffer: list[Transition] = []
        self.position = 0

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ) -> None:
        transition = Transition(
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (
            self.position + 1
        ) % self.capacity

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> list[Transition]:
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough replay samples.")

        if rng is None:
            rng = np.random.default_rng()

        indices = rng.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
        )

        return [
            self.buffer[int(i)]
            for i in indices
        ]

    def __len__(self) -> int:
        return len(self.buffer)


def dqn_target(
    rewards: torch.Tensor,
    next_q: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
        y = r + gamma max_a Q_target(s',a)
    """

    return rewards + (
        gamma
        * (1.0 - dones)
        * next_q.max(dim=1).values
    )


def double_dqn_target(
    rewards: torch.Tensor,
    online_next_q: torch.Tensor,
    target_next_q: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
        a* = argmax_a Q_online(s',a)

        y = r + gamma Q_target(s',a*)
    """

    best_actions = online_next_q.argmax(
        dim=1,
        keepdim=True,
    )

    selected = target_next_q.gather(
        1,
        best_actions,
    ).squeeze(1)

    return rewards + (
        gamma
        * (1.0 - dones)
        * selected
    )


def dueling_q(
    value: torch.Tensor,
    advantage: torch.Tensor,
) -> torch.Tensor:
    return value + (
        advantage
        - advantage.mean(
            dim=-1,
            keepdim=True,
        )
    )
