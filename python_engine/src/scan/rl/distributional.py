from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class C51Network(nn.Module):
    """
    Categorical distributional DQN.

    Instead of estimating one Q-value, estimates:

        P(Z(s,a) = z_i)

    over a fixed support.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.n_actions = int(n_actions)
        self.atoms = int(atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        self.register_buffer(
            "support",
            torch.linspace(
                self.v_min,
                self.v_max,
                self.atoms,
            ),
        )

        self.net = nn.Sequential(
            nn.Linear(
                state_dim,
                hidden_dim,
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                n_actions * atoms,
            ),
        )

    def forward(self, state):
        logits = self.net(state)

        logits = logits.view(
            -1,
            self.n_actions,
            self.atoms,
        )

        probabilities = F.softmax(
            logits,
            dim=-1,
        )

        return probabilities

    def q_values(self, state):
        probabilities = self.forward(state)

        return torch.sum(
            probabilities
            * self.support.view(
                1,
                1,
                -1,
            ),
            dim=-1,
        )
