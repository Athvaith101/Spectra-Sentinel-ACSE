from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class PPOActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(
                state_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
        )

        self.actor = nn.Linear(
            hidden_dim,
            n_actions,
        )

        self.critic = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(self, state):
        features = self.encoder(state)

        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)

        return logits, value


def ppo_clipped_objective(
    log_prob,
    old_log_prob,
    advantage,
    clip_epsilon: float = 0.2,
):
    """
        r_t(theta) =
            pi_theta(a|s) /
            pi_theta_old(a|s)

        L_CLIP =
            E[min(
                r_t A_t,
                clip(r_t,1-eps,1+eps) A_t
            )]
    """

    ratio = torch.exp(
        log_prob - old_log_prob
    )

    clipped_ratio = torch.clamp(
        ratio,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    )

    objective = torch.minimum(
        ratio * advantage,
        clipped_ratio * advantage,
    )

    return objective.mean()


def gae(
    rewards,
    values,
    next_values,
    dones,
    gamma: float = 0.99,
    lam: float = 0.95,
):
    """
        delta_t =
            r_t + gamma V(s_{t+1}) - V(s_t)

        A_t =
            sum_l (gamma lambda)^l delta_{t+l}
    """

    rewards = torch.as_tensor(
        rewards,
        dtype=torch.float32,
    )

    values = torch.as_tensor(
        values,
        dtype=torch.float32,
    )

    next_values = torch.as_tensor(
        next_values,
        dtype=torch.float32,
    )

    dones = torch.as_tensor(
        dones,
        dtype=torch.float32,
    )

    advantages = torch.zeros_like(rewards)

    running = torch.tensor(
        0.0,
        dtype=torch.float32,
    )

    for t in reversed(
        range(len(rewards))
    ):
        delta = (
            rewards[t]
            + gamma
            * next_values[t]
            * (1.0 - dones[t])
            - values[t]
        )

        running = (
            delta
            + gamma
            * lam
            * (1.0 - dones[t])
            * running
        )

        advantages[t] = running

    returns = advantages + values

    return advantages, returns
