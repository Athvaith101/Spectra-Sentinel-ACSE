from __future__ import annotations

import math

import torch
from torch import nn


class NoisyLinear(nn.Module):
    """
    Factorized Gaussian NoisyNet layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma0: float = 0.5,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.weight_mu = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
            )
        )

        self.weight_sigma = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
            )
        )

        self.bias_mu = nn.Parameter(
            torch.empty(out_features)
        )

        self.bias_sigma = nn.Parameter(
            torch.empty(out_features)
        )

        self.register_buffer(
            "weight_epsilon",
            torch.zeros(
                out_features,
                in_features,
            ),
        )

        self.register_buffer(
            "bias_epsilon",
            torch.zeros(out_features),
        )

        self.sigma0 = float(sigma0)

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(
            self.in_features
        )

        self.weight_mu.data.uniform_(
            -bound,
            bound,
        )

        self.bias_mu.data.uniform_(
            -bound,
            bound,
        )

        sigma = (
            self.sigma0
            / math.sqrt(self.in_features)
        )

        self.weight_sigma.data.fill_(sigma)
        self.bias_sigma.data.fill_(sigma)

    @staticmethod
    def _noise(size, device):
        x = torch.randn(
            size,
            device=device,
        )

        return (
            x.sign()
            * torch.sqrt(
                torch.abs(x)
                + 1e-8
            )
        )

    def reset_noise(self):
        epsilon_in = self._noise(
            self.in_features,
            self.weight_mu.device,
        )

        epsilon_out = self._noise(
            self.out_features,
            self.weight_mu.device,
        )

        self.weight_epsilon.copy_(
            epsilon_out[:, None]
            * epsilon_in[None, :]
        )

        self.bias_epsilon.copy_(
            epsilon_out
        )

    def forward(self, x):
        if self.training:
            weight = (
                self.weight_mu
                + self.weight_sigma
                * self.weight_epsilon
            )

            bias = (
                self.bias_mu
                + self.bias_sigma
                * self.bias_epsilon
            )
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return torch.nn.functional.linear(
            x,
            weight,
            bias,
        )


class NoisyQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                state_dim,
                hidden_dim,
            ),
            nn.GELU(),
            NoisyLinear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            NoisyLinear(
                hidden_dim,
                n_actions,
            ),
        )

    def forward(self, x):
        return self.layers(x)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
