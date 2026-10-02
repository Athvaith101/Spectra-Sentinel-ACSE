from __future__ import annotations

import torch
from torch import nn


class GRUBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.net(x)
        return output


class TCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = ((kernel_size - 1) * dilation) // 2

        self.conv = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, C] -> [B, C, T]
        residual = x

        y = x.transpose(1, 2)
        y = self.conv(y)
        y = y.transpose(1, 2)

        return self.norm(residual + y)


class TCNBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        levels: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.projection = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TCNBlock(
                    channels=hidden_dim,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(levels)
            ]
        )

        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.projection(x)

        for block in self.blocks:
            y = block(y)

        return y


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"heads ({heads})."
            )

        self.projection = nn.Linear(input_dim, hidden_dim)

        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_len, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.out_dim = hidden_dim
        self.max_len = max_len

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.size(1)

        if sequence_length > self.max_len:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"Transformer max_len={self.max_len}"
            )

        y = self.projection(x)

        y = y + self.position_embedding[:, :sequence_length]

        return self.encoder(y)


def make_backbone(
    name: str,
    **kwargs,
) -> nn.Module:
    """
    Factory for the STARE temporal representation backbones.

    Supported:
        gru
        tcn
        transformer
        attention
    """

    name = name.lower().strip()

    if name == "gru":
        return GRUBackbone(**kwargs)

    if name == "tcn":
        return TCNBackbone(**kwargs)

    if name in {"transformer", "attention"}:
        return TransformerBackbone(**kwargs)

    raise ValueError(
        f"Unknown STARE backbone '{name}'. "
        f"Expected one of: gru, tcn, transformer."
    )