from __future__ import annotations

import torch
from torch import nn

from .temporal_backbones import make_backbone


class STAREModel(nn.Module):
    """
    STARE temporal representation learner.

    Input:
        [B, T, 5]

    Features:
        ToA
        Frequency
        PulseWidth
        AoA
        Amplitude

    Outputs:
        reconstruction
        temporal embedding
        next-frequency prediction
        activity prediction
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        backbone: str = "gru",
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()

        backbone_kwargs = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        }

        if backbone.lower() == "transformer":
            backbone_kwargs["max_len"] = max_len

        self.backbone_name = backbone

        self.encoder = make_backbone(
            backbone,
            **backbone_kwargs,
        )

        hidden = self.encoder.out_dim

        self.embedding = nn.Sequential(
            nn.Linear(hidden, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, input_dim),
        )

        self.next_frequency_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        self.activity_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        hidden = self.encoder(x)

        pooled = hidden.mean(dim=1)

        embedding = self.embedding(pooled)

        reconstruction = self.decoder(hidden)

        last_state = hidden[:, -1]

        next_frequency = self.next_frequency_head(last_state)

        activity = self.activity_head(last_state)

        return {
            "hidden": hidden,
            "embedding": embedding,
            "reconstruction": reconstruction,
            "next_freq": next_frequency,
            "activity": activity,
        }


def build_model(config: dict):
    return STAREModel(
        input_dim=config.get("input_dim", 5),
        hidden_dim=config.get("hidden_dim", 128),
        embedding_dim=config.get("embedding_dim", 64),
        backbone=config.get("backbone", "gru"),
        dropout=config.get("dropout", 0.1),
        max_len=config.get("max_len", 512),
    )