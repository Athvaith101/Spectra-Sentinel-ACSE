from __future__ import annotations

import torch


def make_mask(
    x: torch.Tensor,
    ratio: float = 0.15,
) -> torch.Tensor:

    return (
        torch.rand(
            x.shape[0],
            x.shape[1],
            device=x.device,
        )
        < ratio
    )


def augment(
    x: torch.Tensor,
    noise_std: float = 0.02,
    feature_dropout: float = 0.05,
) -> torch.Tensor:

    result = x.clone()

    if noise_std > 0:
        result = (
            result
            + torch.randn_like(result) * noise_std
        )

    if feature_dropout > 0:

        dropout_mask = (
            torch.rand(
                result.shape[0],
                result.shape[2],
                device=result.device,
            )
            < feature_dropout
        )

        result = result.masked_fill(
            dropout_mask.unsqueeze(1),
            0.0,
        )

    return result


def build_aux_targets(
    x: torch.Tensor,
):

    # Feature index 1 = normalized frequency.
    next_frequency = x[:, -1, 1]

    # Feature index 4 = normalized amplitude.
    # This is a representation-learning proxy,
    # not a hidden emitter label.
    activity = (
        x[:, :, 4].abs().mean(dim=1)
        > 0.25
    ).float()

    return {
        "next_freq": next_frequency,
        "activity": activity,
    }