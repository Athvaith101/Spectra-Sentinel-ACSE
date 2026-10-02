from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def masked_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:

    error = (prediction - target).pow(2)

    mask = mask.to(error.dtype).unsqueeze(-1)

    numerator = (error * mask).sum()

    denominator = (
        mask.sum().clamp_min(1.0)
        * prediction.size(-1)
    )

    return numerator / denominator


def info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits = torch.matmul(z1, z2.T) / temperature

    labels = torch.arange(
        z1.size(0),
        device=z1.device,
    )

    loss_a = F.cross_entropy(
        logits,
        labels,
    )

    loss_b = F.cross_entropy(
        logits.T,
        labels,
    )

    return 0.5 * (loss_a + loss_b)


class STARELoss(nn.Module):

    def __init__(
        self,
        lambda_mask: float = 1.0,
        lambda_contrastive: float = 0.2,
        lambda_aux: float = 0.2,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.lambda_mask = lambda_mask
        self.lambda_contrastive = lambda_contrastive
        self.lambda_aux = lambda_aux
        self.temperature = temperature

    def forward(
        self,
        outputs,
        target,
        mask,
        contrastive_target,
        aux_target,
    ):

        reconstruction_loss = masked_reconstruction_loss(
            outputs["reconstruction"],
            target,
            mask,
        )

        contrastive_loss = info_nce_loss(
            outputs["embedding"],
            contrastive_target,
            self.temperature,
        )

        activity_loss = F.binary_cross_entropy_with_logits(
            outputs["activity"].squeeze(-1),
            aux_target["activity"],
        )

        frequency_loss = F.smooth_l1_loss(
            outputs["next_freq"].squeeze(-1),
            aux_target["next_freq"],
        )

        auxiliary_loss = (
            activity_loss
            + frequency_loss
        )

        total_loss = (
            self.lambda_mask * reconstruction_loss
            + self.lambda_contrastive * contrastive_loss
            + self.lambda_aux * auxiliary_loss
        )

        metrics = {
            "mask": float(reconstruction_loss.detach()),
            "contrastive": float(contrastive_loss.detach()),
            "aux": float(auxiliary_loss.detach()),
            "total": float(total_loss.detach()),
        }

        return total_loss, metrics