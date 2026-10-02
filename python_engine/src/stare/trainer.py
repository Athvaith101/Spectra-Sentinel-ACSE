from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .augment import (
    augment,
    build_aux_targets,
    make_mask,
)


def seed_everything(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class STARETrainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device="cpu",
        amp=True,
        grad_clip=1.0,
        mask_ratio=0.15,
    ):

        self.model = model.to(device)

        self.optimizer = optimizer

        self.loss_fn = loss_fn

        self.device = torch.device(device)

        self.use_amp = (
            amp
            and self.device.type == "cuda"
        )

        self.grad_clip = grad_clip

        self.mask_ratio = mask_ratio

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp
        )

    def step(
        self,
        x,
        train=True,
    ):

        x = x.to(
            self.device,
            non_blocking=True,
        )

        masked = x.clone()

        mask = make_mask(
            x,
            self.mask_ratio,
        )

        masked[mask] = 0.0

        augmented_1 = augment(x)

        augmented_2 = augment(x)

        with torch.cuda.amp.autocast(
            enabled=self.use_amp
        ):

            outputs = self.model(
                masked
            )

            with torch.no_grad():

                contrastive_target = (
                    self.model(
                        augmented_2
                    )["embedding"]
                )

            # The auxiliary target is computed only
            # from the observed training window.
            auxiliary_target = (
                build_aux_targets(x)
            )

            loss, metrics = self.loss_fn(
                outputs,
                x,
                mask,
                contrastive_target.detach(),
                auxiliary_target,
            )

        if train:

            self.optimizer.zero_grad(
                set_to_none=True
            )

            self.scaler.scale(
                loss
            ).backward()

            self.scaler.unscale_(
                self.optimizer
            )

            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip,
            )

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

        return metrics

    def train_epoch(
        self,
        loader,
        max_batches=None,
    ):

        self.model.train()

        totals = {}
        count = 0

        for batch_index, (x, _) in enumerate(loader):

            metrics = self.step(
                x,
                train=True,
            )

            for key, value in metrics.items():

                totals[key] = (
                    totals.get(key, 0.0)
                    + value
                )

            count += 1

            if (
                max_batches is not None
                and count >= max_batches
            ):
                break

        return {
            key: value / max(count, 1)
            for key, value in totals.items()
        }

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        max_batches=None,
    ):

        self.model.eval()

        totals = {}
        count = 0

        for x, _ in loader:

            metrics = self.step(
                x,
                train=False,
            )

            for key, value in metrics.items():

                totals[key] = (
                    totals.get(key, 0.0)
                    + value
                )

            count += 1

            if (
                max_batches is not None
                and count >= max_batches
            ):
                break

        return {
            key: value / max(count, 1)
            for key, value in totals.items()
        }


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
    config,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    map_location="cpu",
):

    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    return checkpoint