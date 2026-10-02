from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def export_embeddings(
    model,
    loader,
    output,
    device="cpu",
    max_batches=None,
):

    model.eval()

    embeddings = []
    labels = []

    for batch_index, (x, y) in enumerate(loader):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        output_data = model(
            x.to(device)
        )

        embeddings.append(
            output_data["embedding"]
            .cpu()
            .numpy()
        )

        labels.append(
            y[:, -1]
            .cpu()
            .numpy()
        )

    if not embeddings:
        raise RuntimeError(
            "No batches were available for embedding export."
        )

    output = Path(output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output,
        embedding=np.concatenate(
            embeddings,
            axis=0,
        ),
        label=np.concatenate(
            labels,
            axis=0,
        ),
    )

    return output