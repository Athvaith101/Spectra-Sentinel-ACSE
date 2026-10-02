from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset


class STAREShardDataset(IterableDataset):
    """
    Streaming STARE dataset.

    Expected shard format:

        x -> [N, 128, 5]
        y -> [N, 128]

    x feature order:

        ToA
        Frequency
        PulseWidth
        AoA
        Amplitude
    """

    def __init__(
        self,
        shard_dir,
        batch_size=64,
        max_shards=None,
        seed=42,
    ):
        super().__init__()

        self.shard_dir = Path(shard_dir)

        self.files = sorted(
            self.shard_dir.glob("shard_*.npz")
        )

        if max_shards is not None:
            self.files = self.files[:max_shards]

        if not self.files:
            raise FileNotFoundError(
                f"No STARE shards found in: {self.shard_dir}"
            )

        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
            worker_count = 1
        else:
            worker_id = worker_info.id
            worker_count = worker_info.num_workers

        files = self.files[
            worker_id::worker_count
        ]

        rng = np.random.default_rng(
            self.seed
        )

        for shard_path in files:

            with np.load(
                shard_path,
                allow_pickle=False,
            ) as shard:

                x = shard["x"].astype(
                    np.float32,
                    copy=False,
                )

                y = shard["y"]

                indices = np.arange(
                    len(x)
                )

                rng.shuffle(indices)

                for start in range(
                    0,
                    len(indices),
                    self.batch_size,
                ):

                    selected = indices[
                        start:start + self.batch_size
                    ]

                    batch_x = torch.from_numpy(
                        x[selected]
                    )

                    batch_y = torch.from_numpy(
                        y[selected]
                    )

                    yield batch_x, batch_y