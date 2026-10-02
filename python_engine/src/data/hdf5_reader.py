from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


EXPECTED_FEATURES = [
    "ToA",
    "Frequency",
    "PulseWidth",
    "AoA",
    "Amplitude",
]


@dataclass(frozen=True)
class ReceiverConfig:
    dwell_centres_mhz: np.ndarray
    dwell_times_s: np.ndarray
    freq_range_mhz: np.ndarray
    start_position_km: np.ndarray


@dataclass(frozen=True)
class HDF5FileInfo:
    path: str
    n_pulses: int
    n_features: int
    features: tuple[str, ...]
    receiver: ReceiverConfig


class TSRDHDF5Reader:
    """
    Streaming reader for the TSRD HDF5 format.

    The full pulse matrix is never required to be loaded into RAM.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def inspect(self) -> HDF5FileInfo:
        with h5py.File(self.path, "r") as h5:
            features = tuple(
                x.decode() if isinstance(x, bytes) else str(x)
                for x in h5["metadata/feature_names"][()]
            )

            receiver = ReceiverConfig(
                dwell_centres_mhz=h5[
                    "metadata/receiver/dwell_centres_mhz"
                ][()],
                dwell_times_s=h5[
                    "metadata/receiver/dwell_times_s"
                ][()],
                freq_range_mhz=h5[
                    "metadata/receiver/freq_range_mhz"
                ][()],
                start_position_km=h5[
                    "metadata/receiver/start_position_km"
                ][()],
            )

            data_shape = h5["data"].shape

            return HDF5FileInfo(
                path=str(self.path),
                n_pulses=int(data_shape[0]),
                n_features=int(data_shape[1]),
                features=features,
                receiver=receiver,
            )

    def iter_batches(
        self,
        batch_size: int = 8192,
        include_labels: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
        """
        Stream pulse data in batches.

        Yields:
            X: shape [batch, 5]
            y: shape [batch, 1] or None
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        with h5py.File(self.path, "r") as h5:
            data = h5["data"]
            labels = h5["labels"] if include_labels else None

            n = data.shape[0]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)

                X = np.asarray(data[start:end], dtype=np.float32)

                y = None
                if labels is not None:
                    y = np.asarray(
                        labels[start:end],
                        dtype=np.int8,
                    )

                yield X, y

    def iter_windows(
        self,
        window_size: int = 256,
        stride: int = 128,
        include_labels: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
        """
        Stream overlapping temporal pulse windows.

        This method deliberately works file-by-file and does not
        materialize the entire dataset.
        """

        if window_size <= 0:
            raise ValueError("window_size must be positive")

        if stride <= 0:
            raise ValueError("stride must be positive")

        with h5py.File(self.path, "r") as h5:
            data = h5["data"]
            labels = h5["labels"] if include_labels else None

            n = data.shape[0]

            for start in range(0, n - window_size + 1, stride):
                end = start + window_size

                X = np.asarray(
                    data[start:end],
                    dtype=np.float32,
                )

                y = None
                if labels is not None:
                    y = np.asarray(
                        labels[start:end],
                        dtype=np.int8,
                    )

                yield X, y


def validate_schema(info: HDF5FileInfo) -> list[str]:
    errors: list[str] = []

    if info.n_features != len(EXPECTED_FEATURES):
        errors.append(
            f"Expected {len(EXPECTED_FEATURES)} features, "
            f"found {info.n_features}"
        )

    if list(info.features) != EXPECTED_FEATURES:
        errors.append(
            f"Feature mismatch: expected {EXPECTED_FEATURES}, "
            f"found {list(info.features)}"
        )

    if info.receiver.dwell_centres_mhz.shape[0] != 36:
        errors.append(
            "Expected 36 receiver dwell centres"
        )

    if info.receiver.dwell_times_s.shape[0] != 36:
        errors.append(
            "Expected 36 receiver dwell times"
        )

    if info.receiver.freq_range_mhz.shape != (2,):
        errors.append(
            "Frequency range must contain [min, max]"
        )

    return errors