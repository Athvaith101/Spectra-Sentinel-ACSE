from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ScanAction:
    """
    Receiver action.

    channel:
        Discrete receiver channel index.

    frequency_mhz:
        Center frequency corresponding to the channel.

    dwell_time_s:
        Receiver dwell duration.
    """

    channel: int
    frequency_mhz: float
    dwell_time_s: float

    def as_tuple(self):
        return (
            self.channel,
            self.frequency_mhz,
            self.dwell_time_s,
        )


class ScanActionSpace:
    """
    Discrete channel x dwell action space.

    The action space is deliberately explicit so that different
    algorithms can operate on exactly the same actions.
    """

    def __init__(
        self,
        frequencies_mhz: Sequence[float],
        dwell_times_s: Sequence[float],
    ) -> None:

        self.frequencies_mhz = np.asarray(
            frequencies_mhz,
            dtype=np.float64,
        )

        self.dwell_times_s = np.asarray(
            dwell_times_s,
            dtype=np.float64,
        )

        if self.frequencies_mhz.ndim != 1:
            raise ValueError(
                "frequencies_mhz must be one-dimensional."
            )

        if self.dwell_times_s.ndim != 1:
            raise ValueError(
                "dwell_times_s must be one-dimensional."
            )

        if self.frequencies_mhz.size == 0:
            raise ValueError(
                "At least one frequency is required."
            )

        if self.dwell_times_s.size == 0:
            raise ValueError(
                "At least one dwell time is required."
            )

        if np.any(self.dwell_times_s <= 0):
            raise ValueError(
                "All dwell times must be positive."
            )

        self.num_channels = int(
            self.frequencies_mhz.size
        )

        self.num_dwells = int(
            self.dwell_times_s.size
        )

    @property
    def size(self) -> int:
        return self.num_channels * self.num_dwells

    def action(
        self,
        channel: int,
        dwell_index: int = 0,
    ) -> ScanAction:

        if not 0 <= channel < self.num_channels:
            raise IndexError(
                "channel outside action space."
            )

        if not 0 <= dwell_index < self.num_dwells:
            raise IndexError(
                "dwell_index outside action space."
            )

        return ScanAction(
            channel=int(channel),
            frequency_mhz=float(
                self.frequencies_mhz[channel]
            ),
            dwell_time_s=float(
                self.dwell_times_s[dwell_index]
            ),
        )

    def all_actions(self) -> list[ScanAction]:
        actions = []

        for channel in range(self.num_channels):
            for dwell_index in range(self.num_dwells):
                actions.append(
                    self.action(
                        channel,
                        dwell_index,
                    )
                )

        return actions

    def channel_actions(
        self,
        channels: Iterable[int],
    ) -> list[ScanAction]:

        actions = []

        for channel in channels:
            for dwell_index in range(self.num_dwells):
                actions.append(
                    self.action(
                        int(channel),
                        dwell_index,
                    )
                )

        return actions
