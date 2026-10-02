from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ReceiverState:
    """
    Physical/operational state of the cognitive receiver.

    This class intentionally contains only receiver-observable state.
    Ground-truth emitter information must never be placed here.
    """

    current_frequency_mhz: float = 0.0
    current_channel: int = 0
    current_time_s: float = 0.0

    dwell_time_s: float = 0.001
    retune_time_s: float = 0.0

    frequency_min_mhz: float = 0.0
    frequency_max_mhz: float = 0.0

    available_channels_mhz: List[float] = field(default_factory=list)

    total_scans: int = 0
    total_observations: int = 0
    total_detections: int = 0

    channel_visit_count: Dict[int, int] = field(default_factory=dict)
    channel_last_observed_s: Dict[int, float] = field(default_factory=dict)

    cumulative_reward: float = 0.0

    mission_mode: str = "SURVEILLANCE"

    def copy(self) -> "ReceiverState":
        return ReceiverState(
            current_frequency_mhz=self.current_frequency_mhz,
            current_channel=self.current_channel,
            current_time_s=self.current_time_s,
            dwell_time_s=self.dwell_time_s,
            retune_time_s=self.retune_time_s,
            frequency_min_mhz=self.frequency_min_mhz,
            frequency_max_mhz=self.frequency_max_mhz,
            available_channels_mhz=list(self.available_channels_mhz),
            total_scans=self.total_scans,
            total_observations=self.total_observations,
            total_detections=self.total_detections,
            channel_visit_count=dict(self.channel_visit_count),
            channel_last_observed_s=dict(self.channel_last_observed_s),
            cumulative_reward=self.cumulative_reward,
            mission_mode=self.mission_mode,
        )

    def mark_channel_observed(self, channel: int, time_s: float) -> None:
        self.channel_visit_count[channel] = (
            self.channel_visit_count.get(channel, 0) + 1
        )
        self.channel_last_observed_s[channel] = time_s

    def staleness(self, channel: int) -> float:
        last = self.channel_last_observed_s.get(channel)

        if last is None:
            return float("inf")

        return max(0.0, self.current_time_s - last)

    def coverage_fraction(self) -> float:
        n = len(self.available_channels_mhz)

        if n == 0:
            return 1.0

        visited = sum(
            1
            for channel in range(n)
            if self.channel_visit_count.get(channel, 0) > 0
        )

        return visited / float(n)
