from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class PulseObservation:
    """
    One pulse observed by the receiver.

    This represents receiver-observable information only.
    """

    toa_s: float
    frequency_mhz: float
    pulse_width_s: float
    aoa_deg: float
    amplitude: float

    channel: int = -1
    detected: bool = True

    metadata: Dict[str, float] = field(default_factory=dict)

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.toa_s,
                self.frequency_mhz,
                self.pulse_width_s,
                self.aoa_deg,
                self.amplitude,
            ],
            dtype=np.float32,
        )


@dataclass
class ScanObservation:
    """
    Observation produced by one receiver scan.
    """

    time_s: float
    channel: int
    frequency_mhz: float
    dwell_time_s: float

    pulses: List[PulseObservation] = field(default_factory=list)

    measured_energy: float = 0.0
    noise_estimate: float = 0.0

    receiver_observed: bool = False

    @property
    def detection_count(self) -> int:
        return sum(
            1
            for pulse in self.pulses
            if pulse.detected
        )

    @property
    def detected(self) -> bool:
        return self.detection_count > 0

    @property
    def snr_proxy(self) -> float:
        if not self.pulses:
            return 0.0

        amplitudes = np.asarray(
            [pulse.amplitude for pulse in self.pulses],
            dtype=np.float32,
        )

        if amplitudes.size == 0:
            return 0.0

        signal = float(np.mean(np.abs(amplitudes)))
        noise = max(
            abs(float(self.noise_estimate)),
            1e-8,
        )

        return signal / noise

    def feature_matrix(self) -> np.ndarray:
        if not self.pulses:
            return np.empty(
                (0, 5),
                dtype=np.float32,
            )

        return np.stack(
            [
                pulse.as_array()
                for pulse in self.pulses
            ],
            axis=0,
        )

    def to_dict(self) -> Dict:
        return {
            "time_s": self.time_s,
            "channel": self.channel,
            "frequency_mhz": self.frequency_mhz,
            "dwell_time_s": self.dwell_time_s,
            "detection_count": self.detection_count,
            "detected": self.detected,
            "measured_energy": self.measured_energy,
            "noise_estimate": self.noise_estimate,
            "receiver_observed": self.receiver_observed,
        }


@dataclass
class ObservationHistory:
    """
    Bounded receiver-side observation history.

    Prediction, belief and STARE components should consume
    this receiver-observable history rather than hidden truth.
    """

    max_length: int = 512
    observations: List[ScanObservation] = field(
        default_factory=list
    )

    def append(
        self,
        observation: ScanObservation,
    ) -> None:
        self.observations.append(observation)

        if len(self.observations) > self.max_length:
            self.observations = self.observations[
                -self.max_length:
            ]

    def clear(self) -> None:
        self.observations.clear()

    def latest(
        self,
        n: int = 1,
    ) -> List[ScanObservation]:
        if n <= 0:
            return []

        return self.observations[-n:]

    def channels_seen(self) -> List[int]:
        return sorted(
            {
                observation.channel
                for observation in self.observations
                if observation.channel >= 0
            }
        )

    def detection_rate(self) -> float:
        if not self.observations:
            return 0.0

        return float(
            np.mean(
                [
                    1.0
                    if observation.detected
                    else 0.0
                    for observation in self.observations
                ]
            )
        )

    def to_matrix(self) -> np.ndarray:
        """
        Compact receiver-observable history.

        Columns:

        0 = time
        1 = channel
        2 = frequency
        3 = dwell
        4 = detection count
        5 = SNR proxy
        """

        if not self.observations:
            return np.empty(
                (0, 6),
                dtype=np.float32,
            )

        rows = []

        for observation in self.observations:
            rows.append(
                [
                    observation.time_s,
                    observation.channel,
                    observation.frequency_mhz,
                    observation.dwell_time_s,
                    observation.detection_count,
                    observation.snr_proxy,
                ]
            )

        return np.asarray(
            rows,
            dtype=np.float32,
        )
