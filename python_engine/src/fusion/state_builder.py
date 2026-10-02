from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class FusionState:
    """
    Unified receiver-observable SCAN state.

    This object is intentionally model-agnostic.

    Every SCAN policy should ultimately receive an equivalent
    representation derived from this state.

    Ground-truth emitter identity, future frequency, true priority,
    future activity and hidden simulator state must never appear here.
    """

    # ------------------------------------------------------------
    # STARE representation
    # ------------------------------------------------------------

    stare_embedding: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    # ------------------------------------------------------------
    # Belief / channel state
    # ------------------------------------------------------------

    channel_belief: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    staleness: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    freshness: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    transition_prediction: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    activity_prediction: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    hazard_prediction: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    # ------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------

    predictive_uncertainty: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    ood_score: float = 0.0

    change_point_score: float = 0.0

    # ------------------------------------------------------------
    # Mission / safety state
    # ------------------------------------------------------------

    coverage_pressure: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    priority_pressure: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    # ------------------------------------------------------------
    # Receiver state
    # ------------------------------------------------------------

    current_channel: int = 0

    current_frequency_mhz: float = 0.0

    current_time_s: float = 0.0

    current_dwell_s: float = 0.001

    coverage_fraction: float = 0.0

    mission_mode: str = "SURVEILLANCE"

    # ------------------------------------------------------------
    # Derived diagnostics
    # ------------------------------------------------------------

    detection_density: float = 0.0

    mean_snr: float = 0.0

    transition_entropy: float = 0.0

    unique_channel_ratio: float = 0.0

    ood_active: bool = False

    change_point_active: bool = False

    def flatten(self) -> np.ndarray:
        """
        Convert the structured state into one deterministic vector.

        The ordering is fixed so all SCAN models receive the same
        feature layout.
        """

        scalar_features = np.asarray(
            [
                float(self.current_channel),
                float(self.current_frequency_mhz),
                float(self.current_time_s),
                float(self.current_dwell_s),
                float(self.coverage_fraction),
                float(self.detection_density),
                float(self.mean_snr),
                float(self.transition_entropy),
                float(self.unique_channel_ratio),
                float(self.ood_score),
                float(self.change_point_score),
            ],
            dtype=np.float32,
        )

        arrays = [
            self.stare_embedding,
            self.channel_belief,
            self.staleness,
            self.freshness,
            self.transition_prediction,
            self.activity_prediction,
            self.hazard_prediction,
            self.predictive_uncertainty,
            self.coverage_pressure,
            self.priority_pressure,
        ]

        pieces = [scalar_features]

        for array in arrays:
            value = np.asarray(
                array,
                dtype=np.float32,
            ).reshape(-1)

            pieces.append(value)

        if not pieces:
            return np.empty(
                0,
                dtype=np.float32,
            )

        result = np.concatenate(pieces)

        # Numerical safety.
        result = np.nan_to_num(
            result,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        return result.astype(
            np.float32,
            copy=False,
        )

    @property
    def dimension(self) -> int:
        return int(self.flatten().size)

    def copy(self) -> "FusionState":
        return FusionState(
            stare_embedding=self.stare_embedding.copy(),
            channel_belief=self.channel_belief.copy(),
            staleness=self.staleness.copy(),
            freshness=self.freshness.copy(),
            transition_prediction=self.transition_prediction.copy(),
            activity_prediction=self.activity_prediction.copy(),
            hazard_prediction=self.hazard_prediction.copy(),
            predictive_uncertainty=self.predictive_uncertainty.copy(),
            ood_score=self.ood_score,
            change_point_score=self.change_point_score,
            coverage_pressure=self.coverage_pressure.copy(),
            priority_pressure=self.priority_pressure.copy(),
            current_channel=self.current_channel,
            current_frequency_mhz=self.current_frequency_mhz,
            current_time_s=self.current_time_s,
            current_dwell_s=self.current_dwell_s,
            coverage_fraction=self.coverage_fraction,
            mission_mode=self.mission_mode,
            detection_density=self.detection_density,
            mean_snr=self.mean_snr,
            transition_entropy=self.transition_entropy,
            unique_channel_ratio=self.unique_channel_ratio,
            ood_active=self.ood_active,
            change_point_active=self.change_point_active,
        )


class SCANStateBuilder:
    """
    Builds FusionState objects from receiver-observable components.

    Components are deliberately injected rather than hard-coded so that
    individual modules can be benchmarked independently.
    """

    def __init__(
        self,
        num_channels: int,
        stare_dimension: int = 0,
    ) -> None:

        if num_channels <= 0:
            raise ValueError(
                "num_channels must be positive."
            )

        if stare_dimension < 0:
            raise ValueError(
                "stare_dimension cannot be negative."
            )

        self.num_channels = int(num_channels)
        self.stare_dimension = int(stare_dimension)

    # ------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------

    @staticmethod
    def _vector(
        value: Optional[Sequence[float]],
        length: int,
    ) -> np.ndarray:

        if value is None:
            return np.zeros(
                length,
                dtype=np.float32,
            )

        array = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(-1)

        if array.size == length:
            return array

        if array.size > length:
            return array[:length]

        output = np.zeros(
            length,
            dtype=np.float32,
        )

        output[:array.size] = array

        return output

    @staticmethod
    def _normalize_probability(
        value: np.ndarray,
    ) -> np.ndarray:

        value = np.asarray(
            value,
            dtype=np.float32,
        )

        value = np.nan_to_num(
            value,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        value = np.maximum(
            value,
            0.0,
        )

        total = float(np.sum(value))

        if total <= 1e-12:
            return np.full(
                value.shape,
                1.0 / max(value.size, 1),
                dtype=np.float32,
            )

        return value / total

    # ------------------------------------------------------------
    # Main builder
    # ------------------------------------------------------------

    def build(
        self,
        *,
        stare_embedding: Optional[Sequence[float]] = None,
        channel_belief: Optional[Sequence[float]] = None,
        staleness: Optional[Sequence[float]] = None,
        freshness: Optional[Sequence[float]] = None,
        transition_prediction: Optional[Sequence[float]] = None,
        activity_prediction: Optional[Sequence[float]] = None,
        hazard_prediction: Optional[Sequence[float]] = None,
        predictive_uncertainty: Optional[Sequence[float]] = None,
        ood_score: float = 0.0,
        change_point_score: float = 0.0,
        coverage_pressure: Optional[Sequence[float]] = None,
        priority_pressure: Optional[Sequence[float]] = None,
        current_channel: int = 0,
        current_frequency_mhz: float = 0.0,
        current_time_s: float = 0.0,
        current_dwell_s: float = 0.001,
        coverage_fraction: float = 0.0,
        mission_mode: str = "SURVEILLANCE",
        detection_density: float = 0.0,
        mean_snr: float = 0.0,
        transition_entropy: float = 0.0,
        unique_channel_ratio: float = 0.0,
    ) -> FusionState:

        if not 0 <= current_channel < self.num_channels:
            raise ValueError(
                "current_channel is outside the configured channel range."
            )

        stare = (
            np.asarray(
                stare_embedding,
                dtype=np.float32,
            ).reshape(-1)
            if stare_embedding is not None
            else np.zeros(
                self.stare_dimension,
                dtype=np.float32,
            )
        )

        if self.stare_dimension:
            if stare.size > self.stare_dimension:
                stare = stare[:self.stare_dimension]

            elif stare.size < self.stare_dimension:
                padded = np.zeros(
                    self.stare_dimension,
                    dtype=np.float32,
                )

                padded[:stare.size] = stare
                stare = padded

        state = FusionState(
            stare_embedding=stare,
            channel_belief=self._normalize_probability(
                self._vector(
                    channel_belief,
                    self.num_channels,
                )
            ),
            staleness=self._vector(
                staleness,
                self.num_channels,
            ),
            freshness=self._vector(
                freshness,
                self.num_channels,
            ),
            transition_prediction=self._normalize_probability(
                self._vector(
                    transition_prediction,
                    self.num_channels,
                )
            ),
            activity_prediction=self._vector(
                activity_prediction,
                self.num_channels,
            ),
            hazard_prediction=self._vector(
                hazard_prediction,
                self.num_channels,
            ),
            predictive_uncertainty=self._vector(
                predictive_uncertainty,
                self.num_channels,
            ),
            ood_score=float(ood_score),
            change_point_score=float(
                change_point_score
            ),
            coverage_pressure=self._vector(
                coverage_pressure,
                self.num_channels,
            ),
            priority_pressure=self._vector(
                priority_pressure,
                self.num_channels,
            ),
            current_channel=int(current_channel),
            current_frequency_mhz=float(
                current_frequency_mhz
            ),
            current_time_s=float(current_time_s),
            current_dwell_s=float(current_dwell_s),
            coverage_fraction=float(
                np.clip(
                    coverage_fraction,
                    0.0,
                    1.0,
                )
            ),
            mission_mode=str(mission_mode),
            detection_density=float(
                max(0.0, detection_density)
            ),
            mean_snr=float(
                max(0.0, mean_snr)
            ),
            transition_entropy=float(
                max(0.0, transition_entropy)
            ),
            unique_channel_ratio=float(
                np.clip(
                    unique_channel_ratio,
                    0.0,
                    1.0,
                )
            ),
            ood_active=float(ood_score) > 0.5,
            change_point_active=float(
                change_point_score
            ) > 0.5,
        )

        return state
