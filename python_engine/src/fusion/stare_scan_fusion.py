from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.fusion.state_builder import FusionState, SCANStateBuilder


@dataclass
class FusionDiagnostics:
    """
    Diagnostics describing how the fused state was constructed.
    """

    stare_enabled: bool
    stare_dimension: int

    belief_enabled: bool
    prediction_enabled: bool
    uncertainty_enabled: bool

    ood_active: bool
    change_point_active: bool

    state_dimension: int

    components_present: Dict[str, bool]


class STARESCANFusion:
    """
    STARE -> SCAN fusion layer.

    STARE provides a temporal representation.

    SCAN combines it with:
        - belief
        - staleness
        - prediction
        - hazard
        - uncertainty
        - OOD
        - change-point information
        - coverage pressure
        - priority pressure
        - receiver state

    The fusion layer does not itself choose an action.
    """

    def __init__(
        self,
        num_channels: int,
        stare_model: Optional[Any] = None,
        stare_dimension: int = 0,
        freeze_stare: bool = True,
    ) -> None:

        self.num_channels = int(num_channels)
        self.stare_model = stare_model
        self.freeze_stare = bool(freeze_stare)

        self.state_builder = SCANStateBuilder(
            num_channels=num_channels,
            stare_dimension=stare_dimension,
        )

        self.stare_dimension = int(stare_dimension)

        if self.stare_model is not None and self.freeze_stare:
            self._freeze_model(self.stare_model)

    # ------------------------------------------------------------
    # STARE handling
    # ------------------------------------------------------------

    @staticmethod
    def _freeze_model(model: Any) -> None:
        """
        Freeze a PyTorch-style model when possible.
        """

        if hasattr(model, "eval"):
            model.eval()

        if hasattr(model, "parameters"):
            try:
                for parameter in model.parameters():
                    parameter.requires_grad = False
            except TypeError:
                pass

    def encode_stare(
        self,
        sequence: np.ndarray,
    ) -> np.ndarray:

        if self.stare_model is None:
            return np.zeros(
                self.stare_dimension,
                dtype=np.float32,
            )

        x = np.asarray(
            sequence,
            dtype=np.float32,
        )

        # Generic model interface.
        #
        # Supported forms:
        #   model.encode(x)
        #   model.embedding(x)
        #   model(x)
        #
        # We deliberately avoid assuming one specific STARE
        # implementation here.

        if hasattr(self.stare_model, "encode"):
            embedding = self.stare_model.encode(x)

        elif hasattr(self.stare_model, "embedding"):
            embedding = self.stare_model.embedding(x)

        else:
            try:
                import torch

                tensor = torch.from_numpy(x)

                with torch.no_grad():
                    embedding = self.stare_model(
                        tensor
                    )

                if hasattr(
                    embedding,
                    "detach",
                ):
                    embedding = (
                        embedding
                        .detach()
                        .cpu()
                        .numpy()
                    )

            except ImportError as exc:
                raise RuntimeError(
                    "STARE model requires PyTorch, but "
                    "PyTorch is not available."
                ) from exc

        if hasattr(
            embedding,
            "detach",
        ):
            embedding = (
                embedding
                .detach()
                .cpu()
                .numpy()
            )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        # Collapse batch/time dimensions into a stable
        # representation.
        if embedding.ndim == 0:
            embedding = embedding.reshape(1)

        elif embedding.ndim > 1:
            embedding = np.mean(
                embedding,
                axis=tuple(
                    range(
                        embedding.ndim - 1
                    )
                ),
            )

        embedding = embedding.reshape(-1)

        if self.stare_dimension:
            if embedding.size > self.stare_dimension:
                embedding = embedding[
                    :self.stare_dimension
                ]

            elif embedding.size < self.stare_dimension:
                padded = np.zeros(
                    self.stare_dimension,
                    dtype=np.float32,
                )

                padded[:embedding.size] = embedding
                embedding = padded

        return np.nan_to_num(
            embedding,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        ).astype(
            np.float32,
            copy=False,
        )

    # ------------------------------------------------------------
    # Complete fusion
    # ------------------------------------------------------------

    def build_state(
        self,
        *,
        observation_sequence: Optional[np.ndarray] = None,
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

        if (
            stare_embedding is None
            and observation_sequence is not None
            and self.stare_model is not None
        ):
            stare_embedding = self.encode_stare(
                observation_sequence
            )

        return self.state_builder.build(
            stare_embedding=stare_embedding,
            channel_belief=channel_belief,
            staleness=staleness,
            freshness=freshness,
            transition_prediction=transition_prediction,
            activity_prediction=activity_prediction,
            hazard_prediction=hazard_prediction,
            predictive_uncertainty=predictive_uncertainty,
            ood_score=ood_score,
            change_point_score=change_point_score,
            coverage_pressure=coverage_pressure,
            priority_pressure=priority_pressure,
            current_channel=current_channel,
            current_frequency_mhz=current_frequency_mhz,
            current_time_s=current_time_s,
            current_dwell_s=current_dwell_s,
            coverage_fraction=coverage_fraction,
            mission_mode=mission_mode,
            detection_density=detection_density,
            mean_snr=mean_snr,
            transition_entropy=transition_entropy,
            unique_channel_ratio=unique_channel_ratio,
        )

    def diagnostics(
        self,
        state: FusionState,
    ) -> FusionDiagnostics:

        components = {
            "stare": state.stare_embedding.size > 0,
            "belief": state.channel_belief.size > 0,
            "staleness": state.staleness.size > 0,
            "freshness": state.freshness.size > 0,
            "transition_prediction": (
                state.transition_prediction.size > 0
            ),
            "activity_prediction": (
                state.activity_prediction.size > 0
            ),
            "hazard_prediction": (
                state.hazard_prediction.size > 0
            ),
            "uncertainty": (
                state.predictive_uncertainty.size > 0
            ),
            "coverage_pressure": (
                state.coverage_pressure.size > 0
            ),
            "priority_pressure": (
                state.priority_pressure.size > 0
            ),
        }

        return FusionDiagnostics(
            stare_enabled=(
                self.stare_model is not None
            ),
            stare_dimension=(
                state.stare_embedding.size
            ),
            belief_enabled=(
                state.channel_belief.size > 0
            ),
            prediction_enabled=(
                state.transition_prediction.size > 0
                or state.activity_prediction.size > 0
                or state.hazard_prediction.size > 0
            ),
            uncertainty_enabled=(
                state.predictive_uncertainty.size > 0
            ),
            ood_active=state.ood_active,
            change_point_active=(
                state.change_point_active
            ),
            state_dimension=state.dimension,
            components_present=components,
        )
