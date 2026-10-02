from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from src.fusion.state_builder import FusionState
from src.scan.action_space import ScanAction, ScanActionSpace


@dataclass
class CandidateSet:
    """
    Candidate actions plus diagnostics explaining why channels
    were included.
    """

    actions: List[ScanAction]

    belief_channels: List[int]
    prediction_channels: List[int]
    stale_channels: List[int]
    coverage_channels: List[int]
    priority_channels: List[int]

    safe_channels: List[int]

    fallback_used: bool = False


class CandidateGenerator:
    """
    Generates receiver-observable candidate actions.

    Candidate generation never accesses hidden truth.
    """

    def __init__(
        self,
        action_space: ScanActionSpace,
        max_candidates: int = 32,
        top_k_each: int = 4,
        coverage_window_s: float = 5.0,
    ) -> None:

        self.action_space = action_space
        self.max_candidates = int(max_candidates)
        self.top_k_each = int(top_k_each)
        self.coverage_window_s = float(
            coverage_window_s
        )

        if self.max_candidates <= 0:
            raise ValueError(
                "max_candidates must be positive."
            )

    @staticmethod
    def _top_indices(
        values: Sequence[float],
        k: int,
    ) -> List[int]:

        array = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        if array.size == 0:
            return []

        k = min(
            int(k),
            int(array.size),
        )

        indices = np.argsort(
            array
        )[::-1][:k]

        return [
            int(index)
            for index in indices
        ]

    def generate(
        self,
        state: FusionState,
    ) -> CandidateSet:

        n = self.action_space.num_channels

        if state.channel_belief.size != n:
            raise ValueError(
                "FusionState channel count does not match action space."
            )

        belief_channels = self._top_indices(
            state.channel_belief,
            self.top_k_each,
        )

        prediction_score = (
            state.transition_prediction
            + state.activity_prediction
            + state.hazard_prediction
        )

        prediction_channels = self._top_indices(
            prediction_score,
            self.top_k_each,
        )

        stale_channels = self._top_indices(
            state.staleness,
            self.top_k_each,
        )

        coverage_channels = self._top_indices(
            state.coverage_pressure,
            self.top_k_each,
        )

        priority_channels = self._top_indices(
            state.priority_pressure,
            self.top_k_each,
        )

        # --------------------------------------------------------
        # Safety-compatible channels.
        #
        # Channels with high staleness are considered first when
        # coverage pressure is significant.
        # --------------------------------------------------------

        safe_channels = list(
            dict.fromkeys(
                coverage_channels
                + stale_channels
            )
        )

        ordered_channels = list(
            dict.fromkeys(
                priority_channels
                + prediction_channels
                + belief_channels
                + coverage_channels
                + stale_channels
            )
        )

        # Current channel remains available as an exploitation
        # candidate unless another policy later removes it.
        if state.current_channel not in ordered_channels:
            ordered_channels.append(
                state.current_channel
            )

        fallback_used = False

        if not ordered_channels:
            ordered_channels = list(
                range(n)
            )
            fallback_used = True

        # --------------------------------------------------------
        # Construct channel x dwell candidates.
        # --------------------------------------------------------

        actions = self.action_space.channel_actions(
            ordered_channels
        )

        # Deduplicate while preserving order.
        unique = {}
        for action in actions:
            key = (
                action.channel,
                action.dwell_time_s,
            )
            unique[key] = action

        actions = list(
            unique.values()
        )

        if len(actions) > self.max_candidates:
            actions = actions[
                :self.max_candidates
            ]

        return CandidateSet(
            actions=actions,
            belief_channels=belief_channels,
            prediction_channels=prediction_channels,
            stale_channels=stale_channels,
            coverage_channels=coverage_channels,
            priority_channels=priority_channels,
            safe_channels=safe_channels,
            fallback_used=fallback_used,
        )
