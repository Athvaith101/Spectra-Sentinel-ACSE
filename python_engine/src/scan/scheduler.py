from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from src.fusion.state_builder import FusionState
from src.scan.action_space import ScanAction, ScanActionSpace
from src.scan.candidate_actions import (
    CandidateGenerator,
    CandidateSet,
)


@dataclass
class SchedulerDecision:
    """
    Final scheduler decision.
    """

    action: ScanAction

    score: float

    candidates_considered: int

    safety_override: bool

    fallback_used: bool

    reason: str

    diagnostics: Dict[str, float] = field(
        default_factory=dict
    )


class BaseScheduler:
    """
    Common interface for every SCAN policy.

    A scheduler receives:
        FusionState

    and returns:
        channel + dwell action.

    It must not receive hidden emitter truth.
    """

    def __init__(
        self,
        action_space: ScanActionSpace,
        candidate_generator: Optional[
            CandidateGenerator
        ] = None,
    ) -> None:

        self.action_space = action_space

        self.candidate_generator = (
            candidate_generator
            or CandidateGenerator(
                action_space
            )
        )

    def score_action(
        self,
        action: ScanAction,
        state: FusionState,
    ) -> float:
        """
        Override this in learned/model-based policies.
        """

        raise NotImplementedError

    def choose(
        self,
        state: FusionState,
    ) -> SchedulerDecision:

        candidates = self.candidate_generator.generate(
            state
        )

        if not candidates.actions:
            raise RuntimeError(
                "Candidate generator returned no actions."
            )

        scores = np.asarray(
            [
                self.score_action(
                    action,
                    state,
                )
                for action in candidates.actions
            ],
            dtype=np.float64,
        )

        scores = np.nan_to_num(
            scores,
            nan=-np.inf,
            posinf=1e12,
            neginf=-1e12,
        )

        best_index = int(
            np.argmax(scores)
        )

        selected = candidates.actions[
            best_index
        ]

        return SchedulerDecision(
            action=selected,
            score=float(scores[best_index]),
            candidates_considered=len(
                candidates.actions
            ),
            safety_override=False,
            fallback_used=candidates.fallback_used,
            reason="policy_score",
        )


class SequentialScheduler(BaseScheduler):
    """
    Deterministic round-robin baseline.
    """

    def __init__(
        self,
        action_space: ScanActionSpace,
        candidate_generator: Optional[
            CandidateGenerator
        ] = None,
    ) -> None:

        super().__init__(
            action_space,
            candidate_generator,
        )

        self._next_channel = 0

    def score_action(
        self,
        action: ScanAction,
        state: FusionState,
    ) -> float:

        distance = (
            action.channel
            - self._next_channel
        )

        return -abs(distance)

    def choose(
        self,
        state: FusionState,
    ) -> SchedulerDecision:

        channel = self._next_channel

        self._next_channel = (
            self._next_channel + 1
        ) % self.action_space.num_channels

        action = self.action_space.action(
            channel,
            0,
        )

        return SchedulerDecision(
            action=action,
            score=0.0,
            candidates_considered=1,
            safety_override=False,
            fallback_used=False,
            reason="sequential",
        )


class StalenessScheduler(BaseScheduler):
    """
    Greedy staleness policy.

        choose argmax_i s_i

    This is an important baseline because it represents
    a simple coverage/freshness strategy.
    """

    def score_action(
        self,
        action: ScanAction,
        state: FusionState,
    ) -> float:

        if action.channel >= state.staleness.size:
            return -np.inf

        return float(
            state.staleness[
                action.channel
            ]
        )


class InformationGreedyScheduler(BaseScheduler):
    """
    Information-oriented baseline.

    Uses uncertainty + coverage pressure as a receiver-side
    proxy for information gain.

    The actual Information Gain implementation remains separately
    benchmarkable.
    """

    def score_action(
        self,
        action: ScanAction,
        state: FusionState,
    ) -> float:

        channel = action.channel

        uncertainty = (
            float(
                state.predictive_uncertainty[
                    channel
                ]
            )
            if channel < state.predictive_uncertainty.size
            else 0.0
        )

        coverage = (
            float(
                state.coverage_pressure[
                    channel
                ]
            )
            if channel < state.coverage_pressure.size
            else 0.0
        )

        return uncertainty + coverage


class CognitiveGreedyScheduler(BaseScheduler):
    """
    Non-trained cognitive baseline.

    Combines:

        belief
        prediction
        staleness
        uncertainty
        coverage
        priority

    This is intentionally transparent and interpretable.
    """

    def __init__(
        self,
        action_space: ScanActionSpace,
        candidate_generator: Optional[
            CandidateGenerator
        ] = None,
        belief_weight: float = 1.0,
        prediction_weight: float = 1.0,
        staleness_weight: float = 0.5,
        uncertainty_weight: float = 0.5,
        coverage_weight: float = 0.75,
        priority_weight: float = 1.25,
    ) -> None:

        super().__init__(
            action_space,
            candidate_generator,
        )

        self.belief_weight = float(
            belief_weight
        )

        self.prediction_weight = float(
            prediction_weight
        )

        self.staleness_weight = float(
            staleness_weight
        )

        self.uncertainty_weight = float(
            uncertainty_weight
        )

        self.coverage_weight = float(
            coverage_weight
        )

        self.priority_weight = float(
            priority_weight
        )

    def score_action(
        self,
        action: ScanAction,
        state: FusionState,
    ) -> float:

        i = action.channel

        def value(
            array: np.ndarray,
        ) -> float:
            if i >= array.size:
                return 0.0

            return float(
                array[i]
            )

        belief = value(
            state.channel_belief
        )

        prediction = (
            value(
                state.transition_prediction
            )
            + value(
                state.activity_prediction
            )
            + value(
                state.hazard_prediction
            )
        )

        staleness = value(
            state.staleness
        )

        uncertainty = value(
            state.predictive_uncertainty
        )

        coverage = value(
            state.coverage_pressure
        )

        priority = value(
            state.priority_pressure
        )

        score = (
            self.belief_weight * belief
            + self.prediction_weight
            * prediction
            + self.staleness_weight
            * staleness
            + self.uncertainty_weight
            * uncertainty
            + self.coverage_weight
            * coverage
            + self.priority_weight
            * priority
        )

        # Under OOD/change-point conditions, reduce reliance
        # on learned/predictive components and increase exploration.
        if state.ood_active:
            score -= (
                0.5
                * self.prediction_weight
                * prediction
            )

            score += (
                0.5
                * self.uncertainty_weight
                * uncertainty
            )

        if state.change_point_active:
            score += (
                0.5
                * self.coverage_weight
                * coverage
            )

        return float(score)


class FunctionScheduler(BaseScheduler):
    """
    Adapter for arbitrary policy functions.

    Useful for plugging in:
        DQN
        PPO
        MPC
        LinUCB
        Thompson
        Transformer
        GRU
        etc.
    """

    def __init__(
        self,
        action_space: ScanActionSpace,
        policy: Callable[
            [ScanAction, FusionState],
            float,
        ],
        candidate_generator: Optional[
            CandidateGenerator
        ] = None,
        name: str = "custom",
    ) -> None:

        super().__init__(
            action_space,
            candidate_generator,
        )

        self.policy = policy
        self.name = name

    def score_action(
        self,
        action: ScanAction,
        state: FusionState,
    ) -> float:

        return float(
            self.policy(
                action,
                state,
            )
        )

    def choose(
        self,
        state: FusionState,
    ) -> SchedulerDecision:

        decision = super().choose(
            state
        )

        decision.reason = self.name

        return decision
