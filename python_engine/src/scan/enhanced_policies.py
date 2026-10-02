"""
SIH26055 V5 - Steps 14-16 / PHASES 18-20
Enhanced policy components, each toggled through the SAME universal
interface so ablations isolate exactly one mechanism at a time:

  - CoverageShieldedPolicy   (PHASE 18: hard coverage shield BEFORE
    action selection - safe-candidate projection, not post-hoc check)
  - AdaptiveDwellPolicy      (PHASE 19: dwell as a decision variable)
  - OpportunityCostPolicy    (PHASE 20: penalize camping on one channel)

All wrappers only consume receiver-observable PolicyObservation data.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.scan.policy_adapter import (
    PolicyAction,
    PolicyObservation,
    SchedulerPolicy,
)
from src.scan.safety.coverage import CoverageShield


# =====================================================================
# PHASE 18: coverage shield as a PRE-decision constraint
# =====================================================================

class CoverageShieldedPolicy(SchedulerPolicy):
    """
    Architecture (roadmap):

        Candidate actions
              -> Prediction
              -> Utility
              -> Coverage shield      <-- BEFORE policy argmax
              -> Safe candidate set
              -> Policy chooses within safe set
              -> Action

    If the inner policy proposes an unsafe channel (staleness above the
    coverage window) while safe channels exist, the proposal is
    projected onto the best safe channel by inner-policy utility.
    """

    def __init__(
        self,
        inner: SchedulerPolicy,
        num_channels: int,
        coverage_window_s: float = 0.03,
        dwell_time_s: float = 0.001,
        name: str = "CoverageShielded",
    ) -> None:
        self.inner = inner
        self.name = name
        self.shield = CoverageShield(
            n_actions=num_channels,
            max_staleness=coverage_window_s,
        )
        self.dwell_time_s = float(dwell_time_s)
        self._last_staleness = np.zeros(num_channels)
        self._last_inner_utility = np.zeros(num_channels)

    def reset(self, observation=None) -> None:
        self.inner.reset(observation)
        self._last_staleness = np.zeros_like(self._last_staleness)

    def act(self, observation: PolicyObservation) -> PolicyAction:
        self._last_staleness = np.asarray(
            observation.staleness, dtype=np.float64
        )

        action = self.inner.act(observation)

        # Estimate inner utility per channel by re-querying the shield's
        # utility proxy: belief + prediction + staleness pressure.
        utility = (
            np.asarray(observation.channel_belief, dtype=np.float64)
            + np.asarray(observation.transition_prediction, dtype=np.float64)
            + 0.5 * np.minimum(
                np.asarray(observation.staleness, dtype=np.float64), 5.0
            ) / 5.0
        )

        result = self.shield.project(
            proposed_action=int(action.channel),
            staleness=self._last_staleness,
            utility=utility,
        )

        if result.shielded:
            return PolicyAction(
                channel=int(result.action),
                dwell_time_s=float(action.dwell_time_s),
                score=float(action.score),
                reason=f"shielded[{result.reason}]",
                diagnostics={
                    "shielded": 1.0,
                    "original_channel": float(action.channel),
                },
            )

        return action

    def update(self, observation, action, reward, next_observation, info=None):
        self.inner.update(observation, action, reward, next_observation, info)


# =====================================================================
# PHASE 19: adaptive dwell
# =====================================================================

class AdaptiveDwellPolicy(SchedulerPolicy):
    """
    Dwell becomes a decision variable:

        dwell = f(uncertainty, priority pressure, coverage pressure)

    Candidate dwell ladder (roadmap): 0.25 / 0.5 / 1 / 2 / 5 / 10 ms.
    Channel choice is delegated to the inner policy; dwell is chosen
    per-step from receiver-observable pressures.
    """

    DWELL_LADDER_MS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

    def __init__(
        self,
        inner: SchedulerPolicy,
        max_dwell_ms: Optional[float] = None,
        name: str = "AdaptiveDwell",
    ) -> None:
        self.inner = inner
        self.name = name

        # Optional dwell ceiling (deployment receivers often cap the
        # maximum dwell to protect scan budget).
        self.ladder = (
            [d for d in self.DWELL_LADDER_MS if d <= max_dwell_ms]
            if max_dwell_ms is not None
            else list(self.DWELL_LADDER_MS)
        ) or [self.DWELL_LADDER_MS[0]]

    def reset(self, observation=None) -> None:
        self.inner.reset(observation)

    def act(self, observation: PolicyObservation) -> PolicyAction:
        action = self.inner.act(observation)

        channel = int(action.channel)

        uncertainty = _at(observation.predictive_uncertainty, channel)
        priority = _at(observation.priority_pressure, channel)
        coverage = _at(observation.coverage_pressure, channel)

        # High uncertainty or priority -> longer dwell (listen harder).
        # High coverage pressure -> shorter dwell (protect the budget).
        demand = (
            0.5 * min(uncertainty, 1.0)
            + 0.3 * min(priority * 2.0, 1.0)
            + 0.2 * (1.0 - min(coverage, 1.0))
        )

        index = int(np.clip(demand * (len(self.ladder) - 1),
                            0, len(self.ladder) - 1))

        dwell_s = self.ladder[index] / 1000.0

        return PolicyAction(
            channel=channel,
            dwell_time_s=dwell_s,
            score=float(action.score),
            reason=f"adaptive_dwell[{self.ladder[index]:g}ms]",
            diagnostics={"dwell_demand": float(demand)},
        )

    def update(self, observation, action, reward, next_observation, info=None):
        self.inner.update(observation, action, reward, next_observation, info)


# =====================================================================
# PHASE 20: opportunity cost
# =====================================================================

class OpportunityCostPolicy(SchedulerPolicy):
    """
    For every candidate channel:

        adjusted(a) = value(a) - kappa * OpportunityCost(a)

    OpportunityCost(a) = value(best alternative) - value(a)

    This discourages camping on one apparently-promising channel: as
    one channel dominates, the cost of ignoring the alternatives rises
    and exploration is restored.
    """

    def __init__(
        self,
        inner: SchedulerPolicy,
        kappa: float = 0.35,
        camping_window: int = 12,
        name: str = "OpportunityCost",
    ) -> None:
        self.inner = inner
        self.kappa = float(kappa)
        self.camping_window = int(camping_window)
        self.name = name
        self._recent_channels: List[int] = []

    def reset(self, observation=None) -> None:
        self.inner.reset(observation)
        self._recent_channels = []

    def act(self, observation: PolicyObservation) -> PolicyObservation:
        action = self.inner.act(observation)

        # Camping score: how concentrated were recent choices?
        window = self._recent_channels[-self.camping_window:]

        if window:
            top = max(set(window), key=window.count)
            camping_fraction = window.count(top) / len(window)
        else:
            camping_fraction = 0.0

        visit_counts = np.bincount(
            self._recent_channels,
            minlength=max(
                np.asarray(observation.channel_belief).size, 1
            ),
        ).astype(np.float64)

        # Per-channel value estimate (receiver-observable).
        value = (
            np.asarray(observation.channel_belief, dtype=np.float64)
            + np.asarray(observation.transition_prediction, dtype=np.float64)
            + np.asarray(observation.activity_prediction, dtype=np.float64)
        )

        # Opportunity cost against the best alternative.
        best_alternative_value = float(np.max(value))

        adjusted = value.copy()

        for channel in range(value.size):
            cost = best_alternative_value - value[channel]

            # Extra penalty for recently-overused channels.
            if visit_counts.size > channel and len(window) >= 4:
                overuse = visit_counts[channel] / max(len(window), 1)
                cost += self.kappa * overuse * camping_fraction

            adjusted[channel] = value[channel] - self.kappa * max(cost, 0.0)

        # If the inner choice is heavily penalized, take the best
        # adjusted channel instead.
        penalized = adjusted[int(action.channel)]
        best_adjusted = int(np.argmax(adjusted))

        if adjusted[best_adjusted] > penalized * 1.15:
            return PolicyAction(
                channel=best_adjusted,
                dwell_time_s=float(action.dwell_time_s),
                score=float(adjusted[best_adjusted]),
                reason=f"opportunity_cost_override[from {action.channel}]",
                diagnostics={
                    "camping_fraction": float(camping_fraction),
                    "original_channel": float(action.channel),
                },
            )

        return PolicyAction(
            channel=int(action.channel),
            dwell_time_s=float(action.dwell_time_s),
            score=float(penalized),
            reason="opportunity_cost_keep",
            diagnostics={"camping_fraction": float(camping_fraction)},
        )

    def update(self, observation, action, reward, next_observation, info=None):
        self._recent_channels.append(int(action.channel))

        if len(self._recent_channels) > 256:
            self._recent_channels = self._recent_channels[-256:]

        self.inner.update(observation, action, reward, next_observation, info)


# =====================================================================
# Composite: the full enhanced cognitive stack
# =====================================================================

class EnhancedCognitivePolicy(SchedulerPolicy):
    """
    Shield + opportunity cost + adaptive dwell over the cognitive
    utility policy. This is the deployment candidate.
    """

    def __init__(
        self,
        inner: SchedulerPolicy,
        num_channels: int,
        coverage_window_s: float = 0.03,
        kappa: float = 0.35,
        max_dwell_ms: Optional[float] = None,
        name: str = "EnhancedCognitive",
    ) -> None:
        self.name = name
        self._shielded = CoverageShieldedPolicy(
            OpportunityCostPolicy(inner, kappa=kappa),
            num_channels=num_channels,
            coverage_window_s=coverage_window_s,
        )
        self._adaptive = AdaptiveDwellPolicy(
            self._shielded, max_dwell_ms=max_dwell_ms
        )

    def reset(self, observation=None) -> None:
        self._adaptive.reset(observation)

    def act(self, observation: PolicyObservation) -> PolicyAction:
        return self._adaptive.act(observation)

    def update(self, observation, action, reward, next_observation, info=None):
        self._adaptive.update(
            observation, action, reward, next_observation, info
        )


def _at(array: np.ndarray, index: int) -> float:
    array = np.asarray(array)

    if index >= array.size:
        return 0.0

    value = float(array[index])

    return value if np.isfinite(value) else 0.0
