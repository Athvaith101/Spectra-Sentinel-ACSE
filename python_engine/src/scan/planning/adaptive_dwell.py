from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DwellDecision:
    dwell_seconds: float
    reason: str


class AdaptiveDwell:
    """
    Chooses dwell time from uncertainty, detectability,
    priority and coverage pressure.
    """

    def __init__(
        self,
        minimum: float = 0.001,
        maximum: float = 0.050,
        target_uncertainty: float = 0.25,
    ):
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.target_uncertainty = float(
            target_uncertainty
        )

    def choose(
        self,
        uncertainty: float,
        detectability: float,
        priority: float = 0.5,
        coverage_pressure: float = 0.0,
    ) -> DwellDecision:

        uncertainty = max(
            float(uncertainty),
            0.0,
        )

        detectability = min(
            max(float(detectability), 0.0),
            1.0,
        )

        priority = min(
            max(float(priority), 0.0),
            1.0,
        )

        coverage_pressure = min(
            max(float(coverage_pressure), 0.0),
            1.0,
        )

        uncertainty_factor = min(
            uncertainty / max(
                self.target_uncertainty,
                1e-6,
            ),
            2.0,
        )

        priority_factor = (
            0.5
            + priority
        )

        detectability_factor = (
            1.0
            - 0.5 * detectability
        )

        coverage_factor = (
            1.0
            - 0.5 * coverage_pressure
        )

        dwell = (
            self.minimum
            + (
                self.maximum
                - self.minimum
            )
            * min(
                uncertainty_factor
                * priority_factor
                * detectability_factor
                * coverage_factor,
                1.0,
            )
        )

        if uncertainty > self.target_uncertainty:
            reason = "high_uncertainty"
        elif priority > 0.75:
            reason = "high_priority"
        elif coverage_pressure > 0.75:
            reason = "coverage_recovery"
        else:
            reason = "normal"

        return DwellDecision(
            dwell_seconds=float(dwell),
            reason=reason,
        )
