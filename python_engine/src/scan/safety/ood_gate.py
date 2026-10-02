from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OODGateResult:
    allowed: bool
    confidence_multiplier: float
    mode: str


class OODGate:
    """
    Converts OOD/uncertainty into a conservative decision mode.
    """

    def __init__(
        self,
        threshold: float = 5.0,
    ):
        self.threshold = float(threshold)

    def evaluate(
        self,
        score: float,
    ) -> OODGateResult:

        score = float(score)

        if score > self.threshold:
            return OODGateResult(
                allowed=False,
                confidence_multiplier=0.0,
                mode="RECOVER",
            )

        ratio = min(
            score / max(self.threshold, 1e-6),
            1.0,
        )

        return OODGateResult(
            allowed=True,
            confidence_multiplier=float(
                1.0 - 0.5 * ratio
            ),
            mode="NORMAL",
        )
