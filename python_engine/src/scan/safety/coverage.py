from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ShieldResult:
    action: int
    shielded: bool
    safe_actions: np.ndarray
    reason: str


class CoverageShield:
    """
    Hard safety projection.

    Safe action set:

        A_safe(t) =
            {a : T_stale(a) <= W_c}

    The shield is a constraint layer, not a learned preference.
    """

    def __init__(
        self,
        n_actions: int,
        max_staleness: float = 1.0,
    ):
        self.n_actions = int(n_actions)
        self.max_staleness = float(
            max_staleness
        )

    def safe_actions(
        self,
        staleness: np.ndarray,
    ) -> np.ndarray:
        stale = np.asarray(
            staleness,
            dtype=np.float64,
        )

        if stale.size != self.n_actions:
            raise ValueError(
                "staleness dimension mismatch."
            )

        return np.where(
            stale <= self.max_staleness
        )[0]

    def project(
        self,
        proposed_action: int,
        staleness: np.ndarray,
        utility: np.ndarray | None = None,
    ) -> ShieldResult:

        proposed_action = int(
            proposed_action
        )

        safe = self.safe_actions(
            staleness
        )

        if safe.size == 0:
            # If every action is stale, select the
            # least-stale channel as emergency recovery.
            action = int(
                np.argmin(staleness)
            )

            return ShieldResult(
                action=action,
                shielded=True,
                safe_actions=safe,
                reason="all_actions_stale",
            )

        if proposed_action in safe:
            return ShieldResult(
                action=proposed_action,
                shielded=False,
                safe_actions=safe,
                reason="already_safe",
            )

        if utility is not None:
            values = np.asarray(
                utility,
                dtype=np.float64,
            )

            action = int(
                safe[
                    np.argmax(values[safe])
                ]
            )
        else:
            # Deterministic least-stale safe fallback.
            action = int(
                safe[
                    np.argmin(
                        np.asarray(staleness)[safe]
                    )
                ]
            )

        return ShieldResult(
            action=action,
            shielded=True,
            safe_actions=safe,
            reason="coverage_projection",
        )
