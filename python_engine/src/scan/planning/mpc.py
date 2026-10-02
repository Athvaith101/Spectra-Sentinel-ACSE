from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Sequence

import numpy as np


@dataclass
class MPCResult:
    actions: list[int]
    values: list[float]
    objective: float


class MPCPlanner:
    """
    Short-horizon model-predictive controller.

    Conceptually:

        maximize
            sum_k gamma^k U(s_{t+k}, a_{t+k})

    subject to receiver and coverage constraints.
    """

    def __init__(
        self,
        horizon: int = 3,
        gamma: float = 0.95,
        max_sequences: int = 10_000,
    ):
        self.horizon = int(horizon)
        self.gamma = float(gamma)
        self.max_sequences = int(max_sequences)

    def plan(
        self,
        state,
        actions: Sequence[int],
        transition_fn: Callable,
        utility_fn: Callable,
        safe_action_fn: Callable | None = None,
    ) -> MPCResult:

        actions = list(actions)

        if not actions:
            raise ValueError("No actions supplied.")

        candidates = list(
            product(
                actions,
                repeat=self.horizon,
            )
        )

        if len(candidates) > self.max_sequences:
            # Deterministic truncation for development use.
            candidates = candidates[
                : self.max_sequences
            ]

        best_objective = -np.inf
        best_actions = None
        best_values = None

        for sequence in candidates:
            current_state = state
            objective = 0.0
            values = []
            valid = True

            for step, action in enumerate(sequence):
                if (
                    safe_action_fn is not None
                    and not safe_action_fn(
                        current_state,
                        action,
                    )
                ):
                    valid = False
                    break

                utility = float(
                    utility_fn(
                        current_state,
                        action,
                    )
                )

                objective += (
                    self.gamma ** step
                ) * utility

                values.append(utility)

                current_state = transition_fn(
                    current_state,
                    action,
                )

            if valid and objective > best_objective:
                best_objective = objective
                best_actions = list(sequence)
                best_values = values

        if best_actions is None:
            # Safe fallback.
            best_actions = [actions[0]]
            best_values = [
                float(
                    utility_fn(
                        state,
                        actions[0],
                    )
                )
            ]
            best_objective = best_values[0]

        return MPCResult(
            actions=best_actions,
            values=best_values,
            objective=float(best_objective),
        )
