from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class Hypothesis:
    sequence: List[int]
    probability: float


class MultiHypothesisPredictor:
    """
    Maintains multiple possible future channel sequences.

    This is intentionally model-agnostic: any predictor returning a
    probability vector can be used to expand the hypothesis tree.
    """

    def __init__(
        self,
        beam_width: int = 8,
        horizon: int = 3,
    ):
        self.beam_width = int(beam_width)
        self.horizon = int(horizon)

        if self.beam_width < 1:
            raise ValueError("beam_width must be >= 1.")

        if self.horizon < 1:
            raise ValueError("horizon must be >= 1.")

    def predict(
        self,
        initial_sequence: Sequence[int],
        probability_provider,
        n_classes: int,
    ) -> List[Hypothesis]:
        hypotheses = [
            Hypothesis(
                sequence=[int(x) for x in initial_sequence],
                probability=1.0,
            )
        ]

        for _ in range(self.horizon):
            candidates: List[Hypothesis] = []

            for hypothesis in hypotheses:
                context = hypothesis.sequence

                probabilities = np.asarray(
                    probability_provider(context),
                    dtype=np.float64,
                )

                if probabilities.size != n_classes:
                    raise ValueError(
                        "probability_provider returned incorrect number "
                        "of classes."
                    )

                probabilities = np.clip(probabilities, 0.0, None)

                total = probabilities.sum()

                if total <= 0.0:
                    probabilities[:] = 1.0 / n_classes
                else:
                    probabilities /= total

                top_indices = np.argsort(probabilities)[::-1][
                    : self.beam_width
                ]

                for idx in top_indices:
                    p = float(probabilities[idx])

                    if p <= 0.0:
                        continue

                    candidates.append(
                        Hypothesis(
                            sequence=context + [int(idx)],
                            probability=hypothesis.probability * p,
                        )
                    )

            candidates.sort(
                key=lambda h: h.probability,
                reverse=True,
            )

            hypotheses = candidates[: self.beam_width]

        total = sum(h.probability for h in hypotheses)

        if total > 0:
            for h in hypotheses:
                h.probability /= total

        return hypotheses
