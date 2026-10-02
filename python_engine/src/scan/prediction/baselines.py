from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PredictionResult:
    probabilities: np.ndarray
    predicted: int
    confidence: float


class PersistencePredictor:
    """Predict the next frequency/channel as the most recently observed one."""

    def predict(self, sequence: Sequence[int], n_classes: int) -> PredictionResult:
        if not sequence:
            p = np.ones(n_classes, dtype=np.float32) / max(n_classes, 1)
            return PredictionResult(p, int(np.argmax(p)), float(np.max(p)))

        idx = int(sequence[-1])
        p = np.zeros(n_classes, dtype=np.float32)
        if 0 <= idx < n_classes:
            p[idx] = 1.0
        else:
            p[:] = 1.0 / n_classes

        return PredictionResult(p, int(np.argmax(p)), float(np.max(p)))


class FrequencyHistogramPredictor:
    """Empirical frequency predictor."""

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = float(smoothing)

    def predict(
        self,
        sequence: Sequence[int],
        n_classes: int,
    ) -> PredictionResult:
        counts = np.full(n_classes, self.smoothing, dtype=np.float64)

        for value in sequence:
            value = int(value)
            if 0 <= value < n_classes:
                counts[value] += 1.0

        probabilities = counts / counts.sum()

        return PredictionResult(
            probabilities.astype(np.float32),
            int(np.argmax(probabilities)),
            float(np.max(probabilities)),
        )


class MarkovPredictor:
    """First-order empirical Markov transition predictor."""

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = float(smoothing)
        self.transitions: Dict[int, Counter] = defaultdict(Counter)
        self.n_classes: Optional[int] = None

    def fit(
        self,
        sequences: Iterable[Sequence[int]],
        n_classes: int,
    ) -> "MarkovPredictor":
        self.n_classes = int(n_classes)

        for sequence in sequences:
            for a, b in zip(sequence[:-1], sequence[1:]):
                a = int(a)
                b = int(b)

                if (
                    0 <= a < self.n_classes
                    and 0 <= b < self.n_classes
                ):
                    self.transitions[a][b] += 1

        return self

    def predict(
        self,
        sequence: Sequence[int],
        n_classes: Optional[int] = None,
    ) -> PredictionResult:
        n = n_classes or self.n_classes

        if n is None:
            raise ValueError("n_classes must be supplied before prediction.")

        if not sequence:
            p = np.ones(n, dtype=np.float32) / n
            return PredictionResult(p, int(np.argmax(p)), float(np.max(p)))

        current = int(sequence[-1])
        counts = np.full(n, self.smoothing, dtype=np.float64)

        for target, count in self.transitions.get(current, {}).items():
            if 0 <= target < n:
                counts[target] += count

        probabilities = counts / counts.sum()

        return PredictionResult(
            probabilities.astype(np.float32),
            int(np.argmax(probabilities)),
            float(np.max(probabilities)),
        )


class NGramPredictor:
    """Variable-order empirical Markov predictor."""

    def __init__(
        self,
        order: int = 3,
        smoothing: float = 1.0,
    ):
        if order < 1:
            raise ValueError("order must be >= 1")

        self.order = int(order)
        self.smoothing = float(smoothing)
        self.tables: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
        self.n_classes: Optional[int] = None

    def fit(
        self,
        sequences: Iterable[Sequence[int]],
        n_classes: int,
    ) -> "NGramPredictor":
        self.n_classes = int(n_classes)

        for sequence in sequences:
            sequence = [int(x) for x in sequence]

            for i in range(self.order, len(sequence)):
                context = tuple(sequence[i - self.order:i])
                target = sequence[i]

                if 0 <= target < self.n_classes:
                    self.tables[context][target] += 1

        return self

    def predict(
        self,
        sequence: Sequence[int],
        n_classes: Optional[int] = None,
    ) -> PredictionResult:
        n = n_classes or self.n_classes

        if n is None:
            raise ValueError("n_classes must be supplied before prediction.")

        context = tuple(int(x) for x in sequence[-self.order:])

        counts = np.full(n, self.smoothing, dtype=np.float64)

        if context in self.tables:
            for target, count in self.tables[context].items():
                if 0 <= target < n:
                    counts[target] += count
        else:
            # Back off to persistence.
            if sequence:
                target = int(sequence[-1])
                if 0 <= target < n:
                    counts[target] += 1.0

        probabilities = counts / counts.sum()

        return PredictionResult(
            probabilities.astype(np.float32),
            int(np.argmax(probabilities)),
            float(np.max(probabilities)),
        )
