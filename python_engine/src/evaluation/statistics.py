from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    n: int


@dataclass
class PairedComparison:
    difference: float
    lower: float
    upper: float
    confidence: float
    n: int


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[
        [np.ndarray],
        float,
    ] = np.mean,
    confidence: float = 0.95,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> ConfidenceInterval:

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[np.isfinite(x)]

    if x.size == 0:
        return ConfidenceInterval(
            estimate=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            confidence=confidence,
            n=0,
        )

    rng = np.random.default_rng(
        seed
    )

    estimate = float(
        statistic(x)
    )

    if x.size == 1:
        return ConfidenceInterval(
            estimate=estimate,
            lower=estimate,
            upper=estimate,
            confidence=confidence,
            n=1,
        )

    samples = np.empty(
        bootstrap_samples,
        dtype=np.float64,
    )

    for i in range(
        bootstrap_samples
    ):
        sample = rng.choice(
            x,
            size=x.size,
            replace=True,
        )

        samples[i] = float(
            statistic(sample)
        )

    alpha = 1.0 - confidence

    lower = float(
        np.quantile(
            samples,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            samples,
            1.0 - alpha / 2.0,
        )
    )

    return ConfidenceInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence=confidence,
        n=int(x.size),
    )


def paired_bootstrap_difference(
    baseline: Sequence[float],
    treatment: Sequence[float],
    confidence: float = 0.95,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> PairedComparison:

    a = np.asarray(
        baseline,
        dtype=np.float64,
    )

    b = np.asarray(
        treatment,
        dtype=np.float64,
    )

    if a.shape != b.shape:
        raise ValueError(
            "Baseline and treatment must have identical shapes."
        )

    mask = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    a = a[mask]
    b = b[mask]

    if a.size == 0:
        return PairedComparison(
            difference=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            confidence=confidence,
            n=0,
        )

    differences = b - a

    estimate = float(
        np.mean(differences)
    )

    if differences.size == 1:
        return PairedComparison(
            difference=estimate,
            lower=estimate,
            upper=estimate,
            confidence=confidence,
            n=1,
        )

    rng = np.random.default_rng(
        seed
    )

    samples = np.empty(
        bootstrap_samples,
        dtype=np.float64,
    )

    for i in range(
        bootstrap_samples
    ):
        sample = rng.choice(
            differences,
            size=differences.size,
            replace=True,
        )

        samples[i] = float(
            np.mean(sample)
        )

    alpha = 1.0 - confidence

    lower = float(
        np.quantile(
            samples,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            samples,
            1.0 - alpha / 2.0,
        )
    )

    return PairedComparison(
        difference=estimate,
        lower=lower,
        upper=upper,
        confidence=confidence,
        n=int(differences.size),
    )


def aggregate_mean(
    values: Sequence[float],
) -> float:

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[np.isfinite(x)]

    if x.size == 0:
        return float("nan")

    return float(
        np.mean(x)
    )


def aggregate_median(
    values: Sequence[float],
) -> float:

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[np.isfinite(x)]

    if x.size == 0:
        return float("nan")

    return float(
        np.median(x)
    )


def aggregate_std(
    values: Sequence[float],
) -> float:

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[np.isfinite(x)]

    if x.size < 2:
        return 0.0

    return float(
        np.std(
            x,
            ddof=1,
        )
    )
