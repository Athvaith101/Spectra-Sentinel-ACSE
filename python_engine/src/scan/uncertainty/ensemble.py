from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnsembleStatistics:
    mean: np.ndarray
    variance: np.ndarray
    standard_deviation: np.ndarray
    entropy: float


def ensemble_statistics(predictions: np.ndarray) -> EnsembleStatistics:
    """
    Compute uncertainty from K model predictions.

    predictions:
        [K, N] or [K, ...]
    """

    x = np.asarray(predictions, dtype=np.float64)

    if x.ndim < 2:
        raise ValueError("predictions must have at least 2 dimensions.")

    mean = np.mean(x, axis=0)
    variance = np.mean(
        (x - mean) ** 2,
        axis=0,
    )
    std = np.sqrt(np.maximum(variance, 0.0))

    # If the final dimension represents classes, estimate entropy.
    if mean.ndim == 1:
        p = np.clip(mean, 1e-12, None)
        p = p / p.sum()
        entropy = float(-np.sum(p * np.log(p)))
    else:
        entropy = float(np.mean(std))

    return EnsembleStatistics(
        mean=mean,
        variance=variance,
        standard_deviation=std,
        entropy=entropy,
    )


def predictive_entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)

    if p.ndim != 1:
        raise ValueError("probabilities must be one-dimensional.")

    p = np.clip(p, 1e-12, None)
    p /= p.sum()

    return float(-np.sum(p * np.log(p)))


def variation_ratio(predicted_classes: np.ndarray) -> float:
    """
    Variation ratio:

        1 - mode_frequency / K
    """

    values = np.asarray(predicted_classes).reshape(-1)

    if values.size == 0:
        return 0.0

    _, counts = np.unique(values, return_counts=True)

    return float(1.0 - np.max(counts) / values.size)
