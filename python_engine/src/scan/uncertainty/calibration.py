from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationResult:
    ece: float
    brier: float
    accuracy: float
    mean_confidence: float


def brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)

    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])

    if p.ndim != 2:
        raise ValueError("probabilities must be [N, C].")

    y = y.astype(np.int64)

    if len(y) != len(p):
        raise ValueError("probabilities and labels have different lengths.")

    one_hot = np.zeros_like(p)

    valid = (
        (y >= 0)
        & (y < p.shape[1])
    )

    rows = np.arange(len(y))[valid]
    one_hot[rows, y[valid]] = 1.0

    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels).astype(np.int64)

    if p.ndim == 1:
        confidence = np.maximum(p, 1.0 - p)
        predictions = (p >= 0.5).astype(np.int64)
    else:
        predictions = np.argmax(p, axis=1)
        confidence = np.max(p, axis=1)

    if len(predictions) != len(y):
        raise ValueError("predictions and labels have different lengths.")

    edges = np.linspace(0.0, 1.0, bins + 1)

    ece = 0.0
    n = len(y)

    for i in range(bins):
        if i == bins - 1:
            mask = (
                (confidence >= edges[i])
                & (confidence <= edges[i + 1])
            )
        else:
            mask = (
                (confidence >= edges[i])
                & (confidence < edges[i + 1])
            )

        if not np.any(mask):
            continue

        accuracy = np.mean(
            predictions[mask] == y[mask]
        )

        mean_confidence = np.mean(
            confidence[mask]
        )

        ece += (
            np.sum(mask) / max(n, 1)
        ) * abs(
            accuracy - mean_confidence
        )

    return float(ece)


def calibration_summary(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> CalibrationResult:
    p = np.asarray(probabilities)

    if p.ndim == 1:
        predictions = (p >= 0.5).astype(np.int64)
        confidence = np.maximum(p, 1.0 - p)
    else:
        predictions = np.argmax(p, axis=1)
        confidence = np.max(p, axis=1)

    labels = np.asarray(labels).astype(np.int64)

    return CalibrationResult(
        ece=expected_calibration_error(
            p,
            labels,
            bins=bins,
        ),
        brier=brier_score(
            p,
            labels,
        ),
        accuracy=float(
            np.mean(predictions == labels)
        ),
        mean_confidence=float(
            np.mean(confidence)
        ),
    )


def temperature_scale(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Apply temperature scaling:

        p_i = softmax(z_i / T)
    """

    logits = np.asarray(logits, dtype=np.float64)
    temperature = max(float(temperature), 1e-6)

    scaled = logits / temperature

    scaled -= np.max(
        scaled,
        axis=-1,
        keepdims=True,
    )

    exp = np.exp(scaled)

    return exp / np.sum(
        exp,
        axis=-1,
        keepdims=True,
    )
