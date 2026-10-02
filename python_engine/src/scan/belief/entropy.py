from __future__ import annotations

import numpy as np


def entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)

    p = np.clip(p, 1e-12, None)

    total = p.sum()

    if total <= 0:
        return 0.0

    p = p / total

    return float(-np.sum(p * np.log(p)))


def normalized_entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities)

    if p.size <= 1:
        return 0.0

    return float(entropy(p) / np.log(p.size))


def information_gain(
    prior: np.ndarray,
    posterior: np.ndarray,
) -> float:
    """
    IG = H(B) - H(B')
    """

    return float(
        entropy(prior) - entropy(posterior)
    )
