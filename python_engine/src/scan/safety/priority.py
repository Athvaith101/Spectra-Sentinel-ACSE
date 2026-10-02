from __future__ import annotations

import numpy as np


def priority_protection(
    utilities: np.ndarray,
    priority: np.ndarray,
    threshold: float = 0.8,
) -> np.ndarray:
    """
    Raises the decision value of channels with high mission priority.

        U'(a) = U(a) + priority(a)

    only when priority exceeds the protection threshold.
    """

    utilities = np.asarray(
        utilities,
        dtype=np.float64,
    ).copy()

    priority = np.asarray(
        priority,
        dtype=np.float64,
    )

    if utilities.shape != priority.shape:
        raise ValueError(
            "utilities and priority must match."
        )

    mask = priority >= threshold

    utilities[mask] += priority[mask]

    return utilities
