from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass
class InterceptionRecord:
    """
    Ground-truth/evaluation-side interception record.

    This class belongs to evaluation only.

    intercepted:
        Whether the event was successfully intercepted.

    emit_time_s:
        Ground-truth emission time.

    intercept_time_s:
        Ground-truth time at which the receiver successfully
        intercepted the event. None if missed.

    priority:
        Optional mission priority weight.

    high_priority:
        Whether this event belongs to the high-priority subset.
    """

    event_id: int
    intercepted: bool
    emit_time_s: float
    intercept_time_s: Optional[float] = None

    priority: float = 1.0
    high_priority: bool = False

    def latency(
        self,
        horizon_s: Optional[float] = None,
    ) -> float:

        if self.intercepted:
            if self.intercept_time_s is None:
                raise ValueError(
                    "Intercepted event has no intercept_time_s."
                )

            return max(
                0.0,
                float(
                    self.intercept_time_s
                    - self.emit_time_s
                ),
            )

        if horizon_s is None:
            raise ValueError(
                "horizon_s required for censored missed latency."
            )

        return max(
            0.0,
            float(
                horizon_s
                - self.emit_time_s
            ),
        )


@dataclass
class CoverageRecord:
    """
    Evaluation-side coverage observation.
    """

    channel: int
    time_s: float
    compliant: bool


@dataclass
class MetricSummary:
    pd: float
    poi: float
    weighted_poi: float

    mean_interception_time_s: float
    p95_interception_time_s: float
    p99_interception_time_s: float

    censored_mean_latency_s: float

    high_priority_miss_rate: float

    coverage_compliance: float

    mean_staleness_s: float
    max_staleness_s: float

    total_events: int
    intercepted_events: int
    high_priority_events: int


def probability_of_detection(
    records: Sequence[InterceptionRecord],
) -> float:

    if not records:
        return 0.0

    return float(
        np.mean(
            [
                1.0 if record.intercepted else 0.0
                for record in records
            ]
        )
    )


def poi(
    successful_interceptions: int,
    opportunities: int,
) -> float:

    if opportunities <= 0:
        return 0.0

    return float(
        successful_interceptions
        / opportunities
    )


def weighted_poi(
    records: Sequence[InterceptionRecord],
) -> float:

    if not records:
        return 0.0

    weights = np.asarray(
        [
            max(0.0, float(record.priority))
            for record in records
        ],
        dtype=np.float64,
    )

    successes = np.asarray(
        [
            1.0 if record.intercepted else 0.0
            for record in records
        ],
        dtype=np.float64,
    )

    total_weight = float(
        np.sum(weights)
    )

    if total_weight <= 0:
        return 0.0

    return float(
        np.sum(weights * successes)
        / total_weight
    )


def interception_latencies(
    records: Sequence[InterceptionRecord],
) -> np.ndarray:

    values = [
        record.latency()
        for record in records
        if record.intercepted
    ]

    if not values:
        return np.empty(
            0,
            dtype=np.float64,
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def mean_interception_time(
    records: Sequence[InterceptionRecord],
) -> float:

    latencies = interception_latencies(
        records
    )

    if latencies.size == 0:
        return float("nan")

    return float(
        np.mean(latencies)
    )


def percentile_interception_time(
    records: Sequence[InterceptionRecord],
    percentile: float,
) -> float:

    latencies = interception_latencies(
        records
    )

    if latencies.size == 0:
        return float("nan")

    return float(
        np.percentile(
            latencies,
            percentile,
        )
    )


def censored_mean_latency(
    records: Sequence[InterceptionRecord],
    horizon_s: float,
) -> float:

    if not records:
        return float("nan")

    values = np.asarray(
        [
            record.latency(
                horizon_s=horizon_s
            )
            for record in records
        ],
        dtype=np.float64,
    )

    return float(
        np.mean(values)
    )


def high_priority_miss_rate(
    records: Sequence[InterceptionRecord],
) -> float:

    high_priority = [
        record
        for record in records
        if record.high_priority
    ]

    if not high_priority:
        return 0.0

    misses = sum(
        not record.intercepted
        for record in high_priority
    )

    return float(
        misses / len(high_priority)
    )


def coverage_compliance(
    records: Sequence[CoverageRecord],
) -> float:

    if not records:
        return 1.0

    compliant = sum(
        bool(record.compliant)
        for record in records
    )

    return float(
        compliant / len(records)
    )


def staleness_statistics(
    staleness_values_s: Sequence[float],
) -> tuple[float, float]:

    values = np.asarray(
        staleness_values_s,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return float("nan"), float("nan")

    return (
        float(np.mean(values)),
        float(np.max(values)),
    )


def brier_score(
    probabilities: Sequence[float],
    labels: Sequence[int],
) -> float:

    p = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    y = np.asarray(
        labels,
        dtype=np.float64,
    )

    if p.shape != y.shape:
        raise ValueError(
            "probabilities and labels must have identical shapes."
        )

    if p.size == 0:
        return float("nan")

    return float(
        np.mean(
            (p - y) ** 2
        )
    )


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    bins: int = 10,
) -> float:

    p = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    y = np.asarray(
        labels,
        dtype=np.float64,
    )

    if p.shape != y.shape:
        raise ValueError(
            "probabilities and labels must have identical shapes."
        )

    if p.size == 0:
        return float("nan")

    bins = max(
        1,
        int(bins),
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0

    for i in range(bins):
        if i == bins - 1:
            mask = (
                (p >= edges[i])
                & (p <= edges[i + 1])
            )
        else:
            mask = (
                (p >= edges[i])
                & (p < edges[i + 1])
            )

        if not np.any(mask):
            continue

        confidence = float(
            np.mean(p[mask])
        )

        accuracy = float(
            np.mean(y[mask])
        )

        fraction = float(
            np.mean(mask)
        )

        ece += (
            fraction
            * abs(
                accuracy
                - confidence
            )
        )

    return float(ece)


def summarize_episode(
    records: Sequence[InterceptionRecord],
    coverage_records: Sequence[CoverageRecord],
    staleness_values_s: Sequence[float],
    horizon_s: float,
) -> MetricSummary:

    pd_value = probability_of_detection(
        records
    )

    poi_value = poi(
        sum(
            record.intercepted
            for record in records
        ),
        len(records),
    )

    weighted_poi_value = weighted_poi(
        records
    )

    mean_mit = mean_interception_time(
        records
    )

    p95 = percentile_interception_time(
        records,
        95.0,
    )

    p99 = percentile_interception_time(
        records,
        99.0,
    )

    censored = censored_mean_latency(
        records,
        horizon_s,
    )

    hp_miss = high_priority_miss_rate(
        records
    )

    coverage = coverage_compliance(
        coverage_records
    )

    mean_stale, max_stale = (
        staleness_statistics(
            staleness_values_s
        )
    )

    return MetricSummary(
        pd=pd_value,
        poi=poi_value,
        weighted_poi=weighted_poi_value,
        mean_interception_time_s=mean_mit,
        p95_interception_time_s=p95,
        p99_interception_time_s=p99,
        censored_mean_latency_s=censored,
        high_priority_miss_rate=hp_miss,
        coverage_compliance=coverage,
        mean_staleness_s=mean_stale,
        max_staleness_s=max_stale,
        total_events=len(records),
        intercepted_events=sum(
            record.intercepted
            for record in records
        ),
        high_priority_events=sum(
            record.high_priority
            for record in records
        ),
    )
