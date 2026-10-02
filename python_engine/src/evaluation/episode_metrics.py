from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from src.evaluation.metrics import (
    CoverageRecord,
    InterceptionRecord,
    MetricSummary,
    summarize_episode,
)


@dataclass
class EpisodeMetrics:
    """
    Complete metrics container for one episode.
    """

    episode_id: int
    policy_name: str
    scenario_name: str

    summary: MetricSummary

    total_reward: float = 0.0
    decision_count: int = 0

    decision_latency_us: List[float] = field(
        default_factory=list
    )

    reward_trace: List[float] = field(
        default_factory=list
    )

    extra: Dict[str, float] = field(
        default_factory=dict
    )


class EpisodeMetricsBuilder:
    """
    Collects episode-level evaluation evidence.
    """

    def __init__(
        self,
        episode_id: int,
        policy_name: str,
        scenario_name: str,
        horizon_s: float,
    ) -> None:

        self.episode_id = int(
            episode_id
        )

        self.policy_name = str(
            policy_name
        )

        self.scenario_name = str(
            scenario_name
        )

        self.horizon_s = float(
            horizon_s
        )

        self.interceptions: List[
            InterceptionRecord
        ] = []

        self.coverage: List[
            CoverageRecord
        ] = []

        self.staleness: List[float] = []

        self.reward_trace: List[float] = []

        self.decision_latency_us: List[
            float
        ] = []

    def add_interception(
        self,
        record: InterceptionRecord,
    ) -> None:

        self.interceptions.append(
            record
        )

    def add_coverage(
        self,
        record: CoverageRecord,
    ) -> None:

        self.coverage.append(
            record
        )

    def add_staleness(
        self,
        value_s: float,
    ) -> None:

        if np.isfinite(value_s):
            self.staleness.append(
                float(value_s)
            )

    def add_reward(
        self,
        reward: float,
    ) -> None:

        self.reward_trace.append(
            float(reward)
        )

    def add_decision_latency(
        self,
        latency_us: float,
    ) -> None:

        if np.isfinite(latency_us):
            self.decision_latency_us.append(
                float(latency_us)
            )

    def finalize(self) -> EpisodeMetrics:

        summary = summarize_episode(
            records=self.interceptions,
            coverage_records=self.coverage,
            staleness_values_s=self.staleness,
            horizon_s=self.horizon_s,
        )

        return EpisodeMetrics(
            episode_id=self.episode_id,
            policy_name=self.policy_name,
            scenario_name=self.scenario_name,
            summary=summary,
            total_reward=float(
                np.sum(
                    self.reward_trace
                )
            ),
            decision_count=len(
                self.reward_trace
            ),
            decision_latency_us=list(
                self.decision_latency_us
            ),
            reward_trace=list(
                self.reward_trace
            ),
        )
