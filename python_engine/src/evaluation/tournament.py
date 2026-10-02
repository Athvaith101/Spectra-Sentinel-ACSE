"""
SIH26055 V5 - PHASE 5
Development tournament harness.

RULES (see research roadmap):
  - The tournament communicates with policies ONLY through the universal
    interface: reset(observation) / act(observation) / update(...).
  - Truth is injected into the environment only. Policies receive
    receiver-observable state built by the SCAN state builder.
  - Exact MIT comes from environment interception receipts (PHASE 3).
    Misses stay censored - no fabricated timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence
import math

import numpy as np

from src.environment.scan_environment import (
    ScanEnvironment,
)
from src.fusion.state_builder import SCANStateBuilder
from src.scan.action_space import ScanActionSpace
from src.scan.policy_adapter import (
    PolicyAction,
    PolicyObservation,
    SchedulerPolicy,
    make_policy,
    observation_from_fusion_state,
)


# =====================================================================
# Statistical containers
# =====================================================================

@dataclass
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    n: int = 0
    confidence: float = 0.95

    @property
    def difference(self) -> float:
        return self.estimate


# =====================================================================
# Bootstrap helpers
# =====================================================================

def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 12345,
) -> ConfidenceInterval:

    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return ConfidenceInterval(
            estimate=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            n=0,
            confidence=confidence,
        )

    estimate = float(np.mean(values))

    if len(values) == 1:
        return ConfidenceInterval(
            estimate=estimate,
            lower=estimate,
            upper=estimate,
            n=1,
            confidence=confidence,
        )

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(n_bootstrap, len(values)),
        replace=True,
    )

    means = np.mean(samples, axis=1)

    alpha = 1.0 - confidence

    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))

    return ConfidenceInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        n=len(values),
        confidence=confidence,
    )


def paired_bootstrap_difference(
    baseline: Sequence[float],
    treatment: Sequence[float],
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 12345,
) -> ConfidenceInterval:

    a = np.asarray(list(baseline), dtype=float)
    b = np.asarray(list(treatment), dtype=float)

    if len(a) != len(b):
        raise ValueError(
            "Paired comparison requires equal-length arrays."
        )

    valid = np.isfinite(a) & np.isfinite(b)

    a = a[valid]
    b = b[valid]

    if len(a) == 0:
        return ConfidenceInterval(
            estimate=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            n=0,
            confidence=confidence,
        )

    differences = b - a

    estimate = float(np.mean(differences))

    if len(differences) == 1:
        return ConfidenceInterval(
            estimate=estimate,
            lower=estimate,
            upper=estimate,
            n=1,
            confidence=confidence,
        )

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        differences,
        size=(n_bootstrap, len(differences)),
        replace=True,
    )

    means = np.mean(samples, axis=1)

    alpha = 1.0 - confidence

    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))

    return ConfidenceInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        n=len(differences),
        confidence=confidence,
    )


# =====================================================================
# Scenario
# =====================================================================

@dataclass
class Scenario:
    name: str
    truth_events: Sequence[Any]
    horizon_s: float
    seed: int
    frequencies_mhz: Sequence[float]
    dwells_s: Sequence[float]
    retune_time_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_dwell_times(self) -> List[float]:

        frequencies = list(self.frequencies_mhz)
        dwells = list(self.dwells_s)

        if not frequencies:
            raise ValueError(
                f"Scenario '{self.name}' contains no frequencies."
            )

        if not dwells:
            raise ValueError(
                f"Scenario '{self.name}' contains no dwell times."
            )

        if len(dwells) == 1:
            return [float(dwells[0]) for _ in frequencies]

        if len(dwells) != len(frequencies):
            raise ValueError(
                f"Scenario '{self.name}' has "
                f"{len(frequencies)} frequencies but "
                f"{len(dwells)} dwell times."
            )

        return [float(x) for x in dwells]


# =====================================================================
# Episode result
# =====================================================================

@dataclass
class EpisodeResult:
    policy: str
    scenario: str
    episode_index: int
    seed: int

    pd: float = float("nan")
    poi: float = float("nan")
    weighted_poi: float = float("nan")

    mean_interception_time_s: float = float("nan")
    mit_p50_s: float = float("nan")
    mit_p95_s: float = float("nan")
    mit_p99_s: float = float("nan")
    censored_mean_latency_s: float = float("nan")

    hp_miss_rate: float = float("nan")
    coverage: float = float("nan")

    reward: float = float("nan")

    staleness_mean_s: float = float("nan")
    staleness_max_s: float = float("nan")

    detections: int = 0
    opportunities: int = 0
    interceptions: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Policy aggregate result
# =====================================================================

@dataclass
class PolicyResult:
    policy: str
    episodes: List[EpisodeResult]

    pd: ConfidenceInterval
    poi: ConfidenceInterval
    weighted_poi: ConfidenceInterval

    mit_s: ConfidenceInterval
    mit_p50_s: ConfidenceInterval
    mit_p95_s: ConfidenceInterval
    mit_p99_s: ConfidenceInterval
    censored_latency_s: ConfidenceInterval

    hp_miss: ConfidenceInterval
    coverage: ConfidenceInterval

    reward: ConfidenceInterval

    staleness_mean_s: ConfidenceInterval
    staleness_max_s: ConfidenceInterval

    @property
    def mean_interception_time_s(self) -> ConfidenceInterval:
        return self.mit_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "episodes": len(self.episodes),
            "pd": self.pd,
            "poi": self.poi,
            "weighted_poi": self.weighted_poi,
            "mit_s": self.mit_s,
            "mit_p50_s": self.mit_p50_s,
            "mit_p95_s": self.mit_p95_s,
            "mit_p99_s": self.mit_p99_s,
            "censored_latency_s": self.censored_latency_s,
            "hp_miss": self.hp_miss,
            "coverage": self.coverage,
            "reward": self.reward,
            "staleness_mean_s": self.staleness_mean_s,
            "staleness_max_s": self.staleness_max_s,
        }


# =====================================================================
# Tournament summary
# =====================================================================

@dataclass
class TournamentSummary:
    policies: Dict[str, PolicyResult]
    confidence: float = 0.95

    @property
    def results(self):
        return self.policies

    def __getitem__(self, policy: str) -> PolicyResult:
        return self.policies[policy]

    def keys(self):
        return self.policies.keys()

    def values(self):
        return self.policies.values()

    def items(self):
        return self.policies.items()


# =====================================================================
# Policy comparison
# =====================================================================

@dataclass
class PolicyComparison:
    baseline: str
    treatment: str

    pd_delta: ConfidenceInterval
    poi_delta: ConfidenceInterval
    weighted_poi_delta: ConfidenceInterval

    mit_delta_s: ConfidenceInterval
    censored_latency_delta_s: ConfidenceInterval

    hp_miss_delta: ConfidenceInterval
    coverage_delta: ConfidenceInterval

    reward_delta: ConfidenceInterval

    def to_dict(self):
        return {
            "baseline": self.baseline,
            "treatment": self.treatment,
            "pd_delta": self.pd_delta,
            "poi_delta": self.poi_delta,
            "weighted_poi_delta": self.weighted_poi_delta,
            "mit_delta_s": self.mit_delta_s,
            "censored_latency_delta_s": self.censored_latency_delta_s,
            "hp_miss_delta": self.hp_miss_delta,
            "coverage_delta": self.coverage_delta,
            "reward_delta": self.reward_delta,
        }


# =====================================================================
# Belief/prediction/uncertainty online updater
# =====================================================================

class _OnlineChannelModel:
    """
    Lightweight receiver-observable channel model used to feed policies
    with belief / prediction / uncertainty signals.

    - belief: exponential-decay Bayesian-ish activity estimate
    - transition: empirical channel -> channel detection transitions
    - hazard: mean inter-detection interval proxy
    - uncertainty: variance of belief estimates (exploration signal)

    Contains no truth: updated only from scan results.
    """

    def __init__(
        self,
        num_channels: int,
        decay_tau_s: float = 0.05,
        staleness_tau_s: float = 0.05,
    ) -> None:
        self.num_channels = int(num_channels)
        self.decay_tau_s = max(float(decay_tau_s), 1e-9)
        self.staleness_tau_s = max(float(staleness_tau_s), 1e-9)

        self.belief = np.full(num_channels, 0.1, dtype=np.float64)
        self.uncertainty = np.full(num_channels, 1.0, dtype=np.float64)
        self.last_observed_s = np.full(num_channels, np.nan)
        self.last_detection_s = np.full(num_channels, np.nan)
        self.detection_counts = np.zeros(num_channels, dtype=np.float64)
        self.visit_counts = np.zeros(num_channels, dtype=np.float64)
        self.transition_counts = np.zeros(
            (num_channels, num_channels), dtype=np.float64
        )

        self._last_detected_channel: Optional[int] = None

    def observe(
        self,
        *,
        channel: int,
        time_s: float,
        detected: bool,
        dwell_time_s: float,
    ) -> None:

        # Decay belief for every channel to the current time.
        elapsed = np.where(
            np.isnan(self.last_observed_s),
            0.0,
            np.maximum(time_s - self.last_observed_s, 0.0),
        )
        self.belief = self.belief * np.exp(-elapsed / self.decay_tau_s)

        # Update observed channel.
        if detected:
            self.belief[channel] = min(1.0, self.belief[channel] + 0.55)
            self.detection_counts[channel] += 1.0

            if self._last_detected_channel is not None:
                self.transition_counts[
                    self._last_detected_channel, channel
                ] += 1.0
            self._last_detected_channel = int(channel)
            self.last_detection_s[channel] = time_s
        else:
            self.belief[channel] *= 0.75

        self.visit_counts[channel] += 1.0
        self.last_observed_s[channel] = time_s

        # Uncertainty: high for rarely-visited channels.
        self.uncertainty = 1.0 / np.sqrt(self.visit_counts + 1.0)

    # -----------------------------------------------------------------

    def staleness(self, time_s: float) -> np.ndarray:
        values = np.where(
            np.isnan(self.last_observed_s),
            time_s,
            np.maximum(time_s - self.last_observed_s, 0.0),
        )
        return values.astype(np.float64)

    def freshness(self, time_s: float) -> np.ndarray:
        staleness = self.staleness(time_s)
        return np.exp(-staleness / self.staleness_tau_s)

    def transition_prediction(self, current_channel: int) -> np.ndarray:
        row = self.transition_counts[current_channel] + 0.5
        total = row.sum()
        return row / max(total, 1e-9)

    def activity_prediction(self, time_s: float) -> np.ndarray:
        # Belief carried forward one step.
        return self.belief.copy()

    def hazard_prediction(self, time_s: float) -> np.ndarray:
        # Mean inter-detection-interval proxy.
        hazard = np.zeros(self.num_channels, dtype=np.float64)

        for channel in range(self.num_channels):
            last = self.last_detection_s[channel]
            if np.isnan(last):
                continue

            gap = max(time_s - last, 1e-6)
            hazard[channel] = 1.0 / (1.0 + gap)

        return hazard / max(hazard.max(), 1e-9)

    def coverage_pressure(
        self, time_s: float, coverage_window_s: float
    ) -> np.ndarray:
        staleness = self.staleness(time_s)
        # Once a channel misses the configured surveillance deadline, its
        # coverage pressure must be strong enough to compete with an
        # exploitation-heavy detection score.  The previous cap of 2.0 was
        # too weak for short web/API horizons and caused cognitive/STARE/MPC
        # policies to camp on the same detected channel.
        return np.clip(
            staleness / max(coverage_window_s, 1e-9),
            0.0,
            4.0,
        )

    def priority_pressure(self) -> np.ndarray:
        # Receiver-side priority proxy: detection history, NOT truth.
        total = self.detection_counts.sum()
        if total <= 0:
            return np.zeros(self.num_channels, dtype=np.float64)

        return self.detection_counts / total


# =====================================================================
# Tournament harness
# =====================================================================

class TournamentHarness:

    def __init__(
        self,
        scenarios: Optional[Sequence[Scenario]] = None,
        episodes_per_scenario: int = 1,
        bootstrap_samples: int = 2000,
        confidence: float = 0.95,
        seed: int = 12345,
    ):
        self.scenarios = list(scenarios or [])
        self.episodes_per_scenario = int(episodes_per_scenario)
        self.bootstrap_samples = int(bootstrap_samples)
        self.confidence = float(confidence)
        self.seed = int(seed)

    # -----------------------------------------------------------------
    # Environment construction
    # -----------------------------------------------------------------

    @staticmethod
    def _make_environment(
        scenario: Scenario,
        seed: int,
    ) -> ScanEnvironment:

        frequencies = [float(x) for x in scenario.frequencies_mhz]

        if not frequencies:
            raise ValueError(f"Scenario '{scenario.name}' has no channels.")

        dwell_times = scenario.normalized_dwell_times()

        if len(dwell_times) != len(frequencies):
            raise ValueError(
                f"Scenario '{scenario.name}' has "
                f"{len(frequencies)} channels but "
                f"{len(dwell_times)} dwell times."
            )

        # Interactive/API scenarios can provide a mission-specific coverage
        # deadline.  The historical 5 s default is intentionally preserved
        # for research scenarios that do not specify one, but it is far too
        # long for the short 0.5 s web-demo horizon and lets greedy policies
        # camp on one detected channel.
        configured_coverage_window = scenario.metadata.get(
            "coverage_window_s", None
        ) if isinstance(scenario.metadata, dict) else None
        coverage_window_s = (
            float(configured_coverage_window)
            if configured_coverage_window is not None
            else 5.0
        )
        if not np.isfinite(coverage_window_s) or coverage_window_s <= 0:
            raise ValueError(
                f"Scenario '{scenario.name}' has invalid coverage_window_s="
                f"{configured_coverage_window!r}."
            )

        return ScanEnvironment(
            channel_frequencies_mhz=frequencies,
            horizon_s=float(scenario.horizon_s),
            dwell_times_s=dwell_times,
            retune_time_s=float(scenario.retune_time_s),
            coverage_window_s=coverage_window_s,
            seed=int(seed),
        )

    # -----------------------------------------------------------------
    # Observation construction (receiver-observable only)
    # -----------------------------------------------------------------

    @staticmethod
    def _build_observation(
        env: ScanEnvironment,
        model: _OnlineChannelModel,
        builder: SCANStateBuilder,
        detection_density: float,
        mean_snr: float,
        transition_entropy: float,
        unique_channel_ratio: float,
        ood_score: float,
        change_point_score: float,
        stare_embedding: Optional[np.ndarray] = None,
        stare_next_frequency_mhz: float = float("nan"),
    ) -> PolicyObservation:

        time_s = float(env.current_time_s)

        fusion_state = builder.build(
            channel_belief=model.belief.astype(np.float32),
            staleness=model.staleness(time_s).astype(np.float32),
            freshness=model.freshness(time_s).astype(np.float32),
            transition_prediction=model.transition_prediction(
                env.state.current_channel
            ).astype(np.float32),
            activity_prediction=model.activity_prediction(time_s).astype(
                np.float32
            ),
            hazard_prediction=model.hazard_prediction(time_s).astype(
                np.float32
            ),
            predictive_uncertainty=model.uncertainty.astype(np.float32),
            ood_score=ood_score,
            change_point_score=change_point_score,
            coverage_pressure=model.coverage_pressure(
                time_s, env.coverage_window_s
            ).astype(np.float32),
            priority_pressure=model.priority_pressure().astype(np.float32),
            stare_embedding=(
                np.asarray(stare_embedding, dtype=np.float32)
                if stare_embedding is not None
                else None
            ),
            current_channel=int(env.state.current_channel),
            current_frequency_mhz=float(env.state.current_frequency_mhz),
            current_time_s=time_s,
            current_dwell_s=float(env.state.dwell_time_s),
            coverage_fraction=float(env.state.coverage_fraction()),
            detection_density=detection_density,
            mean_snr=mean_snr,
            transition_entropy=transition_entropy,
            unique_channel_ratio=unique_channel_ratio,
        )

        observation = observation_from_fusion_state(fusion_state)
        observation.stare_next_frequency_mhz = float(stare_next_frequency_mhz)
        return observation

    # -----------------------------------------------------------------
    # Episode execution
    # -----------------------------------------------------------------

    def run_episode(
        self,
        scenario: Scenario,
        policy_name: str,
        episode_index: int,
        policy_override: Optional[SchedulerPolicy] = None,
    ) -> EpisodeResult:

        episode_seed = (
            int(scenario.seed)
            + int(episode_index) * 100003
        )

        env = self._make_environment(scenario, episode_seed)

        frequencies = [float(x) for x in scenario.frequencies_mhz]
        dwell_times = scenario.normalized_dwell_times()

        action_space = ScanActionSpace(
            frequencies_mhz=frequencies,
            dwell_times_s=[0.001] + dwell_times,
        )

        builder = SCANStateBuilder(num_channels=len(frequencies))

        model = _OnlineChannelModel(num_channels=len(frequencies))

        env.reset(
            truth_events=[
                dict(event) if isinstance(event, dict) else event
                for event in scenario.truth_events
            ],
            seed=episode_seed,
            start_channel=0,
        )

        observation = self._build_observation(
            env, model, builder,
            detection_density=0.0, mean_snr=0.0,
            transition_entropy=0.0, unique_channel_ratio=0.0,
            ood_score=0.0, change_point_score=0.0,
        )

        # State dimension is derived from the actual observation so
        # learned policies always receive a consistent feature length.
        state_dim = int(observation.flatten().size)

        if policy_override is not None:
            # Ablation/external policies conform to the SAME universal
            # interface and receive the SAME observations.
            policy: SchedulerPolicy = policy_override
        else:
            policy = make_policy(policy_name, action_space, state_dim)

        policy.reset(observation)

        total_reward = 0.0
        detections = 0
        opportunities = 0
        steps = 0

        max_steps = 1_000_000
        previous_time = -1.0

        while steps < max_steps:

            steps += 1

            if env.done or env.current_time_s >= float(scenario.horizon_s):
                break

            # ---------------------------------------------------------
            # UNIVERSAL INTERFACE: policy.act() ONLY.
            # ---------------------------------------------------------

            policy_action: PolicyAction = policy.act(observation)

            channel = int(policy_action.channel)
            dwell = float(policy_action.dwell_time_s)

            if not 0 <= channel < env.num_channels:
                raise ValueError(
                    f"Policy '{policy_name}' selected invalid "
                    f"channel {channel}."
                )

            # ---------------------------------------------------------
            # Step the environment.
            # ---------------------------------------------------------

            from src.environment.scan_environment import ScanAction

            step_result = env.step(
                ScanAction(channel=channel, dwell_time_s=dwell)
            )

            obs_obj = step_result.observation

            reward = float(step_result.reward)
            total_reward += reward
            detections += int(obs_obj.detection_count)
            opportunities += 1

            # ---------------------------------------------------------
            # Update the receiver-observable channel model.
            # ---------------------------------------------------------

            model.observe(
                channel=channel,
                time_s=float(obs_obj.time_s),
                detected=bool(obs_obj.detected),
                dwell_time_s=dwell,
            )

            # Simple OOD proxy: detection surprise.
            ood_score = max(
                0.0,
                min(
                    1.0,
                    abs(float(obs_obj.measured_energy) - model.belief[channel])
                    if obs_obj.detected else 0.0,
                ),
            )

            next_observation = self._build_observation(
                env, model, builder,
                detection_density=float(env.history.detection_rate()),
                mean_snr=float(obs_obj.snr_proxy),
                transition_entropy=float(
                    -np.sum(
                        (model.belief / max(model.belief.sum(), 1e-9))
                        * np.log(
                            (model.belief / max(model.belief.sum(), 1e-9))
                            + 1e-12
                        )
                    )
                ),
                unique_channel_ratio=(
                    float(np.count_nonzero(model.visit_counts))
                    / max(len(frequencies), 1)
                ),
                ood_score=ood_score,
                change_point_score=0.0,
            )

            # ---------------------------------------------------------
            # UNIVERSAL INTERFACE: policy.update().
            # ---------------------------------------------------------

            policy.update(
                observation,
                policy_action,
                reward,
                next_observation,
                info={
                    "detections": int(obs_obj.detection_count),
                    "detected": bool(obs_obj.detected),
                    "time_s": float(obs_obj.time_s),
                    "done": bool(step_result.terminated or step_result.truncated),
                },
            )

            observation = next_observation

            # ---------------------------------------------------------
            # Loop protection
            # ---------------------------------------------------------

            new_time = float(env.current_time_s)

            if previous_time >= 0.0 and new_time <= previous_time:
                raise RuntimeError(
                    f"Environment time failed to advance "
                    f"for policy '{policy_name}' in "
                    f"scenario '{scenario.name}'."
                )

            previous_time = new_time

            if step_result.terminated or step_result.truncated:
                break

        # =============================================================
        # Truth-side evaluation (evaluation-only accessors)
        # =============================================================

        return self._evaluate_episode(
            env=env,
            scenario=scenario,
            policy_name=policy_name,
            episode_index=episode_index,
            episode_seed=episode_seed,
            steps=steps,
            total_reward=total_reward,
            detections=detections,
            opportunities=opportunities,
        )

    # -----------------------------------------------------------------
    # Truth-side evaluation
    # -----------------------------------------------------------------

    def _evaluate_episode(
        self,
        *,
        env: ScanEnvironment,
        scenario: Scenario,
        policy_name: str,
        episode_index: int,
        episode_seed: int,
        steps: int,
        total_reward: float,
        detections: int,
        opportunities: int,
    ) -> EpisodeResult:

        intercepted_ids = set(env.truth_interception_ids())

        # Exact receipts (PHASE 3): real timestamps only.
        receipts = {
            receipt.event_index: receipt
            for receipt in env.interception_receipts()
        }

        events = list(scenario.truth_events)
        n_events = len(events)

        interceptions = len(intercepted_ids)

        pd = (interceptions / n_events) if n_events > 0 else float("nan")

        poi_value = pd

        weighted_numerator = 0.0
        weighted_denominator = 0.0

        hp_total = 0
        hp_missed = 0

        exact_latencies: List[float] = []
        censored_latencies: List[float] = []

        for index, event in enumerate(events):

            if isinstance(event, dict):
                weight = float(
                    event.get("weight", event.get("priority", 1.0))
                )
                priority = float(event.get("priority", 0.0))
                emit_time = event.get(
                    "emit_time_s", event.get("toa_s", event.get("toa"))
                )
            else:
                weight = float(getattr(event, "weight", 1.0))
                priority = float(getattr(event, "priority", 0.0))
                emit_time = getattr(
                    event, "emit_time_s",
                    getattr(event, "toa_s", getattr(event, "toa", None)),
                )

            hit = index in intercepted_ids

            weighted_denominator += max(weight, 0.0)

            if hit:
                weighted_numerator += max(weight, 0.0)

            if priority >= 1.0:
                hp_total += 1
                if not hit:
                    hp_missed += 1

            if emit_time is not None:
                try:
                    emit_time_f = float(emit_time)
                except Exception:
                    continue

                if hit and index in receipts:
                    # Exact MIT from the receipt.
                    receipt = receipts[index]
                    exact_latencies.append(
                        max(
                            0.0,
                            receipt.observation_time_s
                            - receipt.emission_time_s,
                        )
                    )
                elif not hit:
                    # Censored at horizon - no fabricated timestamp.
                    censored_latencies.append(
                        max(0.0, float(scenario.horizon_s) - emit_time_f)
                    )

        weighted_poi = (
            weighted_numerator / weighted_denominator
            if weighted_denominator > 0.0
            else float("nan")
        )

        hp_miss = (
            hp_missed / hp_total if hp_total > 0 else float("nan")
        )

        if exact_latencies:
            lat = np.asarray(exact_latencies, dtype=float)

            # Timestamp-artefact guard: float-ULP differences between
            # emission and observation timestamps (e.g. a pulse landing
            # exactly on scan_end under a closed-window match) produce
            # latencies around 1e-16..1e-17 s. No real detection latency
            # in this simulation is below a nanosecond, so anything
            # under 1e-9 s is an artefact, not a measurement.
            lat = lat[lat > 1e-9]

        if exact_latencies and lat.size > 0:
            mit_s = float(np.mean(lat))
            mit_p50 = float(np.quantile(lat, 0.50))
            mit_p95 = float(np.quantile(lat, 0.95))
            mit_p99 = float(np.quantile(lat, 0.99))
        else:
            mit_s = mit_p50 = mit_p95 = mit_p99 = float("nan")

        censored_latency = (
            float(np.mean(censored_latencies))
            if censored_latencies
            else float("nan")
        )

        # Receiver-side staleness statistics.
        state = env.state
        if state is not None and state.channel_last_observed_s:
            staleness_values = []
            for channel in range(env.num_channels):
                staleness_values.append(state.staleness(channel))
            finite = [s for s in staleness_values if math.isfinite(s)]
            staleness_mean = (
                float(np.mean(finite)) if finite else float("nan")
            )
            staleness_max = (
                float(np.max(finite)) if finite else float("nan")
            )
        else:
            staleness_mean = staleness_max = float("nan")

        coverage = float(state.coverage_fraction()) if state else float("nan")

        return EpisodeResult(
            policy=policy_name,
            scenario=scenario.name,
            episode_index=episode_index,
            seed=episode_seed,

            pd=float(pd),
            poi=float(poi_value),
            weighted_poi=float(weighted_poi),

            mean_interception_time_s=mit_s,
            mit_p50_s=mit_p50,
            mit_p95_s=mit_p95,
            mit_p99_s=mit_p99,

            censored_mean_latency_s=censored_latency,

            hp_miss_rate=float(hp_miss),

            coverage=coverage,

            reward=float(total_reward),

            staleness_mean_s=staleness_mean,
            staleness_max_s=staleness_max,

            detections=int(detections),
            opportunities=int(opportunities),
            interceptions=int(interceptions),

            metadata={
                "steps": steps,
                "channels": len(scenario.frequencies_mhz),
                "dwell_times_s": scenario.normalized_dwell_times(),
                "exact_mit_available": bool(exact_latencies),
                "truth_used_only_for_evaluation": True,
            },
        )

    # -----------------------------------------------------------------
    # Tournament run
    # -----------------------------------------------------------------

    def run(
        self,
        policies: Sequence[str],
    ) -> Dict[str, List[EpisodeResult]]:

        if not self.scenarios:
            raise ValueError("No tournament scenarios were provided.")

        results: Dict[str, List[EpisodeResult]] = {
            policy: [] for policy in policies
        }

        for policy in policies:

            for scenario in self.scenarios:

                for episode_index in range(self.episodes_per_scenario):

                    result = self.run_episode(
                        scenario=scenario,
                        policy_name=policy,
                        episode_index=episode_index,
                    )

                    results[policy].append(result)

        return results

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def _aggregate_metric(
        self,
        episodes: Sequence[EpisodeResult],
        field_name: str,
        seed_offset: int,
    ) -> ConfidenceInterval:

        values = []

        for episode in episodes:

            value = getattr(episode, field_name)

            try:
                value = float(value)
            except Exception:
                continue

            if math.isfinite(value):
                values.append(value)

        return bootstrap_ci(
            values,
            confidence=self.confidence,
            n_bootstrap=self.bootstrap_samples,
            seed=self.seed + seed_offset,
        )

    def summarize(
        self,
        results: Dict[str, List[EpisodeResult]],
    ) -> Dict[str, PolicyResult]:

        summaries = {}

        for policy_index, (policy, episodes) in enumerate(results.items()):

            base = policy_index * 1000

            summaries[policy] = PolicyResult(

                policy=policy,
                episodes=list(episodes),

                pd=self._aggregate_metric(episodes, "pd", base + 1),
                poi=self._aggregate_metric(episodes, "poi", base + 2),
                weighted_poi=self._aggregate_metric(
                    episodes, "weighted_poi", base + 3
                ),

                mit_s=self._aggregate_metric(
                    episodes, "mean_interception_time_s", base + 4
                ),
                mit_p50_s=self._aggregate_metric(
                    episodes, "mit_p50_s", base + 14
                ),
                mit_p95_s=self._aggregate_metric(
                    episodes, "mit_p95_s", base + 15
                ),
                mit_p99_s=self._aggregate_metric(
                    episodes, "mit_p99_s", base + 16
                ),

                censored_latency_s=self._aggregate_metric(
                    episodes, "censored_mean_latency_s", base + 5
                ),

                hp_miss=self._aggregate_metric(
                    episodes, "hp_miss_rate", base + 6
                ),

                coverage=self._aggregate_metric(episodes, "coverage", base + 7),

                reward=self._aggregate_metric(episodes, "reward", base + 8),

                staleness_mean_s=self._aggregate_metric(
                    episodes, "staleness_mean_s", base + 9
                ),

                staleness_max_s=self._aggregate_metric(
                    episodes, "staleness_max_s", base + 10
                ),
            )

        return summaries

    # -----------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------

    def _paired_metric(
        self,
        results: Dict[str, List[EpisodeResult]],
        baseline: str,
        treatment: str,
        field_name: str,
        seed_offset: int,
    ) -> ConfidenceInterval:

        baseline_results = results[baseline]
        treatment_results = results[treatment]

        if len(baseline_results) != len(treatment_results):
            raise ValueError(
                "Paired comparison requires "
                "the baseline and treatment to "
                "have the same number of episodes."
            )

        baseline_values = []
        treatment_values = []

        for b, t in zip(baseline_results, treatment_results):

            try:
                b_value = float(getattr(b, field_name))
                t_value = float(getattr(t, field_name))
            except Exception:
                continue

            if not math.isfinite(b_value):
                continue

            if not math.isfinite(t_value):
                continue

            baseline_values.append(b_value)
            treatment_values.append(t_value)

        return paired_bootstrap_difference(
            baseline_values,
            treatment_values,
            confidence=self.confidence,
            n_bootstrap=self.bootstrap_samples,
            seed=self.seed + seed_offset,
        )

    def compare(
        self,
        results: Dict[str, List[EpisodeResult]],
        baseline: str,
        treatment: str,
    ) -> PolicyComparison:

        if baseline not in results:
            raise KeyError(f"Baseline policy '{baseline}' is not present.")

        if treatment not in results:
            raise KeyError(f"Treatment policy '{treatment}' is not present.")

        return PolicyComparison(

            baseline=baseline,
            treatment=treatment,

            pd_delta=self._paired_metric(results, baseline, treatment, "pd", 101),

            poi_delta=self._paired_metric(results, baseline, treatment, "poi", 102),

            weighted_poi_delta=self._paired_metric(
                results, baseline, treatment, "weighted_poi", 103
            ),

            mit_delta_s=self._paired_metric(
                results, baseline, treatment,
                "mean_interception_time_s", 104,
            ),

            censored_latency_delta_s=self._paired_metric(
                results, baseline, treatment,
                "censored_mean_latency_s", 105,
            ),

            hp_miss_delta=self._paired_metric(
                results, baseline, treatment, "hp_miss_rate", 106
            ),

            coverage_delta=self._paired_metric(
                results, baseline, treatment, "coverage", 107
            ),

            reward_delta=self._paired_metric(
                results, baseline, treatment, "reward", 108
            ),
        )


# =====================================================================
# Development scenarios (PHASE 5, Scenarios A-F)
# =====================================================================

def _events(events):
    return list(events)


def _pulse_train(
    frequency_mhz: float,
    start_s: float,
    stop_s: float,
    pri_s: float,
    priority: float = 1.0,
) -> List[Dict[str, float]]:
    """
    Generate a pulse train (realistic emitter behaviour):
    repeated pulses at a fixed PRI between start and stop.
    """

    events = []

    toa = start_s

    while toa <= stop_s:
        events.append(
            {
                "toa_s": round(toa, 9),
                "frequency_mhz": float(frequency_mhz),
                "priority": float(priority),
                "weight": float(priority),
            }
        )

        toa += pri_s

    return events


def default_development_scenarios() -> List[Scenario]:
    """
    Scenario A - Fixed emitters:       verify detection/receipts
    Scenario B - Sparse emitters:      verify exploration
    Scenario C - Frequency hopping:    verify temporal prediction
    Scenario D - Burst/LPI:            verify rapid reaction
    Scenario E - Dense multi-threat:   verify opportunity cost
    Scenario F - Persistent emitter:   verify coverage/staleness
    Scenario G - Adversarial shift:    verify change-point response
    """

    # A - fixed emitters: two pulse trains (PRI 5 ms)
    fixed_events = _events(
        _pulse_train(1000.0, 0.006, 0.096, 0.005, priority=1.0)
        + _pulse_train(1200.0, 0.008, 0.098, 0.005, priority=0.5)
    )

    # B - sparse: two isolated pulses
    sparse_events = _events([
        {"toa_s": 0.025, "frequency_mhz": 1300.0, "priority": 1.0, "weight": 1.0},
        {"toa_s": 0.070, "frequency_mhz": 1500.0, "priority": 0.5, "weight": 0.5},
    ])

    # C - frequency hopping: alternating pulse trains every 5 ms
    hopping_events = _events(
        _pulse_train(1000.0, 0.006, 0.036, 0.010, priority=1.0)
        + _pulse_train(1200.0, 0.011, 0.041, 0.010, priority=1.0)
        + _pulse_train(1000.0, 0.046, 0.076, 0.010, priority=1.0)
        + _pulse_train(1200.0, 0.051, 0.071, 0.010, priority=1.0)
    )

    # D - burst/LPI: three very short bursts (3 pulses, 1 ms apart)
    burst_events = _events(
        _pulse_train(1400.0, 0.012, 0.014, 0.001, priority=1.0)
        + _pulse_train(1400.0, 0.052, 0.054, 0.001, priority=1.0)
        + _pulse_train(1400.0, 0.092, 0.094, 0.001, priority=1.0)
    )

    # E - dense multi-threat: six emitters with staggered PRIs
    dense_events = _events(
        _pulse_train(1000.0, 0.005, 0.075, 0.010, priority=1.0)
        + _pulse_train(1100.0, 0.007, 0.077, 0.010, priority=0.5)
        + _pulse_train(1200.0, 0.009, 0.079, 0.010, priority=1.0)
        + _pulse_train(1300.0, 0.011, 0.071, 0.020, priority=0.5)
        + _pulse_train(1400.0, 0.013, 0.073, 0.020, priority=1.0)
        + _pulse_train(1500.0, 0.015, 0.075, 0.020, priority=0.25)
    )

    # F - persistent emitter on one channel (coverage/staleness test)
    persistent_events = _events(
        _pulse_train(1000.0, 0.005, 0.115, 0.005, priority=1.0)
    )

    # G - adversarial shift: emitter relocates mid-horizon
    adversarial_events = _events(
        _pulse_train(1000.0, 0.006, 0.030, 0.005, priority=1.0)
        + _pulse_train(1600.0, 0.036, 0.060, 0.005, priority=1.0)
        + _pulse_train(1300.0, 0.076, 0.096, 0.005, priority=1.0)
    )

    return [
        Scenario(
            name="A-Fixed",
            truth_events=fixed_events,
            horizon_s=0.100,
            seed=101,
            frequencies_mhz=[1000.0, 1100.0, 1200.0, 1300.0],
            dwells_s=[0.001],
            retune_time_s=0.0,
            metadata={"type": "development", "purpose": "fixed-detection"},
        ),
        Scenario(
            name="B-Sparse",
            truth_events=sparse_events,
            horizon_s=0.120,
            seed=202,
            frequencies_mhz=[1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
            dwells_s=[0.001],
            retune_time_s=0.0,
            metadata={"type": "development", "purpose": "sparse-exploration"},
        ),
        Scenario(
            name="C-Hopping",
            truth_events=hopping_events,
            horizon_s=0.080,
            seed=303,
            frequencies_mhz=[1000.0, 1100.0, 1200.0, 1300.0],
            dwells_s=[0.001],
            retune_time_s=0.0,
            metadata={"type": "development", "purpose": "frequency-hopping"},
        ),
        Scenario(
            name="D-BurstLPI",
            truth_events=burst_events,
            horizon_s=0.100,
            seed=404,
            frequencies_mhz=[1200.0, 1300.0, 1400.0, 1500.0],
            dwells_s=[0.001],
            retune_time_s=0.0,
            metadata={"type": "development", "purpose": "burst-lpi"},
        ),
        Scenario(
            name="E-Dense",
            truth_events=dense_events,
            horizon_s=0.080,
            seed=505,
            frequencies_mhz=[1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
            dwells_s=[0.001],
            retune_time_s=0.0,
            metadata={"type": "development", "purpose": "dense-multi-threat"},
        ),
        Scenario(
            name="F-Persistent",
            truth_events=persistent_events,
            horizon_s=0.120,
            seed=606,
            frequencies_mhz=[1000.0, 1100.0, 1200.0, 1300.0],
            dwells_s=[0.001],
            retune_time_s=0.0,
            metadata={"type": "development", "purpose": "persistent-coverage"},
        ),
        Scenario(
            name="G-Adversarial",
            truth_events=adversarial_events,
            horizon_s=0.100,
            seed=707,
            frequencies_mhz=[1000.0, 1100.0, 1200.0, 1300.0, 1400.0,
                             1500.0, 1600.0, 1700.0],
            dwells_s=[0.001],
            retune_time_s=0.0,
            metadata={"type": "development", "purpose": "adversarial-shift"},
        ),
    ]


# =====================================================================
# Public API
# =====================================================================

__all__ = [
    "ConfidenceInterval",
    "Scenario",
    "EpisodeResult",
    "PolicyResult",
    "TournamentSummary",
    "PolicyComparison",
    "TournamentHarness",
    "default_development_scenarios",
    "bootstrap_ci",
    "paired_bootstrap_difference",
]
