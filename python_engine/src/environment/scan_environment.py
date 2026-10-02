from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.environment.observation import (
    ObservationHistory,
    PulseObservation,
    ScanObservation,
)
from src.environment.receiver_state import ReceiverState


@dataclass
class ScanAction:
    """
    One receiver action.

    channel:
        Index into the receiver frequency/channel table.

    dwell_time_s:
        How long the receiver observes that channel.
    """

    channel: int
    dwell_time_s: float


@dataclass
class InterceptionReceipt:
    """
    Exact event-level interception receipt (PHASE 3).

    Created only when an emitted event is actually received during a
    scan. MIT = interception_time - emission_time computed ONLY over
    receipt events; misses stay censored (no fabricated timestamps).
    """

    event_index: int
    observation_time_s: float
    emission_time_s: float
    channel: int
    frequency_mhz: float


@dataclass
class ScanStepResult:
    observation: ScanObservation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class ScanEnvironment:
    """
    Receiver-side sequential spectrum scanning environment.

    Important scientific rule:
        The policy receives observations produced by the receiver.
        Ground-truth emitter state is used only internally by the
        environment for evaluation/reward computation.

    The environment follows:

        action
          -> receiver movement
          -> observation
          -> belief/prediction update outside environment
          -> reward/evaluation
          -> next state
    """

    def __init__(
        self,
        channel_frequencies_mhz: Sequence[float],
        horizon_s: float = 30.0,
        dwell_times_s: Optional[Sequence[float]] = None,
        retune_time_s: float = 0.0,
        coverage_window_s: float = 5.0,
        seed: int = 42,
        receiver_observer: Optional[Any] = None,
        reward_function: Optional[Any] = None,
    ) -> None:

        frequencies = np.asarray(
            channel_frequencies_mhz,
            dtype=np.float64,
        )

        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError(
                "channel_frequencies_mhz must be a non-empty 1D sequence."
            )

        self.frequencies_mhz = frequencies
        self.num_channels = int(frequencies.size)

        if dwell_times_s is None:
            self.dwell_times_s = np.full(
                self.num_channels,
                0.001,
                dtype=np.float64,
            )
        else:
            dwell = np.asarray(dwell_times_s, dtype=np.float64)

            if dwell.size != self.num_channels:
                raise ValueError(
                    "dwell_times_s must have one value per channel."
                )

            if np.any(dwell <= 0):
                raise ValueError("All dwell times must be positive.")

            self.dwell_times_s = dwell

        self.horizon_s = float(horizon_s)
        self.retune_time_s = float(retune_time_s)
        self.coverage_window_s = float(coverage_window_s)

        if self.horizon_s <= 0:
            raise ValueError("horizon_s must be positive.")

        self.rng = np.random.default_rng(seed)

        self.receiver_observer = receiver_observer
        self.reward_function = reward_function

        self.state: Optional[ReceiverState] = None
        self.history = ObservationHistory(max_length=512)

        self._truth_events: List[Dict[str, Any]] = []
        self._intercepted_event_ids: set = set()
        self._emitted_event_ids: set = set()
        self._interception_receipts: List[InterceptionReceipt] = []

        self._episode_id = 0

    # ------------------------------------------------------------------
    # Environment lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        truth_events: Optional[Sequence[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        start_channel: int = 0,
    ) -> Dict[str, Any]:

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if not 0 <= start_channel < self.num_channels:
            raise ValueError("start_channel is outside channel range.")

        self.state = ReceiverState(
            current_frequency_mhz=float(
                self.frequencies_mhz[start_channel]
            ),
            current_channel=int(start_channel),
            current_time_s=0.0,
            dwell_time_s=float(
                self.dwell_times_s[start_channel]
            ),
            retune_time_s=self.retune_time_s,
            frequency_min_mhz=float(np.min(self.frequencies_mhz)),
            frequency_max_mhz=float(np.max(self.frequencies_mhz)),
            available_channels_mhz=self.frequencies_mhz.tolist(),
        )

        self.history.clear()

        self._truth_events = list(truth_events or [])
        self._intercepted_event_ids.clear()
        self._emitted_event_ids.clear()
        self._interception_receipts = []

        self._episode_id += 1

        self._validate_truth_events()

        return self.state_dict()

    def step(self, action: ScanAction) -> ScanStepResult:
        if self.state is None:
            raise RuntimeError(
                "Environment has not been reset. Call reset() first."
            )

        self._validate_action(action)

        previous_time = self.state.current_time_s

        # Receiver retuning cost.
        self.state.current_time_s += self.retune_time_s

        # Apply selected dwell.
        self.state.current_channel = int(action.channel)
        self.state.current_frequency_mhz = float(
            self.frequencies_mhz[action.channel]
        )
        self.state.dwell_time_s = float(action.dwell_time_s)

        scan_start = self.state.current_time_s
        scan_end = scan_start + float(action.dwell_time_s)

        observation = self._observe(
            channel=action.channel,
            frequency_mhz=self.frequencies_mhz[action.channel],
            scan_start=scan_start,
            scan_end=scan_end,
        )

        self.state.current_time_s = scan_end
        self.state.total_scans += 1
        self.state.total_observations += 1

        if observation.detected:
            self.state.total_detections += observation.detection_count

        self.state.mark_channel_observed(
            action.channel,
            self.state.current_time_s,
        )

        self.history.append(observation)

        reward, reward_info = self._calculate_reward(
            observation=observation,
            previous_time=previous_time,
        )

        self.state.cumulative_reward += reward

        terminated = self.state.current_time_s >= self.horizon_s
        truncated = False

        info = {
            "episode_id": self._episode_id,
            "time_s": self.state.current_time_s,
            "channel": action.channel,
            "frequency_mhz": float(
                self.frequencies_mhz[action.channel]
            ),
            "dwell_time_s": float(action.dwell_time_s),
            "coverage_fraction": self.state.coverage_fraction(),
            "reward": reward,
            **reward_info,
        }

        return ScanStepResult(
            observation=observation,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    # ------------------------------------------------------------------
    # Receiver observation
    # ------------------------------------------------------------------

    def _observe(
        self,
        *,
        channel: int,
        frequency_mhz: float,
        scan_start: float,
        scan_end: float,
    ) -> ScanObservation:

        if self.receiver_observer is not None:
            return self.receiver_observer(
                channel=channel,
                frequency_mhz=float(frequency_mhz),
                scan_start=float(scan_start),
                scan_end=float(scan_end),
                rng=self.rng,
            )

        # Default deterministic truth-event adapter.
        pulses: List[PulseObservation] = []

        measured_energy = 0.0
        noise_estimate = 1.0

        for event_id, event in enumerate(self._truth_events):

            toa = float(event.get("toa_s", event.get("toa", 0.0)))

            # Half-open window [scan_start, scan_end): a pulse is
            # observable during the scan, but NOT at the closing instant.
            # A closed interval lets pulses landing exactly on scan_end
            # produce zero-latency receipts (MIT underflow artefacts).
            if not scan_start <= toa < scan_end:
                continue

            event_frequency = float(
                event.get(
                    "frequency_mhz",
                    event.get("frequency", frequency_mhz),
                )
            )

            # Frequency matching is deliberately based on receiver
            # bandwidth/channel tolerance rather than emitter identity.
            channel_bandwidth = self._channel_bandwidth(channel)

            if abs(event_frequency - frequency_mhz) > channel_bandwidth / 2.0:
                continue

            pulse = PulseObservation(
                toa_s=toa,
                frequency_mhz=event_frequency,
                pulse_width_s=float(
                    event.get(
                        "pulse_width_s",
                        event.get("pulse_width", 0.0),
                    )
                ),
                aoa_deg=float(
                    event.get(
                        "aoa_deg",
                        event.get("aoa", 0.0),
                    )
                ),
                amplitude=float(
                    event.get(
                        "amplitude",
                        event.get("power", 1.0),
                    )
                ),
                channel=channel,
                detected=True,
            )

            pulses.append(pulse)

            measured_energy += max(0.0, pulse.amplitude)

            self._emitted_event_ids.add(event_id)
            self._intercepted_event_ids.add(event_id)

            # ------------------------------------------------------
            # Exact receipt (PHASE 3):
            #   interception time = end of the scan window in which
            #   the pulse was received (receiver timestamp convention).
            # ------------------------------------------------------
            self._interception_receipts.append(
                InterceptionReceipt(
                    event_index=int(event_id),
                    observation_time_s=float(scan_end),
                    emission_time_s=toa,
                    channel=int(channel),
                    frequency_mhz=float(event_frequency),
                )
            )

        return ScanObservation(
            time_s=float(scan_end),
            channel=int(channel),
            frequency_mhz=float(frequency_mhz),
            dwell_time_s=float(scan_end - scan_start),
            pulses=pulses,
            measured_energy=float(measured_energy),
            noise_estimate=float(noise_estimate),
            receiver_observed=True,
        )

    def _channel_bandwidth(self, channel: int) -> float:
        if self.num_channels <= 1:
            return 1.0

        if channel == 0:
            spacing = abs(
                self.frequencies_mhz[1]
                - self.frequencies_mhz[0]
            )
        elif channel == self.num_channels - 1:
            spacing = abs(
                self.frequencies_mhz[-1]
                - self.frequencies_mhz[-2]
            )
        else:
            spacing = min(
                abs(
                    self.frequencies_mhz[channel]
                    - self.frequencies_mhz[channel - 1]
                ),
                abs(
                    self.frequencies_mhz[channel + 1]
                    - self.frequencies_mhz[channel]
                ),
            )

        return max(float(spacing), 1e-9)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _calculate_reward(
        self,
        *,
        observation: ScanObservation,
        previous_time: float,
    ) -> Tuple[float, Dict[str, float]]:

        if self.reward_function is not None:
            result = self.reward_function(
                observation=observation,
                receiver_state=self.state,
                history=self.history,
            )

            if isinstance(result, tuple):
                reward, info = result
                return float(reward), dict(info)

            return float(result), {}

        # Conservative environment-level default.
        #
        # This is NOT the final research reward.
        # The final reward laboratory implementation should be injected
        # during SCAN training/evaluation.
        interception_reward = float(
            observation.detection_count
        )

        latency_cost = max(
            0.0,
            float(observation.dwell_time_s),
        )

        coverage_bonus = 0.01 * self.state.coverage_fraction()

        reward = (
            interception_reward
            + coverage_bonus
            - 0.01 * latency_cost
        )

        return float(reward), {
            "interception_reward": interception_reward,
            "latency_cost": latency_cost,
            "coverage_bonus": coverage_bonus,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_action(self, action: ScanAction) -> None:
        if not isinstance(action, ScanAction):
            raise TypeError(
                "action must be a ScanAction instance."
            )

        if not 0 <= action.channel < self.num_channels:
            raise ValueError(
                f"channel must be in [0, {self.num_channels - 1}]"
            )

        if not np.isfinite(action.dwell_time_s):
            raise ValueError("dwell_time_s must be finite.")

        if action.dwell_time_s <= 0:
            raise ValueError("dwell_time_s must be positive.")

    def _validate_truth_events(self) -> None:
        for event in self._truth_events:
            if "toa_s" not in event and "toa" not in event:
                raise ValueError(
                    "Every truth event must contain toa_s or toa."
                )

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        if self.state is None:
            raise RuntimeError(
                "Environment has not been reset."
            )

        return {
            "current_frequency_mhz": self.state.current_frequency_mhz,
            "current_channel": self.state.current_channel,
            "current_time_s": self.state.current_time_s,
            "dwell_time_s": self.state.dwell_time_s,
            "retune_time_s": self.state.retune_time_s,
            "frequency_min_mhz": self.state.frequency_min_mhz,
            "frequency_max_mhz": self.state.frequency_max_mhz,
            "total_scans": self.state.total_scans,
            "total_observations": self.state.total_observations,
            "total_detections": self.state.total_detections,
            "coverage_fraction": self.state.coverage_fraction(),
            "mission_mode": self.state.mission_mode,
            "cumulative_reward": self.state.cumulative_reward,
        }

    @property
    def current_time_s(self) -> float:
        if self.state is None:
            return 0.0

        return self.state.current_time_s

    @property
    def done(self) -> bool:
        return self.state is not None and (
            self.state.current_time_s >= self.horizon_s
        )

    @property
    def available_actions(self) -> List[ScanAction]:
        return [
            ScanAction(
                channel=i,
                dwell_time_s=float(self.dwell_times_s[i]),
            )
            for i in range(self.num_channels)
        ]

    def truth_interception_ids(self) -> set:
        """
        Evaluation-only accessor.

        Policies must not call this method.
        """
        return set(self._intercepted_event_ids)

    def interception_receipts(self) -> List[InterceptionReceipt]:
        """
        Evaluation-only accessor.

        Exact receipts for successfully intercepted events. Policies
        must never consume these; they are for the evaluation engine so
        MIT is computed from real observation timestamps only.
        """
        return list(self._interception_receipts)
