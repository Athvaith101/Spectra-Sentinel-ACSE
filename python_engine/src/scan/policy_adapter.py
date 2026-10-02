"""
SIH26055 V5 - PHASE 1 / PHASE 4
Universal scheduler policy interface.

Every scheduling policy (heuristic, bandit, RL, MPC, STARE-assisted,
portfolio) is wrapped into ONE interface:

    reset(observation)
    act(observation) -> action
    update(observation, action, reward, next_observation, info)

The tournament communicates ONLY through this interface and never
inspects the internal policy type. Ground-truth emitter state is never
visible to any adapter.

Observation contract (PolicyObservation)
----------------------------------------
The adapter layer converts the receiver-observable FusionState into a
plain dataclass so policies never depend on fusion internals either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.fusion.state_builder import FusionState
from src.scan.action_space import ScanAction, ScanActionSpace
from src.scan.scheduler import (
    CognitiveGreedyScheduler,
    InformationGreedyScheduler,
    StalenessScheduler,
)


# =====================================================================
# Observation passed to every policy
# =====================================================================

@dataclass
class PolicyObservation:
    """
    Receiver-observable observation handed to every policy.

    Contains no hidden emitter truth:
      - no emitter IDs
      - no future frequencies
      - no future activity
      - no true priority (priority_pressure is a receiver-side estimate)
      - no ground-truth interception results
    """

    channel_belief: np.ndarray
    staleness: np.ndarray
    freshness: np.ndarray
    transition_prediction: np.ndarray
    activity_prediction: np.ndarray
    hazard_prediction: np.ndarray
    predictive_uncertainty: np.ndarray
    coverage_pressure: np.ndarray
    priority_pressure: np.ndarray

    current_channel: int
    current_time_s: float
    current_dwell_s: float
    coverage_fraction: float

    ood_score: float = 0.0
    change_point_score: float = 0.0
    detection_density: float = 0.0
    mean_snr: float = 0.0
    transition_entropy: float = 0.0
    unique_channel_ratio: float = 0.0

    stare_embedding: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    # Receiver-observable temporal prediction produced by the frozen STARE
    # model. NaN means that the temporal model has not warmed up yet.
    stare_next_frequency_mhz: float = float("nan")

    available_channels: Optional[np.ndarray] = None

    def flatten(self) -> np.ndarray:
        """
        Deterministic feature vector for learned policies (DQN/PPO).
        Layout is fixed for all policies and episodes.
        """
        pieces = [
            np.asarray([self.current_channel, self.current_time_s,
                        self.current_dwell_s, self.coverage_fraction,
                        self.ood_score, self.change_point_score,
                        self.detection_density, self.mean_snr,
                        self.transition_entropy, self.unique_channel_ratio],
                       dtype=np.float32),
            self.channel_belief, self.staleness, self.freshness,
            self.transition_prediction, self.activity_prediction,
            self.hazard_prediction, self.predictive_uncertainty,
            self.coverage_pressure, self.priority_pressure,
        ]
        vector = np.concatenate(
            [np.nan_to_num(np.asarray(p, dtype=np.float32).reshape(-1),
                           nan=0.0, posinf=1e6, neginf=-1e6) for p in pieces]
        )
        return vector.astype(np.float32, copy=False)


@dataclass
class PolicyAction:
    """One policy decision: channel + dwell seconds."""

    channel: int
    dwell_time_s: float

    score: float = 0.0
    reason: str = ""
    diagnostics: Dict[str, float] = field(default_factory=dict)


# =====================================================================
# Universal interface
# =====================================================================

class SchedulerPolicy:
    """
    Universal policy interface (PHASE 1).

    Every scheduler MUST conform to this interface. The tournament
    calls ONLY reset/act/update and never reaches into internals.
    """

    name: str = "policy"

    def reset(self, observation: Optional[PolicyObservation] = None) -> None:
        raise NotImplementedError

    def act(self, observation: PolicyObservation) -> PolicyAction:
        raise NotImplementedError

    def update(
        self,
        observation: PolicyObservation,
        action: PolicyAction,
        reward: float,
        next_observation: PolicyObservation,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        return None


# =====================================================================
# Helper: convert FusionState -> PolicyObservation
# =====================================================================

def observation_from_fusion_state(
    state: FusionState,
    available_channels: Optional[Sequence[int]] = None,
) -> PolicyObservation:
    return PolicyObservation(
        channel_belief=np.asarray(state.channel_belief, dtype=np.float32),
        staleness=np.asarray(state.staleness, dtype=np.float32),
        freshness=np.asarray(state.freshness, dtype=np.float32),
        transition_prediction=np.asarray(
            state.transition_prediction, dtype=np.float32),
        activity_prediction=np.asarray(
            state.activity_prediction, dtype=np.float32),
        hazard_prediction=np.asarray(state.hazard_prediction, dtype=np.float32),
        predictive_uncertainty=np.asarray(
            state.predictive_uncertainty, dtype=np.float32),
        coverage_pressure=np.asarray(state.coverage_pressure, dtype=np.float32),
        priority_pressure=np.asarray(state.priority_pressure, dtype=np.float32),
        current_channel=int(state.current_channel),
        current_time_s=float(state.current_time_s),
        current_dwell_s=float(state.current_dwell_s),
        coverage_fraction=float(state.coverage_fraction),
        ood_score=float(state.ood_score),
        change_point_score=float(state.change_point_score),
        detection_density=float(state.detection_density),
        mean_snr=float(state.mean_snr),
        transition_entropy=float(state.transition_entropy),
        unique_channel_ratio=float(state.unique_channel_ratio),
        stare_embedding=np.asarray(state.stare_embedding, dtype=np.float32),
        available_channels=(
            np.asarray(list(available_channels), dtype=np.int64)
            if available_channels is not None
            else None
        ),
    )


# =====================================================================
# Coverage-deadline gate
# =====================================================================

def _coverage_candidate_channels(
    observation: PolicyObservation,
    n_channels: int,
) -> List[int]:
    """Return overdue receiver-observable channels when a coverage deadline is active.

    This is deliberately a gate, not a hidden-truth rule: it uses only the
    policy observation's coverage pressure.  When no channel is overdue, all
    channels remain eligible for exploitation.
    """
    pressure = np.asarray(observation.coverage_pressure, dtype=np.float64).reshape(-1)
    pressure = pressure[:n_channels]
    overdue = np.flatnonzero(pressure >= 1.0)
    if overdue.size:
        return [int(i) for i in overdue]
    return list(range(n_channels))


# =====================================================================
# Adapter for score-based BaseScheduler implementations
# =====================================================================

class ScoredSchedulerAdapter(SchedulerPolicy):
    """
    Adapter for BaseScheduler subclasses exposing score_action().

    Sequential / Staleness / Information / Cognitive / FunctionScheduler
    all route through here.
    """

    def __init__(
        self,
        scheduler: Any,
        action_space: ScanActionSpace,
        name: str,
    ) -> None:
        self.scheduler = scheduler
        self.action_space = action_space
        self.name = name

    def reset(self, observation=None) -> None:
        reset = getattr(self.scheduler, "reset", None)
        if callable(reset):
            try:
                reset()
                return
            except TypeError:
                pass
        # SequentialScheduler carries round-robin state.
        if hasattr(self.scheduler, "_next_channel"):
            self.scheduler._next_channel = 0

    def act(self, observation: PolicyObservation) -> PolicyAction:
        channels = self._candidate_channels(observation)

        best_channel = channels[0]
        best_dwell = float(self.action_space.dwell_times_s[0])
        best_score = -np.inf

        for channel in channels:
            for dwell_index in range(self.action_space.num_dwells):
                action = self.action_space.action(channel, dwell_index)
                score = float(
                    self.scheduler.score_action(action, None)
                    if False
                    else self._score(action, observation)
                )
                if score > best_score:
                    best_score = score
                    best_channel = channel
                    best_dwell = float(action.dwell_time_s)

        return PolicyAction(
            channel=int(best_channel),
            dwell_time_s=best_dwell,
            score=float(best_score),
            reason=f"{self.name}:argmax_score",
        )

    def _score(self, action: ScanAction, observation: PolicyObservation) -> float:
        """
        Evaluate the wrapped scheduler's score_action with a synthetic
        FusionState-like view. BaseScheduler implementations only read
        per-channel arrays plus a few scalars, so we expose the
        observation arrays directly.
        """
        view = _ScoreView(observation)
        return float(self.scheduler.score_action(action, view))

    def _candidate_channels(
        self, observation: PolicyObservation
    ) -> List[int]:
        return _coverage_candidate_channels(
            observation, self.action_space.num_channels
        )


class _ScoreView:
    """
    Minimal read-only view exposing the array attributes that
    BaseScheduler.score_action implementations access on FusionState:
    staleness, predictive_uncertainty, coverage_pressure, channel_belief,
    transition_prediction, activity_prediction, hazard_prediction,
    priority_pressure, plus ood_active / change_point_active flags.
    """

    def __init__(self, observation: PolicyObservation) -> None:
        self.channel_belief = observation.channel_belief
        self.staleness = observation.staleness
        self.freshness = observation.freshness
        self.transition_prediction = observation.transition_prediction
        self.activity_prediction = observation.activity_prediction
        self.hazard_prediction = observation.hazard_prediction
        self.predictive_uncertainty = observation.predictive_uncertainty
        self.coverage_pressure = observation.coverage_pressure
        self.priority_pressure = observation.priority_pressure
        self.ood_score = observation.ood_score
        self.change_point_score = observation.change_point_score
        self.ood_active = observation.ood_score > 0.5
        self.change_point_active = observation.change_point_score > 0.5


# =====================================================================
# Sequential adapter (round-robin must NOT be rescored every step)
# =====================================================================

class SequentialAdapter(SchedulerPolicy):
    """Pure round-robin over channels with the first dwell time."""

    def __init__(
        self,
        action_space: ScanActionSpace,
        name: str = "Sequential",
    ) -> None:
        self.action_space = action_space
        self.name = name
        self._next_channel = 0

    def reset(self, observation=None) -> None:
        self._next_channel = 0

    def act(self, observation: PolicyObservation) -> PolicyAction:
        channel = self._next_channel
        self._next_channel = (
            self._next_channel + 1
        ) % self.action_space.num_channels
        return PolicyAction(
            channel=channel,
            dwell_time_s=float(self.action_space.dwell_times_s[0]),
            score=0.0,
            reason="round_robin",
        )

    def update(self, *args, **kwargs) -> None:
        return None


class RoundRobinDwellAdapter(SequentialAdapter):
    """Round-robin cycling channels AND dwell indices."""

    def __init__(self, action_space: ScanActionSpace) -> None:
        super().__init__(action_space, name="RoundRobinDwell")
        self._dwell_index = 0

    def act(self, observation: PolicyObservation) -> PolicyAction:
        channel = self._next_channel
        dwell_index = self._dwell_index

        self._next_channel = (
            self._next_channel + 1
        ) % self.action_space.num_channels
        self._dwell_index = (
            dwell_index + 1
        ) % self.action_space.num_dwells

        action = self.action_space.action(channel, dwell_index)
        return PolicyAction(
            channel=int(action.channel),
            dwell_time_s=float(action.dwell_time_s),
            score=0.0,
            reason="round_robin_dwell",
        )


# =====================================================================
# Bandit adapters
# =====================================================================

class UCB1Adapter(SchedulerPolicy):
    """
    UCB1 over channels.

    Reward signal = observed detection energy on the chosen channel.
    Bandits NEVER see future rewards: update() receives only the reward
    of the action just taken.
    """

    def __init__(
        self,
        action_space: ScanActionSpace,
        alpha: float = 2.0,
        seed: int = 42,
        name: str = "UCB1",
    ) -> None:
        from src.scan.bandits.ucb import UCB1

        self.action_space = action_space
        self.bandit = UCB1(n_actions=action_space.num_channels, alpha=alpha)
        self.name = name
        self._seed = int(seed)

    def reset(self, observation=None) -> None:
        from src.scan.bandits.ucb import UCB1

        self.bandit = UCB1(
            n_actions=self.action_space.num_channels,
            alpha=self.bandit.alpha,
        )

    def act(self, observation: PolicyObservation) -> PolicyAction:
        channel = self.bandit.select()
        return PolicyAction(
            channel=int(channel),
            dwell_time_s=float(self.action_space.dwell_times_s[0]),
            score=float(self.bandit.values[channel]),
            reason="ucb1",
        )

    def update(self, observation, action, reward, next_observation, info=None):
        # Bandits never see future rewards: update uses only the reward
        # of the action just taken.
        self.bandit.update(action.channel, float(reward))


class LinUCBAdapter(SchedulerPolicy):
    """Contextual LinUCB; context = flattened receiver observation."""

    def __init__(
        self,
        action_space: ScanActionSpace,
        context_dim: int,
        alpha: float = 1.0,
        seed: int = 42,
        name: str = "LinUCB",
    ) -> None:
        from src.scan.bandits.linucb import LinUCB

        self.action_space = action_space
        self.context_dim = int(context_dim)
        self.bandit = LinUCB(
            n_actions=action_space.num_channels,
            context_dim=context_dim,
            alpha=alpha,
        )
        self.name = name

    def reset(self, observation=None) -> None:
        from src.scan.bandits.linucb import LinUCB

        self.bandit = LinUCB(
            n_actions=self.action_space.num_channels,
            context_dim=self.context_dim,
            alpha=self.bandit.alpha,
        )

    def act(self, observation: PolicyObservation) -> PolicyAction:
        context = observation.flatten()
        channel, scores = self.bandit.select(context)
        return PolicyAction(
            channel=int(channel),
            dwell_time_s=float(self.action_space.dwell_times_s[0]),
            score=float(scores[channel]),
            reason="linucb",
        )

    def update(self, observation, action, reward, next_observation, info=None):
        self.bandit.update(
            action.channel,
            observation.flatten(),
            float(reward),
        )


class ThompsonAdapter(SchedulerPolicy):
    """Gaussian Thompson Sampling over channels."""

    def __init__(
        self,
        action_space: ScanActionSpace,
        seed: int = 42,
        name: str = "Thompson",
    ) -> None:
        from src.scan.bandits.thompson import GaussianThompsonSampling

        self.action_space = action_space
        self.bandit = GaussianThompsonSampling(
            n_actions=action_space.num_channels,
            context_dim=1,
            seed=seed,
        )
        self.name = name
        self._seed = int(seed)

    def reset(self, observation=None) -> None:
        from src.scan.bandits.thompson import GaussianThompsonSampling

        self.bandit = GaussianThompsonSampling(
            n_actions=self.action_space.num_channels,
            context_dim=1,
            seed=self._seed,
        )

    def act(self, observation: PolicyObservation) -> PolicyAction:
        # Non-contextual Thompson: use a constant context of 1.
        channel = self.bandit.select(np.ones(1, dtype=np.float64))
        return PolicyAction(
            channel=int(channel),
            dwell_time_s=float(self.action_space.dwell_times_s[0]),
            score=float(self._posterior_mean(channel)),
            reason="thompson",
        )

    def _posterior_mean(self, channel: int) -> float:
        import numpy as _np

        A_inv = _np.linalg.pinv(self.bandit.A[channel])
        return float((A_inv @ self.bandit.b[channel]).item())

    def update(self, observation, action, reward, next_observation, info=None):
        self.bandit.update(
            action.channel,
            np.ones(1, dtype=np.float64),
            float(reward),
        )


# =====================================================================
# Markov / predictive adapter
# =====================================================================

class MarkovAdapter(SchedulerPolicy):
    """
    Online first-order Markov channel predictor.

    Predicts P(next channel | last channel) from observed detections and
    picks the highest-probability channel that is not the current one
    (exploring elsewhere only when belief is flat).
    """

    def __init__(
        self,
        action_space: ScanActionSpace,
        smoothing: float = 1.0,
        name: str = "Markov",
    ) -> None:
        self.action_space = action_space
        self.smoothing = float(smoothing)
        self.name = name
        self._transitions: Dict[int, Dict[int, float]] = {}
        self._last_detected_channel: Optional[int] = None

    def reset(self, observation=None) -> None:
        self._transitions = {}
        self._last_detected_channel = None

    def act(self, observation: PolicyObservation) -> PolicyAction:
        current = observation.current_channel

        counts = np.full(
            self.action_space.num_channels,
            self.smoothing,
            dtype=np.float64,
        )

        row = self._transitions.get(current)
        if row:
            for target, count in row.items():
                counts[target] += count

        probabilities = counts / counts.sum()

        # Prefer predicted-active channels; avoid sitting on the current
        # channel unless it dominates.
        scores = probabilities.copy()
        scores[current] *= 0.5

        channel = int(np.argmax(scores))

        return PolicyAction(
            channel=channel,
            dwell_time_s=float(self.action_space.dwell_times_s[0]),
            score=float(probabilities[channel]),
            reason="markov_prediction",
        )

    def update(self, observation, action, reward, next_observation, info=None):
        # Learn transitions only from receiver-observable detections.
        detected = bool(info and info.get("detections", 0) > 0)

        if detected:
            source = self._last_detected_channel
            if source is not None and source != action.channel:
                row = self._transitions.setdefault(source, {})
                row[action.channel] = row.get(action.channel, 0.0) + 1.0

            self._last_detected_channel = int(action.channel)


# =====================================================================
# RL adapters (DQN family, PPO)
# =====================================================================

class _RLPolicyBase(SchedulerPolicy):
    """
    Shared logic for torch-based RL policies.

    Each RL policy owns:
      - its own network(s)
      - its own replay buffer / rollout storage
      - its own random seed
      - its own epsilon schedule

    Training happens online from ITS OWN rollouts only (PHASE 16 rule:
    no sharing trajectories between algorithms).
    """

    def __init__(
        self,
        action_space: ScanActionSpace,
        state_dim: int,
        seed: int = 42,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 2000,
        batch_size: int = 64,
        train_start: int = 200,
        train_frequency: int = 4,
        target_update_frequency: int = 200,
        device: str = "cpu",
    ) -> None:
        import torch

        self.action_space = action_space
        self.state_dim = int(state_dim)
        self.n_actions = action_space.num_channels
        self.seed = int(seed)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = max(1, int(epsilon_decay_steps))
        self.batch_size = int(batch_size)
        self.train_start = int(train_start)
        self.train_frequency = int(train_frequency)
        self.target_update_frequency = int(target_update_frequency)
        self.device = torch.device(device)

        self.rng = np.random.default_rng(seed)
        self.torch_generator = torch.Generator().manual_seed(seed)

        self._steps = 0
        self._replay: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._replay_capacity = 50_000
        self._replay_position = 0
        self._loss: Optional[float] = None
        self._mode = "train"

    # ---- train/eval mode --------------------------------------------

    def set_mode(self, mode: str) -> None:
        """'train' enables exploration + learning; 'eval' is greedy/frozen."""
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'.")
        self._mode = mode

    @property
    def training(self) -> bool:
        return self._mode == "train"

    # ---- weight checkpointing ----------------------------------------

    _MODULE_ATTRS = ("online", "target", "network")
    _OPTIMIZER_ATTRS = ("optimizer",)

    def save_weights(self, path) -> Dict[str, Any]:
        """
        Persist the actual network weights to a .pt checkpoint.

        The JSON sidecar keeps training lineage only; without the .pt
        the checkpoint contains no model at all.
        """
        import torch

        state_dicts = {}
        for attr in self._MODULE_ATTRS:
            module = getattr(self, attr, None)
            if isinstance(module, torch.nn.Module):
                state_dicts[attr] = module.state_dict()

        optimizers = {}
        for attr in self._OPTIMIZER_ATTRS:
            opt = getattr(self, attr, None)
            if opt is not None:
                optimizers[attr] = opt.state_dict()

        payload = {
            "format": "scan-rl-weights-v1",
            "policy_class": type(self).__name__,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "seed": self.seed,
            "gamma": self.gamma,
            "steps": self._steps,
            "epsilon": self.epsilon,
            "state_dicts": state_dicts,
            "optimizers": optimizers,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return payload

    def load_weights(self, path) -> None:
        """Restore network weights saved by save_weights()."""
        import torch

        payload = torch.load(path, map_location=self.device)

        for attr, state_dict in payload.get("state_dicts", {}).items():
            module = getattr(self, attr, None)
            if isinstance(module, torch.nn.Module):
                module.load_state_dict(state_dict)

        if "optimizer" in payload.get("optimizers", {}):
            opt = getattr(self, "optimizer", None)
            if opt is not None:
                opt.load_state_dict(payload["optimizers"]["optimizer"])

        self._steps = int(payload.get("steps", 0))

    # ---- replay -----------------------------------------------------

    def _remember(self, state, action, reward, next_state, done) -> None:
        transition = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )

        if len(self._replay) < self._replay_capacity:
            self._replay.append(transition)
        else:
            self._replay[self._replay_position] = transition

        self._replay_position = (
            self._replay_position + 1
        ) % self._replay_capacity

    def _sample_batch(self):
        if len(self._replay) < self.batch_size:
            return None

        indices = self.rng.choice(
            len(self._replay),
            size=self.batch_size,
            replace=False,
        )

        batch = [self._replay[int(i)] for i in indices]

        import torch

        states = torch.tensor(
            np.stack([b[0] for b in batch]),
            device=self.device,
        )
        actions = torch.tensor(
            [b[1] for b in batch],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            [b[2] for b in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.tensor(
            np.stack([b[3] for b in batch]),
            device=self.device,
        )
        dones = torch.tensor(
            [b[4] for b in batch],
            dtype=torch.float32,
            device=self.device,
        )

        return states, actions, rewards, next_states, dones

    # ---- exploration ------------------------------------------------

    def _decay_epsilon(self) -> None:
        fraction = min(
            1.0,
            self._steps / self.epsilon_decay_steps,
        )
        self.epsilon = self.epsilon_start + fraction * (
            self.epsilon_end - self.epsilon_start
        )

    def _maybe_select_random(self) -> bool:
        return bool(self.rng.random() < self.epsilon)

    # ---- interface --------------------------------------------------

    def reset(self, observation=None) -> None:
        # Network weights persist across episodes; exploration state too.
        return None

    def act(self, observation: PolicyObservation) -> PolicyAction:
        self._steps += 1

        if self.training:
            self._decay_epsilon()

            if self._maybe_select_random():
                channel = int(
                    self.rng.integers(0, self.n_actions)
                )
                return PolicyAction(
                    channel=channel,
                    dwell_time_s=float(self.action_space.dwell_times_s[0]),
                    score=0.0,
                    reason="explore",
                    diagnostics={"epsilon": self.epsilon},
                )

        channel = self._greedy_channel(observation)
        return PolicyAction(
            channel=int(channel),
            dwell_time_s=float(self.action_space.dwell_times_s[0]),
            score=0.0,
            reason="greedy",
            diagnostics={"epsilon": self.epsilon},
        )

    def _greedy_channel(self, observation: PolicyObservation) -> int:
        raise NotImplementedError

    def update(self, observation, action, reward, next_observation, info=None):
        if not self.training:
            return None

        state = observation.flatten()
        next_state = next_observation.flatten()
        done = bool(info and info.get("done", False))

        self._remember(state, action.channel, reward, next_state, done)
        self._train_step()
        self._post_update(action.channel)

    def _train_step(self) -> None:
        pass

    def _post_update(self, channel: int) -> None:
        pass


class DQNAdapter(_RLPolicyBase):
    """Vanilla DQN with target network."""

    def __init__(self, action_space, state_dim, **kwargs) -> None:
        super().__init__(action_space, state_dim, **kwargs)

        import torch
        from src.scan.rl.dqn import QNetwork, dqn_target

        self.torch = torch
        self._dqn_target = dqn_target

        self.online = QNetwork(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
        ).to(self.device)

        self.target = QNetwork(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
        ).to(self.device)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.AdamW(
            self.online.parameters(), lr=kwargs.get("learning_rate", 1e-3)
        )

        self._update_count = 0

    def _greedy_channel(self, observation: PolicyObservation) -> int:
        import torch

        with torch.no_grad():
            state = torch.tensor(
                observation.flatten(),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            q_values = self.online(state)

        return int(torch.argmax(q_values, dim=1).item())

    def _train_step(self) -> None:
        if self._steps % self.train_frequency != 0:
            return

        batch = self._sample_batch()
        if batch is None:
            return

        import torch

        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            next_q = self.target(next_states)
            target_q = self._dqn_target(rewards, next_q, dones, self.gamma)

        current_q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.torch.nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()

        self._loss = float(loss.detach())
        self._update_count += 1

        if self._update_count % self.target_update_frequency == 0:
            self.target.load_state_dict(self.online.state_dict())

    def _post_update(self, channel: int) -> None:
        return None


class DoubleDQNAdapter(DQNAdapter):
    """Double DQN: online selects, target evaluates."""

    def __init__(self, action_space, state_dim, **kwargs) -> None:
        super().__init__(action_space, state_dim, **kwargs)

        from src.scan.rl.dqn import double_dqn_target

        self._double_target = double_dqn_target

    def _train_step(self) -> None:
        if self._steps % self.train_frequency != 0:
            return

        batch = self._sample_batch()
        if batch is None:
            return

        import torch

        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            online_next_q = self.online(next_states)
            target_next_q = self.target(next_states)
            target_q = self._double_target(
                rewards, online_next_q, target_next_q, dones, self.gamma
            )

        current_q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.torch.nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()

        self._loss = float(loss.detach())
        self._update_count += 1

        if self._update_count % self.target_update_frequency == 0:
            self.target.load_state_dict(self.online.state_dict())


class DuelingDQNAdapter(DQNAdapter):
    """Dueling architecture (value + advantage)."""

    def __init__(self, action_space, state_dim, **kwargs) -> None:
        super().__init__(action_space, state_dim, **kwargs)

        from src.scan.rl.dqn import DuelingQNetwork

        self.online = DuelingQNetwork(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
        ).to(self.device)

        self.target = DuelingQNetwork(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
        ).to(self.device)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()


class DuelingDoubleDQNAdapter(DoubleDQNAdapter, DuelingDQNAdapter):
    """Dueling Double DQN."""

    def __init__(self, action_space, state_dim, **kwargs) -> None:
        # Explicitly chain both mixins without double-init.
        _RLPolicyBase.__init__(self, action_space, state_dim, **kwargs)

        import torch
        from src.scan.rl.dqn import (
            DuelingQNetwork,
            double_dqn_target,
        )

        self.torch = torch
        self._double_target = double_dqn_target

        self.online = DuelingQNetwork(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
        ).to(self.device)

        self.target = DuelingQNetwork(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
        ).to(self.device)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.AdamW(
            self.online.parameters(), lr=kwargs.get("learning_rate", 1e-3)
        )

        self._update_count = 0


class ReinforceAdapter(_RLPolicyBase):
    """
    Policy-gradient baseline (REINFORCE with baseline critic).

    Used where a full PPO is too heavy for the smoke stage; PPO is
    provided separately.
    """

    def __init__(self, action_space, state_dim, **kwargs) -> None:
        super().__init__(action_space, state_dim, **kwargs)

        import torch
        from src.scan.rl.ppo import PPOActorCritic

        self.torch = torch

        self.network = PPOActorCritic(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=kwargs.get("learning_rate", 3e-4)
        )

        self._log_probs: List[float] = []
        self._rewards: List[float] = []

    def reset(self, observation=None) -> None:
        # Episode boundary: finish previous trajectory.
        self._finish_episode()
        self._log_probs = []
        self._rewards = []

    def act(self, observation: PolicyObservation) -> PolicyAction:
        import torch

        self._steps += 1

        state = torch.tensor(
            observation.flatten(),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        logits, _ = self.network(state)

        distribution = torch.distributions.Categorical(logits=logits)

        if self.training and self.rng.random() < self.epsilon:
            action_index = int(self.rng.integers(0, self.n_actions))
        else:
            action_index = int(distribution.sample().item())

        log_prob = distribution.log_prob(
            torch.tensor(action_index, device=self.device)
        )

        self._log_probs.append(float(log_prob.detach()))
        self._last_state = observation.flatten()

        return PolicyAction(
            channel=action_index,
            dwell_time_s=float(self.action_space.dwell_times_s[0]),
            score=float(log_prob.detach()),
            reason="policy_gradient",
        )

    def update(self, observation, action, reward, next_observation, info=None):
        if not self.training:
            return None

        self._rewards.append(float(reward))
        self._train_step()

    def _train_step(self) -> None:
        if len(self._rewards) < 8 or len(self._rewards) % 16 != 0:
            return

        import torch

        returns = []
        running = 0.0

        for reward in reversed(self._rewards):
            running = reward + self.gamma * running
            returns.append(running)

        returns.reverse()

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (
                returns.std() + 1e-8
            )

        log_probs = torch.tensor(
            self._log_probs[-len(returns):],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self._loss = float(loss.detach())

    def _finish_episode(self) -> None:
        return None


class PPOAdapter(_RLPolicyBase):
    """
    PPO with clipped objective and GAE.

    Independent rollouts, independent buffer, independent seed.
    """

    def __init__(
        self,
        action_space,
        state_dim,
        clip_epsilon: float = 0.2,
        ppo_epochs: int = 2,
        rollout_length: int = 64,
        entropy_bonus: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(action_space, state_dim, **kwargs)

        import torch
        from src.scan.rl.ppo import PPOActorCritic

        self.torch = torch
        self.clip_epsilon = float(clip_epsilon)
        self.ppo_epochs = int(ppo_epochs)
        self.rollout_length = int(rollout_length)
        self.entropy_bonus = float(entropy_bonus)

        self.network = PPOActorCritic(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=kwargs.get("learning_rate", 3e-4)
        )

        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._log_probs: List[float] = []
        self._rewards: List[float] = []
        self._values: List[float] = []
        self._dones: List[bool] = []

    def reset(self, observation=None) -> None:
        return None

    def act(self, observation: PolicyObservation) -> PolicyAction:
        import torch

        self._steps += 1

        state = observation.flatten()

        with torch.no_grad():
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            logits, value = self.network(state_tensor)

        distribution = torch.distributions.Categorical(logits=logits)

        if self.training and self.rng.random() < self.epsilon * 0.2:
            action_index = int(self.rng.integers(0, self.n_actions))
        else:
            action_index = int(distribution.sample().item())

        log_prob = float(distribution.log_prob(
            torch.tensor(action_index, device=self.device)
        ))

        self._states.append(state)
        self._actions.append(action_index)
        self._log_probs.append(log_prob)
        self._values.append(float(value.item()))
        self._rewards.append(0.0)
        self._dones.append(False)

        self._last_log_prob = log_prob

        return PolicyAction(
            channel=action_index,
            dwell_time_s=float(self.action_space.dwell_times_s[0]),
            score=log_prob,
            reason="ppo_sample",
        )

    def update(self, observation, action, reward, next_observation, info=None):
        if not self.training:
            return None

        if self._rewards:
            self._rewards[-1] = float(reward)

        done = bool(info and info.get("done", False))
        if self._dones:
            self._dones[-1] = done

        if len(self._rewards) >= self.rollout_length:
            self._ppo_update()

    def _ppo_update(self) -> None:
        if len(self._rewards) < 8:
            return

        import torch

        with torch.no_grad():
            next_state = torch.tensor(
                self._states[-1], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            _, next_value = self.network(next_state)

        advantages, returns = self._gae(
            self._rewards,
            self._values,
            [float(next_value.item())] * len(self._rewards),
            self._dones,
        )

        advantages = torch.tensor(
            advantages, dtype=torch.float32, device=self.device
        )

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        states = torch.tensor(
            np.stack(self._states),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            self._actions, dtype=torch.long, device=self.device
        )
        old_log_probs = torch.tensor(
            self._log_probs, dtype=torch.float32, device=self.device
        )

        for _ in range(self.ppo_epochs):
            logits, values = self.network(states)

            distribution = torch.distributions.Categorical(logits=logits)

            new_log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()

            from src.scan.rl.ppo import ppo_clipped_objective

            policy_loss = -ppo_clipped_objective(
                new_log_probs,
                old_log_probs,
                advantages,
                self.clip_epsilon,
            )

            value_loss = self.torch.nn.functional.smooth_l1_loss(
                values.squeeze(-1),
                torch.tensor(returns, dtype=torch.float32, device=self.device),
            )

            loss = (
                policy_loss
                + 0.5 * value_loss
                - self.entropy_bonus * entropy
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
            self.optimizer.step()

        self._states, self._actions = [], []
        self._log_probs, self._rewards = [], []
        self._values, self._dones = [], []

    @staticmethod
    def _gae(rewards, values, next_values, dones, gamma=0.95, lam=0.95):
        advantages = np.zeros(len(rewards), dtype=np.float32)

        running = 0.0

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + gamma * next_values[t] * (1.0 - dones[t])
                - values[t]
            )

            running = (
                delta
                + gamma * lam * (1.0 - dones[t]) * running
            )

            advantages[t] = running

        returns = advantages + np.asarray(values, dtype=np.float32)

        return advantages.tolist(), returns.tolist()


# =====================================================================
# MPC adapter
# =====================================================================

class MPCAdapter(SchedulerPolicy):
    """
    Model-predictive planning over candidate action sequences.

    World model = receiver-observable belief/prediction decay:
      - belief and prediction decay toward the prior
      - observing a channel resets its staleness

    No hidden state is used for planning.
    """

    def __init__(
        self,
        action_space: ScanActionSpace,
        horizon: int = 3,
        gamma: float = 0.95,
        name: str = "MPC",
    ) -> None:
        from src.scan.planning.mpc import MPCPlanner

        self.action_space = action_space
        self.name = name
        self.planner = MPCPlanner(horizon=horizon, gamma=gamma)

    def reset(self, observation=None) -> None:
        return None

    def act(self, observation: PolicyObservation) -> PolicyAction:
        n = self.action_space.num_channels

        def utility(state: "_MPCWorld", action_index: int) -> float:
            belief = float(state.belief[action_index])
            stale = float(state.staleness[action_index])
            prediction = float(state.prediction[action_index])
            uncertainty = float(state.uncertainty[action_index])

            staleness_value = min(stale, 5.0) / 5.0
            coverage_pressure = float(
                np.asarray(observation.coverage_pressure, dtype=np.float64)[action_index]
            )

            return (
                belief
                + 0.75 * prediction
                + 0.5 * staleness_value
                + 0.25 * uncertainty
                + 1.0 * min(coverage_pressure, 4.0)
            )

        def transition(state: "_MPCWorld", action_index: int) -> "_MPCWorld":
            return state.step(action_index)

        state = _MPCWorld(observation)

        candidate_channels = _coverage_candidate_channels(
            observation, n
        )
        result = self.planner.plan(
            state=state,
            actions=candidate_channels,
            transition_fn=transition,
            utility_fn=utility,
        )

        channel = int(result.actions[0]) if result.actions else 0

        return PolicyAction(
            channel=channel,
            dwell_time_s=float(self.action_space.dwell_times_s[0]),
            score=float(result.objective),
            reason="mpc_planning",
            diagnostics={"horizon": float(self.planner.horizon)},
        )

    def update(self, observation, action, reward, next_observation, info=None):
        return None


class _MPCWorld:
    """Receiver-observable lookahead world for MPC."""

    def __init__(self, observation: PolicyObservation) -> None:
        self.belief = np.asarray(observation.channel_belief, dtype=np.float64)
        self.staleness = np.asarray(observation.staleness, dtype=np.float64)
        self.prediction = np.asarray(
            observation.transition_prediction
            + observation.activity_prediction
            + observation.hazard_prediction,
            dtype=np.float64,
        )
        self.uncertainty = np.asarray(
            observation.predictive_uncertainty,
            dtype=np.float64,
        )
        self.staleness = self.staleness + 0.001  # time advance per step

    def step(self, action_index: int) -> "_MPCWorld":
        world = _MPCWorld.__new__(_MPCWorld)

        world.belief = self.belief * 0.98
        world.prediction = self.prediction * 0.9
        world.uncertainty = self.uncertainty + 0.01
        world.staleness = self.staleness + 0.001

        # Observing a channel refreshes it and confirms belief.
        world.belief[action_index] = min(
            world.belief[action_index] + 0.05, 1.0
        )
        world.staleness[action_index] = 0.0

        return world


class LookaheadAdapter(MPCAdapter):
    """1/2-step lookahead planner (MPC with horizon 1 or 2)."""

    def __init__(self, action_space, horizon: int = 2) -> None:
        super().__init__(action_space, horizon=horizon, name="Lookahead")


# =====================================================================
# STARE-assisted adapter
# =====================================================================

class STARECognitiveAdapter(SchedulerPolicy):
    """
    STARE-assisted cognitive scheduler.

    Utility = cognitive utility + STARE embedding similarity bonus:
      channels whose STARE embedding correlates with detected-pulse
      history get a small exploitation bonus.
    """

    def __init__(
        self,
        action_space: ScanActionSpace,
        stare_weight: float = 0.5,
        name: str = "STARE-Cognitive",
    ) -> None:
        self.action_space = action_space
        self.stare_weight = float(stare_weight)
        self.name = name

    def reset(self, observation=None) -> None:
        return None

    def act(self, observation: PolicyObservation) -> PolicyAction:
        n = self.action_space.num_channels

        stare_bonus = self._stare_channel_bonus(observation, n)

        candidate_channels = _coverage_candidate_channels(
            observation, n
        )
        best_channel, best_score, best_dwell = candidate_channels[0], -np.inf, float(
            self.action_space.dwell_times_s[0]
        )

        for channel in candidate_channels:
            belief = _at(observation.channel_belief, channel)
            prediction = (
                _at(observation.transition_prediction, channel)
                + _at(observation.activity_prediction, channel)
                + _at(observation.hazard_prediction, channel)
            )
            staleness = _at(observation.staleness, channel)
            uncertainty = _at(observation.predictive_uncertainty, channel)
            coverage = _at(observation.coverage_pressure, channel)
            priority = _at(observation.priority_pressure, channel)

            score = (
                belief
                + prediction
                + 0.5 * staleness
                + 0.5 * uncertainty
                + 0.75 * coverage
                + 1.25 * priority
                + self.stare_weight * stare_bonus[channel]
            )

            predicted_frequency = float(
                getattr(observation, "stare_next_frequency_mhz", np.nan)
            )
            if np.isfinite(predicted_frequency):
                channel_frequency = float(
                    self.action_space.frequencies_mhz[channel]
                )
                span = max(
                    float(np.ptp(self.action_space.frequencies_mhz)),
                    1e-9,
                )
                frequency_match = max(
                    0.0,
                    1.0 - abs(channel_frequency - predicted_frequency) / span,
                )
                score += 1.5 * frequency_match

            if score > best_score:
                best_score = score
                best_channel = channel

        return PolicyAction(
            channel=int(best_channel),
            dwell_time_s=float(self.action_space.dwell_times_s[0]),
            score=float(best_score),
            reason="stare_cognitive",
            diagnostics={"stare_bonus_mean": float(np.mean(stare_bonus))},
        )

    @staticmethod
    def _stare_channel_bonus(
        observation: PolicyObservation, n_channels: int
    ) -> np.ndarray:
        embedding = np.asarray(observation.stare_embedding, dtype=np.float32)

        if embedding.size == 0:
            return np.zeros(n_channels, dtype=np.float32)

        # Deterministic projection of the embedding onto channels.
        indices = np.arange(embedding.size)
        weights = np.cos(indices * (2 * np.pi / n_channels))

        bonus = np.zeros(n_channels, dtype=np.float64)

        for channel in range(n_channels):
            phase = weights * np.roll(
                np.arange(embedding.size), channel
            )
            bonus[channel] = float(
                np.tanh(np.dot(embedding, phase) / max(embedding.size, 1))
            )

        return bonus.astype(np.float32)

    def update(self, observation, action, reward, next_observation, info=None):
        return None


# =====================================================================
# Adaptive portfolio adapter (PHASE 26)
# =====================================================================

class AdaptivePortfolioAdapter(SchedulerPolicy):
    """
    Portfolio controller.

    Observes ONLY receiver-observable context statistics:
      detection density, SNR, transition entropy, unique channel ratio,
      staleness statistics, uncertainty, OOD/change-point indicators.

    Chooses ONE expert policy per decision via a multiplicative-weights
    (Hedge) update from realized rewards. Never sees scenario name,
    emitter identity, or truth.
    """

    def __init__(
        self,
        action_space: ScanActionSpace,
        state_dim: int,
        experts: Optional[Sequence[SchedulerPolicy]] = None,
        learning_rate: float = 0.15,
        name: str = "AdaptivePortfolio",
    ) -> None:
        self.action_space = action_space
        self.name = name

        if experts is None:
            experts = build_default_experts(
                action_space,
                state_dim,
                include_portfolio=False,
            )

        self.experts: List[SchedulerPolicy] = list(experts)
        self.learning_rate = float(learning_rate)

        self._weights = np.full(
            len(self.experts),
            1.0 / max(len(self.experts), 1),
            dtype=np.float64,
        )

        self._active_expert = 0
        self._last_observation: Optional[PolicyObservation] = None

    # ---- context features (receiver-observable only) -----------------

    @staticmethod
    def _context(observation: PolicyObservation) -> np.ndarray:
        staleness = np.asarray(observation.staleness, dtype=np.float64)
        staleness = np.nan_to_num(staleness, nan=10.0, posinf=10.0)

        finite = staleness[np.isfinite(staleness)]

        context = np.asarray(
            [
                observation.detection_density,
                observation.mean_snr,
                observation.transition_entropy,
                observation.unique_channel_ratio,
                float(np.mean(finite)) if finite.size else 0.0,
                float(np.max(finite)) if finite.size else 0.0,
                float(np.mean(observation.predictive_uncertainty)),
                observation.ood_score,
                observation.change_point_score,
                observation.coverage_fraction,
            ],
            dtype=np.float64,
        )

        return np.nan_to_num(context, nan=0.0, posinf=1e6, neginf=-1e6)

    # ---- interface ----------------------------------------------------

    def reset(self, observation=None) -> None:
        for expert in self.experts:
            expert.reset(observation)
        self._active_expert = int(np.argmax(self._weights))
        self._last_observation = observation

    def act(self, observation: PolicyObservation) -> PolicyAction:
        self._last_observation = observation

        self._active_expert = int(np.argmax(self._weights))

        action = self.experts[self._active_expert].act(observation)

        action.diagnostics = dict(action.diagnostics)
        action.diagnostics["expert"] = float(self._active_expert)
        action.diagnostics["expert_name_weight"] = float(
            self._weights[self._active_expert]
        )
        action.reason = (
            f"portfolio[{self.experts[self._active_expert].name}]"
        )

        return action

    def update(self, observation, action, reward, next_observation, info=None):
        # Hedge update: experts are scored by the reward the portfolio
        # itself received (counterfactual-free, no future information).
        losses = np.zeros(len(self.experts), dtype=np.float64)

        active = self._active_expert
        losses[active] = -float(reward)

        # Regularize weights toward uniform on OOD to force exploration.
        weights = self._weights * np.exp(-self.learning_rate * losses)

        if self._last_observation is not None and (
            self._last_observation.ood_score > 0.5
        ):
            weights = 0.9 * weights + 0.1 * np.mean(weights)

        total = weights.sum()

        if total <= 1e-12:
            weights = np.full_like(weights, 1.0 / len(weights))
        else:
            weights = weights / total

        self._weights = weights

        # Propagate experience to the active expert only.
        try:
            self.experts[active].update(
                observation, action, reward, next_observation, info
            )
        except Exception:
            pass


# =====================================================================
# Adapter registry
# =====================================================================

def build_default_experts(
    action_space: ScanActionSpace,
    state_dim: int,
    include_portfolio: bool = False,
    seed: int = 42,
) -> List[SchedulerPolicy]:
    """
    All V5 policies conforming to the universal interface.
    """

    experts: List[SchedulerPolicy] = [
        SequentialAdapter(action_space, name="Sequential"),
        RoundRobinDwellAdapter(action_space),
        _make_scheduler_adapter(
            StalenessScheduler(action_space), action_space, "Staleness"
        ),
        _make_scheduler_adapter(
            InformationGreedyScheduler(action_space),
            action_space,
            "InformationGain",
        ),
        UCB1Adapter(action_space, seed=seed),
        LinUCBAdapter(
            action_space,
            context_dim=state_dim,
            alpha=1.0,
            seed=seed,
        ),
        ThompsonAdapter(action_space, seed=seed),
        MarkovAdapter(action_space),
        _make_scheduler_adapter(
            CognitiveGreedyScheduler(action_space),
            action_space,
            "Cognitive",
        ),
        DQNAdapter(action_space, state_dim, seed=seed),
        DoubleDQNAdapter(action_space, state_dim, seed=seed),
        DuelingDQNAdapter(action_space, state_dim, seed=seed),
        PPOAdapter(action_space, state_dim, seed=seed),
        MPCAdapter(action_space, horizon=3),
        STARECognitiveAdapter(action_space),
    ]

    if include_portfolio:
        experts.append(
            AdaptivePortfolioAdapter(
                action_space,
                state_dim,
                experts=build_default_experts(
                    action_space, state_dim, include_portfolio=False,
                    seed=seed,
                ),
            )
        )

    return experts


def _make_scheduler_adapter(
    scheduler,
    action_space: ScanActionSpace,
    name: str,
) -> ScoredSchedulerAdapter:
    return ScoredSchedulerAdapter(scheduler, action_space, name)


# =====================================================================
# Convenience: build a policy by name
# =====================================================================

_POLICY_BUILDERS = {
    "sequential": lambda a, d: SequentialAdapter(a, name="Sequential"),
    "roundrobin": lambda a, d: RoundRobinDwellAdapter(a),
    "staleness": lambda a, d: ScoredSchedulerAdapter(
        StalenessScheduler(a), a, "Staleness"
    ),
    "information": lambda a, d: ScoredSchedulerAdapter(
        InformationGreedyScheduler(a), a, "InformationGain"
    ),
    "ucb1": lambda a, d: UCB1Adapter(a),
    "linucb": lambda a, d: LinUCBAdapter(a, context_dim=d),
    "thompson": lambda a, d: ThompsonAdapter(a),
    "markov": lambda a, d: MarkovAdapter(a),
    "cognitive": lambda a, d: ScoredSchedulerAdapter(
        CognitiveGreedyScheduler(a), a, "Cognitive"
    ),
    "dqn": lambda a, d: DQNAdapter(a, d),
    "double_dqn": lambda a, d: DoubleDQNAdapter(a, d),
    "dueling_dqn": lambda a, d: DuelingDQNAdapter(a, d),
    "dueling_double_dqn": lambda a, d: DuelingDoubleDQNAdapter(a, d),
    "ppo": lambda a, d: PPOAdapter(a, d),
    "mpc": lambda a, d: MPCAdapter(a, horizon=3),
    "lookahead": lambda a, d: LookaheadAdapter(a, horizon=2),
    "stare_cognitive": lambda a, d: STARECognitiveAdapter(a),
    "portfolio": lambda a, d: AdaptivePortfolioAdapter(a, d),
}


def make_policy(name: str, action_space: ScanActionSpace, state_dim: int):
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")

    if normalized not in _POLICY_BUILDERS:
        raise ValueError(
            f"Unknown policy '{name}'. "
            f"Available: {sorted(_POLICY_BUILDERS.keys())}"
        )

    return _POLICY_BUILDERS[normalized](action_space, state_dim)


def available_policies() -> List[str]:
    return sorted(_POLICY_BUILDERS.keys())


# small helper used above
def _at(array: np.ndarray, index: int) -> float:
    array = np.asarray(array)
    if index >= array.size:
        return 0.0
    return float(array[index])
