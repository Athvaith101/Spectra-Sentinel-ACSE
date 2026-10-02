from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardComponents:
    interception: float = 0.0
    priority: float = 0.0
    information: float = 0.0
    latency: float = 0.0
    coverage: float = 0.0
    opportunity_cost: float = 0.0
    receiver_cost: float = 0.0


@dataclass
class RewardConfig:
    eta_info: float = 0.10
    beta_latency: float = 0.10
    gamma_coverage: float = 1.0
    kappa_opportunity: float = 0.10
    receiver_cost_weight: float = 0.01


def compose_reward(
    components: RewardComponents,
    config: RewardConfig | None = None,
) -> float:
    """
    Unified reward:

        R =
            R_int
            + R_prio
            + eta R_info
            - beta latency
            - gamma coverage_violation
            - kappa opportunity_cost
            - receiver_cost
    """

    if config is None:
        config = RewardConfig()

    return float(
        components.interception
        + components.priority
        + config.eta_info * components.information
        - config.beta_latency * components.latency
        - config.gamma_coverage * components.coverage
        - config.kappa_opportunity * components.opportunity_cost
        - config.receiver_cost_weight * components.receiver_cost
    )


def mean_variance_objective(
    mean_reward: float,
    variance: float,
    risk_aversion: float = 1.0,
) -> float:
    """
        J = E[R] - lambda Var(R)
    """

    return float(
        mean_reward
        - risk_aversion * variance
    )


def cvar_lower_tail(
    rewards,
    alpha: float = 0.1,
) -> float:
    """
    Lower-tail CVaR for reward.

    Smaller alpha focuses on the worst outcomes.
    """

    import numpy as np

    rewards = np.asarray(
        rewards,
        dtype=np.float64,
    ).reshape(-1)

    if rewards.size == 0:
        return 0.0

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")

    sorted_rewards = np.sort(rewards)

    count = max(
        1,
        int(np.ceil(alpha * len(sorted_rewards))),
    )

    return float(
        np.mean(sorted_rewards[:count])
    )


def lagrangian_objective(
    expected_reward: float,
    coverage_violation: float,
    hp_miss: float,
    coverage_multiplier: float,
    hp_multiplier: float,
) -> float:
    """
        L =
            E[R]
            - lambda_c C_cov
            - lambda_hp C_hp
    """

    return float(
        expected_reward
        - coverage_multiplier * coverage_violation
        - hp_multiplier * hp_miss
    )
