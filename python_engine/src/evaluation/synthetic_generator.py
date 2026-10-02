"""Deterministic synthetic receiver scenarios used by the public demo API.

The generator produces receiver truth events only for the environment. Policies
still receive PolicyObservation through the canonical tournament harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SyntheticConfig:
    num_emitters: int = 8
    horizon_s: float = 0.5
    frequency_min_mhz: float = 2000.0
    frequency_max_mhz: float = 9750.0
    frequency_step_mhz: float = 250.0
    pulse_rate_hz: float = 180.0


def available_synthetic_scenarios() -> List[str]:
    return ["fixed", "sparse", "hopping", "burst", "burst_lpi", "dense", "adversarial", "mixed"]


def _emit_event(rng: np.random.Generator, emitter: int, toa: float, freq: float, lpi: bool = False) -> Dict:
    return {
        "event_id": f"e{emitter}_{toa:.9f}",
        "toa_s": float(toa),
        "emit_time_s": float(toa),
        "frequency_mhz": float(freq),
        "pulse_width_s": float(rng.uniform(0.4e-6, 2.5e-6) if lpi else rng.uniform(1e-6, 8e-6)),
        "aoa_deg": float(rng.uniform(-90, 90)),
        "amplitude": float(rng.uniform(0.5, 2.0)),
        "priority": 1.0,
        "weight": 1.0,
        "emitter_id": int(emitter),
    }


def generate(name: str, cfg: SyntheticConfig, seed: int = 26055) -> Tuple[List[Dict], List[float]]:
    name = name.strip().lower().replace(" ", "_")
    if name not in available_synthetic_scenarios():
        raise ValueError(f"Unknown synthetic scenario '{name}'")
    rng = np.random.default_rng(seed)
    grid = np.arange(cfg.frequency_min_mhz, cfg.frequency_max_mhz + 0.5 * cfg.frequency_step_mhz, cfg.frequency_step_mhz)
    n = max(1, min(int(cfg.num_emitters), 50))
    base = rng.choice(grid, size=n, replace=False if n <= len(grid) else True)
    events: List[Dict] = []

    for emitter in range(n):
        f0 = float(base[emitter])
        if name == "sparse":
            count = max(1, int(cfg.horizon_s * cfg.pulse_rate_hz * 0.25))
        elif name == "dense":
            count = max(1, int(cfg.horizon_s * cfg.pulse_rate_hz * 2.2))
        elif name in {"burst", "burst_lpi"}:
            count = max(2, int(cfg.horizon_s * cfg.pulse_rate_hz * 1.4))
        else:
            count = max(1, int(cfg.horizon_s * cfg.pulse_rate_hz))

        times = np.sort(rng.uniform(0, max(cfg.horizon_s - 1e-6, 1e-6), size=count))
        for j, t in enumerate(times):
            freq = f0
            if name in {"hopping", "adversarial", "mixed"} and j % (3 if name != "adversarial" else 2) == 0:
                hop = int(rng.integers(-5, 6))
                freq = float(np.clip(f0 + hop * cfg.frequency_step_mhz, grid[0], grid[-1]))
            if name in {"burst", "burst_lpi"}:
                center = cfg.horizon_s * (0.25 + 0.5 * ((emitter + 1) % 3) / 3.0)
                if abs(float(t) - center) > cfg.horizon_s * 0.22:
                    continue
            lpi = name == "burst_lpi" or (name == "adversarial" and j % 5 == 0)
            events.append(_emit_event(rng, emitter, float(t), freq, lpi=lpi))

    if name == "fixed":
        events.sort(key=lambda e: e["toa_s"])
    elif name == "adversarial":
        # Concentrate activity around a small subset of channels to test exploration.
        hot = list(base[:max(1, min(2, n))])
        for i, e in enumerate(events):
            if i % 3:
                e["frequency_mhz"] = float(hot[i % len(hot)])
        events.sort(key=lambda e: e["toa_s"])
    elif name == "mixed":
        # Blend fixed, hopping and sparse bursts deterministically.
        for i, e in enumerate(events):
            if i % 4 == 0:
                e["frequency_mhz"] = float(base[i % n])
            elif i % 4 == 1:
                e["frequency_mhz"] = float(np.clip(e["frequency_mhz"] + rng.choice([-2, 2, 3]) * cfg.frequency_step_mhz, grid[0], grid[-1]))
        events.sort(key=lambda e: e["toa_s"])

    active = sorted({float(round(e["frequency_mhz"], 6)) for e in events})
    return events, active
