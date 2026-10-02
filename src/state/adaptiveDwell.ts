/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ChannelBelief, ReceiverAction } from './types';

// PROVENANCE: copied verbatim from ../src/optimization/adaptiveDwell.ts (repo root).

/**
 * Priority 9 — Detection-Aware Dwell Optimizer:
 * Evaluates candidate dwells d ∈ D_candidates using:
 *
 * J(d) = P_D(d | SNR, PRI) + λ * IG(d) - μ * SwitchingCost(d) - ν * CoverageCost(d)
 *
 * d* = argmax_d J(d)
 *
 * Ensures that low SNR signals receive sufficient pulse integration time even when confidence is high,
 * while preventing information gain alone from prematurely terminating critical radar dwells.
 */
export class AdaptiveDwellManager {
  public static readonly CANDIDATE_DWELLS_MS = [0.55, 0.75, 0.95, 1.15, 1.45, 1.85, 2.35];
  private static readonly LO_SETTLING_MS = 0.035; // 35 μs superheterodyne synthesizer lock time

  public static computeDwellTime(
    reason: ReceiverAction['reason'],
    confidence: number,
    belief: ChannelBelief,
    allBeliefs: ChannelBelief[] = [],
    estimatedPriMs: number = 1.5,
    estimatedSnrDb: number = 14.0,
    isSwitchedChannel: boolean = true
  ): number {
    // 1. Fast confirmation burst override: rapid pulses to cement track formation (Priority 10)
    if (reason === 'confirmation_burst') {
      return 0.65;
    }

    // 2. Immediate override for coverage shield: rapid probe to satisfy deadline (Priority 1)
    if (reason === 'coverage_shield') {
      return 0.60;
    }

    // Fast-path exploit: optimized for high-performance intercept
    if (reason === 'fast_exploit') {
      // Dwell matched to single-pulse or double-pulse capture window
      const baseD = Math.max(0.75, Math.min(1.4, estimatedPriMs * 1.1));
      return Number(baseD.toFixed(2));
    }

    const pri = Math.max(0.35, Math.min(12.0, estimatedPriMs));
    const pOcc = Math.max(0.05, Math.min(0.99, belief.occupancyProb));
    const snr = Math.max(3.0, Math.min(30.0, estimatedSnrDb));

    // Single-pulse P_D based on Marcum Q-function approximation from SNR
    // Albersheim's / Marcum's equation approximation:
    const pdSinglePulse = Math.min(0.99, Math.max(0.2, 1.0 - 0.5 * Math.exp(-Math.pow(10, snr / 10.0) / 4.0)));

    // Coverage staleness pressure from remaining channels
    let meanOtherStaleness = 0;
    if (allBeliefs.length > 1) {
      let sum = 0;
      for (const b of allBeliefs) {
        if (b.channelIndex !== belief.channelIndex) {
          sum += b.stalenessMs;
        }
      }
      meanOtherStaleness = sum / (allBeliefs.length - 1);
    }
    const coverageCostNorm = Math.min(2.0, meanOtherStaleness / 80.0);

    let bestDwell = 1.15;
    let maxJ = -Infinity;

    // Optimization hyperparameters:
    const lambda = 0.85; // IG weight
    const mu = 0.45;     // Switching cost weight
    const nu = 0.55;     // Coverage cost weight

    for (const d of this.CANDIDATE_DWELLS_MS) {
      const tObs = Math.max(0, d - this.LO_SETTLING_MS);
      if (tObs <= 0) continue;

      // Expected number of pulses arriving during active digitization
      const pulsesInWindow = tObs / pri;

      // Cumulative detection probability:
      // P_D(d) = P(occupied) * (1 - (1 - P_D,single)^N)
      const pdWindow = 1.0 - Math.pow(1.0 - pdSinglePulse, Math.max(0.5, pulsesInWindow));
      const expectedPd = pOcc * pdWindow;

      // Mutual Information Gain yield for dwell duration d
      const igYield = belief.informationGain * (1.0 - Math.exp(-d / (pri * 1.5)));

      // Hardware switching penalty: only assessed if LO actually retuned
      const switchingCost = isSwitchedChannel ? (this.LO_SETTLING_MS / d) * 1.2 : 0.0;

      // Opportunity / Fleet Coverage cost: longer dwell deprives other channels
      const coverageCost = (d / 1.5) * coverageCostNorm;

      // Target objective J(d):
      // J(d) = P_D(d) + λ * IG(d) - μ * SwitchingCost(d) - ν * CoverageCost(d)
      let jVal = expectedPd * 4.0 + lambda * igYield - mu * switchingCost - nu * coverageCost;

      // Threat priority scaling
      jVal *= 1.0 + (belief.threatPriority / 5.0) * 0.5;

      if (jVal > maxJ) {
        maxJ = jVal;
        bestDwell = d;
      }
    }

    return Number(bestDwell.toFixed(2));
  }
}
