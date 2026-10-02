/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ChannelBelief, BayesianUpdateMode } from './types';
import { getChannelCenterFreq } from '../simulation/emitterModels';

// PROVENANCE: copied verbatim from ../src/belief/bayesianBelief.ts (repo root).

/**
 * Bayesian Beta-Bernoulli Spectrum Occupancy Engine:
 * Maintains mathematically rigorous posterior probability distributions over channel occupancy.
 *
 * Supports two distinct modes:
 * 1. Exact Conjugate Beta-Bernoulli:
 *    α' = α + x, β' = β + (1 - x) where x ∈ {0, 1}
 * 2. Discounted Beta (with exponential forgetting parameter ρ ∈ (0, 1]):
 *    α' = ρ * α + x
 *    β' = ρ * β + (1 - x)
 *
 * Coverage Accounting Semantics:
 * A channel only resets true surveillance staleness (T_stale = 0) after the physical
 * receiver has completed a valid, settled observation window.
 */
export class BayesianBeliefEngine {
  public beliefs: ChannelBelief[] = [];
  public mode: BayesianUpdateMode = 'discounted_beta';
  public discountFactorRho: number = 0.96; // Forgetting factor for discounted Beta
  private channelCount: number;

  // Base uninformative EW spectrum prior Beta(1.2, 8.8) (~12% prior occupancy)
  private readonly priorAlpha = 1.2;
  private readonly priorBeta = 8.8;

  constructor(channelCount: number = 32, mode: BayesianUpdateMode = 'discounted_beta') {
    this.channelCount = channelCount;
    this.mode = mode;
    this.reset();
  }

  public setMode(mode: BayesianUpdateMode): void {
    this.mode = mode;
  }

  public reset(): void {
    this.beliefs = [];
    for (let c = 0; c < this.channelCount; c++) {
      const alpha = this.priorAlpha;
      const beta = this.priorBeta;
      const mean = alpha / (alpha + beta);
      const variance = (alpha * beta) / (Math.pow(alpha + beta, 2) * (alpha + beta + 1));

      this.beliefs.push({
        channelIndex: c,
        freqMHz: getChannelCenterFreq(c),
        occupancyProb: mean,
        betaAlpha: alpha,
        betaBeta: beta,
        epistemicUncertainty: variance,
        stalenessMs: 0,
        lastObservedTimeMs: 0,
        totalHits: 0,
        totalVisits: 0,
        opportunityCost: 0,
        hazardRate: 0.02,
        survivalProb: 1.0,
        informationGain: 0.1,
        threatPriority: 3,
        shieldAlertLevel: 'green',
      });
    }
  }

  /**
   * Semantics 1: Scheduled Visit
   * Receiver has commanded tuning to this channel. Increments visit counter,
   * but DOES NOT reset surveillance staleness because receiver is still settling.
   */
  public recordScheduledVisit(channelIndex: number): void {
    if (channelIndex >= 0 && channelIndex < this.channelCount) {
      this.beliefs[channelIndex].totalVisits++;
    }
  }

  /**
   * Semantics 2: Successful Observation Opportunity
   * Receiver has successfully settled and dwelled for a non-zero observation window.
   * True surveillance staleness is now legitimately refreshed.
   */
  public recordObservationOpportunity(
    channelIndex: number,
    currentTimeMs: number,
    effectiveObsDurationMs: number,
    isHit: boolean,
    priority: number = 3
  ): void {
    if (channelIndex < 0 || channelIndex >= this.channelCount) return;
    const b = this.beliefs[channelIndex];

    // Only reset staleness if the receiver actually had a valid observation window
    if (effectiveObsDurationMs > 0.05) {
      b.lastObservedTimeMs = currentTimeMs;
      b.stalenessMs = 0;
    }

    if (isHit) {
      b.totalHits++;
      b.threatPriority = Math.max(b.threatPriority, priority);

      if (this.mode === 'exact_beta') {
        // Standard conjugate update: α' = α + 1, β' = β
        b.betaAlpha = Math.min(80, b.betaAlpha + 1.0);
      } else {
        // Discounted Beta: α' = ρ*α + 1, β' = ρ*β
        b.betaAlpha = Math.min(80, this.discountFactorRho * b.betaAlpha + 1.0);
        b.betaBeta = Math.max(1.0, this.discountFactorRho * b.betaBeta);
      }
    } else {
      if (this.mode === 'exact_beta') {
        // Standard conjugate update: α' = α, β' = β + 1
        b.betaBeta = Math.min(80, b.betaBeta + 1.0);
      } else {
        // Discounted Beta: α' = ρ*α, β' = ρ*β + 1
        b.betaAlpha = Math.max(1.0, this.discountFactorRho * b.betaAlpha);
        b.betaBeta = Math.min(80, this.discountFactorRho * b.betaBeta + 1.0);
      }
    }

    this.recomputeBeliefStats(b);
  }

  /**
   * Continuous temporal decay / belief regression toward uninformative prior
   * for channels that have not been observed.
   */
  public ageBeliefs(currentTimeMs: number, maxAllowedStalenessMs: number = 180): void {
    const decayRate = 0.008; // Continuous aging factor per ms

    for (let c = 0; c < this.channelCount; c++) {
      const b = this.beliefs[c];
      b.stalenessMs = Math.max(0, currentTimeMs - b.lastObservedTimeMs);

      // Regress alpha and beta toward default prior (1.2, 8.8)
      const dt = b.stalenessMs;
      const decay = Math.exp(-decayRate * (dt / 10.0));

      b.betaAlpha = this.priorAlpha + (b.betaAlpha - this.priorAlpha) * decay;
      b.betaBeta = this.priorBeta + (b.betaBeta - this.priorBeta) * decay;

      this.recomputeBeliefStats(b);

      // Priority-based shield alert level
      const deadline = maxAllowedStalenessMs / (1.0 + 0.15 * b.threatPriority);
      if (b.stalenessMs >= deadline) {
        b.shieldAlertLevel = 'red';
      } else if (b.stalenessMs >= deadline * 0.75) {
        b.shieldAlertLevel = 'yellow';
      } else {
        b.shieldAlertLevel = 'green';
      }
    }
  }

  private recomputeBeliefStats(b: ChannelBelief): void {
    const total = b.betaAlpha + b.betaBeta;
    b.occupancyProb = b.betaAlpha / total;
    b.epistemicUncertainty = (b.betaAlpha * b.betaBeta) / (Math.pow(total, 2) * (total + 1));
  }
}
