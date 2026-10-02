/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ChannelBelief, HazardModelMode } from '../types';

export interface ChannelRecurrenceStats {
  samples: number[];
  meanMs: number;
  stdMs: number;
  weibullK: number;     // Shape parameter k
  weibullEta: number;   // Scale parameter η in ms
  lastPulseTimeMs: number;
}

/**
 * Research-Grade Survival Analysis & Recurrence Hazard Engine:
 * Replaces ad-hoc constants with mathematically grounded parameter estimation
 * from empirical pulse inter-arrival timing statistics.
 *
 * Supports three analytical survival formulations:
 * 1. Weibull Hazard:
 *    h(t) = (k / η) * (t / η)^(k - 1)
 *    Cumulative Hazard H(t) = (t / η)^k
 *    Survival Function S(t) = exp(-H(t))
 * 2. Exponential (Memoryless Poisson):
 *    h(t) = λ = 1 / μ
 *    H(t) = λ * t
 *    S(t) = exp(-λ * t)
 * 3. Empirical Kaplan-Meier Survival:
 *    S(t) = 1 - (empirical CDF of observed recurrence times ≤ t)
 */
export class HazardSurvivalModel {
  public static mode: HazardModelMode = 'weibull';
  private static channelStats: Map<number, ChannelRecurrenceStats> = new Map();

  public static setMode(mode: HazardModelMode): void {
    this.mode = mode;
  }

  public static reset(): void {
    this.channelStats.clear();
  }

  private static getOrInitStats(channelIndex: number): ChannelRecurrenceStats {
    let stat = this.channelStats.get(channelIndex);
    if (!stat) {
      stat = {
        samples: [],
        meanMs: 25.0, // Default prior mean recurrence
        stdMs: 15.0,
        weibullK: 1.4, // Prior slightly super-exponential (radar periodicity)
        weibullEta: 28.0,
        lastPulseTimeMs: 0,
      };
      this.channelStats.set(channelIndex, stat);
    }
    return stat;
  }

  /**
   * Registers newly intercepted pulse in channel to learn timing recurrence
   */
  public static onPulseObserved(b: ChannelBelief, currentTimeMs: number, priEstimateMs?: number): void {
    const stat = this.getOrInitStats(b.channelIndex);

    if (stat.lastPulseTimeMs > 0) {
      const dt = currentTimeMs - stat.lastPulseTimeMs;
      // Filter out zero-delays from same burst or pulses within receiver window
      if (dt > 0.2 && dt < 2000.0) {
        stat.samples.push(dt);
        if (stat.samples.length > 30) {
          stat.samples.shift();
        }
        this.recomputeParameters(stat);
      }
    } else if (priEstimateMs && priEstimateMs > 0) {
      stat.samples.push(priEstimateMs * 8); // Estimate burst interval from PRI
      this.recomputeParameters(stat);
    }

    stat.lastPulseTimeMs = currentTimeMs;
    b.hazardRate = stat.weibullEta > 0 ? 1.0 / stat.weibullEta : 0.05;
  }

  /**
   * Moment-based Weibull and Poisson parameter estimation
   */
  private static recomputeParameters(stat: ChannelRecurrenceStats): void {
    if (stat.samples.length < 2) return;

    const n = stat.samples.length;
    const mean = stat.samples.reduce((a, b) => a + b, 0) / n;
    const variance = stat.samples.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / (n - 1);
    const std = Math.sqrt(Math.max(0.1, variance));

    stat.meanMs = mean;
    stat.stdMs = std;

    // Menon's Moment Estimator for Weibull shape parameter k:
    // k ≈ (std / mean)^(-1.086)
    const cv = Math.max(0.08, std / mean);
    const k = Math.min(3.5, Math.max(0.8, Math.pow(cv, -1.086)));
    stat.weibullK = k;

    // Scale parameter η = mean / Γ(1 + 1/k)
    // Using Ramanujan/Lanczos approximation for Γ(1 + 1/k)
    const gammaArg = 1.0 + 1.0 / k;
    const gammaVal = this.approxGamma(gammaArg);
    stat.weibullEta = Math.max(1.0, mean / gammaVal);
  }

  /**
   * Updates hazard rates, survival probability S(t), and opportunity cost for all channel beliefs
   */
  public static updateChannelHazards(beliefs: ChannelBelief[], currentTimeMs: number): void {
    for (const b of beliefs) {
      const tau = Math.max(0.1, b.stalenessMs);
      const stat = this.getOrInitStats(b.channelIndex);

      let survival = 1.0;
      let instantaneousHazard = 0.02;

      switch (this.mode) {
        case 'exponential': {
          // Exponential survival: S(t) = exp(-λ * t)
          const lambda = 1.0 / Math.max(1.0, stat.meanMs);
          const cumHazard = lambda * tau;
          survival = Math.exp(-Math.min(15.0, cumHazard));
          instantaneousHazard = lambda;
          break;
        }

        case 'empirical': {
          // Empirical Kaplan-Meier survival estimator
          if (stat.samples.length >= 4) {
            const recurrencesBelowTau = stat.samples.filter((s) => s <= tau).length;
            const empiricalCdf = recurrencesBelowTau / stat.samples.length;
            survival = Math.max(0.01, 1.0 - empiricalCdf);
            instantaneousHazard = empiricalCdf / Math.max(1.0, tau);
          } else {
            // Fallback to exponential
            const lambda = 1.0 / Math.max(1.0, stat.meanMs);
            survival = Math.exp(-Math.min(15.0, lambda * tau));
            instantaneousHazard = lambda;
          }
          break;
        }

        case 'weibull':
        default: {
          // Learned Weibull: H(t) = (t / η)^k
          const k = stat.weibullK;
          const eta = Math.max(1.0, stat.weibullEta);
          const normT = tau / eta;
          const cumHazard = Math.pow(normT, k);

          survival = Math.exp(-Math.min(15.0, cumHazard));
          instantaneousHazard = (k / eta) * Math.pow(Math.max(1e-4, normT), k - 1);
          break;
        }
      }

      b.survivalProb = survival;
      b.hazardRate = Number(instantaneousHazard.toFixed(4));

      // Miss hazard / Opportunity cost factor: F(t) = 1 - S(t)
      const missHazard = 1.0 - survival;
      const threatScale = b.threatPriority / 5.0;

      // Mathematically grounded Opportunity Cost:
      // OC(c) = (1 - S_c(τ)) * (Priority_c / 5) * (E[p_c] + 1.8 * Var[p_c])
      b.opportunityCost = missHazard * threatScale * (b.occupancyProb + b.epistemicUncertainty * 1.8);
    }
  }

  private static approxGamma(z: number): number {
    // Stirling approximation for Gamma(z) around z ∈ [1, 2]
    const g = 7;
    const c = [
      0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.138571095836526, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    if (z < 0.5) {
      return Math.PI / (Math.sin(Math.PI * z) * this.approxGamma(1 - z));
    }
    z -= 1;
    let x = c[0];
    for (let i = 1; i < g + 2; i++) {
      x += c[i] / (z + i);
    }
    const t = z + g + 0.5;
    return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
  }
}
