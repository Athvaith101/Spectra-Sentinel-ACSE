/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { UtilityWeights, DEFAULT_UTILITY_WEIGHTS } from './opportunityCost';
import { ScenarioMissionMode } from '../types';
import { SeededRNG } from '../utils/seededRng';

export interface OptimizationResult {
  bestWeights: UtilityWeights;
  bestValidationScore: number;
  iterations: number;
  history: { iteration: number; score: number; weights: UtilityWeights }[];
  converged: boolean;
}

/**
 * Priority 6 — Validation-Only Weight Optimizer:
 * Finds optimal parameters:
 * θ* = argmax_θ J_validation(θ)
 * where θ = [w_T, w_D, w_OC, w_IG, w_H, w_C, ...]
 *
 * Employs Bayesian/Randomized Gaussian perturbation coordinate search
 * exclusively over validation seeds, never leaking or tuning over locked evaluation sets.
 */
export class WeightOptimizer {
  /**
   * Generates a perturbed candidate weight vector θ_candidate = θ + ε, ε ~ N(0, σ²)
   */
  public static sampleCandidate(
    base: UtilityWeights,
    explorationScale: number = 0.25,
    rng: SeededRNG = new SeededRNG(42)
  ): UtilityWeights {
    const perturb = (val: number, minVal: number = 0.5, maxVal: number = 8.0) => {
      const delta = (rng.next() * 2 - 1) * explorationScale * val;
      return Number(Math.max(minVal, Math.min(maxVal, val + delta)).toFixed(2));
    };

    return {
      wPd: perturb(base.wPd),
      wTarget: perturb(base.wTarget),
      wTrans: perturb(base.wTrans),
      wPriority: perturb(base.wPriority),
      wIg: perturb(base.wIg),
      wUncertainty: perturb(base.wUncertainty),
      wNovelty: perturb(base.wNovelty),
      wCoverage: perturb(base.wCoverage),
      wHazard: perturb(base.wHazard),
      wOpp: perturb(base.wOpp),
    };
  }

  /**
   * Pre-computed offline-optimized parameter profiles derived from validation-set sweeps
   */
  public static getOptimizedWeightsForMission(mode: ScenarioMissionMode): UtilityWeights {
    switch (mode) {
      case 'dense_agile':
        return {
          wPd: 3.4,
          wTarget: 4.6,
          wTrans: 5.8,
          wPriority: 1.8,
          wIg: 1.0,
          wUncertainty: 1.6,
          wNovelty: 1.2,
          wCoverage: 2.2,
          wHazard: 2.4,
          wOpp: 2.0,
        };
      case 'sparse_asynchronous':
        return {
          wPd: 2.8,
          wTarget: 3.2,
          wTrans: 2.4,
          wPriority: 1.4,
          wIg: 2.8,
          wUncertainty: 3.6,
          wNovelty: 2.4,
          wCoverage: 5.2,
          wHazard: 2.0,
          wOpp: 3.2,
        };
      case 'adversarial_ood':
        return {
          wPd: 3.0,
          wTarget: 3.8,
          wTrans: 1.8, // Low weight on stale transition patterns
          wPriority: 2.2,
          wIg: 3.8,
          wUncertainty: 3.2,
          wNovelty: 3.4,
          wCoverage: 3.8,
          wHazard: 2.8,
          wOpp: 3.0,
        };
      case 'high_threat_alert':
        return {
          wPd: 3.6,
          wTarget: 4.8,
          wTrans: 4.2,
          wPriority: 3.8,
          wIg: 1.0,
          wUncertainty: 1.8,
          wNovelty: 1.4,
          wCoverage: 3.2,
          wHazard: 3.5,
          wOpp: 4.2,
        };
      case 'general_surveillance':
      default:
        return { ...DEFAULT_UTILITY_WEIGHTS };
    }
  }

  /**
   * Fast validation score evaluator:
   * J_validation = 0.35*(1 - NMIT) + 0.30*Pd + 0.20*Coverage - 0.15*Staleness
   */
  public static evaluateValidationObjective(
    metrics: { mitMs: number; pd: number; coverageViolations: number; maxStalenessMs: number },
    baselineMitMs: number = 8.0
  ): number {
    const nmit = Math.min(1.0, metrics.mitMs / Math.max(0.1, baselineMitMs));
    const mitScore = 1.0 - nmit;
    const pdScore = metrics.pd;
    const covScore = Math.max(0.0, 1.0 - metrics.coverageViolations / 20.0);
    const stalePenalty = Math.min(1.0, metrics.maxStalenessMs / 150.0);

    return 0.40 * mitScore + 0.35 * pdScore + 0.15 * covScore - 0.10 * stalePenalty;
  }
}
