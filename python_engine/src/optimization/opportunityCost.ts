/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ChannelBelief, ComponentGates, ScenarioMissionMode } from '../types';

export interface UtilityWeights {
  wPd: number;
  wTarget: number;
  wTrans: number;
  wPriority: number;
  wIg: number;
  wUncertainty: number;
  wNovelty: number;
  wCoverage: number;
  wHazard: number;
  wOpp: number;
}

export const DEFAULT_UTILITY_WEIGHTS: UtilityWeights = {
  wPd: 3.2,
  wTarget: 4.0,
  wTrans: 4.8,
  wPriority: 1.5,
  wIg: 1.2,
  wUncertainty: 2.2,
  wNovelty: 1.8,
  wCoverage: 3.5,
  wHazard: 2.6,
  wOpp: 2.4,
};

export class OpportunityCostEngine {
  /**
   * Priority 7 — Scenario-Dependent / Mission-Conditioned Objective Weights:
   * Dynamically adapts weights according to the operational threat environment:
   * - Persistent emitters: w_tracking / w_target boosted
   * - Sparse emitters: w_exploration / w_coverage boosted
   * - Adversarial / OOD: w_information boosted
   * - High threat alert: w_priority boosted
   */
  public static getScenarioWeights(
    mode: ScenarioMissionMode = 'general_surveillance',
    baseWeights: UtilityWeights = DEFAULT_UTILITY_WEIGHTS
  ): UtilityWeights {
    const w = { ...baseWeights };
    switch (mode) {
      case 'dense_agile':
        w.wTrans = 5.6;
        w.wTarget = 4.8;
        w.wPd = 3.6;
        w.wCoverage = 2.0;
        break;
      case 'sparse_asynchronous':
        w.wCoverage = 4.8;
        w.wUncertainty = 3.2;
        w.wIg = 2.4;
        w.wTrans = 2.5;
        break;
      case 'adversarial_ood':
        w.wIg = 3.5;
        w.wNovelty = 3.0;
        w.wUncertainty = 2.8;
        w.wTrans = 2.0; // Don't overfit to shifting Markov models
        break;
      case 'high_threat_alert':
        w.wPriority = 3.0;
        w.wOpp = 3.8;
        w.wTarget = 4.5;
        break;
      case 'general_surveillance':
      default:
        break;
    }
    return w;
  }

  /**
   * Priority 1 — EXPLOIT Mode Utility:
   * Direct aggressive target interception:
   * U_E(a) = w1 * P_D + w2 * P_target + w3 * P_transition + w4 * Priority
   */
  public static scoreExploit(
    c: number,
    belief: ChannelBelief,
    transitionProb: number,
    weights: UtilityWeights = DEFAULT_UTILITY_WEIGHTS
  ): number {
    const pD = Math.max(0.05, Math.min(0.99, 1.0 - Math.exp(-belief.occupancyProb * 2.5)));
    const pTarget = belief.occupancyProb;
    const priorityNorm = belief.threatPriority / 5.0;

    return (
      weights.wPd * pD +
      weights.wTarget * pTarget +
      weights.wTrans * transitionProb +
      weights.wPriority * priorityNorm
    );
  }

  /**
   * Priority 1 — EXPLORE Mode Utility:
   * Deliberate information gathering & uncertainty reduction:
   * U_X(a) = w1 * IG + w2 * Uncertainty + w3 * Novelty + w4 * Coverage + w5 * Hazard
   */
  public static scoreExplore(
    c: number,
    belief: ChannelBelief,
    novelty: number,
    coverageDeadlineMs: number = 100.0,
    weights: UtilityWeights = DEFAULT_UTILITY_WEIGHTS
  ): number {
    const ig = belief.informationGain;
    const uncertainty = belief.epistemicUncertainty;
    const coverageUrgency = Math.min(2.0, belief.stalenessMs / Math.max(1.0, coverageDeadlineMs));
    const hazard = belief.hazardRate;

    return (
      weights.wIg * ig +
      weights.wUncertainty * (uncertainty / 0.05) +
      weights.wNovelty * novelty +
      weights.wCoverage * coverageUrgency +
      weights.wHazard * hazard
    );
  }

  /**
   * Priority 2 & 5 — Gated Multi-Component Utility with Full MHT Distribution:
   * U(c) = U_base(c) + g_hazard * U_hazard + g_ood * U_ood + g_coverage * U_coverage
   * Integrates the full MHT probability distribution P(f_{t+1}=c) across all channels c.
   */
  public static scoreChannelGated(
    c: number,
    belief: ChannelBelief,
    hopDistribution: Float32Array,
    gates: ComponentGates,
    weights: UtilityWeights = DEFAULT_UTILITY_WEIGHTS
  ): number {
    const hopP = c < hopDistribution.length ? hopDistribution[c] : 1.0 / hopDistribution.length;

    // Fast-path: When all gates indicate a normal, stable environment
    if (gates.isFastPathActive) {
      // Use pure fast exploit path
      return this.scoreExploit(c, belief, hopP, weights);
    }

    // Base exploit utility
    const pD = Math.max(0.05, Math.min(0.99, 1.0 - Math.exp(-belief.occupancyProb * 2.5)));
    const uBase =
      weights.wPd * pD +
      weights.wTarget * belief.occupancyProb +
      (gates.gPu > 0.15 ? weights.wTrans * hopP : 0.4 * weights.wTrans * (1.0 / hopDistribution.length));

    // Gated Hazard Utility
    const uHazard = weights.wHazard * belief.hazardRate + weights.wOpp * belief.opportunityCost;

    // Gated OOD / Novelty Utility
    const novelty = Math.min(1.0, belief.stalenessMs / 150.0);
    const uOod = weights.wNovelty * novelty + weights.wIg * belief.informationGain;

    // Gated Coverage Urgency Utility
    const deadlineMs = 100.0 / (1.0 + 0.18 * belief.threatPriority);
    const uCoverage = weights.wCoverage * (belief.stalenessMs / Math.max(1.0, deadlineMs));

    // Priority bonus
    const priorityBonus = weights.wPriority * (belief.threatPriority / 5.0);

    // Composite gated utility
    return (
      uBase +
      gates.gHazard * uHazard +
      gates.gOod * uOod +
      gates.gCoverage * uCoverage +
      priorityBonus
    );
  }

  /**
   * Priority 8 — Hysteresis Mechanism:
   * Switch only if U(new) > U(current) + Delta
   * Reduces unnecessary retuning, receiver settling loss (35 us), wasted dwell, and oscillation.
   */
  public static applyHysteresis(
    channelUtilities: Float32Array | number[],
    currentChannel: number,
    hysteresisMarginDelta: number = 0.38
  ): { bestChannel: number; maxUtility: number; switched: boolean } {
    let bestChannel = 0;
    let maxUtility = -Infinity;

    for (let c = 0; c < channelUtilities.length; c++) {
      if (channelUtilities[c] > maxUtility) {
        maxUtility = channelUtilities[c];
        bestChannel = c;
      }
    }

    // Evaluate if switching from currentChannel is warranted
    if (
      currentChannel >= 0 &&
      currentChannel < channelUtilities.length &&
      bestChannel !== currentChannel
    ) {
      const currentUtility = channelUtilities[currentChannel];
      // If the advantage of switching is less than the switching margin, stay on current channel!
      if (maxUtility < currentUtility + hysteresisMarginDelta) {
        return {
          bestChannel: currentChannel,
          maxUtility: currentUtility,
          switched: false,
        };
      }
    }

    return {
      bestChannel,
      maxUtility,
      switched: bestChannel !== currentChannel,
    };
  }

  /**
   * Backward-compatible legacy call
   */
  public static scoreChannel(
    c: number,
    belief: ChannelBelief,
    transitionProb: number,
    weights = { trans: 4.8, opp: 2.8, occ: 2.0, ig: 0.9 }
  ): number {
    let utility = 0;
    utility += weights.trans * transitionProb;
    utility += weights.opp * belief.opportunityCost;
    utility += weights.occ * belief.occupancyProb;
    utility += weights.ig * belief.informationGain;
    return utility;
  }
}
