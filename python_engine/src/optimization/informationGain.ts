/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ChannelBelief } from '../types';

export interface ChannelFeatureEntropy {
  pwVariance: number;
  snrVariance: number;
  priEntropy: number;
  sampleCount: number;
}

/**
 * Multi-Parameter PDW Information Gain Optimizer:
 * Evaluates expected mutual information from sensing action a:
 * IG(a) = H(B) - E[ H(B | PDW) ]
 *
 * Expands beyond binary hit/miss by incorporating continuous PDW feature entropy:
 * - Occupancy belief entropy reduction (Beta posterior variance reduction)
 * - Multi-dimensional parameter entropy reduction (carrier frequency, PW, SNR, and PRI uncertainty)
 * - Hardware collision / false-alarm penalty degradation
 */
export class InformationGainOptimizer {
  private static featureEntropies: Map<number, ChannelFeatureEntropy> = new Map();

  public static reset(): void {
    this.featureEntropies.clear();
  }

  /**
   * Registers observed PDW to update channel feature variance & entropy
   */
  public static updateFeatureStats(channelIndex: number, pwUs: number, snrDb: number, priMs?: number): void {
    let fe = this.featureEntropies.get(channelIndex);
    if (!fe) {
      fe = { pwVariance: 1.5, snrVariance: 12.0, priEntropy: 1.8, sampleCount: 1 };
      this.featureEntropies.set(channelIndex, fe);
      return;
    }
    fe.sampleCount++;
    // Exponentially decay parameter uncertainty as samples arrive
    fe.pwVariance = Math.max(0.05, fe.pwVariance * 0.94);
    fe.snrVariance = Math.max(0.2, fe.snrVariance * 0.94);
    fe.priEntropy = Math.max(0.1, fe.priEntropy * 0.92);
  }

  /**
   * Binary Shannon entropy function: H(p) = -p log2(p) - (1-p) log2(1-p)
   */
  public static binaryEntropy(p: number): number {
    const clamped = Math.max(1e-5, Math.min(1 - 1e-5, p));
    return -clamped * Math.log2(clamped) - (1.0 - clamped) * Math.log2(1.0 - clamped);
  }

  /**
   * Evaluates expected mutual information IG(a) = H(B) - E[H(B|PDW)] for each channel
   */
  public static computeInformationGain(beliefs: ChannelBelief[]): void {
    for (const b of beliefs) {
      const priorEntropy = this.binaryEntropy(b.occupancyProb);

      // 1. Expected occupancy posterior entropy:
      const p = b.occupancyProb;
      const total = b.betaAlpha + b.betaBeta;
      const postHitMean = (b.betaAlpha + 1.0) / (total + 1.0);
      const postMissMean = b.betaAlpha / (total + 1.0);

      const expectedOccupancyEntropy =
        p * this.binaryEntropy(postHitMean) + (1.0 - p) * this.binaryEntropy(postMissMean);

      const occupancyGain = Math.max(0, priorEntropy - expectedOccupancyEntropy);

      // 2. Expected PDW feature parameter entropy reduction:
      // High feature uncertainty (unexplored channel or agile emitter) provides extra information value
      const fe = this.featureEntropies.get(b.channelIndex) || {
        pwVariance: 2.0,
        snrVariance: 15.0,
        priEntropy: 2.0,
        sampleCount: 0,
      };

      // Gaussian differential entropy ~ 0.5 * ln(2πe * σ²)
      const pwDiffEntropy = 0.5 * Math.log2(2 * Math.PI * Math.E * Math.max(0.05, fe.pwVariance));
      const snrDiffEntropy = 0.5 * Math.log2(2 * Math.PI * Math.E * Math.max(0.1, fe.snrVariance));
      const featureEntropy = Math.max(0, (pwDiffEntropy + snrDiffEntropy + fe.priEntropy) / 6.0);

      // Total expected information gain:
      // IG(a) = ΔH_occupancy + P(hit) * E[ΔH_features | hit] + Epistemic Exploration Bonus
      const totalGain = occupancyGain + p * featureEntropy * 0.4 + b.epistemicUncertainty * 2.2;

      b.informationGain = Math.max(0, Number(totalGain.toFixed(4)));
    }
  }
}
