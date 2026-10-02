/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { PulseDescriptorWord } from './types';

// PROVENANCE: copied verbatim from ../src/safety/oodGate.ts (repo root).

export interface ChannelPDWStats {
  meanFreq: number;
  varFreq: number;
  meanPw: number;
  varPw: number;
  meanSnr: number;
  varSnr: number;
  meanPri: number;
  varPri: number;
  lastToaMs: number;
  count: number;
}

/**
 * Hierarchical Multi-Feature Out-of-Distribution (OOD) Gate:
 * Computes calibrated anomaly scores using a 3-tier hierarchy:
 *
 * 1. D_PDW: Mahalanobis distance over physical pulse parameters [carrier freq, pulse width, SNR]
 * 2. D_behavior: Behavioral timing distance over PRI jitter & pulse recurrence
 * 3. D_transition: Markovian state transition surprise -log(P(c_t | c_{t-1}))
 *
 * Composite Score:
 * OODScore = w1 * D_PDW + w2 * D_behavior + w3 * D_transition
 */
export class OODDetector {
  private channelStats: Map<number, ChannelPDWStats> = new Map();
  private lastChannel: number = -1;

  public evaluatePDW(
    pdw: PulseDescriptorWord,
    transitionProb: number = 0.5
  ): { isOOD: boolean; anomalyScore: number; dPdw: number; dBehavior: number; dTransition: number } {
    if (pdw.isFalseAlarm) {
      return { isOOD: false, anomalyScore: 0, dPdw: 0, dBehavior: 0, dTransition: 0 };
    }

    const c = pdw.channelIndex;
    let stat = this.channelStats.get(c);

    if (!stat) {
      // Unseen channel signature -> initial high novelty
      stat = {
        meanFreq: pdw.freqMHz,
        varFreq: 4.0,
        meanPw: pdw.pulseWidthUs,
        varPw: 0.5,
        meanSnr: pdw.snrDb,
        varSnr: 9.0,
        meanPri: 2.0,
        varPri: 1.0,
        lastToaMs: pdw.toaMs,
        count: 1,
      };
      this.channelStats.set(c, stat);
      this.lastChannel = c;
      return { isOOD: false, anomalyScore: 0.05, dPdw: 0.05, dBehavior: 0.05, dTransition: 0.05 };
    }

    // 1. Tier 1: Mahalanobis distance over physical PDW parameters
    const dFreq = Math.pow(pdw.freqMHz - stat.meanFreq, 2) / Math.max(1.0, stat.varFreq);
    const dPw = Math.pow(pdw.pulseWidthUs - stat.meanPw, 2) / Math.max(0.1, stat.varPw);
    const dSnr = Math.pow(pdw.snrDb - stat.meanSnr, 2) / Math.max(2.0, stat.varSnr);
    // Standardized 3-DoF chi-square-like residual
    const rawMahalanobis = Math.sqrt((dFreq + dPw + dSnr) / 3.0);
    const dPdw = Math.min(1.0, rawMahalanobis / 3.5);

    // 2. Tier 2: Behavioral timing distance (PRI deviation)
    let dBehavior = 0.2;
    if (stat.lastToaMs > 0) {
      const observedPri = pdw.toaMs - stat.lastToaMs;
      if (observedPri > 0.05 && observedPri < 50.0) {
        const priResid = Math.abs(observedPri - stat.meanPri) / Math.sqrt(Math.max(0.2, stat.varPri));
        dBehavior = Math.min(1.0, priResid / 3.0);

        // Update PRI statistics
        const alpha = 0.15;
        stat.varPri = (1 - alpha) * stat.varPri + alpha * Math.pow(observedPri - stat.meanPri, 2);
        stat.meanPri = (1 - alpha) * stat.meanPri + alpha * observedPri;
      }
    }
    stat.lastToaMs = pdw.toaMs;

    // 3. Tier 3: Transition pattern surprise
    // Highly unexpected transitions have very small transitionProb
    const surprise = -Math.log2(Math.max(0.01, Math.min(1.0, transitionProb)));
    const dTransition = Math.min(1.0, surprise / 6.64); // normalized against log2(100)

    // Online statistical updates for PDW params
    const lr = 0.1;
    stat.count++;
    stat.varFreq = (1 - lr) * stat.varFreq + lr * Math.pow(pdw.freqMHz - stat.meanFreq, 2);
    stat.meanFreq = (1 - lr) * stat.meanFreq + lr * pdw.freqMHz;
    stat.varPw = (1 - lr) * stat.varPw + lr * Math.pow(pdw.pulseWidthUs - stat.meanPw, 2);
    stat.meanPw = (1 - lr) * stat.meanPw + lr * pdw.pulseWidthUs;
    stat.varSnr = (1 - lr) * stat.varSnr + lr * Math.pow(pdw.snrDb - stat.meanSnr, 2);
    stat.meanSnr = (1 - lr) * stat.meanSnr + lr * pdw.snrDb;
    this.lastChannel = c;

    // Composite hierarchical OOD score:
    // OODScore = w1 * D_PDW + w2 * D_behavior + w3 * D_transition
    const score = 0.45 * dPdw + 0.3 * dBehavior + 0.25 * dTransition;

    return {
      isOOD: score > 0.65,
      anomalyScore: Number(score.toFixed(3)),
      dPdw: Number(dPdw.toFixed(3)),
      dBehavior: Number(dBehavior.toFixed(3)),
      dTransition: Number(dTransition.toFixed(3)),
    };
  }

  public reset(): void {
    this.channelStats.clear();
    this.lastChannel = -1;
  }
}
