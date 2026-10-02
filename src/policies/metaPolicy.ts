/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  ChannelBelief,
  CognitiveOperatingMode,
  ComponentGates,
  ScenarioMissionMode,
  TrackConfidenceState,
  AdaptationMetrics,
} from '../state/types';

// PROVENANCE: copied verbatim from ../src/decision/metaPolicy.ts (repo root).

/**
 * Lightweight Learned Meta-Policy & Component Gating Controller:
 * Implements:
 * 1. Fast-Path Exploitation vs. Deliberate Cognitive Reasoning
 * 2. Multi-Component Evidence Gating:
 *    U(a) = U_base(a) + g_hazard * U_hazard + g_ood * U_ood + g_coverage * U_coverage
 * 3. Prediction Usefulness Gate:
 *    PU = (Utility(with_pred) - Utility(without_pred)) / Cost
 * 4. Confirmation Burst State Machine
 * 5. Dynamic Track Confidence C_track in [0, 1]
 * 6. Change-Point Recovery & Adaptation Latency (T_adapt) Tracking
 */
export class CognitiveMetaController {
  private channelCount: number;

  // Operating states
  public currentMode: CognitiveOperatingMode = 'EXPLOIT';
  public missionMode: ScenarioMissionMode = 'general_surveillance';

  // Confirmation burst state
  private confirmationBurstRemaining: number = 0;
  private confirmationTargetChannel: number = -1;

  // Track confidence per channel
  private trackConfidenceMap: Map<number, TrackConfidenceState> = new Map();

  // Change-point recovery tracking
  private changePointTriggerTimeMs: number | null = null;
  private preShiftMitWindow: number[] = [];
  private postShiftMitWindow: number[] = [];
  public adaptationMetrics: AdaptationMetrics = {
    tAdaptMs: 0,
    perfBeforeShift: 1.0,
    perfAfterShift: 1.0,
    recoveryRatio: 1.0,
    totalChangePoints: 0,
    meanAdaptationTimeMs: 0,
  };

  // Prediction accuracy tracking for PU gate
  private predictionUsefulnessHistory: number[] = [];
  private recentPredictionAgreements: boolean[] = [];

  constructor(channelCount: number = 32) {
    this.channelCount = channelCount;
    this.reset();
  }

  public reset(): void {
    this.currentMode = 'EXPLOIT';
    this.confirmationBurstRemaining = 0;
    this.confirmationTargetChannel = -1;
    this.changePointTriggerTimeMs = null;
    this.preShiftMitWindow = [];
    this.postShiftMitWindow = [];
    this.predictionUsefulnessHistory = [];
    this.recentPredictionAgreements = [];
    this.adaptationMetrics = {
      tAdaptMs: 0,
      perfBeforeShift: 1.0,
      perfAfterShift: 1.0,
      recoveryRatio: 1.0,
      totalChangePoints: 0,
      meanAdaptationTimeMs: 0,
    };

    this.trackConfidenceMap.clear();
    for (let c = 0; c < this.channelCount; c++) {
      this.trackConfidenceMap.set(c, {
        channelIndex: c,
        cTrack: 0.2, // Prior uncertainty
        regime: 'explore',
        observationCount: 0,
        predictionConsistency: 0.5,
        meanSnr: 12.0,
        stalenessMs: 0,
      });
    }

    this.adaptationMetrics = {
      tAdaptMs: 0,
      perfBeforeShift: 1.0,
      perfAfterShift: 1.0,
      recoveryRatio: 1.0,
      totalChangePoints: 0,
      meanAdaptationTimeMs: 0,
    };
  }

  /**
   * Priority 11 — Track Confidence:
   * C_track = f(observations, prediction consistency, SNR, age, OOD, transition consistency)
   */
  public updateTrackConfidence(
    channel: number,
    hit: boolean,
    snrDb: number,
    isOod: boolean,
    wasPredictionCorrect: boolean,
    stalenessMs: number
  ): TrackConfidenceState {
    let state = this.trackConfidenceMap.get(channel);
    if (!state) {
      state = {
        channelIndex: channel,
        cTrack: 0.2,
        regime: 'explore',
        observationCount: 0,
        predictionConsistency: 0.5,
        meanSnr: snrDb,
        stalenessMs,
      };
      this.trackConfidenceMap.set(channel, state);
    }

    state.stalenessMs = stalenessMs;
    if (hit) {
      state.observationCount++;
      state.meanSnr = 0.85 * state.meanSnr + 0.15 * snrDb;
    }

    // Prediction consistency running average
    const predScore = wasPredictionCorrect ? 1.0 : 0.1;
    state.predictionConsistency = 0.8 * state.predictionConsistency + 0.2 * predScore;
    this.recentPredictionAgreements.push(wasPredictionCorrect);
    if (this.recentPredictionAgreements.length > 20) this.recentPredictionAgreements.shift();

    // Composite Track Confidence Calculation:
    // C_track = w_obs * f_obs + w_pred * predCons + w_snr * snrNorm - w_ood * OOD - w_stale * stalenessDecay
    const fObs = Math.min(1.0, state.observationCount / 4.0);
    const snrNorm = Math.min(1.0, Math.max(0.1, (state.meanSnr - 6.0) / 18.0));
    const stalenessFactor = Math.exp(-stalenessMs / 120.0);
    const oodPenalty = isOod ? 0.45 : 0.0;

    const rawConfidence =
      0.35 * fObs +
      0.25 * state.predictionConsistency +
      0.20 * snrNorm +
      0.20 * stalenessFactor -
      oodPenalty;

    state.cTrack = Math.min(0.99, Math.max(0.05, rawConfidence));

    if (state.cTrack > 0.85) {
      state.regime = 'exploit';
    } else if (state.cTrack > 0.45) {
      state.regime = 'verify';
    } else {
      state.regime = 'explore';
    }

    return state;
  }

  /**
   * Priority 2 & 4 — Component Gating & Prediction Usefulness:
   * Calculates whether each intelligent component actually has enough evidence to matter.
   */
  public evaluateGates(
    beliefs: ChannelBelief[],
    maxHopProb: number,
    oodScore: number,
    imminentHazardProb: number,
    coverageViolationRatio: number,
    isChangePointActive: boolean
  ): ComponentGates {
    // 1. g_ood = P(OOD)
    const gOod = Math.min(1.0, Math.max(0.0, (oodScore - 0.35) / 0.45));

    // 2. g_hazard = P(imminent event)
    const gHazard = Math.min(1.0, Math.max(0.0, imminentHazardProb));

    // 3. g_coverage = P(T_stale > threshold)
    const gCoverage = Math.min(1.0, Math.max(0.0, (coverageViolationRatio - 0.65) / 0.35));

    // 4. g_mht: activates when multiple candidates compete (ambiguity)
    let entropyH = 0;
    for (let i = 0; i < Math.min(4, beliefs.length); i++) {
      const p = beliefs[i].occupancyProb;
      if (p > 0.05 && p < 0.95) {
        entropyH += -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
      }
    }
    const gMht = Math.min(1.0, Math.max(0.0, entropyH / 3.0));

    // 5. g_ig: activates when epistemic uncertainty is elevated
    let maxUncertainty = 0;
    for (const b of beliefs) {
      if (b.epistemicUncertainty > maxUncertainty) {
        maxUncertainty = b.epistemicUncertainty;
      }
    }
    const gIg = Math.min(1.0, maxUncertainty / 0.07);

    // 6. g_cp: change-point alarm
    const gCp = isChangePointActive ? 1.0 : 0.0;

    // 7. g_pu: Prediction Usefulness Gate (Priority 4)
    // PU = (Utility(with_pred) - Utility(without_pred)) / Cost
    // If prediction confidence is low or agreement is poor, PU < tau
    const recentAgreementRate =
      this.recentPredictionAgreements.length > 0
        ? this.recentPredictionAgreements.filter(Boolean).length /
          this.recentPredictionAgreements.length
        : 0.6;
    const pu = (maxHopProb * recentAgreementRate - 0.15) / 0.85;
    const gPu = Math.min(1.0, Math.max(0.0, pu));

    // Fast-path evaluation:
    // When environment is normal (no OOD, no coverage crisis, no change point, low epistemic noise)
    const isFastPathActive =
      gOod < 0.25 &&
      gCoverage < 0.3 &&
      !isChangePointActive &&
      this.confirmationBurstRemaining === 0;

    return {
      gOod,
      gHazard,
      gCoverage,
      gMht,
      gIg,
      gCp,
      gPu,
      isFastPathActive,
    };
  }

  /**
   * Priority 1 & 15 — Meta-Policy Mode Selector:
   * Evaluates input vector x_t:
   * x_t = [uncertainty, OOD, staleness, hazard, predConfidence, trackConfidence, disagreement, SNR]
   * Outputs regime: EXPLOIT, EXPLORE, VERIFY, COVERAGE, RECOVER
   */
  public selectOperatingMode(
    gates: ComponentGates,
    maxStalenessRatio: number,
    bestTrackConfidence: number
  ): CognitiveOperatingMode {
    // 1. Coverage emergency takes safety priority
    if (gates.gCoverage >= 0.95 || maxStalenessRatio >= 1.0) {
      this.currentMode = 'COVERAGE';
      return 'COVERAGE';
    }

    // 2. Change point or severe OOD triggers RECOVER mode
    if (gates.gCp > 0.5 || gates.gOod > 0.75) {
      this.currentMode = 'RECOVER';
      return 'RECOVER';
    }

    // 3. Active Confirmation Burst takes precedence for 2-3 dwells
    if (this.confirmationBurstRemaining > 0) {
      this.currentMode = 'VERIFY';
      return 'VERIFY';
    }

    // 4. High track confidence / low uncertainty triggers aggressive EXPLOIT mode
    if (bestTrackConfidence > 0.75 && gates.gOod < 0.3 && gates.gIg < 0.5) {
      this.currentMode = 'EXPLOIT';
      return 'EXPLOIT';
    }

    // 5. Moderate track confidence with recent suspicion -> VERIFY
    if (bestTrackConfidence >= 0.45 && bestTrackConfidence <= 0.75) {
      this.currentMode = 'VERIFY';
      return 'VERIFY';
    }

    // 6. Otherwise: high uncertainty / staleness requires EXPLORE mode
    this.currentMode = 'EXPLORE';
    return 'EXPLORE';
  }

  /**
   * Priority 10 — Confirmation Burst Trigger:
   * Call when suspicious channel is detected (P_target > 0.7 or new emitter hit)
   */
  public triggerConfirmationBurst(channelIndex: number, burstCount: number = 2): void {
    this.confirmationTargetChannel = channelIndex;
    this.confirmationBurstRemaining = burstCount;
  }

  public getConfirmationState(): { active: boolean; channel: number; remaining: number } {
    return {
      active: this.confirmationBurstRemaining > 0,
      channel: this.confirmationTargetChannel,
      remaining: this.confirmationBurstRemaining,
    };
  }

  public decrementConfirmationBurst(): void {
    if (this.confirmationBurstRemaining > 0) {
      this.confirmationBurstRemaining--;
      if (this.confirmationBurstRemaining === 0) {
        this.confirmationTargetChannel = -1;
      }
    }
  }

  /**
   * Priority 12 — Change-Point Recovery & Adaptation Latency (T_adapt)
   */
  public onChangePointDetected(currentTimeMs: number): void {
    this.changePointTriggerTimeMs = currentTimeMs;
    this.adaptationMetrics.totalChangePoints++;
    // Boost explore mode & reset confirmation bursts
    this.confirmationBurstRemaining = 0;
  }

  public recordInterceptionDelay(delayMs: number, currentTimeMs: number): void {
    if (this.changePointTriggerTimeMs === null) {
      this.preShiftMitWindow.push(delayMs);
      if (this.preShiftMitWindow.length > 30) this.preShiftMitWindow.shift();
    } else {
      this.postShiftMitWindow.push(delayMs);
      // Measure adaptation latency T_adapt when post-shift intercepts stabilize (>= 8 samples)
      if (this.postShiftMitWindow.length >= 8 && this.adaptationMetrics.tAdaptMs === 0) {
        this.adaptationMetrics.tAdaptMs = Number(
          (currentTimeMs - this.changePointTriggerTimeMs).toFixed(1)
        );
        this.adaptationMetrics.meanAdaptationTimeMs = this.adaptationMetrics.tAdaptMs;

        const meanPre =
          this.preShiftMitWindow.length > 0
            ? this.preShiftMitWindow.reduce((a, b) => a + b, 0) / this.preShiftMitWindow.length
            : 1.0;
        const meanPost =
          this.postShiftMitWindow.reduce((a, b) => a + b, 0) / this.postShiftMitWindow.length;

        this.adaptationMetrics.perfBeforeShift = Number(meanPre.toFixed(2));
        this.adaptationMetrics.perfAfterShift = Number(meanPost.toFixed(2));
        this.adaptationMetrics.recoveryRatio = Number((meanPre / Math.max(0.1, meanPost)).toFixed(2));
      }
    }
  }

  public getAdaptationMetrics(): AdaptationMetrics {
    return { ...this.adaptationMetrics };
  }

  public getTrackConfidence(channel: number): number {
    return this.trackConfidenceMap.get(channel)?.cTrack ?? 0.2;
  }

  public getTrackConfidenceMap(): Map<number, TrackConfidenceState> {
    return this.trackConfidenceMap;
  }
}
