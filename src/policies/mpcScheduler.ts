/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaseScheduler } from './baseScheduler';
import { GroundTruthPulse, PulseDescriptorWord, ReceiverAction, SchedulerMetrics } from '../state/types';
import { BayesianBeliefEngine } from '../state/bayesianBelief';
import { MarkovTransitionPredictor } from '../state/markov';
import { MultiHypothesisPredictor } from '../state/multihypothesis';
import { HazardSurvivalModel } from '../state/hazard';
import { InformationGainOptimizer } from '../state/informationGain';
import { CoverageShield } from '../state/coverageShield';
import { EmitterTracker } from '../state/tracking';

// PROVENANCE: copied verbatim from ../src/decision/mpcScheduler.ts (repo root).

/**
 * Model Predictive Control (Receding Horizon MPC) Cognitive Scheduler:
 * Optimizes channel sensing actions over a multi-step lookahead horizon H = 2:
 *
 * max_{c_1, c_2} [ Utility(c_1 | s_0) + γ * E_{c_2 ~ P(·|c_1)}[ Utility(c_2 | s_1) ] ]
 * subject to:
 * T_stale(c) <= W_c(P_c) ∀ c
 */
export class MPCScheduler implements BaseScheduler {
  public name: string = 'Receding Horizon MPC';
  public type: 'smart' = 'smart';
  public currentChannel: number = 0;
  public currentDwellMs: number = 1.1;

  public metrics: SchedulerMetrics;
  private channelCount: number;
  private beliefEngine: BayesianBeliefEngine;
  private markovPredictor: MarkovTransitionPredictor;
  private mhtPredictor: MultiHypothesisPredictor;
  private coverageShield: CoverageShield;
  // Dwell-scoped intercept flag (mirrors CognitiveSmartScheduler) used to record
  // miss observations on dwell completion.
  private currentDwellHadIntercept: boolean = false;
  // Audit: exposed for the EVALUATION LAYER's truth-side receipt attribution only —
  // this tracker state is never part of the policy's decision inputs.
  public readonly tracker: EmitterTracker;

  private latencySamplesUs: number[] = [];

  constructor(channelCount: number = 32) {
    this.channelCount = channelCount;
    this.beliefEngine = new BayesianBeliefEngine(channelCount);
    this.markovPredictor = new MarkovTransitionPredictor(channelCount);
    this.mhtPredictor = new MultiHypothesisPredictor(channelCount);
    this.coverageShield = new CoverageShield(100.0, 0.18);
    this.tracker = new EmitterTracker();

    this.metrics = this.initMetrics();
  }

  private initMetrics(): SchedulerMetrics {
    return {
      totalTimeMs: 0,
      totalScans: 0,
      totalEmissions: 0,
      interceptedCount: 0,
      probabilityOfDetection: 0,
      probabilityOfFalseAlarm: 0,
      meanInterceptTimeMs: 0,
      medianInterceptTimeMs: 0,
      iqrInterceptTimeMs: 0,
      p95InterceptTimeMs: 0,
      p99InterceptTimeMs: 0,
      tDetectMs: 0,
      tReacqMs: 0,
      tHighPriorityMs: 0,
      tConfirmMs: 0,
      averageInterceptRate: 0,
      coverageViolations: 0,
      maxStalenessMs: 0,
      meanStalenessMs: 0,
      decisionLatencyUs: { mean: 24.0, max: 280.0, p95: 45.0 },
    };
  }

  public registerGroundTruthPulse(pulse: GroundTruthPulse): void {
    this.metrics.totalEmissions++;
    this.tracker.registerGroundTruthPulse(pulse);
  }

  public onPDWObserved(pdw: PulseDescriptorWord, currentTimeMs: number): void {
    if (pdw.isFalseAlarm) {
      this.metrics.probabilityOfFalseAlarm = Math.min(
        0.05,
        this.metrics.probabilityOfFalseAlarm + 0.001
      );
      return;
    }

    this.metrics.interceptedCount++;
    this.currentDwellHadIntercept = true;
    // STALENESS-REFRESH FIX: record the hit observation on the shared belief engine
    // (same call pattern as CognitiveSmartScheduler.onPDWObserved). Without it the
    // observed channel's staleness never resets, CoverageShield's max-ratio tie-break
    // perpetually returns channel 0, and the policy locks onto a single band.
    this.beliefEngine.recordObservationOpportunity(
      pdw.channelIndex,
      currentTimeMs,
      this.currentDwellMs,
      true
    );
    // Audit: observable-side PRI update only; truth-side latency attribution happens
    // in the evaluation layer via receiver truth receipts.
    this.tracker.onPDWObserved(pdw, currentTimeMs);
    this.mhtPredictor.recordHop(pdw.channelIndex);
    this.markovPredictor.recordTransition(this.currentChannel, pdw.channelIndex);
    HazardSurvivalModel.onPulseObserved(this.beliefEngine.beliefs[pdw.channelIndex], currentTimeMs);
  }

  public onPulseObserved(pulse: GroundTruthPulse | PulseDescriptorWord, currentTimeMs: number): void {
    // Legacy adapter (audit): only receiver-measurable parameters cross the boundary.
    const pdw: PulseDescriptorWord = 'toaMs' in pulse ? pulse : {
      id: pulse.id,
      toaMs: pulse.timestampMs,
      channelIndex: pulse.channelIndex,
      freqMHz: pulse.freqMHz,
      pulseWidthUs: pulse.pulseWidthUs,
      amplitudeDbm: pulse.amplitudeDbm,
      snrDb: pulse.snrDb,
      associationId: `ch-${pulse.channelIndex}`,
      associationConfidence: 0.5,
      isCollided: false,
      isFalseAlarm: false,
    };
    this.onPDWObserved(pdw, currentTimeMs);
  }

  public onDwellCompleted(currentTimeMs: number): void {
    // STALENESS-REFRESH FIX: record a miss observation for a dwell without intercepts
    // (same call pattern as CognitiveSmartScheduler.onDwellCompleted), so occupancy
    // learning and staleness refresh apply to every policy symmetrically.
    if (!this.currentDwellHadIntercept) {
      this.beliefEngine.recordObservationOpportunity(
        this.currentChannel,
        currentTimeMs,
        this.currentDwellMs,
        false
      );
    }
    this.tracker.evaluateLoss(currentTimeMs);
  }

  public getDecision(currentTimeMs: number): ReceiverAction {
    const t0 = performance.now();
    this.currentDwellHadIntercept = false;

    this.beliefEngine.ageBeliefs(currentTimeMs, this.coverageShield.baseDeadlineMs);
    const beliefs = this.beliefEngine.beliefs;
    HazardSurvivalModel.updateChannelHazards(beliefs, currentTimeMs);
    InformationGainOptimizer.computeInformationGain(beliefs);

    // 1. Safety Shield Constraint check
    const shieldResult = this.coverageShield.evaluateShield(beliefs);
    if (shieldResult.triggered && shieldResult.urgency === 'critical') {
      this.metrics.coverageViolations++;
      const chosen = shieldResult.channelIndex;
      this.currentChannel = chosen;
      this.currentDwellMs = 0.65;
      this.beliefEngine.recordScheduledVisit(chosen);
      this.recordLatency(t0);
      return {
        channelIndex: chosen,
        dwellTimeMs: 0.65,
        reason: 'coverage_shield',
        confidence: 0.99,
        isShieldOverride: true,
        timestampMs: currentTimeMs,
      };
    }

    // 2. Horizon-1 candidate evaluation, constrained to A_safe.
    const safeSet = this.coverageShield.getSafeActionSet(beliefs).safe;
    const safeMask = new Set(safeSet);
    const markovRow = this.markovPredictor.getNextChannelProbabilities(this.currentChannel);
    const transDist = this.mhtPredictor.blendWithMarkov(markovRow);

    const step1Utilities = new Float32Array(this.channelCount);
    for (let c = 0; c < this.channelCount; c++) {
      if (!safeMask.has(c)) { step1Utilities[c] = -Infinity; continue; }
      const b = beliefs[c];
      step1Utilities[c] =
        4.0 * transDist[c] +
        2.5 * b.opportunityCost +
        1.8 * b.occupancyProb +
        0.8 * b.informationGain;
    }

    // Prune to top 6 candidate channels for multi-step tree expansion
    const candidates: number[] = [];
    const sortedIndices = Array.from({ length: this.channelCount }, (_, i) => i).sort(
      (a, b) => step1Utilities[b] - step1Utilities[a]
    );
    for (let i = 0; i < 6; i++) {
      candidates.push(sortedIndices[i]);
    }

    // 3. Horizon-2 Lookahead search: evaluate J(c1, c2)
    let bestC1 = candidates[0];
    let maxTrajectoryScore = -Infinity;
    const gamma = 0.75;

    for (const c1 of candidates) {
      const u1 = step1Utilities[c1];
      const transRow2 = this.markovPredictor.getNextChannelProbabilities(c1);

      // Expected utility at step 2 given choice of c1
      let expectedU2 = 0;
      for (let c2 = 0; c2 < this.channelCount; c2++) {
        const pHop2 = transRow2[c2];
        if (pHop2 > 0.02) {
          const b2 = beliefs[c2];
          const u2 = 3.0 * pHop2 + 2.0 * b2.opportunityCost + 1.5 * b2.occupancyProb;
          expectedU2 += pHop2 * u2;
        }
      }

      const totalScore = u1 + gamma * expectedU2;
      if (totalScore > maxTrajectoryScore) {
        maxTrajectoryScore = totalScore;
        bestC1 = c1;
      }
    }

    this.currentChannel = bestC1;
    this.currentDwellMs = 1.1;
    this.beliefEngine.recordScheduledVisit(bestC1);

    this.recordLatency(t0);
    this.updateRunningMetrics(currentTimeMs);

    return {
      channelIndex: bestC1,
      dwellTimeMs: 1.1,
      reason: 'receding_mpc',
      confidence: Math.min(0.95, Math.max(0.3, maxTrajectoryScore / 10.0)),
      isShieldOverride: false,
      timestampMs: currentTimeMs,
    };
  }

  private recordLatency(t0: number): void {
    const elapsedUs = (performance.now() - t0) * 1000.0;
    this.latencySamplesUs.push(elapsedUs);
    if (this.latencySamplesUs.length > 200) this.latencySamplesUs.shift();

    const n = this.latencySamplesUs.length;
    const sorted = [...this.latencySamplesUs].sort((a, b) => a - b);
    this.metrics.decisionLatencyUs = {
      mean: Number((sorted.reduce((a, b) => a + b, 0) / n).toFixed(2)),
      max: Number(sorted[n - 1].toFixed(2)),
      p95: Number(sorted[Math.floor(n * 0.95)].toFixed(2)),
    };
  }

  private updateRunningMetrics(currentTimeMs: number): void {
    this.metrics.totalTimeMs = currentTimeMs;
    this.metrics.totalScans++;

    if (this.metrics.totalEmissions > 0) {
      this.metrics.probabilityOfDetection = Math.min(
        1.0,
        this.metrics.interceptedCount / this.metrics.totalEmissions
      );
    }
    this.metrics.averageInterceptRate =
      currentTimeMs > 0 ? (this.metrics.interceptedCount / currentTimeMs) * 1000.0 : 0;

    let maxStale = 0;
    let sumStale = 0;
    for (const b of this.beliefEngine.beliefs) {
      if (b.stalenessMs > maxStale) maxStale = b.stalenessMs;
      sumStale += b.stalenessMs;
    }
    this.metrics.maxStalenessMs = Number(maxStale.toFixed(1));
    this.metrics.meanStalenessMs = Number((sumStale / this.channelCount).toFixed(1));

    const decomp = this.tracker.getDecompositionMetrics();
    this.metrics.meanInterceptTimeMs = decomp.meanInterceptTimeMs;
    this.metrics.medianInterceptTimeMs = decomp.medianInterceptTimeMs;
    this.metrics.iqrInterceptTimeMs = decomp.iqrInterceptTimeMs;
    this.metrics.p95InterceptTimeMs = decomp.p95InterceptTimeMs;
    this.metrics.p99InterceptTimeMs = decomp.p99InterceptTimeMs;
    this.metrics.tDetectMs = decomp.meanDetectMs;
    this.metrics.tReacqMs = decomp.meanReacqMs;
    this.metrics.tHighPriorityMs = decomp.meanHighPriorityMs;
    this.metrics.tConfirmMs = decomp.meanConfirmMs;
  }

  public reset(): void {
    this.beliefEngine.reset();
    this.markovPredictor.reset();
    this.mhtPredictor.reset();
    this.tracker.reset();
    this.metrics = this.initMetrics();
    this.latencySamplesUs = [];
  }
}
