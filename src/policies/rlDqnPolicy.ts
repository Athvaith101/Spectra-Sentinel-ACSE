/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaseScheduler } from './baseScheduler';
import { GroundTruthPulse, PulseDescriptorWord, ReceiverAction, SchedulerMetrics } from '../state/types';
import { BayesianBeliefEngine } from '../state/bayesianBelief';
import { MarkovTransitionPredictor } from '../state/markov';
import { MultiHypothesisPredictor } from '../state/multihypothesis';
import { CoverageShield } from '../state/coverageShield';
import { EmitterTracker } from '../state/tracking';
import { SeededRNG } from '../state/seededRng';

// PROVENANCE: copied verbatim from ../src/decision/rlDqnPolicy.ts (repo root).

/**
 * Deep Q-Network (DQN / Double DQN) Learned Cognitive Policy Scheduler:
 * Employs a reinforcement learning agent learning state-action value functions Q(s, a).
 *
 * State Vector s_t:
 * - Occupancy belief E[p_c] for each channel c
 * - Normalized staleness T_stale(c) / W_c(P_c)
 * - Threat priority P_c / 5.0
 * - Blended transition probability P_trans(c)
 *
 * Action Space:
 * Channel selection a_t ∈ {0, ..., C-1}
 *
 * Architecture:
 * Final Action = CoverageShield( argmax_a Q(s_t, a) )
 */
export class RLDQNScheduler implements BaseScheduler {
  public name: string = 'Learned RL Cognitive (DQN)';
  public type: 'smart' = 'smart';
  public currentChannel: number = 0;
  public currentDwellMs: number = 1.1;

  public metrics: SchedulerMetrics;
  private channelCount: number;
  private beliefEngine: BayesianBeliefEngine;
  private markovPredictor: MarkovTransitionPredictor;
  private mhtPredictor: MultiHypothesisPredictor;
  private coverageShield: CoverageShield;
  // Audit: exposed for the EVALUATION LAYER's truth-side receipt attribution only —
  // this tracker state is never part of the policy's decision inputs.
  public readonly tracker: EmitterTracker;

  // Q-Learning weights matrix: [channelCount x featureDim]
  // Feature vector per channel: [pOcc, stalenessRatio, priorityNorm, transProb, bias]
  private readonly featureDim = 5;
  private qWeights: Float32Array;
  private targetWeights: Float32Array; // Double DQN target network
  private gamma: number = 0.85;        // Discount factor
  private lr: number = 0.04;           // Learning rate
  private epsilon: number = 0.05;      // Exploration floor

  private lastState: Float32Array | null = null;
  private lastAction: number = 0;
  private totalSteps: number = 0;
  private rng: SeededRNG;

  // Latency telemetry
  private latencySamplesUs: number[] = [];
  // Dwell-scoped intercept flag (mirrors CognitiveSmartScheduler) used to record
  // miss observations on dwell completion.
  private currentDwellHadIntercept: boolean = false;

  constructor(channelCount: number = 32, seed = 42) {
    this.channelCount = channelCount;
    this.rng = new SeededRNG(seed);
    this.beliefEngine = new BayesianBeliefEngine(channelCount);
    this.markovPredictor = new MarkovTransitionPredictor(channelCount);
    this.mhtPredictor = new MultiHypothesisPredictor(channelCount);
    this.coverageShield = new CoverageShield(100.0, 0.18);
    this.tracker = new EmitterTracker();

    this.qWeights = new Float32Array(this.channelCount * this.featureDim);
    this.targetWeights = new Float32Array(this.channelCount * this.featureDim);
    this.initializeWeights();

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
      decisionLatencyUs: { mean: 18.0, max: 220.0, p95: 35.0 },
    };
  }

  private initializeWeights(): void {
    // Seed with initial inductive bias (positive reward for occupancy, priority, and transition)
    for (let c = 0; c < this.channelCount; c++) {
      const offset = c * this.featureDim;
      this.qWeights[offset + 0] = 2.5; // pOcc
      this.qWeights[offset + 1] = 1.8; // stalenessRatio
      this.qWeights[offset + 2] = 2.0; // priority
      this.qWeights[offset + 3] = 3.2; // transProb
      this.qWeights[offset + 4] = 0.5; // bias
    }
    this.targetWeights.set(this.qWeights);
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

    // Online Q-learning reward update (positive reward for intercept)
    if (this.lastState !== null) {
      const reward = 10.0 + (pdw.snrDb > 15 ? 2.0 : 0);
      this.updateQValue(this.lastState, this.lastAction, reward);
    }
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
    this.totalSteps++;

    // 1. Refresh belief staleness
    this.beliefEngine.ageBeliefs(currentTimeMs, this.coverageShield.baseDeadlineMs);
    const beliefs = this.beliefEngine.beliefs;

    // 2. Check Coverage Safety Shield first (Constraint)
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

    // 3. Extract feature state representation
    const state = this.extractStateVector(beliefs);
    this.lastState = state;

    // 4. Project the learned policy onto the hard-safe action set.
    const feasible = this.coverageShield.getSafeActionSet(beliefs);
    const feasibleSet = new Set(feasible.safe);
    let bestAction = feasible.safe[0] ?? 0;
    let maxQ = -Infinity;

    for (const c of feasible.safe) {
      const q = this.computeQValue(state, c, this.qWeights);
      if (q > maxQ) {
        maxQ = q;
        bestAction = c;
      }
    }

    // Epsilon-greedy exploration
    if (!feasible.forcedRecovery && this.rng.next() < this.epsilon) {
      bestAction = feasible.safe[this.rng.int(0, feasible.safe.length - 1)];
    }

    this.lastAction = bestAction;
    this.currentChannel = bestAction;
    this.currentDwellMs = 1.1;
    this.beliefEngine.recordScheduledVisit(bestAction);

    // Periodically sync Double DQN target network
    if (this.totalSteps % 100 === 0) {
      this.targetWeights.set(this.qWeights);
    }

    this.recordLatency(t0);
    this.updateRunningMetrics(currentTimeMs);

    return {
      channelIndex: bestAction,
      dwellTimeMs: 1.1,
      reason: 'fast_exploit',
      confidence: Math.min(0.95, Math.max(0.2, maxQ / 15.0)),
      isShieldOverride: false,
      timestampMs: currentTimeMs,
    };
  }

  private extractStateVector(beliefs: typeof this.beliefEngine.beliefs): Float32Array {
    const s = new Float32Array(this.channelCount * this.featureDim);
    const markovRow = this.markovPredictor.getNextChannelProbabilities(this.currentChannel);
    const transDist = this.mhtPredictor.blendWithMarkov(markovRow);

    for (let c = 0; c < this.channelCount; c++) {
      const b = beliefs[c];
      const deadline = this.coverageShield.getDeadlineMs(b.threatPriority);
      const offset = c * this.featureDim;

      s[offset + 0] = b.occupancyProb;
      s[offset + 1] = Math.min(2.0, b.stalenessMs / deadline);
      s[offset + 2] = b.threatPriority / 5.0;
      s[offset + 3] = transDist[c];
      s[offset + 4] = 1.0; // bias term
    }
    return s;
  }

  private computeQValue(state: Float32Array, action: number, weights: Float32Array): number {
    const offset = action * this.featureDim;
    let q = 0;
    for (let i = 0; i < this.featureDim; i++) {
      q += state[offset + i] * weights[offset + i];
    }
    return q;
  }

  private updateQValue(state: Float32Array, action: number, reward: number): void {
    const currentQ = this.computeQValue(state, action, this.qWeights);

    // Double DQN target estimation: argmax over online network, value from target network
    let bestNextAction = 0;
    let maxNextQ = -Infinity;
    for (let c = 0; c < this.channelCount; c++) {
      const qOnline = this.computeQValue(state, c, this.qWeights);
      if (qOnline > maxNextQ) {
        maxNextQ = qOnline;
        bestNextAction = c;
      }
    }
    const targetQVal = this.computeQValue(state, bestNextAction, this.targetWeights);
    const target = reward + this.gamma * targetQVal;
    const tdError = target - currentQ;

    // Gradient descent step on weights
    const offset = action * this.featureDim;
    for (let i = 0; i < this.featureDim; i++) {
      this.qWeights[offset + i] += this.lr * tdError * state[offset + i];
      // Clamp weights
      this.qWeights[offset + i] = Math.max(-20, Math.min(20, this.qWeights[offset + i]));
    }
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

  public reset(seed = 42): void {
    this.rng = new SeededRNG(seed);
    this.beliefEngine.reset();
    this.markovPredictor.reset();
    this.mhtPredictor.reset();
    this.tracker.reset();
    this.metrics = this.initMetrics();
    this.latencySamplesUs = [];
    this.lastState = null;
    this.totalSteps = 0;
  }
}
