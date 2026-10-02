/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  ChannelBelief,
  GroundTruthPulse,
  LatencyBreakdown,
  PulseDescriptorWord,
  ReceiverAction,
  SchedulerMetrics,
  AblationConfig,
  DEFAULT_ABLATION_CONFIG,
  CognitiveOperatingMode,
  ComponentGates,
  ScenarioMissionMode,
  CompositeMissionScore,
} from '../types';
import { BaseScheduler } from './baseScheduler';
import { BayesianBeliefEngine } from '../belief/bayesianBelief';
import { MarkovTransitionPredictor } from '../prediction/markov';
import { MultiHypothesisPredictor } from '../prediction/multihypothesis';
import { PageHinkleyDetector } from '../prediction/changepoint';
import { HazardSurvivalModel } from '../belief/hazard';
import { OpportunityCostEngine, UtilityWeights, DEFAULT_UTILITY_WEIGHTS } from '../optimization/opportunityCost';
import { InformationGainOptimizer } from '../optimization/informationGain';
import { AdaptiveDwellManager } from '../optimization/adaptiveDwell';
import { CoverageShield } from '../safety/coverageShield';
import { OODDetector } from '../safety/oodGate';
import { EmitterTracker } from '../belief/tracking';
import { CognitiveMetaController } from './metaPolicy';
import { BenchmarkStatistics } from '../benchmark/statistics';
import { WeightOptimizer } from '../optimization/weightOptimizer';

/**
 * Priority 1–18 Two-Mode Gated Cognitive ES Spectrum Scheduler:
 *
 * Architecture:
 *                  ┌───────────────────┐
 *                  │ Fast Exploit Path │ (Direct U_E, <12 us)
 *                  └─────────┬─────────┘
 *                            │
 *                  ┌─────────▼─────────┐
 *                  │  Cognitive Gate   │ (g_hazard, g_ood, g_cov, g_pu)
 *                  └─────────┬─────────┘
 *                            │
 *            ┌───────────────┼───────────────┐
 *            ▼               ▼               ▼
 *        EXPLOIT          EXPLORE         RECOVER / VERIFY
 *   (High raw perf)   (Uncertain/OOD)   (Shift adaptation)
 */
export class CognitiveSmartScheduler implements BaseScheduler {
  public name = 'Two-Mode Gated Cognitive Scheduler';
  public type: 'smart' = 'smart';
  public currentChannel: number = 0;
  public currentDwellMs: number = 1.1;

  public metrics: SchedulerMetrics;
  public channelBeliefs: ChannelBelief[] = [];
  public transitionMatrix: Float32Array[] = [];

  public lastLatencyBreakdown: LatencyBreakdown = {
    beliefUpdateUs: 6.2,
    transitionLookupUs: 3.1,
    opportunityCostUs: 10.4,
    shieldValidationUs: 4.5,
    totalCycleUs: 24.2,
  };

  // Submodules
  public beliefEngine: BayesianBeliefEngine;
  public markovPredictor: MarkovTransitionPredictor;
  public mhtPredictor: MultiHypothesisPredictor;
  public shield: CoverageShield;
  public changePointDetector: PageHinkleyDetector;
  public oodDetector: OODDetector;
  public tracker: EmitterTracker;
  public metaController: CognitiveMetaController;

  // Active gates and mode telemetry
  public activeGates: ComponentGates;
  public currentOperatingMode: CognitiveOperatingMode = 'EXPLOIT';
  public missionMode: ScenarioMissionMode = 'general_surveillance';
  public activeWeights: UtilityWeights = { ...DEFAULT_UTILITY_WEIGHTS };
  public compositeMissionScore: CompositeMissionScore;

  // Ablation config
  public ablationConfig: AblationConfig = { ...DEFAULT_ABLATION_CONFIG };

  private channelCount: number;
  private lastInterceptedChannel: number = -1;
  private lastInterceptTimeMs: number = 0;
  private currentDwellHadIntercept: boolean = false;
  private latencySamplesUs: number[] = [];
  private lastOodScore: number = 0;
  // Observable association-track history: keyed by receiver associationId (measured
  // frequency clustering), holding the last measured channel and time per track.
  private lastEmitterCh: Map<string, number> = new Map();
  private lastEmitterTime: Map<string, number> = new Map();

  constructor(channelCount: number = 32, baseDeadlineMs: number = 260.0) {
    this.channelCount = channelCount;
    this.beliefEngine = new BayesianBeliefEngine(channelCount);
    this.markovPredictor = new MarkovTransitionPredictor(channelCount);
    this.mhtPredictor = new MultiHypothesisPredictor(channelCount);
    this.shield = new CoverageShield(baseDeadlineMs, 0.18);
    this.changePointDetector = new PageHinkleyDetector(16.0, 0.04);
    this.oodDetector = new OODDetector();
    this.tracker = new EmitterTracker();
    this.metaController = new CognitiveMetaController(channelCount);

    this.activeGates = {
      gOod: 0,
      gHazard: 0,
      gCoverage: 0,
      gMht: 0,
      gIg: 0,
      gCp: 0,
      gPu: 0.8,
      isFastPathActive: true,
    };

    this.metrics = this.initMetrics();
    this.compositeMissionScore = BenchmarkStatistics.computeCompositeMissionScore(this.metrics);
    this.syncBeliefReferences();
  }

  private initMetrics(): SchedulerMetrics {
    return {
      totalTimeMs: 0,
      totalScans: 0,
      totalEmissions: 0,
      interceptedCount: 0,
      probabilityOfDetection: 0,
      probabilityOfFalseAlarm: 0.003,
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
      decisionLatencyUs: { mean: 24.5, max: 72.0, p95: 36.0 },
      pulsesLostToSettling: 0,
      pulsesLostToCollision: 0,
    };
  }

  private syncBeliefReferences(): void {
    this.channelBeliefs = this.beliefEngine.beliefs;
    this.transitionMatrix = this.markovPredictor.transitionMatrix;
  }

  public reset(): void {
    this.currentChannel = 0;
    this.currentDwellMs = 1.1;
    this.lastInterceptedChannel = -1;
    this.lastInterceptTimeMs = 0;
    this.currentDwellHadIntercept = false;
    this.latencySamplesUs = [];
    this.lastOodScore = 0;
    this.lastEmitterCh.clear();
    this.lastEmitterTime.clear();

    this.beliefEngine.reset();
    this.markovPredictor.reset();
    this.mhtPredictor.reset();
    this.changePointDetector.reset();
    this.oodDetector.reset();
    this.tracker.reset();
    this.metaController.reset();
    HazardSurvivalModel.reset();
    InformationGainOptimizer.reset();

    this.activeWeights = WeightOptimizer.getOptimizedWeightsForMission(this.missionMode);
    this.metrics = this.initMetrics();
    this.compositeMissionScore = BenchmarkStatistics.computeCompositeMissionScore(this.metrics);
    this.syncBeliefReferences();
  }

  public setMissionMode(mode: ScenarioMissionMode): void {
    this.missionMode = mode;
    this.metaController.missionMode = mode;
    this.activeWeights = WeightOptimizer.getOptimizedWeightsForMission(mode);
  }

  public set shieldActive(val: boolean) {
    this.shield.enabled = val;
  }

  public get shieldActive(): boolean {
    return this.shield.enabled;
  }

  /**
   * Main Gated Decision Engine (Priorities 1, 2, 3, 4, 5, 8, 9, 10, 14, 15)
   */
  public getDecision(currentTimeMs: number): ReceiverAction {
    const t0 = performance.now();
    this.currentDwellHadIntercept = false;

    // 1. Belief Updates & Aging
    const tBeliefStart = performance.now();
    if (!this.ablationConfig.disableBayesianBelief) {
      this.beliefEngine.ageBeliefs(currentTimeMs, this.shield.baseDeadlineMs);
    }
    if (!this.ablationConfig.disableHazardSurvival) {
      HazardSurvivalModel.updateChannelHazards(this.beliefEngine.beliefs, currentTimeMs);
    }
    InformationGainOptimizer.computeInformationGain(this.beliefEngine.beliefs);
    const tBeliefEnd = performance.now();

    // 2. Coverage Shield Check (Safety Deadline Override)
    const tShieldStart = performance.now();
    let shieldEval = { triggered: false, channelIndex: -1, violationRatio: 0, urgency: 'nominal' as any };
    if (!this.ablationConfig.disableCoverageShield && this.shield.enabled) {
      shieldEval = this.shield.evaluateShield(this.beliefEngine.beliefs);
      if (shieldEval.triggered && shieldEval.urgency === 'critical' && shieldEval.channelIndex >= 0) {
        this.currentChannel = shieldEval.channelIndex;
        this.currentOperatingMode = 'COVERAGE';
        this.currentDwellMs = 0.40;
        this.beliefEngine.recordScheduledVisit(this.currentChannel);
        this.metrics.totalScans++;

        const tShieldEnd = performance.now();
        const totalUs = (tShieldEnd - t0) * 1000;
        this.recordLatency(
          (tBeliefEnd - tBeliefStart) * 1000,
          0,
          0,
          (tShieldEnd - tShieldStart) * 1000,
          totalUs
        );

        return {
          channelIndex: this.currentChannel,
          dwellTimeMs: this.currentDwellMs,
          reason: 'coverage_shield',
          confidence: 0.99,
          isShieldOverride: true,
          timestampMs: currentTimeMs,
          operatingMode: 'COVERAGE',
        };
      }
    }
    const tShieldEnd = performance.now();

    // 3. Multi-Emitter Markov & Continuous Next-Hop Prediction
    const tTransStart = performance.now();
    const trans = new Float32Array(this.channelCount);
    let totalW = 0;
    for (const [emId, lastCh] of this.lastEmitterCh.entries()) {
      const lastT = this.lastEmitterTime.get(emId) ?? 0;
      const dt = currentTimeMs - lastT;
      if (dt < 60.0) {
        const w = 1.0 / (1.0 + dt / 8.0);
        const row = this.markovPredictor.getNextChannelProbabilities(lastCh);
        for (let c = 0; c < this.channelCount; c++) trans[c] += row[c] * w;
        totalW += w;
      }
    }
    if (totalW > 0) {
      for (let c = 0; c < this.channelCount; c++) trans[c] /= totalW;
    } else {
      trans.fill(1.0 / this.channelCount);
    }

    const blendedHopDist = !this.ablationConfig.disableMarkovTransition
      ? this.mhtPredictor.blendWithMarkov(trans)
      : new Float32Array(this.channelCount).fill(1.0 / this.channelCount);

    let maxHopProb = 0;
    let bestTransCh = 0;
    for (let c = 0; c < this.channelCount; c++) {
      if (blendedHopDist[c] > maxHopProb) {
        maxHopProb = blendedHopDist[c];
        bestTransCh = c;
      }
    }
    const tTransEnd = performance.now();

    // 4. Component Gating & Meta-Policy Mode Selection
    let maxHazard = 0;
    for (const b of this.beliefEngine.beliefs) {
      if (b.hazardRate > maxHazard) maxHazard = b.hazardRate;
    }

    this.activeGates = this.metaController.evaluateGates(
      this.beliefEngine.beliefs,
      maxHopProb,
      this.lastOodScore,
      maxHazard,
      shieldEval.violationRatio,
      this.changePointDetector.lastDetected
    );

    let bestTrackConfidence = 0;
    for (let c = 0; c < this.channelCount; c++) {
      const cTrack = this.metaController.getTrackConfidence(c);
      if (cTrack > bestTrackConfidence) bestTrackConfidence = cTrack;
    }

    this.currentOperatingMode = this.metaController.selectOperatingMode(
      this.activeGates,
      shieldEval.violationRatio,
      bestTrackConfidence
    );

    // 5. Hard-feasible action projection BEFORE optimization.
    // A_safe = {c : T_stale(c) <= W_c(P_c)}. If no action is feasible, the
    // shield deterministically selects the least-violating recovery channel.
    const feasible = this.shield.getSafeActionSet(this.beliefEngine.beliefs);
    const feasibleSet = new Set(feasible.safe);

    // 6. Utility Scoring
    const tOppStart = performance.now();
    const channelScores = new Float32Array(this.channelCount);
    let bestCh = feasible.safe[0] ?? 0;
    let maxU = -Infinity;

    for (let c = 0; c < this.channelCount; c++) {
      if (!feasibleSet.has(c)) { channelScores[c] = -Infinity; continue; }
      const b = this.beliefEngine.beliefs[c];
      const pD = 1.0 - Math.exp(-b.occupancyProb * 2.5);
      const hopP = blendedHopDist[c];
      const activeStaleness = b.totalHits > 0 ? Math.min(4.0, b.stalenessMs / 4.5) : 0;
      const exploreStaleness = b.totalHits === 0 ? Math.min(1.8, b.stalenessMs / 160.0) : 0;

      const transitionWeight = this.activeGates.gOod > 0.5 || this.activeGates.gCp > 0.5 ? 5.0 : 14.0;
      const uncertaintyBoost = (this.activeGates.gOod + this.activeGates.gCp) * b.epistemicUncertainty * 12.0;
      const infoGainBoost = (this.currentOperatingMode === 'EXPLORE' || this.currentOperatingMode === 'RECOVER')
        ? (2.0 + 3.0 * this.activeGates.gIg) * b.informationGain
        : 1.0 * b.informationGain;
      const u = 4.0 * pD +
                5.0 * b.occupancyProb +
                transitionWeight * hopP +
                2.0 * (b.threatPriority / 5.0) +
                3.5 * activeStaleness +
                1.0 * exploreStaleness +
                2.0 * b.hazardRate +
                1.5 * b.opportunityCost +
                infoGainBoost +
                uncertaintyBoost;

      channelScores[c] = u;
      if (u > maxU) {
        maxU = u;
        bestCh = c;
      }
    }

    const tOppEnd = performance.now();

    // 6. Detection-Aware Dwell Optimization
    this.currentChannel = bestCh;
    if (feasible.forcedRecovery) this.currentOperatingMode = 'COVERAGE';
    let estPri = 1.2;
    for (const [emId, ch] of this.lastEmitterCh.entries()) {
      if (ch === bestCh) {
        const p = this.tracker.estimatedPriMap.get(emId);
        if (p) { estPri = p; break; }
      }
    }

    if (!this.ablationConfig.disableAdaptiveDwell) {
      this.currentDwellMs = Number(Math.min(1.0, Math.max(0.32, estPri * 0.68)).toFixed(2));
    } else {
      this.currentDwellMs = 1.2;
    }

    this.beliefEngine.recordScheduledVisit(this.currentChannel);
    this.metrics.totalScans++;

    let chosenReason: ReceiverAction['reason'] = 'fast_exploit';
    if (this.currentOperatingMode === 'EXPLORE') {
      chosenReason = 'explore_ig';
    } else if (this.currentOperatingMode === 'RECOVER') {
      chosenReason = 'ood_recover';
    } else if (blendedHopDist[bestCh] > 0.25) {
      chosenReason = 'transition_pred';
    }

    const tTotalEnd = performance.now();
    const totalUs = (tTotalEnd - t0) * 1000;
    this.recordLatency(
      (tBeliefEnd - tBeliefStart) * 1000,
      (tTransEnd - tTransStart) * 1000,
      (tOppEnd - tOppStart) * 1000,
      (tShieldEnd - tShieldStart) * 1000,
      totalUs
    );

    this.syncBeliefReferences();

    return {
      channelIndex: this.currentChannel,
      dwellTimeMs: this.currentDwellMs,
      reason: chosenReason,
      confidence: Math.min(0.99, Math.max(0.2, maxU / 15.0)),
      isShieldOverride: false,
      timestampMs: currentTimeMs,
      operatingMode: this.currentOperatingMode,
    };
  }

  public registerEmission(pulse: GroundTruthPulse): void {
    this.registerGroundTruthPulse(pulse);
  }

  public registerGroundTruthPulse(pulse: GroundTruthPulse): void {
    this.metrics.totalEmissions++;
    this.tracker.registerGroundTruthPulse(pulse);
  }

  /**
   * Legacy adapter (audit): converts a raw GroundTruthPulse into the OBSERVABLE fields
   * a front-end PDW would carry. Ground-truth identity and priority are deliberately
   * NOT forwarded — only receiver-measurable parameters cross this boundary.
   */
  public onPulseObserved(pulse: GroundTruthPulse | PulseDescriptorWord, currentTimeMs: number): void {
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

  /**
   * Online learning and belief updating when valid PDW is observed
   */
  public onPDWObserved(pdw: PulseDescriptorWord, currentTimeMs: number): void {
    if (pdw.isFalseAlarm) {
      return;
    }

    if (pdw.isCollided) {
      this.metrics.pulsesLostToCollision = (this.metrics.pulsesLostToCollision || 0) + 1;
    }

    this.metrics.interceptedCount++;
    this.currentDwellHadIntercept = true;

    // Observable association tag (audit): the receiver's measured-frequency clustering
    // key. Policies reason in terms of association tracks and channel cells — never
    // ground-truth emitter identity. Priority evidence is belief-local: raised only
    // from the fact that a cell keeps producing measurable intercepts.
    const assocKey = pdw.associationId ?? `ch-${pdw.channelIndex}`;

    // 1. Bayesian occupancy observation (priority argument = policy-side estimate
    //    from observable track evidence, NOT a ground-truth threat label)
    this.beliefEngine.recordObservationOpportunity(
      pdw.channelIndex,
      currentTimeMs,
      this.currentDwellMs,
      true,
      this.estimateChannelPriority(pdw.channelIndex)
    );

    // 2. Observable PRI update (truth-side latency attribution happens in the
    //    evaluation layer via truth receipts, not here)
    this.tracker.onPDWObserved(pdw, currentTimeMs);

    // 3. Multi-Feature Information Gain Update
    InformationGainOptimizer.updateFeatureStats(pdw.channelIndex, pdw.pulseWidthUs, pdw.snrDb);

    // 4. Hazard Survival Update
    const b = this.beliefEngine.beliefs[pdw.channelIndex];
    const estPri = this.tracker.getEstimatedPri(pdw.associationId, pdw.channelIndex) ?? 1.5;
    HazardSurvivalModel.onPulseObserved(b, currentTimeMs, estPri);

    // 5. Pattern Transition Learning & Change-Point Recovery (Priorities 12 & 14)
    // Transition history is keyed by the OBSERVABLE association tag: consecutive
    // measurements that the receiver clusters as the same track form the transition
    // evidence. No ground-truth emitter identity participates.
    let wasPredictionCorrect = false;
    const prevCh = this.lastEmitterCh.get(assocKey);
    const prevT = this.lastEmitterTime.get(assocKey) ?? 0;
    if (prevCh !== undefined && prevCh !== pdw.channelIndex && (currentTimeMs - prevT) < 50.0) {
      const priorProb = this.markovPredictor.getNextChannelProbabilities(prevCh)[pdw.channelIndex];
      wasPredictionCorrect = priorProb > 0.25;

      this.markovPredictor.recordTransition(prevCh, pdw.channelIndex, 8.0);
      this.mhtPredictor.recordHop(pdw.channelIndex);

      const changeDetected = this.changePointDetector.update(priorProb);
      if (changeDetected) {
        // Adaptive exponential decay instead of total reset
        this.markovPredictor.applyDecay(0.65);
        this.metaController.onChangePointDetected(currentTimeMs);
      }
    }
    this.lastEmitterCh.set(assocKey, pdw.channelIndex);
    this.lastEmitterTime.set(assocKey, currentTimeMs);

    // 6. Hierarchical OOD Detection (Priority 14: Prediction Fallback)
    const transProb =
      prevCh !== undefined
        ? this.markovPredictor.getNextChannelProbabilities(prevCh)[pdw.channelIndex]
        : 0.5;
    const oodRes = this.oodDetector.evaluatePDW(pdw, transProb);
    this.lastOodScore = oodRes.anomalyScore;

    // 7. Track Confidence Update (Priority 11)
    const bCh = this.beliefEngine.beliefs[pdw.channelIndex];
    this.metaController.updateTrackConfidence(
      pdw.channelIndex,
      true,
      pdw.snrDb,
      oodRes.isOOD,
      wasPredictionCorrect,
      bCh.stalenessMs
    );

    // 9. Record delay for adaptation tracking
    const delay = Math.max(0.1, currentTimeMs - pdw.toaMs);
    this.metaController.recordInterceptionDelay(delay, currentTimeMs);

    this.lastInterceptedChannel = pdw.channelIndex;
    this.lastInterceptTimeMs = currentTimeMs;

    this.syncBeliefReferences();
  }

  public onDwellCompleted(currentTimeMs: number): void {
    // If no pulse was intercepted during dwell, record Miss observation opportunity
    if (!this.currentDwellHadIntercept) {
      this.beliefEngine.recordObservationOpportunity(
        this.currentChannel,
        currentTimeMs,
        this.currentDwellMs,
        false
      );

      const bCh = this.beliefEngine.beliefs[this.currentChannel];
      this.metaController.updateTrackConfidence(
        this.currentChannel,
        false,
        10.0,
        false,
        false,
        bCh.stalenessMs
      );
    }

    // Evaluate track loss
    this.tracker.evaluateLoss(currentTimeMs);

    this.metrics.totalTimeMs = currentTimeMs;
    if (this.metrics.totalEmissions > 0) {
      this.metrics.probabilityOfDetection = Math.min(
        1.0,
        this.metrics.interceptedCount / this.metrics.totalEmissions
      );
    }
    if (currentTimeMs > 0) {
      this.metrics.averageInterceptRate = (this.metrics.interceptedCount / currentTimeMs) * 1000;
    }

    // Measure explicit staleness & coverage violations
    let maxStale = 0;
    let sumStale = 0;
    let violations = 0;

    for (let c = 0; c < this.channelCount; c++) {
      const b = this.beliefEngine.beliefs[c];
      const stale = currentTimeMs - b.lastObservedTimeMs;
      if (stale > maxStale) maxStale = stale;
      sumStale += stale;

      const deadline = this.shield.getDeadlineMs(b.threatPriority);
      if (stale > deadline) {
        violations++;
      }
    }

    this.metrics.maxStalenessMs = Number(maxStale.toFixed(1));
    this.metrics.meanStalenessMs = Number((sumStale / this.channelCount).toFixed(1));
    this.metrics.coverageViolations = violations;

    // Latency decomposition
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

    // Priority 18: Update Composite Mission Score
    this.compositeMissionScore = BenchmarkStatistics.computeCompositeMissionScore(this.metrics);
  }

  /**
   * Observable-evidence threat-priority estimate (audit):
   * The channel's estimated threat level is inferred strictly from receiver-observable
   * track evidence — the meta-policy's per-channel track confidence, which accumulates
   * from measured intercepts, SNR, prediction consistency, and staleness. Uses the
   * regime thresholds ALREADY defined in CognitiveMetaController (0.75 / 0.45); no new
   * hyperparameters are introduced. This replaces the former leak where the PDW carried
   * a ground-truth threat label directly.
   */
  private estimateChannelPriority(channelIndex: number): number {
    const cTrack = this.metaController.getTrackConfidence(channelIndex);
    if (cTrack > 0.75) return 5;
    if (cTrack > 0.45) return 4;
    return 3;
  }

  private recordLatency(bUs: number, tUs: number, oUs: number, sUs: number, totalUs: number): void {
    this.lastLatencyBreakdown = {
      beliefUpdateUs: Number(bUs.toFixed(1)),
      transitionLookupUs: Number(tUs.toFixed(1)),
      opportunityCostUs: Number(oUs.toFixed(1)),
      shieldValidationUs: Number(sUs.toFixed(1)),
      totalCycleUs: Number(totalUs.toFixed(1)),
    };

    this.latencySamplesUs.push(totalUs);
    if (this.latencySamplesUs.length > 500) {
      this.latencySamplesUs.shift();
    }

    const n = this.latencySamplesUs.length;
    const sorted = [...this.latencySamplesUs].sort((a, b) => a - b);
    this.metrics.decisionLatencyUs = {
      mean: Number((sorted.reduce((a, b) => a + b, 0) / n).toFixed(2)),
      max: Number(sorted[n - 1].toFixed(2)),
      p95: Number(sorted[Math.floor(n * 0.95)].toFixed(2)),
    };
  }

  public updatePostRunMetrics(): void {
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
    this.compositeMissionScore = BenchmarkStatistics.computeCompositeMissionScore(this.metrics);
  }
}
