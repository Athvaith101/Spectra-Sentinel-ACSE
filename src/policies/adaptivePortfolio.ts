/**
 * Adaptive Portfolio Scheduler
 *
 * Scenario-adaptive expert selection without truth/oracle features.
 * A short receiver-observable scouting phase estimates:
 *   density, transition predictability, burstiness, SNR and spectral spread.
 * An offline-prior + online evidence router then selects one expert for the
 * remainder of the episode. Hard safety constraints remain enforced by the
 * selected expert.
 */
import { BaseScheduler } from './baseScheduler';
import { CognitiveSmartScheduler } from './cognitive';
import { MPCScheduler } from './mpcScheduler';
import { RLDQNScheduler } from './rlDqnPolicy';
import { StalenessGreedyScheduler } from './staleness';
import { SequentialScheduler } from './sequential';
import {
  GroundTruthPulse, PulseDescriptorWord, ReceiverAction, SchedulerMetrics,
  ChannelBelief, LatencyBreakdownUs, ComponentGates, CompositeMissionScore,
  AdaptationMetrics, CognitiveOperatingMode, ScenarioMissionMode, BayesianUpdateMode
} from '../state/types';

// PROVENANCE: copied verbatim from ../src/decision/adaptivePortfolio.ts (repo root).

export type PortfolioExpertId = 'cognitive' | 'mpc' | 'dqn' | 'staleness' | 'sequential';

export interface ObservableScenarioContext {
  durationMs: number;
  observationCount: number;
  detectionRatePerSec: number;
  meanSnrDb: number;
  snrStdDb: number;
  uniqueChannelRatio: number;
  transitionEntropy: number;
  predictability: number;
  interArrivalCv: number;
  burstiness: number;
  spectralSpread: number;
  oodProxy: number;
  coveragePressure: number;
}

export interface PortfolioDecisionTelemetry {
  selectedExpert: PortfolioExpertId;
  confidence: number;
  scoutingComplete: boolean;
  context: ObservableScenarioContext;
  scores: Record<PortfolioExpertId, number>;
  selectionReason: string;
}

interface ExpertArm {
  id: PortfolioExpertId;
  scheduler: BaseScheduler;
  prior: number;
  evidence: number;
  uses: number;
}

function clamp(x: number, lo = 0, hi = 1): number {
  return Math.max(lo, Math.min(hi, x));
}

function normalizeEntropy(prob: number[]): number {
  if (prob.length <= 1) return 0;
  if (!prob.length) return 1;
  const h = -prob.reduce((s, p) => s + (p > 1e-12 ? p * Math.log(p) : 0), 0);
  return clamp(h / Math.log(prob.length));
}

export class AdaptivePortfolioScheduler implements BaseScheduler {
  public name = 'Adaptive Scenario Portfolio';
  public type: 'smart' = 'smart';
  public currentChannel = 0;
  public currentDwellMs = 2.0;
  public metrics: SchedulerMetrics;

  // UI telemetry is sourced from the always-running observable cognitive core.
  public readonly telemetryCore: CognitiveSmartScheduler;
  public get channelBeliefs(): ChannelBelief[] { return this.telemetryCore.channelBeliefs; }
  public get transitionMatrix(): Float32Array[] { return this.telemetryCore.transitionMatrix; }
  public get lastLatencyBreakdown(): LatencyBreakdownUs { return this.telemetryCore.lastLatencyBreakdown; }
  public get activeGates(): ComponentGates { return this.telemetryCore.activeGates; }
  public get compositeMissionScore(): CompositeMissionScore { return this.telemetryCore.compositeMissionScore; }
  public get metaController() { return this.telemetryCore.metaController; }
  public get currentOperatingMode(): CognitiveOperatingMode { return this.telemetryCore.currentOperatingMode; }
  public get missionMode(): ScenarioMissionMode { return this.telemetryCore.missionMode; }
  public get beliefEngine() { return this.telemetryCore.beliefEngine; }
  public get tracker() { return this.telemetryCore.tracker; }
  public set shieldActive(v: boolean) { this.telemetryCore.shieldActive = v; }
  public get shieldActive() { return this.telemetryCore.shieldActive; }

  private readonly channelCount: number;
  private readonly scoutingMs: number;
  private scoutingStartMs = 0;
  private scouting = true;
  private lastDecisionTimeMs = 0;
  private selectedExpertId: PortfolioExpertId = 'cognitive';
  private selectedExpert: BaseScheduler;
  private readonly arms: ExpertArm[];
  private readonly observations: PulseDescriptorWord[] = [];
  private readonly interArrivals: number[] = [];
  private lastObservationMs: number | null = null;
  private seenChannels = new Set<number>();
  private seenFreqs: number[] = [];
  private selectedExpertUses = 0;
  private selectionCount = 0;
  private lastProxyReward = 0;
  private context: ObservableScenarioContext = this.emptyContext();
  private telemetry: PortfolioDecisionTelemetry;

  constructor(channelCount = 32, scoutingMs = 80, seed = 42) {
    this.channelCount = channelCount;
    this.scoutingMs = scoutingMs;
    this.telemetryCore = new CognitiveSmartScheduler(channelCount);
    const cognitive = this.telemetryCore;
    const mpc = new MPCScheduler(channelCount);
    const dqn = new RLDQNScheduler(channelCount, seed + 17);
    const staleness = new StalenessGreedyScheduler(channelCount);
    const sequential = new SequentialScheduler(channelCount, 2.0);
    this.arms = [
      { id: 'cognitive', scheduler: cognitive, prior: 0.72, evidence: 0, uses: 0 },
      { id: 'mpc', scheduler: mpc, prior: 0.66, evidence: 0, uses: 0 },
      { id: 'dqn', scheduler: dqn, prior: 0.62, evidence: 0, uses: 0 },
      { id: 'staleness', scheduler: staleness, prior: 0.58, evidence: 0, uses: 0 },
      { id: 'sequential', scheduler: sequential, prior: 0.50, evidence: 0, uses: 0 },
    ];
    this.selectedExpert = cognitive;
    this.metrics = this.initMetrics();
    this.telemetry = {
      selectedExpert: 'cognitive', confidence: 0, scoutingComplete: false,
      context: this.context, scores: this.scoreExperts(this.context),
      selectionReason: 'scouting',
    };
  }

  private initMetrics(): SchedulerMetrics {
    return {
      totalTimeMs: 0, totalScans: 0, totalEmissions: 0, interceptedCount: 0,
      probabilityOfDetection: 0, probabilityOfFalseAlarm: 0,
      meanInterceptTimeMs: 0, medianInterceptTimeMs: 0, iqrInterceptTimeMs: 0,
      p95InterceptTimeMs: 0, p99InterceptTimeMs: 0, tDetectMs: 0, tReacqMs: 0,
      tHighPriorityMs: 0, tConfirmMs: 0, averageInterceptRate: 0,
      coverageViolations: 0, maxStalenessMs: 0, meanStalenessMs: 0,
      decisionLatencyUs: { mean: 0, max: 0, p95: 0 },
      pulsesLostToSettling: 0, pulsesLostToCollision: 0,
    };
  }

  private emptyContext(): ObservableScenarioContext {
    return {
      durationMs: 0, observationCount: 0, detectionRatePerSec: 0, meanSnrDb: 0,
      snrStdDb: 0, uniqueChannelRatio: 0, transitionEntropy: 1,
      predictability: 0, interArrivalCv: 0, burstiness: 0, spectralSpread: 0,
      oodProxy: 0, coveragePressure: 0,
    };
  }

  public reset(): void {
    this.currentChannel = 0; this.currentDwellMs = 2.0;
    this.scouting = true; this.scoutingStartMs = 0; this.lastDecisionTimeMs = 0;
    this.selectedExpertId = 'cognitive'; this.selectedExpert = this.telemetryCore;
    this.observations.length = 0; this.interArrivals.length = 0;
    this.lastObservationMs = null; this.seenChannels.clear(); this.seenFreqs.length = 0;
    this.selectedExpertUses = 0; this.selectionCount = 0; this.lastProxyReward = 0;
    this.context = this.emptyContext();
    for (const arm of this.arms) { arm.scheduler.reset(); arm.evidence = 0; arm.uses = 0; }
    this.telemetryCore.reset();
    this.metrics = this.initMetrics();
    this.telemetry = {
      selectedExpert: 'cognitive', confidence: 0, scoutingComplete: false,
      context: this.context, scores: this.scoreExperts(this.context), selectionReason: 'scouting',
    };
  }

  public setMissionMode(mode: ScenarioMissionMode): void {
    this.telemetryCore.setMissionMode(mode);
  }
  public setBayesianMode(mode: BayesianUpdateMode): void { this.telemetryCore.beliefEngine.setMode(mode); }

  public registerGroundTruthPulse(pulse: GroundTruthPulse): void {
    this.metrics.totalEmissions++;
    // Truth registration is evaluation bookkeeping only; every expert receives the
    // same count. No scheduler decision consumes truth identity or priority.
    for (const arm of this.arms) arm.scheduler.registerGroundTruthPulse(pulse);
  }

  public onPDWObserved(pdw: PulseDescriptorWord, currentTimeMs: number): void {
    // Every expert receives the same observable stream, enabling clean switching
    // without resetting its internal belief/prediction state.
    for (const arm of this.arms) arm.scheduler.onPDWObserved(pdw, currentTimeMs);
    if (!pdw.isFalseAlarm) {
      this.observations.push(pdw);
      this.seenChannels.add(pdw.channelIndex);
      this.seenFreqs.push(pdw.freqMHz);
      if (this.lastObservationMs !== null) {
        const dt = pdw.toaMs - this.lastObservationMs;
        if (dt > 0) this.interArrivals.push(dt);
      }
      this.lastObservationMs = pdw.toaMs;
      this.context = this.extractContext(currentTimeMs);
    }
  }

  public onDwellCompleted(currentTimeMs: number): void {
    for (const arm of this.arms) arm.scheduler.onDwellCompleted(currentTimeMs);
    this.updateMetrics(currentTimeMs);
    if (!this.scouting && this.selectedExpertUses > 0) {
      const reward = this.proxyReward();
      const arm = this.arms.find(a => a.id === this.selectedExpertId);
      if (arm) {
        // Exponentially weighted online evidence; bounded so a single lucky pulse
        // cannot permanently lock the portfolio.
        arm.evidence = 0.92 * arm.evidence + 0.08 * reward;
        arm.uses++;
        this.lastProxyReward = reward;
      }
    }
  }

  public getDecision(currentTimeMs: number): ReceiverAction {
    if (this.scoutingStartMs === 0) this.scoutingStartMs = currentTimeMs;

    const elapsedScoutMs = currentTimeMs - this.scoutingStartMs;
    const earlyEvidence = this.observations.length >= 3;
    const effectiveScoutDeadline = earlyEvidence ? Math.min(this.scoutingMs, 50) : this.scoutingMs;
    if (this.scouting && elapsedScoutMs < effectiveScoutDeadline) {
      this.currentChannel = (this.currentChannel + 1) % this.channelCount;
      this.currentDwellMs = 2.0;
      this.lastDecisionTimeMs = currentTimeMs;
      return {
        channelIndex: this.currentChannel, dwellTimeMs: this.currentDwellMs,
        reason: 'sequential_sweep', confidence: 0.55, isShieldOverride: false,
        timestampMs: currentTimeMs, operatingMode: 'EXPLORE',
      };
    }

    if (this.scouting) this.selectExpert(currentTimeMs);

    // Keep the chosen expert's starting point aligned with the receiver's current tune.
    this.selectedExpert.currentChannel = this.currentChannel;
    this.selectedExpert.currentDwellMs = this.currentDwellMs;
    let action = this.selectedExpert.getDecision(currentTimeMs);

    // Portfolio-level hard safety projection. This remains active even when the
    // selected expert is a baseline that has no native shield.
    if (this.telemetryCore.shieldActive) {
      const beliefs = this.telemetryCore.channelBeliefs;
      const feasible = this.telemetryCore.shield.getSafeActionSet(beliefs);
      const isSafe = feasible.safe.includes(action.channelIndex);
      if (!isSafe) {
        let safeChannel = feasible.safe[0] ?? feasible.worstChannel;
        let bestScore = -Infinity;
        for (const c of feasible.safe) {
          const b = beliefs[c];
          const score = 2.0 * b.occupancyProb + 1.5 * b.hazardRate +
            1.2 * b.opportunityCost + 0.8 * b.informationGain;
          if (score > bestScore) { bestScore = score; safeChannel = c; }
        }
        action = {
          ...action, channelIndex: safeChannel, dwellTimeMs: 0.65,
          reason: 'coverage_shield', confidence: 0.99, isShieldOverride: true,
          operatingMode: 'COVERAGE', timestampMs: currentTimeMs,
        };
      }
      if (feasible.forcedRecovery) {
        action = {
          ...action, channelIndex: feasible.worstChannel, dwellTimeMs: 0.65,
          reason: 'coverage_shield', confidence: 0.995, isShieldOverride: true,
          operatingMode: 'COVERAGE', timestampMs: currentTimeMs,
        };
      }
    }

    this.currentChannel = action.channelIndex;
    this.currentDwellMs = action.dwellTimeMs;
    this.selectedExpertUses++;
    this.lastDecisionTimeMs = currentTimeMs;
    this.updateMetrics(currentTimeMs);

    return {
      ...action,
      confidence: Math.min(action.confidence, 0.995),
      reason: action.isShieldOverride ? action.reason : 'gated_hybrid',
    };
  }

  private selectExpert(currentTimeMs: number): void {
    this.context = this.extractContext(currentTimeMs);
    const scores = this.scoreExperts(this.context);
    const ranked = (Object.keys(scores) as PortfolioExpertId[]).sort((a, b) => scores[b] - scores[a]);
    const chosen = ranked[0];
    const margin = scores[chosen] - scores[ranked[1]];
    this.selectedExpertId = chosen;
    this.selectedExpert = this.arms.find(a => a.id === chosen)!.scheduler;
    this.scouting = false;
    this.selectionCount++;
    this.telemetry = {
      selectedExpert: chosen,
      confidence: clamp(0.55 + margin * 0.7),
      scoutingComplete: true,
      context: this.context,
      scores,
      selectionReason: this.explainSelection(chosen, this.context),
    };
  }

  private scoreExperts(c: ObservableScenarioContext): Record<PortfolioExpertId, number> {
    // Validation-derived priors are intentionally weak. Observable context supplies
    // the scenario adaptation; online evidence can later be incorporated by retraining
    // the profile table from locked validation episodes.
    const randomish = 1 - c.predictability;
    const persistent = 1 - c.burstiness;
    const weak = c.observationCount >= 3 ? clamp((14 - c.meanSnrDb) / 10) : 0;
    const dense = clamp(c.detectionRatePerSec / 250);
    const burst = c.burstiness;
    const ood = c.oodProxy;
    const uninformative = clamp((3 - c.observationCount) / 3);
    const scores: Record<PortfolioExpertId, number> = {
      // Robust default: mixed regimes, high density, OOD and coverage pressure.
      cognitive: 0.78 + 0.32 * ood + 0.24 * c.coveragePressure + 0.18 * dense + 0.12 * burst,
      // Model-based lookahead is valuable only when a stable transition structure
      // is actually observable; it is penalized under strong distribution shift.
      mpc: 0.64 + 0.72 * c.predictability + 0.10 * persistent - 0.28 * ood,
      // Learned adaptive policy gets priority for sparse/bursty/weak-signal regimes.
      dqn: 0.66 + 0.48 * burst + 0.28 * weak + 0.20 * dense + 0.08 * c.spectralSpread,
      // Cheap coverage-oriented policy for persistent/dense regimes.
      staleness: 0.57 + 0.34 * dense + 0.20 * persistent + 0.24 * c.coveragePressure - 0.16 * c.predictability,
      // Sequential is a last-resort safe fallback when there is genuinely too little
      // observable structure to justify adaptation, not a generic winner for random data.
      sequential: 0.38 + 0.28 * uninformative + 0.08 * randomish,
    };
    for (const arm of this.arms) {
      // Online evidence is deliberately capped to avoid early lock-in.
      scores[arm.id] += clamp(arm.evidence, -0.25, 0.25) * Math.min(1, arm.uses / 20);
    }
    return scores;
  }

  private extractContext(nowMs: number): ObservableScenarioContext {
    const n = this.observations.length;
    const snrs = this.observations.map(p => p.snrDb);
    const mean = n ? snrs.reduce((a,b)=>a+b,0)/n : 0;
    const variance = n ? snrs.reduce((s,x)=>s+(x-mean)*(x-mean),0)/n : 0;
    const ia = this.interArrivals;
    const iaMean = ia.length ? ia.reduce((a,b)=>a+b,0)/ia.length : 0;
    const iaVar = ia.length ? ia.reduce((s,x)=>s+(x-iaMean)*(x-iaMean),0)/ia.length : 0;
    const cv = iaMean > 0 ? Math.sqrt(iaVar)/iaMean : 0;
    const transitionCounts = new Map<number, Map<number, number>>();
    const byAssoc = new Map<string, number>();
    for (const p of this.observations) {
      const k = p.associationId ?? `ch-${p.channelIndex}`;
      const prev = byAssoc.get(k);
      if (prev !== undefined) {
        let row = transitionCounts.get(prev);
        if (!row) { row = new Map(); transitionCounts.set(prev, row); }
        row.set(p.channelIndex, (row.get(p.channelIndex) ?? 0) + 1);
      }
      byAssoc.set(k, p.channelIndex);
    }
    const entropies: number[] = [];
    for (const row of transitionCounts.values()) {
      const total = [...row.values()].reduce((a,b)=>a+b,0);
      const probs = [...row.values()].map(v=>v/Math.max(1,total));
      entropies.push(normalizeEntropy(probs));
    }
    const transitionEntropyRaw = entropies.length ? entropies.reduce((a,b)=>a+b,0)/entropies.length : 1;
    const transitionEntropy = Number.isFinite(transitionEntropyRaw) ? transitionEntropyRaw : 1;
    const predictability = clamp(1 - transitionEntropy);
    const spectralMean = this.seenFreqs.length ? this.seenFreqs.reduce((a,b)=>a+b,0)/this.seenFreqs.length : 0;
    const spectralVar = this.seenFreqs.length ? this.seenFreqs.reduce((s,x)=>s+(x-spectralMean)*(x-spectralMean),0)/this.seenFreqs.length : 0;
    const spectralSpread = clamp(Math.sqrt(spectralVar) / 2500);
    const burstiness = clamp((cv - 0.15) / 1.5);
    const detectionRatePerSec = n / Math.max(0.001, nowMs / 1000);
    const oodProxyRaw = 0.5 * spectralSpread + 0.5 * burstiness * (1 - predictability);
    const oodProxy = Number.isFinite(oodProxyRaw) ? clamp(oodProxyRaw) : spectralSpread;
    const coveragePressure = clamp(this.telemetryCore.channelBeliefs.length
      ? this.telemetryCore.channelBeliefs.reduce((m,b) => Math.max(m, b.stalenessMs / Math.max(1,b.stalenessMs + 100)),0)
      : 0);
    return {
      durationMs: nowMs,
      observationCount: n,
      detectionRatePerSec,
      meanSnrDb: mean,
      snrStdDb: Math.sqrt(variance),
      uniqueChannelRatio: clamp(this.seenChannels.size / Math.max(1,this.channelCount)),
      transitionEntropy,
      predictability,
      interArrivalCv: cv,
      burstiness,
      spectralSpread,
      oodProxy,
      coveragePressure,
    };
  }

  private proxyReward(): number {
    const hits = this.selectedExpert.metrics.interceptedCount;
    const scans = Math.max(1, this.selectedExpert.metrics.totalScans);
    const pfaPenalty = this.selectedExpert.metrics.probabilityOfFalseAlarm;
    return clamp(0.55 * (hits / scans) + 0.25 * (1 - pfaPenalty) + 0.20 * (1 - this.context.coveragePressure));
  }

  private explainSelection(id: PortfolioExpertId, c: ObservableScenarioContext): string {
    if (id === 'mpc') return `Predictable transition structure ${(c.predictability*100).toFixed(0)}% with low OOD pressure → lookahead expert.`;
    if (id === 'dqn') return `Sparse/bursty or weak-signal regime detected → learned adaptive expert.`;
    if (id === 'staleness') return `High density/persistence or coverage pressure → low-compute coverage expert.`;
    if (id === 'sequential') return `Low transition predictability ${(c.predictability*100).toFixed(0)}% → robust open-loop fallback.`;
    return `Mixed/uncertain regime or distribution shift → full cognitive expert.`;
  }

  private updateMetrics(currentTimeMs: number): void {
    const m = this.selectedExpert.metrics;
    const d = this.telemetryCore.tracker.getDecompositionMetrics();
    // Decision/coverage counters come from the selected expert; authoritative
    // emission-to-observation latency comes from the portfolio's truth-side
    // evaluation tracker. This prevents policy-side TOA error from masquerading
    // as interception latency.
    this.metrics = {
      ...m,
      totalTimeMs: currentTimeMs,
      meanInterceptTimeMs: d.meanInterceptTimeMs,
      medianInterceptTimeMs: d.medianInterceptTimeMs,
      iqrInterceptTimeMs: d.iqrInterceptTimeMs,
      p95InterceptTimeMs: d.p95InterceptTimeMs,
      p99InterceptTimeMs: d.p99InterceptTimeMs,
      tDetectMs: d.meanDetectMs,
      tReacqMs: d.meanReacqMs,
      tHighPriorityMs: d.meanHighPriorityMs,
      tConfirmMs: d.meanConfirmMs,
    };
  }

  public getPortfolioTelemetry(): PortfolioDecisionTelemetry { return { ...this.telemetry, context: { ...this.context }, scores: { ...this.telemetry.scores } }; }
  public get selectedExpertName(): string { return this.selectedExpert.name; }
  public get selectedExpertIdValue(): PortfolioExpertId { return this.selectedExpertId; }
}
