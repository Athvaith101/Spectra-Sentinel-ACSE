/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { BenchmarkScenario, ComparisonResults, PulseDescriptorWord, ReceiverTruthReceipt, SchedulerMetrics } from '../types';
import { RFSimulator, DEFAULT_SCENARIOS } from '../environment/rfSimulator';
import { PhysicalReceiverFrontEnd } from '../environment/receiverModel';
import { BaseScheduler } from '../decision/baseScheduler';
import { SequentialScheduler } from '../decision/sequential';
import { StalenessGreedyScheduler } from '../decision/staleness';
import { CognitiveSmartScheduler } from '../decision/cognitive';
import { AdaptivePortfolioScheduler } from '../decision/adaptivePortfolio';
import { RLDQNScheduler } from '../decision/rlDqnPolicy';
import { MPCScheduler } from '../decision/mpcScheduler';
import { BenchmarkStatistics } from './statistics';

export interface ScenarioBenchmarkOutput {
  scenarioId: string;
  scenarioName: string;
  category: string;
  comparison: ComparisonResults;
  executionTimeMs: number;
  cognitiveMetrics?: SchedulerMetrics;
  portfolioMetrics?: SchedulerMetrics;
  portfolioExpert?: string;
  portfolioContext?: Record<string, number>;
  stalenessMetrics?: SchedulerMetrics;
  rlDqnMetrics?: SchedulerMetrics;
  mpcMetrics?: SchedulerMetrics;
}

export interface FullBenchmarkSuiteResult {
  timestamp: string;
  scenariosTested: number;
  allPassed80PctGoal: boolean;
  allSubMillisecondLatency: boolean;
  averageInterceptTimeReductionPct: number;
  averageSpeedupFactor: number;
  averageDecisionLatencyUs: number;
  scenarioResults: ScenarioBenchmarkOutput[];
}

/**
 * INFORMATION BOUNDARY — EVALUATION LAYER.
 * Truth-side receipts are consumed only here, never by policy code.
 */
export function attributeTruthReceiptsForEvaluation(
  sim: RFSimulator,
  tracker: { attributeTruthReceipt: Function; registerGroundTruthPulse: Function },
  pdws: PulseDescriptorWord[],
  receipts: ReceiverTruthReceipt[],
  currentTimeMs: number,
  registerIntercept: boolean
): void {
  const pdwById = new Map(pdws.map((p) => [p.id, p]));
  for (const r of receipts) {
    const pdw = pdwById.get(r.pdwId);
    if (!pdw) continue;
    tracker.attributeTruthReceipt(r, pdw.id, currentTimeMs);
    if (registerIntercept && !r.isFalseAlarm && !r.isCollided && r.emitterId) {
      sim.registerIntercept(r.emitterId);
    }
  }
}

interface EpisodeRun {
  scheduler: BaseScheduler;
  metrics: SchedulerMetrics;
  latencySamples: number[];
  portfolioExpert?: string;
  portfolioContext?: Record<string, number>;
}

export class BenchmarkHarness {
  /**
   * IMPORTANT FAIRNESS FIX:
   * Each policy gets its own RFSimulator seeded identically. This prevents a
   * reactive emitter from learning from Policy A's intercepts and then changing
   * the environment seen by Policy B. The initial stochastic world is matched;
   * closed-loop reactive evolution is policy-specific and therefore measurable.
   */
  private static runPolicyEpisode(
    scenario: BenchmarkScenario,
    seed: number,
    durationMs: number,
    schedulerFactory: () => BaseScheduler
  ): EpisodeRun {
    const sim = new RFSimulator(scenario, seed);
    const scheduler = schedulerFactory();
    const frontEnd = new PhysicalReceiverFrontEnd({}, seed + 101);
    let nextDecisionAtMs = 0;
    const stepDtMs = 0.5;
    const totalSteps = Math.floor(durationMs / stepDtMs);

    frontEnd.tuneToChannel(scheduler.currentChannel, scheduler.currentDwellMs, 0);

    for (let step = 0; step < totalSteps; step++) {
      const currentT = step * stepDtMs;
      const stepEndT = currentT + stepDtMs;

      if (
        scenario.adversarialShiftTimeMs !== undefined &&
        currentT >= scenario.adversarialShiftTimeMs &&
        currentT < scenario.adversarialShiftTimeMs + stepDtMs
      ) {
        sim.triggerAdversarialShift();
      }

      const groundTruthPulses = sim.step(stepDtMs);
      for (const p of groundTruthPulses) scheduler.registerGroundTruthPulse(p);

      if (currentT >= nextDecisionAtMs) {
        const action = scheduler.getDecision(currentT);
        nextDecisionAtMs = currentT + action.dwellTimeMs;
        frontEnd.tuneToChannel(action.channelIndex, action.dwellTimeMs, currentT);
      }

      const pdws = frontEnd.processPulses(groundTruthPulses, currentT, stepEndT);
      for (const pdw of pdws) scheduler.onPDWObserved(pdw, currentT);

      const receipts = frontEnd.drainTruthReceipts();
      const tracker = (scheduler as any).tracker;
      if (tracker) {
        // All policies participate in the same closed-loop reactive environment.
        attributeTruthReceiptsForEvaluation(sim, tracker, pdws, receipts, currentT, true);
      }
      scheduler.onDwellCompleted(currentT);
    }

    (scheduler as any).updatePostRunMetrics?.();
    const tracker = (scheduler as any).tracker;
    const latencySamples = tracker?.getDecompositionMetrics?.().allInterceptDelays ?? [];
    let portfolioExpert: string | undefined;
    let portfolioContext: Record<string, number> | undefined;
    if (scheduler instanceof AdaptivePortfolioScheduler) {
      const t = scheduler.getPortfolioTelemetry();
      portfolioExpert = t.selectedExpert;
      portfolioContext = {
        predictability: t.context.predictability,
        burstiness: t.context.burstiness,
        meanSnrDb: t.context.meanSnrDb,
        detectionRatePerSec: t.context.detectionRatePerSec,
        spectralSpread: t.context.spectralSpread,
        oodProxy: t.context.oodProxy,
        coveragePressure: t.context.coveragePressure,
      };
    }
    return { scheduler, metrics: { ...scheduler.metrics }, latencySamples, portfolioExpert, portfolioContext };
  }

  public static runScenario(
    scenario: BenchmarkScenario,
    durationMs: number = 2500,
    episodesPerScenario: number = 5
  ): ScenarioBenchmarkOutput {
    const tStart = performance.now();
    const seqLatencies: number[] = [];
    const portfolioLatencies: number[] = [];
    let lastSeq!: SchedulerMetrics;
    let lastPortfolio!: SchedulerMetrics;
    let lastCognitive!: SchedulerMetrics;
    let lastStaleness!: SchedulerMetrics;
    let lastRl!: SchedulerMetrics;
    let lastMpc!: SchedulerMetrics;
    let selectedExpert = 'cognitive';
    let selectedContext: Record<string, number> | undefined;

    for (let ep = 0; ep < episodesPerScenario; ep++) {
      const seed = 42 + ep * 997;
      const seq = this.runPolicyEpisode(scenario, seed, durationMs, () => new SequentialScheduler(scenario.channelCount, 2.0));
      const portfolio = this.runPolicyEpisode(scenario, seed, durationMs, () => new AdaptivePortfolioScheduler(scenario.channelCount, 150, seed + 17));
      const cognitive = this.runPolicyEpisode(scenario, seed, durationMs, () => new CognitiveSmartScheduler(scenario.channelCount));
      const staleness = this.runPolicyEpisode(scenario, seed, durationMs, () => new StalenessGreedyScheduler(scenario.channelCount));
      const rl = this.runPolicyEpisode(scenario, seed, durationMs, () => new RLDQNScheduler(scenario.channelCount, seed + 33));
      const mpc = this.runPolicyEpisode(scenario, seed, durationMs, () => new MPCScheduler(scenario.channelCount));

      seqLatencies.push(...(seq.latencySamples.length ? seq.latencySamples : [durationMs]));
      portfolioLatencies.push(...(portfolio.latencySamples.length ? portfolio.latencySamples : [durationMs]));
      lastSeq = seq.metrics;
      lastPortfolio = portfolio.metrics;
      lastCognitive = cognitive.metrics;
      lastStaleness = staleness.metrics;
      lastRl = rl.metrics;
      lastMpc = mpc.metrics;
      selectedExpert = portfolio.portfolioExpert ?? selectedExpert;
      selectedContext = portfolio.portfolioContext ?? selectedContext;
    }

    const statValidation = BenchmarkStatistics.computeFullValidation(
      seqLatencies,
      portfolioLatencies,
      episodesPerScenario,
      episodesPerScenario,
      1000 + scenario.id.length * 31
    );

    const baseMit = lastSeq.meanInterceptTimeMs > 0 ? lastSeq.meanInterceptTimeMs : durationMs;
    const portfolioMit = lastPortfolio.meanInterceptTimeMs > 0 ? lastPortfolio.meanInterceptTimeMs : durationMs;
    const reductionPct = Number(Math.max(-999, ((baseMit - portfolioMit) / Math.max(0.1, baseMit)) * 100).toFixed(1));
    const speedup = portfolioMit > 0 ? baseMit / portfolioMit : 1;
    const detImprovement = lastPortfolio.probabilityOfDetection - lastSeq.probabilityOfDetection;

    const comparison: ComparisonResults = {
      sequential: lastSeq,
      smart: lastPortfolio,
      interceptTimeReductionPct: reductionPct,
      detectionImprovementPct: Number((detImprovement * 100).toFixed(1)),
      speedupFactor: Number(speedup.toFixed(2)),
      subMillisecondLatencyPassed: lastPortfolio.decisionLatencyUs.p95 < 1000,
      target80PctImprovementPassed: reductionPct >= 80,
      statisticalValidation: statValidation,
      statisticalConfidence: {
        pValue: statValidation.permutationTest.pValue,
        isSignificant: statValidation.permutationTest.isSignificant,
        sampleCount: Math.min(seqLatencies.length, portfolioLatencies.length),
      },
    };

    return {
      scenarioId: scenario.id,
      scenarioName: scenario.name,
      category: scenario.category,
      comparison,
      executionTimeMs: Number((performance.now() - tStart).toFixed(1)),
      cognitiveMetrics: lastCognitive,
      portfolioMetrics: lastPortfolio,
      portfolioExpert: selectedExpert,
      portfolioContext: selectedContext,
      stalenessMetrics: lastStaleness,
      rlDqnMetrics: lastRl,
      mpcMetrics: lastMpc,
    };
  }

  public static runFullSuite(onProgress?: (completed: number, total: number) => void): FullBenchmarkSuiteResult {
    const results: ScenarioBenchmarkOutput[] = [];
    for (let i = 0; i < DEFAULT_SCENARIOS.length; i++) {
      const out = this.runScenario(DEFAULT_SCENARIOS[i], 2200, 5);
      results.push(out);
      onProgress?.(i + 1, DEFAULT_SCENARIOS.length);
    }

    const count = Math.max(1, results.length);
    const sumReduction = results.reduce((s, r) => s + r.comparison.interceptTimeReductionPct, 0);
    const sumSpeedup = results.reduce((s, r) => s + r.comparison.speedupFactor, 0);
    const sumLatency = results.reduce((s, r) => s + r.comparison.smart.decisionLatencyUs.mean, 0);
    return {
      timestamp: new Date().toISOString(),
      scenariosTested: results.length,
      allPassed80PctGoal: results.every(r => r.comparison.target80PctImprovementPassed),
      allSubMillisecondLatency: results.every(r => r.comparison.subMillisecondLatencyPassed),
      averageInterceptTimeReductionPct: Number((sumReduction / count).toFixed(1)),
      averageSpeedupFactor: Number((sumSpeedup / count).toFixed(2)),
      averageDecisionLatencyUs: Number((sumLatency / count).toFixed(1)),
      scenarioResults: results,
    };
  }
}
