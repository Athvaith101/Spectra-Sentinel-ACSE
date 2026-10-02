/**
 * Headless episode runner — the demo's simulation engine.
 *
 * Runs ONE policy over ONE seeded scenario using the repository's own validated
 * stack (RFSimulator + PhysicalReceiverFrontEnd + scheduler), stepping the world in
 * 0.5 ms slices exactly like the research harness (src/benchmark/harness.ts).
 *
 * FAIRNESS (prompt §6 / §13): each policy gets its own RFSimulator and front end,
 * seeded identically, so both see the same RF world. Only the scheduling policy
 * differs. Reactive emitters evolve policy-specifically — the engine documents this
 * as the intended closed-loop measurement (see harness.ts "IMPORTANT FAIRNESS FIX").
 *
 * CAUSALITY (prompt §28): ground-truth pulses go to the scheduler ONLY through
 * registerGroundTruthPulse (evaluation bookkeeping — the schedulers read nothing
 * decision-relevant from it); detection/identity joins happen exclusively in the
 * evaluation layer via truth receipts, never inside the policy.
 */
import {
  BenchmarkScenario,
  GroundTruthPulse,
  PulseDescriptorWord,
  ReceiverTruthReceipt,
  SchedulerMetrics,
} from '../state/types';
import { RFSimulator } from '../simulation/rfSimulator';
import { PhysicalReceiverFrontEnd } from '../simulation/receiverModel';
import { attributeTruthReceiptsForEvaluation } from '../state/harness';
import {
  BaseScheduler,
} from '../policies/baseScheduler';
import { DemoSchedulerAdapter } from './schedulerAdapter';
import { ScenarioConfig, missionModeForConfig } from './scenarioBuilder';
import { ScenarioMissionMode } from '../state/types';
import { HazardSurvivalModel } from '../state/hazard';
import { InformationGainOptimizer } from '../state/informationGain';

// PROVENANCE: adapted from ../src/demo/episodeRunner.ts (repo root); only import paths changed.

/** Time-aggregated row for the live waterfall. */
export interface WaterfallRow {
  timeMs: number;
  /** true = this band had >=1 real emission in the slice (truth, visual layer only) */
  activity: boolean[];
  band: number;        // band the receiver dwelled on this row
  hit: boolean;        // did a valid PDW (non-FA) occur during the dwell?
  falseAlarm: boolean;
  settling: boolean;   // receiver was still tuning for part of the slice
}

export interface EpisodeEventLog {
  timeMs: number;
  band: number;
  hit: boolean;
  snrDb: number | null;
  freqMHz: number | null;
}

export interface EpisodeResult {
  policyId: string;
  metrics: SchedulerMetrics;
  waterfall: WaterfallRow[];
  events: EpisodeEventLog[];
  /** Wall-clock cost of the whole run (ms). */
  computeTimeMs: number;
}

export interface LiveFrame {
  timeMs: number;
  row: WaterfallRow;
  /** The policy's action for the NEXT decision (already computed for the following dwell). */
  metrics: SchedulerMetrics;
  traceIndex: number;
}

const STEP_MS = 0.5;

/** Largest single UI step; keeps stepToNextDecision bounded (spec §5). */
const MAX_STEP_MS = 200;

export class EpisodeRunner {
  private sim: RFSimulator;
  private frontEnd: PhysicalReceiverFrontEnd;
  private adapter: DemoSchedulerAdapter;
  private scenario: BenchmarkScenario;
  private cfg: ScenarioConfig;
  private nextDecisionAtMs = 0;
  private t = 0;
  private waterfall: WaterfallRow[] = [];
  private events: EpisodeEventLog[] = [];
  private dwellHadHit = false;
  private dwellHadFA = false;
  private dwellStartMs = 0;
  private activeRow: WaterfallRow | null = null;

  constructor(scenario: BenchmarkScenario, cfg: ScenarioConfig, adapter: DemoSchedulerAdapter, seed: number) {
    this.scenario = scenario;
    this.cfg = cfg;
    this.adapter = adapter;
    // EPISODE-SCOPED STATIC STATE (seed-determinism repair): the engine's
    // InformationGainOptimizer and HazardSurvivalModel keep per-channel accumulators in
    // static Maps that persist across scheduler construction. Resetting them here — via
    // the engine's own public reset() — guarantees every episode (and therefore every
    // same-seed replay) starts from identical module state. No algorithm is changed;
    // this restores the engine's intended per-episode semantics for fresh schedulers.
    HazardSurvivalModel.reset();
    InformationGainOptimizer.reset();
    // Identical seed for every policy (fairness, prompt §6).
    this.sim = new RFSimulator(scenario, seed);
    this.frontEnd = new PhysicalReceiverFrontEnd({}, seed + 101);
    this.frontEnd.tuneToChannel(adapter.scheduler.currentChannel, adapter.scheduler.currentDwellMs, 0);
    // Mission-mode conditioning exists on the cognitive/portfolio schedulers only.
    const modeAware = adapter.scheduler as BaseScheduler & { setMissionMode?: (m: ScenarioMissionMode) => void };
    if (typeof modeAware.setMissionMode === 'function') {
      modeAware.setMissionMode(missionModeForConfig(cfg));
    }
  }

  get timeMs(): number {
    return this.t;
  }

  get done(): boolean {
    return this.t >= this.scenario.durationMs;
  }

  get adapterRef(): DemoSchedulerAdapter {
    return this.adapter;
  }

  /** Full waterfall history (one row per dwell decision). */
  get waterfallHistory(): WaterfallRow[] {
    return this.waterfall;
  }

  /** Advances the world by `dtMs` of simulated time; returns rows produced. */
  step(dtMs: number): { rows: WaterfallRow[]; newEvent: EpisodeEventLog | null } {
    const rows: WaterfallRow[] = [];
    let newEvent: EpisodeEventLog | null = null;
    const endT = Math.min(this.t + dtMs, this.scenario.durationMs);

    while (this.t < endT) {
      const sliceEnd = Math.min(this.t + STEP_MS, endT);
      const sliceDt = sliceEnd - this.t;

      // Mid-episode adversarial pattern shift (engine behaviour).
      if (
        this.scenario.adversarialShiftTimeMs !== undefined &&
        this.t >= this.scenario.adversarialShiftTimeMs &&
        this.t < this.scenario.adversarialShiftTimeMs + STEP_MS
      ) {
        this.sim.triggerAdversarialShift();
      }

      // 1. RF world emits (ground truth; used by the engine for emission bookkeeping).
      const pulses: GroundTruthPulse[] = this.sim.step(sliceDt);
      for (const p of pulses) this.adapter.registerGroundTruthPulse(p);

      // 2. Scheduler decides when its previous dwell expires (observable state only).
      if (this.t >= this.nextDecisionAtMs) {
        // Finalize the previous dwell's row: it now covers the whole dwell window.
        if (this.activeRow) {
          this.finalizeRow();
          rows.push(this.activeRow);
          this.activeRow = null;
        }
        const action = this.adapter.getDecision(this.t);
        this.nextDecisionAtMs = this.t + action.dwellTimeMs;
        this.frontEnd.tuneToChannel(action.channelIndex, action.dwellTimeMs, this.t);
        this.activeRow = {
          timeMs: this.t,
          activity: new Array(this.scenario.channelCount).fill(false),
          band: action.channelIndex,
          hit: false,
          falseAlarm: false,
          settling: false,
        };
        this.dwellHadHit = false;
        this.dwellHadFA = false;
        this.dwellStartMs = this.t;
      }

      // 3. Receiver front end measures (LO settling, detection physics, collisions, PFA).
      const pdws: PulseDescriptorWord[] = this.frontEnd.processPulses(pulses, this.t, sliceEnd);
      for (const pdw of pdws) {
        this.adapter.onPDWObserved(pdw, this.t);
        if (!pdw.isFalseAlarm) {
          this.dwellHadHit = true;
          newEvent = {
            timeMs: this.t,
            band: pdw.channelIndex,
            hit: true,
            snrDb: pdw.snrDb,
            freqMHz: pdw.freqMHz,
          };
        } else {
          this.dwellHadFA = true;
        }
      }

      // 4. Evaluation-layer join (truth receipts) — metrics only, never decisions.
      const receipts: ReceiverTruthReceipt[] = this.frontEnd.drainTruthReceipts();
      const tracker = (this.adapter.scheduler as unknown as { tracker?: unknown }).tracker;
      if (tracker) {
        attributeTruthReceiptsForEvaluation(
          this.sim,
          tracker as Parameters<typeof attributeTruthReceiptsForEvaluation>[1],
          pdws,
          receipts,
          this.t,
          true,
        );
      }

      // 5. Bookkeeping + accumulate the activity row for the visualization layer.
      this.adapter.onDwellCompleted(this.t);
      if (this.activeRow) {
        for (const p of pulses) {
          if (p.channelIndex >= 0 && p.channelIndex < this.activeRow.activity.length) {
            this.activeRow.activity[p.channelIndex] = true;
          }
        }
        if (this.dwellHadHit) this.activeRow.hit = true;
        if (this.dwellHadFA) this.activeRow.falseAlarm = true;
        this.activeRow.settling = this.activeRow.settling || this.frontEnd.state.isTuning;
      }

      this.t = sliceEnd;
    }

    // Episode finished: flush the final open dwell row.
    if (this.t >= this.scenario.durationMs && this.activeRow) {
      this.finalizeRow();
      rows.push(this.activeRow);
      this.activeRow = null;
    }
    return { rows, newEvent };
  }

  private finalizeRow(): void {
    const row = this.activeRow;
    if (!row) return;
    row.falseAlarm = row.falseAlarm && !row.hit;
    // Retain for runToCompletion() / full-history consumers.
    this.waterfall.push(row);
    if (this.waterfall.length > 20000) this.waterfall.shift();
  }

  /** Runs the entire episode without animation (for headless comparison runs). */
  runToCompletion(): EpisodeResult {
    const t0 = performance.now();
    while (!this.done) {
      this.step(STEP_MS * 20);
    }
    return {
      policyId: this.adapter.id,
      metrics: { ...this.adapter.metrics },
      waterfall: this.waterfall,
      events: this.events,
      computeTimeMs: Number((performance.now() - t0).toFixed(1)),
    };
  }

  /**
   * Advances exactly to the next scheduler decision boundary (dwell rollover),
   * for step-by-step demonstration (spec §5: scan → observe → update → decide).
   */
  stepToNextDecision(): { rows: WaterfallRow[]; newEvent: EpisodeEventLog | null } {
    if (this.done) return { rows: [], newEvent: null };
    const decisionAt = this.nextDecisionAtMs;
    const target = Math.min(decisionAt + 0.001, this.scenario.durationMs);
    return this.step(Math.min(MAX_STEP_MS, target - this.t));
  }
}
