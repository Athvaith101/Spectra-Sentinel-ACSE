/**
 * DemoSchedulerAdapter — the single integration point between the demo UI and any
 * scan-scheduling policy (spec §18/§19: Frontend → SchedulerAdapter → policy).
 *
 * Contract (mirrors the repository's BaseScheduler so the validated engine stays
 * the source of truth):
 *   INPUT  : receiver-observable state only (PDW history, beliefs, staleness,
 *            transition predictions, current band/dwell, available bands).
 *   OUTPUT : { band, dwellMs, confidence, reason, factors } — no ground truth,
 *            no future information.
 *
 * ─── MODEL INTEGRATION BOUNDARY ────────────────────────────────────────────────
 * Every policy in the demo is wrapped here. When the final learned scheduler is
 * validated, add a new PolicyId whose factory wraps the deployed model behind the
 * same BaseScheduler surface (getDecision / onPDWObserved / registerGroundTruthPulse /
 * onDwellCompleted / reset) and add it to POLICY_CATALOG. No UI redesign is needed —
 * pages read POLICY_CATALOG and construct adapters through forPolicy().
 *
 * ─── CANONICAL POLICY CATALOG (single source of truth) ─────────────────────────
 * POLICY_CATALOG is the ONE policy list for the entire website: Scenario Lab scan-policy
 * selector, Policy Arena selection + comparison, Benchmark page, homepage snapshot and
 * the backend adapter all read it. No page maintains its own list.
 *
 * Availability model (exactly one class per policy):
 *   'connected'  — the local packaged engine implements the policy and executes it in
 *                  the browser today. Selectable and runnable everywhere.
 *   'pending'    — the policy genuinely exists in the backend project (documented in
 *                  latest readme.md §16/§17/§19 and implemented in src/scan/
 *                  policy_adapter.py) but no browser/backend execution path is connected
 *                  yet. Shown everywhere with "Backend integration pending", NOT
 *                  executable, never produces results.
 *   Category C policies (not part of the current project's README policy set) are NOT
 *   listed here at all — removed, not hidden.
 *
 * Every entry carries `backendAdapter`: the policy identifier used by the backend's
 * locked evaluation (results/locked/locked_test_report.json), so the backend runner maps
 * PolicyId → backend policy 1:1. Sequential is the canonical baseline.
 */
import { CognitiveSmartScheduler } from '../policies/cognitive';
import { SequentialScheduler } from '../policies/sequential';
import { StalenessGreedyScheduler } from '../policies/staleness';
import { MPCScheduler } from '../policies/mpcScheduler';
import { BaseScheduler } from '../policies/baseScheduler';
import { PulseDescriptorWord, ReceiverAction, GroundTruthPulse } from '../state/types';

// PROVENANCE: adapted from ../src/demo/schedulerAdapter.ts (repo root); extended for
// the multi-page demo (Policy Arena) per spec §9/§19.

/**
 * Canonical policy identities — exactly the latest README's backend policy set
 * (readme.md §19 locked evaluation; same ids in src/scan/policy_adapter.py
 * available_policies()). No demo-invented ids remain.
 */
export type PolicyId =
  | 'sequential'
  | 'staleness'
  | 'information'
  | 'ucb1'
  | 'linucb'
  | 'thompson'
  | 'markov'
  | 'cognitive'
  | 'mpc'
  | 'stare_cognitive'
  | 'portfolio';

/** Where a policy's execution actually happens today. */
export type PolicyAvailability = 'connected' | 'pending';

/** Registry category: baseline reference / adaptive decision policies. */
export type PolicyCategory = 'baseline' | 'adaptive';

/** Conceptual runner interface: PolicyId → actual implementation (backend adapter). */
export type PolicyRunner = (id: PolicyId, channelCount: number) => DemoSchedulerAdapter;

export interface PolicyMeta {
  id: PolicyId;
  /** Full judge-facing name. */
  displayName: string;
  /** Compact chip/label name. */
  shortName: string;
  /** One-line, judge-facing description of the actual mechanism. */
  description: string;
  category: PolicyCategory;
  /** Where execution happens today: connected (runnable now) or pending. */
  availability: PolicyAvailability;
  /** Convenience flag: availability === 'connected'. */
  executable: boolean;
  /** The policy has documented locked-evaluation benchmark results (readme §19). */
  benchmarkAvailable: boolean;
  /** Backend policy identifier for the integration seam (never null here: every policy
   *  in this catalog is a real backend policy from the README set). */
  backendAdapter: PolicyId;
  /** Human note shown in the UI; required for pending policies. */
  availabilityNote?: string;
}

const PENDING_NOTE = 'Backend integration pending — implemented in the project backend; not yet executable in the browser demo.';

export const POLICY_CATALOG: Record<PolicyId, PolicyMeta> = {
  sequential: {
    id: 'sequential',
    displayName: 'Sequential — Baseline',
    shortName: 'Sequential',
    description: 'Reference baseline: sweeps all 32 bands in a fixed order, ignoring observations.',
    category: 'baseline',
    availability: 'connected',
    executable: true,
    benchmarkAvailable: true,
    backendAdapter: 'sequential',
  },
  staleness: {
    id: 'staleness',
    displayName: 'Staleness',
    shortName: 'Staleness',
    description: 'Revisits the least-recently-scanned band; hit history breaks ties.',
    category: 'adaptive',
    availability: 'connected',
    executable: true,
    benchmarkAvailable: true,
    backendAdapter: 'staleness',
  },
  information: {
    id: 'information',
    displayName: 'Information',
    shortName: 'Information',
    description: 'Chooses the band whose next observation maximises expected information gain.',
    category: 'adaptive',
    availability: 'pending',
    executable: false,
    benchmarkAvailable: true,
    backendAdapter: 'information',
    availabilityNote: PENDING_NOTE,
  },
  ucb1: {
    id: 'ucb1',
    displayName: 'UCB1',
    shortName: 'UCB1',
    description: 'Bandit: upper confidence bound on per-band reward, exploration by uncertainty.',
    category: 'adaptive',
    availability: 'pending',
    executable: false,
    benchmarkAvailable: true,
    backendAdapter: 'ucb1',
    availabilityNote: PENDING_NOTE,
  },
  linucb: {
    id: 'linucb',
    displayName: 'LinUCB',
    shortName: 'LinUCB',
    description: 'Contextual bandit: linear reward model over belief features with UCB exploration.',
    category: 'adaptive',
    availability: 'pending',
    executable: false,
    benchmarkAvailable: true,
    backendAdapter: 'linucb',
    availabilityNote: PENDING_NOTE,
  },
  thompson: {
    id: 'thompson',
    displayName: 'Thompson',
    shortName: 'Thompson',
    description: 'Bandit: Thompson sampling over per-band detection posterior.',
    category: 'adaptive',
    availability: 'pending',
    executable: false,
    benchmarkAvailable: true,
    backendAdapter: 'thompson',
    availabilityNote: PENDING_NOTE,
  },
  markov: {
    id: 'markov',
    displayName: 'Markov',
    shortName: 'Markov',
    description: 'Predicts the next band from emitter transition statistics (Markov model).',
    category: 'adaptive',
    availability: 'pending',
    executable: false,
    benchmarkAvailable: true,
    backendAdapter: 'markov',
    availabilityNote: PENDING_NOTE,
  },
  cognitive: {
    id: 'cognitive',
    displayName: 'Cognitive (belief-driven)',
    shortName: 'Cognitive',
    description: 'Belief + hazard + information gain + staleness utility; the validated adaptive scheduler.',
    category: 'adaptive',
    availability: 'connected',
    executable: true,
    benchmarkAvailable: true,
    backendAdapter: 'cognitive',
  },
  mpc: {
    id: 'mpc',
    displayName: 'MPC',
    shortName: 'MPC',
    description: 'Receding-horizon lookahead (H=2) over predicted emitter transitions.',
    category: 'adaptive',
    availability: 'connected',
    executable: true,
    benchmarkAvailable: true,
    backendAdapter: 'mpc',
  },
  stare_cognitive: {
    id: 'stare_cognitive',
    displayName: 'STARE-Cognitive',
    shortName: 'STARE-Cognitive',
    description: 'Cognitive scheduler plus STARE temporal-representation bonus (README: identical locked result to Cognitive).',
    category: 'adaptive',
    availability: 'pending',
    executable: false,
    benchmarkAvailable: true,
    backendAdapter: 'stare_cognitive',
    availabilityNote: PENDING_NOTE,
  },
  portfolio: {
    id: 'portfolio',
    displayName: 'Portfolio',
    shortName: 'Portfolio',
    description: 'Hedges across scheduling experts; measured at the sequential baseline level in the locked evaluation (Pd 0.0657).',
    category: 'adaptive',
    availability: 'pending',
    executable: false,
    benchmarkAvailable: true,
    backendAdapter: 'portfolio',
    availabilityNote: PENDING_NOTE,
  },
};

/**
 * CANONICAL REGISTRY VIEW — typed accessor over POLICY_CATALOG.
 * Scenario Lab, Policy Arena, Benchmark, Methodology and the backend adapter must derive
 * every policy list from this object; no page maintains its own list.
 */
export const PolicyRegistry = {
  entries: POLICY_CATALOG,
  all: () => Object.values(POLICY_CATALOG),
  get: (id: PolicyId): PolicyMeta => POLICY_CATALOG[id],
  /** Category A — backend-connected, selectable AND runnable in Lab/Arena. */
  connected: () => Object.values(POLICY_CATALOG).filter((p) => p.availability === 'connected'),
  connectedIds: (): PolicyId[] =>
    Object.values(POLICY_CATALOG).filter((p) => p.availability === 'connected').map((p) => p.id),
  /** Category B — real project policies whose backend integration is pending. */
  pending: () => Object.values(POLICY_CATALOG).filter((p) => p.availability === 'pending'),
  /** Policies selectable in the UI pickers (connected + pending, availability-labelled). */
  selectable: () => Object.values(POLICY_CATALOG),
  /** Policies with documented locked-evaluation evidence (Benchmark page). */
  benchmarkBacked: () => Object.values(POLICY_CATALOG).filter((p) => p.benchmarkAvailable),
  /** The canonical baseline (Sequential — Baseline). */
  baselineId: 'sequential' as PolicyId,
};

/** Policies that can actually execute in the browser today. */
export const EXECUTABLE_POLICIES: PolicyId[] = PolicyRegistry.connectedIds();

/**
 * Observable decision factors — exactly the terms the CognitiveSmartScheduler's
 * utility actually sums (see src/policies/cognitive.ts getDecision). These are
 * labelled "observable scheduler features": they explain the decision rule that
 * produced the choice, not a neural network's internal reasoning.
 */
export interface DecisionFactor {
  key: string;
  label: string;
  /** LOW / MEDIUM / HIGH rating derived from the actual numeric term. */
  level: 'LOW' | 'MEDIUM' | 'HIGH';
  /** The actual value used by the scheduler's utility. */
  value: string;
}

export interface DecisionTraceEntry {
  timeMs: number;
  band: number;
  bandLabel: string;
  dwellMs: number;
  reason: ReceiverAction['reason'];
  reasonLabel: string;
  confidence: number;
  /** Observable feature snapshot for the SELECTED band at decision time. */
  factors: DecisionFactor[];
  /** Short "What happened?" sentence generated only from available state. */
  explanation: string;
}

const REASON_LABELS: Record<ReceiverAction['reason'], string> = {
  transition_pred: 'Predicted hop target',
  hazard_opp_cost: 'Hazard / opportunity cost',
  coverage_shield: 'Coverage shield override',
  novelty_explore: 'Novelty exploration',
  sequential_sweep: 'Open-loop sequential sweep',
  information_gain: 'Information gain',
  staleness_greedy: 'Staleness-driven revisit',
  receding_mpc: 'MPC lookahead',
  fast_exploit: 'Fast exploit path',
  explore_ig: 'Exploration (uncertainty)',
  confirmation_burst: 'Confirmation burst',
  ood_recover: 'Distribution-shift recovery',
  gated_hybrid: 'Gated adaptive decision',
};

function level(v: number, hi: number, lo: number): 'LOW' | 'MEDIUM' | 'HIGH' {
  if (v >= hi) return 'HIGH';
  if (v <= lo) return 'LOW';
  return 'MEDIUM';
}

export function bandLabel(band: number): string {
  const f = 2000 + band * 250;
  return `${(f / 1000).toFixed(2)}–${((f + 250) / 1000).toFixed(2)} GHz`;
}

/**
 * PolicyRunner implementation: maps PolicyId → the actual local-engine implementation.
 * Only connected policies have an engine mapping; pending policies throw — they must
 * never silently produce fabricated results. When the backend service lands, extend this
 * seam (or backendContract.BackendRunnerClient) per policy; the UI needs no changes.
 */
export class DemoSchedulerAdapter {
  readonly id: PolicyId;
  readonly label: string;
  /** The wrapped, validated engine scheduler. */
  readonly scheduler: BaseScheduler;
  public trace: DecisionTraceEntry[] = [];
  private lastAction: ReceiverAction | null = null;
  private consecutiveSameBand = 0;
  private lastBand = -1;

  constructor(id: PolicyId, scheduler: BaseScheduler) {
    this.id = id;
    this.label = POLICY_CATALOG[id].displayName;
    this.scheduler = scheduler;
  }

  static forPolicy: PolicyRunner = (id: PolicyId, channelCount: number): DemoSchedulerAdapter => {
    const meta = POLICY_CATALOG[id];
    if (!meta.executable) {
      throw new Error(
        `Policy "${id}" is not executable (backend integration pending). ` +
        'No local implementation exists; refusing to fabricate a run.',
      );
    }
    switch (id) {
      case 'sequential':
        return new DemoSchedulerAdapter(id, new SequentialScheduler(channelCount, 2.0));
      case 'staleness':
        return new DemoSchedulerAdapter(id, new StalenessGreedyScheduler(channelCount));
      case 'cognitive':
        return new DemoSchedulerAdapter(id, new CognitiveSmartScheduler(channelCount));
      case 'mpc':
        return new DemoSchedulerAdapter(id, new MPCScheduler(channelCount));
      default:
        // Reachable only for connected==true ids lacking an engine case — a catalog/adapter
        // inconsistency. Fail loudly rather than substitute another policy.
        throw new Error(`No local engine implementation for connected policy "${id}".`);
    }
  };

  get metrics() {
    return this.scheduler.metrics;
  }

  get currentChannel(): number {
    return this.scheduler.currentChannel;
  }

  reset(): void {
    this.scheduler.reset();
    this.trace = [];
    this.lastAction = null;
    this.consecutiveSameBand = 0;
    this.lastBand = -1;
  }

  getDecision(currentTimeMs: number): ReceiverAction {
    const action = this.scheduler.getDecision(currentTimeMs);
    this.lastAction = action;

    if (action.channelIndex === this.lastBand) {
      this.consecutiveSameBand++;
    } else {
      this.consecutiveSameBand = 0;
    }
    this.lastBand = action.channelIndex;

    if (this.trace.length > 240) this.trace.shift();
    this.trace.push(this.buildTrace(action, currentTimeMs));
    return action;
  }

  onPDWObserved(pdw: PulseDescriptorWord, currentTimeMs: number): void {
    this.scheduler.onPDWObserved(pdw, currentTimeMs);
  }

  registerGroundTruthPulse(pulse: GroundTruthPulse): void {
    this.scheduler.registerGroundTruthPulse(pulse);
  }

  onDwellCompleted(currentTimeMs: number): void {
    this.scheduler.onDwellCompleted(currentTimeMs);
  }

  get action(): ReceiverAction | null {
    return this.lastAction;
  }

  /** Latest trace entry (built at decision time). */
  get lastTrace(): DecisionTraceEntry | null {
    return this.trace.length ? this.trace[this.trace.length - 1] : null;
  }

  private buildTrace(action: ReceiverAction, timeMs: number): DecisionTraceEntry {
    const band = action.channelIndex;
    const factors: DecisionFactor[] = [];
    let explanation: string;

    if (this.id === 'sequential') {
      factors.push(
        { key: 'policy', label: 'Policy', level: 'LOW', value: 'Open-loop raster' },
        { key: 'next', label: 'Next band in sweep', level: 'LOW', value: `${band + 1} of 32` },
      );
      explanation =
        'The sequential scheduler does not use observations: it always sweeps to the next band in a fixed raster, so the decision is independent of what was detected.';
    } else if (this.id === 'staleness') {
      factors.push(
        { key: 'rule', label: 'Rule', level: 'MEDIUM', value: 'Most-stale band first' },
        { key: 'next', label: 'Selected band', level: 'LOW', value: `${band + 1} of 32` },
      );
      explanation = `The staleness policy always revisits the band that has gone unobserved the longest (band ${band + 1}); earlier detections there add a small priority boost.`;
    } else {
      // Observable scheduler features (same quantities the cognitive utility sums).
      // 'cognitive' exposes its own beliefs. MPC keeps its belief engine private, so only
      // the decision reason is reported for it — nothing is invented.
      const beliefs = 'channelBeliefs' in this.scheduler
        ? (this.scheduler as CognitiveSmartScheduler).channelBeliefs
        : null;
      const b = beliefs ? beliefs[band] : null;

      if (b) {
        const occ = b.occupancyProb;
        const occLevel = level(occ, 0.25, 0.12);
        factors.push({
          key: 'recent_activity',
          label: 'Recent activity (belief)',
          level: occLevel,
          value: `${(occ * 100).toFixed(0)}%`,
        });
        factors.push({
          key: 'staleness',
          label: 'Scan staleness',
          level: level(b.stalenessMs, 250, 80),
          value: `${b.stalenessMs.toFixed(0)} ms since last observed`,
        });
        factors.push({
          key: 'history',
          label: 'Detection history',
          level: level(b.totalHits, 4, 1),
          value: `${b.totalHits} hits in ${b.totalVisits} visits`,
        });
        factors.push({
          key: 'uncertainty',
          label: 'Epistemic uncertainty',
          level: level(b.epistemicUncertainty, 0.04, 0.015),
          value: b.epistemicUncertainty.toFixed(4),
        });
        factors.push({
          key: 'hazard',
          label: 'Recurrence hazard',
          level: level(b.hazardRate, 0.08, 0.02),
          value: b.hazardRate.toFixed(3),
        });
        factors.push({
          key: 'opp_cost',
          label: 'Opportunity cost',
          level: level(b.opportunityCost, 0.25, 0.08),
          value: b.opportunityCost.toFixed(3),
        });
        factors.push({
          key: 'info_gain',
          label: 'Expected information gain',
          level: level(b.informationGain, 0.35, 0.15),
          value: b.informationGain.toFixed(3),
        });
        factors.push({
          key: 'retune',
          label: 'Retune cost',
          level: level(Math.abs(band - this.lastActionBeforeDecide), 8, 3),
          value: `${Math.abs(band - this.lastActionBeforeDecide)} band(s) away`,
        });

        const bits: string[] = [];
        if (action.reason === 'coverage_shield') {
          bits.push(
            `Band ${band} had not been observed for ${b.stalenessMs.toFixed(0)} ms, exceeding its coverage deadline, so the safety shield forced a revisit`
          );
        } else if (action.reason === 'transition_pred') {
          bits.push(`The transition predictor assigned high probability to a hop into band ${band}`);
        } else if (action.reason === 'explore_ig' || action.reason === 'information_gain') {
          bits.push(`Uncertainty in band ${band} was high (unexplored or recently changed), so the scheduler chose to gather information there`);
        } else if (action.reason === 'ood_recover') {
          bits.push('A distribution shift was detected, so the scheduler fell back to broad uncertainty-driven exploration');
        } else {
          bits.push(
            `Band ${band} combined ${occLevel === 'LOW' ? 'low' : occLevel === 'HIGH' ? 'high' : 'moderate'} estimated activity with staleness of ${b.stalenessMs.toFixed(0)} ms`
          );
        }
        if (b.stalenessMs > 250 && action.reason !== 'coverage_shield') {
          bits.push('and had not been scanned recently');
        }
        bits.push(`— the utility sum therefore prioritised it (dwell ${(action.dwellTimeMs ?? 1).toFixed(2)} ms)`);
        explanation = bits.join(' ') + '.';
      } else {
        factors.push(
          { key: 'lookahead', label: 'Mechanism', level: 'MEDIUM', value: 'H=2 receding horizon' },
          { key: 'next', label: 'Selected band', level: 'LOW', value: `${band + 1} of 32` },
        );
        explanation = `The MPC scheduler evaluated two-step lookahead utility over predicted emitter transitions and selected band ${band} (${REASON_LABELS[action.reason]}).`;
      }
    }

    return {
      timeMs,
      band,
      bandLabel: bandLabel(band),
      dwellMs: action.dwellTimeMs,
      reason: action.reason,
      reasonLabel: REASON_LABELS[action.reason],
      confidence: action.confidence,
      factors,
      explanation,
    };
  }

  /** Band the adapter was previously on, used for retune-cost display. */
  private get lastActionBeforeDecide(): number {
    return this.lastBand < 0 ? 0 : this.lastBand;
  }
}
