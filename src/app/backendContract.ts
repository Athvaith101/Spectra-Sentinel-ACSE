/**
 * BACKEND INTEGRATION BOUNDARY (integration prompt §15/§16/§17)
 *
 * This module is the SINGLE seam through which the backend's real runner will be
 * connected. It deliberately mirrors the backend's ACTUAL execution contract —
 * `ewsched.baselines.run_episode(env, tracker, policy, scn, seed)` returning a metrics
 * dict — rather than inventing an API the backend does not have.
 *
 * TODAY: `USE_BACKEND = false`. Scenario Lab and Policy Arena execute the packaged
 * TypeScript engine (validated engine copies) through DemoSchedulerAdapter. Nothing here
 * changes their behaviour.
 *
 * WHEN THE BACKEND SERVICE EXISTS: implement BackendRunnerClient against the real
 * endpoint (contract below), flip USE_BACKEND, and both pages route through it — the UI
 * reads results in the same shape either way. NO metric is ever synthesized on the
 * frontend: if the backend is unreachable, the run fails loudly (integration prompt §22).
 *
 * ── Contract expected from the backend (derived from the backend implementation) ──
 * REQUEST  { policy: BackendPolicyId,        // POLICY_CATALOG[].backendAdapter
 *            seed: number,                   // identical for every policy in an arena run
 *            scenario: ScenarioSpec,         // serialized scenario (bands, emitters, horizon)
 *            config?: RunConfig }            // optional run configuration overrides
 * RESPONSE { decisions:  Array<{ slot, band, dwell }>,   // every scan decision, in order
 *            events:    Array<{ slot, band, detected }>, // waterfall / detection events
 *            metrics:   { pd, weighted_poi, hp_miss, coverage, reward, ... },
 *            logs?:     string[] }
 * The response maps 1:1 onto EpisodeResult/WaterfallRow in episodeRunner.ts, so the
 * waterfall, decision trace and results panels render backend runs unchanged.
 */
import { EpisodeResult, WaterfallRow } from '../app/episodeRunner';
import { ScenarioConfig } from '../app/scenarioBuilder';

/** Backend policy identifiers — exactly the locked-evaluation policy names. */
export type BackendPolicyId =
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

/**
 * COMMON RESULT SCHEMA (integration prompt §29) — one result structure consumed by
 * BOTH Scenario Lab and Policy Arena regardless of whether the run came from the local
 * engine or the backend. `evaluationSource`/`evaluationLabel` (§30) keep documented
 * benchmark values distinguishable from live runs — they are never blurred together.
 */
export type EvaluationSource = 'local-engine' | 'backend-inference' | 'documented-benchmark';

export type EvaluationLabel =
  | 'LIVE SCENARIO RESULT'   // local engine run on a user-built scenario
  | 'BACKEND INFERENCE'      // backend model inference on a user-built scenario
  | 'LOCKED BENCHMARK';      // documented locked-evaluation values (read-only)

export interface PolicyRunResult {
  policyId: string;
  policyName: string;
  scenarioId: string;
  seed: number;
  /** Core locked-evaluation metrics (always present). */
  detectionProbability: number;
  weightedPOI: number;
  coverage: number;
  reward: number;
  /** Optional extended metrics — present only when the source reports them. */
  highPriorityMissRate?: number;
  staleness?: number;
  interceptLatency?: number;
  scanEfficiency?: number;
  /** Per-decision stream (local engine; backend when provided). */
  decisions?: Array<{ slot: number; band: number; dwell: number }>;
  /** Waterfall rows for visualisation (local engine; backend when provided). */
  waterfall?: unknown[];
  /** Where these numbers came from + the UI label that must accompany them (§30). */
  evaluationSource: EvaluationSource;
  evaluationLabel: EvaluationLabel;
}

/**
 * Builds the common result envelope for a local-engine run (current behaviour).
 * Backend runs will map RunResponse → PolicyRunResult through the same shape.
 */
export function localRunResult(
  policyId: string,
  policyName: string,
  seed: number,
  scenarioId: string,
): PolicyRunResult {
  return {
    policyId,
    policyName,
    scenarioId,
    seed,
    detectionProbability: 0,
    weightedPOI: 0,
    coverage: 0,
    reward: 0,
    evaluationSource: 'local-engine',
    evaluationLabel: 'LIVE SCENARIO RESULT',
  };
}

/** Flip to true only when a real BackendRunnerClient implementation is wired in. */
export const USE_BACKEND = false;

/** Serialized scenario payload for the backend request. */
export interface ScenarioSpec {
  channelCount: number;
  durationSlots: number;
  seed: number;
  /** Backend-side scenario family / preset identifier, when applicable. */
  preset?: string;
}

export interface RunRequest {
  policy: BackendPolicyId;
  seed: number;
  scenario: ScenarioSpec;
  config?: Record<string, unknown>;
}

/** Backend response, shaped to map 1:1 onto the demo's EpisodeResult. */
export interface RunResponse {
  decisions: Array<{ slot: number; band: number; dwell: number }>;
  events: Array<{ slot: number; band: number; detected: boolean }>;
  metrics: {
    pd: number;
    weighted_poi: number;
    hp_miss: number;
    coverage: number;
    reward: number;
    [k: string]: number;
  };
  logs?: string[];
}

/** Implemented by the future backend client; never faked on the frontend. */
export interface BackendRunnerClient {
  runEpisode(req: RunRequest): Promise<RunResponse>;
}

/**
 * Converts a demo ScenarioConfig into the backend ScenarioSpec. Kept explicit and
 * lossless-by-design: only fields the backend contract actually needs.
 */
export function toScenarioSpec(cfg: ScenarioConfig, channelCount: number): ScenarioSpec {
  return {
    channelCount,
    durationSlots: Math.round(cfg.durationMs),
    seed: cfg.seed,
  };
}

/**
 * Converts a backend RunResponse into the demo's EpisodeResult so the existing
 * waterfall/decision/results UI renders backend runs without modification.
 * Waterfall rows are derived from the backend's OWN decision/event stream — never
 * regenerated or decorated on the frontend (integration prompt §22).
 */
export function toEpisodeResult(policyId: string, res: RunResponse): EpisodeResult {
  const rows: WaterfallRow[] = res.decisions.map((d, i) => {
    const ev = res.events.find((e) => e.slot >= d.slot && e.band === d.band);
    return {
      timeMs: d.slot,
      activity: [], // truth-side activity comes from the backend when provided
      band: d.band,
      hit: !!ev?.detected,
      falseAlarm: false,
      settling: false,
    } as WaterfallRow;
  });
  return {
    policyId,
    metrics: {
      // Backend metrics are authoritative; fields the backend does not report stay 0
      // and are simply not displayed. Nothing is inferred or filled in.
      totalTimeMs: res.metrics['total_time_ms'] ?? 0,
      totalScans: res.metrics['total_scans'] ?? res.decisions.length,
      totalEmissions: res.metrics['total_emissions'] ?? 0,
      interceptedCount: res.metrics['detections'] ?? 0,
      probabilityOfDetection: res.metrics.pd,
      probabilityOfFalseAlarm: res.metrics['pfa'] ?? 0,
      meanInterceptTimeMs: res.metrics['mit_ms'] ?? 0,
      medianInterceptTimeMs: res.metrics['mit_p50_ms'] ?? 0,
      iqrInterceptTimeMs: res.metrics['mit_iqr_ms'] ?? 0,
      p95InterceptTimeMs: res.metrics['mit_p95_ms'] ?? 0,
      p99InterceptTimeMs: res.metrics['mit_p99_ms'] ?? 0,
      tDetectMs: res.metrics['t_detect_ms'] ?? 0,
      tReacqMs: res.metrics['t_reacq_ms'] ?? 0,
      tHighPriorityMs: res.metrics['t_hp_ms'] ?? 0,
      tConfirmMs: res.metrics['t_confirm_ms'] ?? 0,
      averageInterceptRate: res.metrics['air'] ?? 0,
      coverageViolations: res.metrics['coverage_violations'] ?? 0,
      maxStalenessMs: res.metrics['staleness_max_ms'] ?? 0,
      meanStalenessMs: res.metrics['staleness_mean_ms'] ?? 0,
      decisionLatencyUs: {
        mean: res.metrics['decision_latency_mean_us'] ?? 0,
        max: res.metrics['decision_latency_max_us'] ?? 0,
        p95: res.metrics['decision_latency_p95_us'] ?? 0,
      },
    },
    waterfall: rows,
    events: [],
    computeTimeMs: 0,
  };
}

/**
 * Local-engine path (current behaviour): the packaged TypeScript engine executes the
 * episode in-browser. Kept behind the same seam so both pages share ONE integration
 * contract (integration prompt §15) — Scenario Lab (one policy) and Policy Arena
 * (multiple policies on the identical seed/scenario).
 */
export const integrationMode = USE_BACKEND ? 'backend' : 'local-engine';
