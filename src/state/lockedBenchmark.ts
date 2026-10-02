/**
 * AUTHORITATIVE BENCHMARK DATA — the single source of truth for every locked-evaluation
 * number displayed anywhere on the demo website (integration prompt §29: one clearly
 * identified evaluation source; no mixing of protocols).
 *
 * SOURCE OF RECORD (verbatim transcription — do not edit values without re-deriving
 * them from the source documents):
 *   - `latest readme.md` (SIH26055 full technical report), §1, §19, §20, §22, §38
 *     — the locked-evaluation sections of the supplied model-training/comparison report.
 *   - `results/locked/locked_test_report.json` — the machine-readable locked-test report
 *     of the same evaluation line (values for Pd / wPOI / reward / coverage match).
 *
 * PROVENANCE RULES (integration prompt §1, §29):
 *   - Browser-simulation results, demo smoke-test results and historical V3/V5 numbers
 *     are NOT benchmark data and must never be added to LOCKED_EVALUATION.
 *   - Secondary protocols (development tournament, RL research track) are kept in
 *     separate, clearly-labelled blocks below and are never merged into the locked table.
 *   - The locked report's mean-intercept-time (MIT) column is a timestamp artefact
 *     (values ≈ 1e-17 s; see results/locked/locked_test_v6_audit.json note: re-run after
 *     the MIT half-open-scan-window root fix). MIT is therefore NOT quoted as a locked
 *     headline metric anywhere in the UI.
 */

export interface LockedPolicyResult {
  /** Backend identifier, exactly as it appears in the locked report. */
  id:
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
  label: string;
  role: 'baseline' | 'adaptive' | 'reference';
  /** Detection probability. */
  pd: number;
  /** 95% bootstrap CI for Pd (500 resamples, per the report's discipline block). */
  pdLow: number;
  pdHigh: number;
  /** Weighted probability of intercept (priority-weighted coverage of activity). */
  wPoi: number;
  /** Fraction of high-priority emitter activity missed. */
  hpMiss: number;
  /** Coverage fraction under the evaluation protocol (1.000 = deadline respected). */
  coverage: number;
  /** Mean staleness (seconds) as reported. */
  meanStalenessS: number;
  /** Composite reward of the evaluation protocol. */
  reward: number;
}

/** Locked evaluation: 3 seeds × 24 scenario instances × 2 episodes/scenario × 11 policies. */
export const LOCKED_EVALUATION = {
  title: 'Locked evaluation',
  /** Report + configuration line (readme §19). */
  source:
    'SIH26055 locked evaluation report — readme.md §19/§20 (results/locked/locked_test_report.json)',
  scale: {
    /** TSRD-derived synthetic PDW corpus (readme §3): QA-passed files and pulses. */
    corpusFiles: 4689,
    corpusPulses: 2466102218,
    corpusQaFailures: 0,
    /** Locked-test configuration (readme §19/§38). */
    seeds: 3,
    episodes: 528,
    events: 1158,
    scenarioInstances: 24,
    policies: 11,
    bootstrapSamples: 500,
    confidence: 0.95,
  },
  policies: [
    { id: 'sequential', label: 'Sequential', role: 'baseline', pd: 0.0657, pdLow: 0.051, pdHigh: 0.082, wPoi: 0.0726, hpMiss: 0.9158, coverage: 1.0, meanStalenessS: 0.0035, reward: 3.551 },
    { id: 'staleness', label: 'Staleness', role: 'baseline', pd: 0.0657, pdLow: 0.049, pdHigh: 0.082, wPoi: 0.0726, hpMiss: 0.9158, coverage: 1.0, meanStalenessS: 0.0035, reward: 3.551 },
    { id: 'information', label: 'Information', role: 'baseline', pd: 0.0657, pdLow: 0.05, pdHigh: 0.082, wPoi: 0.0726, hpMiss: 0.9158, coverage: 1.0, meanStalenessS: 0.0035, reward: 3.551 },
    { id: 'ucb1', label: 'UCB1', role: 'baseline', pd: 0.1402, pdLow: 0.115, pdHigh: 0.162, wPoi: 0.1371, hpMiss: 0.87, coverage: 1.0, meanStalenessS: 0.0035, reward: 5.301 },
    { id: 'linucb', label: 'LinUCB', role: 'baseline', pd: 0.1465, pdLow: 0.13, pdHigh: 0.164, wPoi: 0.1431, hpMiss: 0.8608, coverage: 1.0, meanStalenessS: 0.0049, reward: 4.676 },
    { id: 'thompson', label: 'Thompson', role: 'baseline', pd: 0.2459, pdLow: 0.212, pdHigh: 0.281, wPoi: 0.2727, hpMiss: 0.6705, coverage: 1.0, meanStalenessS: 0.0093, reward: 6.265 },
    { id: 'markov', label: 'Markov', role: 'baseline', pd: 0.1819, pdLow: 0.14, pdHigh: 0.229, wPoi: 0.2038, hpMiss: 0.7525, coverage: 0.25, meanStalenessS: 0.0005, reward: 4.395 },
    { id: 'cognitive', label: 'Cognitive', role: 'adaptive', pd: 0.2403, pdLow: 0.196, pdHigh: 0.287, wPoi: 0.2076, hpMiss: 0.8545, coverage: 1.0, meanStalenessS: 0.069, reward: 6.426 },
    { id: 'mpc', label: 'MPC', role: 'adaptive', pd: 0.2138, pdLow: 0.167, pdHigh: 0.265, wPoi: 0.1988, hpMiss: 0.8145, coverage: 1.0, meanStalenessS: 0.076, reward: 6.551 },
    { id: 'stare_cognitive', label: 'STARE-Cognitive', role: 'adaptive', pd: 0.2403, pdLow: 0.195, pdHigh: 0.287, wPoi: 0.2076, hpMiss: 0.8545, coverage: 1.0, meanStalenessS: 0.069, reward: 6.426 },
    { id: 'portfolio', label: 'Portfolio', role: 'reference', pd: 0.0657, pdLow: 0.051, pdHigh: 0.082, wPoi: 0.0726, hpMiss: 0.9158, coverage: 1.0, meanStalenessS: 0.0035, reward: 3.551 },
  ] as LockedPolicyResult[],
} as const;

/** Paired Cognitive-vs-Sequential analysis from the locked report (readme §20). */
export const LOCKED_HEADLINE = {
  cognitivePd: 0.2403,
  sequentialPd: 0.0657,
  deltaPd: 0.1746,
  deltaPdCi: [0.125, 0.2258] as const,
  cognitiveReward: 6.426,
  sequentialReward: 3.551,
  deltaReward: 2.875,
  /** 0.2403 / 0.0657 ≈ 3.66 — a locked-configuration measurement, not a field guarantee. */
  pdRatio: 3.66,
} as const;

/** Paired MPC-vs-Cognitive analysis from the locked report (readme §22). */
export const LOCKED_MPC_VS_COGNITIVE = {
  deltaPd: -0.0265,
  pdCi: [-0.0831, 0.023] as const,
  deltaReward: 0.125,
} as const;

/**
 * Metric definitions — only terms/wordings supported by the project report.
 * (readme §2 conceptual objective, §15 coverage, §19 table columns.)
 */
export const METRIC_DEFINITIONS: Record<string, string> = {
  pd: 'Detection probability — share of emitter activity the receiver actually intercepted during the episode.',
  wPoi: 'Weighted POI — intercept coverage of activity weighted by mission priority.',
  hpMiss: 'High-priority miss rate — fraction of high-priority emitter activity not intercepted.',
  coverage: 'Coverage — fraction of the spectrum kept within the evaluation protocol\u2019s revisit deadline (1.000 = fully respected).',
  reward: 'Reward — the evaluation protocol\u2019s composite score (detection, information, priority, coverage and cost terms).',
  staleness: 'Mean staleness — average time a channel went unobserved.',
};

/** Scientific caveats, stated with the report's own wording (readme §1, §21, §26, App. B). */
export const LOCKED_CAVEATS: { title: string; text: string }[] = [
  {
    title: 'STARE attribution',
    text: 'STARE-Cognitive produced the same locked-test values as Cognitive. The locked evaluation validates the adaptive cognitive scheduling architecture, but does not isolate STARE as the cause of the improvement.',
  },
  {
    title: 'MPC comparison',
    text: 'The locked paired MPC-vs-Cognitive Pd interval (−0.0265, 95% CI [−0.0831, +0.0230]) crosses zero: no clear Pd separation between the two is established by this evaluation.',
  },
  {
    title: 'Synthetic data',
    text: 'The corpus is the TSRD-derived synthetic PDW dataset. It enables controlled experiments but does not guarantee real-RF performance.',
  },
  {
    title: 'MIT artefact',
    text: 'The original locked report\u2019s mean-intercept-time column contained a timestamp artefact (≈1e-17 s). The disclosed v6 audit re-run (results/locked/locked_test_v6_audit.json) root-fixed it; MIT is therefore not quoted as a locked headline metric here.',
  },
  {
    title: 'Demo vs locked evidence',
    text: 'The interactive scenarios on this site are browser simulations used for explanation. Quantitative claims come only from the locked research evaluation, which ran separately on the evaluation pipeline.',
  },
];

/**
 * SECONDARY PROTOCOL — development tournament (readme §17). Exploratory, run BEFORE the
 * model freeze; never merged with the locked table. Shown only on the Benchmark page,
 * always under its own label.
 */
export const DEV_TOURNAMENT = {
  title: 'Development tournament (exploratory — pre-freeze, separate protocol)',
  source: 'readme.md §17 (results/tournament/dev_tournament.json)',
  config: '12 policies × 7 scenario families (A-Fixed, B-Sparse, C-Hopping, D-BurstLPI, E-Dense, F-Persistent, G-Adversarial)',
  rows: [
    { id: 'sequential', label: 'Sequential', pd: 0.1455, wPoi: 0.152, reward: 4.55 },
    { id: 'ucb1', label: 'UCB1', pd: 0.197, wPoi: 0.2086, reward: 5.264 },
    { id: 'linucb', label: 'LinUCB', pd: 0.3128, wPoi: 0.3085, reward: 6.978 },
    { id: 'thompson', label: 'Thompson', pd: 0.224, wPoi: 0.2343, reward: 5.675 },
    { id: 'markov', label: 'Markov', pd: 0.2259, wPoi: 0.2273, reward: 5.556 },
    { id: 'cognitive', label: 'Cognitive', pd: 0.368, wPoi: 0.3508, reward: 8.121 },
    { id: 'mpc', label: 'MPC', pd: 0.4244, wPoi: 0.4099, reward: 8.978 },
    { id: 'stare_cognitive', label: 'STARE-Cognitive', pd: 0.368, wPoi: 0.3508, reward: 8.121 },
  ],
  cognitiveVsSequential: {
    deltaPd: 0.2225,
    deltaPdCi: [0.0451, 0.407] as const,
    deltaHpMiss: -0.1537,
    deltaReward: 3.5714,
  },
} as const;

/**
 * SECONDARY PROTOCOL — RL research track (readme §16). Separate environment and
 * configuration from the locked test; shown only as a labelled research result.
 */
export const RL_TRACK = {
  title: 'RL research track (separate protocol)',
  source: 'readme.md §16',
  config: 'Seed 26055 · 200 episodes per algorithm · simulation environment',
  rows: [
    { id: 'dqn', label: 'DQN', reward: 4.271, detectionRate: 0.125 },
    { id: 'ddqn', label: 'Double DQN', reward: 8.463, detectionRate: 0.246 },
    { id: 'dueling_dqn', label: 'Dueling DQN', reward: 0.277, detectionRate: 0.0 },
    { id: 'dueling_ddqn', label: 'Dueling Double DQN', reward: 5.432, detectionRate: 0.154 },
    { id: 'ppo', label: 'PPO', reward: 5.058, detectionRate: 0.143 },
  ],
  note: 'Environment/configuration-specific result, not a universal claim about RL algorithms.',
} as const;

/** Adaptive-decision loop steps shown on the Benchmark page (readme §2, §6, §11–§15). */
export const ADAPTIVE_LOOP_STEPS = [
  'Observation (pulse descriptor word on the tuned channel)',
  'Belief / activity estimate per channel',
  'Uncertainty + temporal context (staleness, hazard, change detection)',
  'Prediction / information value',
  'Adaptive scan decision (channel + dwell)',
  'HIT / MISS feedback',
  'Updated belief',
] as const;
