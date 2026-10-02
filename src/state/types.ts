/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type EmitterType = 
  | 'frequency_agile'     // Radar hopping across channels with Markovian pattern
  | 'periodic_scan'       // Mechanically/electronically scanning radar (mainbeam periodic crossing)
  | 'burst_lpi'           // Sporadic short-burst low-probability-of-intercept
  | 'continuous'          // CW tracking illuminator
  | 'adversarial_shift'   // Hopping pattern dynamically shifts halfway through
  | 'novel_unseen'        // Previously unknown threat with high priority
  | 'reactive_jammer';    // Closed-loop threat that alters hopping if intercepted

export type ReactiveEvasionType =
  | 'typeA_freq_shift'       // Jumps to non-adjacent channel
  | 'typeB_pri_stagger'      // Introduces +/-25% PRI jitter
  | 'typeC_pw_reduction'     // Lowers pulse width to reduce intercept SNR
  | 'typeD_silence_burst'    // Goes RF silent for burst interval
  | 'typeE_decoy_emission'   // Generates simultaneous decoy beacon
  | 'typeF_deceptive_markov' // Inverts predictable transition sequence
  | 'typeG_randomization';   // Reverts to uniform white noise hopping

export type BayesianUpdateMode = 'exact_beta' | 'discounted_beta';
export type HazardModelMode = 'weibull' | 'exponential' | 'empirical';

export interface EmitterDefinition {
  id: string;
  name: string;
  type: EmitterType;
  baseFreqMHz: number;
  channelSpan: number;    // Number of channels it hops across
  hoppingSequence: number[]; // e.g. [3, 8, 14, 2, 9, 21]
  currentHopIndex: number;
  priMs: number;          // Pulse Repetition Interval in ms (e.g. 1.2 ms)
  pulseWidthUs: number;   // Pulse width in microseconds
  powerDbm: number;       // Signal power
  priority: number;       // Threat priority (1 to 5)
  scanPeriodMs?: number;  // Scan rotation period (e.g. 120 ms)
  beamwidthDeg?: number;  // e.g. 15 degrees
  currentAzimuthDeg?: number;
  burstDurationMs?: number; // Burst duration
  burstIntervalMs?: number; // Time between bursts
  isBurstActive?: boolean;
  snrDb: number;
  isReactive?: boolean;
  reactiveTactic?: ReactiveEvasionType;
  interceptCount?: number;
  evasionCount?: number;
}

/**
 * INFORMATION BOUNDARY — TRUTH SIDE (never observable by any scheduler/policy):
 * GroundTruthPulse is the simulator's omniscient emission record. Fields such as
 * emitterId, emitterName, emitterType, isMainbeam and priority are generating-process
 * state. They may be consumed ONLY by (a) the physical propagation/receiver layer to
 * derive stochastic observables, (b) truth-side scoring bookkeeping (episode/latency
 * joins, Pd normalization), and (c) the evaluation/visualization layer. No decision
 * policy, belief update, or predictor may branch on any field of this interface.
 */
export interface GroundTruthPulse {
  id: string;
  timestampMs: number;
  emitterId: string;
  emitterName: string;
  emitterType: EmitterType;
  channelIndex: number;
  freqMHz: number;
  pulseWidthUs: number;
  amplitudeDbm: number;
  snrDb: number;
  isMainbeam: boolean;
  priority: number;
}

/**
 * INFORMATION BOUNDARY — RECEIVER-OBSERVABLE SIDE (the only signal interface policies
 * may consume): A Pulse Descriptor Word contains strictly receiver-measurable parameters
 * plus the receiver's own within-front-end association tag. Ground-truth emitter identity
 * and threat priority MUST NEVER appear here. If evaluation or the closed-loop
 * reactive-emitter loop needs the generating emission's identity, consume the truth-side
 * ReceiverTruthReceipt sidecar — deliberately kept out of this interface.
 */
export interface PulseDescriptorWord {
  id: string;
  toaMs: number;
  channelIndex: number;
  freqMHz: number;
  pulseWidthUs: number;
  amplitudeDbm: number;
  snrDb: number;
  /** Receiver-side association tag: clusters successive PDWs on the same measured
   *  (channel, carrier-frequency) cell. Stable only within one PhysicalReceiverFrontEnd
   *  instance. This is a receiver clustering heuristic — NOT ground-truth identity. */
  associationId?: string;
  /** [0,1] receiver confidence in the association (garbled/collided blobs associate weakly). */
  associationConfidence?: number;
  isFalseAlarm?: boolean;
  isCollided?: boolean;
  detectedBy?: 'sequential' | 'smart' | 'both';
}

/**
 * INFORMATION BOUNDARY — TRUTH-SIDE SIDECAR (not policy-consumable):
 * Mirrors the generating emission's identity for bookkeeping that must see through the
 * channel: episode/latency scoring joins and the simulator's reactive RWR counter-closure
 * (RFSimulator.registerIntercept). Deliberately separate from PulseDescriptorWord so no
 * policy can accidentally consume ground-truth identity.
 */
export interface ReceiverTruthReceipt {
  pdwId: string;
  pulseId: string;    // GroundTruthPulse.id that generated this PDW
  emitterId: string;  // Ground-truth identity, verbatim (incl. 'decoy-*' IDs)
  toaMs: number;
  channelIndex: number;
  isCollided: boolean;
  isFalseAlarm: boolean;
}

export interface ReceiverFrontEndConfig {
  settlingTimeUs: number;      // Synthesizer LO tuning settling time (e.g. 35 us = 0.035 ms)
  noiseFloorDbm: number;       // Thermal noise floor (-95 dBm)
  detectionThresholdDb: number;// CFAR threshold SNR for detection (e.g. 10.5 dB)
  pfaRatePerSec: number;       // False alarm generation rate (e.g. 50/sec across band)
  minPulseOverlapUs: number;   // Minimum overlap between pulse and active dwell window (0.5 us)
  channelBandwidthMHz: number; // 250 MHz per channel
  collisionToleranceDb: number;// Overlap collision amplitude threshold (e.g. within 3 dB)
}

export interface ReceiverFrontEndState {
  currentChannel: number;
  dwellStartTimeMs: number;
  dwellDurationMs: number;
  settlingTimeMs: number;
  isTuning: boolean;
  activeWindowStartMs: number;
  activeWindowEndMs: number;
  totalPulsesArrived: number;
  totalPulsesDuringTuning: number;
  totalPulsesCollided: number;
  totalFalseAlarmsGenerated: number;
  totalValidPDWsProduced: number;
}

export interface ReceiverAction {
  channelIndex: number;
  dwellTimeMs: number;
  reason: 
    | 'transition_pred' 
    | 'hazard_opp_cost' 
    | 'coverage_shield' 
    | 'novelty_explore' 
    | 'sequential_sweep' 
    | 'information_gain'
    | 'staleness_greedy'
    | 'receding_mpc'
    | 'fast_exploit'
    | 'explore_ig'
    | 'confirmation_burst'
    | 'ood_recover'
    | 'gated_hybrid';
  confidence: number;
  isShieldOverride: boolean;
  timestampMs: number;
  operatingMode?: CognitiveOperatingMode;
}

export type CognitiveOperatingMode = 'EXPLOIT' | 'EXPLORE' | 'VERIFY' | 'COVERAGE' | 'RECOVER';

export type ScenarioMissionMode =
  | 'dense_agile'
  | 'sparse_asynchronous'
  | 'adversarial_ood'
  | 'high_threat_alert'
  | 'general_surveillance';

export interface ComponentGates {
  gOod: number;         // P(OOD)
  gHazard: number;      // P(imminent event)
  gCoverage: number;    // P(T_stale > threshold)
  gMht: number;         // P(multi-hypothesis ambiguity > tau)
  gIg: number;          // P(epistemic uncertainty > tau)
  gCp: number;          // Change-point alarm weight
  gPu: number;          // Prediction Usefulness ratio
  isFastPathActive: boolean; // True when executing low-overhead fast path
}

export interface TrackConfidenceState {
  channelIndex: number;
  cTrack: number; // 0 to 1
  regime: 'exploit' | 'verify' | 'explore';
  observationCount: number;
  predictionConsistency: number;
  meanSnr: number;
  stalenessMs: number;
}

export interface CompositeMissionScore {
  totalUtilityJ: number;
  normalizedMitScore: number;
  detectionScore: number;
  coverageScore: number;
  highPriorityDefenseScore: number;
  infoGainScore: number;
  falseAlarmPenalty: number;
  latencyPenalty: number;
}

export interface AdaptationMetrics {
  tAdaptMs: number;       // Time to re-stabilize policy after change point (t_stable - t_change)
  perfBeforeShift: number;// MIT before shift
  perfAfterShift: number; // MIT after shift
  recoveryRatio: number;  // Performance retention after shift
  totalChangePoints: number;
  meanAdaptationTimeMs: number;
}

export interface ChannelBelief {
  channelIndex: number;
  freqMHz: number;
  occupancyProb: number;      // Posterior mean E[p] from Beta(alpha, beta)
  betaAlpha: number;          // Conjugate Beta prior successes
  betaBeta: number;           // Conjugate Beta prior failures
  epistemicUncertainty: number; // Variance Var[p]
  stalenessMs: number;
  lastObservedTimeMs: number;
  totalHits: number;
  totalVisits: number;
  opportunityCost: number;
  hazardRate: number;
  survivalProb: number;
  informationGain: number;
  threatPriority: number;
  shieldAlertLevel: 'green' | 'yellow' | 'red';
}

export interface LatencyBreakdownUs {
  beliefUpdateUs: number;
  transitionLookupUs: number;
  opportunityCostUs: number;
  shieldValidationUs: number;
  totalCycleUs: number;
}
export type LatencyBreakdown = LatencyBreakdownUs;

export interface SchedulerMetrics {
  totalTimeMs: number;
  totalScans: number;
  totalEmissions: number;
  interceptedCount: number;
  probabilityOfDetection: number; // 0 to 1 (Pd)
  probabilityOfFalseAlarm: number; // Pfa
  meanInterceptTimeMs: number;    // MIT
  medianInterceptTimeMs: number;  // Median MIT
  iqrInterceptTimeMs: number;     // Inter-quartile range
  p95InterceptTimeMs: number;
  p99InterceptTimeMs: number;
  // Rigorous latency decomposition:
  tDetectMs: number;              // Initial detection latency T_detect = t_firstPDW - t_emissionStart
  tReacqMs: number;               // Track reacquisition latency T_reacq = t_reacquired - t_lost
  tHighPriorityMs: number;        // High-priority threat interception latency T_HP (Priority >= 4)
  tConfirmMs: number;             // Track confirmation latency T_confirm (N >= 3 pulses)
  averageInterceptRate: number;   // AIR (intercepts per second)
  coverageViolations: number;
  maxStalenessMs: number;
  meanStalenessMs: number;
  decisionLatencyUs: {
    mean: number;
    max: number;
    p95: number;
  };
  pulsesLostToSettling?: number;
  pulsesLostToCollision?: number;
}

export interface BootstrapConfidenceInterval {
  lower: number;
  upper: number;
  meanDiff: number;
  medianDiff: number;
  confidenceLevel: number; // e.g. 0.95
}

export interface PermutationTestResult {
  pValue: number;
  tStat: number;
  permutationsCount: number;
  isSignificant: boolean;
}

export interface PairedStatisticalValidation {
  bootstrapCI: BootstrapConfidenceInterval;
  permutationTest: PermutationTestResult;
  cohensD: number;
  meanReductionPct: number;
  medianReductionPct: number;
  p95ReductionPct: number;
  p99ReductionPct: number;
  episodesCount: number;
  seedsCount: number;
  pairedDifferences: number[];
  isStatisticallyDefensible: boolean;
}

export interface ComparisonResults {
  sequential: SchedulerMetrics;
  smart: SchedulerMetrics;
  interceptTimeReductionPct: number; // >= 80% target
  detectionImprovementPct: number;
  speedupFactor: number;
  subMillisecondLatencyPassed: boolean;
  target80PctImprovementPassed: boolean;
  statisticalValidation?: PairedStatisticalValidation;
  // Retain backward-compatible statisticalConfidence
  statisticalConfidence?: {
    pValue: number;
    isSignificant: boolean;
    sampleCount: number;
  };
}

export interface BenchmarkScenario {
  id: string;
  name: string;
  category: string;
  description: string;
  durationMs: number;
  channelCount: number;
  emitters: EmitterDefinition[];
  adversarialShiftTimeMs?: number;
}

export interface AblationConfig {
  disableCoverageShield: boolean;
  disableMarkovTransition: boolean;
  disableHazardSurvival: boolean;
  disableAdaptiveDwell: boolean;
  disableBayesianBelief: boolean;
}

export const DEFAULT_ABLATION_CONFIG: AblationConfig = {
  disableCoverageShield: false,
  disableMarkovTransition: false,
  disableHazardSurvival: false,
  disableAdaptiveDwell: false,
  disableBayesianBelief: false,
};

export interface AblationRunResult {
  variantId: string;
  variantName: string;
  description: string;
  mitMs: number;
  mitReductionPct: number;
  probabilityOfDetection: number;
  maxStalenessMs: number;
  coverageViolations: number;
  meanDecisionLatencyUs: number;
  status: 'optimal' | 'degraded' | 'critical_failure';
}

export interface ParetoDataPoint {
  schedulerId: string;
  name: string;
  type: 'smart' | 'sequential' | 'ablation' | 'bandit' | 'staleness';
  mitMs: number;
  maxStalenessMs: number;
  decisionLatencyUs: number;
  probabilityOfDetection: number;
  isParetoOptimal: boolean;
}
