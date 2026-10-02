/**
 * Seeded RF scenario builder for the SIH26055 demonstration layer.
 *
 * COMPOSITION-FIRST (spec §5): the user sets a quantity per emitter class; the total
 * is derived, never typed. Emitters are generated with ONLY parameters that the
 * packaged engine actually implements (src/simulation/emitterModels.ts):
 *   - frequency_agile      (hops through a precomputed sequence per pulse)
 *   - periodic_scan        (fixed channel, rotating mainbeam window)
 *   - burst_lpi            (burst-gated transmission, fixed channel)
 *   - continuous           (steady CW-style illuminator on a fixed channel — engine-
 *                           supported EmitterType, exposed here for the first time)
 *   - adversarial_shift    (agile emitter whose sequence shifts mid-episode — added
 *                           automatically when "adversarial pattern shift" is enabled)
 *   - isReactive           (closed-loop emitter that evades after repeated intercepts)
 *
 * Everything is driven by a displayed seed (SeededRNG / Mulberry32) so any scenario
 * can be reproduced exactly. The spectrum layout is fixed by the engine: 32 bands x
 * 250 MHz spanning 2.0-10.0 GHz; the receiver's instantaneous bandwidth is one band.
 */
import { BenchmarkScenario, EmitterDefinition, ScenarioMissionMode } from '../state/types';
import { SeededRNG } from '../state/seededRng';
import { TOTAL_CHANNELS, BASE_FREQ_MHZ } from '../simulation/rfSimulator';

// PROVENANCE: adapted from ../src/demo/scenarioBuilder.ts (repo root); composition-first
// redesign per the demo spec (§5, §23, §24). Engine physics untouched.

export const LIMITS = {
  minEmitters: 1,
  maxEmitters: 12,
  maxPerClass: 8,
  minDurationMs: 2000,
  maxDurationMs: 6000,
};

export interface EmitterComposition {
  agile: number;      // frequency_agile
  scan: number;       // periodic_scan
  burst: number;      // burst_lpi
  continuous: number; // continuous
}

export interface ScenarioConfig {
  seed: number;
  /** Emitter composition — the source of truth. Total = sum of components. */
  composition: EmitterComposition;
  reactiveEnabled: boolean;
  adversarialShift: boolean;
  durationMs: number;
  /** 0..1 — higher = shorter PRIs / tighter burst cycles */
  activity: number;
  /** 0..1 — higher = wider hop span for agile emitters */
  agility: number;
  /** 0..1 — higher = lower SNR (harder intercepts) */
  difficulty: number;
}

export const DEFAULT_CONFIG: ScenarioConfig = {
  seed: 26055,
  composition: { agile: 2, scan: 1, burst: 2, continuous: 0 },
  reactiveEnabled: false,
  adversarialShift: false,
  durationMs: 4000,
  activity: 0.5,
  agility: 0.5,
  difficulty: 0.35,
};

export function totalEmitters(c: EmitterComposition): number {
  return c.agile + c.scan + c.burst + c.continuous;
}

export function missionModeForConfig(cfg: ScenarioConfig): ScenarioMissionMode {
  const total = totalEmitters(cfg.composition);
  if (cfg.adversarialShift) return 'adversarial_ood';
  if (total >= 6 && cfg.composition.agile >= 2) return 'dense_agile';
  if (cfg.composition.burst >= Math.max(1, Math.floor(total / 2))) return 'sparse_asynchronous';
  return 'general_surveillance';
}

/** Builds a spread, non-colliding hop sequence for an agile emitter. */
function hopSequence(rng: SeededRNG, span: number, offset: number): number[] {
  const len = Math.max(3, Math.min(10, Math.round(3 + span * 0.28)));
  const stride = Math.max(2, Math.round(TOTAL_CHANNELS / Math.max(2, len)));
  const seq: number[] = [];
  let ch = ((offset % TOTAL_CHANNELS) + TOTAL_CHANNELS) % TOTAL_CHANNELS;
  for (let i = 0; i < len; i++) {
    seq.push(ch);
    ch = (ch + stride + rng.int(0, 2)) % TOTAL_CHANNELS;
  }
  return seq;
}

export function buildScenario(cfg: ScenarioConfig): BenchmarkScenario {
  const rng = new SeededRNG(cfg.seed >>> 0);
  const comp = cfg.composition;

  const snrBase = 24 - cfg.difficulty * 10;        // 24 dB (easy) .. 14 dB (hard)
  const priScale = 1.7 - cfg.activity * 1.1;       // 1.7 (quiet) .. 0.6 (busy)
  const spanBase = 4 + Math.round(cfg.agility * 20);
  const emitters: EmitterDefinition[] = [];
  let idx = 0;

  const channelOffset = () => (idx * 9 + rng.int(0, 6)) % TOTAL_CHANNELS;

  for (let i = 0; i < comp.agile; i++, idx++) {
    const snr = Math.max(6, snrBase + rng.float(-3, 2));
    emitters.push({
      id: `gen-${cfg.seed}-agile-${i}`,
      name: `Agile Radar #${i + 1}`,
      type: 'frequency_agile',
      baseFreqMHz: BASE_FREQ_MHZ,
      channelSpan: spanBase,
      hoppingSequence: hopSequence(rng, spanBase, channelOffset()),
      currentHopIndex: 0,
      priMs: Number(Math.min(3.0, Math.max(0.6, rng.float(0.9, 1.9) * priScale)).toFixed(2)),
      pulseWidthUs: Number(rng.float(0.6, 2.2).toFixed(2)),
      powerDbm: Number(-(15 + cfg.difficulty * 8 + rng.float(0, 6)).toFixed(1)),
      priority: rng.next() < 0.5 ? 5 : 4,
      snrDb: Number(snr.toFixed(1)),
    });
  }

  for (let i = 0; i < comp.scan; i++, idx++) {
    const snr = Math.max(6, snrBase + rng.float(-2, 3));
    emitters.push({
      id: `gen-${cfg.seed}-scan-${i}`,
      name: `Scanning Radar #${i + 1}`,
      type: 'periodic_scan',
      baseFreqMHz: BASE_FREQ_MHZ,
      channelSpan: 1,
      hoppingSequence: [channelOffset()],
      currentHopIndex: 0,
      priMs: Number(Math.min(3.2, Math.max(0.9, rng.float(1.5, 2.5) * priScale)).toFixed(2)),
      pulseWidthUs: Number(rng.float(2.5, 5.0).toFixed(2)),
      powerDbm: Number(-(14 + cfg.difficulty * 8 + rng.float(0, 4)).toFixed(1)),
      priority: rng.next() < 0.4 ? 4 : 3,
      scanPeriodMs: Number(rng.float(100, 180).toFixed(1)),
      beamwidthDeg: Number(rng.float(14, 30).toFixed(1)),
      snrDb: Number(snr.toFixed(1)),
    });
  }

  for (let i = 0; i < comp.burst; i++, idx++) {
    const snr = Math.max(5, snrBase - 4 + rng.float(-2, 2));
    const burstDurationMs = Number(rng.float(6, 14).toFixed(1));
    const burstIntervalMs = Number(Math.max(25, rng.float(45, 95) - cfg.activity * 20).toFixed(1));
    emitters.push({
      id: `gen-${cfg.seed}-burst-${i}`,
      name: `Burst LPI Emitter #${i + 1}`,
      type: 'burst_lpi',
      baseFreqMHz: BASE_FREQ_MHZ,
      channelSpan: 1,
      hoppingSequence: [channelOffset()],
      currentHopIndex: 0,
      priMs: Number(Math.min(2.2, Math.max(0.7, rng.float(0.9, 1.4) * priScale)).toFixed(2)),
      pulseWidthUs: Number(rng.float(0.5, 1.2).toFixed(2)),
      powerDbm: Number(-(20 + cfg.difficulty * 6 + rng.float(0, 5)).toFixed(1)),
      priority: 5,
      burstDurationMs,
      burstIntervalMs,
      snrDb: Number(snr.toFixed(1)),
    });
  }

  for (let i = 0; i < comp.continuous; i++, idx++) {
    // Engine-supported 'continuous' type: steady emission on one channel. A forgiving
    // easy-to-find signal; useful as a calibration baseline in judge demos.
    const snr = Math.max(8, snrBase + rng.float(-1, 4));
    emitters.push({
      id: `gen-${cfg.seed}-cont-${i}`,
      name: `Continuous Illuminator #${i + 1}`,
      type: 'continuous',
      baseFreqMHz: BASE_FREQ_MHZ,
      channelSpan: 1,
      hoppingSequence: [channelOffset()],
      currentHopIndex: 0,
      priMs: 1.0,
      pulseWidthUs: 5.0,
      powerDbm: Number(-(12 + cfg.difficulty * 6 + rng.float(0, 4)).toFixed(1)),
      priority: 2,
      snrDb: Number(snr.toFixed(1)),
    });
  }

  // Closed-loop reactive evasion (existing ReactiveEmitterManager behaviour).
  if (cfg.reactiveEnabled && emitters.length > 0) {
    const host = [...emitters].reverse().find((e) => e.type === 'frequency_agile') ?? emitters[emitters.length - 1];
    host.isReactive = true;
    host.name = `${host.name} (reactive)`;
  }

  // Mid-episode adversarial pattern shift on a dedicated agile emitter.
  const adversarialShiftTimeMs = cfg.adversarialShift ? Math.round(cfg.durationMs / 2) : undefined;
  if (cfg.adversarialShift) {
    const snr = Math.max(6, snrBase + 2);
    emitters.push({
      id: `gen-${cfg.seed}-adv-0`,
      name: 'Adversarial Shift Radar',
      type: 'adversarial_shift',
      baseFreqMHz: BASE_FREQ_MHZ,
      channelSpan: spanBase,
      hoppingSequence: hopSequence(rng, spanBase, (emitters.length * 7 + 3) % TOTAL_CHANNELS),
      currentHopIndex: 0,
      priMs: Number(Math.min(2.5, Math.max(0.7, rng.float(1.2, 1.8) * priScale)).toFixed(2)),
      pulseWidthUs: Number(rng.float(1.2, 2.0).toFixed(2)),
      powerDbm: Number(-(16 + cfg.difficulty * 8).toFixed(1)),
      priority: 5,
      snrDb: Number(snr.toFixed(1)),
    });
  }

  return {
    id: `gen-${cfg.seed}`,
    name: `Generated Scenario (seed ${cfg.seed})`,
    category: 'Generated',
    description: 'Seeded synthetic environment generated in-browser by the demo scenario builder.',
    durationMs: Math.min(LIMITS.maxDurationMs, Math.max(LIMITS.minDurationMs, cfg.durationMs)),
    channelCount: TOTAL_CHANNELS,
    emitters,
    adversarialShiftTimeMs,
  };
}

export function newSeed(): number {
  return Math.floor(Math.random() * 90000) + 1000;
}

export interface ScenarioPreset {
  id: string;
  name: string;
  description: string;
  config: ScenarioConfig;
}

export const PRESETS: ScenarioPreset[] = [
  {
    id: 'balanced',
    name: 'Balanced Surveillance',
    description: 'Mixed agile, scanning and burst emitters — the default demo scenario.',
    config: { ...DEFAULT_CONFIG },
  },
  {
    id: 'agile-arena',
    name: 'Agile Threat Arena',
    description: 'Fast frequency-hopping radars across a wide span.',
    config: {
      ...DEFAULT_CONFIG,
      seed: 4102,
      composition: { agile: 5, scan: 1, burst: 0, continuous: 0 },
      activity: 0.7,
      agility: 0.8,
      difficulty: 0.3,
    },
  },
  {
    id: 'lpi-hunt',
    name: 'Sparse LPI Hunt',
    description: 'Few, short-burst emitters — hard to find with a blind sweep.',
    config: {
      ...DEFAULT_CONFIG,
      seed: 7309,
      composition: { agile: 0, scan: 1, burst: 2, continuous: 0 },
      activity: 0.25,
      agility: 0.3,
      difficulty: 0.55,
    },
  },
  {
    id: 'dense-stress',
    name: 'Dense + Adversarial',
    description: 'Many emitters, reactive evasion and a mid-run pattern shift.',
    config: {
      ...DEFAULT_CONFIG,
      seed: 90210,
      composition: { agile: 5, scan: 1, burst: 2, continuous: 0 },
      reactiveEnabled: true,
      adversarialShift: true,
      activity: 0.75,
      agility: 0.7,
      difficulty: 0.5,
    },
  },
  {
    id: 'calm-baseline',
    name: 'Calm Baseline',
    description: 'Steady, easy-to-find signals — for understanding the display first.',
    config: {
      ...DEFAULT_CONFIG,
      seed: 1111,
      composition: { agile: 1, scan: 1, burst: 0, continuous: 2 },
      activity: 0.35,
      agility: 0.25,
      difficulty: 0.2,
    },
  },
];

/** UI metadata for each emitter class — names/descriptions only; physics is the engine's. */
export const EMITTER_CLASS_INFO: Record<keyof EmitterComposition, { label: string; hint: string }> = {
  agile: { label: 'Agile scanner', hint: 'Frequency-hopping radar; hops channels pulse by pulse' },
  scan: { label: 'Periodic emitter', hint: 'Rotating surveillance radar; brief mainbeam passes on one band' },
  burst: { label: 'Burst / LPI', hint: 'Short intermittent bursts separated by silence' },
  continuous: { label: 'Continuous emitter', hint: 'Steady illuminator on one band; easy to hold' },
};

export function scenarioSummary(cfg: ScenarioConfig): string[] {
  const comp = cfg.composition;
  const lines = [
    `Emitters: ${totalEmitters(comp)} (${comp.agile} agile, ${comp.scan} scanning, ${comp.burst} burst LPI, ${comp.continuous} continuous${cfg.reactiveEnabled ? ', reactive evasion' : ''})`,
    `Spectrum: 2.0-10.0 GHz in ${TOTAL_CHANNELS} bands of 250 MHz (receiver instantaneous bandwidth: 1 band)`,
    `Horizon: ${(cfg.durationMs / 1000).toFixed(1)} s of simulated time`,
    `Activity ${(cfg.activity * 100).toFixed(0)}% | Agility ${(cfg.agility * 100).toFixed(0)}% | Difficulty ${(cfg.difficulty * 100).toFixed(0)}%`,
  ];
  if (cfg.adversarialShift) lines.push('Adversarial pattern shift at mid-episode');
  return lines;
}
