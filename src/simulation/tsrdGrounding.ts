/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { EmitterDefinition, EmitterType } from '../state/types';
import { SeededRNG } from '../state/seededRng';

// PROVENANCE: copied verbatim from ../src/environment/tsrdGrounding.ts (repo root).

export interface TSRDClassDistribution {
  className: string;
  threatPriority: number;
  priMinMs: number;
  priMaxMs: number;
  priJitterPct: number;
  pwMinUs: number;
  pwMaxUs: number;
  snrMinDb: number;
  snrMaxDb: number;
  hopAgility: 'fixed' | 'markov_cycle' | 'pseudo_random' | 'evasive_reactive';
  channels: number[];
  dwellBurstCount: number;
}

/**
 * Priority 17 — TSRD (Tactical Spectrum Radar Dataset) Grounding Layer:
 * Statistical parameters derived from empirical Electronic Support / ELINT radar signal datasets:
 * Combines TSRD statistical realism with synthetic experimental controllability.
 */
export const TSRD_RADAR_DISTRIBUTIONS: TSRDClassDistribution[] = [
  {
    className: 'SAM Fire-Control Tracking Radar (X-Band)',
    threatPriority: 5,
    priMinMs: 0.65,
    priMaxMs: 1.25,
    priJitterPct: 8.0,
    pwMinUs: 0.4,
    pwMaxUs: 1.2,
    snrMinDb: 16.0,
    snrMaxDb: 26.0,
    hopAgility: 'markov_cycle',
    channels: [8, 9, 10, 11],
    dwellBurstCount: 16,
  },
  {
    className: 'Airborne Interceptor Multi-Mode Radar (Ku-Band)',
    threatPriority: 5,
    priMinMs: 0.45,
    priMaxMs: 0.95,
    priJitterPct: 15.0,
    pwMinUs: 0.2,
    pwMaxUs: 0.8,
    snrMinDb: 14.0,
    snrMaxDb: 24.0,
    hopAgility: 'pseudo_random',
    channels: [16, 17, 18, 19, 20, 21, 22, 23],
    dwellBurstCount: 12,
  },
  {
    className: 'Naval 3D Air Surveillance Search Radar (S-Band)',
    threatPriority: 3,
    priMinMs: 2.4,
    priMaxMs: 4.8,
    priJitterPct: 4.0,
    pwMinUs: 8.0,
    pwMaxUs: 22.0,
    snrMinDb: 18.0,
    snrMaxDb: 28.0,
    hopAgility: 'fixed',
    channels: [3, 4],
    dwellBurstCount: 6,
  },
  {
    className: 'Anti-Ship Missile Active Radar Seeker (Ka-Band)',
    threatPriority: 5,
    priMinMs: 0.35,
    priMaxMs: 0.70,
    priJitterPct: 20.0,
    pwMinUs: 0.15,
    pwMaxUs: 0.5,
    snrMinDb: 12.0,
    snrMaxDb: 22.0,
    hopAgility: 'evasive_reactive',
    channels: [24, 25, 26, 27, 28, 29, 30, 31],
    dwellBurstCount: 24,
  },
  {
    className: 'Tactical Battlefield Surveillance Radar (C-Band)',
    threatPriority: 4,
    priMinMs: 1.2,
    priMaxMs: 2.2,
    priJitterPct: 10.0,
    pwMinUs: 1.5,
    pwMaxUs: 4.5,
    snrMinDb: 15.0,
    snrMaxDb: 22.0,
    hopAgility: 'markov_cycle',
    channels: [12, 13, 14, 15],
    dwellBurstCount: 10,
  },
  {
    className: 'Low-Probability-of-Intercept (LPI) Frequency Hopper',
    threatPriority: 4,
    priMinMs: 0.8,
    priMaxMs: 1.8,
    priJitterPct: 25.0,
    pwMinUs: 0.3,
    pwMaxUs: 1.0,
    snrMinDb: 8.0, // Low SNR challenge
    snrMaxDb: 15.0,
    hopAgility: 'pseudo_random',
    channels: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    dwellBurstCount: 8,
  },
];

export class TSRDGroundingEngine {
  /**
   * Samples a realistic emitter configuration from TSRD empirical distributions
   */
  public static sampleEmitterFromDistribution(
    dist: TSRDClassDistribution,
    idPrefix: string = 'tsrd',
    index: number = 0,
    rng: SeededRNG = new SeededRNG(42)
  ): EmitterDefinition {
    const pri = dist.priMinMs + rng.next() * (dist.priMaxMs - dist.priMinMs);
    const pw = dist.pwMinUs + rng.next() * (dist.pwMaxUs - dist.pwMinUs);
    const snr = dist.snrMinDb + rng.next() * (dist.snrMaxDb - dist.snrMinDb);
    const primaryCh = dist.channels[index % dist.channels.length];

    let pattern: number[] = [];
    if (dist.hopAgility === 'markov_cycle') {
      pattern = [...dist.channels];
    } else if (dist.hopAgility === 'pseudo_random') {
      // Shuffled pseudo-random sequence
      pattern = [...dist.channels];
      for (let i = pattern.length - 1; i > 0; i--) { const j = rng.int(0, i); [pattern[i], pattern[j]] = [pattern[j], pattern[i]]; }
    } else {
      pattern = [primaryCh];
    }

    let eType: EmitterType = 'frequency_agile';
    if (dist.hopAgility === 'fixed') eType = 'periodic_scan';
    else if (dist.hopAgility === 'evasive_reactive') eType = 'reactive_jammer';
    else if (dist.className.includes('LPI')) eType = 'burst_lpi';

    return {
      id: `${idPrefix}-${dist.className.split(' ')[0].toLowerCase()}-${index}`,
      name: `${dist.className} #${index + 1}`,
      type: eType,
      baseFreqMHz: 2000 + primaryCh * 250,
      channelSpan: dist.channels.length,
      hoppingSequence: pattern,
      currentHopIndex: 0,
      priMs: Number(pri.toFixed(2)),
      pulseWidthUs: Number(pw.toFixed(2)),
      powerDbm: Number((snr - 95.0).toFixed(1)),
      priority: dist.threatPriority,
      snrDb: Number(snr.toFixed(1)),
      scanPeriodMs: 120.0,
      beamwidthDeg: 18.0,
      currentAzimuthDeg: 0,
      burstDurationMs: dist.dwellBurstCount * pri,
      burstIntervalMs: 8.0,
      isBurstActive: true,
      isReactive: dist.hopAgility === 'evasive_reactive',
    };
  }

  /**
   * Generates a complete TSRD-grounded scenario with controlled emitter density
   */
  public static generateTSRDScenario(
    scenarioName: string = 'TSRD Grounded High-Fidelity Tactical Threat Environment',
    density: 'light' | 'moderate' | 'dense' = 'moderate',
    seed = 42
  ): EmitterDefinition[] {
    const emitters: EmitterDefinition[] = [];
    const rng = new SeededRNG(seed);
    const countPerClass = density === 'light' ? 1 : density === 'moderate' ? 2 : 3;

    TSRD_RADAR_DISTRIBUTIONS.forEach((dist, dIdx) => {
      for (let i = 0; i < countPerClass; i++) {
        emitters.push(this.sampleEmitterFromDistribution(dist, `tsrd-${dIdx}`, i, rng));
      }
    });

    return emitters;
  }
}
