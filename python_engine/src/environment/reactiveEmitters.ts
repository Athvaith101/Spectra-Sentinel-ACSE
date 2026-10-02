/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { EmitterDefinition, ReactiveEvasionType, GroundTruthPulse } from '../types';
import { SeededRNG } from '../utils/seededRng';

export interface EvasionLogEvent {
  emitterId: string;
  timestampMs: number;
  tactic: ReactiveEvasionType;
  description: string;
}

/**
 * Closed-Loop Adversarial Reactive Threat Manager:
 * Simulates intelligent cognitive radars that monitor hostile EW intercepts
 * (via Radar Warning Receivers / ES illumination detectors) and dynamically adapt
 * their transmission tactics:
 *
 * P(e_{t+1} | e_t, a_t) instead of open-loop P(e_{t+1} | e_t)
 *
 * Implements 7 distinct evasive counter-strategies:
 * - Type A: Frequency Shift (agile jump to non-adjacent band)
 * - Type B: PRI Stagger (dynamic jitter to break PRI trackers)
 * - Type C: Pulse-Width Reduction (lowers intercept energy)
 * - Type D: Silence/Burst Blanking (evades dwell pursuit)
 * - Type E: Decoy Emission (spawns decoy pulses on decoy channels)
 * - Type F: Deceptive Markov Pattern (inverts predictable sequence)
 * - Type G: Uniform Randomization (destroys Markov predictability)
 */
export class ReactiveEmitterManager {
  private interceptCounts: Map<string, number> = new Map();
  private lastAdaptTime: Map<string, number> = new Map();
  public evasionLogs: EvasionLogEvent[] = [];
  public decoyPulses: GroundTruthPulse[] = [];
  private rng: SeededRNG;

  constructor(seed = 42) { this.rng = new SeededRNG(seed); }

  public registerIntercept(
    emitterId: string,
    currentTimeMs: number,
    emitters: EmitterDefinition[]
  ): { adapted: boolean; tactic?: ReactiveEvasionType } {
    const count = (this.interceptCounts.get(emitterId) || 0) + 1;
    this.interceptCounts.set(emitterId, count);

    const em = emitters.find((e) => e.id === emitterId);
    if (!em || !em.isReactive) return { adapted: false };

    em.interceptCount = count;

    const lastAdapt = this.lastAdaptTime.get(emitterId) || 0;
    // Adapt at most once every 250 ms after at least 6 intercepts
    if (count >= 6 && currentTimeMs - lastAdapt > 250) {
      this.lastAdaptTime.set(emitterId, currentTimeMs);
      const tactic = this.applyEvasiveTactics(em, currentTimeMs);
      em.evasionCount = (em.evasionCount || 0) + 1;
      return { adapted: true, tactic };
    }

    return { adapted: false };
  }

  private applyEvasiveTactics(em: EmitterDefinition, currentTimeMs: number): ReactiveEvasionType {
    const channelCount = 32;
    // Pick tactic from definition, or cycle across strategies
    const tactics: ReactiveEvasionType[] = [
      'typeA_freq_shift',
      'typeB_pri_stagger',
      'typeC_pw_reduction',
      'typeD_silence_burst',
      'typeE_decoy_emission',
      'typeF_deceptive_markov',
      'typeG_randomization',
    ];

    const chosenTactic: ReactiveEvasionType =
      em.reactiveTactic ?? tactics[this.rng.int(0, tactics.length - 1)];

    let logDesc = '';

    switch (chosenTactic) {
      case 'typeA_freq_shift': {
        // Shift hopping channels by a large non-adjacent stride
        const stride = this.rng.int(0, 8) + 8;
        em.hoppingSequence = em.hoppingSequence.map((ch) => (ch + stride) % channelCount);
        logDesc = `Frequency Shift: Shifted hop channels by +${stride} bins to evade lock`;
        break;
      }

      case 'typeB_pri_stagger': {
        // PRI stagger / jitter ±25%
        const jitter = 0.75 + this.rng.next() * 0.5;
        em.priMs = Number((Math.max(0.6, em.priMs * jitter)).toFixed(2));
        logDesc = `PRI Stagger: Jittered PRI to ${em.priMs.toFixed(2)} ms to break dwell alignment`;
        break;
      }

      case 'typeC_pw_reduction': {
        // Pulse width reduction to minimize receiver intercept energy
        em.pulseWidthUs = Math.max(0.3, Number((em.pulseWidthUs * 0.7).toFixed(2)));
        logDesc = `PW Reduction: Compressed pulse width to ${em.pulseWidthUs.toFixed(2)} μs (-3 dB energy)`;
        break;
      }

      case 'typeD_silence_burst': {
        // Emission blanking / radar silence
        em.isBurstActive = false;
        em.burstIntervalMs = 120 + this.rng.next() * 80;
        logDesc = `Silence Blanking: Emitter entered RF silence for ${em.burstIntervalMs.toFixed(0)} ms`;
        break;
      }

      case 'typeE_decoy_emission': {
        // Generate decoy pulses
        const decoyCh = (em.hoppingSequence[0] + 13) % channelCount;
        this.decoyPulses.push({
          id: `decoy-${currentTimeMs}`,
          timestampMs: currentTimeMs,
          emitterId: `decoy-${em.id}`,
          emitterName: `${em.name} [Decoy]`,
          emitterType: 'burst_lpi',
          channelIndex: decoyCh,
          freqMHz: 2000 + decoyCh * 125,
          pulseWidthUs: em.pulseWidthUs,
          amplitudeDbm: em.powerDbm - 3,
          snrDb: 14.0,
          isMainbeam: true,
          priority: 2,
        });
        logDesc = `Decoy Emission: Spawned ghost decoy beacon on Channel ${decoyCh}`;
        break;
      }

      case 'typeF_deceptive_markov': {
        // Invert / reverse sequence to break Markov prediction
        em.hoppingSequence.reverse();
        logDesc = `Deceptive Markov: Inverted transition pattern to trigger predictive misdirection`;
        break;
      }

      case 'typeG_randomization': {
        // Uniform white-noise randomized channel allocation
        const len = em.hoppingSequence.length;
        em.hoppingSequence = Array.from({ length: len }, () => this.rng.int(0, channelCount - 1));
        logDesc = `Uniform Randomization: Collapsed Markov structure to white-noise hops`;
        break;
      }
    }

    this.evasionLogs.push({
      emitterId: em.id,
      timestampMs: currentTimeMs,
      tactic: chosenTactic,
      description: logDesc,
    });
    if (this.evasionLogs.length > 50) this.evasionLogs.shift();

    return chosenTactic;
  }

  public drainDecoyPulses(): GroundTruthPulse[] {
    const decoys = [...this.decoyPulses];
    this.decoyPulses = [];
    return decoys;
  }

  public forceTactic(tactic: ReactiveEvasionType, emitters: EmitterDefinition[], currentTimeMs: number): void {
    if (emitters.length === 0) return;
    const em = emitters[0];
    em.reactiveTactic = tactic;
    this.applyEvasiveTactics(em, currentTimeMs);
    em.evasionCount = (em.evasionCount || 0) + 1;
  }

  public reset(seed?: number): void {
    if (seed !== undefined) this.rng = new SeededRNG(seed);
    this.interceptCounts.clear();
    this.lastAdaptTime.clear();
    this.evasionLogs = [];
    this.decoyPulses = [];
  }
}
