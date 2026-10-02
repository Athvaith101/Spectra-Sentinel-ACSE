/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { EmitterDefinition, GroundTruthPulse } from '../types';

export const TOTAL_CHANNELS = 32;
export const BASE_FREQ_MHZ = 2000;
export const CHANNEL_BANDWIDTH_MHZ = 250;

export function getChannelCenterFreq(channelIndex: number): number {
  return BASE_FREQ_MHZ + channelIndex * CHANNEL_BANDWIDTH_MHZ + CHANNEL_BANDWIDTH_MHZ / 2;
}

export class EmitterPhysics {
  /**
   * Generates ground truth pulses for a single emitter across a time step [startMs, endMs].
   */
  public static generatePulses(
    em: EmitterDefinition,
    startMs: number,
    endMs: number,
    lastEmitTime: number,
    nextPulseId: { val: number }
  ): { pulses: GroundTruthPulse[]; updatedLastEmitTime: number } {
    const pulses: GroundTruthPulse[] = [];

    // Handle burst gating for LPI emitters
    if (em.type === 'burst_lpi' && em.burstDurationMs && em.burstIntervalMs) {
      const cycleTime = em.burstDurationMs + em.burstIntervalMs;
      const modTime = endMs % cycleTime;
      em.isBurstActive = modTime < em.burstDurationMs;
      if (!em.isBurstActive) {
        return { pulses, updatedLastEmitTime: endMs };
      }
    }

    // Determine current channel index
    let channelIndex = 0;
    if (em.type === 'frequency_agile' || em.type === 'adversarial_shift' || em.type === 'novel_unseen' || em.type === 'reactive_jammer') {
      const seq = em.hoppingSequence;
      if (seq.length > 0) {
        channelIndex = seq[em.currentHopIndex % seq.length];
      }
    } else if (em.type === 'periodic_scan') {
      channelIndex = em.hoppingSequence[0] ?? 4;
    } else {
      channelIndex = em.hoppingSequence[0] ?? 0;
    }

    // Spatial antenna beam pattern (Mainbeam vs Sidelobes)
    let isMainbeam = true;
    if (em.type === 'periodic_scan' && em.scanPeriodMs) {
      const scanPeriod = em.scanPeriodMs;
      const beamwidth = em.beamwidthDeg || 20;
      const beamWindowMs = (beamwidth / 360.0) * scanPeriod;
      const phase = endMs % scanPeriod;
      isMainbeam = phase < beamWindowMs;
    }

    let updatedLastEmit = lastEmitTime;
    let nextDue = lastEmitTime === 0 ? startMs : lastEmitTime + em.priMs;

    while (nextDue <= endMs) {
      if (nextDue >= startMs) {
        const freqMHz = getChannelCenterFreq(channelIndex);
        const amplitudeDbm = isMainbeam ? em.powerDbm : em.powerDbm - 32;
        const snrDb = isMainbeam ? em.snrDb : Math.max(0, em.snrDb - 32);

        pulses.push({
          id: `p-${nextPulseId.val++}`,
          timestampMs: nextDue,
          emitterId: em.id,
          emitterName: em.name,
          emitterType: em.type,
          channelIndex,
          freqMHz,
          pulseWidthUs: em.pulseWidthUs,
          amplitudeDbm,
          snrDb,
          isMainbeam,
          priority: em.priority,
        });

        // Advance hop index on agile emitters
        if (em.type === 'frequency_agile' || em.type === 'adversarial_shift' || em.type === 'novel_unseen' || em.type === 'reactive_jammer') {
          em.currentHopIndex = (em.currentHopIndex + 1) % em.hoppingSequence.length;
          channelIndex = em.hoppingSequence[em.currentHopIndex];
        }
      }
      updatedLastEmit = nextDue;
      nextDue += em.priMs;
    }

    return { pulses, updatedLastEmitTime: updatedLastEmit };
  }
}
