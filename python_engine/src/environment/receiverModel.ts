/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  GroundTruthPulse,
  PulseDescriptorWord,
  ReceiverFrontEndConfig,
  ReceiverFrontEndState,
  ReceiverTruthReceipt,
} from '../types';
import { DetectionModel } from './detectionModel';
import { getChannelCenterFreq } from './emitterModels';
import { SeededRNG } from '../utils/seededRng';

export const DEFAULT_RECEIVER_CONFIG: ReceiverFrontEndConfig = {
  settlingTimeUs: 35, // 35 microseconds LO settling time
  noiseFloorDbm: -95,
  detectionThresholdDb: 10.5,
  pfaRatePerSec: 15, // Noise false alarms per second across instantaneous bandwidth
  minPulseOverlapUs: 0.4,
  channelBandwidthMHz: 250,
  collisionToleranceDb: 3.0,
};

/**
 * INFORMATION BOUNDARY NOTE (audit):
 * The front end is the ONLY legitimate bridge between the truth side (GroundTruthPulse)
 * and the observable side (PulseDescriptorWord). Policies may consume PDWs exclusively.
 * Ground-truth emitter identity / threat priority are therefore NOT copied into PDWs.
 * - Policy-observable emitter continuity is provided by the receiver's own
 *   measured-frequency association heuristic (associationId) — an clustering of
 *   successive measured (channel, carrier-frequency) cells, not an identity oracle.
 * - Where bookkeeping genuinely requires the generating emission's identity (truth-side
 *   episode scoring joins, the simulator's reactive RWR counter-closure), the front end
 *   exposes a separate ReceiverTruthReceipt sidecar channel that no policy reads.
 */
export class PhysicalReceiverFrontEnd {
  public config: ReceiverFrontEndConfig;
  public state: ReceiverFrontEndState;
  private currentChannel: number = 0;
  private nextPdwId: number = 1;
  // Receiver-side association state: per-channel tracking of the last measured carrier
  // frequency, used to cluster successive PDWs into observable "tracks". This is derived
  // purely from receiver measurements — never from ground-truth emitter identity.
  private associationRegistry: Map<number, { associationId: string; lastMeasuredFreqMHz: number; count: number }> = new Map();
  // Truth-side sidecar receipts (never exposed to schedulers): consumed by evaluation
  // joins and by RFSimulator.registerIntercept for the closed-loop reactive emitter model.
  private pendingTruthReceipts: ReceiverTruthReceipt[] = [];
  private rng: SeededRNG;

  constructor(config: Partial<ReceiverFrontEndConfig> = {}, seed = 42) {
    this.config = { ...DEFAULT_RECEIVER_CONFIG, ...config };
    this.rng = new SeededRNG(seed);
    this.state = this.initState();
  }

  private initState(): ReceiverFrontEndState {
    const settlingMs = this.config.settlingTimeUs / 1000.0;
    return {
      currentChannel: 0,
      dwellStartTimeMs: 0,
      dwellDurationMs: 2.0,
      settlingTimeMs: settlingMs,
      isTuning: false,
      activeWindowStartMs: settlingMs,
      activeWindowEndMs: 2.0,
      totalPulsesArrived: 0,
      totalPulsesDuringTuning: 0,
      totalPulsesCollided: 0,
      totalFalseAlarmsGenerated: 0,
      totalValidPDWsProduced: 0,
    };
  }

  public reset(initialChannel: number = 0, currentTimeMs: number = 0, seed?: number): void {
    if (seed !== undefined) { this.rng = new SeededRNG(seed); }
    this.currentChannel = initialChannel;
    this.nextPdwId = 1;
    const settlingMs = this.config.settlingTimeUs / 1000.0;
    this.state = {
      currentChannel: initialChannel,
      dwellStartTimeMs: currentTimeMs,
      dwellDurationMs: 2.0,
      settlingTimeMs: settlingMs,
      isTuning: false,
      activeWindowStartMs: currentTimeMs + settlingMs,
      activeWindowEndMs: currentTimeMs + 2.0,
      totalPulsesArrived: 0,
      totalPulsesDuringTuning: 0,
      totalPulsesCollided: 0,
      totalFalseAlarmsGenerated: 0,
      totalValidPDWsProduced: 0,
    };
  }

  /**
   * Receiver-side association heuristic (observable information only):
   * PDWs measured on the same channel within the receiver's frequency resolution of the
   * previous measurement continue the same association tag; otherwise a new tag is minted.
   * Deliberately frequency-cluster-based: two distinct emitters on the same channel cell
   * (or a hopper returning to a cell) merge/split purely as the measurements dictate —
   * this is exactly the ambiguity a real ES receiver faces.
   */
  private getOrCreateAssociation(channelIndex: number, measuredFreqMHz: number): { associationId: string; confidence: number } {
    const freqToleranceMHz = 0.01 * this.config.channelBandwidthMHz; // measurement-resolution tolerance
    const entry = this.associationRegistry.get(channelIndex);
    if (
      entry &&
      Math.abs(measuredFreqMHz - entry.lastMeasuredFreqMHz) <= freqToleranceMHz
    ) {
      entry.lastMeasuredFreqMHz = measuredFreqMHz;
      entry.count++;
      return { associationId: entry.associationId, confidence: 0.9 };
    }
    const newId = `assoc-${channelIndex}-${this.nextPdwId}-${this.rng.next().toString(36).slice(2, 8)}`;
    this.associationRegistry.set(channelIndex, {
      associationId: newId,
      lastMeasuredFreqMHz: measuredFreqMHz,
      count: 1,
    });
    return { associationId: newId, confidence: 0.9 };
  }

  /**
   * Receiver measurement imperfections only (boundary audit): a superheterodyne ES
   * receiver measures carrier frequency, amplitude and SNR with finite resolution and
   * post-detection noise. Applied AFTER all detection physics — never alters ground
   * truth, Pd, collisions, or false alarms.
   */
  private applyMeasurementNoise(p: GroundTruthPulse): { freqMHz: number; amplitudeDbm: number; snrDb: number } {
    const freqErrMHz = (this.rng.next() - 0.5) * 0.01 * this.config.channelBandwidthMHz;
    const ampErrDb = (this.rng.next() - 0.5) * 1.2;
    return {
      freqMHz: p.freqMHz + freqErrMHz,
      amplitudeDbm: p.amplitudeDbm + ampErrDb,
      snrDb: Math.max(0.0, p.snrDb + ampErrDb * 0.5),
    };
  }

  /**
   * Called when scheduler tunes receiver to new channel and assigns dwell duration.
   */
  public tuneToChannel(channelIndex: number, dwellDurationMs: number, currentTimeMs: number): void {
    const channelChanged = channelIndex !== this.currentChannel;
    // Information boundary (audit): leaving a channel drops that cell's measurement
    // context — the receiver has no memory of associations on channels it is not measuring.
    if (channelChanged) {
      this.associationRegistry.delete(this.currentChannel);
    }
    this.currentChannel = channelIndex;
    const settlingMs = channelChanged ? this.config.settlingTimeUs / 1000.0 : 0.005; // 5us minimal retune if same channel

    this.state.currentChannel = channelIndex;
    this.state.dwellStartTimeMs = currentTimeMs;
    this.state.dwellDurationMs = dwellDurationMs;
    this.state.settlingTimeMs = settlingMs;
    this.state.isTuning = settlingMs > 0;
    this.state.activeWindowStartMs = currentTimeMs + settlingMs;
    this.state.activeWindowEndMs = currentTimeMs + dwellDurationMs;
  }

  /**
   * Physically processes pulses that arrive in the RF spectrum during [timeStartMs, timeEndMs].
   * Applies front-end physics: LO settling, pulse overlap, SNR detection, collision, and false alarms.
   *
   * INFORMATION BOUNDARY (audit): output PDWs carry strictly receiver-observable parameters
   * (with receiver measurement noise) and the receiver's own associationId. Ground-truth
   * emitter identity / threat priority are NOT copied into PDWs. The generating emission's
   * identity is delivered only via the truth-side ReceiverTruthReceipt sidecar, retrievable
   * through drainTruthReceipts() by evaluation / the simulator's reactive loop.
   */
  public processPulses(
    pulses: GroundTruthPulse[],
    timeStartMs: number,
    timeEndMs: number
  ): PulseDescriptorWord[] {
    const validPDWs: PulseDescriptorWord[] = [];
    const ch = this.currentChannel;
    const activeStart = this.state.activeWindowStartMs;
    const activeEnd = this.state.activeWindowEndMs;

    // Filter pulses arriving on receiver's tuned channel
    const candidatePulses = pulses.filter((p) => p.channelIndex === ch);

    // Track pulse collision candidate buckets within this time slice
    const collidingCandidates: GroundTruthPulse[] = [];

    for (const p of candidatePulses) {
      this.state.totalPulsesArrived++;

      const pStart = p.timestampMs;
      const pEnd = pStart + p.pulseWidthUs / 1000.0;

      // Check 1: LO Settling Time Gate
      // If pulse finishes before LO finishes settling, it is completely lost!
      if (pEnd <= activeStart) {
        this.state.totalPulsesDuringTuning++;
        continue;
      }

      // If pulse starts after dwell active window ended, it's outside dwell
      if (pStart >= activeEnd) {
        continue;
      }

      // Check 2: Minimum Pulse / Dwell Overlap Window
      const overlapStart = Math.max(pStart, activeStart);
      const overlapEnd = Math.min(pEnd, activeEnd);
      const overlapDurationUs = Math.max(0, (overlapEnd - overlapStart) * 1000.0);
      const minRequiredUs = Math.min(p.pulseWidthUs, this.config.minPulseOverlapUs);

      if (overlapDurationUs < minRequiredUs) {
        continue; // Insufficient overlap to trigger video detector integrator
      }

      // Check 3: Stochastic Detection Probability (Albersheim / Marcum)
      const isDetected = DetectionModel.sampleDetection(p.snrDb, p.pulseWidthUs, 1e-4, this.rng);
      if (!isDetected) {
        continue;
      }

      collidingCandidates.push(p);
    }

    // Check 4: Simultaneous Pulse Overlap & Collision
    // If two pulses overlap in time and power difference is within collision tolerance,
    // receiver front-end discriminator is blinded / pulse is garbled.
    // AUDIT (information boundary): a time-overlapping stronger signal within the
    // collision tolerance physically garbles the weaker pulse as well — the front-end
    // discriminator cannot separate them. The stronger pulse remains detectable, but
    // ONLY if it still satisfies the minimum-overlap integrator requirement.
    for (let i = 0; i < collidingCandidates.length; i++) {
      const p = collidingCandidates[i];
      let hasCollision = false;

      for (let j = 0; j < collidingCandidates.length; j++) {
        if (i === j) continue;
        const other = collidingCandidates[j];
        const p1Start = p.timestampMs;
        const p1End = p1Start + p.pulseWidthUs / 1000.0;
        const p2Start = other.timestampMs;
        const p2End = p2Start + other.pulseWidthUs / 1000.0;

        // Check time collision
        const timeOverlap = !(p1End < p2Start || p2End < p1Start);
        if (timeOverlap) {
          const powerDiffDb = Math.abs(p.amplitudeDbm - other.amplitudeDbm);
          if (powerDiffDb <= this.config.collisionToleranceDb) {
            hasCollision = true;
            break;
          }
        }
      }

      if (hasCollision) {
        this.state.totalPulsesCollided++;
        // Collided pulse descriptor word is flagged — still a measurement, but garbled:
        // association confidence is degraded, and no truth identity leaks through.
        this.pushMeasuredPDW(p, ch, timeStartMs, activeStart, activeEnd, true, validPDWs);
      } else {
        this.state.totalValidPDWsProduced++;
        this.pushMeasuredPDW(p, ch, timeStartMs, activeStart, activeEnd, false, validPDWs);
      }
    }

    // Check 5: Noise False Alarms (Poisson generation during active dwell window)
    const activeSliceDurationMs = Math.max(0, Math.min(timeEndMs, activeEnd) - Math.max(timeStartMs, activeStart));
    if (activeSliceDurationMs > 0) {
      const faCount = DetectionModel.generateNoiseFalseAlarms(activeSliceDurationMs, this.config.pfaRatePerSec, this.rng);
      for (let f = 0; f < faCount; f++) {
        this.state.totalFalseAlarmsGenerated++;
        const randomToa = Math.max(timeStartMs, activeStart) + this.rng.next() * activeSliceDurationMs;
        // Thermal-noise false alarm: measured parameters drawn from the noise-plus-
        // threshold distribution, evaluated at the receiver's actual tuned center frequency.
        const measuredFreqMHz = getChannelCenterFreq(ch) + (this.rng.next() - 0.5) * 0.01 * this.config.channelBandwidthMHz;
        validPDWs.push({
          id: `pdw-fa-${this.nextPdwId++}`,
          toaMs: randomToa,
          channelIndex: ch,
          freqMHz: measuredFreqMHz,
          pulseWidthUs: 0.5 + this.rng.next() * 1.5,
          amplitudeDbm: this.config.noiseFloorDbm + this.config.detectionThresholdDb + this.rng.next() * 2.5,
          snrDb: this.config.detectionThresholdDb + this.rng.next() * 1.5,
          associationId: `assoc-noise-${ch}`,
          associationConfidence: 0.1,
          isCollided: false,
          isFalseAlarm: true,
        });
        // Noise false alarms are receiver-internal events with no generating emission:
        // they produce a truth-side receipt only to keep receipt/PDW bookkeeping aligned.
        this.pendingTruthReceipts.push({
          pdwId: `pdw-fa-${this.nextPdwId - 1}`,
          pulseId: '',
          emitterId: '',
          toaMs: randomToa,
          channelIndex: ch,
          isCollided: false,
          isFalseAlarm: true,
        });
      }
    }

    return validPDWs;
  }

  /**
   * Constructs one observable PDW from a detected emission (or records its loss):
   * applies receiver measurement noise, computes the receiver-side association tag,
   * and appends the truth-side receipt for bookkeeping consumers.
   */
  private pushMeasuredPDW(
    p: GroundTruthPulse,
    ch: number,
    timeStartMs: number,
    activeStartMs: number,
    activeEndMs: number,
    isCollided: boolean,
    out: PulseDescriptorWord[]
  ): void {
    const pStart = p.timestampMs;
    const pEnd = pStart + p.pulseWidthUs / 1000.0;

    // Physical video-detector integrator behaviour (audit): a pulse is produced as a
    // descriptor word only if the receiver measured it for at least the configured
    // minimum overlap within its active window. Partially-overlapped blobs with less
    // measurable energy than that are not parameterized by the front end at all.
    const overlapStart = Math.max(pStart, activeStartMs);
    const overlapEnd = Math.min(pEnd, activeEndMs);
    const overlapUs = Math.max(0, (overlapEnd - overlapStart) * 1000.0);
    if (overlapUs < Math.min(p.pulseWidthUs, this.config.minPulseOverlapUs)) {
      // Detected energy was insufficient for parameter measurement: no PDW, no receipt.
      return;
    }

    const measured = this.applyMeasurementNoise(p);
    const assoc = this.getOrCreateAssociation(ch, measured.freqMHz);

    out.push({
      id: `pdw-${this.nextPdwId++}`,
      toaMs: p.timestampMs,
      channelIndex: ch,
      freqMHz: measured.freqMHz,
      pulseWidthUs: p.pulseWidthUs,
      amplitudeDbm: measured.amplitudeDbm,
      snrDb: measured.snrDb,
      associationId: assoc.associationId,
      associationConfidence: isCollided ? 0.25 : assoc.confidence,
      isCollided,
      isFalseAlarm: false,
    });

    // Truth-side sidecar (NOT part of the observable PDW): preserves the generating
    // emission's identity solely for scoring joins and the reactive RWR closure.
    this.pendingTruthReceipts.push({
      pdwId: `pdw-${this.nextPdwId - 1}`,
      pulseId: p.id,
      emitterId: p.emitterId,
      toaMs: p.timestampMs,
      channelIndex: ch,
      isCollided,
      isFalseAlarm: false,
    });
  }

  /**
   * Truth-side sidecar channel (audit): returns and clears receipts linking produced
   * PDWs to their generating ground-truth emissions. Consumers: benchmark/episode
   * scoring joins and RFSimulator.registerIntercept (reactive emitter closure).
   * Schedulers/policies MUST NOT consume this data.
   */
  public drainTruthReceipts(): ReceiverTruthReceipt[] {
    const receipts = this.pendingTruthReceipts;
    this.pendingTruthReceipts = [];
    return receipts;
  }
}
