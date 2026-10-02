/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GroundTruthPulse, PulseDescriptorWord } from './types';

// PROVENANCE: copied verbatim from ../src/belief/tracking.ts (repo root).

export interface EmitterEpisodeState {
  emitterId: string;
  emitterName: string;
  priority: number;
  emissionStartMs: number;
  lastEmissionMs: number;
  firstPDWMs: number | null;
  confirmedPDWMs: number | null;
  highPriorityDetectedMs: number | null;
  consecutivePDWs: number;
  totalPDWs: number;
  lastPDWMs: number | null;
  lastChannel: number;
  isLost: boolean;
  lostTimeMs: number | null;
}

export interface LatencyDecompositionMetrics {
  tDetectSamples: number[];
  tReacqSamples: number[];
  tHighPrioritySamples: number[];
  tConfirmSamples: number[];
  allInterceptDelays: number[];
  meanDetectMs: number;
  meanReacqMs: number;
  meanHighPriorityMs: number;
  meanConfirmMs: number;
  meanInterceptTimeMs: number;
  medianInterceptTimeMs: number;
  iqrInterceptTimeMs: number;
  p95InterceptTimeMs: number;
  p99InterceptTimeMs: number;
}

/**
 * Truth-side episode scorer (INFORMATION BOUNDARY — audit):
 *
 * The tracker maintains TWO strictly separated ledgers:
 *
 * 1. TRUTH SIDE — `episodes`, fed by `registerGroundTruthPulse` (omniscient emission
 *    records) and attributed by `attributeTruthReceipt` (ReceiverTruthReceipt sidecar
 *    emitted by the physical receiver). This ledger exists solely for evaluation
 *    metrics: T_detect / T_reacq / T_HP / T_confirm latency decomposition. It is never
 *    consulted by any decision, belief, or prediction pathway.
 *
 * 2. OBSERVABLE SIDE — `estimatedPriMap`, fed exclusively by `onPDWObserved` and keyed
 *    by the receiver's own `associationId` (a measured-frequency clustering tag, not an
 *    identity oracle). This is the only tracker state a scheduler may read at decision
 *    time, and it is derivable from receiver measurements alone.
 *
 * There is deliberately NO join between the two ledgers: policies see association-keyed
 * observables; scoring sees truth-keyed episodes.
 */
export class EmitterTracker {
  // Truth-side ledger (scoring only; not policy-readable state)
  private episodes: Map<string, EmitterEpisodeState> = new Map();
  private tDetectDelays: number[] = [];
  private tReacqDelays: number[] = [];
  private tHighPriorityDelays: number[] = [];
  private tConfirmDelays: number[] = [];
  private allInterceptDelays: number[] = [];

  // Observable-side ledger (policy-readable): PRI estimates keyed by receiver
  // associationId. Derived purely from measured TOA sequences.
  public estimatedPriMap: Map<string, number> = new Map();
  private recentToas: Map<string, number[]> = new Map();

  /**
   * TRUTH SIDE. Called for every simulated emission regardless of detectability.
   * Establishes the ground-truth episode bookkeeping used by latency decomposition.
   */
  public registerGroundTruthPulse(pulse: GroundTruthPulse): void {
    let ep = this.episodes.get(pulse.emitterId);
    if (!ep) {
      ep = {
        emitterId: pulse.emitterId,
        emitterName: pulse.emitterName,
        priority: pulse.priority,
        emissionStartMs: pulse.timestampMs,
        lastEmissionMs: pulse.timestampMs,
        firstPDWMs: null,
        confirmedPDWMs: null,
        highPriorityDetectedMs: null,
        consecutivePDWs: 0,
        totalPDWs: 0,
        lastPDWMs: null,
        lastChannel: pulse.channelIndex,
        isLost: false,
        lostTimeMs: null,
      };
      this.episodes.set(pulse.emitterId, ep);
    } else {
      ep.lastEmissionMs = pulse.timestampMs;
      // If emitter hops channel, previous track is considered lost until reacquired
      if (ep.lastChannel !== pulse.channelIndex && ep.firstPDWMs !== null && !ep.isLost) {
        ep.isLost = true;
        ep.lostTimeMs = pulse.timestampMs;
        ep.consecutivePDWs = 0;
      }
      ep.lastChannel = pulse.channelIndex;
    }
  }

  /**
   * TRUTH SIDE attribution join (audit):
   * Links a produced PDW back to its generating emission via the receiver's truth
   * receipt, recording detection timestamps on the ground-truth episode. Consumed
   * exclusively by the evaluation layer — never by schedulers, beliefs, or predictors.
   */
  public attributeTruthReceipt(
    receipt: { pulseId: string; emitterId: string; toaMs: number; channelIndex: number; isCollided: boolean; isFalseAlarm: boolean },
    pdwId: string,
    observationTimeMs: number
  ): void {
    if (receipt.isFalseAlarm || receipt.isCollided) return;

    const ep = this.episodes.get(receipt.emitterId);
    if (!ep) {
      // Emission never registered (e.g. decoy created mid-step): synthesize a minimal
      // scoring-only episode so the intercept is still accounted for in metrics.
      this.episodes.set(receipt.emitterId, {
        emitterId: receipt.emitterId,
        emitterName: receipt.emitterId,
        priority: 3,
        emissionStartMs: receipt.toaMs,
        lastEmissionMs: receipt.toaMs,
        firstPDWMs: observationTimeMs,
        confirmedPDWMs: null,
        highPriorityDetectedMs: null,
        consecutivePDWs: 1,
        totalPDWs: 1,
        lastPDWMs: observationTimeMs,
        lastChannel: receipt.channelIndex,
        isLost: false,
        lostTimeMs: null,
      });
      const delay = Math.max(0.1, observationTimeMs - receipt.toaMs);
      this.tDetectDelays.push(delay);
      this.allInterceptDelays.push(delay);
      return;
    }

    // 1. Initial detection latency T_detect = t_firstPDW - t_emissionStart
    if (ep.firstPDWMs === null) {
      ep.firstPDWMs = observationTimeMs;
      const tDetect = Math.max(0.1, observationTimeMs - ep.emissionStartMs);
      this.tDetectDelays.push(tDetect);
      this.allInterceptDelays.push(tDetect);

      // High-priority latency T_HP (Priority >= 4): ground-truth episode priority
      if (ep.priority >= 4) {
        ep.highPriorityDetectedMs = observationTimeMs;
        this.tHighPriorityDelays.push(tDetect);
      }
    } else if (ep.isLost && ep.lostTimeMs !== null) {
      // 2. Track reacquisition latency T_reacq = t_reacquired - t_lost
      const tReacq = Math.max(0.1, observationTimeMs - ep.lostTimeMs);
      this.tReacqDelays.push(tReacq);
      this.allInterceptDelays.push(tReacq);
      ep.isLost = false;
      ep.lostTimeMs = null;
    } else {
      // General pulse-to-pulse latency sample
      const prev = ep.lastPDWMs ?? ep.emissionStartMs;
      const d = Math.max(0.1, observationTimeMs - prev);
      this.allInterceptDelays.push(d);
    }

    ep.totalPDWs++;
    ep.consecutivePDWs++;
    ep.lastPDWMs = observationTimeMs;

    // 3. Track confirmation latency T_confirm (N >= 3 consecutive/consistent pulses)
    if (ep.consecutivePDWs >= 3 && ep.confirmedPDWMs === null) {
      ep.confirmedPDWMs = observationTimeMs;
      const tConfirm = Math.max(0.1, observationTimeMs - ep.emissionStartMs);
      this.tConfirmDelays.push(tConfirm);
    }
  }

  /**
   * OBSERVABLE SIDE update (audit): learns PRI statistics exclusively from measured
   * PDW parameters, keyed by the receiver association tag. No ground-truth identity
   * or priority is read here; collided garbled blobs and false alarms are rejected.
   */
  public onPDWObserved(pdw: PulseDescriptorWord, currentTimeMs: number): void {
    if (pdw.isFalseAlarm) return;
    if (pdw.isCollided) return; // garbled blob: timing is unreliable

    // Receiver association tag (fallback key: channel-scoped placeholder when the
    // front end did not provide one). This is an observable clustering key, NOT
    // ground-truth identity.
    const assocKey = pdw.associationId ?? `ch-${pdw.channelIndex}`;

    let toas = this.recentToas.get(assocKey);
    if (!toas) {
      toas = [];
      this.recentToas.set(assocKey, toas);
    }
    toas.push(pdw.toaMs);
    if (toas.length > 8) toas.shift();
    if (toas.length >= 2) {
      const deltas: number[] = [];
      for (let i = 1; i < toas.length; i++) {
        const d = toas[i] - toas[i - 1];
        if (d > 0.05 && d < 25.0) deltas.push(d);
      }
      if (deltas.length > 0) {
        deltas.sort((a, b) => a - b);
        this.estimatedPriMap.set(assocKey, deltas[Math.floor(deltas.length / 2)]);
      }
    }
  }

  /**
   * OBSERVABLE SIDE lookup: PRI estimate for a receiver association tag, for scheduler
   * dwell adaptation. Returns the most recent estimate across tags when the exact tag
   * has no data (e.g. freshly re-associated track) — uses only observable statistics.
   */
  public getEstimatedPri(associationId: string | undefined, channelIndex: number): number | undefined {
    if (associationId) {
      const direct = this.estimatedPriMap.get(associationId);
      if (direct !== undefined) return direct;
    }
    // Channel-scoped fallback key (matches onPDWObserved default keying)
    const fallback = this.estimatedPriMap.get(`ch-${channelIndex}`);
    return fallback;
  }

  /**
   * OBSERVABLE SIDE convenience: most recent PRI estimate on a channel cell across all
   * associations observed on it (pure measurement statistics).
   */
  public getChannelPRI(channelIndex: number): number | undefined {
    let best: number | undefined;
    for (const key of this.estimatedPriMap.keys()) {
      if (key === `ch-${channelIndex}` || key.startsWith(`assoc-${channelIndex}-`)) {
        const v = this.estimatedPriMap.get(key)!;
        if (best === undefined || v < best) best = v;
      }
    }
    return best;
  }

  /**
   * TRUTH SIDE: track-loss evaluation over ground-truth episodes (scoring only).
   */
  public evaluateLoss(currentTimeMs: number): void {
    for (const ep of this.episodes.values()) {
      if (ep.firstPDWMs !== null && !ep.isLost && ep.lastPDWMs !== null) {
        // Loss gating uses the truth episode's own channel so scoring semantics are
        // unchanged; the threshold uses the last observable PRI estimate available
        // for that channel cell (measurement-derived), defaulting as before.
        const pri = this.getChannelPRI(ep.lastChannel) ?? 2.0;
        // If not seen for 3x PRI and > 10ms, mark as lost track requiring reacquisition
        if (currentTimeMs - ep.lastPDWMs > Math.max(10.0, pri * 3.5)) {
          ep.isLost = true;
          ep.lostTimeMs = currentTimeMs;
          ep.consecutivePDWs = 0;
        }
      }
    }
  }

  public getDecompositionMetrics(): LatencyDecompositionMetrics {
    const calcMean = (arr: number[]) => (arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length);
    const sortedAll = [...this.allInterceptDelays].sort((a, b) => a - b);
    const n = sortedAll.length;

    const median = n === 0 ? 0 : sortedAll[Math.floor(n * 0.5)];
    const q1 = n === 0 ? 0 : sortedAll[Math.floor(n * 0.25)];
    const q3 = n === 0 ? 0 : sortedAll[Math.floor(n * 0.75)];
    const iqr = q3 - q1;
    const p95 = n === 0 ? 0 : sortedAll[Math.min(n - 1, Math.floor(n * 0.95))];
    const p99 = n === 0 ? 0 : sortedAll[Math.min(n - 1, Math.floor(n * 0.99))];

    return {
      tDetectSamples: [...this.tDetectDelays],
      tReacqSamples: [...this.tReacqDelays],
      tHighPrioritySamples: [...this.tHighPriorityDelays],
      tConfirmSamples: [...this.tConfirmDelays],
      allInterceptDelays: sortedAll,
      meanDetectMs: Number(calcMean(this.tDetectDelays).toFixed(2)),
      meanReacqMs: Number(calcMean(this.tReacqDelays).toFixed(2)),
      meanHighPriorityMs: Number(calcMean(this.tHighPriorityDelays).toFixed(2)),
      meanConfirmMs: Number(calcMean(this.tConfirmDelays).toFixed(2)),
      meanInterceptTimeMs: Number(calcMean(this.allInterceptDelays).toFixed(2)),
      medianInterceptTimeMs: Number(median.toFixed(2)),
      iqrInterceptTimeMs: Number(iqr.toFixed(2)),
      p95InterceptTimeMs: Number(p95.toFixed(2)),
      p99InterceptTimeMs: Number(p99.toFixed(2)),
    };
  }

  public reset(): void {
    this.episodes.clear();
    this.tDetectDelays = [];
    this.tReacqDelays = [];
    this.tHighPriorityDelays = [];
    this.tConfirmDelays = [];
    this.allInterceptDelays = [];
    this.estimatedPriMap.clear();
    this.recentToas.clear();
  }
}
