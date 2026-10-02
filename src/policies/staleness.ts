/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GroundTruthPulse, PulseDescriptorWord, ReceiverAction, SchedulerMetrics } from '../state/types';
import { BaseScheduler } from './baseScheduler';

// PROVENANCE: copied verbatim from ../src/decision/staleness.ts (repo root).

export class StalenessGreedyScheduler implements BaseScheduler {
  public name = 'Greedy Staleness Baseline';
  public type: 'staleness' = 'staleness';
  public currentChannel: number = 0;
  public currentDwellMs: number = 1.5;

  public metrics: SchedulerMetrics;
  private channelCount: number;
  private lastVisitTime: number[];
  private channelHits: number[];
  private firstDetectTimeMap: Map<string, number> = new Map();
  private interceptDelays: number[] = [];

  constructor(channelCount: number = 32) {
    this.channelCount = channelCount;
    this.lastVisitTime = new Array(channelCount).fill(0);
    this.channelHits = new Array(channelCount).fill(0);
    this.metrics = this.initMetrics();
  }

  private initMetrics(): SchedulerMetrics {
    return {
      totalTimeMs: 0,
      totalScans: 0,
      totalEmissions: 0,
      interceptedCount: 0,
      probabilityOfDetection: 0,
      probabilityOfFalseAlarm: 0.005,
      meanInterceptTimeMs: 0,
      medianInterceptTimeMs: 0,
      iqrInterceptTimeMs: 0,
      p95InterceptTimeMs: 0,
      p99InterceptTimeMs: 0,
      tDetectMs: 0,
      tReacqMs: 0,
      tHighPriorityMs: 0,
      tConfirmMs: 0,
      averageInterceptRate: 0,
      coverageViolations: 0,
      maxStalenessMs: 0,
      meanStalenessMs: 0,
      decisionLatencyUs: { mean: 4.5, max: 12.0, p95: 6.2 },
    };
  }

  public reset(): void {
    this.currentChannel = 0;
    this.lastVisitTime.fill(0);
    this.channelHits.fill(0);
    this.firstDetectTimeMap.clear();
    this.interceptDelays = [];
    this.metrics = this.initMetrics();
  }

  public getDecision(currentTimeMs: number): ReceiverAction {
    let maxStale = -1;
    let bestChannel = 0;

    for (let c = 0; c < this.channelCount; c++) {
      const stale = currentTimeMs - this.lastVisitTime[c];
      const priorityBoost = 1.0 + (this.channelHits[c] > 0 ? 0.5 : 0);
      const score = stale * priorityBoost;
      if (score > maxStale) {
        maxStale = score;
        bestChannel = c;
      }
    }

    this.currentChannel = bestChannel;
    this.lastVisitTime[bestChannel] = currentTimeMs;
    this.metrics.totalScans++;

    return {
      channelIndex: this.currentChannel,
      dwellTimeMs: this.currentDwellMs,
      reason: 'staleness_greedy',
      confidence: 0.7,
      isShieldOverride: false,
      timestampMs: currentTimeMs,
    };
  }

  public onPDWObserved(pdw: PulseDescriptorWord, currentTimeMs: number): void {
    if (pdw.isFalseAlarm) return;
    this.metrics.interceptedCount++;
    this.channelHits[pdw.channelIndex]++;

    // Audit (information boundary): detection-latency bookkeeping is now keyed by the
    // receiver's observable association tag (measured-frequency clustering), NOT by
    // ground-truth emitter identity. The emission-start estimate falls back to the
    // first measurement of the association rather than a truth timestamp.
    const assocKey = pdw.associationId ?? `ch-${pdw.channelIndex}`;
    const key = `${assocKey}-${Math.floor(pdw.toaMs / 60)}`;
    if (!this.firstDetectTimeMap.has(key)) {
      this.firstDetectTimeMap.set(key, currentTimeMs);
      // Observable estimate of emission start: the measurement time itself (no truth
      // timestamps are available on this side of the boundary).
      const emissionStart = pdw.toaMs;
      const delay = Math.max(0.1, currentTimeMs - emissionStart);
      this.interceptDelays.push(delay);
      const sum = this.interceptDelays.reduce((a, b) => a + b, 0);
      this.metrics.meanInterceptTimeMs = sum / this.interceptDelays.length;
    }
  }

  /**
   * Audit: registration counts emissions for Pd normalization only. Truth emission
   * timestamps are no longer copied into a policy-side map (the former
   * emissionTimeMap join leaked ground-truth timing keyed by truth identity).
   */
  public registerGroundTruthPulse(pulse: GroundTruthPulse): void {
    this.metrics.totalEmissions++;
  }

  public onDwellCompleted(currentTimeMs: number): void {
    this.metrics.totalTimeMs = currentTimeMs;
    if (this.metrics.totalEmissions > 0) {
      this.metrics.probabilityOfDetection = Math.min(1.0, this.metrics.interceptedCount / this.metrics.totalEmissions);
    }
  }
}
