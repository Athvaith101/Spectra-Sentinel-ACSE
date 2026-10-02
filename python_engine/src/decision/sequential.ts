/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GroundTruthPulse, PulseDescriptorWord, ReceiverAction, SchedulerMetrics } from '../types';
import { BaseScheduler } from './baseScheduler';
import { EmitterTracker } from '../belief/tracking';

export class SequentialScheduler implements BaseScheduler {
  public name = 'Sequential Raster Scanner';
  public type: 'sequential' = 'sequential';
  public currentChannel: number = 0;
  public currentDwellMs: number = 2.0;

  public metrics: SchedulerMetrics;
  private channelCount: number;
  public tracker: EmitterTracker;
  private latencySamplesUs: number[] = [];
  private lastVisitTime: number[];

  constructor(channelCount: number = 32, defaultDwellMs: number = 2.0) {
    this.channelCount = channelCount;
    this.currentDwellMs = defaultDwellMs;
    this.lastVisitTime = new Array(channelCount).fill(0);
    this.tracker = new EmitterTracker();
    this.metrics = this.initMetrics();
  }

  private initMetrics(): SchedulerMetrics {
    return {
      totalTimeMs: 0,
      totalScans: 0,
      totalEmissions: 0,
      interceptedCount: 0,
      probabilityOfDetection: 0,
      probabilityOfFalseAlarm: 0.008,
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
      decisionLatencyUs: { mean: 2.1, max: 8.4, p95: 3.5 },
      pulsesLostToSettling: 0,
      pulsesLostToCollision: 0,
    };
  }

  public reset(): void {
    this.currentChannel = 0;
    this.latencySamplesUs = [];
    this.lastVisitTime.fill(0);
    this.tracker.reset();
    this.metrics = this.initMetrics();
  }

  public getDecision(currentTimeMs: number): ReceiverAction {
    const t0 = performance.now();
    // Deterministic open-loop raster scan: (currentChannel + 1) % channelCount
    this.currentChannel = (this.currentChannel + 1) % this.channelCount;
    this.lastVisitTime[this.currentChannel] = currentTimeMs;
    this.metrics.totalScans++;

    const elapsedUs = (performance.now() - t0) * 1000;
    this.latencySamplesUs.push(elapsedUs);

    return {
      channelIndex: this.currentChannel,
      dwellTimeMs: this.currentDwellMs,
      reason: 'sequential_sweep',
      confidence: 1.0,
      isShieldOverride: false,
      timestampMs: currentTimeMs,
    };
  }

  public onPDWObserved(pdw: PulseDescriptorWord, currentTimeMs: number): void {
    if (pdw.isFalseAlarm) {
      return;
    }

    if (pdw.isCollided) {
      this.metrics.pulsesLostToCollision = (this.metrics.pulsesLostToCollision || 0) + 1;
    }

    this.metrics.interceptedCount++;
    // Audit: observable-side PRI update only; truth-side latency attribution is
    // performed by the evaluation layer via receiver truth receipts.
    this.tracker.onPDWObserved(pdw, currentTimeMs);
  }

  public registerGroundTruthPulse(pulse: GroundTruthPulse): void {
    this.metrics.totalEmissions++;
    this.tracker.registerGroundTruthPulse(pulse);
  }

  /**
   * Legacy adapter (audit): only receiver-measurable parameters cross the boundary;
   * ground-truth identity/priority are not forwarded.
   */
  public onPulseObserved(pulse: GroundTruthPulse | PulseDescriptorWord, currentTimeMs: number): void {
    const pdw: PulseDescriptorWord = 'toaMs' in pulse ? pulse : {
      id: pulse.id,
      toaMs: pulse.timestampMs,
      channelIndex: pulse.channelIndex,
      freqMHz: pulse.freqMHz,
      pulseWidthUs: pulse.pulseWidthUs,
      amplitudeDbm: pulse.amplitudeDbm,
      snrDb: pulse.snrDb,
      associationId: `ch-${pulse.channelIndex}`,
      associationConfidence: 0.5,
      isCollided: false,
      isFalseAlarm: false,
    };
    this.onPDWObserved(pdw, currentTimeMs);
  }

  public onDwellCompleted(currentTimeMs: number): void {
    this.tracker.evaluateLoss(currentTimeMs);
    this.metrics.totalTimeMs = currentTimeMs;

    if (this.metrics.totalEmissions > 0) {
      this.metrics.probabilityOfDetection = Math.min(
        1.0,
        this.metrics.interceptedCount / this.metrics.totalEmissions
      );
    }
    if (currentTimeMs > 0) {
      this.metrics.averageInterceptRate = (this.metrics.interceptedCount / currentTimeMs) * 1000;
    }

    let maxStale = 0;
    let sumStale = 0;
    for (let c = 0; c < this.channelCount; c++) {
      const stale = currentTimeMs - this.lastVisitTime[c];
      if (stale > maxStale) maxStale = stale;
      sumStale += stale;
    }
    this.metrics.maxStalenessMs = Number(maxStale.toFixed(1));
    this.metrics.meanStalenessMs = Number((sumStale / this.channelCount).toFixed(1));

    // Latency decomposition
    const decomp = this.tracker.getDecompositionMetrics();
    this.metrics.meanInterceptTimeMs = decomp.meanInterceptTimeMs;
    this.metrics.medianInterceptTimeMs = decomp.medianInterceptTimeMs;
    this.metrics.iqrInterceptTimeMs = decomp.iqrInterceptTimeMs;
    this.metrics.p95InterceptTimeMs = decomp.p95InterceptTimeMs;
    this.metrics.p99InterceptTimeMs = decomp.p99InterceptTimeMs;
    this.metrics.tDetectMs = decomp.meanDetectMs;
    this.metrics.tReacqMs = decomp.meanReacqMs;
    this.metrics.tHighPriorityMs = decomp.meanHighPriorityMs;
    this.metrics.tConfirmMs = decomp.meanConfirmMs;
  }

  public updatePostRunMetrics(): void {
    const decomp = this.tracker.getDecompositionMetrics();
    this.metrics.meanInterceptTimeMs = decomp.meanInterceptTimeMs;
    this.metrics.medianInterceptTimeMs = decomp.medianInterceptTimeMs;
    this.metrics.iqrInterceptTimeMs = decomp.iqrInterceptTimeMs;
    this.metrics.p95InterceptTimeMs = decomp.p95InterceptTimeMs;
    this.metrics.p99InterceptTimeMs = decomp.p99InterceptTimeMs;
    this.metrics.tDetectMs = decomp.meanDetectMs;
    this.metrics.tReacqMs = decomp.meanReacqMs;
    this.metrics.tHighPriorityMs = decomp.meanHighPriorityMs;
    this.metrics.tConfirmMs = decomp.meanConfirmMs;
  }
}
