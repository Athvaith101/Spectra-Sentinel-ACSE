/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GroundTruthPulse, PulseDescriptorWord, ReceiverAction, SchedulerMetrics } from '../state/types';

// PROVENANCE: copied verbatim from ../src/decision/baseScheduler.ts (repo root).

export interface BaseScheduler {
  name: string;
  type: 'sequential' | 'smart' | 'staleness' | 'bandit';
  currentChannel: number;
  currentDwellMs: number;
  metrics: SchedulerMetrics;
  getDecision(currentTimeMs: number): ReceiverAction;
  onPDWObserved(pdw: PulseDescriptorWord, currentTimeMs: number): void;
  registerGroundTruthPulse(pulse: GroundTruthPulse): void;
  onDwellCompleted(currentTimeMs: number): void;
  reset(): void;
}
