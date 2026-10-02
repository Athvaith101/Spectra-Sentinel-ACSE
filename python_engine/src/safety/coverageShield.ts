/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ChannelBelief } from '../types';

/**
 * Tactical Coverage Shield:
 * Enforces explicit mathematical surveillance deadlines across all spectrum channels.
 *
 * Explicit Deadline Formulation:
 * T_stale(c) <= W_c(P_c)
 *
 * where:
 * W_c(P_c) = W_base / (1 + λ * P_c)
 *
 * Higher threat priority directly tightens the surveillance deadline window W_c.
 * E.g., for W_base = 120 ms, λ = 0.18:
 * - Priority 5 (Lethal Target Tracker): W_c = 120 / (1 + 0.90) = 63.2 ms
 * - Priority 4 (Fire Control Radar):    W_c = 120 / (1 + 0.72) = 69.8 ms
 * - Priority 3 (Standard Surveillance):  W_c = 120 / (1 + 0.54) = 77.9 ms
 * - Priority 1 (Background Emitter):    W_c = 120 / (1 + 0.18) = 101.7 ms
 */
export class CoverageShield {
  public baseDeadlineMs: number;
  public lambda: number;
  public enabled: boolean = true;

  constructor(baseDeadlineMs: number = 260.0, lambda: number = 0.18) {
    this.baseDeadlineMs = baseDeadlineMs;
    this.lambda = lambda;
  }

  /**
   * Computes the explicit surveillance deadline in milliseconds for a given priority level
   */
  public getDeadlineMs(priority: number, totalHits: number = 1): number {
    const base = totalHits === 0 ? this.baseDeadlineMs * 1.8 : this.baseDeadlineMs;
    return base / (1.0 + this.lambda * Math.max(1, Math.min(5, priority)));
  }

  /**
   * Returns the hard-feasible action set A_safe = {c : T_stale(c) <= W_c}.
   * If every channel is overdue, the least-violating channel is returned so the
   * optimizer still has a deterministic recovery action. This method is intended
   * to be applied BEFORE utility/RL/MPC argmax, not as a post-hoc override.
   */
  public getSafeActionSet(beliefs: ChannelBelief[]): { safe: number[]; forcedRecovery: boolean; worstChannel: number } {
    if (!this.enabled || beliefs.length === 0) {
      return { safe: beliefs.map((_, i) => i), forcedRecovery: false, worstChannel: -1 };
    }
    const safe: number[] = [];
    let worstChannel = 0;
    let worstRatio = -Infinity;
    for (let c = 0; c < beliefs.length; c++) {
      const b = beliefs[c];
      const deadline = this.getDeadlineMs(b.threatPriority, b.totalHits);
      const ratio = b.stalenessMs / Math.max(1, deadline);
      if (ratio <= 1.0) safe.push(c);
      if (ratio > worstRatio) { worstRatio = ratio; worstChannel = c; }
    }
    return { safe: safe.length ? safe : [worstChannel], forcedRecovery: safe.length === 0, worstChannel };
  }

  /**
   * Evaluates channel beliefs against explicit priority deadlines W_c(P_c).
   * Triggers safety override if T_stale(c) >= W_c(P_c).
   */
  public evaluateShield(beliefs: ChannelBelief[]): {
    triggered: boolean;
    channelIndex: number;
    urgency: 'critical' | 'warning' | 'none';
    stalenessMs: number;
    deadlineMs: number;
    violationRatio: number;
  } {
    if (!this.enabled || beliefs.length === 0) {
      return {
        triggered: false,
        channelIndex: -1,
        urgency: 'none',
        stalenessMs: 0,
        deadlineMs: this.baseDeadlineMs,
        violationRatio: 0,
      };
    }

    let maxViolationRatio = -1;
    let worstChannel = -1;
    let worstStaleness = 0;
    let worstDeadline = this.baseDeadlineMs;

    for (let c = 0; c < beliefs.length; c++) {
      const b = beliefs[c];
      const deadline = this.getDeadlineMs(b.threatPriority, b.totalHits);
      const ratio = b.stalenessMs / Math.max(1.0, deadline);

      if (ratio > maxViolationRatio) {
        maxViolationRatio = ratio;
        worstChannel = c;
        worstStaleness = b.stalenessMs;
        worstDeadline = deadline;
      }
    }

    // 1. Critical hard deadline violation: T_stale(c) >= W_c(P_c) (ratio >= 1.25 or ratio >= 1.0 for active threats)
    const isCritical =
      worstChannel !== -1 &&
      (maxViolationRatio >= 1.25 ||
        (maxViolationRatio >= 1.0 && beliefs[worstChannel].threatPriority >= 4));

    if (isCritical) {
      return {
        triggered: true,
        channelIndex: worstChannel,
        urgency: 'critical',
        stalenessMs: worstStaleness,
        deadlineMs: worstDeadline,
        violationRatio: maxViolationRatio,
      };
    }

    // 2. Preemptive soft warning: T_stale(c) >= 0.90 * W_c(P_c) for high-priority threats
    if (
      worstChannel !== -1 &&
      beliefs[worstChannel].threatPriority >= 4 &&
      maxViolationRatio >= 0.90
    ) {
      return {
        triggered: true,
        channelIndex: worstChannel,
        urgency: 'warning',
        stalenessMs: worstStaleness,
        deadlineMs: worstDeadline,
        violationRatio: maxViolationRatio,
      };
    }

    return {
      triggered: false,
      channelIndex: -1,
      urgency: 'none',
      stalenessMs: worstStaleness,
      deadlineMs: worstDeadline,
      violationRatio: maxViolationRatio,
    };
  }
}
