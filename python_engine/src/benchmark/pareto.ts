/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ParetoDataPoint } from '../types';

export class ParetoAnalysis {
  /**
   * Computes Pareto optimality across candidates.
   * Multi-objective criteria:
   * - Minimize MIT (mean intercept time)
   * - Minimize Max Staleness
   * - Minimize Decision Latency
   * - Maximize Probability of Detection (Pd)
   */
  public static computeFrontier(points: ParetoDataPoint[]): ParetoDataPoint[] {
    const results: ParetoDataPoint[] = points.map((pt) => ({ ...pt, isParetoOptimal: true }));

    for (let i = 0; i < results.length; i++) {
      const p1 = results[i];
      for (let j = 0; j < results.length; j++) {
        if (i === j) continue;
        const p2 = results[j];

        // Does p2 strictly dominate p1?
        // p2 is at least as good in all 4 metrics and strictly better in at least one
        const p2BetterOrEqualMit = p2.mitMs <= p1.mitMs;
        const p2BetterOrEqualStale = p2.maxStalenessMs <= p1.maxStalenessMs;
        const p2BetterOrEqualLat = p2.decisionLatencyUs <= p1.decisionLatencyUs;
        const p2BetterOrEqualPd = p2.probabilityOfDetection >= p1.probabilityOfDetection;

        const p2StrictlyBetter =
          p2.mitMs < p1.mitMs ||
          p2.maxStalenessMs < p1.maxStalenessMs ||
          p2.decisionLatencyUs < p1.decisionLatencyUs ||
          p2.probabilityOfDetection > p1.probabilityOfDetection;

        if (
          p2BetterOrEqualMit &&
          p2BetterOrEqualStale &&
          p2BetterOrEqualLat &&
          p2BetterOrEqualPd &&
          p2StrictlyBetter
        ) {
          p1.isParetoOptimal = false;
          break;
        }
      }
    }

    return results;
  }
}
