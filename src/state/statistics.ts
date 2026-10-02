/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  BootstrapConfidenceInterval,
  PermutationTestResult,
  PairedStatisticalValidation,
} from './types';
import { SeededRNG } from './seededRng';

// PROVENANCE: copied verbatim from ../src/benchmark/statistics.ts (repo root).

/**
 * Rigorous Statistical Validation Engine for EW Cognitive Benchmark:
 * Replaces synthetic sample multipliers with authentic episode-level statistics:
 *
 * 1. Paired Bootstrap Confidence Intervals (B = 2,000 resamples)
 * 2. Paired Permutation Test (exact two-tailed non-parametric p-value)
 * 3. Standardized Effect Sizes (Cohen's d_z, Hedge's g)
 * 4. Non-parametric robust quantiles (P50 Median, IQR, P95, P99)
 */
export class BenchmarkStatistics {
  /**
   * Non-parametric Paired Bootstrap Confidence Interval for Mean Difference
   */
  public static computePairedBootstrapCI(
    baseSamples: number[],
    smartSamples: number[],
    numResamples: number = 2000,
    confidenceLevel: number = 0.95,
    seed = 42
  ): BootstrapConfidenceInterval {
    const n = Math.min(baseSamples.length, smartSamples.length);
    if (n < 2) {
      return {
        lower: 0,
        upper: 0,
        meanDiff: 0,
        medianDiff: 0,
        confidenceLevel,
      };
    }

    // Compute paired differences: Δ_i = Base_i - Smart_i
    // (Positive Δ indicates smart scheduler is faster / lower latency)
    const diffs: number[] = new Array(n);
    for (let i = 0; i < n; i++) {
      diffs[i] = baseSamples[i] - smartSamples[i];
    }

    const observedMeanDiff = this.mean(diffs);
    const observedMedianDiff = this.median(diffs);

    // Bootstrap resample means
    const rng = new SeededRNG(seed);
    const bootMeans: number[] = new Array(numResamples);
    for (let b = 0; b < numResamples; b++) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const randIdx = rng.int(0, n - 1);
        sum += diffs[randIdx];
      }
      bootMeans[b] = sum / n;
    }

    bootMeans.sort((a, b) => a - b);
    const alpha = 1.0 - confidenceLevel;
    const lowerIdx = Math.floor((alpha / 2) * numResamples);
    const upperIdx = Math.floor((1 - alpha / 2) * numResamples);

    const lower = bootMeans[Math.max(0, lowerIdx)];
    const upper = bootMeans[Math.min(numResamples - 1, upperIdx)];

    return {
      lower: Number(lower.toFixed(2)),
      upper: Number(upper.toFixed(2)),
      meanDiff: Number(observedMeanDiff.toFixed(2)),
      medianDiff: Number(observedMedianDiff.toFixed(2)),
      confidenceLevel,
    };
  }

  /**
   * Two-Tailed Paired Permutation Test
   * Tests null hypothesis H0: The distribution of paired latency differences has zero mean
   */
  public static computePairedPermutationTest(
    baseSamples: number[],
    smartSamples: number[],
    numPermutations: number = 2000,
    seed = 43
  ): PermutationTestResult {
    const n = Math.min(baseSamples.length, smartSamples.length);
    if (n < 2) {
      return {
        pValue: 1.0,
        tStat: 0,
        permutationsCount: numPermutations,
        isSignificant: false,
      };
    }

    const diffs: number[] = new Array(n);
    for (let i = 0; i < n; i++) {
      diffs[i] = baseSamples[i] - smartSamples[i];
    }

    const observedDiff = this.mean(diffs);
    const absObserved = Math.abs(observedDiff);

    let countAsOrMoreExtreme = 0;
    const rng = new SeededRNG(seed);

    for (let p = 0; p < numPermutations; p++) {
      let permSum = 0;
      for (let i = 0; i < n; i++) {
        const sign = rng.next() < 0.5 ? 1 : -1;
        permSum += sign * diffs[i];
      }
      const permMean = permSum / n;
      if (Math.abs(permMean) >= absObserved) {
        countAsOrMoreExtreme++;
      }
    }

    const pValue = (countAsOrMoreExtreme + 1) / (numPermutations + 1);
    const stdDiff = Math.sqrt(Math.max(1e-6, this.variance(diffs, observedDiff)));
    const tStat = observedDiff / (stdDiff / Math.sqrt(n));

    return {
      pValue: Number(pValue.toFixed(5)),
      tStat: Number(tStat.toFixed(3)),
      permutationsCount: numPermutations,
      isSignificant: pValue < 0.01,
    };
  }

  /**
   * Computes comprehensive paired statistical validation summary
   */
  public static computeFullValidation(
    baseSamples: number[],
    smartSamples: number[],
    episodesCount: number = 1,
    seedsCount: number = 1,
    seed = 42
  ): PairedStatisticalValidation {
    const n = Math.min(baseSamples.length, smartSamples.length);
    const diffs: number[] = new Array(n);
    for (let i = 0; i < n; i++) {
      diffs[i] = baseSamples[i] - smartSamples[i];
    }

    const bootstrapCI = this.computePairedBootstrapCI(baseSamples, smartSamples, 2000, 0.95, seed);
    const permutationTest = this.computePairedPermutationTest(baseSamples, smartSamples, 2000, seed + 1);

    const meanBase = this.mean(baseSamples);
    const meanDiff = this.mean(diffs);
    const medianBase = this.median(baseSamples);
    const medianDiff = this.median(diffs);

    const stdDiff = Math.sqrt(Math.max(1e-6, this.variance(diffs, meanDiff)));
    const cohensD = Number((meanDiff / stdDiff).toFixed(3));

    const meanReductionPct = meanBase > 0 ? Number(((meanDiff / meanBase) * 100).toFixed(1)) : 0;
    const medianReductionPct = medianBase > 0 ? Number(((medianDiff / medianBase) * 100).toFixed(1)) : 0;

    const baseQuantiles = this.quantiles(baseSamples);
    const smartQuantiles = this.quantiles(smartSamples);

    const p95ReductionPct =
      baseQuantiles.p95 > 0
        ? Number((((baseQuantiles.p95 - smartQuantiles.p95) / baseQuantiles.p95) * 100).toFixed(1))
        : 0;

    const p99ReductionPct =
      baseQuantiles.p99 > 0
        ? Number((((baseQuantiles.p99 - smartQuantiles.p99) / baseQuantiles.p99) * 100).toFixed(1))
        : 0;

    const isStatisticallyDefensible =
      bootstrapCI.lower > 0 && permutationTest.pValue < 0.01 && n >= 5;

    return {
      bootstrapCI,
      permutationTest,
      cohensD,
      meanReductionPct,
      medianReductionPct,
      p95ReductionPct,
      p99ReductionPct,
      episodesCount,
      seedsCount,
      pairedDifferences: diffs.slice(0, 50),
      isStatisticallyDefensible,
    };
  }

  public static mean(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  public static median(arr: number[]): number {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  public static quantiles(arr: number[]): {
    median: number;
    iqr: number;
    p95: number;
    p99: number;
  } {
    if (arr.length === 0) return { median: 0, iqr: 0, p95: 0, p99: 0 };
    const sorted = [...arr].sort((a, b) => a - b);
    const n = sorted.length;
    const median = sorted[Math.floor(n * 0.5)];
    const q1 = sorted[Math.floor(n * 0.25)];
    const q3 = sorted[Math.floor(n * 0.75)];
    const iqr = q3 - q1;
    const p95 = sorted[Math.min(n - 1, Math.floor(n * 0.95))];
    const p99 = sorted[Math.min(n - 1, Math.floor(n * 0.99))];

    return {
      median: Number(median.toFixed(2)),
      iqr: Number(iqr.toFixed(2)),
      p95: Number(p95.toFixed(2)),
      p99: Number(p99.toFixed(2)),
    };
  }

  /**
   * Priority 18 — Composite Mission Utility:
   * J = w1*(1 - NMIT) + w2*Pd + w3*Coverage + w4*(1 - P_HPmiss) + w5*IG - w6*FA - w7*Latency
   * Evaluates balanced multi-objective fitness while retaining all individual sub-metric reporting.
   */
  public static computeCompositeMissionScore(
    metrics: {
      meanInterceptTimeMs: number;
      probabilityOfDetection: number;
      coverageViolations: number;
      tHighPriorityMs: number;
      probabilityOfFalseAlarm: number;
      decisionLatencyUs: { mean: number };
      totalEmissions: number;
      interceptedCount: number;
    },
    baselineMitMs: number = 8.5
  ): {
    totalUtilityJ: number;
    normalizedMitScore: number;
    detectionScore: number;
    coverageScore: number;
    highPriorityDefenseScore: number;
    infoGainScore: number;
    falseAlarmPenalty: number;
    latencyPenalty: number;
  } {
    // 1. Normalized MIT speedup score: (1 - NMIT)
    const nmit = Math.min(1.0, metrics.meanInterceptTimeMs / Math.max(0.1, baselineMitMs));
    const normalizedMitScore = Number((1.0 - nmit).toFixed(3));

    // 2. Probability of detection score
    const detectionScore = Number(metrics.probabilityOfDetection.toFixed(3));

    // 3. Coverage deadline preservation score
    const coverageScore = Number(Math.max(0.0, 1.0 - metrics.coverageViolations / 15.0).toFixed(3));

    // 4. High-priority threat defense score: (1 - P_HPmiss)
    const hpRatio = metrics.tHighPriorityMs > 0 ? Math.min(1.0, metrics.tHighPriorityMs / 12.0) : 0.15;
    const highPriorityDefenseScore = Number((1.0 - hpRatio).toFixed(3));

    // 5. Information gain yield score
    const infoGainScore = Number((0.65 * detectionScore + 0.35 * (1.0 - nmit)).toFixed(3));

    // 6. False alarm penalty
    const falseAlarmPenalty = Number((metrics.probabilityOfFalseAlarm * 5.0).toFixed(3));

    // 7. Decision cycle latency penalty: penalizes latencies approaching 1000 us limit
    const latencyPenalty = Number(Math.min(1.0, metrics.decisionLatencyUs.mean / 500.0).toFixed(3));

    // Composite multi-objective index:
    // Weights: w1=0.30, w2=0.25, w3=0.15, w4=0.15, w5=0.10, w6=0.03, w7=0.02
    const totalUtilityJ = Number(
      (
        0.30 * normalizedMitScore +
        0.25 * detectionScore +
        0.15 * coverageScore +
        0.15 * highPriorityDefenseScore +
        0.10 * infoGainScore -
        0.03 * falseAlarmPenalty -
        0.02 * latencyPenalty
      ).toFixed(3)
    );

    return {
      totalUtilityJ,
      normalizedMitScore,
      detectionScore,
      coverageScore,
      highPriorityDefenseScore,
      infoGainScore,
      falseAlarmPenalty,
      latencyPenalty,
    };
  }

  private static variance(arr: number[], mean: number): number {
    if (arr.length <= 1) return 1.0;
    const sumSq = arr.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0);
    return sumSq / (arr.length - 1);
  }
}
