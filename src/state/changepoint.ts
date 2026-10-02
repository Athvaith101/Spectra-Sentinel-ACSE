/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Page-Hinkley Cumulative Sum Change Point Detector:
 * Sequentially monitors surprise and prediction residuals to detect abrupt adversarial
 * hopping sequence changes or countermeasure switches.
 */
export class PageHinkleyDetector {
  private thresholdLambda: number;
  private driftDelta: number;
  private cumSum: number = 0;
  private minCumSum: number = 0;
  private sampleCount: number = 0;
  private runningMean: number = 0.5;
  public lastDetected: boolean = false;

  constructor(thresholdLambda: number = 14.0, driftDelta: number = 0.05) {
    this.thresholdLambda = thresholdLambda;
    this.driftDelta = driftDelta;
  }

  public update(observedProb: number): boolean {
    this.lastDetected = false;
    this.sampleCount++;
    // Surprise metric: s = -log2(P(c_t | c_{t-1}))
    const surprise = -Math.log2(Math.max(0.01, Math.min(1.0, observedProb)));

    // Welford running mean update
    this.runningMean += (surprise - this.runningMean) / this.sampleCount;

    // Cumulative sum update with tolerance drift
    this.cumSum += surprise - this.runningMean - this.driftDelta;

    if (this.cumSum < this.minCumSum) {
      this.minCumSum = this.cumSum;
    }

    // Change point detected if cumulative deviation exceeds threshold lambda
    if (this.cumSum - this.minCumSum > this.thresholdLambda) {
      this.lastDetected = true;
      this.reset();
      this.lastDetected = true;
      return true;
    }

    return false;
  }

  public reset(): void {
    this.lastDetected = false;
    this.cumSum = 0;
    this.minCumSum = 0;
    this.sampleCount = 0;
    this.runningMean = 0.5;
  }
}
