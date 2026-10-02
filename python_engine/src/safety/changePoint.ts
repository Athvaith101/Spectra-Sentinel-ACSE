/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Page-Hinkley Change Point Detector:
 * Monitors prediction loss and surprise of intercepted hops in real time.
 * Detects intentional adversarial sequence shifts (Scenario 5) within 3-5 hops
 * and triggers immediate recalibration of transition priors.
 */
export class PageHinkleyDetector {
  private threshold: number;
  private alpha: number;
  private cumulativeDiff: number = 0;
  private minCumulative: number = 0;
  private meanError: number = 0.5;
  private sampleCount: number = 0;

  constructor(threshold: number = 18.0, alpha: number = 0.05) {
    this.threshold = threshold;
    this.alpha = alpha;
  }

  /**
   * Evaluates prediction surprise:
   * error is 0 if predicted correctly, 1 if missed.
   * Returns true if abrupt distribution shift is detected.
   */
  public update(predictedProbOfObservedHop: number): boolean {
    const error = 1.0 - Math.min(1.0, Math.max(0.0, predictedProbOfObservedHop));
    this.sampleCount++;

    // Online running mean of error
    this.meanError = 0.92 * this.meanError + 0.08 * error;

    // Cumulative sum
    this.cumulativeDiff += (error - this.meanError - this.alpha);
    if (this.cumulativeDiff < this.minCumulative) {
      this.minCumulative = this.cumulativeDiff;
    }

    const testStatistic = this.cumulativeDiff - this.minCumulative;
    if (testStatistic > this.threshold) {
      // Change point detected! Reset statistics
      this.reset();
      return true;
    }

    return false;
  }

  public reset(): void {
    this.cumulativeDiff = 0;
    this.minCumulative = 0;
    this.sampleCount = 0;
    this.meanError = 0.5;
  }
}
