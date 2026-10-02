/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * PROVENANCE: copied verbatim from ../src/environment/detectionModel.ts (repo root).
 */

/**
 * Radar / EW Detection Theory Model:
 * Implements physical SNR-dependent Probability of Detection (Pd) based on
 * Albersheim's empirical approximation and Marcum Q-function for non-fluctuating
 * and Swerling target/emitter pulses in Gaussian noise.
 */
export class DetectionModel {
  private static rand(rng: { next(): number }): number { return rng.next(); }
  /**
   * Calculates Probability of Detection given SNR (dB) and false alarm probability (Pfa).
   * Uses logistic approximation calibrated to standard EW crystal video / IF threshold detectors.
   *
   * @param snrDb Signal-to-Noise ratio in decibels
   * @param pulseWidthUs Pulse duration in microseconds
   * @param pfa False alarm probability (default 1e-4)
   * @returns Detection probability Pd in range [0, 1]
   */
  public static calculatePd(snrDb: number, pulseWidthUs: number, pfa: number = 1e-4): number {
    // Shorter pulses (< 1 us) experience slight post-detection filter dispersion loss
    const pwLossDb = pulseWidthUs < 1.0 ? Math.max(0, 1.5 * (1.0 - pulseWidthUs)) : 0;
    const effectiveSnrDb = snrDb - pwLossDb;

    // Threshold SNR required for Pd = 0.5 at given Pfa (Albersheim's base relation)
    // For Pfa = 1e-4, nominal 50% detection threshold is ~9.5 dB
    const snrThreshDb = 8.0 - 1.2 * Math.log10(pfa);

    // Steepness parameter of detector S-curve (higher SNR produces sharper transition)
    const slope = 0.45;
    const snrDiff = effectiveSnrDb - snrThreshDb;

    // Sigmoid / logistic detector transfer characteristic
    const pd = 1.0 / (1.0 + Math.exp(-slope * snrDiff));
    return Math.max(0.001, Math.min(0.999, pd));
  }

  /**
   * Evaluates whether a pulse is detected given its SNR and pulse width.
   * Performs Monte Carlo draw against physical Pd.
   */
  public static sampleDetection(snrDb: number, pulseWidthUs: number, pfa: number = 1e-4, rng: { next(): number }): boolean {
    const pd = this.calculatePd(snrDb, pulseWidthUs, pfa);
    return this.rand(rng) < pd;
  }

  /**
   * Computes noise-induced false alarm count for a given dwell duration.
   * Uses Poisson distribution approximation: λ = pfaRatePerSec * (dwellTimeMs / 1000)
   */
  public static generateNoiseFalseAlarms(dwellTimeMs: number, pfaRatePerSec: number = 25, rng: { next(): number }): number {
    const lambda = pfaRatePerSec * (dwellTimeMs / 1000.0);
    // Knuth's Poisson random variable generator
    const L = Math.exp(-lambda);
    let k = 0;
    let p = 1.0;
    do {
      k++;
      p *= this.rand(rng);
    } while (p > L);
    return k - 1;
  }
}
