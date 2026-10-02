/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface SequenceHypothesis {
  id: string;
  pattern: number[];
  matchScore: number;
  lastMatchedIndex: number;
}

/**
 * Multi-Hypothesis Tracking (MHT) Pattern Predictor:
 * Generates continuous probability distributions P(f_{t+1} = c) across all spectrum channels
 * by maintaining multiple concurrent hypotheses over pseudo-random radar hopping sequences.
 */
export class MultiHypothesisPredictor {
  private hypotheses: SequenceHypothesis[] = [];
  private hopHistory: number[] = [];
  private channelCount: number;

  constructor(channelCount: number = 32) {
    this.channelCount = channelCount;
  }

  public recordHop(channel: number): void {
    this.hopHistory.push(channel);
    if (this.hopHistory.length > 64) {
      this.hopHistory.shift();
    }
    this.updateHypotheses();
  }

  private updateHypotheses(): void {
    if (this.hopHistory.length < 6) return;

    // Evaluate sequence period lengths from 3 to 16
    for (let period = 3; period <= 16; period++) {
      if (this.hopHistory.length < period * 2) continue;

      const len = this.hopHistory.length;
      let matches = 0;
      let comparisons = 0;

      for (let i = 0; i < period; i++) {
        const valRecent = this.hopHistory[len - 1 - i];
        const valPast = this.hopHistory[len - 1 - i - period];
        if (valRecent === valPast) {
          matches++;
        }
        comparisons++;
      }

      const score = comparisons > 0 ? matches / comparisons : 0;
      if (score >= 0.5) {
        const candidatePattern = this.hopHistory.slice(len - period);
        const hypId = `hyp-p${period}`;
        const existing = this.hypotheses.find((h) => h.id === hypId);
        if (existing) {
          existing.matchScore = 0.75 * existing.matchScore + 0.25 * score;
          existing.pattern = candidatePattern;
        } else {
          this.hypotheses.push({
            id: hypId,
            pattern: candidatePattern,
            matchScore: score,
            lastMatchedIndex: 0,
          });
        }
      }
    }
  }

  /**
   * Produces full probability distribution vector P(f_{t+1} = c) across all C channels.
   * Eliminates the binary confidence threshold cliff.
   */
  public getHopDistribution(): Float32Array {
    const dist = new Float32Array(this.channelCount);
    if (this.hopHistory.length === 0 || this.hypotheses.length === 0) {
      const uniform = 1.0 / this.channelCount;
      dist.fill(uniform);
      return dist;
    }

    const lastCh = this.hopHistory[this.hopHistory.length - 1];
    let totalWeight = 0.05; // Base exploration floor

    // Uniform baseline
    for (let c = 0; c < this.channelCount; c++) {
      dist[c] = 0.05 / this.channelCount;
    }

    // Accumulate predictions from all active hypotheses weighted by matchScore
    for (const hyp of this.hypotheses) {
      if (hyp.pattern.length === 0 || hyp.matchScore <= 0.1) continue;

      const idxInPattern = hyp.pattern.indexOf(lastCh);
      if (idxInPattern >= 0) {
        const nextIdx = (idxInPattern + 1) % hyp.pattern.length;
        const predictedCh = hyp.pattern[nextIdx];
        if (predictedCh >= 0 && predictedCh < this.channelCount) {
          // Weight scales exponentially with pattern confidence
          const w = Math.pow(hyp.matchScore, 2.0);
          dist[predictedCh] += w;
          totalWeight += w;
        }
      }
    }

    // Normalize to proper probability simplex Σ P(c) = 1
    if (totalWeight > 0) {
      for (let c = 0; c < this.channelCount; c++) {
        dist[c] /= totalWeight;
      }
    }

    return dist;
  }

  /**
   * Blends Markov transition row with MHT distribution
   */
  public blendWithMarkov(markovRow: Float32Array): Float32Array {
    const mhtDist = this.getHopDistribution();
    const blended = new Float32Array(this.channelCount);

    let maxMhtConfidence = 0;
    for (const h of this.hypotheses) {
      if (h.matchScore > maxMhtConfidence) maxMhtConfidence = h.matchScore;
    }

    // Dynamic blending factor: higher MHT confidence gives MHT greater weight
    const mhtWeight = Math.min(0.85, Math.max(0.15, maxMhtConfidence));
    const markovWeight = 1.0 - mhtWeight;

    for (let c = 0; c < this.channelCount; c++) {
      blended[c] = markovWeight * markovRow[c] + mhtWeight * mhtDist[c];
    }

    return blended;
  }

  public reset(): void {
    this.hypotheses = [];
    this.hopHistory = [];
  }
}
