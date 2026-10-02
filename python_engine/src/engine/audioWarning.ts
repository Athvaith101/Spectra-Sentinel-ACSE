/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

class AudioWarningEngine {
  private ctx: AudioContext | null = null;
  public isMuted: boolean = true; // Default muted for smooth UX

  public toggleMute(): boolean {
    this.isMuted = !this.isMuted;
    if (!this.isMuted && !this.ctx) {
      try {
        const AudioCtxClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        this.ctx = new AudioCtxClass();
      } catch {
        this.isMuted = true;
      }
    }
    return this.isMuted;
  }

  public playInterceptBeep(priority: number = 3, freqMHz: number = 3000): void {
    if (this.isMuted) return;
    try {
      if (!this.ctx) {
        const AudioCtxClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        this.ctx = new AudioCtxClass();
      }
      if (this.ctx.state === 'suspended') {
        this.ctx.resume();
      }

      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      // Pitch based on radar frequency & priority
      const pitch = 400 + (freqMHz % 2000) * 0.5 + priority * 120;
      osc.type = priority >= 5 ? 'sawtooth' : 'sine';
      osc.frequency.setValueAtTime(pitch, this.ctx.currentTime);

      gain.gain.setValueAtTime(0.04, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + 0.05);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start();
      osc.stop(this.ctx.currentTime + 0.05);
    } catch {
      // Audio autoplay restrictions or headless environment
    }
  }
}

export const audioWarning = new AudioWarningEngine();
