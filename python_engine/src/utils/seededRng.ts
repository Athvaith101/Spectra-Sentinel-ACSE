/** Deterministic RNG for reproducible simulation, benchmark and training paths. */
export class SeededRNG {
  private state: number;
  constructor(seed = 42) { this.state = seed >>> 0; }
  next(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  int(min: number, max: number): number { return Math.floor(this.next() * (max - min + 1)) + min; }
  float(min: number, max: number): number { return min + this.next() * (max - min); }
  gaussian(mean = 0, std = 1): number {
    let u1 = this.next(); const u2 = this.next();
    while (u1 < 1e-15) u1 = this.next();
    return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  pick<T>(arr: T[]): T { return arr[Math.floor(this.next() * arr.length)]; }
}
