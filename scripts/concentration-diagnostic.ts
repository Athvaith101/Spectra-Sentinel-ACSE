/**
 * Channel-concentration diagnostic (integration prompt §21/§22).
 *
 * Run from demo_web/:  npx tsx scripts/concentration-diagnostic.ts
 *
 * For every UI-exposed policy and multiple seeds, runs a REAL episode with the packaged
 * engine and records:
 *   - unique channels used
 *   - maximum single-channel share
 *   - top-3 channel share
 *   - channel coverage (share of channels visited at least once)
 *   - number of channel switches
 *   - detections by channel
 *
 * Nothing here changes the engine, the waterfall or any metric — measurement only
 * (integration prompt §22: never fix results by making them look better).
 */
import { buildScenario, ScenarioConfig, DEFAULT_CONFIG } from '../src/app/scenarioBuilder';
import { DemoSchedulerAdapter, EXECUTABLE_POLICIES } from '../src/app/schedulerAdapter';
import { EpisodeRunner } from '../src/app/episodeRunner';

const SEEDS = [26055, 4102, 7309, 90210, 1111, 5150];

const CONFIGS: { name: string; cfg: ScenarioConfig }[] = [
  { name: 'default (balanced)', cfg: { ...DEFAULT_CONFIG } },
  {
    name: 'agile arena',
    cfg: {
      ...DEFAULT_CONFIG, seed: 4102,
      composition: { agile: 5, scan: 1, burst: 0, continuous: 0 },
      activity: 0.7, agility: 0.8, difficulty: 0.3,
    },
  },
  {
    name: 'sparse LPI',
    cfg: {
      ...DEFAULT_CONFIG, seed: 7309,
      composition: { agile: 0, scan: 1, burst: 2, continuous: 0 },
      activity: 0.25, agility: 0.3, difficulty: 0.55,
    },
  },
];

interface ChannelStats {
  unique: number;
  maxShare: number;
  top3Share: number;
  coverage: number;
  switches: number;
  detByChannel: Record<number, number>;
  scans: number;
}

function statsFor(bands: number[], detections: Map<number, number>, channelCount: number): ChannelStats {
  const counts = new Map<number, number>();
  let switches = 0;
  let prev = -1;
  for (const b of bands) {
    counts.set(b, (counts.get(b) ?? 0) + 1);
    if (b !== prev) switches++;
    prev = b;
  }
  const total = bands.length;
  const sorted = [...counts.values()].sort((a, b) => b - a);
  const unique = counts.size;
  return {
    unique,
    maxShare: total ? (sorted[0] ?? 0) / total : 0,
    top3Share: total ? sorted.slice(0, 3).reduce((s, v) => s + v, 0) / total : 0,
    coverage: unique / channelCount,
    switches,
    detByChannel: Object.fromEntries([...detections.entries()].sort((a, b) => a[0] - b[0])),
    scans: total,
  };
}

console.log('CHANNEL-CONCENTRATION DIAGNOSTIC — real episodes, packaged engine');
console.log(`connected (runnable) policies: ${EXECUTABLE_POLICIES.join(', ')}`);
console.log(`seeds per scenario config: ${SEEDS.length}, configs: ${CONFIGS.length}\n`);

let worst = { policy: '', cfg: '', seed: 0, maxShare: 0 };

for (const { name, cfg: base } of CONFIGS) {
  for (const seed of SEEDS) {
    const cfg: ScenarioConfig = { ...base, seed };
    const scenario = buildScenario(cfg);
    console.log(`── ${name} · seed ${seed} · ${scenario.emitters.length} emitters ──`);
    for (const id of EXECUTABLE_POLICIES) {
      const adapter = DemoSchedulerAdapter.forPolicy(id, scenario.channelCount);
      const runner = new EpisodeRunner(scenario, cfg, adapter, seed);
      runner.runToCompletion();

      const bands = runner.waterfallHistory.map((r) => r.band);
      // Detections by channel: from the waterfall's HIT rows (truth-joined hits of this run).
      const detections = new Map<number, number>();
      for (const r of runner.waterfallHistory) {
        if (r.hit) detections.set(r.band, (detections.get(r.band) ?? 0) + 1);
      }
      const st = statsFor(bands, detections, scenario.channelCount);
      const m = runner.adapterRef.scheduler.metrics;

      console.log(
        `  ${id.padEnd(11)} scans=${String(st.scans).padStart(4)}  used=${String(st.unique).padStart(2)}/32`
        + `  maxShare=${(st.maxShare * 100).toFixed(1).padStart(5)}%  top3=${(st.top3Share * 100).toFixed(1).padStart(5)}%`
        + `  coverage=${(st.coverage * 100).toFixed(0).padStart(3)}%  switches=${String(st.switches).padStart(4)}`
        + `  detCh=${JSON.stringify(st.detByChannel)}  pd=${(m.probabilityOfDetection * 100).toFixed(1)}%`,
      );
      if (st.maxShare > worst.maxShare) worst = { policy: id, cfg: name, seed, maxShare: st.maxShare };
    }
    console.log('');
  }
}

console.log('SUMMARY');
console.log(`  worst single-channel concentration: ${worst.policy} @ ${worst.cfg} seed ${worst.seed}`
  + ` — ${(worst.maxShare * 100).toFixed(1)}% of dwells on one channel`);
console.log('  Interpretation: check whether high concentration is systematic (same policy every seed)'
  + ' or seed-specific. Connected policies are belief/coverage-driven and must not collapse to one band;'
  + ' if a concentrated behaviour appears after backend integration, it must be fixed at the backend/policy'
  + ' layer, never cosmetically redistributed in the UI (correction spec: seed/one-channel issue).');
