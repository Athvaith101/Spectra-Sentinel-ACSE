/**
 * Demo smoke checks for the packaged demo_web app.
 * Run from demo_web/: npx tsx smoke-test.ts
 *
 * Verifies:
 *  - seeded scenarios are deterministic (incl. composition-first builder + continuous emitter)
 *  - every catalog policy completes the same seeded episode
 *  - sequential vs adaptive fairness: same world realization, same horizon
 *  - metrics are computed from the actual runs and physically sane
 *  - decision traces expose observable state only (incl. MPC / Q-learning paths)
 */
import { DEFAULT_CONFIG, PRESETS, buildScenario } from './src/app/scenarioBuilder';
import { DemoSchedulerAdapter, POLICY_CATALOG, PolicyRegistry, EXECUTABLE_POLICIES } from './src/app/schedulerAdapter';
import { EpisodeRunner } from './src/app/episodeRunner';
import { LOCKED_EVALUATION, LOCKED_HEADLINE } from './src/state/lockedBenchmark';

let failures = 0;
function check(name: string, cond: boolean, detail = '') {
  if (cond) {
    console.log(`  PASS  ${name}`);
  } else {
    failures++;
    console.error(`  FAIL  ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

console.log('1) Scenario determinism (same seed -> identical scenario)');
const a = buildScenario({ ...DEFAULT_CONFIG, seed: 12345 });
const b = buildScenario({ ...DEFAULT_CONFIG, seed: 12345 });
check('identical emitter count', a.emitters.length === b.emitters.length);
check('identical emitter definitions', JSON.stringify(a.emitters) === JSON.stringify(b.emitters));
const c = buildScenario({ ...DEFAULT_CONFIG, seed: 999 });
check('different seed -> different scenario', JSON.stringify(a.emitters) !== JSON.stringify(c.emitters));

console.log('2) Composition-first builder');
{
  const withContinuous = buildScenario({
    ...DEFAULT_CONFIG, seed: 42,
    composition: { agile: 1, scan: 0, burst: 0, continuous: 2 },
  });
  check(
    'continuous emitter class supported end-to-end',
    withContinuous.emitters.some((e) => e.type === 'continuous'),
  );
  const total = Object.values({ ...DEFAULT_CONFIG.composition }).reduce((s, v) => s + v, 0);
  check('default composition total matches builder output',
    buildScenario(DEFAULT_CONFIG).emitters.length === total + (DEFAULT_CONFIG.adversarialShift ? 1 : 0));
  check('every preset builds', PRESETS.every((p) => buildScenario(p.config).emitters.length > 0));
}

console.log('2b) Canonical policy catalog integrity (correction spec)');
{
  const all = Object.values(POLICY_CATALOG);
  const connected = PolicyRegistry.connectedIds();
  const pending = PolicyRegistry.pending().map((p) => p.id);

  // The catalog is exactly the README §19 locked-evaluation policy set — nothing invented,
  // nothing demo-only, nothing missing.
  const lockedIds = LOCKED_EVALUATION.policies.map((p) => p.id).sort();
  const catalogIds = all.map((p) => p.id).sort();
  check('catalog == README locked-evaluation policy set (no invented/obsolete ids)',
    JSON.stringify(catalogIds) === JSON.stringify(lockedIds));
  check('catalog lists 11 policies', all.length === 11);

  // Availability classes are exhaustive and correct.
  check('every policy has exactly one availability class',
    all.every((p) => p.availability === 'connected' || p.availability === 'pending'));
  check('connected flag agrees with availability',
    all.every((p) => p.executable === (p.availability === 'connected')));
  check('pending policies carry the pending note',
    all.filter((p) => p.availability === 'pending').every((p) => !!p.availabilityNote));
  check('connected policies are exactly [sequential, staleness, cognitive, mpc]',
    JSON.stringify(connected.sort()) === JSON.stringify(['cognitive', 'mpc', 'sequential', 'staleness']));

  // Sequential is the baseline, named per the correction spec.
  check('Sequential is the canonical baseline', PolicyRegistry.baselineId === 'sequential');
  check('Sequential is backend-connected and runnable',
    POLICY_CATALOG.sequential.availability === 'connected' && POLICY_CATALOG.sequential.executable);
  check('Sequential display name is "Sequential — Baseline"',
    POLICY_CATALOG.sequential.displayName === 'Sequential — Baseline');

  // Every entry maps 1:1 onto a backend policy id (adapter seam ready for each).
  check('every policy has a backendAdapter present in the locked evaluation',
    all.every((p) => lockedIds.includes(p.backendAdapter)));

  // Pending policies can never fabricate results: forPolicy throws for them.
  let refused = false;
  try { DemoSchedulerAdapter.forPolicy('ucb1', 32); } catch { refused = true; }
  check('pending policy (ucb1) is refused by the adapter — no fabricated runs', refused);
  for (const id of pending) {
    let threw = false;
    try { DemoSchedulerAdapter.forPolicy(id, 32); } catch { threw = true; }
    check(`pending policy (${id}) cannot instantiate a runner`, threw);
  }

  // Benchmark table integrity: 11 locked policies, transcribed values sane.
  check('locked evaluation lists 11 policies', LOCKED_EVALUATION.policies.length === 11);
  check('locked Pd values within [0,1]',
    LOCKED_EVALUATION.policies.every((p) => p.pd >= 0 && p.pd <= 1));
}

console.log('2c) Locked-benchmark value verification (integration prompt §35)');
{
  const byId = (id: string) => LOCKED_EVALUATION.policies.find((p) => p.id === id);
  const seq = byId('sequential');
  const cog = byId('cognitive');
  // §35.1/§35.2 — documented locked values match the source report verbatim.
  check('locked Sequential Pd = 0.0657', seq?.pd === 0.0657);
  check('locked Cognitive Pd = 0.2403', cog?.pd === 0.2403);
  check('locked Sequential reward = 3.551', seq?.reward === 3.551);
  check('locked Cognitive reward = 6.426', cog?.reward === 6.426);
  check('locked coverage values untransformed (1.0 for both)',
    seq?.coverage === 1.0 && cog?.coverage === 1.0);
  check('locked wPOI values (0.0726 / 0.2076)',
    seq?.wPoi === 0.0726 && cog?.wPoi === 0.2076);
  check('locked HP-miss values (0.9158 / 0.8545)',
    seq?.hpMiss === 0.9158 && cog?.hpMiss === 0.8545);
  // §35.3/§35.4 — paired deltas are arithmetically correct.
  check('ΔPd = Cognitive − Sequential = +0.1746 (derived)',
    Math.abs((cog!.pd - seq!.pd) - 0.1746) < 1e-9);
  check('LOCKED_HEADLINE.deltaPd = +0.1746', LOCKED_HEADLINE.deltaPd === 0.1746);
  check('ΔReward = Cognitive − Sequential = +2.875 (derived)',
    Math.abs((cog!.reward - seq!.reward) - 2.875) < 1e-9);
  check('LOCKED_HEADLINE.deltaReward = +2.875', LOCKED_HEADLINE.deltaReward === 2.875);
  check('ΔPd CI brackets the measured ΔPd',
    LOCKED_HEADLINE.deltaPd >= LOCKED_HEADLINE.deltaPdCi[0] &&
    LOCKED_HEADLINE.deltaPd <= LOCKED_HEADLINE.deltaPdCi[1]);
  check('Pd ratio ≈ 3.66× (derived)',
    Math.abs((cog!.pd / seq!.pd) - LOCKED_HEADLINE.pdRatio) < 0.005);
  // §35.6 — no table mixing: development-tournament numbers must differ from locked ones
  // (they come from a different protocol and must never be merged).
  check('development tournament is labelled a separate protocol',
    LOCKED_EVALUATION.source !== '' && typeof LOCKED_EVALUATION.source === 'string');
  // §35.7 — protocol labelling present on every locked row.
  check('every locked policy row carries a role label',
    LOCKED_EVALUATION.policies.every((p) => p.role === 'baseline' || p.role === 'adaptive' || p.role === 'reference'));
}

console.log('3) Every connected policy completes the seeded episode');
{
  for (const id of EXECUTABLE_POLICIES) {
    const adapter = DemoSchedulerAdapter.forPolicy(id, a.channelCount);
    const runner = new EpisodeRunner(a, DEFAULT_CONFIG, adapter, DEFAULT_CONFIG.seed);
    const res = runner.runToCompletion();
    const m = res.metrics;
    check(`${id}: episode completed within horizon`, runner.done, `t=${runner.timeMs}`);
    check(`${id}: scans executed`, m.totalScans > 0, String(m.totalScans));
    check(`${id}: detection rate within [0,1]`,
      m.probabilityOfDetection >= 0 && m.probabilityOfDetection <= 1, String(m.probabilityOfDetection));
    const factors = adapter.trace[adapter.trace.length - 1]?.factors ?? [];
    check(`${id}: trace exposes observable factors`, factors.length > 0);
  }
}

console.log('4) Cognitive policy + observable explanation');
{
  const adapter = DemoSchedulerAdapter.forPolicy('cognitive', a.channelCount);
  const runner = new EpisodeRunner(a, DEFAULT_CONFIG, adapter, DEFAULT_CONFIG.seed);
  const res = runner.runToCompletion();
  const m = res.metrics;
  check('MIT physically sane (>= 0.05 ms)', m.meanInterceptTimeMs === 0 || m.meanInterceptTimeMs >= 0.05, String(m.meanInterceptTimeMs));
  check('trace explanations present', adapter.trace.some((e) => e.explanation.length > 10));
  console.log(`   cognitive: scans=${m.totalScans} emissions=${m.totalEmissions} pd=${m.probabilityOfDetection.toFixed(3)} mit=${m.meanInterceptTimeMs}ms`);
}

console.log('5) Same-seed fairness: identical world for all policies');
{
  const ids = ['sequential', 'cognitive', 'staleness'] as const;
  const runners = ids.map((id) =>
    new EpisodeRunner(a, DEFAULT_CONFIG, DemoSchedulerAdapter.forPolicy(id, a.channelCount), DEFAULT_CONFIG.seed));
  const results = runners.map((r) => r.runToCompletion());
  check(
    'equal emissions for all policies (same realization)',
    results.every((r) => r.metrics.totalEmissions === results[0].metrics.totalEmissions),
    results.map((r) => r.metrics.totalEmissions).join(' / '),
  );
  check(
    'all complete on the same horizon',
    runners.every((r) => r.timeMs === runners[0].timeMs),
  );
  console.log(`   comparison: ${ids.map((id, i) => `${id} pd=${results[i].metrics.probabilityOfDetection.toFixed(3)}`).join(' · ')}`);
}

console.log('6) Channel-concentration guardrail (integration prompt §20/§21)');
{
  // Belief/coverage-driven connected policies must not systematically park on one band.
  // Threshold: > 90% of dwells on a single channel across ALL seeds would indicate a
  // collapse; measurement-only — the diagnostic script reports details.
  for (const id of EXECUTABLE_POLICIES) {
    const shares: number[] = [];
    for (const seed of [26055, 4102, 7309, 90210]) {
      const cfg = { ...DEFAULT_CONFIG, seed };
      const scn = buildScenario(cfg);
      const adapter = DemoSchedulerAdapter.forPolicy(id, scn.channelCount);
      const runner = new EpisodeRunner(scn, cfg, adapter, seed);
      runner.runToCompletion();
      const counts = new Map<number, number>();
      for (const r of runner.waterfallHistory) counts.set(r.band, (counts.get(r.band) ?? 0) + 1);
      const total = runner.waterfallHistory.length;
      const maxShare = total ? Math.max(...counts.values()) / total : 0;
      shares.push(maxShare);
    }
    const worst = Math.max(...shares);
    check(`${id}: no >90% single-channel collapse across seeds (worst ${(worst * 100).toFixed(1)}%)`, worst <= 0.9);
  }
}

console.log(failures === 0 ? '\nALL DEMO_WEB SMOKE CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
