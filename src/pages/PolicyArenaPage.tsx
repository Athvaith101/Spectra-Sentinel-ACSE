/**
 * Policy Arena (spec §6–§8) — dedicated comparison environment.
 *
 * SCENARIO SOURCE (spec §7): Build Scenario (the same canonical ScenarioBuilderPanel as
 * the Scenario Lab: presets, Custom, emitter composition, sliders, editable seed) or
 * Upload Dataset (the shared structural-validation panel — honest: validation is not
 * model evaluation, no fabricated results).
 *
 * FAIRNESS (spec §8/§20): in Build mode, 2–4 CONNECTED policies run headlessly on the
 * IDENTICAL seeded scenario — one scenario object, one seed, only the scheduler differs.
 * Results are labelled LIVE SCENARIO RESULT: a behaviour demonstration, never a claim.
 *
 * CORRECTION: the policy grid shows the full canonical catalog. Pending (backend
 * integration pending) policies are visible and labelled but CANNOT be selected into a
 * live comparison — only runnable policies participate, and nothing can fabricate results.
 */
import React, { useRef, useState } from 'react';
import { ScenarioConfig, buildScenario } from '../app/scenarioBuilder';
import { POLICY_CATALOG, PolicyId, PolicyRegistry, DemoSchedulerAdapter } from '../app/schedulerAdapter';
import { EpisodeRunner, EpisodeResult } from '../app/episodeRunner';
import { ScenarioBuilderPanel } from '../components/ScenarioBuilder/ScenarioBuilderPanel';
import { DatasetValidationPanel } from '../components/DatasetUpload/DatasetValidationPanel';
import { WaterfallView } from '../components/Waterfall/WaterfallView';
import { DEFAULT_CONFIG, newSeed, PRESETS } from '../app/scenarioBuilder';
import { CUSTOM_PRESET } from '../components/ScenarioBuilder/ScenarioBuilderPanel';

const ARENA_COLORS: Record<PolicyId, string> = {
  sequential: '#2563a8',
  staleness: '#9a6700',
  information: '#0f766e',
  ucb1: '#7c5cbf',
  linucb: '#b45309',
  thompson: '#0e7490',
  markov: '#6d28d9',
  cognitive: '#0a7d57',
  mpc: '#0f766e',
  stare_cognitive: '#4d7c0f',
  portfolio: '#57534e',
};

const DEFAULT_SELECTION: PolicyId[] = ['sequential', 'cognitive'];

type ScenarioSource = 'build' | 'upload';

interface RunState {
  running: boolean;
  progress: number;
  results: Map<PolicyId, EpisodeResult>;
}

function deltaPct(adp: number, base: number, lowerIsBetter: boolean): { text: string; cls: string } {
  if (Math.abs(base) < 1e-9) return { text: '—', cls: '' };
  const diff = adp - base;
  const pct = (diff / Math.abs(base)) * 100;
  if (Math.abs(pct) < 0.05) return { text: '±0%', cls: '' };
  const better = lowerIsBetter ? diff < 0 : diff > 0;
  return { text: `${pct > 0 ? '+' : ''}${pct.toFixed(0)}%`, cls: better ? 'best' : 'text-bad' };
}

export const PolicyArenaPage: React.FC = () => {
  const [source, setSource] = useState<ScenarioSource>('build');

  // Build-mode scenario state — mirrors the Scenario Lab's useLabSession config handling.
  const [cfg, setCfg] = useState<ScenarioConfig>({ ...DEFAULT_CONFIG });
  const [activePreset, setActivePreset] = useState<string>('balanced');
  const applyConfig = (next: ScenarioConfig, presetId: string | null) => {
    setCfg(next);
    setActivePreset(presetId ?? CUSTOM_PRESET);
  };
  const editConfig = (next: ScenarioConfig) => applyConfig(next, null);
  const applyPreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id);
    if (p) applyConfig({ ...p.config }, id);
  };

  const [selected, setSelected] = useState<PolicyId[]>(DEFAULT_SELECTION);
  const [run, setRun] = useState<RunState>({ running: false, progress: 0, results: new Map() });
  const cancelRef = useRef(false);

  const scenario = React.useMemo(() => buildScenario(cfg), [cfg]);
  // Full canonical catalog — the same identities as Scenario Lab (single source of truth).
  // Selection of pending policies into live runs is prevented in toggle() + runComparison().
  const policies = PolicyRegistry.selectable();
  const connectedCount = PolicyRegistry.connectedIds().length;

  const toggle = (id: PolicyId) => {
    // Pending policies must never enter a live comparison (no fabricated results).
    if (!POLICY_CATALOG[id].executable) return;
    setSelected((prev) => {
      if (prev.includes(id)) return prev.length > 1 ? prev.filter((x) => x !== id) : prev;
      if (prev.length >= 4) return prev;
      return [...prev, id];
    });
  };

  const runComparison = async () => {
    if (selected.length < 2 || run.running) return;
    cancelRef.current = false;
    setRun({ running: true, progress: 0, results: new Map() });

    // FAIRNESS: build the scenario ONCE, then one runner per policy on that same object
    // and the same seed — identical realization, identical information constraints.
    // Only connected policies reach this point (enforced in toggle + the guard below).
    const runnable = selected.filter((id) => POLICY_CATALOG[id].executable);
    if (runnable.length < 2) return;
    const scenarioSnapshot = buildScenario(cfg);
    const runners = runnable.map((id) => {
      const adapter = DemoSchedulerAdapter.forPolicy(id, scenarioSnapshot.channelCount);
      return new EpisodeRunner(scenarioSnapshot, cfg, adapter, cfg.seed);
    });

    // Yield first so the UI paints the RUNNING state before compute.
    await new Promise((r) => setTimeout(r, 30));

    while (runners.some((r) => !r.done)) {
      for (const r of runners) if (!r.done) r.step(10);
      if (cancelRef.current) return;
      await new Promise((r) => setTimeout(r, 0));
      const doneMs = Math.min(...runners.map((r) => r.timeMs));
      setRun((prev) => ({ ...prev, progress: doneMs / scenarioSnapshot.durationMs }));
    }

    const results = new Map<PolicyId, EpisodeResult>();
    runnable.forEach((id, i) => {
      const r = runners[i];
      results.set(id, {
        policyId: id,
        metrics: { ...r.adapterRef.scheduler.metrics },
        waterfall: r.waterfallHistory,
        events: [],
        computeTimeMs: 0,
      });
    });
    setRun({ running: false, progress: 1, results });
  };

  const ordered = selected
    .map((id) => ({ id, res: run.results.get(id) }))
    .filter((x) => x.res) as { id: PolicyId; res: EpisodeResult }[];

  return (
    <div className="page">
      <div className="spread" style={{ marginBottom: 12 }}>
        <div>
          <h1 className="page-title">Policy Arena</h1>
          <div className="section-note">
            Run 2–4 schedulers on the same seeded scenario and compare actual behaviour.
          </div>
        </div>
        <a className="btn btn-quiet" href="#/lab">Open Scenario Lab →</a>
      </div>

      {/* SCENARIO SOURCE — Build vs Upload (spec §7) */}
      <div className="panel panel-pad" style={{ marginBottom: 12 }}>
        <div className="spread" style={{ flexWrap: 'wrap', gap: 10 }}>
          <div className="row" style={{ gap: 10 }}>
            <span className="panel-title">Scenario source</span>
            <div className="theme-toggle" role="group" aria-label="Scenario source">
              <button className={source === 'build' ? 'on' : ''} onClick={() => setSource('build')}>Build scenario</button>
              <button className={source === 'upload' ? 'on' : ''} onClick={() => setSource('upload')}>Upload dataset</button>
            </div>
          </div>
          {source === 'upload' && (
            <span className="tiny" style={{ color: 'var(--warn)' }}>
              Upload mode validates dataset structure only — it does not run model evaluation.
            </span>
          )}
        </div>
      </div>

      {source === 'upload' ? (
        <div style={{ maxWidth: 620 }}>
          <DatasetValidationPanel />
        </div>
      ) : (
        <>
          {/* Build mode: canonical scenario builder + policy selection */}
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 420px) 1fr', gap: 12, alignItems: 'start', marginBottom: 12 }}>
            <ScenarioBuilderPanel
              cfg={cfg}
              onChange={editConfig}
              onPreset={applyPreset}
              onCustom={() => setActivePreset(CUSTOM_PRESET)}
              onSeedChange={(seed) => applyConfig({ ...cfg, seed }, CUSTOM_PRESET)}
              onRandomizeSeed={() => applyConfig({ ...cfg, seed: newSeed() }, CUSTOM_PRESET)}
              activePreset={activePreset}
            />

            <div className="panel panel-pad">
              <div className="panel-title" style={{ marginBottom: 8 }}>2 · Policies</div>
              <div className="panel-sub" style={{ marginBottom: 8 }}>
                Select 2–4 connected policies. Sequential — Baseline anchors every comparison;
                pending policies are shown for completeness but cannot run.
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {policies.map((p) => {
                  const on = selected.includes(p.id);
                  const order = on ? selected.indexOf(p.id) + 1 : null;
                  const pending = !p.executable;
                  return (
                    <button
                      key={p.id}
                      onClick={() => toggle(p.id)}
                      title={pending ? `${p.displayName} — ${p.availabilityNote ?? ''}` : p.description}
                      className="chip"
                      disabled={pending}
                      style={{
                        cursor: pending ? 'not-allowed' : 'pointer', padding: '5px 10px', fontSize: 11,
                        background: on ? 'var(--accent-soft)' : 'var(--surface)',
                        borderColor: on ? 'var(--accent)' : pending ? 'var(--border)' : 'var(--border-strong)',
                        color: pending ? 'var(--faint)' : on ? 'var(--accent)' : 'var(--muted)',
                        opacity: pending ? 0.65 : 1,
                        textDecoration: pending ? 'line-through' : undefined,
                      }}
                    >
                      {on ? `${order}. ` : ''}{p.shortName}{pending ? ' · pending' : ''}
                    </button>
                  );
                })}
              </div>
              <p className="tiny" style={{ color: 'var(--faint)', margin: '8px 0 0' }}>
                {connectedCount} of {policies.length} canonical policies are backend-connected and runnable.
                Entries marked “Backend integration pending” exist in the project backend (see the Benchmark page for
                their documented locked-evaluation results) but cannot execute in the browser yet — they never produce
                fabricated results here.
              </p>

              <hr className="hr" />

              <div className="panel-title" style={{ marginBottom: 8 }}>3 · Run comparison</div>
              <div className="row" style={{ gap: 10, flexWrap: 'wrap' }}>
                <button className="btn btn-primary" onClick={() => void runComparison()}
                  disabled={run.running || selected.length < 2}>
                  {run.running ? 'Running…' : 'Run comparison'}
                </button>
                {run.running && (
                  <div style={{ width: 180, height: 6, background: 'var(--surface-2)', border: '1px solid var(--border)' }}>
                    <div style={{ width: `${Math.round(run.progress * 100)}%`, height: '100%', background: 'var(--accent)' }} />
                  </div>
                )}
                <span className="tiny" style={{ color: 'var(--faint)' }}>
                  identical seed {cfg.seed} for every policy — only the scheduler differs
                </span>
              </div>
            </div>
          </div>

          {/* Waterfalls */}
          {ordered.length > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(ordered.length, 2)}, 1fr)`, gap: 12 }}>
              {ordered.map(({ id, res }) => (
                <WaterfallView
                  key={id}
                  rows={res.waterfall}
                  accent={ARENA_COLORS[id]}
                  mode="full"
                  height={230}
                  title={`${POLICY_CATALOG[id].displayName} — full episode`}
                />
              ))}
            </div>
          )}

          {/* Structured results (integration prompt §21): Baseline → Policy result →
              Relative to Sequential → one-line interpretation. Plus the §22 fairness note. */}
          {ordered.length > 0 && (
            <div className="panel panel-pad" style={{ marginTop: 12 }}>
              <div className="spread" style={{ marginBottom: 10 }}>
                <span className="panel-title">Results</span>
                <span className="chip" style={{ fontSize: 9 }}>LIVE SCENARIO RESULT</span>
              </div>

              {/* §22 — fairness note, kept visually close to the numbers it qualifies */}
              <p className="tiny" style={{ color: 'var(--muted)', margin: '0 0 12px', fontFamily: 'var(--mono)' }}>
                All selected policies are evaluated on the same seeded scenario realization
                (seed {cfg.seed} · identical emitters · identical horizon · identical sensing constraints).
              </p>

              {/* §21.1 — Baseline block: the first selected policy. If Sequential is among
                  the selection it is promoted to the baseline slot, matching the benchmark's
                  convention and making every Δ row read "vs Sequential". */}
              {(() => {
                const seqIn = ordered.find((x) => x.id === 'sequential');
                const baseline = seqIn ?? ordered[0];
                const others = ordered.filter((x) => x.id !== baseline.id);
                const fmtDelta = (d: number, digits: number) => `${d >= 0 ? '+' : ''}${d.toFixed(digits)}`;
                return (
                  <div style={{ display: 'grid', gap: 12 }}>
                    {/* Baseline */}
                    <div>
                      <div className="panel-title" style={{ marginBottom: 6 }}>Baseline</div>
                      <div className="res-grid">
                        <div className="res-cell">
                          <div className="rl">Policy</div>
                          <div className="rv" style={{ fontSize: 16 }}>{POLICY_CATALOG[baseline.id].displayName}</div>
                          <div className="rh">first selected policy is the comparison baseline</div>
                        </div>
                        <div className="res-cell">
                          <div className="rl">Pd</div>
                          <div className="rv">{(baseline.res.metrics.probabilityOfDetection * 100).toFixed(1)}%</div>
                          <div className="rh">{baseline.res.metrics.interceptedCount} of {baseline.res.metrics.totalEmissions} emissions</div>
                        </div>
                        <div className="res-cell">
                          <div className="rl">Avg intercept (ms)</div>
                          <div className="rv">{baseline.res.metrics.meanInterceptTimeMs.toFixed(1)}</div>
                          <div className="rh">p95 {baseline.res.metrics.p95InterceptTimeMs.toFixed(1)} ms</div>
                        </div>
                        <div className="res-cell">
                          <div className="rl">Max staleness (ms)</div>
                          <div className="rv">{baseline.res.metrics.maxStalenessMs.toFixed(0)}</div>
                          <div className="rh">scans {baseline.res.metrics.totalScans}</div>
                        </div>
                      </div>
                    </div>

                    {/* Policy results */}
                    <div>
                      <div className="panel-title" style={{ marginBottom: 6 }}>Policy result</div>
                      <div style={{ display: 'grid', gap: 8 }}>
                        {others.map(({ id, res }) => (
                          <div key={id} className="res-grid" style={{ borderTop: '1px solid var(--border)', paddingTop: 8 }}>
                            <div className="res-cell">
                              <div className="rl">Policy</div>
                              <div className="rv" style={{ fontSize: 16 }}>
                                <span className="dot" style={{ background: ARENA_COLORS[id], marginRight: 6 }} />
                                {POLICY_CATALOG[id].displayName}
                              </div>
                              <div className="rh">same scenario realization</div>
                            </div>
                            <div className="res-cell">
                              <div className="rl">Pd</div>
                              <div className="rv">{(res.metrics.probabilityOfDetection * 100).toFixed(1)}%</div>
                              <div className="rh">{res.metrics.interceptedCount} of {res.metrics.totalEmissions} emissions</div>
                            </div>
                            <div className="res-cell">
                              <div className="rl">Avg intercept (ms)</div>
                              <div className="rv">{res.metrics.meanInterceptTimeMs.toFixed(1)}</div>
                              <div className="rh">p95 {res.metrics.p95InterceptTimeMs.toFixed(1)} ms</div>
                            </div>
                            <div className="res-cell">
                              <div className="rl">Max staleness (ms)</div>
                              <div className="rv">{res.metrics.maxStalenessMs.toFixed(0)}</div>
                              <div className="rh">scans {res.metrics.totalScans}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Relative to baseline */}
                    <div>
                      <div className="panel-title" style={{ marginBottom: 6 }}>Relative to {POLICY_CATALOG[baseline.id].shortName}</div>
                      <table className="tbl">
                        <thead>
                          <tr>
                            <th>Policy</th>
                            <th className="num">Δ Detections</th>
                            <th className="num">Δ Pd</th>
                            <th className="num">Δ Avg intercept (ms)</th>
                            <th className="num">Δ P95 (ms)</th>
                            <th className="num">Δ Max staleness (ms)</th>
                            <th className="num">Δ Scans</th>
                          </tr>
                        </thead>
                        <tbody>
                          {others.map(({ id, res }) => (
                            <tr key={id}>
                              <td>
                                <span className="dot" style={{ background: ARENA_COLORS[id], marginRight: 6 }} />
                                {POLICY_CATALOG[id].displayName}
                              </td>
                              <td className={`num ${deltaPct(res.metrics.interceptedCount, baseline.res.metrics.interceptedCount, false).cls}`}>{deltaPct(res.metrics.interceptedCount, baseline.res.metrics.interceptedCount, false).text}</td>
                              <td className={`num ${deltaPct(res.metrics.probabilityOfDetection, baseline.res.metrics.probabilityOfDetection, false).cls}`}>{deltaPct(res.metrics.probabilityOfDetection, baseline.res.metrics.probabilityOfDetection, false).text}</td>
                              <td className={`num ${deltaPct(res.metrics.meanInterceptTimeMs, baseline.res.metrics.meanInterceptTimeMs, true).cls}`}>{deltaPct(res.metrics.meanInterceptTimeMs, baseline.res.metrics.meanInterceptTimeMs, true).text}</td>
                              <td className={`num ${deltaPct(res.metrics.p95InterceptTimeMs, baseline.res.metrics.p95InterceptTimeMs, true).cls}`}>{deltaPct(res.metrics.p95InterceptTimeMs, baseline.res.metrics.p95InterceptTimeMs, true).text}</td>
                              <td className={`num ${deltaPct(res.metrics.maxStalenessMs, baseline.res.metrics.maxStalenessMs, true).cls}`}>{deltaPct(res.metrics.maxStalenessMs, baseline.res.metrics.maxStalenessMs, true).text}</td>
                              <td className={`num ${deltaPct(res.metrics.totalScans, baseline.res.metrics.totalScans, true).cls}`}>{deltaPct(res.metrics.totalScans, baseline.res.metrics.totalScans, true).text}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {/* One-line interpretation per policy — magnitude is shown, not "better" */}
                      <div style={{ display: 'grid', gap: 4, marginTop: 8 }}>
                        {others.map(({ id, res }) => {
                          const dPd = res.metrics.probabilityOfDetection - baseline.res.metrics.probabilityOfDetection;
                          const line = dPd === 0
                            ? `Pd equal to the ${POLICY_CATALOG[baseline.id].shortName} baseline under this scenario.`
                            : `Pd is ${dPd > 0 ? 'higher' : 'lower'} than the ${POLICY_CATALOG[baseline.id].shortName} baseline under this scenario (Δ ${(dPd * 100).toFixed(1)} pp).`;
                          return (
                            <p key={id} className="tiny" style={{ margin: 0, color: 'var(--muted)' }}>
                              <strong style={{ color: 'var(--text)' }}>{POLICY_CATALOG[id].shortName}:</strong> {line}
                            </p>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                );
              })()}

              <p className="tiny" style={{ color: 'var(--muted)', margin: '12px 0 0', lineHeight: 1.5 }}>
                Reading this: each policy ran on the identical environment realization (same seed, same
                emitters, same horizon, same receiver limits). Values are from the actual runs; Δ values are
                simple differences relative to the baseline policy. A single live scenario does not
                establish statistical superiority — it demonstrates behaviour. Scientific claims come only
                from the locked multi-seed research evaluation.
              </p>
            </div>
          )}

          {!ordered.length && !run.running && (
            <div className="panel panel-pad tiny" style={{ color: 'var(--faint)' }}>
              Select at least two policies and press Run comparison. The waterfalls and metric table appear here.
            </div>
          )}
        </>
      )}
    </div>
  );
};
