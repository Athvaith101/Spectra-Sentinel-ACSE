/**
 * Benchmark page (integration prompt §15–§19, §31) — the detailed evidence page a jury
 * visits after asking "Okay, how do you know this works?". The homepage carries the
 * short snapshot; this page explains the evidence: overview → protocol → policy
 * comparison (with metric/view selector, §18) → baseline rationale (§16.4) → proposed
 * policy (§16.5) → interpretation → secondary protocols → limitations (§16.8).
 *
 * Every number is read from src/state/lockedBenchmark.ts; nothing is computed or invented.
 */
import React, { useState } from 'react';
import {
  LOCKED_EVALUATION,
  LOCKED_HEADLINE,
  LOCKED_MPC_VS_COGNITIVE,
  LOCKED_CAVEATS,
  DEV_TOURNAMENT,
  RL_TRACK,
  ADAPTIVE_LOOP_STEPS,
} from '../state/lockedBenchmark';
import { BenchmarkSnapshot } from '../components/Benchmark/BenchmarkSnapshot';

const COLOR_ADAPTIVE = 'var(--pos)';
const COLOR_BASELINE = 'var(--blue)';
const COLOR_BAD = 'var(--bad)';

/** Metrics that actually exist in the locked evaluation (integration prompt §18:
 *  only display metrics that exist for the selected benchmark). */
type MetricKey = 'pd' | 'wPoi' | 'reward' | 'coverage' | 'hpMiss' | 'meanStalenessS';

const METRICS: { key: MetricKey; label: string; short: string; fmt: (v: number) => string; higherIsBetter: boolean }[] = [
  { key: 'pd', label: 'Detection Probability', short: 'Pd', fmt: (v) => v.toFixed(4), higherIsBetter: true },
  { key: 'wPoi', label: 'Weighted POI', short: 'wPOI', fmt: (v) => v.toFixed(4), higherIsBetter: true },
  { key: 'reward', label: 'Reward', short: 'Reward', fmt: (v) => v.toFixed(3), higherIsBetter: true },
  { key: 'coverage', label: 'Coverage', short: 'Coverage', fmt: (v) => v.toFixed(3), higherIsBetter: true },
  { key: 'hpMiss', label: 'High-priority miss rate', short: 'HP miss', fmt: (v) => v.toFixed(4), higherIsBetter: false },
  { key: 'meanStalenessS', label: 'Mean staleness (s)', short: 'Staleness', fmt: (v) => v.toFixed(4), higherIsBetter: false },
];

type ViewMode = 'all' | 'relative' | 'proposed';

function MetricBar({ value, max, color, dim }: { value: number; max: number; color: string; dim?: boolean }) {
  const width = Math.max(1.5, (Math.abs(value) / max) * 100);
  return (
    <div style={{ flex: 1, height: 9, background: 'var(--surface-2)', border: '1px solid var(--border)', minWidth: 80 }}>
      <div style={{ width: `${width}%`, height: '100%', background: color, opacity: dim ? 0.5 : 1 }} />
    </div>
  );
}

/** Δ bar growing from a centre line — used by the Sequential-relative view (§17 E). */
function DeltaBar({ delta, scale, goodDirection }: { delta: number; scale: number; goodDirection: 1 | -1 }) {
  const pct = Math.min(50, (Math.abs(delta) / scale) * 50);
  const positive = delta >= 0;
  const good = positive ? goodDirection === 1 : goodDirection === -1;
  const color = good ? COLOR_ADAPTIVE : COLOR_BAD;
  return (
    <div style={{ flex: 1, height: 9, background: 'var(--surface-2)', border: '1px solid var(--border)', position: 'relative', minWidth: 80 }}>
      <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: 'var(--border-strong)' }} />
      <div
        style={{
          position: 'absolute', top: 0, bottom: 0, background: color,
          width: `${pct}%`,
          left: positive ? '50%' : `${50 - pct}%`,
        }}
      />
    </div>
  );
}

export const BenchmarkPage: React.FC = () => {
  const policies = LOCKED_EVALUATION.policies;
  const cognitive = policies.find((p) => p.id === 'cognitive')!;
  const sequential = policies.find((p) => p.id === 'sequential')!;
  const thompson = policies.find((p) => p.id === 'thompson')!;
  const s = LOCKED_EVALUATION.scale;

  const [metric, setMetric] = useState<MetricKey>('pd');
  const [view, setView] = useState<ViewMode>('all');
  const metricDef = METRICS.find((m) => m.key === metric)!;

  return (
    <div className="page" style={{ maxWidth: 1080 }}>
      <div style={{ marginBottom: 14 }}>
        <h1 className="page-title">Benchmark</h1>
        <div className="section-note">
          Controlled evaluation of adaptive scan scheduling — the project's quantitative evidence.
        </div>
      </div>

      {/* §16.1 — Benchmark overview (snapshot reused; identical source of truth) */}
      <BenchmarkSnapshot />

      {/* ── §16.2 — Evaluation protocol ──────────────────────────────────────── */}
      <section style={{ padding: '20px 0 6px', borderBottom: '1px solid var(--border)' }}>
        <div className="panel-title" style={{ marginBottom: 8 }}>Evaluation protocol</div>
        <ul className="small" style={{ margin: 0, paddingLeft: 18, lineHeight: 1.8, maxWidth: 820 }}>
          <li>
            <strong>What was evaluated.</strong> 11 scan policies — the adaptive cognitive scheduler, bandit and
            Markov baselines, MPC, a STARE-assisted variant and a portfolio — under the project's evaluation protocol.
          </li>
          <li>
            <strong>Scale.</strong> {s.seeds} seeds × {s.scenarioInstances} scenario instances × 2 episodes per
            scenario per seed = {s.episodes} episodes per policy discipline; {s.events.toLocaleString('en-US')} scored
            events; bootstrap CIs from {s.bootstrapSamples} resamples at {Math.round(s.confidence * 100)}%.
          </li>
          <li>
            <strong>Corpus.</strong> The TSRD-derived synthetic PDW corpus: {s.corpusFiles.toLocaleString('en-US')}
            {' '}files, 2,466,102,218 pulses, 0 QA failures. Evaluation was corpus-grounded; the locked test itself ran
            on fixed evaluation scenarios derived from that data pipeline.
          </li>
          <li>
            <strong>Fairness.</strong> Truth/policy isolation (no ground-truth identity or future information reaches
            any policy), no tuning on test data, multi-seed evaluation — as recorded in the report's discipline block.
          </li>
          <li>
            <strong>Freeze discipline.</strong> Artifacts were frozen (24 artifacts, MODEL FREEZE: PASS) before the
            locked evaluation ran, so the tested models are fixed artifacts rather than continually changing ones.
          </li>
        </ul>
      </section>

      {/* ── §16.3 + §17/§18 — Policy comparison with metric/view selector ────── */}
      <section style={{ padding: '20px 0 6px', borderBottom: '1px solid var(--border)' }}>
        <div className="panel-title" style={{ marginBottom: 8 }}>Policy comparison — locked evaluation</div>
        <div className="panel" style={{ overflowX: 'auto' }}>
          <table className="tbl">
            <thead>
              <tr>
                <th>Policy</th>
                <th className="num">Pd</th>
                <th className="num">Pd 95% CI</th>
                <th className="num">Weighted POI</th>
                <th className="num">HP miss</th>
                <th className="num">Coverage</th>
                <th className="num">Mean staleness (s)</th>
                <th className="num">Reward</th>
              </tr>
            </thead>
            <tbody>
              {policies.map((p) => (
                <tr key={p.id} style={p.role === 'adaptive' ? { background: 'var(--pos-soft)' } : undefined}>
                  <td style={{ whiteSpace: 'nowrap', fontWeight: p.role === 'adaptive' ? 700 : 400 }}>
                    {p.label}
                    {p.id === 'sequential' && <span className="tiny" style={{ color: 'var(--faint)' }}> (baseline)</span>}
                  </td>
                  <td className="num">{p.pd.toFixed(4)}</td>
                  <td className="num" style={{ color: 'var(--muted)' }}>[{p.pdLow.toFixed(3)}, {p.pdHigh.toFixed(3)}]</td>
                  <td className="num">{p.wPoi.toFixed(4)}</td>
                  <td className="num">{p.hpMiss.toFixed(4)}</td>
                  <td className="num">{p.coverage.toFixed(3)}</td>
                  <td className="num">{p.meanStalenessS.toFixed(4)}</td>
                  <td className="num">{p.reward.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="tiny" style={{ color: 'var(--muted)', margin: '8px 0 0' }}>
          Verbatim from the locked-evaluation report ({LOCKED_EVALUATION.source}). Markov measured coverage 0.250 under
          the protocol — reported as measured, not adjusted. Pd CIs are bootstrap intervals per the report's discipline block.
        </p>

        {/* §18 — metric / view selector driving the chart below */}
        <div className="panel panel-pad" style={{ marginTop: 14 }}>
          <div className="spread" style={{ flexWrap: 'wrap', gap: 10, marginBottom: 12 }}>
            <span className="panel-title">Metric comparison</span>
            <div className="row" style={{ gap: 10, flexWrap: 'wrap' }}>
              <label className="row" style={{ gap: 6 }}>
                <span className="tiny" style={{ color: 'var(--muted)' }}>Compare metric:</span>
                <select
                  className="control"
                  style={{ width: 190 }}
                  value={metric}
                  onChange={(e) => setMetric(e.target.value as MetricKey)}
                >
                  {METRICS.map((m) => (
                    <option key={m.key} value={m.key}>{m.label}</option>
                  ))}
                </select>
              </label>
              <div className="row" style={{ gap: 6 }}>
                <span className="tiny" style={{ color: 'var(--muted)' }}>View:</span>
                <div className="theme-toggle" role="group" aria-label="Chart view">
                  <button className={view === 'all' ? 'on' : ''} onClick={() => setView('all')}>All policies</button>
                  <button className={view === 'relative' ? 'on' : ''} onClick={() => setView('relative')}>Sequential-relative</button>
                  <button className={view === 'proposed' ? 'on' : ''} onClick={() => setView('proposed')}>Proposed vs baseline</button>
                </div>
              </div>
            </div>
          </div>

          {/* §17 A–D — all-policy bars for the selected metric */}
          {view === 'all' && (
            <div style={{ display: 'grid', gap: 6 }}>
              {[...policies].sort((a, b) => b[metric] - a[metric]).map((p) => (
                <div key={p.id} className="row" style={{ gap: 10 }}>
                  <span className="mono tiny" style={{ width: 170, flex: 'none', fontWeight: p.role === 'adaptive' ? 700 : 400 }}>
                    {p.label}
                    {p.id === 'cognitive' && <Tag color={COLOR_ADAPTIVE}>PROPOSED · ADAPTIVE</Tag>}
                    {p.id === 'sequential' && <Tag color={COLOR_BASELINE}>BASELINE</Tag>}
                  </span>
                  <MetricBar
                    value={p[metric]}
                    max={Math.max(...policies.map((q) => q[metric])) || 1}
                    color={p.role === 'adaptive' ? COLOR_ADAPTIVE : COLOR_BASELINE}
                    dim={p.role !== 'adaptive' && p.id !== 'sequential'}
                  />
                  <span className="mono tiny" style={{ width: 62, textAlign: 'right', flex: 'none' }}>{metricDef.fmt(p[metric])}</span>
                </div>
              ))}
            </div>
          )}

          {/* §17 E — Sequential-relative Δ for the selected metric */}
          {view === 'relative' && (
            <div style={{ display: 'grid', gap: 6 }}>
              {policies.filter((p) => p.id !== 'sequential').map((p) => {
                const delta = p[metric] - sequential[metric];
                const scale = Math.max(...policies.filter((q) => q.id !== 'sequential').map((q) => Math.abs(q[metric] - sequential[metric]))) || 1;
                return (
                  <div key={p.id} className="row" style={{ gap: 10 }}>
                    <span className="mono tiny" style={{ width: 170, flex: 'none', fontWeight: p.role === 'adaptive' ? 700 : 400 }}>
                      {p.label}
                      {p.id === 'cognitive' && <Tag color={COLOR_ADAPTIVE}>PROPOSED · ADAPTIVE</Tag>}
                    </span>
                    <DeltaBar delta={delta} scale={scale} goodDirection={metricDef.higherIsBetter ? 1 : -1} />
                    <span
                      className="mono tiny"
                      style={{
                        width: 78, textAlign: 'right', flex: 'none',
                        color: (delta >= 0) === metricDef.higherIsBetter ? COLOR_ADAPTIVE : COLOR_BAD,
                      }}
                    >
                      {delta >= 0 ? '+' : ''}{metricDef.fmt(delta)}
                    </span>
                  </div>
                );
              })}
              <p className="tiny" style={{ color: 'var(--muted)', margin: '6px 0 0' }}>
                Δ = policy − Sequential on {metricDef.label.toLowerCase()} under the locked evaluation. Bars grow from
                zero; {metricDef.higherIsBetter ? 'higher' : 'lower'} is better for this metric.
              </p>
            </div>
          )}

          {/* §18 — Proposed vs selected (baseline) head-to-head */}
          {view === 'proposed' && (
            <div style={{ display: 'grid', gap: 10 }}>
              <div className="row" style={{ gap: 10 }}>
                <span className="mono tiny" style={{ width: 170, flex: 'none', fontWeight: 700 }}>
                  {cognitive.label} <Tag color={COLOR_ADAPTIVE}>PROPOSED · ADAPTIVE</Tag>
                </span>
                <MetricBar value={cognitive[metric]} max={Math.max(cognitive[metric], sequential[metric]) || 1} color={COLOR_ADAPTIVE} />
                <span className="mono tiny" style={{ width: 62, textAlign: 'right', flex: 'none' }}>{metricDef.fmt(cognitive[metric])}</span>
              </div>
              <div className="row" style={{ gap: 10 }}>
                <span className="mono tiny" style={{ width: 170, flex: 'none', fontWeight: 700 }}>
                  {sequential.label} <Tag color={COLOR_BASELINE}>BASELINE</Tag>
                </span>
                <MetricBar value={sequential[metric]} max={Math.max(cognitive[metric], sequential[metric]) || 1} color={COLOR_BASELINE} />
                <span className="mono tiny" style={{ width: 62, textAlign: 'right', flex: 'none' }}>{metricDef.fmt(sequential[metric])}</span>
              </div>
              <div className="mono tiny" style={{ color: 'var(--muted)' }}>
                Δ ({cognitive.label} − {sequential.label}) = {(cognitive[metric] - sequential[metric] >= 0 ? '+' : '')}
                {metricDef.fmt(cognitive[metric] - sequential[metric])} · {metricDef.higherIsBetter ? 'higher' : 'lower'} is better
              </div>
            </div>
          )}
        </div>

        {/* §16.7 — paired interpretation (locked report's own paired analyses) */}
        <div className="panel panel-pad" style={{ marginTop: 14 }}>
          <div className="panel-title" style={{ marginBottom: 10 }}>Adaptive (Cognitive) vs Sequential — paired locked differences</div>
          <div className="res-grid">
            <div className="res-cell">
              <div className="rl">ΔPd</div>
              <div className="rv" style={{ color: 'var(--pos)' }}>+{LOCKED_HEADLINE.deltaPd.toFixed(4)}</div>
              <div className="rh">95% CI [{LOCKED_HEADLINE.deltaPdCi[0].toFixed(4)}, {LOCKED_HEADLINE.deltaPdCi[1].toFixed(4)}]</div>
            </div>
            <div className="res-cell">
              <div className="rl">ΔReward</div>
              <div className="rv" style={{ color: 'var(--pos)' }}>+{LOCKED_HEADLINE.deltaReward.toFixed(3)}</div>
              <div className="rh">Cognitive {LOCKED_HEADLINE.cognitiveReward.toFixed(3)} vs {LOCKED_HEADLINE.sequentialReward.toFixed(3)}</div>
            </div>
            <div className="res-cell">
              <div className="rl">Pd ratio</div>
              <div className="rv">≈ {LOCKED_HEADLINE.pdRatio.toFixed(2)}×</div>
              <div className="rh">measured under the locked configuration</div>
            </div>
            <div className="res-cell">
              <div className="rl">Coverage</div>
              <div className="rv">1.000 / 1.000</div>
              <div className="rh">Cognitive / Sequential</div>
            </div>
          </div>
          <p className="tiny" style={{ color: 'var(--muted)', margin: '10px 0 0' }}>
            MPC vs Cognitive, paired: ΔPd {LOCKED_MPC_VS_COGNITIVE.deltaPd.toFixed(4)}, 95% CI
            [{LOCKED_MPC_VS_COGNITIVE.pdCi[0].toFixed(4)}, +{LOCKED_MPC_VS_COGNITIVE.pdCi[1].toFixed(4)}] — the interval
            crosses zero, so no clear Pd separation is established between the two. Thompson measured the highest single-policy Pd
            ({thompson.pd.toFixed(4)}); Cognitive's headline result is its paired improvement over the sequential baseline.
          </p>
        </div>
      </section>

      {/* ── §16.4 — Sequential baseline rationale ───────────────────────────── */}
      <section style={{ padding: '20px 0 6px', borderBottom: '1px solid var(--border)' }}>
        <div className="panel-title" style={{ marginBottom: 8 }}>The Sequential baseline</div>
        <p className="small" style={{ maxWidth: 800, margin: 0 }}>
          The sequential scheduler sweeps every channel in a fixed order (<span className="mono">1 → 2 → 3 → … → N</span>)
          without using observations. It is the baseline because it represents the default behaviour of a conventional
          scanner: predictable, uniform coverage, but no explicit decision about which observation is most valuable at
          the current moment. Every adaptive policy in the evaluation is measured against it under identical conditions,
          which makes ΔPd / ΔReward interpretable as the measured value of deciding <em>where to look next</em>.
        </p>
      </section>

      {/* ── §16.5 — Proposed/adaptive policy ────────────────────────────────── */}
      <section style={{ padding: '20px 0 6px', borderBottom: '1px solid var(--border)' }}>
        <div className="panel-title" style={{ marginBottom: 8 }}>The proposed adaptive policy</div>
        <p className="small" style={{ maxWidth: 800, margin: '0 0 12px' }}>
          The cognitive scheduler treats scanning as a sequential decision problem: instead of a fixed raster, it
          maintains an evolving state of the spectrum and chooses the next channel — and how long to stay — from that
          state. Relative to sequential scanning it adds:
        </p>
        <ul className="small" style={{ margin: 0, paddingLeft: 18, lineHeight: 1.8, maxWidth: 820 }}>
          <li><strong>Belief-driven selection</strong> — per-channel activity estimates updated from its own observations.</li>
          <li><strong>Temporal context</strong> — staleness, hazard and change detection, so a channel can become important because its behaviour is changing.</li>
          <li><strong>Information value</strong> — channels it does not yet understand are worth observing; understood ones are not re-observed redundantly.</li>
          <li><strong>Adaptive dwell</strong> — the sensing budget is allocated dynamically rather than spent uniformly.</li>
          <li><strong>Coverage constraint</strong> — adaptivity never destroys broad awareness (coverage 1.000 in the locked test).</li>
        </ul>
        <div className="flow" style={{ maxWidth: 560, margin: '14px auto 0' }}>
          {ADAPTIVE_LOOP_STEPS.map((step, i) => (
            <React.Fragment key={step}>
              {i > 0 && <div className="farrow">↓</div>}
              <div className={`fbox${i === 4 ? '' : ' fbox-dim'}`}>{step}</div>
            </React.Fragment>
          ))}
        </div>
        <p className="tiny" style={{ color: 'var(--muted)', margin: '12px 0 0', maxWidth: 780 }}>
          Belief, uncertainty, hazard, information gain, staleness and adaptive dwell are the components the project
          report documents for this loop. STARE-Cognitive produced the same locked-test values as Cognitive: the locked
          evaluation validates the adaptive cognitive scheduling architecture but does not isolate STARE as the cause
          of the improvement.
        </p>
      </section>

      {/* ── §16.8 — Secondary protocols — labelled separately (integration prompt §28/§29) ─ */}
      <section style={{ padding: '20px 0 6px', borderBottom: '1px solid var(--border)' }}>
        <div className="panel-title" style={{ marginBottom: 4 }}>Secondary evaluation runs — separate protocols</div>
        <p className="section-note" style={{ marginBottom: 12 }}>
          Exploratory runs kept strictly apart from the locked table above; different configuration, not comparable numbers.
        </p>

        <div className="panel panel-pad" style={{ marginBottom: 12 }}>
          <div className="spread" style={{ flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
            <span className="mono tiny" style={{ fontWeight: 700 }}>{DEV_TOURNAMENT.title}</span>
            <span className="chip chip-warn" style={{ fontSize: 9 }}>EXPLORATORY · PRE-FREEZE</span>
          </div>
          <div className="tiny" style={{ color: 'var(--muted)', marginBottom: 8 }}>{DEV_TOURNAMENT.config}</div>
          <table className="tbl">
            <thead>
              <tr><th>Policy</th><th className="num">Pd</th><th className="num">wPOI</th><th className="num">Reward</th></tr>
            </thead>
            <tbody>
              {DEV_TOURNAMENT.rows.map((r) => (
                <tr key={r.id}>
                  <td>{r.label}</td>
                  <td className="num">{r.pd.toFixed(4)}</td>
                  <td className="num">{r.wPoi.toFixed(4)}</td>
                  <td className="num">{r.reward.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="tiny" style={{ color: 'var(--muted)', margin: '8px 0 0' }}>
            Cognitive vs Sequential (paired bootstrap): ΔPd +{DEV_TOURNAMENT.cognitiveVsSequential.deltaPd.toFixed(4)},
            95% CI [+{DEV_TOURNAMENT.cognitiveVsSequential.deltaPdCi[0].toFixed(4)}, +{DEV_TOURNAMENT.cognitiveVsSequential.deltaPdCi[1].toFixed(4)}];
            ΔHP-miss {DEV_TOURNAMENT.cognitiveVsSequential.deltaHpMiss.toFixed(4)}; ΔReward
            +{DEV_TOURNAMENT.cognitiveVsSequential.deltaReward.toFixed(4)}. These results were exploratory and were
            followed by the locked evaluation.
          </p>
        </div>

        <div className="panel panel-pad">
          <div className="spread" style={{ flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
            <span className="mono tiny" style={{ fontWeight: 700 }}>{RL_TRACK.title}</span>
            <span className="chip" style={{ fontSize: 9 }}>RESEARCH TRACK</span>
          </div>
          <div className="tiny" style={{ color: 'var(--muted)', marginBottom: 8 }}>{RL_TRACK.config}</div>
          <table className="tbl">
            <thead>
              <tr><th>Algorithm</th><th className="num">Evaluation reward</th><th className="num">Detection rate</th></tr>
            </thead>
            <tbody>
              {RL_TRACK.rows.map((r) => (
                <tr key={r.id}>
                  <td>{r.label}</td>
                  <td className="num">{r.reward.toFixed(3)}</td>
                  <td className="num">{r.detectionRate.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="tiny" style={{ color: 'var(--muted)', margin: '8px 0 0' }}>{RL_TRACK.note}</p>
        </div>
      </section>

      {/* ── §16.8 — Limitations / evidence status ───────────────────────────── */}
      <section style={{ padding: '20px 0 6px' }}>
        <div className="panel-title" style={{ marginBottom: 10 }}>Scientific integrity — stated limitations</div>
        <div style={{ display: 'grid', gap: 8, maxWidth: 860 }}>
          {LOCKED_CAVEATS.map((c) => (
            <div key={c.title} className="note note-neutral">
              <strong>{c.title}.</strong> {c.text}
            </div>
          ))}
          <div className="note">
            <strong>Source discipline.</strong> One evaluation source per table. Locked values, the development
            tournament and the RL track are never mixed; browser-simulation results never appear in any benchmark table.
          </div>
        </div>
      </section>
    </div>
  );
};

/** Restrained PROPOSED / BASELINE tag (integration prompt §19). */
function Tag({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <span
      className="mono"
      style={{
        fontSize: 7.5, letterSpacing: '0.06em', padding: '1px 4px', borderRadius: 2,
        border: '1px solid var(--border-strong)', color, whiteSpace: 'nowrap', marginLeft: 6,
      }}
    >
      {children}
    </span>
  );
}
