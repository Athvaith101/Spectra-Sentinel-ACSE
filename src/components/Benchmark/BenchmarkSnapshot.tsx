/**
 * Benchmark snapshot (integration prompt §5–§9) — the homepage's evidence block,
 * reused on the Benchmark page. Every number is read from src/state/lockedBenchmark.ts
 * (verbatim transcription of the locked evaluation report); nothing here is computed
 * from browser runs, and no metric absent from the source is displayed.
 *
 * Layout: technical tags → evaluation scale → principal comparison (Cognitive vs
 * Sequential with Δ and CI) → full supported-policy Pd bars → benchmark reading →
 * optional Explore Full Benchmark CTA (homepage only, §14).
 */
import React from 'react';
import {
  LOCKED_EVALUATION,
  LOCKED_HEADLINE,
} from '../../state/lockedBenchmark';

const COLOR_ADAPTIVE = 'var(--pos)';
const COLOR_BASELINE = 'var(--blue)';

/** Compact technical tags (integration prompt §13). No "trained model" tag: the
 *  backend integration is not yet verified, so the UI must not claim it. */
const TECH_TAGS = [
  'Locked benchmark',
  'Reproducible evaluation',
  'Adaptive scan scheduling',
  'Belief-driven sensing',
  'Temporal context',
  'Coverage-aware decisions',
] as const;

function PdBar({ value, max, color, dim }: { value: number; max: number; color: string; dim?: boolean }) {
  const width = Math.max(1.5, (value / max) * 100);
  return (
    <div style={{ flex: 1, height: 9, background: 'var(--surface-2)', border: '1px solid var(--border)', minWidth: 90 }}>
      <div style={{ width: `${width}%`, height: '100%', background: color, opacity: dim ? 0.5 : 1 }} />
    </div>
  );
}

export const BenchmarkSnapshot: React.FC<{ showCta?: boolean }> = ({ showCta = false }) => {
  const policies = LOCKED_EVALUATION.policies;
  const cognitive = policies.find((p) => p.id === 'cognitive')!;
  const sequential = policies.find((p) => p.id === 'sequential')!;
  const others = policies
    .filter((p) => p.id !== 'cognitive' && p.id !== 'sequential')
    .sort((a, b) => b.pd - a.pd);
  const maxPd = Math.max(...policies.map((p) => p.pd));
  const s = LOCKED_EVALUATION.scale;

  const scaleItems = [
    { v: s.corpusFiles.toLocaleString('en-US'), l: 'corpus files' },
    { v: '2.466B', l: 'pulses (synthetic PDW)' },
    { v: String(s.seeds), l: 'seeds' },
    { v: String(s.episodes), l: 'episodes' },
    { v: String(s.scenarioInstances), l: 'scenario instances' },
    { v: String(s.policies), l: 'policies' },
  ];

  return (
    <section id="benchmark-snapshot" style={{ padding: '22px 0 8px', borderBottom: '1px solid var(--border)' }}>
      <div className="spread" style={{ flexWrap: 'wrap', gap: 8, marginBottom: 4 }}>
        <div className="panel-title" style={{ fontSize: 13 }}>Benchmark snapshot</div>
        <span className="chip chip-accent" style={{ fontSize: 9 }}>LOCKED EVALUATION · READ ONLY</span>
      </div>
      <div className="section-note" style={{ marginBottom: 14, maxWidth: 820 }}>
        Corpus-grounded evaluation using the TSRD-derived synthetic PDW corpus —{' '}
        {s.corpusFiles.toLocaleString('en-US')} QA-passed files containing 2,466,102,218 pulses.
        Results below are the project's locked-evaluation measurements, not live browser runs.
      </div>

      {/* Technical tags (§13) — compact, factual, integration-state aware */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 14 }}>
        {TECH_TAGS.map((t) => (
          <span key={t} className="chip" style={{ fontSize: 9 }}>{t}</span>
        ))}
      </div>

      {/* A. Evaluation scale */}
      <div
        style={{
          display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 1,
          background: 'var(--border)', border: '1px solid var(--border)', borderRadius: 4, overflow: 'hidden',
          marginBottom: 18,
        }}
      >
        {scaleItems.map((x) => (
          <div key={x.l} style={{ background: 'var(--surface)', padding: '10px 12px' }}>
            <div className="mono" style={{ fontSize: 18, fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>{x.v}</div>
            <div className="micro" style={{ color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{x.l}</div>
          </div>
        ))}
      </div>

      {/* B. Principal comparison — Cognitive vs Sequential */}
      <div className="panel panel-pad" style={{ marginBottom: 18 }}>
        <div className="panel-title" style={{ marginBottom: 10 }}>Adaptive vs sequential — measured Pd (locked configuration)</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 1, background: 'var(--border)', border: '1px solid var(--border)', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{ background: 'var(--pos-soft)', padding: '12px 14px' }}>
            <div className="res-cell-rl">Adaptive (Cognitive)</div>
            <div className="mono" style={{ fontSize: 26, fontWeight: 700, color: 'var(--pos)' }}>{LOCKED_HEADLINE.cognitivePd.toFixed(4)}</div>
            <div className="tiny" style={{ color: 'var(--muted)' }}>Pd · coverage 1.000 · reward {LOCKED_HEADLINE.cognitiveReward.toFixed(3)}</div>
          </div>
          <div style={{ background: 'var(--surface)', padding: '12px 14px' }}>
            <div className="res-cell-rl">Sequential (baseline)</div>
            <div className="mono" style={{ fontSize: 26, fontWeight: 700, color: 'var(--blue)' }}>{LOCKED_HEADLINE.sequentialPd.toFixed(4)}</div>
            <div className="tiny" style={{ color: 'var(--muted)' }}>Pd · coverage 1.000 · reward {LOCKED_HEADLINE.sequentialReward.toFixed(3)}</div>
          </div>
          <div style={{ background: 'var(--surface)', padding: '12px 14px' }}>
            <div className="res-cell-rl">ΔPd (paired)</div>
            <div className="mono" style={{ fontSize: 26, fontWeight: 700, color: 'var(--accent)' }}>
              +{LOCKED_HEADLINE.deltaPd.toFixed(4)}
            </div>
            <div className="tiny mono" style={{ color: 'var(--muted)' }}>
              95% CI [{LOCKED_HEADLINE.deltaPdCi[0].toFixed(4)}, {LOCKED_HEADLINE.deltaPdCi[1].toFixed(4)}]
            </div>
          </div>
          <div style={{ background: 'var(--surface)', padding: '12px 14px' }}>
            <div className="res-cell-rl">ΔReward (paired)</div>
            <div className="mono" style={{ fontSize: 26, fontWeight: 700, color: 'var(--accent)' }}>
              +{LOCKED_HEADLINE.deltaReward.toFixed(3)}
            </div>
            <div className="tiny" style={{ color: 'var(--muted)' }}>≈ {LOCKED_HEADLINE.pdRatio.toFixed(2)}× Pd ratio in this configuration</div>
          </div>
        </div>
        <p className="tiny" style={{ color: 'var(--muted)', margin: '10px 0 0' }}>
          Paired locked-test difference (Cognitive − Sequential), {s.seeds} seeds, {s.scenarioInstances} scenario instances,
          bootstrap {s.bootstrapSamples} resamples at {Math.round(s.confidence * 100)}% confidence.
        </p>
      </div>

      {/* C. Policy comparison bars — supported policies only */}
      <div className="panel panel-pad" style={{ marginBottom: 14 }}>
        <div className="spread" style={{ marginBottom: 10 }}>
          <span className="panel-title">Detection probability by policy</span>
          <span className="chip" style={{ fontSize: 9 }}>{policies.length} locked policies</span>
        </div>
        <div style={{ display: 'grid', gap: 6 }}>
          <PolicyBarRow p={cognitive} maxPd={maxPd} color={COLOR_ADAPTIVE} tag="PROPOSED · ADAPTIVE" />
          <PolicyBarRow p={sequential} maxPd={maxPd} color={COLOR_BASELINE} tag="BASELINE" />
          {others.map((p) => (
            <PolicyBarRow key={p.id} p={p} maxPd={maxPd} color={p.role === 'adaptive' ? COLOR_ADAPTIVE : COLOR_BASELINE} dim />
          ))}
        </div>
        <p className="tiny" style={{ color: 'var(--muted)', margin: '10px 0 0' }}>
          Thompson measured the highest single-policy Pd ({policies.find((p) => p.id === 'thompson')!.pd.toFixed(4)});
          Cognitive delivered the best locked paired ΔPd and ΔReward against the sequential baseline.
        </p>
      </div>

      {/* E. Benchmark reading (integration prompt §11) — concise interpretation of the
          chart above, using only documented locked values. */}
      <div className="panel panel-pad" style={{ marginBottom: 14, background: 'var(--surface-2)' }}>
        <div className="panel-title" style={{ marginBottom: 6 }}>Benchmark reading</div>
        <p className="small" style={{ margin: 0, maxWidth: 820 }}>
          Under the locked benchmark configuration, Cognitive measured higher Pd than the Sequential
          baseline while maintaining the same reported coverage. The measured Pd difference was
          <strong> +{LOCKED_HEADLINE.deltaPd.toFixed(4)}</strong>, with a paired 95% CI of
          <span className="mono"> [+{LOCKED_HEADLINE.deltaPdCi[0].toFixed(4)}, +{LOCKED_HEADLINE.deltaPdCi[1].toFixed(4)}]</span>.
        </p>
        <div className="mono tiny" style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: '4px 18px', color: 'var(--muted)' }}>
          <span><strong style={{ color: 'var(--text)' }}>Pd:</strong> {LOCKED_HEADLINE.cognitivePd.toFixed(4)} vs {LOCKED_HEADLINE.sequentialPd.toFixed(4)}</span>
          <span><strong style={{ color: 'var(--text)' }}>ΔPd:</strong> +{LOCKED_HEADLINE.deltaPd.toFixed(4)}</span>
          <span><strong style={{ color: 'var(--text)' }}>Reward:</strong> {LOCKED_HEADLINE.cognitiveReward.toFixed(3)} vs {LOCKED_HEADLINE.sequentialReward.toFixed(3)}</span>
          <span><strong style={{ color: 'var(--text)' }}>Pd ratio:</strong> ≈{LOCKED_HEADLINE.pdRatio.toFixed(2)}× measured under the locked configuration</span>
        </div>
      </div>

      {/* D. Evaluation-integrity statement */}
      <p className="section-note" style={{ maxWidth: 820, marginBottom: 14 }}>
        All values are quoted verbatim from the project's locked-evaluation report. The interactive
        demos on this site are separate browser simulations; their results are labelled
        <span className="mono"> LIVE SCENARIO RESULT</span> and are never mixed with these numbers.
      </p>

      {/* F. Homepage CTA (integration prompt §14) — prominent navigation to the full
          evidence page. Rendered on the homepage only. */}
      {showCta && (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '4px 0 18px' }}>
          <a href="#/benchmark" className="btn btn-primary" style={{ fontSize: 13, padding: '9px 22px' }}>
            Explore Full Benchmark →
          </a>
        </div>
      )}
    </section>
  );
};

function PolicyBarRow({ p, maxPd, color, dim, tag }: { p: { id: string; label: string; pd: number; role: string }; maxPd: number; color: string; dim?: boolean; tag?: string }) {
  return (
    <div className="row" style={{ gap: 10 }}>
      <span
        className="mono tiny"
        style={{ width: 150, flex: 'none', fontWeight: tag ? 700 : 400, color: dim ? 'var(--muted)' : 'var(--text)', display: 'flex', alignItems: 'center', gap: 6 }}
      >
        {p.label}
        {tag && (
          <span
            className="mono"
            style={{
              fontSize: 7.5, letterSpacing: '0.06em', padding: '1px 4px', borderRadius: 2,
              border: '1px solid var(--border-strong)', color: color, whiteSpace: 'nowrap',
            }}
          >
            {tag}
          </span>
        )}
      </span>
      <PdBar value={p.pd} max={maxPd} color={color} dim={dim} />
      <span className="mono tiny" style={{ width: 58, textAlign: 'right', flex: 'none', fontVariantNumeric: 'tabular-nums' }}>
        {p.pd.toFixed(4)}
      </span>
    </div>
  );
}
