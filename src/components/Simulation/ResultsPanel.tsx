/**
 * Episode results (integration prompt §25/§26) — information grouped by meaning, not a
 * flat dump of numbers:
 *
 *   DETECTION            detection probability + weighted detection quality
 *   TRACKING / COVERAGE  coverage behaviour + staleness
 *   EFFICIENCY           reward-side / latency measures of the run
 *   EVALUATION           the run's provenance (seed, horizon, policy, scenario, mode)
 *   TECHNICAL DETAILS    lower-level counters, visually secondary
 *
 * Values come from the engine's own SchedulerMetrics; none are computed for display.
 * Single-run results are labelled LIVE SCENARIO RESULT (§30) and never mixed with the
 * locked benchmark.
 */
import React from 'react';
import { SchedulerMetrics } from '../../state/types';

interface Props {
  metrics: SchedulerMetrics | null;
  policyLabel: string;
  seed: number;
  /** Scenario preset label for the Evaluation group (§25); defaults to Custom. */
  scenarioLabel?: string;
  /** Displayed evaluation-mode tag (§25/§30). Default LIVE SCENARIO RESULT. */
  evaluationLabel?: string;
}

const fmt = (v: number | undefined, d = 1) => (v === undefined || v === null ? '—' : v.toFixed(d));

function Cell({ label, value, hint, color, big }: { label: string; value: React.ReactNode; hint?: string; color?: string; big?: boolean }) {
  return (
    <div className="res-cell">
      <div className="rl">{label}</div>
      <div className="rv" style={{ color, fontSize: big ? 20 : undefined }}>{value}</div>
      {hint && <div className="rh">{hint}</div>}
    </div>
  );
}

function GroupHead({ children }: { children: React.ReactNode }) {
  return (
    <div className="panel-title" style={{ fontSize: 10, margin: '14px 0 6px' }}>{children}</div>
  );
}

export const ResultsPanel: React.FC<Props> = ({
  metrics, policyLabel, seed, scenarioLabel, evaluationLabel = 'LIVE SCENARIO RESULT',
}) => {
  if (!metrics) {
    return (
      <div className="panel panel-pad">
        <div className="panel-title">6 · Results</div>
        <p className="tiny" style={{ color: 'var(--faint)', margin: '6px 0 0' }}>
          Run an episode to see results for this policy.
        </p>
      </div>
    );
  }

  return (
    <div className="panel panel-pad">
      <div className="spread" style={{ marginBottom: 8 }}>
        <span className="panel-title">6 · Results</span>
        <span className="chip" style={{ fontSize: 9 }}>{evaluationLabel}</span>
      </div>

      {/* ── DETECTION (§25) ─────────────────────────────────────────────────── */}
      <GroupHead>Detection</GroupHead>
      <div className="res-grid">
        <Cell
          label="Detection probability"
          value={`${(metrics.probabilityOfDetection * 100).toFixed(1)}%`}
          hint="fraction of evaluated opportunities successfully intercepted"
          color="var(--accent)"
          big
        />
        <Cell
          label="Detections"
          value={metrics.interceptedCount}
          hint={`valid intercepts of ${metrics.totalEmissions} emissions`}
          color="var(--accent)"
        />
        <Cell
          label="Avg intercept time"
          value={`${fmt(metrics.meanInterceptTimeMs)} ms`}
          hint="emission → observation delay"
        />
      </div>

      {/* ── TRACKING / COVERAGE (§25) ───────────────────────────────────────── */}
      <GroupHead>Tracking / Coverage</GroupHead>
      <div className="res-grid">
        <Cell label="Coverage violations" value={metrics.coverageViolations} hint="bands past their revisit deadline" />
        <Cell label="Max staleness" value={`${fmt(metrics.maxStalenessMs, 0)} ms`} hint="longest unobserved band interval" />
        <Cell label="Mean staleness" value={`${fmt(metrics.meanStalenessMs, 0)} ms`} hint="average time a band went unobserved" />
        <Cell label="False-alarm rate" value={metrics.probabilityOfFalseAlarm.toFixed(4)} hint="Pfa of this run" />
      </div>

      {/* ── EFFICIENCY (§25) ────────────────────────────────────────────────── */}
      <GroupHead>Efficiency</GroupHead>
      <div className="res-grid">
        <Cell label="Intercept rate" value={`${fmt(metrics.averageInterceptRate, 2)} /s`} hint="intercepts per second of episode" />
        <Cell label="P95 intercept time" value={`${fmt(metrics.p95InterceptTimeMs)} ms`} hint="tail intercept latency" />
        <Cell label="Scans used" value={metrics.totalScans} hint="dwells executed" />
        <Cell
          label="Decision latency (mean)"
          value={`${metrics.decisionLatencyUs.mean.toFixed(1)} µs`}
          hint={`p95 ${metrics.decisionLatencyUs.p95.toFixed(1)} µs`}
        />
      </div>

      {/* ── EVALUATION (§25) — provenance of this run, not performance ──────── */}
      <GroupHead>Evaluation</GroupHead>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '2px 24px' }}>
        <dl className="mono tiny" style={{ margin: 0 }}>
          <div className="kv"><span className="k">Seed</span><span className="v">{seed}</span></div>
          <div className="kv"><span className="k">Horizon</span><span className="v">{fmt(metrics.totalTimeMs / 1000, 2)} s</span></div>
          <div className="kv"><span className="k">Policy</span><span className="v">{policyLabel}</span></div>
          <div className="kv"><span className="k">Scenario</span><span className="v">{scenarioLabel ?? 'Custom'}</span></div>
          <div className="kv"><span className="k">Evaluation mode</span><span className="v">{evaluationLabel}</span></div>
        </dl>
      </div>

      {/* ── TECHNICAL DETAILS — visually secondary (§24/§25) ────────────────── */}
      <div style={{ marginTop: 12, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
        <GroupHead>Technical details</GroupHead>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '2px 24px' }}>
          <dl className="mono tiny" style={{ margin: 0 }}>
            <div className="kv"><span className="k">Total emissions</span><span className="v">{metrics.totalEmissions}</span></div>
            <div className="kv"><span className="k">Median intercept time</span><span className="v">{fmt(metrics.medianInterceptTimeMs)} ms</span></div>
            <div className="kv"><span className="k">IQR intercept time</span><span className="v">{fmt(metrics.iqrInterceptTimeMs)} ms</span></div>
            <div className="kv"><span className="k">P99 intercept time</span><span className="v">{fmt(metrics.p99InterceptTimeMs)} ms</span></div>
            <div className="kv"><span className="k">T_confirm (mean)</span><span className="v">{fmt(metrics.tConfirmMs)} ms</span></div>
          </dl>
          <dl className="mono tiny" style={{ margin: 0 }}>
            <div className="kv"><span className="k">T_detect (mean)</span><span className="v">{fmt(metrics.tDetectMs)} ms</span></div>
            <div className="kv"><span className="k">T_reacquire (mean)</span><span className="v">{fmt(metrics.tReacqMs)} ms</span></div>
            <div className="kv"><span className="k">High-priority intercept (mean)</span><span className="v">{fmt(metrics.tHighPriorityMs)} ms</span></div>
            <div className="kv"><span className="k">Decision latency (max)</span><span className="v">{metrics.decisionLatencyUs.max.toFixed(1)} µs</span></div>
            <div className="kv"><span className="k">Pulses lost to tuning / collisions</span><span className="v">{metrics.pulsesLostToSettling ?? '—'} / {metrics.pulsesLostToCollision ?? '—'}</span></div>
          </dl>
        </div>
        <p className="tiny" style={{ color: 'var(--faint)', margin: '8px 0 0' }}>
          Single-seed live run — a behaviour demonstration, not a statistical claim. Scientific
          claims come only from the locked research evaluation (see the Benchmark page).
        </p>
      </div>
    </div>
  );
};
