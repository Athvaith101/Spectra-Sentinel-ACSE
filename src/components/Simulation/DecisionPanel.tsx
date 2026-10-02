/**
 * Decision panel — WHAT the scheduler chose + WHY, vertically organised for the narrow
 * right column beneath Run & Playback. Everything shown is observable scheduler state
 * or receiver output; nothing is deleted (spec: keep all information, compact rows).
 *
 * Scroll contract: the outer .lab-decision-inner wrapper scrolls vertically ONLY
 * (overflow-x hidden); content never forces the panel wider — values wrap.
 */
import React from 'react';
import { DemoSchedulerAdapter, DecisionTraceEntry, bandLabel } from '../../app/schedulerAdapter';
import { getChannelCenterFreq } from '../../simulation/rfSimulator';
import { EpisodeEventLog } from '../../app/episodeRunner';

interface Props {
  adapter: DemoSchedulerAdapter | null;
  currentBand: number;
  running: boolean;
  simTimeMs: number;
  lastEvent: EpisodeEventLog | null;
  durationMs: number;
}

const LEVEL_CLASS = { HIGH: 'lvl lvl-high', MEDIUM: 'lvl lvl-med', LOW: 'lvl lvl-low' } as const;

/** Section sub-heading inside the trace column. */
const SubHead: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="panel-title" style={{ fontSize: 10, margin: '0 0 4px' }}>{children}</div>
);

export const DecisionPanel: React.FC<Props> = ({
  adapter, currentBand, running, simTimeMs, lastEvent, durationMs,
}) => {
  const trace: DecisionTraceEntry | null = adapter?.lastTrace ?? null;
  const hitFresh = !!lastEvent && simTimeMs - lastEvent.timeMs < 600;
  const status = hitFresh ? 'HIT' : running ? 'SCANNING' : simTimeMs >= durationMs ? 'COMPLETE' : 'IDLE';

  return (
    <div className="panel panel-pad" style={{ minWidth: 0, overflowWrap: 'anywhere' }}>
      <div className="spread" style={{ marginBottom: 8, gap: 8 }}>
        <span className="panel-title">5 · Decision trace</span>
        <span className={`chip ${status === 'HIT' ? 'chip-pos' : status === 'SCANNING' ? 'chip-accent' : ''}`}
          style={{ flex: 'none' }}>
          {status === 'HIT' ? '● HIT' : status === 'SCANNING' ? '▸ SCANNING' : status === 'COMPLETE' ? '✓ COMPLETE' : '○ IDLE'}
        </span>
      </div>

      {/* ── CURRENT DECISION — what was selected ──────────────────────────── */}
      <SubHead>What was selected</SubHead>
      <div className="mono tiny" style={{ minWidth: 0 }}>
        <div className="kv"><span className="k">Frequency</span><span className="v">{(getChannelCenterFreq(currentBand) / 1000).toFixed(3)} GHz</span></div>
        <div className="kv"><span className="k">Band</span><span className="v">{bandLabel(currentBand)}</span></div>
        <div className="kv"><span className="k">Dwell</span><span className="v">{trace ? `${trace.dwellMs.toFixed(2)} ms` : '—'}</span></div>
        <div className="kv">
          <span className="k">HIT / MISS</span>
          <span className="v" style={{ color: hitFresh ? 'var(--pos)' : undefined }}>
            {hitFresh ? 'HIT' : status === 'SCANNING' ? '— (scanning)' : status === 'COMPLETE' ? '— (end)' : '—'}
          </span>
        </div>
        {hitFresh && lastEvent && (
          <div className="kv"><span className="k">Intercept detail</span><span className="v">
            {lastEvent.freqMHz?.toFixed(0)} MHz · SNR {lastEvent.snrDb?.toFixed(1)} dB
          </span></div>
        )}
      </div>

      <hr className="hr" style={{ margin: '8px 0' }} />

      {/* ── WHY — decision factors from actual scheduler state ────────────── */}
      <SubHead>Why — observable factors</SubHead>
      {trace && trace.factors.length > 0 ? (
        <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
          <tbody>
            {trace.factors.map((f) => (
              <tr key={f.key}>
                <td className="tiny" style={{ color: 'var(--muted)', padding: '1.5px 0', minWidth: 0, overflowWrap: 'anywhere' }}>{f.label}</td>
                <td className="tiny mono" style={{ textAlign: 'right', padding: '1.5px 4px', minWidth: 0, overflowWrap: 'anywhere' }}>{f.value}</td>
                <td style={{ width: 50, textAlign: 'right', padding: '1.5px 0', flex: 'none' }}>
                  <span className={LEVEL_CLASS[f.level]}>{f.level}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="tiny" style={{ color: 'var(--faint)', margin: '4px 0 0' }}>Waiting for the first decision…</p>
      )}

      <hr className="hr" style={{ margin: '8px 0' }} />

      {/* ── EXPLANATION ───────────────────────────────────────────────────── */}
      <SubHead>Explanation</SubHead>
      {trace ? (
        <p className="small" style={{
          margin: 0, lineHeight: 1.45, minWidth: 0, overflowWrap: 'anywhere',
          background: 'var(--accent-soft)', borderLeft: '3px solid var(--accent)',
          padding: '5px 8px', borderRadius: '0 3px 3px 0',
        }}>
          {trace.explanation}
        </p>
      ) : (
        <p className="tiny" style={{ color: 'var(--faint)', margin: 0 }}>—</p>
      )}

      <hr className="hr" style={{ margin: '8px 0' }} />

      {/* ── DECISION LOG ──────────────────────────────────────────────────── */}
      <SubHead>Decision log</SubHead>
      <div className="log tiny" style={{ minWidth: 0 }}>
        {(adapter?.trace ?? []).slice(-8).reverse().map((e, i) => (
          <div key={`${e.timeMs}-${i}`} className="log-row" style={{ minWidth: 0 }}>
            <span className="log-t">{(e.timeMs / 1000).toFixed(2)}s</span>
            <span className="log-b" style={{ overflowWrap: 'anywhere' }}>{e.bandLabel}</span>
            <span className="log-r" style={{ minWidth: 0, overflowWrap: 'anywhere' }}>{e.reasonLabel}</span>
          </div>
        ))}
        {!adapter?.trace.length && <span className="tiny" style={{ color: 'var(--faint)' }}>—</span>}
      </div>

      <hr className="hr" style={{ margin: '8px 0' }} />

      {/* ── TECHNICAL DETAILS — always present in the trace column ────────── */}
      <details className="tech">
        <summary>Technical details</summary>
        <div className="tech-body">
          <dl className="mono tiny" style={{ margin: 0, minWidth: 0 }}>
            {adapter && (
              <>
                <div className="kv"><span className="k">Policy (engine class)</span><span className="v">{adapter.label}</span></div>
                <div className="kv"><span className="k">Scheduler type</span><span className="v">{adapter.id === 'sequential' ? 'open-loop raster' : adapter.id === 'staleness' ? 'greedy revisit' : 'belief-driven adaptive'}</span></div>
                <div className="kv"><span className="k">Scans / intercepts</span><span className="v">{adapter.metrics.totalScans} / {adapter.metrics.interceptedCount}</span></div>
                <div className="kv"><span className="k">Decision latency (mean)</span><span className="v">{adapter.metrics.decisionLatencyUs.mean.toFixed(1)} µs</span></div>
                <div className="kv"><span className="k">Receiver front end</span><span className="v">35 µs LO settling, CFAR Pfa</span></div>
                <div className="kv"><span className="k">Information boundary</span><span className="v">PDWs only; truth is evaluation-side</span></div>
              </>
            )}
          </dl>
        </div>
      </details>
    </div>
  );
};
