/**
 * Scenario Lab (spec §3 page 2) — one coherent top-to-bottom workflow:
 * configure → choose policy → run → watch → understand the decision → see results.
 * The waterfall is the visual centrepiece; nothing buries it.
 */
import React, { useEffect, useRef, useState } from 'react';
import { useLabSession } from '../state/labSession';
import { ScenarioBuilderPanel } from '../components/ScenarioBuilder/ScenarioBuilderPanel';
import { PolicySelector } from '../components/ScenarioBuilder/PolicySelector';
import { PlaybackControls } from '../components/Simulation/PlaybackControls';
import { WaterfallView } from '../components/Waterfall/WaterfallView';
import { DecisionPanel } from '../components/Simulation/DecisionPanel';
import { ResultsPanel } from '../components/Simulation/ResultsPanel';
import { PlaybackSnapshot } from '../state/playback';
import { WaterfallRow } from '../app/episodeRunner';
import { POLICY_CATALOG } from '../app/schedulerAdapter';
import { PRESETS } from '../app/scenarioBuilder';

const ACCENT: Record<string, string> = {
  sequential: '#2563a8',
  cognitive: '#0a7d57',
  staleness: '#7c5cbf',
  mpc: '#9a6700',
};

const MAX_LIVE_ROWS = 900;

export const ScenarioLabPage: React.FC = () => {
  const lab = useLabSession();
  const [liveRows, setLiveRows] = useState<WaterfallRow[]>([]);
  const [lockedResult, setLockedResult] = useState<any>(null);
  const lastSnapshotRef = useRef<PlaybackSnapshot | null>(null);

  // Drain rows appended by the playback loop into render state (once per publish).
  useEffect(() => {
    if (lastSnapshotRef.current === lab.snapshot) return;
    lastSnapshotRef.current = lab.snapshot;
    const rows = lab.snapshot.rows;
    if (rows.length) {
      setLiveRows((prev) => {
        const next = [...prev, ...rows];
        return next.length > MAX_LIVE_ROWS ? next.slice(next.length - MAX_LIVE_ROWS) : next;
      });
    }
  }, [lab.snapshot]);

  // Runner rebuilt (config/policy change, reset, replay) → clear the live view.
  useEffect(() => {
    setLiveRows([]);
  }, [lab.engineVersion]);

  const adapter = lab.adapterRef.current;
  const policyMeta = POLICY_CATALOG[lab.policyId];
  const preset = PRESETS.find((p) => p.id === lab.activePreset);
  const scenarioLabel = lab.activePreset === 'custom' ? 'Custom' : (preset?.name ?? 'Custom');
  const accent = ACCENT[lab.policyId] ?? '#0a7d57';
  const done = lab.playState === 'done';

  // Lock only the first completed result. The waterfall remains continuous and keeps updating.
  useEffect(() => {
    if (done && !lockedResult) {
      setLockedResult({
        metrics: lab.snapshot.metrics ?? lab.adapterRef.current?.metrics ?? null,
        seed: lab.cfg.seed,
        policyId: lab.policyId,
        policyLabel: POLICY_CATALOG[lab.policyId]?.displayName ?? lab.policyId,
        scenarioLabel: lab.activePreset === 'custom' ? 'Custom' : (PRESETS.find((p) => p.id === lab.activePreset)?.name ?? 'Custom'),
      });
    }
  }, [done, lockedResult, lab.snapshot.metrics, lab.cfg.seed, lab.policyId, lab.activePreset, lab.adapterRef]);

  // Continuous receiver demonstrator: every completed episode restarts with a fresh deterministic seed.
  useEffect(() => {
    if (!done) return;
    const timer = window.setTimeout(() => lab.startNextRun(), 120);
    return () => window.clearTimeout(timer);
  }, [done, lab.startNextRun]);

  return (
    <div className="page">
      <div className="spread" style={{ marginBottom: 12 }}>
        <div>
          <h1 className="page-title">Scenario Lab</h1>
          <div className="section-note">
            Build a seeded RF environment, choose the scheduler, and watch a continuously running receiver sweep across the full 2–10 GHz spectrum.
          </div>
        </div>
        <a className="btn btn-quiet" href="#/arena">Compare policies →</a>
      </div>

      {/* GEOMETRY: config (left, spans both rows) · policy→waterfall (center) ·
          run→decision (right). Results full-width below the grid. */}
      <div className="lab-grid">
        {/* COLUMN 1 — Scenario Configuration (tall left column) */}
        <div className="lab-config">
          <ScenarioBuilderPanel
            cfg={lab.cfg}
            onChange={lab.editConfig}
            onPreset={lab.applyPreset}
            onCustom={lab.chooseCustom}
            onSeedChange={lab.setSeed}
            onRandomizeSeed={lab.randomizeSeed}
            activePreset={lab.activePreset}
          />
        </div>

        {/* COLUMN 2 — Scan Policy (top) + Live Spectrum (below) */}
        <div className="lab-policy">
          <PolicySelector value={lab.policyId} onChange={lab.setPolicyId} />
        </div>
        <div className="lab-run">
          <PlaybackControls
            state={lab.playState}
            speed={lab.speed}
            onSpeed={lab.setSpeed}
            onRun={lab.playState === 'paused' ? lab.resume : lab.startRun}
            onPause={lab.pause}
            onReset={lab.reset}
            onReplay={lab.replay}
            onStep={lab.stepOnce}
            progress={lab.progress}
            timeMs={lab.timeMs}
            durationMs={lab.cfg.durationMs}
          />
        </div>
        <div className="lab-waterfall">
          <WaterfallView
            rows={liveRows}
            accent={accent}
            title={`4 · Live spectrum — ${policyMeta.displayName}`}
            mode="live"
            height={330}
            showLegend
          />
        </div>

        {/* COLUMN 3 — Decision Trace (under Run & Playback, scrolls internally) */}
        <div className="lab-decision">
          <div className="lab-decision-inner">
            <DecisionPanel
              adapter={adapter}
              currentBand={adapter?.currentChannel ?? 0}
              running={lab.playState === 'running'}
              simTimeMs={lab.timeMs}
              lastEvent={lab.snapshot.lastEvent}
              durationMs={lab.cfg.durationMs}
            />
          </div>
        </div>
      </div>

      {/* BOTTOM — first completed result stays locked while the waterfall continues */}
      <div style={{ marginTop: 12 }}>
        {lockedResult ? (
          <div>
            <div className="panel panel-pad" style={{ marginBottom: 8, borderColor: 'var(--pos)' }}>
              <div className="spread">
                <div><div className="panel-title">First run locked</div><div className="panel-sub">This result stays fixed while the waterfall continues through new deterministic scenarios.</div></div>
                <span className="chip chip-pos">LOCKED · SEED {lockedResult.seed}</span>
              </div>
            </div>
            <ResultsPanel metrics={lockedResult.metrics} policyLabel={lockedResult.policyLabel} seed={lockedResult.seed} scenarioLabel={lockedResult.scenarioLabel} />
          </div>
        ) : (
          <div className="panel panel-pad tiny" style={{ color: 'var(--faint)' }}>
            Continuous mode: the first completed run will be locked here while the receiver waterfall keeps running.
          </div>
        )}
      </div>
    </div>
  );
};
