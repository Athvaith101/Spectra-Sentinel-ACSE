/**
 * Simulation controls (spec §5): Run / Pause / Reset / Replay / Next Decision and a
 * nonlinear (log-scale) speed slider from 0.05× to 10× so slow demonstration speeds
 * (0.05/0.1/0.25×) get real control resolution. Speed scales wall-clock playback only —
 * the engine's own 0.5 ms steps are untouched (spec §17).
 */
import React from 'react';
import { PlaybackState } from '../../state/playback';

/** Useful stops on the slider track (index → speed). Roughly logarithmic. */
export const SPEED_STOPS = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10] as const;

function nearestStopIndex(s: number): number {
  let best = 0;
  for (let i = 1; i < SPEED_STOPS.length; i++) {
    if (Math.abs(Math.log(SPEED_STOPS[i]) - Math.log(s)) < Math.abs(Math.log(SPEED_STOPS[best]) - Math.log(s))) best = i;
  }
  return best;
}

interface Props {
  state: PlaybackState;
  speed: number;
  onSpeed: (s: number) => void;
  onRun: () => void;
  onPause: () => void;
  onReset: () => void;
  onReplay: () => void;
  onStep: () => void;
  progress: number;          // 0..1
  timeMs: number;
  durationMs: number;
}

export const PlaybackControls: React.FC<Props> = ({
  state, speed, onSpeed, onRun, onPause, onReset, onReplay, onStep, progress, timeMs, durationMs,
}) => {
  const stopIdx = nearestStopIndex(speed);
  return (
    <div className="panel panel-pad">
      <div className="panel-title" style={{ marginBottom: 8 }}>3 · Run & playback</div>
      <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
        {state === 'running' ? (
          <button className="btn" onClick={onPause}>Pause</button>
        ) : (
          <button className="btn btn-primary" onClick={onRun}>
            {state === 'idle' ? 'Run' : state === 'paused' ? 'Resume' : 'Re-run'}
          </button>
        )}
        <button className="btn btn-quiet" onClick={onStep} disabled={state === 'running' || state === 'done'}
          title="Advance to the next scheduler decision">
          Next decision
        </button>
        <button className="btn btn-quiet" onClick={onReset}>Reset</button>
        <button className="btn btn-quiet" onClick={onReplay} title="Re-run the same seeded scenario">
          Replay
        </button>
      </div>

      {/* Speed — nonlinear slider with useful stops */}
      <div style={{ marginTop: 12 }}>
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <span className="field" style={{ marginBottom: 0 }}>
            <span style={{ marginBottom: 0 }}>Playback speed</span>
          </span>
          <span className="chip chip-accent" style={{ fontSize: 11, padding: '2px 10px' }}>
            {speed}× {speed <= 0.25 ? '· inspect' : speed < 1 ? '· slow' : speed === 1 ? '· normal' : '· fast'}
          </span>
        </div>
        <input
          type="range" className="slider" min={0} max={SPEED_STOPS.length - 1} step={1}
          value={stopIdx}
          onChange={(e) => onSpeed(SPEED_STOPS[Number(e.target.value)])}
          aria-label="Playback speed"
        />
        <div className="spread mono" style={{ fontSize: 9, color: 'var(--faint)', marginTop: -4 }}>
          <span>0.05×</span>
          <span>0.25×</span>
          <span>1×</span>
          <span>10×</span>
        </div>
      </div>

      <div className="row" style={{ gap: 10, marginTop: 12 }}>
        <span className="mono tiny" style={{ color: 'var(--muted)', width: 112, flex: 'none' }}>
          t = {(timeMs / 1000).toFixed(2)} / {(durationMs / 1000).toFixed(1)} s
        </span>
        <div style={{ flex: 1, height: 6, background: 'var(--surface-2)', border: '1px solid var(--border)' }}>
          <div style={{ width: `${Math.min(100, progress * 100)}%`, height: '100%', background: 'var(--accent)' }} />
        </div>
        <span className="chip" style={{ fontSize: 9 }}>
          {state === 'running' ? 'running' : state === 'paused' ? 'paused' : state === 'done' ? 'complete' : 'ready'}
        </span>
      </div>
    </div>
  );
};
