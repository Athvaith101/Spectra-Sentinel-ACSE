/**
 * Shared playback clock for the Scenario Lab (spec §5, §17).
 *
 * Runs the EpisodeRunner headlessly against a wall-clock-paced loop and publishes a
 * throttled snapshot for rendering, so simulation state and React re-renders stay
 * decoupled. Playback speed multiplies wall-clock time only — it never changes the
 * simulation's own 0.5 ms step semantics.
 */
import { useRef, useState, useEffect } from 'react';
import { EpisodeRunner, WaterfallRow, EpisodeEventLog } from '../app/episodeRunner';
import { DemoSchedulerAdapter } from '../app/schedulerAdapter';
import { SchedulerMetrics } from '../state/types';

export type PlaybackState = 'idle' | 'running' | 'paused' | 'done';

export interface PlaybackSnapshot {
  rows: WaterfallRow[];       // rows since the last publish (appended by the page)
  timeMs: number;
  lastEvent: EpisodeEventLog | null;
  metrics: SchedulerMetrics | null;
}

const PUBLISH_MS = 100; // render cadence — independent of simulation speed

export function usePlayback(
  runnerRef: React.MutableRefObject<EpisodeRunner | null>,
  adapterRef: React.MutableRefObject<DemoSchedulerAdapter | null>,
  speed: number,
  state: PlaybackState,
  onFinish: () => void,
) {
  const [snapshot, setSnapshot] = useState<PlaybackSnapshot>({
    rows: [], timeMs: 0, lastEvent: null, metrics: null,
  });
  const pendingRef = useRef<PlaybackSnapshot>({
    rows: [], timeMs: 0, lastEvent: null, metrics: null,
  });
  const lastPublishRef = useRef(0);
  const lastFrameRef = useRef(0);
  const rafRef = useRef(0);
  const finishedRef = useRef(false);
  const onFinishRef = useRef(onFinish);
  onFinishRef.current = onFinish;

  useEffect(() => {
    if (state !== 'running') {
      lastFrameRef.current = performance.now();
      return;
    }
    const publish = (now: number) => {
      lastPublishRef.current = now;
      const pend = pendingRef.current;
      setSnapshot({ ...pend, rows: [...pend.rows] });
      pend.rows = [];
    };

    const loop = (now: number) => {
      const runner = runnerRef.current;
      if (!runner || finishedRef.current) return; // stop the loop at episode end
      rafRef.current = requestAnimationFrame(loop);

      const dtWall = Math.min(200, now - lastFrameRef.current);
      lastFrameRef.current = now;
      const dtSim = dtWall * speed;

      const { rows, newEvent } = runner.step(dtSim);
      const pend = pendingRef.current;
      pend.rows.push(...rows);
      if (rows.length) pend.timeMs = rows[rows.length - 1].timeMs;
      if (newEvent) pend.lastEvent = newEvent;
      const sched = adapterRef.current?.scheduler;
      if (sched && sched.metrics.totalScans > 0) pend.metrics = { ...sched.metrics };

      if (now - lastPublishRef.current >= PUBLISH_MS) publish(now);

      if (runner.done) {
        publish(now);
        finishedRef.current = true;
        onFinishRef.current();
      }
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [state, speed, runnerRef, adapterRef]);

  /** Arms the finished flag for a fresh episode. */
  const armRun = () => {
    finishedRef.current = false;
    pendingRef.current = { rows: [], timeMs: 0, lastEvent: null, metrics: null };
    setSnapshot({ rows: [], timeMs: 0, lastEvent: null, metrics: null });
  };

  /** Immediate publish (used by Next Decision while paused). */
  const publishNow = () => {
    const runner = runnerRef.current;
    const pend = pendingRef.current;
    if (runner) {
      pend.timeMs = runner.timeMs;
      const sched = adapterRef.current?.scheduler;
      if (sched && sched.metrics.totalScans > 0) pend.metrics = { ...sched.metrics };
      if (runner.done) finishedRef.current = true;
    }
    setSnapshot({ ...pend, rows: [...pend.rows] });
    pend.rows = [];
  };

  return { snapshot, armRun, publishNow };
}
