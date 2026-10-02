/**
 * Shared scenario-lab session machinery: config → scenario → runner/adapter lifecycle
 * plus playback state. Used by the Scenario Lab page (single-policy playback).
 * The Policy Arena reuses the same ScenarioBuilderPanel with the same handlers.
 */
import { useRef, useState, useCallback, useEffect } from 'react';
import { ScenarioConfig, DEFAULT_CONFIG, buildScenario, newSeed, PRESETS } from '../app/scenarioBuilder';
import { BenchmarkScenario } from '../state/types';
import { DemoSchedulerAdapter, PolicyId } from '../app/schedulerAdapter';
import { EpisodeRunner } from '../app/episodeRunner';
import { PlaybackState, usePlayback } from './playback';
import { CUSTOM_PRESET } from '../components/ScenarioBuilder/ScenarioBuilderPanel';

export function useLabSession(initialPolicy: PolicyId = 'cognitive') {
  const [cfg, setCfg] = useState<ScenarioConfig>({ ...DEFAULT_CONFIG });
  const [scenario, setScenario] = useState<BenchmarkScenario>(() => buildScenario(DEFAULT_CONFIG));
  const [activePreset, setActivePreset] = useState<string>('balanced');
  const [policyId, setPolicyId] = useState<PolicyId>(initialPolicy);

  const [playState, setPlayState] = useState<PlaybackState>('idle');
  const [speed, setSpeed] = useState(2.5); // fast continuous demonstration speed (spec §3)

  const runnerRef = useRef<EpisodeRunner | null>(null);
  const adapterRef = useRef<DemoSchedulerAdapter | null>(null);
  const [engineVersion, setEngineVersion] = useState(0); // bumps when refs are rebuilt

  const buildRunner = useCallback((sc: BenchmarkScenario, config: ScenarioConfig, pol: PolicyId) => {
    const adapter = DemoSchedulerAdapter.forPolicy(pol, sc.channelCount);
    adapterRef.current = adapter;
    runnerRef.current = new EpisodeRunner(sc, config, adapter, config.seed);
    setEngineVersion((v) => v + 1);
  }, []);

  const { snapshot, armRun, publishNow } = usePlayback(runnerRef, adapterRef, speed, playState, () => {
    setPlayState('done');
  });

  // Policy change → rebuild runner (paused) so the next Run uses the new policy.
  useEffect(() => {
    buildRunner(scenario, cfg, policyId);
    setPlayState('idle');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [policyId]);

  // Config change → rebuild runner (paused) so Run always uses the shown scenario.
  useEffect(() => {
    if (engineVersion === 0) return; // initial mount already built it
    buildRunner(scenario, cfg, policyId);
    setPlayState('idle');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cfg]);

  // Reconcile panel state when refs were rebuilt while idle.
  useEffect(() => {
    if (playState === 'idle') publishNow();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engineVersion]);

  /** Core config application: sets cfg + rebuilds the world deterministically. */
  const applyConfig = useCallback((next: ScenarioConfig, presetId: string | null) => {
    setCfg(next);
    setActivePreset(presetId ?? CUSTOM_PRESET);
    setScenario(buildScenario(next));
  }, []);

  /** Named preset chosen from the dropdown. */
  const applyPreset = useCallback((presetId: string) => {
    const p = PRESETS.find((x) => x.id === presetId);
    if (!p) return;
    applyConfig({ ...p.config }, presetId);
  }, [applyConfig]);

  /** "Custom" explicitly chosen in the dropdown: keep current values, tag as custom. */
  const chooseCustom = useCallback(() => {
    setActivePreset(CUSTOM_PRESET);
  }, []);

  /** Any manual edit: keep values, move the scenario to Custom (and keep it there). */
  const editConfig = useCallback((next: ScenarioConfig) => {
    setCfg(next);
    setActivePreset(CUSTOM_PRESET);
    setScenario(buildScenario(next));
  }, []);

  /** Seed typed directly: deterministic world rebuild for the new seed. */
  const setSeed = useCallback((seed: number) => {
    applyConfig({ ...cfg, seed }, CUSTOM_PRESET);
  }, [cfg, applyConfig]);

  /** Randomize button: fresh seed, scenario becomes Custom. */
  const randomizeSeed = useCallback(() => {
    applyConfig({ ...cfg, seed: newSeed() }, CUSTOM_PRESET);
  }, [cfg, applyConfig]);

  const startRun = useCallback(() => {
    if (!runnerRef.current) return;
    // Always start from a fresh runner for a clean episode.
    buildRunner(scenario, cfg, policyId);
    armRun();
    setPlayState('running');
  }, [buildRunner, scenario, cfg, policyId, armRun]);

  const startNextRun = useCallback(() => {
    if (!runnerRef.current) return;
    const nextCfg = { ...cfg, seed: newSeed() };
    const nextScenario = buildScenario(nextCfg);
    buildRunner(nextScenario, nextCfg, policyId);
    armRun();
    setPlayState('running');
  }, [cfg, policyId, buildRunner, armRun]);

  const pause = useCallback(() => setPlayState('paused'), []);

  const resume = useCallback(() => {
    if (!runnerRef.current) return;
    if (runnerRef.current.done) {
      startRun();
    } else {
      setPlayState('running');
    }
  }, [runnerRef, startRun]);

  const reset = useCallback(() => {
    buildRunner(scenario, cfg, policyId);
    armRun();
    setPlayState('idle');
  }, [buildRunner, scenario, cfg, policyId, armRun]);

  const replay = useCallback(() => {
    startRun(); // same seed → identical scenario (research integrity)
  }, [startRun]);

  const stepOnce = useCallback(() => {
    const runner = runnerRef.current;
    if (!runner || runner.done) return;
    if (playState === 'idle') {
      armRun();
      setPlayState('paused');
    }
    runner.stepToNextDecision();
    publishNow();
    if (runner.done) setPlayState('done');
  }, [playState, armRun, publishNow]);

  const timeMs = snapshot.timeMs || runnerRef.current?.timeMs || 0;

  return {
    cfg, applyConfig, editConfig, scenario,
    activePreset, applyPreset, chooseCustom,
    setSeed, randomizeSeed,
    policyId, setPolicyId,
    playState, speed, setSpeed,
    startRun, startNextRun, pause, resume, reset, replay, stepOnce,
    runnerRef, adapterRef, engineVersion,
    snapshot, publishNow,
    timeMs,
    progress: Math.min(1, timeMs / Math.max(1, scenario.durationMs)),
  };
}
