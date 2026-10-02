/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  BenchmarkScenario,
  ChannelBelief,
  GroundTruthPulse,
  LatencyBreakdownUs,
  ReceiverAction,
  ReceiverFrontEndState,
  SchedulerMetrics,
  BayesianUpdateMode,
  HazardModelMode,
  ReactiveEvasionType,
  CognitiveOperatingMode,
  ComponentGates,
  CompositeMissionScore,
  AdaptationMetrics,
  ScenarioMissionMode,
} from './types';
import { RFSimulator, DEFAULT_SCENARIOS, TOTAL_CHANNELS, getChannelCenterFreq, getMissionModeForScenario } from './engine/rfSimulator';
import { SequentialScheduler, AdaptivePortfolioScheduler } from './engine/schedulers';
import { PhysicalReceiverFrontEnd } from './environment/receiverModel';
import { attributeTruthReceiptsForEvaluation } from './benchmark/harness';
import { HazardSurvivalModel } from './belief/hazard';
import { audioWarning } from './engine/audioWarning';
import { SpectrumWaterfall } from './components/SpectrumWaterfall';
import { HeadToHeadComparison } from './components/HeadToHeadComparison';
import { PhysicalReceiverTelemetry } from './components/PhysicalReceiverTelemetry';
import { TacticalSpectrumView } from './components/TacticalSpectrumView';
import { TransitionMatrixView } from './components/TransitionMatrixView';
import { HighLoadLatencyMonitor } from './components/HighLoadLatencyMonitor';
import { BenchmarkRunnerModal } from './components/BenchmarkRunnerModal';
import { AblationStudyModal } from './components/AblationStudyModal';
import { ParetoFrontierModal } from './components/ParetoFrontierModal';
import { CognitiveMetaPolicyHUD } from './components/CognitiveMetaPolicyHUD';
import {
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Award,
  Zap,
  Radio,
  Sliders,
  ShieldCheck,
  AlertCircle,
  PlusCircle,
  Shuffle,
  Layers,
  TrendingUp,
  Cpu,
  Target,
} from 'lucide-react';

export default function App() {
  // Scenario state
  const [selectedScenarioIndex, setSelectedScenarioIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [simSpeed, setSimSpeed] = useState<number>(1.0);
  const [isAudioMuted, setIsAudioMuted] = useState<boolean>(true);
  const [isShieldActive, setIsShieldActive] = useState<boolean>(true);
  const [isBenchmarkModalOpen, setIsBenchmarkModalOpen] = useState<boolean>(false);
  const [isAblationModalOpen, setIsAblationModalOpen] = useState<boolean>(false);
  const [isParetoModalOpen, setIsParetoModalOpen] = useState<boolean>(false);

  // Research Architecture Modes
  const [bayesianMode, setBayesianMode] = useState<BayesianUpdateMode>('discounted_beta');
  const [hazardMode, setHazardMode] = useState<HazardModelMode>('weibull');
  const [selectedTactic, setSelectedTactic] = useState<ReactiveEvasionType>('typeA_freq_shift');
  const [activeToast, setActiveToast] = useState<string | null>(null);

  // Live simulation references
  const simRef = useRef<RFSimulator>(new RFSimulator(DEFAULT_SCENARIOS[0]));
  const seqRef = useRef<SequentialScheduler>(new SequentialScheduler(TOTAL_CHANNELS, 2.0));
  const smartRef = useRef<AdaptivePortfolioScheduler>(new AdaptivePortfolioScheduler(TOTAL_CHANNELS, 150, 42));
  const seqFrontEndRef = useRef<PhysicalReceiverFrontEnd>(new PhysicalReceiverFrontEnd());
  const smartFrontEndRef = useRef<PhysicalReceiverFrontEnd>(new PhysicalReceiverFrontEnd());

  const seqNextDecisionAtMsRef = useRef<number>(0);
  const smartNextDecisionAtMsRef = useRef<number>(0);
  const lastFrameTimeRef = useRef<number>(performance.now());
  const reqAnimRef = useRef<number | null>(null);

  // Reactive state for UI updates
  const [currentTimeMs, setCurrentTimeMs] = useState<number>(0);
  const [currentPulses, setCurrentPulses] = useState<GroundTruthPulse[]>([]);
  const [seqAction, setSeqAction] = useState<ReceiverAction | null>(null);
  const [smartAction, setSmartAction] = useState<ReceiverAction | null>(null);
  const [recentSeqHits, setRecentSeqHits] = useState<GroundTruthPulse[]>([]);
  const [recentSmartHits, setRecentSmartHits] = useState<GroundTruthPulse[]>([]);
  const [seqMetrics, setSeqMetrics] = useState<SchedulerMetrics>(seqRef.current.metrics);
  const [smartMetrics, setSmartMetrics] = useState<SchedulerMetrics>(smartRef.current.metrics);
  const [channelBeliefs, setChannelBeliefs] = useState<ChannelBelief[]>(smartRef.current.channelBeliefs);
  const [transitionMatrix, setTransitionMatrix] = useState<Float32Array[]>(smartRef.current.transitionMatrix);
  const [latencyBreakdown, setLatencyBreakdown] = useState<LatencyBreakdownUs>(smartRef.current.lastLatencyBreakdown);
  const [smartFrontEndState, setSmartFrontEndState] = useState<ReceiverFrontEndState>(smartFrontEndRef.current.state);
  const [operatingMode, setOperatingMode] = useState<CognitiveOperatingMode>('EXPLOIT');
  const [componentGates, setComponentGates] = useState<ComponentGates>(smartRef.current.activeGates);
  const [compositeMissionScore, setCompositeMissionScore] = useState<CompositeMissionScore>(smartRef.current.compositeMissionScore);
  const [adaptationMetrics, setAdaptationMetrics] = useState<AdaptationMetrics>(smartRef.current.metaController.getAdaptationMetrics());
  const [missionMode, setMissionMode] = useState<ScenarioMissionMode>('dense_agile');
  const [portfolioExpert, setPortfolioExpert] = useState<string>('SCOUTING');

  // Initialize/Reset Simulation
  const resetSimulation = useCallback((scenarioIndex: number = selectedScenarioIndex) => {
    const sc = DEFAULT_SCENARIOS[scenarioIndex];
    simRef.current.loadScenario(sc);
    seqRef.current.reset();
    smartRef.current.reset();

    const currentMission = getMissionModeForScenario(scenarioIndex);
    smartRef.current.setMissionMode(currentMission);
    setMissionMode(currentMission);

    smartRef.current.shieldActive = isShieldActive;
    smartRef.current.beliefEngine.setMode(bayesianMode);
    HazardSurvivalModel.setMode(hazardMode);

    seqFrontEndRef.current.reset();
    smartFrontEndRef.current.reset();
    seqFrontEndRef.current.tuneToChannel(0, 2.0, 0);
    smartFrontEndRef.current.tuneToChannel(0, 2.0, 0);

    seqNextDecisionAtMsRef.current = 0;
    smartNextDecisionAtMsRef.current = 0;
    lastFrameTimeRef.current = performance.now();

    setCurrentTimeMs(0);
    setCurrentPulses([]);
    setSeqAction(null);
    setSmartAction(null);
    setRecentSeqHits([]);
    setRecentSmartHits([]);
    setSeqMetrics({ ...seqRef.current.metrics });
    setSmartMetrics({ ...smartRef.current.metrics });
    setChannelBeliefs([...smartRef.current.channelBeliefs]);
    setTransitionMatrix([...smartRef.current.transitionMatrix]);
    setSmartFrontEndState({ ...smartFrontEndRef.current.state });
    setOperatingMode(smartRef.current.currentOperatingMode);
    setComponentGates({ ...smartRef.current.activeGates });
    setCompositeMissionScore({ ...smartRef.current.compositeMissionScore });
    setAdaptationMetrics({ ...smartRef.current.metaController.getAdaptationMetrics() });
    setPortfolioExpert(smartRef.current.getPortfolioTelemetry().scoutingComplete ? smartRef.current.getPortfolioTelemetry().selectedExpert.toUpperCase() : 'SCOUTING');
  }, [selectedScenarioIndex, isShieldActive, bayesianMode, hazardMode]);

  // Handle Scenario Switch
  const handleSelectScenario = (index: number) => {
    setSelectedScenarioIndex(index);
    resetSimulation(index);
  };

  // Toggle Coverage Shield
  const handleToggleShield = () => {
    const nextVal = !isShieldActive;
    setIsShieldActive(nextVal);
    smartRef.current.shieldActive = nextVal;
  };

  // Toggle Audio Alert
  const handleToggleAudio = () => {
    const muted = audioWarning.toggleMute();
    setIsAudioMuted(muted);
  };

  // Inject Adversarial Shift
  const handleTriggerAdversarialShift = () => {
    simRef.current.triggerAdversarialShift();
    showToast('Adversarial Hopping Shift Triggered (+11 Stride)');
  };

  // Inject Novel Unknown Threat
  const handleInjectNovelThreat = () => {
    simRef.current.injectNovelEmitter();
    showToast('Novel Unknown Threat Injected in Spectrum (OOD Monitoring Active)');
  };

  // Handle Bayesian Mode change
  const handleBayesianModeChange = (mode: BayesianUpdateMode) => {
    setBayesianMode(mode);
    smartRef.current.beliefEngine.setMode(mode);
    showToast(`Bayesian Belief Mode: ${mode === 'discounted_beta' ? 'Discounted Beta (ρ=0.96)' : 'Exact Conjugate Beta-Bernoulli'}`);
  };

  // Handle Hazard Mode change
  const handleHazardModeChange = (mode: HazardModelMode) => {
    setHazardMode(mode);
    HazardSurvivalModel.setMode(mode);
    showToast(`Recurrence Hazard Function: ${mode.toUpperCase()} model`);
  };

  // Inject specific reactive tactic
  const handleTriggerSpecificTactic = () => {
    simRef.current.forceReactiveTactic(selectedTactic);
    const tacticLabels: Record<ReactiveEvasionType, string> = {
      typeA_freq_shift: 'Type A: Agile Frequency Hop Stride (+Bins)',
      typeB_pri_stagger: 'Type B: PRI Jitter & Stagger (±25%)',
      typeC_pw_reduction: 'Type C: Pulse-Width Reduction (-3 dB Energy)',
      typeD_silence_burst: 'Type D: Radar Silence / Transmission Blanking',
      typeE_decoy_emission: 'Type E: Ghost Decoy Beacon Pulse Emission',
      typeF_deceptive_markov: 'Type F: Inverted Deceptive Markov Hopping Pattern',
      typeG_randomization: 'Type G: White-Noise Uniform Random Hopping',
    };
    showToast(`Radar Evasion Executed: ${tacticLabels[selectedTactic]}`);
  };

  const showToast = (msg: string) => {
    setActiveToast(msg);
    setTimeout(() => setActiveToast(null), 3500);
  };

  // Core 60fps Simulation Loop
  useEffect(() => {
    const loop = (now: number) => {
      if (isPlaying) {
        const deltaRealMs = Math.min(50, now - lastFrameTimeRef.current);
        const dtSimMs = deltaRealMs * simSpeed;

        const sim = simRef.current;
        const seq = seqRef.current;
        const smart = smartRef.current;
        const seqFe = seqFrontEndRef.current;
        const smartFe = smartFrontEndRef.current;

        // Step simulation time in fine slices (0.5ms steps)
        const stepSizeMs = 0.5;
        const subSteps = Math.max(1, Math.round(dtSimMs / stepSizeMs));
        const dtSlice = dtSimMs / subSteps;

        let lastPulses: GroundTruthPulse[] = [];
        let curSeqAct = seqAction;
        let curSmartAct = smartAction;
        const seqHits: GroundTruthPulse[] = [];
        const smartHits: GroundTruthPulse[] = [];

        for (let s = 0; s < subSteps; s++) {
          const t = sim.currentTimeMs;
          const tNext = t + dtSlice;

          // 1. Advance RF Spectrum Environment
          const pulses = sim.step(dtSlice);
          if (pulses.length > 0) {
            lastPulses = pulses;
          }

          // Register ground truth emissions for accurate Pd calculation
          for (const p of pulses) {
            seq.registerGroundTruthPulse(p);
            smart.registerGroundTruthPulse(p);
          }

          // 2. Sequential Scheduler Step
          if (t >= seqNextDecisionAtMsRef.current) {
            curSeqAct = seq.getDecision(t);
            seqNextDecisionAtMsRef.current = t + curSeqAct.dwellTimeMs;
            seqFe.tuneToChannel(curSeqAct.channelIndex, curSeqAct.dwellTimeMs, t);
          }
          const seqPDWs = seqFe.processPulses(pulses, t, tNext);
          // Truth-side receipt sidecar (audit): drained after front-end processing and
          // consumed ONLY by the evaluation join below — never by the scheduler.
          const seqReceipts = seqFe.drainTruthReceipts();
          for (const pdw of seqPDWs) {
            seq.onPDWObserved(pdw, t);
            if (!pdw.isFalseAlarm) {
              // Evaluation-layer join (audit): display correlation uses the truth-side
              // receipt, never a policy-visible field. Falls back to time/frequency
              // correlation against this slice's emissions for context.
              const pdwReceipt = seqReceipts.find((r) => r.pdwId === pdw.id);
              const matched = pdwReceipt
                ? pulses.find((p) => p.id === pdwReceipt.pulseId)
                : pulses.find(
                    (p) =>
                      p.channelIndex === pdw.channelIndex &&
                      Math.abs(p.timestampMs - pdw.toaMs) < 0.01 &&
                      Math.abs(p.freqMHz - pdw.freqMHz) < 3.0
                  );
              seqHits.push({
                id: pdw.id,
                timestampMs: pdw.toaMs,
                channelIndex: pdw.channelIndex,
                freqMHz: pdw.freqMHz,
                pulseWidthUs: pdw.pulseWidthUs,
                amplitudeDbm: pdw.amplitudeDbm,
                snrDb: pdw.snrDb,
                emitterId: matched?.emitterId || pdw.associationId || 'assoc-unknown',
                emitterName: matched?.emitterName || pdw.associationId || 'assoc-unknown',
                emitterType: matched?.emitterType || 'frequency_agile',
                priority: matched?.priority || 3,
                isMainbeam: matched?.isMainbeam ?? true,
              });
            }
          }
          attributeTruthReceiptsForEvaluation(sim, seq.tracker, seqPDWs, seqReceipts, t, false);
          seq.onDwellCompleted(t);

          // 3. Cognitive Smart Scheduler Step
          if (t >= smartNextDecisionAtMsRef.current) {
            curSmartAct = smart.getDecision(t);
            smartNextDecisionAtMsRef.current = t + curSmartAct.dwellTimeMs;
            smartFe.tuneToChannel(curSmartAct.channelIndex, curSmartAct.dwellTimeMs, t);
          }
          const smartPDWs = smartFe.processPulses(pulses, t, tNext);
          // Truth-side receipt sidecar (audit): drained after front-end processing and
          // consumed ONLY by the evaluation join below — never by the scheduler.
          const smartReceipts = smartFe.drainTruthReceipts();
          for (const pdw of smartPDWs) {
            smart.onPDWObserved(pdw, t);
            if (!pdw.isFalseAlarm) {
              // Evaluation-layer join (audit): same receipt-based correlation as the
              // sequential branch; interception closure flows through truth receipts.
              const pdwReceipt = smartReceipts.find((r) => r.pdwId === pdw.id);
              const matched = pdwReceipt
                ? pulses.find((p) => p.id === pdwReceipt.pulseId)
                : pulses.find(
                    (p) =>
                      p.channelIndex === pdw.channelIndex &&
                      Math.abs(p.timestampMs - pdw.toaMs) < 0.01 &&
                      Math.abs(p.freqMHz - pdw.freqMHz) < 3.0
                  );
              smartHits.push({
                id: pdw.id,
                timestampMs: pdw.toaMs,
                channelIndex: pdw.channelIndex,
                freqMHz: pdw.freqMHz,
                pulseWidthUs: pdw.pulseWidthUs,
                amplitudeDbm: pdw.amplitudeDbm,
                snrDb: pdw.snrDb,
                emitterId: matched?.emitterId || pdw.associationId || 'assoc-unknown',
                emitterName: matched?.emitterName || pdw.associationId || 'assoc-unknown',
                emitterType: matched?.emitterType || 'frequency_agile',
                priority: matched?.priority || 4,
                isMainbeam: matched?.isMainbeam ?? true,
              });

              if (!audioWarning.isMuted) {
                audioWarning.playInterceptBeep(4, pdw.freqMHz);
              }
            }
          }
          // Closed-loop reactive trigger via truth-side receipts (evaluation layer only)
          attributeTruthReceiptsForEvaluation(sim, smart.tracker, smartPDWs, smartReceipts, t, true);
          smart.onDwellCompleted(t);
        }

        // Attach front-end loss statistics to metrics
        seq.metrics.pulsesLostToSettling = seqFe.state.totalPulsesDuringTuning;
        seq.metrics.pulsesLostToCollision = seqFe.state.totalPulsesCollided;
        smart.metrics.pulsesLostToSettling = smartFe.state.totalPulsesDuringTuning;
        smart.metrics.pulsesLostToCollision = smartFe.state.totalPulsesCollided;

        // Push to React state periodically
        setCurrentTimeMs(sim.currentTimeMs);
        if (lastPulses.length > 0) {
          setCurrentPulses(lastPulses);
        }
        if (curSeqAct) setSeqAction(curSeqAct);
        if (curSmartAct) setSmartAction(curSmartAct);
        if (seqHits.length > 0) {
          setRecentSeqHits((prev) => [...prev.slice(-40), ...seqHits]);
        }
        if (smartHits.length > 0) {
          setRecentSmartHits((prev) => [...prev.slice(-40), ...smartHits]);
        }

        setSeqMetrics({ ...seq.metrics });
        setSmartMetrics({ ...smart.metrics });
        setChannelBeliefs([...smart.channelBeliefs]);
        setTransitionMatrix([...smart.transitionMatrix]);
        setLatencyBreakdown({ ...smart.lastLatencyBreakdown });
        setSmartFrontEndState({ ...smartFe.state });
        setOperatingMode(smart.currentOperatingMode);
        setComponentGates({ ...smart.activeGates });
        setCompositeMissionScore({ ...smart.compositeMissionScore });
        setAdaptationMetrics({ ...smart.metaController.getAdaptationMetrics() });
        setPortfolioExpert(smart.getPortfolioTelemetry().scoutingComplete ? smart.getPortfolioTelemetry().selectedExpert.toUpperCase() : 'SCOUTING');
      }

      lastFrameTimeRef.current = now;
      reqAnimRef.current = requestAnimationFrame(loop);
    };

    reqAnimRef.current = requestAnimationFrame(loop);
    return () => {
      if (reqAnimRef.current) cancelAnimationFrame(reqAnimRef.current);
    };
  }, [isPlaying, simSpeed, seqAction, smartAction]);

  const activeScenario = DEFAULT_SCENARIOS[selectedScenarioIndex];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500/30 selection:text-emerald-200">
      {/* Top Tactical Navigation & Control Bar */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur-md sticky top-0 z-40 px-4 py-2.5">
        <div className="max-w-7xl mx-auto flex flex-col lg:flex-row items-center justify-between gap-3">
          {/* Title & Badge */}
          <div className="flex items-center gap-3 w-full lg:w-auto justify-between lg:justify-start">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/40 flex items-center justify-center text-emerald-400 shadow-[0_0_12px_#10b98122]">
                <Radio className="w-4 h-4 animate-pulse" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-sm font-bold tracking-wider uppercase font-mono text-white">
                    SIH26055 — EW Smart Scan Cognitive Receiver
                    <span className="ml-3 text-[10px] text-cyan-300 font-mono border border-cyan-500/30 px-2 py-1 rounded">AUTO EXPERT: {portfolioExpert}</span>
                  </h1>
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                    DRDO SPEC
                  </span>
                </div>
                <p className="text-[11px] font-mono text-slate-400">
                  Electronic Support Receiver Scheduler &bull; Wideband Surveillance Against Agile Emitters
                </p>
              </div>
            </div>

            {/* Quick Benchmark Button */}
            <button
              onClick={() => setIsBenchmarkModalOpen(true)}
              className="lg:hidden flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-mono text-xs font-bold transition-all shadow-lg shadow-emerald-950/50"
            >
              <Award className="w-3.5 h-3.5" /> Test Suite
            </button>
          </div>

          {/* Scenario Selector & Controls */}
          <div className="flex flex-wrap items-center gap-2.5 w-full lg:w-auto justify-end">
            {/* Scenario dropdown */}
            <div className="flex items-center gap-1.5 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs font-mono">
              <span className="text-slate-400 text-[11px]">Scenario:</span>
              <select
                value={selectedScenarioIndex}
                onChange={(e) => handleSelectScenario(Number(e.target.value))}
                className="bg-transparent text-emerald-300 font-semibold focus:outline-hidden cursor-pointer"
              >
                {DEFAULT_SCENARIOS.map((sc, idx) => (
                  <option key={sc.id} value={idx} className="bg-slate-900 text-slate-200">
                    {sc.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Play/Pause & Reset */}
            <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-0.5">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="p-1.5 rounded hover:bg-slate-800 text-slate-300 hover:text-white transition-colors"
                title={isPlaying ? 'Pause Simulation' : 'Resume Simulation'}
              >
                {isPlaying ? <Pause className="w-4 h-4 text-amber-400 fill-current" /> : <Play className="w-4 h-4 text-emerald-400 fill-current" />}
              </button>
              <button
                onClick={() => resetSimulation()}
                className="p-1.5 rounded hover:bg-slate-800 text-slate-300 hover:text-white transition-colors"
                title="Reset Scenario"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>

            {/* Sim Speed Selector */}
            <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300 p-0.5">
              {[0.5, 1.0, 2.0].map((s) => (
                <button
                  key={s}
                  onClick={() => setSimSpeed(s)}
                  className={`px-2 py-1 rounded text-[11px] font-semibold transition-colors ${
                    simSpeed === s ? 'bg-slate-800 text-emerald-400' : 'hover:text-white'
                  }`}
                >
                  {s}x
                </button>
              ))}
            </div>

            {/* Audio Toggle */}
            <button
              onClick={handleToggleAudio}
              className={`p-2 rounded-lg border text-xs font-mono transition-colors ${
                isAudioMuted
                  ? 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                  : 'bg-emerald-950/60 border-emerald-500/50 text-emerald-300'
              }`}
              title={isAudioMuted ? 'Unmute Tactical EW Radar Audio' : 'Mute Audio Alerts'}
            >
              {isAudioMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4 animate-pulse" />}
            </button>

            {/* Research & Benchmark Suite Triggers */}
            <div className="hidden lg:flex items-center gap-1.5">
              <button
                onClick={() => setIsAblationModalOpen(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-950/80 hover:bg-indigo-900/90 text-indigo-300 border border-indigo-500/40 font-mono text-xs font-semibold transition-all shadow-md"
                title="Systematic Component Ablation Matrix"
              >
                <Layers className="w-3.5 h-3.5" /> Ablation Study
              </button>

              <button
                onClick={() => setIsParetoModalOpen(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-emerald-400 border border-emerald-500/40 font-mono text-xs font-semibold transition-all shadow-md"
                title="Multi-Objective Pareto Trade-off Frontier"
              >
                <TrendingUp className="w-3.5 h-3.5" /> Pareto Frontier
              </button>

              <button
                onClick={() => setIsBenchmarkModalOpen(true)}
                className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-mono text-xs font-bold transition-all shadow-lg shadow-emerald-950/50"
              >
                <Award className="w-4 h-4" /> Run Automated Test Suite
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Floating Tactical Notification Toast */}
      {activeToast && (
        <div className="fixed top-16 right-4 z-50 bg-slate-900/95 border border-emerald-500/50 text-emerald-300 text-xs font-mono px-4 py-2 rounded-lg shadow-2xl flex items-center gap-2 animate-bounce">
          <Zap className="w-4 h-4 text-emerald-400" />
          <span>{activeToast}</span>
        </div>
      )}

      {/* Main Tactical Dashboard Body */}
      <main className="max-w-7xl mx-auto w-full p-4 flex-1 space-y-4">
        {/* Advanced Research Architecture & Adversarial Tactic Controls */}
        <div className="p-3.5 bg-slate-900/90 border border-slate-800 rounded-lg flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3 text-xs font-mono">
          <div className="flex flex-wrap items-center gap-3">
            {/* Bayesian Model Toggle */}
            <div className="flex items-center gap-1.5 bg-slate-950 border border-slate-800 rounded px-2 py-1">
              <span className="text-slate-400 text-[11px]">Bayesian Engine:</span>
              <button
                onClick={() => handleBayesianModeChange('discounted_beta')}
                className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                  bayesianMode === 'discounted_beta'
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Discounted Beta (ρ=0.96)
              </button>
              <button
                onClick={() => handleBayesianModeChange('exact_beta')}
                className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                  bayesianMode === 'exact_beta'
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Exact Beta
              </button>
            </div>

            {/* Hazard Function Toggle */}
            <div className="flex items-center gap-1.5 bg-slate-950 border border-slate-800 rounded px-2 py-1">
              <span className="text-slate-400 text-[11px]">Recurrence Hazard:</span>
              {(['weibull', 'exponential', 'empirical'] as HazardModelMode[]).map((mode) => (
                <button
                  key={mode}
                  onClick={() => handleHazardModeChange(mode)}
                  className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${
                    hazardMode === mode
                      ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>

          {/* Adversarial Evasion Tactic Trigger */}
          <div className="flex items-center gap-2">
            <span className="text-amber-400 text-[11px] font-bold">Threat Evasion:</span>
            <select
              value={selectedTactic}
              onChange={(e) => setSelectedTactic(e.target.value as ReactiveEvasionType)}
              className="bg-slate-950 border border-slate-800 rounded text-amber-300 text-[11px] font-mono px-2 py-1 focus:outline-hidden"
            >
              <option value="typeA_freq_shift">Type A: Frequency Hop Shift (+Bins)</option>
              <option value="typeB_pri_stagger">Type B: PRI Jitter & Stagger</option>
              <option value="typeC_pw_reduction">Type C: Pulse Width Reduction</option>
              <option value="typeD_silence_burst">Type D: Radar Silence Blanking</option>
              <option value="typeE_decoy_emission">Type E: Ghost Decoy Pulse</option>
              <option value="typeF_deceptive_markov">Type F: Deceptive Markov Inversion</option>
              <option value="typeG_randomization">Type G: Uniform Randomization</option>
            </select>
            <button
              onClick={handleTriggerSpecificTactic}
              className="px-2.5 py-1 bg-amber-600 hover:bg-amber-500 text-white rounded text-[11px] font-bold transition-colors shadow-md"
            >
              Execute Tactic
            </button>
            <button
              onClick={handleInjectNovelThreat}
              className="px-2.5 py-1 bg-rose-700 hover:bg-rose-600 text-white rounded text-[11px] font-bold transition-colors shadow-md"
            >
              Inject Unknown Radar
            </button>
          </div>
        </div>

        {/* 1. Strict Requirement Head-To-Head Verification Card */}
        <HeadToHeadComparison seqMetrics={seqMetrics} smartMetrics={smartMetrics} />

        {/* 1b. Physical Receiver Front-End Telemetry */}
        <PhysicalReceiverTelemetry
          smartFrontEndState={smartFrontEndState}
          seqMetrics={seqMetrics}
          smartMetrics={smartMetrics}
        />

        {/* 2. High-Resolution RF Waterfall Spectrogram */}
        <SpectrumWaterfall
          pulses={currentPulses}
          seqAction={seqAction}
          smartAction={smartAction}
          recentSeqHits={recentSeqHits}
          recentSmartHits={recentSmartHits}
          currentTimeMs={currentTimeMs}
        />

        {/* 3. Tactical 32-Channel Belief & Coverage Shield Map */}
        <TacticalSpectrumView
          beliefs={channelBeliefs}
          activePulses={currentPulses}
          seqAction={seqAction}
          smartAction={smartAction}
          shieldActive={isShieldActive}
          onToggleShield={handleToggleShield}
        />

        {/* 4. Dual Grid: Sub-Millisecond Latency Profiler & Learned Markov Matrix */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <HighLoadLatencyMonitor breakdown={latencyBreakdown} metrics={smartMetrics} />
          <TransitionMatrixView transitionMatrix={transitionMatrix} />
        </div>

        {/* 5. Active Emitter Threat Inventory & Intercept Table */}
        <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 shadow-xl text-xs font-mono">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800 mb-2">
            <span className="font-semibold text-slate-300 uppercase tracking-wider text-[11px]">
              Active Emitter Threat Table (Truth vs Interception)
            </span>
            <span className="text-slate-400 text-[11px]">
              Simulation Time: {(currentTimeMs / 1000).toFixed(2)} s
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="text-slate-400 border-b border-slate-800">
                  <th className="pb-1.5 font-semibold">Threat Name</th>
                  <th className="pb-1.5 font-semibold">Type</th>
                  <th className="pb-1.5 font-semibold">Band / Channels</th>
                  <th className="pb-1.5 font-semibold">PRI</th>
                  <th className="pb-1.5 font-semibold">SNR</th>
                  <th className="pb-1.5 font-semibold">Priority</th>
                  <th className="pb-1.5 font-semibold">Smart Intercept Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-200">
                {simRef.current.emitters.map((em) => {
                  const isCurActive = currentPulses.some((p) => p.emitterId === em.id);
                  const isSmartTracking = smartAction && em.hoppingSequence.includes(smartAction.channelIndex);

                  return (
                    <tr key={em.id} className="hover:bg-slate-900/40">
                      <td className="py-1.5 font-bold flex items-center gap-2">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            isCurActive ? 'bg-rose-500 animate-ping' : 'bg-slate-600'
                          }`}
                        />
                        {em.name}
                      </td>
                      <td className="py-1.5 text-slate-400 uppercase text-[11px]">{em.type.replace('_', ' ')}</td>
                      <td className="py-1.5 text-sky-400">
                        {em.hoppingSequence.slice(0, 4).join(', ')}
                        {em.hoppingSequence.length > 4 ? '...' : ''}
                      </td>
                      <td className="py-1.5 text-slate-300">{em.priMs.toFixed(1)} ms</td>
                      <td className="py-1.5 text-slate-300">{em.snrDb} dB</td>
                      <td className="py-1.5">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                            em.priority >= 5
                              ? 'bg-rose-500/20 text-rose-300'
                              : 'bg-amber-500/20 text-amber-300'
                          }`}
                        >
                          P{em.priority}
                        </span>
                      </td>
                      <td className="py-1.5">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            isSmartTracking
                              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                              : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          {isSmartTracking ? 'LOCKED / INTERCEPTING' : 'MONITORING'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </main>

      {/* Automated Benchmark & Verification Modal */}
      <BenchmarkRunnerModal
        isOpen={isBenchmarkModalOpen}
        onClose={() => setIsBenchmarkModalOpen(false)}
      />

      {/* Component Ablation Study Modal */}
      <AblationStudyModal
        isOpen={isAblationModalOpen}
        onClose={() => setIsAblationModalOpen(false)}
      />

      {/* Multi-Objective Pareto Frontier Modal */}
      <ParetoFrontierModal
        isOpen={isParetoModalOpen}
        onClose={() => setIsParetoModalOpen(false)}
      />
    </div>
  );
}
