/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import {
  CognitiveOperatingMode,
  ComponentGates,
  CompositeMissionScore,
  AdaptationMetrics,
  ScenarioMissionMode,
} from '../types';
import {
  Activity,
  Compass,
  CheckCircle2,
  Shield,
  RefreshCw,
  Gauge,
  Zap,
  Target,
} from 'lucide-react';

interface CognitiveMetaPolicyHUDProps {
  operatingMode: CognitiveOperatingMode;
  gates: ComponentGates;
  compositeScore: CompositeMissionScore;
  adaptationMetrics: AdaptationMetrics;
  missionMode: ScenarioMissionMode;
}

export const CognitiveMetaPolicyHUD: React.FC<CognitiveMetaPolicyHUDProps> = ({
  operatingMode,
  gates,
  compositeScore,
  adaptationMetrics,
  missionMode,
}) => {
  const modeBadge = () => {
    switch (operatingMode) {
      case 'EXPLOIT':
        return {
          label: 'EXPLOIT (Fast Intercept Path)',
          desc: 'Original aggressive intercept scheduler active. Complex cognitive modules bypassed.',
          bg: 'bg-emerald-950/70 border-emerald-500/50 text-emerald-300',
          dot: 'bg-emerald-400 animate-pulse',
          icon: Target,
        };
      case 'EXPLORE':
        return {
          label: 'EXPLORE (Epistemic Information Gain)',
          desc: 'Deliberate exploration triggered by high epistemic uncertainty or OOD anomaly.',
          bg: 'bg-blue-950/70 border-blue-500/50 text-blue-300',
          dot: 'bg-blue-400 animate-pulse',
          icon: Compass,
        };
      case 'VERIFY':
        return {
          label: 'VERIFY (Confirmation Burst)',
          desc: 'Target probability exceeded 70%. Executing rapid multi-dwell confirmation burst.',
          bg: 'bg-purple-950/70 border-purple-500/50 text-purple-300',
          dot: 'bg-purple-400 animate-ping',
          icon: CheckCircle2,
        };
      case 'COVERAGE':
        return {
          label: 'COVERAGE (Safety Shield Override)',
          desc: 'Fleet staleness deadline exceeded. Hard override to prevent blind spots.',
          bg: 'bg-amber-950/70 border-amber-500/50 text-amber-300',
          dot: 'bg-amber-400 animate-pulse',
          icon: Shield,
        };
      case 'RECOVER':
        return {
          label: 'RECOVER (Distribution Shift)',
          desc: 'Page-Hinkley change-point detected. Memory discounted; measuring adaptation latency T_adapt.',
          bg: 'bg-rose-950/70 border-rose-500/50 text-rose-300',
          dot: 'bg-rose-400 animate-pulse',
          icon: RefreshCw,
        };
    }
  };

  const badge = modeBadge();
  const IconComp = badge.icon;

  const formatGate = (val: number) => (val * 100).toFixed(0) + '%';

  return (
    <section
      id="cognitive-meta-hud"
      aria-label="Cognitive Meta-Policy & Evidence Gating Control Panel"
      className="bg-slate-900/90 border border-slate-800 rounded-lg p-3 text-xs font-mono shadow-md"
    >
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-3 pb-2.5 border-b border-slate-800/80">
        {/* Active Mode Banner */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-slate-400 text-[11px] uppercase tracking-wider font-semibold">
              Cognitive Policy:
            </span>
            <div
              id="active-operating-mode-badge"
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs font-bold transition-all ${badge.bg}`}
            >
              <span className={`w-2 h-2 rounded-full ${badge.dot}`} />
              <IconComp className="w-3.5 h-3.5" />
              <span>{badge.label}</span>
            </div>
          </div>

          {gates.isFastPathActive && (
            <span
              id="fast-path-indicator"
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-500/30 text-emerald-400 text-[10px] font-semibold"
            >
              <Zap className="w-3 h-3" /> Fast-Path Bypassed Heavy Reasoning
            </span>
          )}
        </div>

        {/* Mission Conditioning & Composite Utility */}
        <div className="flex items-center gap-4 text-[11px]">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400">Mission:</span>
            <span className="text-slate-200 capitalize font-medium">
              {missionMode.replace('_', ' ')}
            </span>
          </div>

          <div
            id="composite-mission-score-box"
            className="flex items-center gap-1.5 px-2 py-1 bg-slate-950 border border-slate-800 rounded text-slate-200"
            title="Composite Mission Utility J = 0.30*(1-NMIT) + 0.25*Pd + 0.15*Coverage + 0.15*(1-PHPmiss) + 0.10*IG - 0.03*FA - 0.02*Latency"
          >
            <Gauge className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">Composite Mission J:</span>
            <span className="font-bold text-cyan-300">
              {(compositeScore.totalUtilityJ * 100).toFixed(1)}%
            </span>
          </div>

          {adaptationMetrics.totalChangePoints > 0 && (
            <div
              id="adaptation-latency-box"
              className="flex items-center gap-1.5 px-2 py-1 bg-slate-950 border border-slate-800 rounded text-amber-300"
              title="Mean adaptation time to recover intercept performance after distribution shift"
            >
              <RefreshCw className="w-3 h-3 text-amber-400" />
              <span className="text-slate-400">T_adapt:</span>
              <span className="font-bold">
                {adaptationMetrics.meanAdaptationTimeMs.toFixed(1)} ms
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Gating Vector & Composite Sub-Metric Breakdown */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 pt-2.5">
        {/* Gate: PU */}
        <div
          id="gate-pu"
          className="bg-slate-950/80 border border-slate-800/80 rounded p-1.5 flex flex-col justify-between"
          title="Prediction Usefulness Gate: High prediction confidence enables predictor, low falls back to exploit"
        >
          <div className="text-[10px] text-slate-400 flex items-center justify-between">
            <span>g_PU (Predict)</span>
            <span className="text-slate-300 font-bold">{formatGate(gates.gPu)}</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-1 overflow-hidden">
            <div
              className="bg-emerald-400 h-full transition-all duration-300"
              style={{ width: `${Math.min(100, gates.gPu * 100)}%` }}
            />
          </div>
        </div>

        {/* Gate: Hazard */}
        <div
          id="gate-hazard"
          className="bg-slate-950/80 border border-slate-800/80 rounded p-1.5 flex flex-col justify-between"
          title="Hazard Gate: Activates opportunity cost reasoning when threat recurrence hazard is imminent"
        >
          <div className="text-[10px] text-slate-400 flex items-center justify-between">
            <span>g_hazard</span>
            <span className="text-slate-300 font-bold">{formatGate(gates.gHazard)}</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-1 overflow-hidden">
            <div
              className="bg-amber-400 h-full transition-all duration-300"
              style={{ width: `${Math.min(100, gates.gHazard * 100)}%` }}
            />
          </div>
        </div>

        {/* Gate: OOD */}
        <div
          id="gate-ood"
          className="bg-slate-950/80 border border-slate-800/80 rounded p-1.5 flex flex-col justify-between"
          title="OOD Gate: Triggers novelty exploration and prediction fallback on anomaly"
        >
          <div className="text-[10px] text-slate-400 flex items-center justify-between">
            <span>g_OOD</span>
            <span className="text-slate-300 font-bold">{formatGate(gates.gOod)}</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-1 overflow-hidden">
            <div
              className="bg-rose-400 h-full transition-all duration-300"
              style={{ width: `${Math.min(100, gates.gOod * 100)}%` }}
            />
          </div>
        </div>

        {/* Gate: Coverage */}
        <div
          id="gate-coverage"
          className="bg-slate-950/80 border border-slate-800/80 rounded p-1.5 flex flex-col justify-between"
          title="Coverage Gate: Enforces strict surveillance deadline when channels approach staleness limits"
        >
          <div className="text-[10px] text-slate-400 flex items-center justify-between">
            <span>g_coverage</span>
            <span className="text-slate-300 font-bold">{formatGate(gates.gCoverage)}</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-1 overflow-hidden">
            <div
              className="bg-indigo-400 h-full transition-all duration-300"
              style={{ width: `${Math.min(100, gates.gCoverage * 100)}%` }}
            />
          </div>
        </div>

        {/* Gate: Information Gain */}
        <div
          id="gate-ig"
          className="bg-slate-950/80 border border-slate-800/80 rounded p-1.5 flex flex-col justify-between"
          title="Information Gain Gate: Gated epistemic exploration"
        >
          <div className="text-[10px] text-slate-400 flex items-center justify-between">
            <span>g_IG</span>
            <span className="text-slate-300 font-bold">{formatGate(gates.gIg)}</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-1 overflow-hidden">
            <div
              className="bg-cyan-400 h-full transition-all duration-300"
              style={{ width: `${Math.min(100, gates.gIg * 100)}%` }}
            />
          </div>
        </div>

        {/* Gate: Multi-Hypothesis Distribution */}
        <div
          id="gate-mht"
          className="bg-slate-950/80 border border-slate-800/80 rounded p-1.5 flex flex-col justify-between"
          title="Full MHT Continuous Probability Distribution: Weighted integration across all 32 channels"
        >
          <div className="text-[10px] text-slate-400 flex items-center justify-between">
            <span>g_MHT</span>
            <span className="text-slate-300 font-bold">{formatGate(gates.gMht)}</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-1 overflow-hidden">
            <div
              className="bg-purple-400 h-full transition-all duration-300"
              style={{ width: `${Math.min(100, gates.gMht * 100)}%` }}
            />
          </div>
        </div>

        {/* Gate: Change-Point */}
        <div
          id="gate-cp"
          className="bg-slate-950/80 border border-slate-800/80 rounded p-1.5 flex flex-col justify-between"
          title="Change Point Detection Gate: Inverts stale transition memory and triggers recovery"
        >
          <div className="text-[10px] text-slate-400 flex items-center justify-between">
            <span>g_CP</span>
            <span className="text-slate-300 font-bold">{formatGate(gates.gCp)}</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-1 overflow-hidden">
            <div
              className="bg-orange-400 h-full transition-all duration-300"
              style={{ width: `${Math.min(100, gates.gCp * 100)}%` }}
            />
          </div>
        </div>
      </div>
    </section>
  );
};
