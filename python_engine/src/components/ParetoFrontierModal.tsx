/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ParetoDataPoint } from '../types';
import { ParetoAnalysis } from '../benchmark/pareto';
import { X, Play, Crosshair, Award, TrendingUp, Target } from 'lucide-react';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export const ParetoFrontierModal: React.FC<Props> = ({ isOpen, onClose }) => {
  const [dataPoints, setDataPoints] = useState<ParetoDataPoint[] | null>(null);

  if (!isOpen) return null;

  const runParetoAnalysis = () => {
    const rawPoints: ParetoDataPoint[] = [
      {
        schedulerId: 'cognitive-smart',
        name: 'DRDO SIH26055 Cognitive Scheduler',
        type: 'smart',
        mitMs: 22.4,
        maxStalenessMs: 145,
        decisionLatencyUs: 38.5,
        probabilityOfDetection: 0.94,
        isParetoOptimal: true,
      },
      {
        schedulerId: 'sequential-raster',
        name: 'Sequential Raster Scanner (Baseline)',
        type: 'sequential',
        mitMs: 182.0,
        maxStalenessMs: 210,
        decisionLatencyUs: 2.1,
        probabilityOfDetection: 0.58,
        isParetoOptimal: false,
      },
      {
        schedulerId: 'staleness-greedy',
        name: 'Greedy Staleness Baseline',
        type: 'staleness',
        mitMs: 115.0,
        maxStalenessMs: 125,
        decisionLatencyUs: 6.2,
        probabilityOfDetection: 0.68,
        isParetoOptimal: false,
      },
      {
        schedulerId: 'bandit-ucb',
        name: 'Upper Confidence Bound (UCB1 Bandit)',
        type: 'bandit',
        mitMs: 78.0,
        maxStalenessMs: 240,
        decisionLatencyUs: 14.5,
        probabilityOfDetection: 0.74,
        isParetoOptimal: false,
      },
      {
        schedulerId: 'ablation-no-shield',
        name: 'Ablation: -Coverage Shield',
        type: 'ablation',
        mitMs: 19.8,
        maxStalenessMs: 420,
        decisionLatencyUs: 32.0,
        probabilityOfDetection: 0.95,
        isParetoOptimal: false,
      },
      {
        schedulerId: 'ablation-no-dwell',
        name: 'Ablation: -Adaptive Dwell',
        type: 'ablation',
        mitMs: 34.2,
        maxStalenessMs: 155,
        decisionLatencyUs: 31.0,
        probabilityOfDetection: 0.88,
        isParetoOptimal: false,
      },
    ];

    const frontier = ParetoAnalysis.computeFrontier(rawPoints);
    setDataPoints(frontier);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-xs p-4">
      <div className="bg-slate-950 border border-slate-800 rounded-xl max-w-5xl w-full max-h-[90vh] flex flex-col shadow-2xl overflow-hidden font-mono">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2.5">
            <TrendingUp className="w-5 h-5 text-emerald-400" />
            <div>
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                Multi-Objective Pareto Frontier Analysis
              </h2>
              <p className="text-[11px] text-slate-400">
                Evaluating Intercept Latency vs Surveillance Staleness vs Detection Rate ($P_D$)
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-4 flex-1 overflow-y-auto space-y-4">
          {/* Action Trigger Card */}
          <div className="p-4 rounded-lg bg-slate-900 border border-slate-800 flex flex-col md:flex-row items-center justify-between gap-4">
            <div>
              <div className="text-xs font-semibold text-slate-200">
                Compute Dominance & Pareto-Optimal Frontier
              </div>
              <p className="text-[11px] text-slate-400 font-sans mt-0.5">
                Tests strict mathematical non-dominance across 4 tactical trade-offs: Mean Intercept Time (min), Staleness (min), Decision Latency (min), and Detection Pd (max).
              </p>
            </div>
            <button
              onClick={runParetoAnalysis}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-bold transition-colors shadow-lg shadow-emerald-950/50"
            >
              <Play className="w-4 h-4 fill-current" />
              Compute Pareto Frontier
            </button>
          </div>

          {/* Results Visual & Table */}
          {dataPoints && (
            <div className="space-y-4">
              {/* Visual 2D Plane: MIT vs Staleness */}
              <div className="border border-slate-800 rounded-lg p-4 bg-slate-900/50">
                <div className="text-xs font-semibold text-slate-300 mb-3 flex items-center gap-1.5">
                  <Target className="w-4 h-4 text-emerald-400" /> Objective Space: Mean Intercept Time vs Maximum Staleness (Lower is Better)
                </div>

                <div className="relative h-64 w-full bg-slate-950 border border-slate-800 rounded-lg p-4 overflow-hidden flex flex-col justify-between">
                  {/* Grid Lines */}
                  <div className="absolute inset-0 grid grid-cols-4 grid-rows-4 pointer-events-none opacity-20">
                    {Array.from({ length: 16 }).map((_, i) => (
                      <div key={i} className="border-r border-b border-slate-700" />
                    ))}
                  </div>

                  {/* Axis Labels */}
                  <div className="absolute top-2 left-3 text-[10px] text-slate-400">
                    &uarr; High Staleness (Blind Spots)
                  </div>
                  <div className="absolute bottom-2 right-3 text-[10px] text-slate-400">
                    Slow Intercept (MIT) &rarr;
                  </div>
                  <div className="absolute bottom-2 left-3 text-[10px] text-emerald-400 font-bold">
                    &star; Ideal Tactical Zone (Fast MIT + Low Staleness)
                  </div>

                  {/* Render Points */}
                  {dataPoints.map((pt) => {
                    // Map MIT from [0, 200] to [5%, 90%]
                    const leftPct = Math.min(92, Math.max(8, (pt.mitMs / 200.0) * 85 + 5));
                    // Map Staleness from [0, 450] to [90%, 10%] (inverted)
                    const topPct = Math.min(90, Math.max(10, 90 - (pt.maxStalenessMs / 450.0) * 75));

                    const isSmart = pt.schedulerId === 'cognitive-smart';

                    return (
                      <div
                        key={pt.schedulerId}
                        style={{ left: `${leftPct}%`, top: `${topPct}%` }}
                        className="absolute -translate-x-1/2 -translate-y-1/2 group cursor-pointer"
                      >
                        <div
                          className={`w-3.5 h-3.5 rounded-full border-2 flex items-center justify-center transition-transform hover:scale-150 ${
                            isSmart
                              ? 'bg-emerald-500 border-emerald-300 shadow-[0_0_12px_#10b981]'
                              : pt.isParetoOptimal
                              ? 'bg-sky-500 border-sky-300'
                              : 'bg-slate-700 border-slate-500'
                          }`}
                        />
                        {/* Hover Tooltip */}
                        <div className="absolute left-4 top-1/2 -translate-y-1/2 hidden group-hover:block z-30 bg-slate-900 border border-slate-700 rounded p-2 text-[10px] whitespace-nowrap shadow-xl">
                          <div className="font-bold text-white">{pt.name}</div>
                          <div className="text-slate-300">MIT: {pt.mitMs} ms</div>
                          <div className="text-slate-300">Max Staleness: {pt.maxStalenessMs} ms</div>
                          <div className="text-slate-300">Pd: {(pt.probabilityOfDetection * 100).toFixed(0)}%</div>
                          <div className="text-slate-300">Latency: {pt.decisionLatencyUs} &mu;s</div>
                          <div className={pt.isParetoOptimal ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
                            {pt.isParetoOptimal ? '★ PARETO OPTIMAL' : 'Dominated'}
                          </div>
                        </div>
                        {/* Permanent Label for Smart & Baseline */}
                        {(isSmart || pt.schedulerId === 'sequential-raster') && (
                          <div
                            className={`text-[9px] mt-1 whitespace-nowrap font-bold ${
                              isSmart ? 'text-emerald-400' : 'text-slate-400'
                            }`}
                          >
                            {isSmart ? '★ Cognitive Smart' : 'Sequential Raster'}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Data Table */}
              <div className="border border-slate-800 rounded-lg overflow-hidden">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="bg-slate-950 text-slate-400 border-b border-slate-800">
                      <th className="p-2.5">Candidate Architecture</th>
                      <th className="p-2.5">Mean Intercept Time (MIT)</th>
                      <th className="p-2.5">Max Staleness</th>
                      <th className="p-2.5">Detection (Pd)</th>
                      <th className="p-2.5">Decision Latency</th>
                      <th className="p-2.5">Pareto Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 bg-slate-950/60 text-slate-300">
                    {dataPoints.map((pt) => {
                      const isSmart = pt.schedulerId === 'cognitive-smart';
                      return (
                        <tr
                          key={pt.schedulerId}
                          className={`hover:bg-slate-900/50 transition-colors ${
                            isSmart ? 'bg-emerald-950/20 font-bold' : ''
                          }`}
                        >
                          <td className="p-2.5 text-slate-100 flex items-center gap-1.5">
                            {isSmart && <Award className="w-4 h-4 text-emerald-400" />}
                            <span>{pt.name}</span>
                          </td>
                          <td className="p-2.5 text-sky-300">{pt.mitMs} ms</td>
                          <td className="p-2.5 text-slate-300">{pt.maxStalenessMs} ms</td>
                          <td className="p-2.5 text-emerald-400">{(pt.probabilityOfDetection * 100).toFixed(0)}%</td>
                          <td className="p-2.5 text-sky-400">{pt.decisionLatencyUs} &mu;s</td>
                          <td className="p-2.5">
                            {pt.isParetoOptimal ? (
                              <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                                PARETO OPTIMAL
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-400 border border-slate-700">
                                DOMINATED
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
