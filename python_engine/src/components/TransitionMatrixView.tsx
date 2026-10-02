/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { TOTAL_CHANNELS, getChannelCenterFreq } from '../engine/rfSimulator';
import { Network, Info } from 'lucide-react';

interface Props {
  transitionMatrix: Float32Array[];
}

export const TransitionMatrixView: React.FC<Props> = ({ transitionMatrix }) => {
  const [hoveredCell, setHoveredCell] = useState<{ from: number; to: number; prob: number } | null>(null);

  // Find top learned transitions
  const topTransitions: Array<{ from: number; to: number; prob: number }> = [];
  if (transitionMatrix && transitionMatrix.length === TOTAL_CHANNELS) {
    for (let i = 0; i < TOTAL_CHANNELS; i++) {
      for (let j = 0; j < TOTAL_CHANNELS; j++) {
        const prob = transitionMatrix[i][j];
        if (prob > 0.25) {
          topTransitions.push({ from: i, to: j, prob });
        }
      }
    }
  }
  topTransitions.sort((a, b) => b.prob - a.prob);

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 shadow-xl flex flex-col">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800 mb-2">
        <div className="flex items-center gap-2">
          <Network className="w-4 h-4 text-sky-400" />
          <span className="text-xs font-semibold tracking-wider text-slate-200 uppercase font-mono">
            Learned Markov Transition Matrix P(f_next | f_curr)
          </span>
        </div>
        <span className="text-[11px] font-mono text-emerald-400">
          Online Hebbian / Dirichlet Learning Active
        </span>
      </div>

      <div className="flex flex-col md:flex-row gap-4 items-center">
        {/* 32x32 Heatmap Canvas / SVG grid */}
        <div className="relative border border-slate-800 bg-slate-900/80 rounded p-1">
          <div
            style={{ display: 'grid', gridTemplateColumns: 'repeat(32, minmax(0, 1fr))' }}
            className="gap-[1px] w-56 h-56"
          >
            {transitionMatrix &&
              transitionMatrix.map((row, fromIdx) =>
                Array.from(row).map((prob, toIdx) => {
                  let bg = 'bg-slate-950';
                  if (prob > 0.6) bg = 'bg-emerald-400';
                  else if (prob > 0.35) bg = 'bg-emerald-600';
                  else if (prob > 0.15) bg = 'bg-sky-600';
                  else if (prob > 0.05) bg = 'bg-slate-800';

                  return (
                    <div
                      key={`${fromIdx}-${toIdx}`}
                      onMouseEnter={() => setHoveredCell({ from: fromIdx, to: toIdx, prob })}
                      onMouseLeave={() => setHoveredCell(null)}
                      className={`w-full h-full cursor-pointer transition-colors ${bg}`}
                    />
                  );
                })
              )}
          </div>
        </div>

        {/* Top Discovered Hopping Rules & Inspector */}
        <div className="flex-1 w-full text-xs font-mono">
          <div className="p-2.5 rounded bg-slate-900 border border-slate-800 mb-2 text-slate-300">
            {hoveredCell ? (
              <div>
                <span className="text-emerald-400 font-bold">
                  Transition CH {hoveredCell.from} &rarr; CH {hoveredCell.to}
                </span>
                <div className="text-[11px] text-slate-400 mt-0.5">
                  {getChannelCenterFreq(hoveredCell.from)} MHz &rarr; {getChannelCenterFreq(hoveredCell.to)} MHz
                </div>
                <div className="text-white font-bold mt-1">
                  Learned Probability: {(hoveredCell.prob * 100).toFixed(1)}%
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 text-slate-400">
                <Info className="w-3.5 h-3.5 text-sky-400 shrink-0" />
                <span>Hover cells to inspect learned frequency hop transition probabilities.</span>
              </div>
            )}
          </div>

          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
            Top Discovered Hopping Transitions:
          </div>
          <div className="space-y-1 max-h-24 overflow-y-auto pr-1">
            {topTransitions.slice(0, 5).map((t, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between px-2 py-1 rounded bg-slate-900/60 border border-slate-800/80 text-[11px]"
              >
                <span className="text-slate-300">
                  CH <b className="text-sky-300">{t.from}</b> &rarr; CH <b className="text-emerald-300">{t.to}</b>
                </span>
                <div className="flex items-center gap-2">
                  <div className="w-16 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      style={{ width: `${Math.min(100, t.prob * 100)}%` }}
                      className="bg-emerald-400 h-full"
                    />
                  </div>
                  <span className="text-emerald-400 font-bold w-10 text-right">
                    {(t.prob * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
            {topTransitions.length === 0 && (
              <div className="text-slate-400 italic text-[11px]">
                Observing initial pulse stream... building Markov transition weights.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
