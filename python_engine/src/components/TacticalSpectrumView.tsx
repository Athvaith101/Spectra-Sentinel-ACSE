/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ChannelBelief, GroundTruthPulse, ReceiverAction } from '../types';
import { ShieldAlert, Zap, Radio, Activity } from 'lucide-react';

interface Props {
  beliefs: ChannelBelief[];
  activePulses: GroundTruthPulse[];
  seqAction: ReceiverAction | null;
  smartAction: ReceiverAction | null;
  shieldActive: boolean;
  onToggleShield: () => void;
}

export const TacticalSpectrumView: React.FC<Props> = ({
  beliefs,
  activePulses,
  seqAction,
  smartAction,
  shieldActive,
  onToggleShield,
}) => {
  const [selectedChannel, setSelectedChannel] = useState<number | null>(null);

  const activeChannelMap = new Set(activePulses.map((p) => p.channelIndex));

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 shadow-xl">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800 mb-2">
        <div className="flex items-center gap-2">
          <Radio className="w-4 h-4 text-emerald-400" />
          <span className="text-xs font-semibold tracking-wider text-slate-200 uppercase font-mono">
            32-Channel Tactical Belief & Coverage Shield Map
          </span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={onToggleShield}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-mono font-medium transition-colors border ${
              shieldActive
                ? 'bg-emerald-950/60 border-emerald-500/50 text-emerald-300'
                : 'bg-rose-950/40 border-rose-500/50 text-rose-300'
            }`}
          >
            <ShieldAlert className="w-3.5 h-3.5" />
            Coverage Shield: {shieldActive ? 'ACTIVE (180ms)' : 'BYPASS'}
          </button>
        </div>
      </div>

      {/* 32 Channel Bars Grid */}
      <div
        style={{ display: 'grid', gridTemplateColumns: 'repeat(32, minmax(0, 1fr))' }}
        className="gap-1 items-end h-32 bg-slate-900/60 p-2 rounded border border-slate-800/80 mb-2 overflow-x-auto"
      >
        {beliefs.map((b) => {
          const isEmitting = activeChannelMap.has(b.channelIndex);
          const isSeqTuned = seqAction?.channelIndex === b.channelIndex;
          const isSmartTuned = smartAction?.channelIndex === b.channelIndex;
          const isSelected = selectedChannel === b.channelIndex;

          const occupancyHeightPct = Math.max(8, Math.min(100, b.occupancyProb * 100));

          let barBg = 'bg-slate-700/60';
          if (isEmitting) {
            barBg = 'bg-rose-500 animate-pulse';
          } else if (b.occupancyProb > 0.4) {
            barBg = 'bg-emerald-500';
          } else if (b.shieldAlertLevel === 'red') {
            barBg = 'bg-amber-500';
          }

          return (
            <div
              key={b.channelIndex}
              onClick={() => setSelectedChannel(b.channelIndex)}
              className={`relative flex flex-col justify-end h-full w-full cursor-pointer rounded-xs transition-all ${
                isSelected ? 'ring-2 ring-emerald-400' : ''
              } ${isSmartTuned ? 'bg-emerald-950/40' : ''}`}
              title={`CH ${b.channelIndex} (${b.freqMHz} MHz)\nBelief: ${(b.occupancyProb * 100).toFixed(0)}%\nStaleness: ${b.stalenessMs.toFixed(0)}ms\nHits: ${b.totalHits}`}
            >
              {/* Tuner Indicator Flags */}
              {isSmartTuned && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_#10b981]" />
              )}
              {isSeqTuned && (
                <div className="absolute -top-5 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-sky-400" />
              )}

              {/* Bar Fill */}
              <div
                style={{ height: `${occupancyHeightPct}%` }}
                className={`w-full rounded-xs transition-all duration-150 ${barBg}`}
              />

              {/* Staleness indicator dot at bottom */}
              <div
                className={`w-full h-1 mt-0.5 rounded-xs ${
                  b.shieldAlertLevel === 'red'
                    ? 'bg-rose-500'
                    : b.shieldAlertLevel === 'yellow'
                    ? 'bg-amber-400'
                    : 'bg-slate-700'
                }`}
              />
            </div>
          );
        })}
      </div>

      {/* Channel inspector details */}
      <div className="flex flex-wrap items-center justify-between text-[11px] font-mono text-slate-400 pt-1">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" /> High Belief
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-rose-500 inline-block" /> Active Pulse
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-amber-400 inline-block" /> Stale (Shield Warning)
          </span>
        </div>

        {selectedChannel !== null ? (
          <div className="flex items-center gap-3 text-slate-200">
            <span className="font-bold text-emerald-400">CH {selectedChannel} ({beliefs[selectedChannel]?.freqMHz} MHz):</span>
            <span>Occupancy Belief: {(beliefs[selectedChannel]?.occupancyProb * 100).toFixed(0)}%</span>
            <span>Staleness: {beliefs[selectedChannel]?.stalenessMs.toFixed(0)}ms</span>
            <span>Opportunity Cost: {beliefs[selectedChannel]?.opportunityCost.toFixed(2)}</span>
            <span>Hits: {beliefs[selectedChannel]?.totalHits}</span>
          </div>
        ) : (
          <span className="text-slate-400">Click any channel above to inspect belief & opportunity parameters</span>
        )}
      </div>
    </div>
  );
};
