/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef } from 'react';
import { GroundTruthPulse, ReceiverAction } from '../types';
import { TOTAL_CHANNELS, getChannelCenterFreq } from '../engine/rfSimulator';

interface WaterfallProps {
  pulses: GroundTruthPulse[];
  seqAction: ReceiverAction | null;
  smartAction: ReceiverAction | null;
  recentSeqHits: GroundTruthPulse[];
  recentSmartHits: GroundTruthPulse[];
  currentTimeMs: number;
}

export const SpectrumWaterfall: React.FC<WaterfallProps> = ({
  pulses,
  seqAction,
  smartAction,
  recentSeqHits,
  recentSmartHits,
  currentTimeMs,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const historyRef = useRef<Array<{
    pulses: GroundTruthPulse[];
    seqChannel: number;
    smartChannel: number;
    seqHitChannels: number[];
    smartHitChannels: number[];
    t: number;
  }>>([]);

  // Record history buffer
  useEffect(() => {
    const history = historyRef.current;
    history.push({
      pulses: [...pulses],
      seqChannel: seqAction?.channelIndex ?? -1,
      smartChannel: smartAction?.channelIndex ?? -1,
      seqHitChannels: recentSeqHits.map((h) => h.channelIndex),
      smartHitChannels: recentSmartHits.map((h) => h.channelIndex),
      t: currentTimeMs,
    });

    if (history.length > 120) {
      history.shift();
    }
  }, [pulses, seqAction, smartAction, recentSeqHits, recentSmartHits, currentTimeMs]);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const channelWidth = width / TOTAL_CHANNELS;

    // Dark tactical radar background
    ctx.fillStyle = '#060d16';
    ctx.fillRect(0, 0, width, height);

    // Channel grid lines
    ctx.strokeStyle = '#0f2438';
    ctx.lineWidth = 1;
    for (let c = 0; c <= TOTAL_CHANNELS; c++) {
      const x = c * channelWidth;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    const history = historyRef.current;
    const rowHeight = height / Math.max(1, history.length);

    // Draw waterfall rows from bottom (newest) to top (oldest)
    for (let r = 0; r < history.length; r++) {
      const entry = history[r];
      const y = height - (history.length - r) * rowHeight;

      // Draw pulses in this time slice
      for (const p of entry.pulses) {
        const x = p.channelIndex * channelWidth;
        // Color based on emitter priority and mainbeam
        if (p.emitterType === 'burst_lpi') {
          ctx.fillStyle = p.isMainbeam ? '#f43f5e' : '#881337';
        } else if (p.emitterType === 'frequency_agile' || p.emitterType === 'adversarial_shift') {
          ctx.fillStyle = p.isMainbeam ? '#38bdf8' : '#0369a1';
        } else if (p.emitterType === 'periodic_scan') {
          ctx.fillStyle = p.isMainbeam ? '#fbbf24' : '#78350f';
        } else {
          ctx.fillStyle = '#a855f7';
        }

        ctx.fillRect(x + 1, y, channelWidth - 2, Math.max(2, rowHeight));
      }

      // Mark sequential receiver dwell
      if (entry.seqChannel >= 0) {
        const sx = entry.seqChannel * channelWidth;
        ctx.fillStyle = 'rgba(56, 189, 248, 0.15)';
        ctx.fillRect(sx, y, channelWidth, rowHeight);
      }

      // Mark smart receiver dwell
      if (entry.smartChannel >= 0) {
        const mx = entry.smartChannel * channelWidth;
        ctx.fillStyle = 'rgba(16, 185, 129, 0.25)';
        ctx.fillRect(mx, y, channelWidth, rowHeight);
      }

      // Mark smart intercepts with bright emerald glow
      for (const ch of entry.smartHitChannels) {
        const hx = ch * channelWidth;
        ctx.fillStyle = '#10b981';
        ctx.fillRect(hx + 2, y, channelWidth - 4, Math.max(3, rowHeight));
      }
    }

    // Draw bottom active receiver cursor indicators
    if (seqAction) {
      const sx = seqAction.channelIndex * channelWidth;
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.strokeRect(sx + 1, height - 16, channelWidth - 2, 14);
      ctx.fillStyle = '#38bdf8';
      ctx.font = '9px monospace';
      ctx.fillText('SEQ', sx + 2, height - 5);
    }

    if (smartAction) {
      const mx = smartAction.channelIndex * channelWidth;
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(mx + 1, height - 28, channelWidth - 2, 26);
      ctx.fillStyle = '#10b981';
      ctx.font = 'bold 9px monospace';
      ctx.fillText('SMART', mx + 1, height - 18);
    }
  }, [pulses, seqAction, smartAction, recentSeqHits, recentSmartHits, currentTimeMs]);

  return (
    <div className="flex flex-col bg-slate-950 border border-slate-800 rounded-lg p-3 shadow-xl">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800/80 mb-2">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-xs font-semibold tracking-wider text-slate-200 uppercase font-mono">
            High-Resolution RF Waterfall Spectrogram (2.0 – 10.0 GHz)
          </span>
        </div>
        <div className="flex items-center gap-4 text-[11px] font-mono">
          <span className="flex items-center gap-1.5 text-sky-400">
            <span className="w-2.5 h-1.5 bg-sky-500 inline-block rounded-xs" /> Sequential Dwell
          </span>
          <span className="flex items-center gap-1.5 text-emerald-400 font-bold">
            <span className="w-2.5 h-1.5 bg-emerald-500 inline-block rounded-xs" /> Cognitive Smart Scan
          </span>
          <span className="flex items-center gap-1.5 text-rose-400">
            <span className="w-2.5 h-1.5 bg-rose-500 inline-block rounded-xs" /> LPI Burst
          </span>
          <span className="flex items-center gap-1.5 text-amber-400">
            <span className="w-2.5 h-1.5 bg-amber-500 inline-block rounded-xs" /> Periodic Mainbeam
          </span>
        </div>
      </div>

      <div className="relative w-full overflow-hidden rounded border border-slate-900">
        <canvas
          ref={canvasRef}
          width={800}
          height={260}
          className="w-full h-52 block cursor-crosshair"
        />
      </div>

      {/* Frequency channel tick marks along bottom */}
      <div className="grid grid-cols-8 gap-0 text-[10px] font-mono text-slate-400 mt-1.5 pt-1 border-t border-slate-900">
        <span className="text-left">2.0 GHz (CH 0)</span>
        <span className="text-center">3.0 GHz</span>
        <span className="text-center">4.0 GHz</span>
        <span className="text-center">5.0 GHz</span>
        <span className="text-center">6.0 GHz</span>
        <span className="text-center">7.0 GHz</span>
        <span className="text-center">8.0 GHz</span>
        <span className="text-right">10.0 GHz (CH 31)</span>
      </div>
    </div>
  );
};
