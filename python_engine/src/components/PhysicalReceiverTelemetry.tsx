/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { ReceiverFrontEndState, SchedulerMetrics } from '../types';
import { Radio, Activity, AlertCircle, ShieldAlert, Zap } from 'lucide-react';

interface Props {
  seqFrontEndState?: ReceiverFrontEndState;
  smartFrontEndState?: ReceiverFrontEndState;
  seqMetrics: SchedulerMetrics;
  smartMetrics: SchedulerMetrics;
}

export const PhysicalReceiverTelemetry: React.FC<Props> = ({
  smartFrontEndState,
  seqMetrics,
  smartMetrics,
}) => {
  const pulsesLostSettlingSeq = seqMetrics.pulsesLostToSettling || 0;
  const pulsesLostSettlingSmart = smartMetrics.pulsesLostToSettling || 0;

  const collisionsSeq = seqMetrics.pulsesLostToCollision || 0;
  const collisionsSmart = smartMetrics.pulsesLostToCollision || 0;

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-lg p-3.5 shadow-xl font-mono">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2 mb-3">
        <div className="flex items-center gap-2">
          <Radio className="w-4 h-4 text-emerald-400" />
          <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
            Physical RF Receiver Front-End Telemetry
          </span>
        </div>
        <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
          LO SETTLING: 35 &mu;s | IF BW: 250 MHz
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        {/* Metric 1: LO Settling Dead Time Loss */}
        <div className="bg-slate-900/70 border border-slate-800/80 rounded-md p-2.5">
          <div className="flex items-center gap-1.5 text-[10px] text-slate-400 font-sans">
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            <span>LO Settling Drops</span>
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-lg font-black text-slate-100">{pulsesLostSettlingSmart}</span>
            <span className="text-[10px] text-slate-400">pulses</span>
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">
            vs {pulsesLostSettlingSeq} on Sequential
          </div>
        </div>

        {/* Metric 2: Simultaneous Pulse Collisions */}
        <div className="bg-slate-900/70 border border-slate-800/80 rounded-md p-2.5">
          <div className="flex items-center gap-1.5 text-[10px] text-slate-400 font-sans">
            <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
            <span>Pulse Overlap Collisions</span>
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-lg font-black text-slate-100">{collisionsSmart}</span>
            <span className="text-[10px] text-slate-400">flagged</span>
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">
            &Delta;P &le; 3dB discriminator
          </div>
        </div>

        {/* Metric 3: Noise False Alarms Generated */}
        <div className="bg-slate-900/70 border border-slate-800/80 rounded-md p-2.5">
          <div className="flex items-center gap-1.5 text-[10px] text-slate-400 font-sans">
            <Activity className="w-3.5 h-3.5 text-sky-400" />
            <span>Noise False Alarms</span>
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-lg font-black text-slate-100">
              {smartFrontEndState?.totalFalseAlarmsGenerated || 0}
            </span>
            <span className="text-[10px] text-sky-300">Pfa ~ 10⁻⁴</span>
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">
            Rejected by Bayesian filter
          </div>
        </div>

        {/* Metric 4: Front-End Detection Pd */}
        <div className="bg-slate-900/70 border border-slate-800/80 rounded-md p-2.5">
          <div className="flex items-center gap-1.5 text-[10px] text-slate-400 font-sans">
            <AlertCircle className="w-3.5 h-3.5 text-emerald-400" />
            <span>Physical Detection Pd</span>
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-lg font-black text-emerald-400">
              {(smartMetrics.probabilityOfDetection * 100).toFixed(1)}%
            </span>
            <span className="text-[10px] text-slate-400">Albersheim model</span>
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">
            CFAR Threshold: 10.5 dB SNR
          </div>
        </div>
      </div>
    </div>
  );
};
