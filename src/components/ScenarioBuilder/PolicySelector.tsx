/**
 * Scan policy selector (spec §4) — placed near the top of the Scenario Lab, above the
 * run controls and the waterfall. Options come from the canonical PolicyRegistry so a
 * newly connected policy becomes selectable without touching the UI.
 *
 * CORRECTION MODEL: connected policies are selectable/runnable; real-but-pending policies
 * appear with "Backend integration pending", are disabled, and can never run or produce
 * fabricated results.
 */
import React from 'react';
import { POLICY_CATALOG, PolicyId } from '../../app/schedulerAdapter';

interface Props {
  value: PolicyId;
  onChange: (id: PolicyId) => void;
}

export const PolicySelector: React.FC<Props> = ({ value, onChange }) => {
  const meta = POLICY_CATALOG[value];
  const policies = Object.values(POLICY_CATALOG);

  return (
    <div className="panel panel-pad">
      <div className="spread" style={{ marginBottom: 8 }}>
        <span className="panel-title">2 · Scan policy</span>
        <span className="chip" style={{ fontSize: 9 }}>
          {meta.category === 'baseline' ? 'baseline' : 'adaptive'}
        </span>
        <span className="divider-v" style={{ height: 12 }} />
        <span className="tiny" style={{ color: 'var(--faint)' }}>
          {policies.filter((p) => p.executable).length} connected · {policies.length} total
        </span>
      </div>

      <label className="field">
        <span>Scheduler under test</span>
        <select className="control" value={value} onChange={(e) => onChange(e.target.value as PolicyId)}>
          {policies.map((p) => (
            <option key={p.id} value={p.id} disabled={!p.executable}>
              {p.displayName}{p.executable ? '' : ' — Backend integration pending'}
            </option>
          ))}
        </select>
      </label>

      {value !== 'sequential' && (
        <p className="tiny" style={{ margin: '6px 0 0', color: 'var(--faint)', fontFamily: 'var(--mono)' }}>
          {meta.category === 'baseline' ? 'Reference baseline policy' : 'Adaptive policy — compared against the Sequential baseline.'}
        </p>
      )}

      <p className="tiny" style={{ color: 'var(--muted)', margin: '8px 0 0', minHeight: 30 }}>
        {meta.description}
        {meta.availabilityNote && (
          <span style={{ color: 'var(--warn)' }}> {meta.availabilityNote}</span>
        )}
      </p>
    </div>
  );
};
