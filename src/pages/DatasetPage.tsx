/**
 * Dataset Evaluation (spec §11/§12) — honest upload validation via the shared
 * DatasetValidationPanel, plus the real evaluation-pipeline description.
 */
import React from 'react';
import { DatasetValidationPanel } from '../components/DatasetUpload/DatasetValidationPanel';

const PIPELINE = [
  'Upload',
  'Validate structure (extension, HDF5 signature, size)',
  'Read supported PDW fields (data [N×5]: ToA, Frequency, PulseWidth, AoA, Amplitude)',
  'Normalize / map fields against metadata/feature_names',
  'Build evaluation representation',
  'Run scheduler evaluation (research pipeline)',
  'Compute metrics (Pd, MIT, staleness, coverage)',
];

export const DatasetPage: React.FC = () => (
  <div className="page">
    <div style={{ marginBottom: 14 }}>
      <h1 className="page-title">Dataset Evaluation</h1>
      <div className="section-note">
        Validate a recorded pulse capture against the project's evaluation pipeline.
      </div>
    </div>

    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 460px) 1fr', gap: 12, alignItems: 'start' }}>
      <DatasetValidationPanel />

      <div style={{ display: 'grid', gap: 12 }}>
        <div className="panel panel-pad">
          <div className="panel-title" style={{ marginBottom: 8 }}>Evaluation pipeline</div>
          <div className="flow">
            {PIPELINE.map((s, i) => (
              <React.Fragment key={s}>
                {i > 0 && <div className="farrow">↓</div>}
                <div className={`fbox${i >= 4 ? ' fbox-dim' : ''}`}>{s}</div>
              </React.Fragment>
            ))}
          </div>
          <p className="tiny" style={{ color: 'var(--muted)', marginTop: 10, marginBottom: 0 }}>
            Steps 1–4 are performed here (structural validation only). Steps 5–7 run on the
            research pipeline — the hosted demo does not silently transform arbitrary files into
            compatible data.
          </p>
        </div>

        <div className="panel panel-pad">
          <div className="panel-title" style={{ marginBottom: 8 }}>What this page does not do</div>
          <ul className="tiny" style={{ margin: 0, paddingLeft: 18, color: 'var(--muted)', lineHeight: 1.7 }}>
            <li>It does not execute uploaded files.</li>
            <li>It does not convert CSV, audio or IQ recordings into the TSRD schema.</li>
            <li>It does not fabricate evaluation metrics for uploaded data.</li>
            <li>It does not replace the locked research evaluation; it triages compatibility.</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
);
