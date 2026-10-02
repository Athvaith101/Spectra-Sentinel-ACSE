/**
 * Dataset upload validation — the ONE canonical implementation, used by the Dataset
 * page and embedded in the Policy Arena's Upload Dataset mode.
 *
 * HONESTY (spec §7/§11/§12): performs STRUCTURAL validation only (extension, HDF5
 * signature, size). Schema validation (metadata/feature_names, [N×5] layout) runs on
 * the research pipeline. Nothing is executed; no model results are fabricated. The
 * panel states plainly that "dataset validation" ≠ "model evaluation".
 */
import React, { useRef, useState } from 'react';

const REQUIRED_FEATURES = ['ToA', 'Frequency', 'PulseWidth', 'AoA', 'Amplitude'];
const MAX_SIZE_MB = 50;
const HDF5_SIG = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

interface Checks {
  fileName: string;
  sizeMb: number;
  isH5: boolean;
  sizeOk: boolean;
  headerOk: boolean | null;
  note: string;
}

export const DatasetValidationPanel: React.FC = () => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [checks, setChecks] = useState<Checks | null>(null);
  const [inspecting, setInspecting] = useState(false);

  const handleFile = async (file: File) => {
    setInspecting(true);
    const sizeMb = file.size / (1024 * 1024);
    const isH5 = /\.(h5|hdf5|he5)$/i.test(file.name);

    let headerOk: boolean | null = null;
    let note = '';

    if (!isH5) {
      note = 'Unsupported extension. The evaluation pipeline reads TSRD-format HDF5 pulse files only.';
    } else if (sizeMb > MAX_SIZE_MB) {
      note = `File exceeds the ${MAX_SIZE_MB} MB browser-validation limit. Use a smaller sample — the full file can be evaluated with the research pipeline scripts.`;
    } else {
      try {
        const head = await file.slice(0, 8).arrayBuffer();
        const sig = new Uint8Array(head);
        headerOk = HDF5_SIG.every((b, i) => sig[i] === b);
        note = headerOk
          ? 'Valid HDF5 signature. Full schema validation (metadata/feature_names, receiver metadata, data [N×5]) runs on the research pipeline — the hosted demo does not execute uploaded content.'
          : 'File does not carry the HDF5 signature — it cannot be a TSRD HDF5 file.';
      } catch {
        headerOk = null;
        note = 'Could not read the file header.';
      }
    }

    setChecks({ fileName: file.name, sizeMb, isH5, sizeOk: sizeMb <= MAX_SIZE_MB, headerOk, note });
    setInspecting(false);
  };

  const row = (ok: boolean | null, label: string) => (
    <div className="row tiny mono" style={{ gap: 8 }}>
      <span style={{ color: ok === null ? 'var(--faint)' : ok ? 'var(--pos)' : 'var(--bad)', width: 12 }}>
        {ok === null ? '…' : ok ? '✓' : '✗'}
      </span>
      <span style={{ color: ok === false ? 'var(--muted)' : 'var(--text)' }}>{label}</span>
    </div>
  );

  const compatible = checks?.isH5 && checks.sizeOk && checks.headerOk === true;

  return (
    <div className="panel panel-pad">
      <div className="spread" style={{ marginBottom: 8 }}>
        <span className="panel-title">Dataset validation</span>
        <span className="chip" style={{ fontSize: 9 }}>structural only</span>
      </div>

      <div className="kv"><span className="k">Format</span><span className="v">TSRD-compatible HDF5</span></div>
      <div className="kv"><span className="k">Pulse matrix</span><span className="v">data [N×5] float</span></div>
      <div className="kv"><span className="k">Feature order</span><span className="v">{REQUIRED_FEATURES.join(', ')}</span></div>
      <div className="kv"><span className="k">Optional</span><span className="v">labels, metadata/receiver/*</span></div>

      <div className="row" style={{ gap: 10, marginTop: 12, flexWrap: 'wrap' }}>
        <button className="btn btn-primary" onClick={() => inputRef.current?.click()}>Choose file</button>
        <input
          ref={inputRef} type="file" accept=".h5,.hdf5,.he5"
          style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void handleFile(f);
            e.target.value = '';
          }}
        />
        <span className="tiny" style={{ color: 'var(--faint)' }}>
          Max {MAX_SIZE_MB} MB · files are inspected structurally, never executed
        </span>
      </div>

      {inspecting && <div className="tiny mono" style={{ marginTop: 8, color: 'var(--accent)' }}>Inspecting file…</div>}

      {checks && (
        <div style={{ marginTop: 10, borderTop: '1px solid var(--border)', paddingTop: 10 }}>
          <div className="tiny mono" style={{ fontWeight: 700, marginBottom: 6 }}>
            Validation — {checks.fileName} ({checks.sizeMb.toFixed(2)} MB)
          </div>
          {row(true, 'File received and inspected (no execution)')}
          {row(checks.isH5, 'HDF5 file extension')}
          {row(checks.sizeOk, `Size within ${MAX_SIZE_MB} MB browser-validation limit`)}
          {row(checks.headerOk, 'HDF5 signature (first 8 bytes)')}
          {row(null, 'Schema check: metadata/feature_names + [N×5] layout → research pipeline')}
          <p className="tiny" style={{
            marginTop: 8, marginBottom: 0, padding: '6px 9px',
            background: compatible ? 'var(--pos-soft)' : 'var(--bad-soft)',
            borderLeft: `3px solid ${compatible ? 'var(--pos)' : 'var(--bad)'}`,
            color: compatible ? 'var(--pos)' : 'var(--bad)',
          }}>
            {compatible
              ? 'Structurally compatible so far — continue with the research pipeline for full schema validation and model evaluation.'
              : 'File structure is not compatible with the current evaluation pipeline. ' + checks.note}
          </p>
        </div>
      )}

      <p className="tiny" style={{ color: 'var(--muted)', margin: '10px 0 0', lineHeight: 1.5 }}>
        Dataset validation is not model evaluation: the hosted demo cannot run the learned-scheduler
        pipeline on uploaded data, and no results are fabricated for it.
      </p>
    </div>
  );
};
