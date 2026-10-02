/**
 * Scenario configuration panel — the ONE canonical builder, used by the Scenario Lab
 * and (embedded) by the Policy Arena's Build Scenario mode.
 *
 * - Preset dropdown with a real, selectable "Custom" state: any manual edit moves the
 *   scenario to Custom and it stays there (no silent switch-back).
 * - Emitter composition is the source of truth: quantity per class, total derived.
 * - Seed is editable + randomizable; same seed → identical world (deterministic replay).
 */
import React from 'react';
import {
  ScenarioConfig, LIMITS, PRESETS, EMITTER_CLASS_INFO, EmitterComposition,
  totalEmitters, scenarioSummary,
} from '../../app/scenarioBuilder';

export const CUSTOM_PRESET = 'custom';

interface Props {
  cfg: ScenarioConfig;
  /** Any manual edit (sliders, steppers, toggles). Caller marks the scenario Custom. */
  onChange: (cfg: ScenarioConfig) => void;
  /** A named preset was selected from the dropdown. */
  onPreset: (id: string) => void;
  /** "Custom" was explicitly selected in the dropdown (keeps current values). */
  onCustom: () => void;
  /** Seed edited directly. */
  onSeedChange: (seed: number) => void;
  /** Randomize seed requested. */
  onRandomizeSeed: () => void;
  /** 'custom' or a preset id. */
  activePreset: string;
}

const COMPOSITION_KEYS: (keyof EmitterComposition)[] = ['agile', 'scan', 'burst', 'continuous'];
const SEED_MAX = 4294967295;

function Slider(props: {
  label: string; value: number; min: number; max: number; step: number;
  format?: (v: number) => string; onChange: (v: number) => void;
}) {
  return (
    <label className="field">
      <span>
        {props.label}
        <span style={{ float: 'right', color: 'var(--text)', fontWeight: 600 }}>
          {props.format ? props.format(props.value) : props.value}
        </span>
      </span>
      <input
        type="range" className="slider" min={props.min} max={props.max} step={props.step}
        value={props.value} onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  );
}

export const ScenarioBuilderPanel: React.FC<Props> = ({
  cfg, onChange, onPreset, onCustom, onSeedChange, onRandomizeSeed, activePreset,
}) => {
  const total = totalEmitters(cfg.composition);
  const preset = PRESETS.find((p) => p.id === activePreset);
  const isCustom = activePreset === CUSTOM_PRESET || !preset;
  const set = (patch: Partial<ScenarioConfig>) => onChange({ ...cfg, ...patch });

  const bump = (key: keyof EmitterComposition, delta: number) => {
    const cur = cfg.composition[key];
    const next = Math.max(0, Math.min(LIMITS.maxPerClass, cur + delta));
    if (next === cur) return;
    // Respect the global cap so the engine workload stays bounded.
    if (delta > 0 && total >= LIMITS.maxEmitters) return;
    set({ composition: { ...cfg.composition, [key]: next } });
  };

  const seedInput = (raw: string) => {
    const n = Math.floor(Number(raw));
    if (!Number.isFinite(n)) return;
    onSeedChange(Math.max(0, Math.min(SEED_MAX, n)));
  };

  return (
    <div className="panel panel-pad">
      <div className="spread" style={{ marginBottom: 10 }}>
        <span className="panel-title">1 · Scenario configuration</span>
        {isCustom && <span className="chip chip-accent" style={{ fontSize: 9 }}>custom</span>}
      </div>

      {/* Preset dropdown — Custom is a real state (§4) */}
      <label className="field" style={{ marginBottom: 10 }}>
        <span>Preset</span>
        <select
          className="control"
          value={isCustom ? CUSTOM_PRESET : preset!.id}
          onChange={(e) => {
            if (e.target.value === CUSTOM_PRESET) onCustom();
            else onPreset(e.target.value);
          }}
        >
          <option value={CUSTOM_PRESET}>Custom</option>
          {PRESETS.map((p) => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
      </label>
      <div className="preset-desc" style={{ marginTop: -4, marginBottom: 10 }}>
        {preset ? preset.description : 'Hand-tuned scenario — every control below stays editable.'}
      </div>

      {/* Seed — editable + randomizable (§5) */}
      <div className="row" style={{ gap: 6, marginBottom: 10, alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: 1 }}>
          <span>Scenario seed</span>
          <input
            className="control" type="number" min={0} max={SEED_MAX} step={1}
            value={cfg.seed} onChange={(e) => seedInput(e.target.value)}
          />
        </label>
        <button className="btn btn-quiet" onClick={onRandomizeSeed} title="Generate a new random seed">
          Randomize
        </button>
      </div>

      {/* Emitter composition — source of truth */}
      <div style={{ marginBottom: 8 }}>
        <div className="panel-title" style={{ marginBottom: 6 }}>Emitter composition</div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <tbody>
            {COMPOSITION_KEYS.map((key) => (
              <tr key={key}>
                <td style={{ padding: '2px 0' }}>
                  <div className="small" style={{ fontWeight: 500 }}>{EMITTER_CLASS_INFO[key].label}</div>
                  <div className="tiny" style={{ color: 'var(--faint)' }}>{EMITTER_CLASS_INFO[key].hint}</div>
                </td>
                <td style={{ textAlign: 'right', width: 92 }}>
                  <span className="stepper">
                    <button onClick={() => bump(key, -1)} aria-label={`remove ${key}`} disabled={cfg.composition[key] === 0}>−</button>
                    <span className="val">{cfg.composition[key]}</span>
                    <button onClick={() => bump(key, +1)} aria-label={`add ${key}`} title={deltaBlocked(cfg) ? `Max ${LIMITS.maxEmitters} emitters total` : undefined}>+</button>
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="spread mono tiny" style={{ marginTop: 5, color: 'var(--muted)' }}>
          <span>Total emitters</span>
          <span style={{ fontWeight: 700, fontSize: 13, color: 'var(--text)' }}>{total}</span>
        </div>
      </div>

      {/* Scenario modifiers */}
      <div className="row" style={{ gap: 16, marginBottom: 8 }}>
        <label className="row small" style={{ gap: 6, cursor: 'pointer' }}
          title="A reactive emitter evades after repeated intercepts (engine-supported behaviour)">
          <input type="checkbox" className="cb" checked={cfg.reactiveEnabled}
            onChange={(e) => set({ reactiveEnabled: e.target.checked })} />
          Reactive evasion
        </label>
        <label className="row small" style={{ gap: 6, cursor: 'pointer' }}
          title="An agile emitter shifts its hopping pattern at mid-episode">
          <input type="checkbox" className="cb" checked={cfg.adversarialShift}
            onChange={(e) => set({ adversarialShift: e.target.checked })} />
          Pattern shift
        </label>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 16px' }}>
        <Slider label="Duration" value={cfg.durationMs} min={LIMITS.minDurationMs} max={LIMITS.maxDurationMs} step={500}
          format={(v) => `${(v / 1000).toFixed(1)} s`} onChange={(v) => set({ durationMs: v })} />
        <Slider label="Activity" value={cfg.activity} min={0.1} max={0.9} step={0.05}
          format={(v) => `${(v * 100).toFixed(0)}%`} onChange={(v) => set({ activity: v })} />
        <Slider label="Agility" value={cfg.agility} min={0} max={1} step={0.05}
          format={(v) => `${(v * 100).toFixed(0)}%`} onChange={(v) => set({ agility: v })} />
        <Slider label="Difficulty" value={cfg.difficulty} min={0} max={1} step={0.05}
          format={(v) => `${(v * 100).toFixed(0)}%`} onChange={(v) => set({ difficulty: v })} />
      </div>

      <div className="preset-desc">
        {scenarioSummary(cfg).slice(1).map((line, i) => <div key={i} className="mono" style={{ fontSize: 10 }}>{line}</div>)}
      </div>
    </div>
  );
};

function deltaBlocked(cfg: ScenarioConfig): boolean {
  return totalEmitters(cfg.composition) >= LIMITS.maxEmitters;
}
