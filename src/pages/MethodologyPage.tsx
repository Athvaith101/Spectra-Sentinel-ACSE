import React from 'react';

export const MethodologyPage: React.FC = () => (
  <div className="page" style={{ maxWidth: 1040 }}>
    <div style={{ marginBottom: 14 }}>
      <div className="hero-kicker">SIH26055 · V6.2 · CURRENT WORKING ARCHITECTURE</div>
      <h1 className="page-title">Methodology</h1>
      <div className="section-note">The current end-to-end design, what the final model actually consumes, and how the demonstration is evaluated.</div>
    </div>

    <div style={{ display: 'grid', gap: 12 }}>
      <section className="panel panel-pad" style={{ borderColor: 'var(--accent)' }}>
        <div className="spread" style={{ alignItems: 'flex-start' }}>
          <div>
            <div className="panel-title">Final integrated model</div>
            <p className="small" style={{ margin: '7px 0 0', maxWidth: 820 }}>
              <strong>STARE-Cognitive</strong> is the canonical model path used for the main demonstration: a frozen STARE temporal encoder produces a 64-dimensional representation from a 128-pulse receiver window, and the cognitive scheduler uses that representation together with receiver-observable belief, prediction, uncertainty, staleness, coverage and priority signals to choose the next channel and dwell time.
            </p>
          </div>
          <span className="chip chip-pos">FINAL MODEL</span>
        </div>
        <div className="pipe" style={{ marginTop: 12 }}>
          <span className="pstep">HDF5 / PDWs</span><span className="parrow">→</span>
          <span className="pstep">normalization</span><span className="parrow">→</span>
          <span className="pstep">STARE GRU</span><span className="parrow">→</span>
          <span className="pstep">64-D embedding</span><span className="parrow">+</span>
          <span className="pstep">receiver belief state</span><span className="parrow">→</span>
          <span className="pstep">Cognitive scheduler</span><span className="parrow">→</span>
          <span className="pstep">channel + dwell</span>
        </div>
      </section>

      <section className="panel panel-pad">
        <div className="panel-title" style={{ marginBottom: 6 }}>1 · The EW sensing problem</div>
        <p className="small" style={{ margin: 0, maxWidth: 840 }}>
          The receiver cannot observe the entire 2–10 GHz spectrum simultaneously. Its instantaneous sensing bandwidth is one 250 MHz band, so it must repeatedly decide <em>where to look next</em> and <em>how long to dwell</em>. The objective is to improve interception opportunities while controlling coverage, stale-band risk and sensing cost.
        </p>
      </section>

      <section className="panel panel-pad">
        <div className="panel-title" style={{ marginBottom: 6 }}>2 · Receiver-observable information</div>
        <p className="small" style={{ margin: 0, maxWidth: 850 }}>
          The scheduler is deliberately separated from ground truth. Its state contains channel belief, staleness/freshness, transition prediction, activity prediction, predictive uncertainty, coverage pressure, priority pressure and receiver-level statistics such as detection density, SNR proxy, transition entropy and unique-channel coverage. For the final model, the same receiver window is encoded by STARE and its 64-D embedding is supplied to the STARE-Cognitive policy.
        </p>
      </section>

      <section className="panel panel-pad">
        <div className="panel-title" style={{ marginBottom: 6 }}>3 · STARE temporal encoder</div>
        <pre className="mono tiny" style={{ margin: 0, lineHeight: 1.55, overflowX: 'auto' }}>{`Receiver pulse window: 128 × 5
Features: ToA · Frequency · PulseWidth · AoA · Amplitude
                         │
                         ▼
                 normalization
                         │
                         ▼
                  STARE GRU
              hidden size = 128
                         │
                         ├── activity prediction
                         ├── next-frequency prediction
                         └── 64-D embedding

Frozen deployment artifact:
models/frozen/stare_inference.pt`}</pre>
        <p className="tiny" style={{ color: 'var(--muted)', margin: '8px 0 0' }}>
          The web backend loads the frozen TorchScript artifact for inference. It does not train the model from the browser.
        </p>
      </section>

      <section className="panel panel-pad">
        <div className="panel-title" style={{ marginBottom: 6 }}>4 · Cognitive scan decision</div>
        <p className="small" style={{ margin: 0, maxWidth: 850 }}>
          The cognitive layer scores candidate channels using the deployed policy weights: belief, prediction, staleness, predictive uncertainty, coverage pressure and priority pressure. The current frozen policy configuration is recorded in <span className="mono">models/frozen/cognitive_policy.json</span>. OOD conditions reduce reliance on prediction and increase exploration; change-point conditions increase coverage pressure.
        </p>
        <div className="pipe" style={{ marginTop: 10 }}>
          <span className="pstep">belief</span><span className="parrow">+</span>
          <span className="pstep">prediction</span><span className="parrow">+</span>
          <span className="pstep">staleness</span><span className="parrow">+</span>
          <span className="pstep">uncertainty</span><span className="parrow">+</span>
          <span className="pstep">coverage</span><span className="parrow">+</span>
          <span className="pstep">priority</span><span className="parrow">→</span>
          <span className="pstep">channel / dwell</span>
        </div>
      </section>

      <section className="panel panel-pad">
        <div className="panel-title" style={{ marginBottom: 6 }}>5 · End-to-end execution</div>
        <pre className="mono tiny" style={{ margin: 0, lineHeight: 1.55, overflowX: 'auto' }}>{`HDF5 / synthetic RF scenario
        │
        ▼
Receiver front end
(one 250 MHz band at a time)
        │
        ▼
PDW / receiver observation
        │
        ├──────────────► STARE temporal encoder
        │                         │
        │                         ▼
        │                    64-D embedding
        │                         │
        ▼                         ▼
Belief / prediction / uncertainty / coverage state
        │
        └──────────────► STARE-Cognitive scheduler
                                  │
                                  ▼
                           next channel + dwell
                                  │
                                  ▼
                              receiver step
                                  │
                                  └──────────► repeat

Evaluation truth is joined only after execution.`}</pre>
      </section>

      <section className="panel panel-pad">
        <div className="panel-title" style={{ marginBottom: 6 }}>6 · Real HDF5 demonstration path</div>
        <div className="console-grid-2">
          <div>
            <div className="panel-sub">Upload</div>
            <p className="small">The backend validates the TSRD-compatible schema and reads bounded temporal windows. It does not materialize an entire multi-million/billion-pulse capture just to run a web preview.</p>
          </div>
          <div>
            <div className="panel-sub">Final verification</div>
            <p className="small">The FINAL MODEL page runs frozen STARE inference and the STARE-Cognitive receiver replay, then reports the model artifact, embedding statistics and replay diagnostics.</p>
          </div>
        </div>
        <div className="note note-neutral" style={{ marginTop: 10 }}>
          <strong>Important:</strong> uploaded-file replay metrics are diagnostic measurements on bounded receiver windows. They are not presented as a replacement for the locked benchmark evaluation.
        </div>
      </section>

      <section className="panel panel-pad">
        <div className="panel-title" style={{ marginBottom: 6 }}>7 · Continuous waterfall demonstration</div>
        <p className="small" style={{ margin: 0, maxWidth: 850 }}>
          Scenario Lab is the live visual receiver demonstrator. It continuously runs the validated browser receiver/RF visualization across the full 2–10 GHz range, drawing the complete channel activity field rather than only the currently selected channel. Blue represents observed activity, green marks receiver hits and the orange cursor shows the current scan position. This visual loop is intentionally kept independent from the authoritative Python evaluation so the waterfall never freezes while a backend experiment is running.
        </p>
      </section>

      <section className="panel panel-pad">
        <div className="panel-title" style={{ marginBottom: 6 }}>8 · Research integrity and model status</div>
        <ul className="small" style={{ margin: 0, paddingLeft: 18, lineHeight: 1.75, color: 'var(--text)' }}>
          <li><strong>Receiver-only policy input:</strong> ground-truth emitter identity and untuned-band future activity are not passed into the scheduler.</li>
          <li><strong>Independent policy episodes:</strong> Policy Arena creates a fresh policy/environment trajectory for each selected policy.</li>
          <li><strong>First-run lock:</strong> the first completed tournament table remains on screen while later experiments may run in the background.</li>
          <li><strong>RL checkpoint integrity:</strong> a trained RL checkpoint is used only when its saved architecture matches the live action/state space. Incompatible checkpoints are rejected rather than silently replaced by a random network.</li>
          <li><strong>Final-model boundary:</strong> STARE-Cognitive is the canonical integrated demonstration path. Current benchmark evidence does not claim that STARE alone causes a performance improvement over Cognitive; the locked benchmark should be interpreted accordingly.</li>
        </ul>
      </section>

      <section className="panel panel-pad">
        <div className="panel-title" style={{ marginBottom: 6 }}>9 · Where to start</div>
        <div className="console-grid-2">
          <a className="entry" href="#/arena"><div className="eg">RECOMMENDED DEMO ENTRY</div><div className="et">Policy Comparison · Final Model</div><div className="ed">Verify STARE-Cognitive against an uploaded receiver capture.</div></a>
          <a className="entry" href="#/lab"><div className="eg">VISUAL ENTRY</div><div className="et">Scenario Lab</div><div className="ed">Show the continuous adaptive waterfall and receiver scan behavior.</div></a>
        </div>
      </section>
    </div>
  </div>
);
