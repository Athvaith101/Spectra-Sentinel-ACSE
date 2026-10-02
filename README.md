# SIH26055 — Demo Web Application

The judge-facing web demonstration of the **Adaptive Cognitive Scan Scheduling Engine**
(SIH 2026 / SIH26055): a wideband receiver that can only listen to one 250 MHz band at a
time must decide *where and when to scan next*. The homepage leads with the project's
measured benchmark evidence; the interactive pages let you build scenarios and watch the
adaptive decisions live.

This folder is **self-contained**: it can be zipped, handed to another developer, and
built without the rest of the repository.

---

## 1. Pages

| Page | Route | Purpose |
|---|---|---|
| Home | `#/` | Identity → Explore → **Benchmark snapshot** → short context. Theme toggle (top-right). |
| Scenario Lab | `#/lab` | Build a seeded RF environment, pick a scan policy, watch each scan decision live, see results. |
| Policy Arena | `#/arena` | Run 2–4 exposed schedulers on the identical seeded scenario; side-by-side waterfalls + metric deltas. |
| Benchmark | `#/benchmark` | The detailed evidence: locked-evaluation table, comparisons, methodology, labelled secondary protocols, stated limitations. |
| Dataset | `#/dataset` | Structural validation of TSRD-compatible HDF5 uploads; honest pipeline description. |
| Methodology | `#/method` | Problem, approach, architecture diagram, fairness constraints, model integration boundary. |

Routing is hash-based (`src/app/App.tsx`), so the static build works on any host without
server-side rewrites.

## 2. Benchmark data integrity

- The **single source of benchmark numbers** is `src/state/lockedBenchmark.ts` — a verbatim
  transcription of the project's locked-evaluation report (`latest readme.md` §19/§20;
  `results/locked/locked_test_report.json` of the same evaluation line).
- The locked evaluation: **4,689-file TSRD-derived synthetic PDW corpus (2.466B pulses),
  3 seeds, 528 episodes, 24 scenario instances, 11 policies.** Headline: Cognitive Pd
  0.2403 vs Sequential 0.0657 — ΔPd +0.1746, 95% CI [+0.1250, +0.2258], ΔReward +2.875.
- Secondary protocols (development tournament, RL research track) are displayed only on
  the Benchmark page, under their own labels, **never merged** with the locked table.
- Browser runs are labelled `LIVE SCENARIO RESULT` and are never mixed into benchmark tables.
- The locked report's MIT column contained a timestamp artefact (see
  `results/locked/locked_test_v6_audit.json`); MIT is not quoted as a locked headline metric.

## 3. What the demo does NOT do

- It does **not** claim the final trained backend model is deployed. Scenario Lab and
  Policy Arena execute the packaged TypeScript engine (validated engine copies).
- It does **not** expose any policy outside the latest README's locked-evaluation set.
  The UI catalog (`POLICY_CATALOG` in `src/app/schedulerAdapter.ts`) contains exactly those
  11 backend policies; the 4 with a local engine path are runnable, the other 7 are shown
  as "Backend integration pending" and cannot execute or produce results.
- It does **not** let schedulers see ground truth or future information (information
  boundary documented in `DEMO_ARCHITECTURE.md`).
- It does **not** execute uploaded files. Dataset validation is structural only.
- It is **not** a training pipeline. No training code exists in this folder.

## 4. Architecture (short form)

```
pages/         HomePage · ScenarioLabPage · PolicyArenaPage · BenchmarkPage · DatasetPage · MethodologyPage
components/    Benchmark/ · ScenarioBuilder/ · Simulation/ · Waterfall/   (presentation only)
app/           App shell · routing · scenarioBuilder · schedulerAdapter · episodeRunner · backendContract
state/         lockedBenchmark (authoritative numbers) · playback · lab session · engine state modules
simulation/    rfSimulator · emitterModels · receiverModel · detectionModel · …
policies/      sequential · cognitive · staleness · mpc (engine files for other families are not bundled)
scripts/       concentration-diagnostic.ts (channel-concentration measurement)
```

Full data-flow diagram with the truth/observable boundary: see `DEMO_ARCHITECTURE.md`.

## 5. Policy catalog (single source)

`POLICY_CATALOG` / `PolicyRegistry` in `src/app/schedulerAdapter.ts` is the **one** policy
list for the whole website: Scenario Lab selector, Policy Arena selection + comparison,
Benchmark page, homepage snapshot and the backend adapter all read it — no page keeps its
own list. The catalog contains **exactly the latest README's locked-evaluation policy set**
(11 policies, `backendAdapter` = the backend policy id, 1:1). Availability model:

- **`connected`** — executable in the browser today: **Sequential — Baseline, Staleness,
  Cognitive, MPC.** Selectable and runnable in the Scenario Lab and Policy Arena.
- **`pending`** — real project policies whose browser execution path is not connected:
  Information, UCB1, LinUCB, Thompson, Markov, STARE-Cognitive, Portfolio. Shown
  everywhere, labelled **“Backend integration pending”**, disabled in pickers, excluded
  from live runs — they never produce fabricated results.

Removed entirely (not part of the README policy set): the demo-only “Adaptive portfolio”
router and the Q-learning expert that previously appeared in the frontend.

Backend-adapter seam: `PolicyRunner` / `DemoSchedulerAdapter.forPolicy` maps PolicyId →
implementation, and **throws for pending policies** (no silent substitution, no fabricated
runs). Connecting a policy = adding its runner case + flipping its availability to
`connected` in the catalog — no page rewrites.

## 6. Backend integration

`src/app/backendContract.ts` is the single integration seam. It documents the contract
derived from the backend's actual runner (`run_episode(env, tracker, policy, scn, seed)` →
metrics dict + decision/event streams):

```
REQUEST  { policy: backendId, seed, scenario, config? }
RESPONSE { decisions[], events[], metrics{pd, weighted_poi, hp_miss, coverage, reward}, logs? }
```

`USE_BACKEND = false` today. When the backend service exists, implement
`BackendRunnerClient`, flip the flag, and both Scenario Lab and Policy Arena route through
the same seam — the waterfall/decision/results UI renders backend runs unchanged. The
frontend never fakes inference or decorates results.

`backendContract.ts` also defines the **common result schema** (`PolicyRunResult`) consumed
by both pages: core metrics (Pd, weighted POI, coverage, reward), optional extended
metrics, and an `evaluationSource` / `evaluationLabel` pair (`LOCKED BENCHMARK` vs
`LIVE SCENARIO RESULT` vs `BACKEND INFERENCE`) so documented values and live runs are
never blurred together.

## 7. Scenario builder (composition-first)

Unchanged from the previous version: emitter composition is the source of truth, presets
populate an editable form, everything is seed-deterministic (`SeededRNG` / Mulberry32),
and Replay rebuilds the exact world.

## 8. Playback

`src/state/playback.ts` runs the engine headlessly on a wall-clock-paced loop and
publishes throttled snapshots; speed presets (0.05×–10×) scale wall-clock time only.
Controls: Run / Pause / Reset / Replay / Next decision.

## 9. Where metrics are calculated

- Per-episode metrics (Pd, MIT, staleness, scans, latency): each scheduler's own
  `SchedulerMetrics` (`src/policies/baseScheduler.ts`), presented hierarchically in
  `ResultsPanel.tsx` (primary → coverage/efficiency → behaviour → technical details).
- Truth-side latency attribution: `src/state/tracking.ts` via `src/state/harness.ts` —
  evaluation only, never decisions.
- Benchmark numbers: `src/state/lockedBenchmark.ts` only (see §2).

## 10. How to install / run / verify

```bash
cd demo_web
npm install --legacy-peer-deps

npm run dev                    # http://localhost:3000
npm run typecheck              # strict TypeScript check (tsc --noEmit)
npm run build                  # production build → dist/
npx tsx smoke-test.ts          # engine + canonical-catalog/availability checks + locked-value
                               # verification (§35) + concentration guardrails
npx tsx scripts/concentration-diagnostic.ts   # full multi-seed channel-concentration report
```

## 11. How to deploy

The build is fully static — any static host works:

- **Netlify**: `npm run build`, publish directory `dist/` (config in `netlify.toml`).
- **Vercel**: import the folder; `vercel.json` rewrites all routes to `index.html`.
- Any other host: upload the contents of `dist/`.

No server-side runtime, database or API keys are required.

## 12. Research-integrity labels used in the UI

- `LIVE SCENARIO RESULT` — metric produced by the current seeded run.
- `LOCKED EVALUATION · READ ONLY` — verbatim benchmark values from the locked report.
- `EXPLORATORY · PRE-FREEZE` / `RESEARCH TRACK` — secondary protocols, shown separately.
- Decision factors are labelled observable scheduler state, not neural-network internals.

## Production deployment

This package is self-contained for the web/backend integration:

- React/Vite frontend: `src/`
- Canonical Python v6.2 engine: `python_engine/src/`
- Frozen local artifacts: `models/frozen/` and `models/rl/`
- Docker backend: `Dockerfile`
- Render service definition: `render.yaml`
- Hugging Face upload helper: `scripts/upload_models_to_hf.py`

The frontend reads `VITE_BACKEND_URL`. The backend can use bundled artifacts (`MODEL_SOURCE=local`) or hydrate them from Hugging Face (`MODEL_SOURCE=huggingface`, `HF_MODEL_REPO=...`).
