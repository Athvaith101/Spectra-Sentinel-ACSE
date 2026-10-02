# SIH26055 Demo — Architecture & Information Boundary

## Demonstration loop

```
        ┌─────────────────────────────────────────────────────────────┐
        │                       SIMULATION SIDE                       │
        │                                                             │
        │   Scenario (seeded emitter composition)                     │
        │        ↓  src/app/scenarioBuilder.ts                        │
        │   RF Environment                                            │
        │        (32 bands × 250 MHz, 2.0–10.0 GHz)                   │
        │        ↓  src/simulation/rfSimulator.ts                     │
        │   Ground-truth pulses  ────────────────┐                    │
        │        ↓                               │                    │
        │   Receiver front end                   │                    │
        │        (LO settling 35 µs, detection   │                    │
        │         physics, collisions, PFA)      │                    │
        │        ↓  src/simulation/receiverModel.ts                   │
        │   Pulse Descriptor Words (PDWs)        │                    │
        │        ↓                               │                    │
        ├────────────────────────────────────────┼────────────────────┤
        │            SCHEDULER SIDE              │ (truth receipt     │
        │                                        │  sidecar —         │
        │   Observable State                     │  evaluation ONLY)  │
        │   (PDWs, Beta beliefs, staleness,      │                    │
        │    Markov/MHT transitions, hazard)     │                    │
        │        ↓  src/state/*                  │                    │
        │   Scheduler (policy)                   │                    │
        │        ↓  src/policies/*               │                    │
        │   Action: next band + dwell            │                    │
        │        ↓                               │                    │
        ├────────────────────────────────────────┴────────────────────┤
        │                      OBSERVATION LOOP                       │
        │   Action → receiver tunes → PDWs → belief update →          │
        │   next decision                                             │
        └─────────────────────────────────────────────────────────────┘
                                ↓
        PRESENTATION (src/pages/*, src/components/*)
        Home · Scenario Lab · Policy Arena · Dataset · Methodology
        waterfall · decision panel · playback · results · comparison
                                ↓
        EVALUATION (src/state/harness.ts, tracking.ts, statistics.ts)
        Truth receipts joined to emissions AFTER the fact → Pd, MIT,
        staleness, coverage — never fed back into the scheduler.
```

## Pages and their responsibilities

| Route | Page | Reads | Owns |
|---|---|---|---|
| `#/` | Home (`pages/HomePage.tsx`) | static content | — |
| `#/lab` | Scenario Lab (`pages/ScenarioLabPage.tsx`) | `useLabSession` snapshot | live playback, decision panel, results |
| `#/arena` | Policy Arena (`pages/PolicyArenaPage.tsx`) | headless `EpisodeRunner` runs | comparison table + waterfalls |
| `#/dataset` | Dataset (`pages/DatasetPage.tsx`) | uploaded file header (bytes only) | structural validation state |
| `#/method` | Methodology (`pages/MethodologyPage.tsx`) | `POLICY_CATALOG` | — |

## What the scheduler sees vs what the simulation knows

| Information | Scheduler | Simulation/Evaluation |
|---|---|---|
| Tuned band, dwell time | ✦ decides it | receives it |
| PDWs of the current band (TOA, freq, PW, SNR, amplitude) | ✓ | ✓ |
| False alarms, collided blobs | ✓ (flagged) | ✓ |
| Receiver association tags (observable clusters) | ✓ | ✓ |
| Occupancy beliefs, staleness, hazard, transitions | ✓ (own state) | — |
| **True emitter identity / type / priority** | ✗ never | ✓ (truth side) |
| **Emissions on untuned bands** | ✗ never | ✓ |
| **Future ground truth** | ✗ never | ✓ (for scoring only) |

✓ = available · ✗ = withheld by design · ✦ = the only output the scheduler produces

This boundary is inherited verbatim from the repository engine
(`INFORMATION BOUNDARY` comments in `receiverModel.ts`, `tracking.ts`, `harness.ts`)
and is a scientific requirement, not a UI choice: the demo must obey the same causal
constraints as the research evaluation.

## Where each responsibility lives

| Responsibility | File(s) |
|---|---|
| Navigation shell / routing | `src/app/App.tsx` (hash routes — static-host safe) |
| Scenario generation (seeded, composition-first) | `src/app/scenarioBuilder.ts` |
| RF world / emitter physics | `src/simulation/rfSimulator.ts`, `emitterModels.ts`, `reactiveEmitters.ts`, `tsrdGrounding.ts` |
| Receiver physics | `src/simulation/receiverModel.ts`, `detectionModel.ts` |
| Policies (sequential/adaptive/…) | `src/policies/*` |
| Beliefs, prediction, safety | `src/state/bayesianBelief.ts`, `markov.ts`, `multihypothesis.ts`, `changepoint.ts`, `hazard.ts`, `coverageShield.ts`, `oodGate.ts` |
| Adapter (UI ↔ policy contract, policy catalog) | `src/app/schedulerAdapter.ts` |
| Episode runner (0.5 ms steps, decision-boundary stepping) | `src/app/episodeRunner.ts` |
| Playback clock (throttled, speed-scaled) | `src/state/playback.ts` |
| Lab session lifecycle | `src/state/labSession.ts` |
| Evaluation joins + metrics | `src/state/harness.ts`, `tracking.ts`, `statistics.ts` |
| Visualization & controls | `src/components/*`, `src/pages/*` |

## Model integration boundary

```
Frontend (pages & panels)
      ↓
DemoSchedulerAdapter           ← the ONLY integration point
      ↓                   ↘
engine policies          future validated model
(src/policies/*)         (same BaseScheduler contract;
                          backend-backed adapter allowed)
```

To connect the final trained scheduler when it is validated:

1. Add a `PolicyId` + `POLICY_CATALOG` entry with an honest availability note.
2. Add a `forPolicy()` case that wraps the model behind the `BaseScheduler` surface
   (`getDecision / onPDWObserved / registerGroundTruthPulse / onDwellCompleted / reset`).
   If the artifact cannot run in the browser, the adapter calls a backend service —
   the frontend never fakes inference.
3. No page or component changes are required: the lab and arena render whatever the
   catalog exposes.

## Provenance

All engine files under `simulation/`, `policies/` and most of `state/` are verbatim
copies of the repository's validated TypeScript engine (repo-root `src/environment`,
`src/decision`, `src/belief`, `src/prediction`, `src/optimization`, `src/safety`,
`src/benchmark`) with only import paths rewritten; each file carries a `PROVENANCE:`
header. Behaviour, seeds and metrics are identical to the original engine — verified by
`smoke-test.ts`.
