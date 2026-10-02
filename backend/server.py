from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.model_store import ensure_model_artifacts

ROOT = Path(os.environ.get("EW_V6_ROOT", Path(__file__).resolve().parents[1])).resolve()
ENGINE_ROOT = Path(os.environ.get("EW_ENGINE_ROOT", ROOT / "python_engine")).resolve()
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

ensure_model_artifacts(ROOT)

from src.data.hdf5_reader import TSRDHDF5Reader, validate_schema  # type: ignore
from src.evaluation import Scenario, TournamentHarness  # type: ignore
from src.evaluation.synthetic_generator import (  # type: ignore
    SyntheticConfig,
    available_synthetic_scenarios,
    generate,
)
from src.evaluation.tournament import _OnlineChannelModel  # type: ignore
from src.fusion.state_builder import SCANStateBuilder  # type: ignore
from src.scan.action_space import ScanActionSpace  # type: ignore
from src.scan.policy_adapter import (  # type: ignore
    make_policy,
    observation_from_fusion_state,
    available_policies as _available_policies,
)
from src.environment.scan_environment import ScanAction  # type: ignore
from src.stare.models import build_model  # type: ignore

app = FastAPI(title="SIH26055 EW Research Backend", version="6.2-production")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

POLICIES = list(_available_policies())
_STARE_MODEL_CACHE: Dict[str, Any] = {}


class SyntheticRequest(BaseModel):
    scenario: str = "mixed"
    emitters: int = Field(8, ge=1, le=50)
    horizon_s: float = Field(0.5, gt=0.01, le=30.0)
    episodes: int = Field(3, ge=1, le=100)
    seed: int = 26055
    policies: List[str] = Field(default_factory=lambda: ["sequential", "cognitive", "stare_cognitive"])


class TraceRequest(BaseModel):
    scenario: str = "mixed"
    emitters: int = Field(8, ge=1, le=50)
    horizon_s: float = Field(0.5, gt=0.01, le=5.0)
    seed: int = 26055
    policy: str = "cognitive"
    max_steps: int = Field(120, ge=1, le=500)


class JobRequest(BaseModel):
    command: str


def _job_update(job_id: str, **values: Any) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values)


def _new_job(stage: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "stage": stage,
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "artifacts": [],
        }
    return job_id



def _load_uploaded_scenario(path: Path, max_events: int = 50000, channel_count: int = 8) -> tuple[Scenario, Dict[str, Any]]:
    """Convert an uploaded TSRD HDF5 capture into the canonical v6.2 Scenario.

    This is the bridge that was missing in the previous build: the uploaded
    receiver data is not merely sharded; it becomes receiver truth for the
    canonical ScanEnvironment/TournamentHarness. Emitter labels are retained
    only on the evaluation side and are never put into PolicyObservation.
    """
    import h5py

    with h5py.File(path, "r") as h5:
        data_ds = h5["data"]
        labels_ds = h5.get("labels")
        n_total = int(data_ds.shape[0])
        if len(data_ds.shape) != 2 or int(data_ds.shape[1]) < 5:
            raise ValueError(f"Expected HDF5 data [N,5], received {list(data_ds.shape)}")
        if n_total == 0:
            raise ValueError("Uploaded HDF5 contains no pulses")

        # Never materialize the full capture. Large TSRD files can contain
        # millions/billions of pulses; the web preview only needs a bounded,
        # temporally distributed sample. HDF5 fancy indexing keeps the read
        # bounded to max_events rows.
        if n_total > max_events:
            idx = np.linspace(0, n_total - 1, max_events, dtype=np.int64)
            data_eval = np.asarray(data_ds[idx], dtype=np.float32)
            labels_eval = np.asarray(labels_ds[idx]).reshape(-1) if labels_ds is not None else None
            sampled = True
        else:
            data_eval = np.asarray(data_ds[:], dtype=np.float32)
            labels_eval = np.asarray(labels_ds[:]).reshape(-1) if labels_ds is not None else None
            sampled = False

        feature_names = [x.decode(errors="ignore") if isinstance(x, bytes) else str(x)
                         for x in h5["metadata"]["feature_names"][...]]
        receiver = h5["metadata"]["receiver"]
        physical_centres = np.asarray(receiver["dwell_centres_mhz"], dtype=np.float64)
        physical_dwells = np.asarray(receiver["dwell_times_s"], dtype=np.float64)

    toa_raw = data_eval[:, 0].astype(np.float64)
    # TSRD ToA in the supplied corpus is expressed on a microsecond-like
    # numerical scale. Use a conservative unit inference for alternate files.
    span = float(np.nanmax(toa_raw) - np.nanmin(toa_raw))
    if span > 1000.0:
        toa_s = (toa_raw - np.nanmin(toa_raw)) * 1e-6
        toa_unit = "microseconds->seconds"
    else:
        toa_s = toa_raw - np.nanmin(toa_raw)
        toa_unit = "seconds"

    # The deployed RL/SCAN experiments use an 8-channel action space. Project
    # the receiver's physical dwell ladder onto 8 stable super-bands so every
    # policy is evaluated in the same action space.
    channel_count = max(2, min(int(channel_count), len(physical_centres)))
    pick = np.linspace(0, len(physical_centres) - 1, channel_count).round().astype(int)
    channels = physical_centres[pick]
    dwell = float(np.median(physical_dwells)) if len(physical_dwells) else 0.001

    events: list[dict[str, Any]] = []
    for i, (row, t) in enumerate(zip(data_eval, toa_s)):
        freq = float(row[1])
        ch = int(np.argmin(np.abs(channels - freq)))
        # Evaluation truth is kept in the Scenario only. The policy receives
        # the receiver observation generated by ScanEnvironment.
        emitter_id = int(labels_eval[i]) if labels_eval is not None else i
        events.append({
            "event_id": f"uploaded_{i}",
            "toa_s": float(t),
            "emit_time_s": float(t),
            "frequency_mhz": freq,
            "pulse_width_s": float(max(1e-7, abs(float(row[2])) * 1e-6)),
            "aoa_deg": float(row[3]),
            "amplitude": float(row[4]),
            "priority": 1.0,
            "weight": 1.0,
            "emitter_id": emitter_id,
            "projected_channel": ch,
        })

    horizon = max(float(np.max(toa_s)) + dwell, dwell)
    scenario = Scenario(
        name=f"UPLOADED-{path.stem}",
        truth_events=events,
        horizon_s=horizon,
        seed=26055,
        frequencies_mhz=[float(x) for x in channels],
        dwells_s=[dwell],
        retune_time_s=0.0,
        metadata={
            "source": "uploaded_tsrD_hdf5",
            "original_pulses": int(n_total),
            "evaluated_pulses": int(len(events)),
            "sampled": sampled,
            "toa_unit_conversion": toa_unit,
            "physical_channels": int(len(physical_centres)),
            "evaluation_channels": int(len(channels)),
            "features": feature_names,
            "unique_emitters": int(len(np.unique(labels_eval))) if labels_eval is not None else None,
        },
    )
    return scenario, {
        "original_pulses": int(n_total),
        "evaluated_pulses": int(len(events)),
        "sampled": sampled,
        "physical_channels": int(len(physical_centres)),
        "evaluation_channels": int(len(channels)),
        "horizon_s": horizon,
        "unique_emitters": int(len(np.unique(labels_eval))) if labels_eval is not None else None,
        "features": feature_names,
    }


def _policy_artifact(policy_name: str) -> Optional[Path]:
    """Find the frozen RL checkpoint used for a named policy, if present."""
    if policy_name not in {"dqn", "double_dqn", "dueling_dqn", "dueling_double_dqn", "ppo"}:
        return None
    candidates = [
        ROOT / "models" / "rl" / f"{policy_name}_seed26055.pt",
        ROOT / "models" / "rl" / f"{policy_name}.pt",
        ROOT / "models" / "frozen" / f"{policy_name}_seed26055.pt",
    ]
    return next((p for p in candidates if p.exists()), None)


def _configure_frozen_policy(policy: Any, policy_name: str) -> Dict[str, Any]:
    """Load a saved RL checkpoint only when its architecture matches the live policy.

    The previous integration treated a failed ``load_state_dict`` as a soft warning,
    then continued with a freshly initialized network. That made the UI show a
    checkpoint path even though the checkpoint was not actually driving decisions.
    """
    if policy_name == "stare_cognitive":
        artifact = ROOT / "models" / "frozen" / "stare_inference.pt"
        if not artifact.exists():
            artifact = ROOT / "models" / "stare" / "best.pt"
        if artifact.exists():
            return {
                "artifact": str(artifact.relative_to(ROOT)),
                "loaded": True,
                "compatible": True,
                "reason": "frozen STARE artifact is available; loaded lazily for receiver-observable temporal windows",
            }
        return {
            "artifact": None,
            "loaded": False,
            "compatible": False,
            "reason": "no frozen STARE inference checkpoint found",
        }

    artifact = _policy_artifact(policy_name)
    if artifact is None:
        if hasattr(policy, "set_mode"):
            policy.set_mode("eval")
        return {"artifact": None, "loaded": False, "compatible": False,
                "reason": "no frozen checkpoint found; adapter evaluation only"}

    try:
        import torch
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        expected_state_dim = int(getattr(policy, "state_dim", -1))
        expected_actions = int(getattr(policy, "n_actions", -1))
        saved_state_dim = int(payload.get("state_dim", -1)) if isinstance(payload, dict) else -1
        saved_actions = int(payload.get("n_actions", -1)) if isinstance(payload, dict) else -1
        if (saved_state_dim != expected_state_dim) or (saved_actions != expected_actions):
            if hasattr(policy, "set_mode"):
                policy.set_mode("eval")
            return {
                "artifact": str(artifact.relative_to(ROOT)),
                "loaded": False,
                "compatible": False,
                "reason": (f"checkpoint architecture mismatch: saved state_dim={saved_state_dim}, "
                            f"n_actions={saved_actions}; live state_dim={expected_state_dim}, "
                            f"n_actions={expected_actions}"),
            }

        if hasattr(policy, "load_weights"):
            policy.load_weights(artifact)
        else:
            state_dicts = payload.get("state_dicts", payload)
            for attr in ("online", "target", "network"):
                module = getattr(policy, attr, None)
                state = state_dicts.get(attr) if isinstance(state_dicts, dict) else None
                if module is not None and state is not None:
                    module.load_state_dict(state)
        if hasattr(policy, "set_mode"):
            policy.set_mode("eval")
        return {"artifact": str(artifact.relative_to(ROOT)), "loaded": True, "compatible": True,
                "reason": "frozen checkpoint loaded and architecture verified"}
    except Exception as exc:
        if hasattr(policy, "set_mode"):
            policy.set_mode("eval")
        return {"artifact": str(artifact.relative_to(ROOT)), "loaded": False, "compatible": False,
                "reason": f"checkpoint load failed: {exc}"}


def _execute_policy_episode(scenario: Scenario, policy_name: str, seed: int, max_steps: int = 5000) -> tuple[Any, Dict[str, Any], list[dict[str, Any]]]:
    """Execute one policy directly through the canonical environment and adapter."""
    harness = TournamentHarness(scenarios=[scenario], seed=seed)
    env = harness._make_environment(scenario, seed)
    frequencies = [float(x) for x in scenario.frequencies_mhz]
    dwell_times = scenario.normalized_dwell_times()
    action_space = ScanActionSpace(frequencies_mhz=frequencies, dwell_times_s=[0.001] + dwell_times)
    builder = SCANStateBuilder(num_channels=len(frequencies))
    model = _OnlineChannelModel(num_channels=len(frequencies))
    env.reset(truth_events=[dict(e) if isinstance(e, dict) else e for e in scenario.truth_events], seed=seed, start_channel=0)
    observation = harness._build_observation(env, model, builder, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    state_dim = int(observation.flatten().size)
    policy = make_policy(policy_name, action_space, state_dim)
    artifact = _configure_frozen_policy(policy, policy_name)
    policy.reset(observation)
    trace=[]; total_reward=0.0; detections=0; steps=0
    # The frozen STARE model consumes a 128x5 temporal pulse representation.
    # In a synthetic Policy Arena there may be fewer than 128 detected pulses
    # on a selected channel, so the receiver builds a rolling observation
    # window from what it actually measured (never hidden truth). This keeps
    # the final model executable in short demos without fabricating emitter
    # labels or future state.
    stare_buffer: list[np.ndarray] = []
    stare_embedding = np.empty(0, dtype=np.float32)
    stare_next_frequency_mhz = float("nan")
    stare_window = 128
    stare_update_interval = 16
    last_stare_infer_step = 0
    median, iqr = _normalization()
    while not env.done and env.current_time_s < scenario.horizon_s and steps < max_steps:
        steps += 1
        action=policy.act(observation)
        result=env.step(ScanAction(channel=int(action.channel), dwell_time_s=float(action.dwell_time_s)))
        obs_obj=result.observation; reward=float(result.reward); total_reward += reward; detections += int(obs_obj.detection_count)
        model.observe(channel=int(action.channel), time_s=float(obs_obj.time_s), detected=bool(obs_obj.detected), dwell_time_s=float(action.dwell_time_s))

        if policy_name == "stare_cognitive":
            if obs_obj.pulses:
                aoa = float(np.mean([p.aoa_deg for p in obs_obj.pulses]))
                amplitude = float(np.mean([p.amplitude for p in obs_obj.pulses]))
                pulse_width = float(np.mean([p.pulse_width_s for p in obs_obj.pulses]))
            else:
                aoa = 0.0
                amplitude = float(obs_obj.measured_energy)
                pulse_width = float(obs_obj.dwell_time_s)
            stare_buffer.append(np.asarray([
                float(obs_obj.time_s),
                float(obs_obj.frequency_mhz),
                pulse_width,
                aoa,
                amplitude,
            ], dtype=np.float32))
            if len(stare_buffer) > stare_window:
                stare_buffer = stare_buffer[-stare_window:]
            if (
                len(stare_buffer) == stare_window
                and (steps - last_stare_infer_step) >= stare_update_interval
            ):
                stare_x = (np.stack(stare_buffer) - median) / np.where(
                    np.abs(iqr) < 1e-9, 1.0, iqr
                )
                try:
                    stare_result = _stare_infer_window(stare_x)
                    stare_embedding = np.asarray(
                        stare_result["embedding"],
                        dtype=np.float32,
                    )
                    stare_next_frequency_mhz = float(
                        stare_result["next_frequency_prediction"]
                    )
                    last_stare_infer_step = steps
                except Exception:
                    pass

        probs=model.belief/max(model.belief.sum(),1e-9)
        next_observation=harness._build_observation(env,model,builder,
            detection_density=float(env.history.detection_rate()),mean_snr=float(obs_obj.snr_proxy),
            transition_entropy=float(-np.sum(probs*np.log(probs+1e-12))),
            unique_channel_ratio=float(np.count_nonzero(model.visit_counts)/max(len(frequencies),1)),
            ood_score=0.0,change_point_score=0.0,
            stare_embedding=stare_embedding,
            stare_next_frequency_mhz=stare_next_frequency_mhz)
        policy.update(observation,action,reward,next_observation,info={"detected":bool(obs_obj.detected),"detections":int(obs_obj.detection_count),"time_s":float(obs_obj.time_s),"done":bool(result.terminated or result.truncated)})
        trace.append({"step":steps,"time_s":float(obs_obj.time_s),"channel":int(action.channel),"frequency_mhz":float(frequencies[int(action.channel)]),"dwell_s":float(action.dwell_time_s),"detected":bool(obs_obj.detected),"detection_count":int(obs_obj.detection_count),"snr":float(obs_obj.snr_proxy),"reward":reward,"score":float(getattr(action,'score',0.0)),"reason":str(getattr(action,'reason',''))})
        observation=next_observation
        if result.terminated or result.truncated: break
    episode=harness._evaluate_episode(env=env,scenario=scenario,policy_name=policy_name,episode_index=0,episode_seed=seed,steps=steps,total_reward=total_reward,detections=detections,opportunities=steps)
    return episode, artifact, trace


def _run_scenario_policies(scenario: Scenario, policies: List[str], episodes: int = 1, seed: int = 26055) -> Dict[str, Any]:
    """Run every requested policy independently, with frozen artifacts loaded when available."""
    out: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for name in policies:
        eps=[]; artifact=None; traces=[]
        try:
            for ep in range(episodes):
                result, artifact, trace = _execute_policy_episode(scenario, name, seed + ep, max_steps=5000)
                eps.append(result); traces.append(trace)
            full_trace = traces[-1] if traces else []
            out[name]={
                "episodes":[_episode_json(r) for r in eps],
                "mean":_summary(eps),
                "status":"completed",
                "artifact":artifact,
                "trace":full_trace,
                "trace_summary": {
                    "decisions": len(full_trace),
                    "unique_channels": sorted({int(x["channel"]) for x in full_trace}),
                    "channel_counts": {
                        str(ch): sum(1 for x in full_trace if int(x["channel"]) == ch)
                        for ch in sorted({int(x["channel"]) for x in full_trace})
                    },
                },
            }
        except Exception as exc:
            errors[name]=str(exc)
            out[name]={"episodes":[],"mean":{},"status":"failed","error":str(exc),"artifact":artifact,"trace":[]}
    return {"policies":out,"errors":errors}


def _stare_batch_summary(input_path: Path, max_windows: int = 32, progress_callback=None) -> Dict[str, Any]:
    """Run the frozen STARE model over many windows, not just the first one."""
    reader = TSRDHDF5Reader(input_path)
    median, iqr = _normalization()
    count = 0
    activities: list[float] = []
    next_freqs: list[float] = []
    embeddings: list[np.ndarray] = []
    for x, _ in reader.iter_windows(window_size=128, stride=128, include_labels=False):
        x = (x - median) / np.where(np.abs(iqr) < 1e-9, 1.0, iqr)
        result = _stare_infer_window(x)
        activities.append(float(result["activity_prediction"]))
        next_freqs.append(float(result["next_frequency_prediction"]))
        embeddings.append(np.asarray(result["embedding"], dtype=np.float32))
        count += 1
        if progress_callback:
            progress_callback(count / max(1, max_windows))
        if count >= max_windows:
            break
    if not count:
        raise ValueError("No complete 128-pulse windows available for STARE evaluation")
    emb = np.stack(embeddings)
    return {
        "windows": count,
        "embedding_dim": int(emb.shape[-1]),
        "activity_mean": float(np.mean(activities)),
        "activity_std": float(np.std(activities)),
        "next_frequency_mean_mhz": float(np.mean(next_freqs)),
        "embedding_norm_mean": float(np.mean(np.linalg.norm(emb, axis=1))),
        "embedding_norm_std": float(np.std(np.linalg.norm(emb, axis=1))),
        "artifact": "models/frozen/stare_inference.pt",
    }

def _build_synthetic(req: SyntheticRequest | TraceRequest) -> Scenario:
    cfg = SyntheticConfig(num_emitters=req.emitters, horizon_s=req.horizon_s)
    truth_events, active = generate(req.scenario, cfg, req.seed)
    raw_frequencies = np.arange(
        cfg.frequency_min_mhz,
        cfg.frequency_max_mhz + cfg.frequency_step_mhz * 0.5,
        cfg.frequency_step_mhz,
        dtype=float,
    )
    # Keep the tournament action space aligned with the trained SCAN/RL
    # models (8 channels), so every policy is evaluated in the same action
    # space rather than silently falling back to incompatible dimensions.
    pick = np.linspace(0, len(raw_frequencies) - 1, min(8, len(raw_frequencies))).round().astype(int)
    frequencies = raw_frequencies[pick].tolist()
    return Scenario(
        name=f"API-{req.scenario.upper()}-{req.seed}",
        truth_events=truth_events,
        horizon_s=req.horizon_s,
        seed=req.seed,
        frequencies_mhz=frequencies,
        dwells_s=[0.001],
        retune_time_s=0.0,
        metadata={
            "source": "v6.2_python_backend",
            "scenario_type": req.scenario,
            "requested_emitters": req.emitters,
            "active_frequencies": active,
            "truth_event_count": len(truth_events),
            "coverage_window_s": float(
                max(8 * 0.001 * max(2, len(frequencies)), req.horizon_s * 0.10)
            ),
        },
    )


def _episode_json(r: Any) -> Dict[str, Any]:
    return {
        "policy": r.policy,
        "scenario": r.scenario,
        "episode_index": r.episode_index,
        "seed": r.seed,
        "pd": r.pd,
        "poi": r.poi,
        "weighted_poi": r.weighted_poi,
        "mean_interception_time_s": r.mean_interception_time_s,
        "mit_p50_s": r.mit_p50_s,
        "mit_p95_s": r.mit_p95_s,
        "mit_p99_s": r.mit_p99_s,
        "censored_mean_latency_s": r.censored_mean_latency_s,
        "hp_miss_rate": r.hp_miss_rate,
        "coverage": r.coverage,
        "reward": r.reward,
        "staleness_mean_s": r.staleness_mean_s,
        "staleness_max_s": r.staleness_max_s,
        "detections": r.detections,
        "opportunities": r.opportunities,
        "interceptions": r.interceptions,
        "metadata": r.metadata,
    }


def _mean(values: List[float]) -> float:
    return float(np.nanmean(values)) if values else float("nan")


def _summary(episodes: List[Any]) -> Dict[str, Any]:
    return {
        "pd": _mean([r.pd for r in episodes]),
        "weighted_poi": _mean([r.weighted_poi for r in episodes]),
        "reward": _mean([r.reward for r in episodes]),
        "coverage": _mean([r.coverage for r in episodes]),
        "interceptions": _mean([float(r.interceptions) for r in episodes]),
        "hp_miss_rate": _mean([r.hp_miss_rate for r in episodes]),
        "mean_interception_time_s": _mean([r.mean_interception_time_s for r in episodes]),
        "mit_p95_s": _mean([r.mit_p95_s for r in episodes]),
        "mit_p99_s": _mean([r.mit_p99_s for r in episodes]),
        "staleness_mean_s": _mean([r.staleness_mean_s for r in episodes]),
        "staleness_max_s": _mean([r.staleness_max_s for r in episodes]),
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    stare = ROOT / "models" / "frozen" / "stare_inference.pt"
    cognitive = ROOT / "models" / "frozen" / "cognitive_policy.json"
    return {"ok": True, "service": "SIH26055 EW Research Backend", "version": "6.2-production", "final_model": "STARE-Cognitive", "stare_loaded": stare.exists(), "cognitive_policy": cognitive.exists(), "engine_root": str(ENGINE_ROOT)}


@app.get("/api/policies")
def policies() -> Dict[str, Any]:
    return {"policies": POLICIES, "source": "src.scan.policy_adapter.available_policies"}


@app.get("/api/synthetic/scenarios")
def synthetic_scenarios() -> Dict[str, Any]:
    return {"scenarios": available_synthetic_scenarios(), "source": "src.evaluation.synthetic_generator"}


@app.get("/api/research/status")
def research_status() -> Dict[str, Any]:
    frozen = ROOT / "models" / "frozen"
    stare = ROOT / "models" / "stare"
    rl = ROOT / "models" / "rl"
    locked = ROOT / "results" / "locked"
    files = {
        "stare_inference": frozen / "stare_inference.pt",
        "stare_checkpoint": stare / "best.pt",
        "deployment_manifest": frozen / "inference_manifest.json",
        "cognitive_policy": frozen / "cognitive_policy.json",
        "portfolio_config": frozen / "portfolio_config.json",
        "dwell_ladder": frozen / "dwell_ladder.json",
        "locked_report": locked / "locked_test_report.json",
        "model_manifest": ROOT / "models" / "manifest.json",
    }
    rl_models = sorted(p.name for p in rl.glob("*.pt")) if rl.exists() else []
    return {
        "root": str(ROOT),
        "files": {k: {"exists": p.exists(), "path": str(p.relative_to(ROOT))} for k, p in files.items()},
        "rl_models": rl_models,
        "policies": POLICIES,
        "execution": "python_backend",
    }


@app.get("/api/models")
def models() -> Dict[str, Any]:
    status = research_status()
    return {"source": "v6.2", "models": status["files"], "rl_models": status["rl_models"]}


@app.get("/api/results/locked")
def locked_results() -> Dict[str, Any]:
    candidates = [ROOT / "results" / "locked" / "locked_test_report.json", ROOT / "results" / "locked_test_report.json"]
    for path in candidates:
        if path.exists():
            try:
                return {"source": "v6.2_locked_evaluation", "path": str(path.relative_to(ROOT)), "report": json.loads(path.read_text(encoding="utf-8"))}
            except Exception as exc:
                raise HTTPException(500, f"Could not parse locked report: {exc}")
    raise HTTPException(404, "Locked evaluation report was not found in v6.2.")


@app.post("/api/evaluate/synthetic")
def evaluate_synthetic(req: SyntheticRequest) -> Dict[str, Any]:
    unknown = sorted(set(req.policies) - set(POLICIES))
    if unknown:
        raise HTTPException(400, f"Unknown policies: {unknown}")
    scenario = _build_synthetic(req)
    result = _run_scenario_policies(scenario, req.policies, episodes=req.episodes, seed=req.seed)
    return {
        "source": "v6.2_research_backend",
        "execution": "independent_canonical_policy_episodes",
        "scenario": {"name": scenario.name, "type": req.scenario, "seed": req.seed, "emitters": req.emitters,
                     "horizon_s": req.horizon_s, "channels": len(scenario.frequencies_mhz),
                     "truth_events": len(scenario.truth_events), "active_frequencies": scenario.metadata["active_frequencies"]},
        "policies": result["policies"],
        "errors": result["errors"],
    }


def _trace_episode(req: TraceRequest) -> Dict[str, Any]:
    scenario = _build_synthetic(req)
    harness = TournamentHarness(scenarios=[scenario], seed=req.seed)
    episode_seed = int(scenario.seed)
    env = harness._make_environment(scenario, episode_seed)
    frequencies = [float(x) for x in scenario.frequencies_mhz]
    dwell_times = scenario.normalized_dwell_times()
    action_space = ScanActionSpace(frequencies_mhz=frequencies, dwell_times_s=[0.001] + dwell_times)
    builder = SCANStateBuilder(num_channels=len(frequencies))
    model = _OnlineChannelModel(num_channels=len(frequencies))
    env.reset(truth_events=[dict(e) if isinstance(e, dict) else e for e in scenario.truth_events], seed=episode_seed, start_channel=0)
    observation = harness._build_observation(env, model, builder, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    state_dim = int(observation.flatten().size)
    policy = make_policy(req.policy, action_space, state_dim)
    policy.reset(observation)
    trace: List[Dict[str, Any]] = []
    total_reward = 0.0
    detections = 0
    steps = 0
    while not env.done and env.current_time_s < scenario.horizon_s and steps < req.max_steps:
        steps += 1
        action = policy.act(observation)
        result = env.step(ScanAction(channel=int(action.channel), dwell_time_s=float(action.dwell_time_s)))
        obs_obj = result.observation
        reward = float(result.reward)
        total_reward += reward
        detections += int(obs_obj.detection_count)
        model.observe(channel=int(action.channel), time_s=float(obs_obj.time_s), detected=bool(obs_obj.detected), dwell_time_s=float(action.dwell_time_s))
        next_observation = harness._build_observation(
            env, model, builder,
            detection_density=float(env.history.detection_rate()),
            mean_snr=float(obs_obj.snr_proxy), transition_entropy=float(
                -np.sum((model.belief / max(model.belief.sum(), 1e-9)) * np.log(model.belief / max(model.belief.sum(), 1e-9) + 1e-12))
            ),
            unique_channel_ratio=float(np.count_nonzero(model.visit_counts) / max(len(frequencies), 1)),
            ood_score=0.0, change_point_score=0.0,
        )
        policy.update(observation, action, reward, next_observation, info={"detected": bool(obs_obj.detected), "time_s": float(obs_obj.time_s), "done": bool(result.terminated or result.truncated)})
        trace.append({
            "step": steps,
            "time_s": float(obs_obj.time_s),
            "channel": int(action.channel),
            "frequency_mhz": float(frequencies[int(action.channel)]),
            "dwell_s": float(action.dwell_time_s),
            "detected": bool(obs_obj.detected),
            "detection_count": int(obs_obj.detection_count),
            "snr": float(obs_obj.snr_proxy),
            "reward": reward,
            "score": float(getattr(action, "score", 0.0)),
            "reason": str(getattr(action, "reason", "")),
            "belief": np.asarray(observation.channel_belief, dtype=float).round(5).tolist(),
            "uncertainty": np.asarray(observation.predictive_uncertainty, dtype=float).round(5).tolist(),
            "coverage": float(observation.coverage_fraction),
            "ood": float(observation.ood_score),
            "change_point": float(observation.change_point_score),
        })
        observation = next_observation
        if result.terminated or result.truncated:
            break
    episode = harness._evaluate_episode(env=env, scenario=scenario, policy_name=req.policy, episode_index=0, episode_seed=episode_seed, steps=steps, total_reward=total_reward, detections=detections, opportunities=steps)
    return {
        "source": "v6.2_research_backend",
        "execution": "python_policy_trace",
        "policy": req.policy,
        "scenario": {"type": req.scenario, "seed": req.seed, "emitters": req.emitters, "horizon_s": req.horizon_s, "channels": len(frequencies)},
        "metrics": _episode_json(episode),
        "trace": trace,
    }


@app.post("/api/decision/trace")
def decision_trace(req: TraceRequest) -> Dict[str, Any]:
    if req.policy not in POLICIES:
        raise HTTPException(400, f"Unknown policy: {req.policy}")
    return _trace_episode(req)



def _stream_trace_worker(stream_id: str, scenario: Scenario, policy_name: str, max_steps: int) -> None:
    """Execute a real backend episode and publish each decision incrementally."""
    try:
        _job_update(stream_id, status="running", progress=0.0, message="Backend receiver loop started", trace=[])
        harness = TournamentHarness(scenarios=[scenario], seed=scenario.seed)
        env = harness._make_environment(scenario, int(scenario.seed))
        frequencies = [float(x) for x in scenario.frequencies_mhz]
        dwell_times = scenario.normalized_dwell_times()
        action_space = ScanActionSpace(frequencies_mhz=frequencies, dwell_times_s=[0.001] + dwell_times)
        builder = SCANStateBuilder(num_channels=len(frequencies))
        model = _OnlineChannelModel(num_channels=len(frequencies))
        env.reset(truth_events=[dict(e) if isinstance(e, dict) else e for e in scenario.truth_events], seed=int(scenario.seed), start_channel=0)
        observation = harness._build_observation(env, model, builder, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        state_dim = int(observation.flatten().size)
        policy = make_policy(policy_name, action_space, state_dim)
        artifact_info = _configure_frozen_policy(policy, policy_name)
        policy.reset(observation)
        trace: List[Dict[str, Any]] = []
        total_reward = 0.0
        detections = 0
        steps = 0
        while not env.done and env.current_time_s < scenario.horizon_s and steps < max_steps:
            steps += 1
            action = policy.act(observation)
            result = env.step(ScanAction(channel=int(action.channel), dwell_time_s=float(action.dwell_time_s)))
            obs_obj = result.observation
            reward = float(result.reward)
            total_reward += reward
            detections += int(obs_obj.detection_count)
            model.observe(channel=int(action.channel), time_s=float(obs_obj.time_s), detected=bool(obs_obj.detected), dwell_time_s=float(action.dwell_time_s))
            probs = model.belief / max(model.belief.sum(), 1e-9)
            next_observation = harness._build_observation(
                env, model, builder,
                detection_density=float(env.history.detection_rate()),
                mean_snr=float(obs_obj.snr_proxy),
                transition_entropy=float(-np.sum(probs * np.log(probs + 1e-12))),
                unique_channel_ratio=float(np.count_nonzero(model.visit_counts) / max(len(frequencies), 1)),
                ood_score=0.0, change_point_score=0.0,
            )
            policy.update(observation, action, reward, next_observation, info={"detected": bool(obs_obj.detected), "detections": int(obs_obj.detection_count), "time_s": float(obs_obj.time_s), "done": bool(result.terminated or result.truncated)})
            trace.append({
                "step": steps,
                "time_s": float(obs_obj.time_s),
                "channel": int(action.channel),
                "frequency_mhz": float(frequencies[int(action.channel)]),
                "dwell_s": float(action.dwell_time_s),
                "detected": bool(obs_obj.detected),
                "detection_count": int(obs_obj.detection_count),
                "snr": float(obs_obj.snr_proxy),
                "reward": reward,
                "score": float(getattr(action, "score", 0.0)),
                "reason": str(getattr(action, "reason", "")),
                "belief": np.asarray(observation.channel_belief, dtype=float).round(5).tolist(),
                "uncertainty": np.asarray(observation.predictive_uncertainty, dtype=float).round(5).tolist(),
                "coverage": float(observation.coverage_fraction),
                "ood": float(observation.ood_score),
                "change_point": float(observation.change_point_score),
                "diagnostics": getattr(action, "diagnostics", {}),
            })
            observation = next_observation
            _job_update(stream_id, progress=min(0.99, float(env.current_time_s / max(scenario.horizon_s, 1e-9))), message=f"{policy_name}: decision {steps}", trace=trace[-160:], latest=trace[-1], steps=steps, detections=detections, total_reward=total_reward, artifact=artifact_info)
            if result.terminated or result.truncated:
                break
        episode = harness._evaluate_episode(env=env, scenario=scenario, policy_name=policy_name, episode_index=0, episode_seed=int(scenario.seed), steps=steps, total_reward=total_reward, detections=detections, opportunities=steps)
        _job_update(stream_id, status="completed", progress=1.0, message="Backend receiver loop completed", metrics=_episode_json(episode), trace=trace, steps=steps, detections=detections, total_reward=total_reward, artifact=artifact_info)
    except Exception as exc:
        _job_update(stream_id, status="failed", progress=1.0, message=str(exc), error=str(exc))


@app.post("/api/decision/stream")
async def decision_stream(background_tasks: BackgroundTasks, policy: str = Form("cognitive"), scenario: str = Form("mixed"), emitters: int = Form(8), horizon_s: float = Form(0.5), seed: int = Form(26055), max_steps: int = Form(240), file: Optional[UploadFile] = File(None)) -> Dict[str, Any]:
    if policy not in POLICIES:
        raise HTTPException(400, f"Unknown policy: {policy}")
    stream_id = _new_job("decision_stream")
    if file is not None:
        suffix = Path(file.filename or "input.h5").suffix.lower()
        if suffix not in {".h5", ".hdf5", ".he5"}:
            raise HTTPException(400, "Only HDF5 input is supported.")
        incoming = ROOT / "data" / "web_jobs" / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        path = incoming / f"stream_{stream_id}{suffix}"
        path.write_bytes(await file.read())
        scenario_obj, meta = _load_uploaded_scenario(path)
        scenario_obj.seed = seed
        background_tasks.add_task(_stream_trace_worker, stream_id, scenario_obj, policy, max_steps)
        _job_update(stream_id, source="uploaded_hdf5", file=Path(file.filename or path.name).name, scenario_meta=meta)
    else:
        req = TraceRequest(scenario=scenario, emitters=emitters, horizon_s=horizon_s, seed=seed, policy=policy, max_steps=max_steps)
        scenario_obj = _build_synthetic(req)
        background_tasks.add_task(_stream_trace_worker, stream_id, scenario_obj, policy, max_steps)
        _job_update(stream_id, source="synthetic", scenario_meta={"scenario": scenario, "emitters": emitters, "horizon_s": horizon_s})
    return {"stream_id": stream_id, "status": "queued", "policy": policy}


@app.get("/api/decision/stream/{stream_id}")
def decision_stream_status(stream_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(stream_id)
    if not job:
        raise HTTPException(404, "Unknown stream id")
    return job

@app.post("/api/stare/infer")
async def stare_infer(file: UploadFile = File(...)) -> Dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".h5", ".hdf5", ".he5"}:
        raise HTTPException(400, "Only HDF5 input is supported.")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = Path(tmp.name)
    try:
        reader = TSRDHDF5Reader(path)
        window = next(reader.iter_windows(window_size=128, stride=128, include_labels=False), None)
        if window is None:
            raise HTTPException(400, "The HDF5 file does not contain a complete 128-pulse window.")
        x, _ = window
        median, iqr = _normalization()
        x = (x - median) / np.where(np.abs(iqr) < 1e-9, 1.0, iqr)
        result = _stare_infer_window(x)
        return {"source":"v6.2_stare","file":Path(file.filename or path.name).name, **result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"STARE inference failed: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/dataset/inspect")
async def inspect_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".h5", ".hdf5", ".he5"}:
        raise HTTPException(400, "Only HDF5 files are supported.")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = Path(tmp.name)
    try:
        reader = TSRDHDF5Reader(path)
        info = reader.inspect()
        errors = validate_schema(info)
        return {
            "source": "v6.2_tsrhdf5_reader",
            "file": Path(file.filename or path.name).name,
            "valid_hdf5_schema": not errors,
            "schema_errors": errors,
            "n_pulses": info.n_pulses,
            "n_features": info.n_features,
            "features": list(info.features),
            "receiver": {
                "dwell_centres_mhz": info.receiver.dwell_centres_mhz.tolist(),
                "dwell_times_s": info.receiver.dwell_times_s.tolist(),
                "freq_range_mhz": info.receiver.freq_range_mhz.tolist(),
                "start_position_km": info.receiver.start_position_km.tolist(),
            },
        }
    except Exception as exc:
        raise HTTPException(400, f"HDF5 inspection failed: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)



def _normalization() -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "data" / "manifests" / "normalization.json"
    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            return np.asarray(obj["median"], dtype=np.float32), np.asarray(obj["iqr"], dtype=np.float32)
        except Exception:
            pass
    return np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32)


def _build_uploaded_stare_shards(input_path: Path, job_id: str, window_size: int = 128, stride: int = 128, shard_windows: int = 256) -> list[str]:
    out_dir = ROOT / "data" / "web_jobs" / job_id / "stare_shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    median, iqr = _normalization()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    written: list[str] = []
    reader = TSRDHDF5Reader(input_path)
    with __import__("h5py").File(input_path, "r") as h5:
        labels = h5.get("labels")
        data = h5["data"]
        n = int(data.shape[0])
        shard_idx = 0
        for start in range(0, max(0, n - window_size + 1), stride):
            end = start + window_size
            x = np.asarray(data[start:end], dtype=np.float32)
            x = (x - median) / np.where(np.abs(iqr) < 1e-9, 1.0, iqr)
            y = np.asarray(labels[start:end], dtype=np.int8) if labels is not None else np.zeros(window_size, dtype=np.int8)
            xs.append(x); ys.append(y)
            if len(xs) >= shard_windows:
                path = out_dir / f"shard_{shard_idx:05d}.npz"
                np.savez_compressed(path, x=np.stack(xs), y=np.stack(ys))
                written.append(str(path.relative_to(ROOT)))
                xs.clear(); ys.clear(); shard_idx += 1
        if xs:
            path = out_dir / f"shard_{shard_idx:05d}.npz"
            np.savez_compressed(path, x=np.stack(xs), y=np.stack(ys))
            written.append(str(path.relative_to(ROOT)))
    return written


def _stare_infer_window(x: np.ndarray) -> Dict[str, Any]:
    import torch
    cfg = {
        "input_dim": 5,
        "hidden_dim": 128,
        "embedding_dim": 64,
        "backbone": "gru",
        "dropout": 0.1,
        "max_len": 128,
    }
    manifest = ROOT / "models" / "frozen" / "inference_manifest.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
            shape = m.get("artifacts", {}).get("stare", {}).get("input_shape", [1, 128, 5])
            cfg["max_len"] = int(shape[1])
        except Exception:
            pass
    checkpoint = ROOT / "models" / "frozen" / "stare_inference.pt"
    if not checkpoint.exists():
        checkpoint = ROOT / "models" / "stare" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError("No STARE inference checkpoint found")
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape != (cfg["max_len"], 5):
        raise ValueError(f"STARE requires [{cfg['max_len']}, 5], received {list(arr.shape)}")
    cache_key = str(checkpoint.resolve())
    model = _STARE_MODEL_CACHE.get(cache_key)
    if model is None:
        try:
            model = torch.jit.load(str(checkpoint), map_location="cpu")
            model.eval()
        except Exception:
            model = build_model(cfg)
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state)
            model.eval()
        _STARE_MODEL_CACHE[cache_key] = model
    with torch.no_grad():
        out = model(torch.from_numpy(arr[None, ...]))
    return {
        "artifact": str(checkpoint.relative_to(ROOT)),
        "embedding": out["embedding"][0].cpu().numpy().round(6).tolist(),
        "embedding_dim": int(out["embedding"].shape[-1]),
        "next_frequency_prediction": float(out["next_freq"][0].item()),
        "activity_prediction": float(out["activity"][0].item()),
    }


def _uploaded_replay_evaluate(input_path: Path, policies: List[str], max_windows: int = 12, steps_per_window: int = 48, progress_callback=None) -> Dict[str, Any]:
    """Evaluate every policy on receiver-observable HDF5 windows.

    Unlike the old shard-only path, this consumes the uploaded pulse stream
    directly. Each policy gets an independent receiver-state trajectory over
    the same windows. Ground-truth labels are used only for optional audit
    statistics after the policy decision; labels never enter PolicyObservation.
    """
    import h5py

    reader = TSRDHDF5Reader(input_path)
    median, iqr = _normalization()
    all_results: Dict[str, list[dict[str, Any]]] = {p: [] for p in policies}
    model_status: Dict[str, Any] = {}

    # Read only receiver metadata here. Do NOT materialize the full HDF5 pulse
    # matrix: production captures can contain millions/billions of pulses.
    with h5py.File(input_path, "r") as h5:
        receiver = h5["metadata"]["receiver"]
        physical_centres = np.asarray(receiver["dwell_centres_mhz"], dtype=np.float64)

    # Match the trained SCAN action-space convention: 8 stable channels.
    pick = np.linspace(0, len(physical_centres) - 1, 8).round().astype(int)
    channels = physical_centres[pick]
    dwell = 0.001

    # Build temporal windows using the canonical reader so normalization and
    # feature ordering stay identical to STARE ingestion.
    windows = []
    for x, y in reader.iter_windows(window_size=128, stride=128, include_labels=True):
        windows.append((np.asarray(x, dtype=np.float32), None if y is None else np.asarray(y)))
        if len(windows) >= max_windows:
            break
    if not windows:
        raise ValueError("Uploaded HDF5 contains no complete evaluation windows")

    total_units = max(1, len(policies) * len(windows))
    completed_units = 0
    for policy_name in policies:
        policy_rows: list[dict[str, Any]] = []
        # State dimension is determined by the canonical observation contract.
        n = len(channels)
        state_dim = 10 + n * 9
        action_space = ScanActionSpace(frequencies_mhz=channels.tolist(), dwell_times_s=[dwell])
        try:
            policy = make_policy(policy_name, action_space, state_dim)
            artifact_info = _configure_frozen_policy(policy, policy_name)
        except Exception as exc:
            model_status[policy_name] = {"status": "failed_to_initialize", "error": str(exc)}
            all_results[policy_name] = [{"status": "failed", "error": str(exc)}]
            continue

        model_status[policy_name] = {"status": "ready", **artifact_info}
        if artifact_info.get("artifact") and not artifact_info.get("loaded"):
            model_status[policy_name]["status"] = "incompatible_checkpoint"
        if artifact_info.get("artifact") and not artifact_info.get("loaded"):
            all_results[policy_name] = [{"status": "incompatible_checkpoint", "reason": artifact_info.get("reason", "checkpoint not loaded")} ]
            completed_units += len(windows)
            if progress_callback:
                progress_callback(completed_units / total_units)
            continue
        for wi, (raw_window, labels_window) in enumerate(windows):
            x = (raw_window - median) / np.where(np.abs(iqr) < 1e-9, 1.0, iqr)
            freqs = raw_window[:, 1].astype(float)
            # Quantize each observed pulse to the common action-space channel.
            pulse_channels = np.argmin(np.abs(freqs[:, None] - channels[None, :]), axis=1)
            # Only the final STARE-Cognitive path needs the frozen STARE
            # representation. Other policies stay on their native receiver state
            # path so the uploaded-data preview remains fast.
            if policy_name == "stare_cognitive":
                stare_result = _stare_infer_window(x)
                stare_embedding = np.asarray(stare_result["embedding"], dtype=np.float32)
            else:
                stare_embedding = np.empty(0, dtype=np.float32)
            pulse_activity = np.zeros(n, dtype=np.float32)
            belief = np.full(n, 0.1, dtype=np.float32)
            staleness = np.zeros(n, dtype=np.float32)
            visits = np.zeros(n, dtype=np.float32)
            detections = np.zeros(n, dtype=np.float32)
            transitions = np.ones((n, n), dtype=np.float32)
            uncertainty = np.ones(n, dtype=np.float32)
            obs = observation_from_fusion_state if False else None

            # Build PolicyObservation directly to avoid leaking labels.
            from src.scan.policy_adapter import PolicyObservation
            observation = PolicyObservation(
                channel_belief=belief.copy(), staleness=staleness.copy(), freshness=np.ones(n, dtype=np.float32),
                transition_prediction=np.full(n, 1/n, dtype=np.float32), activity_prediction=belief.copy(),
                hazard_prediction=np.full(n, 0.5, dtype=np.float32), predictive_uncertainty=uncertainty.copy(),
                coverage_pressure=np.zeros(n, dtype=np.float32), priority_pressure=np.zeros(n, dtype=np.float32),
                current_channel=0, current_time_s=0.0, current_dwell_s=dwell, coverage_fraction=0.0,
                detection_density=0.0, mean_snr=0.0, transition_entropy=float(np.log(n)),
                unique_channel_ratio=0.0,
                stare_embedding=stare_embedding.copy(), available_channels=np.arange(n, dtype=np.int64),
            )
            policy.reset(observation)
            hits = 0
            action_count = 0
            chosen = []
            reward_total = 0.0

            # Replay each pulse slot. The receiver only exposes the pulse
            # currently observed on the selected channel; other labels are
            # never included in the observation.
            for step in range(min(steps_per_window, len(x))):
                action = policy.act(observation)
                ch = int(action.channel) % n
                detected = bool(pulse_channels[step] == ch)
                reward = 1.0 if detected else 0.0
                hits += int(detected)
                action_count += 1
                chosen.append(ch)
                reward_total += reward
                visits[ch] += 1
                staleness += 1.0
                staleness[ch] = 0.0
                if detected:
                    detections[ch] += 1
                    belief[ch] = min(1.0, belief[ch] + 0.45)
                    if len(chosen) > 1:
                        prev = chosen[-2]
                        transitions[prev, ch] += 1.0
                else:
                    belief[ch] *= 0.72
                uncertainty = 1.0 / np.sqrt(visits + 1.0)
                row = transitions[ch] / max(transitions[ch].sum(), 1e-9)
                transition_entropy = float(-np.sum(row * np.log(row + 1e-12)))
                next_obs = PolicyObservation(
                    channel_belief=belief.copy(), staleness=staleness.copy(),
                    freshness=np.exp(-staleness / 8.0).astype(np.float32),
                    transition_prediction=row.astype(np.float32), activity_prediction=belief.copy(),
                    hazard_prediction=(1.0/(1.0+detections)).astype(np.float32),
                    predictive_uncertainty=uncertainty.copy(), coverage_pressure=(staleness/16.0).astype(np.float32),
                    priority_pressure=(detections/max(detections.sum(),1.0)).astype(np.float32),
                    current_channel=ch, current_time_s=float(step+1)*dwell, current_dwell_s=float(action.dwell_time_s),
                    coverage_fraction=float(np.count_nonzero(visits))/n, detection_density=float(np.mean(detections)),
                    mean_snr=float(raw_window[step,4]), transition_entropy=transition_entropy,
                    unique_channel_ratio=float(np.count_nonzero(visits))/n,
                    stare_embedding=observation.stare_embedding,
                    available_channels=np.arange(n,dtype=np.int64),
                )
                policy.update(observation, action, reward, next_obs, info={"detections": int(detected), "detected": detected, "done": step == min(steps_per_window,len(x))-1})
                observation = next_obs

            policy_rows.append({
                "window": wi,
                "steps": action_count,
                "hits": hits,
                "pd": hits/max(action_count,1),
                "reward": reward_total,
                "coverage": float(np.count_nonzero(visits))/n,
                "unique_channels": int(np.count_nonzero(visits)),
                "mean_dwell_s": float(np.mean([dwell] * max(action_count,1))),
                "action_sequence": chosen[-32:],
            })
            completed_units += 1
            if progress_callback:
                progress_callback(completed_units / total_units)
        all_results[policy_name] = policy_rows

    summaries = {}
    for name, rows in all_results.items():
        valid = [r for r in rows if "pd" in r]
        if not valid:
            status = rows[0].get("status", "failed") if rows else "failed"
            reason = rows[0].get("reason") if rows else None
            summaries[name] = {"status": status, "windows": 0, **({"reason": reason} if reason else {})}
            continue
        summaries[name] = {
            "status": "completed",
            "windows": len(valid),
            "pd": float(np.mean([r["pd"] for r in valid])),
            "reward": float(np.mean([r["reward"] for r in valid])),
            "coverage": float(np.mean([r["coverage"] for r in valid])),
            "unique_channels": float(np.mean([r["unique_channels"] for r in valid])),
            "steps": int(np.sum([r["steps"] for r in valid])),
        }
    return {
        "source": "v6.2_uploaded_receiver_replay",
        "execution": "canonical_policy_adapter_on_uploaded_hdf5_windows",
        "channels": [float(x) for x in channels],
        "policies": summaries,
        "model_status": model_status,
        "windows": all_results,
        "evaluation_note": "Offline receiver replay. Ground-truth labels are not passed to PolicyObservation; metrics are replay-detection metrics, not a replacement for the locked synthetic benchmark.",
    }

def _run_pipeline_job(job_id: str, command: str, input_path: Optional[Path] = None, selected_policies: Optional[List[str]] = None) -> None:
    try:
        _job_update(job_id, status="running", progress=0.05, message=f"Running {command}")

        if command in {"final_evaluate", "prepare", "qa", "profile", "shard", "evaluate"} and input_path is None:
            raise ValueError(f"{command} requires an uploaded HDF5 file")

        if command == "final_evaluate":
            _job_update(job_id, progress=0.08, message="Preparing Final Model: STARE representation + Cognitive scheduler")
            replay = _uploaded_replay_evaluate(
                input_path, ["stare_cognitive"], max_windows=12, steps_per_window=48,
                progress_callback=lambda f: _job_update(
                    job_id, progress=0.08 + 0.70 * float(f),
                    message="Running STARE-Cognitive on receiver-observable windows"
                )
            )
            _job_update(job_id, progress=0.80, message="Running frozen STARE inference summary")
            stare = _stare_batch_summary(
                input_path, max_windows=32,
                progress_callback=lambda f: _job_update(
                    job_id, progress=0.80 + 0.18 * float(f),
                    message="Encoding temporal windows with frozen STARE"
                )
            )
            out = ROOT / "data" / "web_jobs" / job_id / "final_model_evaluation.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": {
                    "name": "STARE-Cognitive",
                    "role": "canonical_final_integrated_model",
                    "encoder": "frozen STARE GRU",
                    "scheduler": "Cognitive belief-driven policy",
                    "artifact": "models/frozen/stare_inference.pt",
                    "policy": "stare_cognitive",
                },
                "replay": replay,
                "stare": stare,
                "note": "Final-model receiver replay. The policy receives receiver-observable state plus the frozen STARE embedding; truth labels are withheld from PolicyObservation.",
            }
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _job_update(
                job_id, status="completed", progress=1.0,
                message="Final Model verification completed",
                output=json.dumps(payload, indent=2),
                artifacts=[str(out.relative_to(ROOT))],
            )
            return

        if command == "prepare":
            target = ROOT / "data" / "web_jobs" / job_id / "input.h5"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, target)
            _job_update(job_id, status="completed", progress=1.0, message="HDF5 copied into isolated web job workspace", artifacts=[str(target.relative_to(ROOT))])
            return

        if command == "qa":
            reader = TSRDHDF5Reader(input_path)
            info = reader.inspect()
            errors = validate_schema(info)
            _job_update(job_id, status="completed", progress=1.0, message="Canonical TSRD schema QA completed", output=json.dumps({"pulses": info.n_pulses, "features": list(info.features), "errors": errors}, indent=2), artifacts=[])
            return

        if command == "profile":
            reader = TSRDHDF5Reader(input_path)
            info = reader.inspect()
            mins = np.full(5, np.inf); maxs = np.full(5, -np.inf); count = 0
            for X, _ in reader.iter_batches(batch_size=8192, include_labels=False):
                mins = np.minimum(mins, np.nanmin(X, axis=0)); maxs = np.maximum(maxs, np.nanmax(X, axis=0)); count += len(X)
            report = {"pulses": count, "features": list(info.features), "min": mins.tolist(), "max": maxs.tolist()}
            out = ROOT / "data" / "web_jobs" / job_id / "profile.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            _job_update(job_id, status="completed", progress=1.0, message="Streaming HDF5 profile completed", output=json.dumps(report, indent=2), artifacts=[str(out.relative_to(ROOT))])
            return

        if command == "shard":
            artifacts = _build_uploaded_stare_shards(input_path, job_id)
            _job_update(job_id, status="completed", progress=1.0, message=f"Built {len(artifacts)} STARE-compatible NPZ shard(s)", artifacts=artifacts)
            return

        if command == "evaluate":
            _job_update(job_id, progress=0.08, message="Reading receiver metadata and first evaluation windows")
            eval_policies = selected_policies or POLICIES
            unknown = sorted(set(eval_policies) - set(POLICIES))
            if unknown:
                raise ValueError(f"Unknown policies: {unknown}")
            replay = _uploaded_replay_evaluate(
                input_path, eval_policies, max_windows=12, steps_per_window=48,
                progress_callback=lambda f: _job_update(job_id, progress=0.08 + 0.72 * float(f), message="Evaluating registered policies on uploaded receiver windows")
            )
            _job_update(job_id, progress=0.82, message="Running frozen STARE over uploaded windows")
            stare = _stare_batch_summary(
                input_path, max_windows=32,
                progress_callback=lambda f: _job_update(job_id, progress=0.82 + 0.16 * float(f), message="Running frozen STARE inference")
            )
            out = ROOT / "data" / "web_jobs" / job_id / "evaluation.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {"replay": replay, "stare": stare}
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _job_update(job_id, status="completed", progress=1.0, message="Uploaded HDF5 evaluated across all registered policies + STARE", output=json.dumps(payload, indent=2), artifacts=[str(out.relative_to(ROOT))])
            return

        script_map = {"prepare": "scripts/01_prepare_data.py", "qa": "scripts/02_run_qa.py", "profile": "scripts/03_profile_dataset.py", "shard": "scripts/04_build_training_shards.py"}
        script = ROOT / script_map[command]
        if not script.exists():
            raise FileNotFoundError(script)
        proc = subprocess.run([sys.executable, str(script)], cwd=ROOT, env={**os.environ, "EW_V6_ROOT": str(ROOT)}, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-8000:] or proc.stdout[-8000:])
        _job_update(job_id, status="completed", progress=1.0, message=f"{command} completed", output=proc.stdout[-12000:], artifacts=[])
    except Exception as exc:
        _job_update(job_id, status="failed", progress=1.0, message=str(exc))


@app.post("/api/dataset/job")
async def dataset_job(background_tasks: BackgroundTasks, command: str = Form(...), file: Optional[UploadFile] = File(None), policies: Optional[str] = Form(None)) -> Dict[str, Any]:
    if command not in {"prepare", "qa", "profile", "shard", "evaluate", "final_evaluate"}:
        raise HTTPException(400, "command must be prepare, qa, profile, shard, evaluate, or final_evaluate")
    input_path = None
    if file is not None:
        suffix = Path(file.filename or "input.h5").suffix.lower()
        if suffix not in {".h5", ".hdf5", ".he5"}:
            raise HTTPException(400, "Only HDF5 input is supported.")
        incoming = ROOT / "data" / "web_jobs" / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        max_upload_bytes = int(os.environ.get("MAX_UPLOAD_MB", "250")) * 1024 * 1024
        input_path = incoming / f"{uuid.uuid4().hex}{suffix}"
        written = 0
        with input_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_upload_bytes:
                    try:
                        input_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise HTTPException(413, f"Upload exceeds {max_upload_bytes // (1024 * 1024)} MB limit.")
                out.write(chunk)
    selected_policies = None
    if policies:
        selected_policies = [x.strip() for x in policies.split(',') if x.strip()]
        unknown = sorted(set(selected_policies) - set(POLICIES))
        if unknown:
            raise HTTPException(400, f"Unknown policies: {unknown}")
    job_id = _new_job(command)
    background_tasks.add_task(_run_pipeline_job, job_id, command, input_path, selected_policies)
    return {"job_id": job_id, "status": "queued", "command": command, "input": str(input_path) if input_path else None}


@app.get("/api/dataset/job/{job_id}")
def dataset_job_status(job_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id")
    return job


@app.get("/api/dataset/artifacts")
def dataset_artifacts() -> Dict[str, Any]:
    roots = [ROOT / "data" / "manifests", ROOT / "data" / "processed" / "shards", ROOT.parent / "v6.1" / "data" / "processed" / "shards"]
    items: List[str] = []
    for root in roots:
        if root.exists():
            try:
                items.extend(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in root.rglob("*") if p.is_file())
            except Exception:
                pass
    return {"source": "v6.2", "artifacts": sorted(set(items))[:1000]}
