# SIH26055 v6.2 Backend Bridge

This FastAPI service must run **inside the v6.2 research environment**. It imports the canonical v6.2 policy/evaluation/data/model modules instead of recreating them.

## Install

From:

```text
C:\Users\pushp\Downloads\Balaji\v6.2
```

```cmd
call .venv\Scripts\activate.bat
python -m pip install -r backend\requirements.txt
```

## Start

```cmd
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

## Main endpoints

```text
GET  /api/health
GET  /api/policies
GET  /api/synthetic/scenarios
GET  /api/research/status
GET  /api/models
GET  /api/results/locked

POST /api/evaluate/synthetic
POST /api/decision/trace
POST /api/stare/infer

POST /api/dataset/inspect
POST /api/dataset/job
GET  /api/dataset/job/{job_id}
GET  /api/dataset/artifacts
```

## Data jobs

When an HDF5 file is supplied, the web job path is isolated under:

```text
v6.2\data\web_jobs\<job_id>\
```

`shard` creates STARE-compatible NPZ shards with the canonical shape:

```text
x = [N, 128, 5]
y = [N, 128]
```

The backend uses the canonical `TSRDHDF5Reader` and existing v6.2 normalization when available.

## Safety/integrity boundary

Policy code receives receiver-observable state only. Truth-side event information is retained by the evaluation environment and used after execution for metrics.

The frontend never silently falls back to the archived browser simulator.
