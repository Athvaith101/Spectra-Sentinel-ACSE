# SIH26055 v6.2 — Final Frontend + Canonical Backend Package

This package uses the uploaded SIH26055 frontend as the visual base and connects it to the canonical v6.2 Python engine and frozen model artifacts.

## Canonical flow

- **Final Model** — STARE-Cognitive; real HDF5 inspection, frozen STARE TorchScript inference, receiver-only policy replay.
- **Scenario Lab** — continuous browser RF/receiver waterfall; first completed result is locked while new seeded episodes continue.
- **Policy Arena** — independent backend policy execution with strict RL checkpoint compatibility checks.
- **Benchmark** — locked research evidence; not mixed with uploaded-file diagnostics.
- **Methodology** — current working architecture and information boundary.

## Package layout

`src/` is the React frontend. `python_engine/src/` is the canonical Python research engine. `models/` contains the frozen deployment artifacts for local/offline operation. The huge TSRD training corpus is intentionally excluded.

## Local

Backend:

```cmd
call .venv\Scripts\activate.bat
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

Frontend:

```cmd
npm install
set VITE_BACKEND_URL=http://127.0.0.1:8000
npm run dev
```

## Production

- Vercel: React/Vite frontend.
- Render: Dockerized FastAPI backend using `Dockerfile` / `render.yaml`.
- Hugging Face: optional model registry. Set `MODEL_SOURCE=huggingface` and `HF_MODEL_REPO` on Render to hydrate model artifacts at startup.
- Public uploads are bounded by `MAX_UPLOAD_MB` (default 250 MB).

Do not upload the full TSRD corpus to GitHub, Vercel or Render.
