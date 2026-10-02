# SIH26055 v6.2 — START HERE

## The one page that is the final system

Open **Final Model** first:

`#/final`

This is the canonical **STARE-Cognitive** demonstration path.

### What it runs

1. Upload a TSRD-compatible HDF5 capture.
2. Inspect the real HDF5 schema.
3. Run bounded receiver-window replay through the v6.2 policy adapter.
4. Run the frozen STARE TorchScript encoder over temporal windows.
5. Feed the resulting STARE representation into STARE-Cognitive.
6. Show diagnostic receiver-replay metrics and artifact status.

Uploaded replay metrics are explicitly diagnostic and are not presented as the locked benchmark.

## Other pages

- **Scenario Lab** — continuous browser-side RF/receiver waterfall. The first completed result is locked while the waterfall continues with fresh seeds.
- **Policy Arena** — independent backend policy execution and checkpoint compatibility reporting.
- **Benchmark** — frozen locked-evaluation evidence from the research report.
- **Dataset** — HDF5 inspection and pipeline utilities.
- **Methodology** — current v6.2 architecture and information boundary.
- **Research Console** — engineering/research diagnostics.

## Local run

### Backend

Run from this self-contained package root:

```cmd
cd /d C:\path\to\SIH26055-production
call .venv\Scripts\activate.bat
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

### Frontend

In this web package:

```cmd
npm install
set VITE_BACKEND_URL=http://127.0.0.1:8000
npm run dev
```

Then open the Vite URL and choose **Final Model**.

## Deployment architecture

```text
Vercel frontend
      |
      | HTTPS
      v
Render Docker/FastAPI backend
      |
      +--> v6.2 canonical source + policy engine
      |
      +--> Hugging Face frozen deployment artifacts
```

The full TSRD training corpus is intentionally not part of this web package.
