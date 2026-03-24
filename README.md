# Detector (Python / FastAPI)

This service exposes a single inference endpoint:

- `POST /infer` (multipart form): returns `overall` (REAL/FAKE + probabilities) and `segments` (timestamp localization).

## Run locally

1. Install dependencies:
   - `pip install -r requirements.txt`

2. Start the server:
   - `uvicorn main:app --host 0.0.0.0 --port 8000`

3. Health check:
   - `GET http://localhost:8000/healthz`

## API

`POST /infer`

Form fields:
- `file`: uploaded video (mp4/mov/etc)
- `windowSec` (default `1.5`)
- `overlapSec` (default `0.5`)
- `sampleFps` (default `8`)

Response (shape):
- `overall`: `{ prediction, prob_fake, prob_real }`
- `segments`: `[{ startSec, endSec, score, label }]`
- `windowScores`: per-window scores used to build `segments`
- `heuristic`: metadata about the current detector backend

## Swap in a different detector

Right now this repo implements a training-free `audio_video_mismatch_proxy` inside:

- `infer_heuristic(...)` in `main.py`

To swap in a model-based detector later:
1. Implement a new function, e.g. `infer_model(...)`, returning the same response schema (at least `overall`, `segments`, `windowScores`).
2. Update `infer(...)` endpoint in `main.py` to call `infer_model` based on an environment variable (example: `DETECTOR_BACKEND=model`).

This keeps the Node/Next app stable because `/infer` stays compatible.

## Logs & runs

Each `/infer` call is appended to:
- `detector/runs/infer_log.jsonl`

Use the log file for debugging and later calibration/evaluation.

