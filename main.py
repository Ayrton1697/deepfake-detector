import os
import json
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import librosa
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
import soundfile as sf
from uuid import uuid4
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


VIDEO_SAMPLE_FPS_DEFAULT = 8.0
AUDIO_SR = 16000

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
os.makedirs(RUNS_DIR, exist_ok=True)
INFER_LOG_PATH = os.path.join(RUNS_DIR, "infer_log.jsonl")


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{err}")


def ffprobe_duration_sec(video_path: str) -> float:
    # `imageio-ffmpeg` bundles `ffmpeg` but not necessarily `ffprobe`.
    # Parse the "Duration: HH:MM:SS.xx" string from ffmpeg stderr.
    p = subprocess.run(
        [get_ffmpeg_exe(), "-i", video_path],
        capture_output=True,
        text=True,
    )
    stderr = (p.stderr or "").strip()
    # Typical line: "Duration: 00:01:23.45, start: 0.000000, bitrate: ..."
    import re

    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if not m:
        raise RuntimeError(f"Could not parse duration from ffmpeg output:\n{stderr[-4000:]}")
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = float(m.group(3))
    dur = hh * 3600 + mm * 60 + ss
    if not math.isfinite(dur) or dur <= 0:
        raise RuntimeError(f"Could not determine duration for: {video_path}")
    return dur


def extract_audio_wav_16k(video_path: str, wav_path: str, duration_sec: float | None = None) -> None:
    cmd = [
        get_ffmpeg_exe(),
        "-nostdin",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vn",
        "-ar",
        str(AUDIO_SR),
        "-ac",
        "1",
        "-f",
        "wav",
        wav_path,
    ]
    try:
        _run(cmd)
        return
    except Exception:
        # Some test videos might not include an audio stream.
        # Fall back to silent audio so the pipeline can still run.
        n_samples = int((duration_sec or 1.0) * AUDIO_SR)
        n_samples = max(AUDIO_SR // 2, n_samples)
        wav = np.zeros((n_samples,), dtype=np.float32)
        sf.write(wav_path, wav, AUDIO_SR)


def minmax_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def var_laplacian(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    return float(np.mean((a - b) ** 2))


@dataclass
class VideoMetrics:
    times: np.ndarray  # seconds
    sharpness: np.ndarray  # Laplacian variance
    motion: np.ndarray  # face-frame motion proxy (MSE)


def extract_video_metrics(
    video_path: str,
    sample_fps: float,
    face_min_size: Tuple[int, int] = (60, 60),
) -> VideoMetrics:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video via OpenCV")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    input_fps = float(input_fps) if input_fps and input_fps > 1e-6 else 25.0

    frame_interval = max(1, int(round(input_fps / max(sample_fps, 0.1))))

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    prev_face_gray = None
    idx = 0

    times: List[float] = []
    sharpness: List[float] = []
    motion: List[float] = []

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if idx % frame_interval != 0:
            idx += 1
            continue

        t = idx / input_fps
        idx += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(
            frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=face_min_size
        )
        if len(faces) == 0:
            prev_face_gray = None  # avoid huge motion after missing faces
            continue

        # Pick largest face.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_gray = frame_gray[y : y + h, x : x + w]

        if face_gray.size == 0:
            prev_face_gray = None
            continue

        # Normalize face shape for stable motion comparison.
        face_gray = cv2.resize(face_gray, (96, 96), interpolation=cv2.INTER_AREA)
        sh = var_laplacian(face_gray)

        if prev_face_gray is None:
            mot = 0.0
        else:
            mot = mse(face_gray, prev_face_gray)
        prev_face_gray = face_gray

        times.append(t)
        sharpness.append(sh)
        motion.append(mot)

    cap.release()

    if len(times) == 0:
        # Return empty arrays; the caller will handle fallbacks.
        return VideoMetrics(
            times=np.zeros((0,), dtype=np.float32),
            sharpness=np.zeros((0,), dtype=np.float32),
            motion=np.zeros((0,), dtype=np.float32),
        )

    return VideoMetrics(
        times=np.asarray(times, dtype=np.float32),
        sharpness=np.asarray(sharpness, dtype=np.float32),
        motion=np.asarray(motion, dtype=np.float32),
    )


def extract_audio_rms(
    audio_wav_path: str,
    hop_length: int,
    win_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    y, sr = librosa.load(audio_wav_path, sr=AUDIO_SR, mono=True)
    if y.size == 0:
        y = np.zeros((AUDIO_SR,), dtype=np.float32)

    rms = librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(
        np.arange(rms.shape[0]), sr=sr, hop_length=hop_length, n_fft=win_length
    )
    return times.astype(np.float32), rms.astype(np.float32)


def merge_segments_from_window_scores(
    window_scores: List[Dict[str, Any]],
    enter_thr: float,
    exit_thr: float,
    min_windows_in_segment: int,
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    active = False
    seg_start = 0.0
    seg_end = 0.0
    seg_scores: List[float] = []

    for w in window_scores:
        s = float(w["score"])
        start_sec = float(w["startSec"])
        end_sec = float(w["endSec"])

        if not active and s >= enter_thr:
            active = True
            seg_start = start_sec
            seg_end = end_sec
            seg_scores = [s]
        elif active:
            if s >= exit_thr:
                seg_end = end_sec
                seg_scores.append(s)
            else:
                # close segment
                if len(seg_scores) >= min_windows_in_segment:
                    segments.append(
                        {
                            "startSec": float(seg_start),
                            "endSec": float(seg_end),
                            "score": float(np.median(np.asarray(seg_scores, dtype=np.float32))),
                        }
                    )
                active = False
                seg_scores = []

    if active and len(seg_scores) >= min_windows_in_segment:
        segments.append(
            {
                "startSec": float(seg_start),
                "endSec": float(seg_end),
                "score": float(np.median(np.asarray(seg_scores, dtype=np.float32))),
            }
        )

    return segments


def infer_heuristic(
    video_path: str,
    window_sec: float,
    overlap_sec: float,
    sample_fps: float,
) -> Dict[str, Any]:
    duration = ffprobe_duration_sec(video_path)
    window_sec = max(0.2, float(window_sec))
    overlap_sec = max(0.0, min(float(overlap_sec), window_sec - 0.05))

    step = max(0.1, window_sec - overlap_sec)
    num_windows = max(1, int(math.ceil((duration - window_sec) / step)) + 1)

    # Extract full-video measurements once, then slice into windows.
    audio_tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_tmp = tmp.name
        extract_audio_wav_16k(video_path, audio_tmp, duration_sec=duration)

        # Audio analysis: use hop aligned roughly with window fractions.
        hop_length = int(AUDIO_SR / max(20, sample_fps * 2))
        hop_length = max(128, min(1024, hop_length))
        win_length = 2 * hop_length
        audio_times, audio_rms = extract_audio_rms(audio_tmp, hop_length, win_length)

        video_metrics = extract_video_metrics(video_path, sample_fps=sample_fps)
        v_times = video_metrics.times
        v_sharp = video_metrics.sharpness
        v_motion = video_metrics.motion

        # Compute per-window aggregates.
        window_scores_raw: List[float] = []
        window_scores: List[Dict[str, Any]] = []
        v_motion_means: List[float] = []
        v_sharp_means: List[float] = []
        a_rms_means: List[float] = []

        for i in range(num_windows):
            start_sec = i * step
            end_sec = min(duration, start_sec + window_sec)

            if end_sec <= start_sec:
                continue

            # slice audio
            a_mask = (audio_times >= start_sec) & (audio_times < end_sec)
            a_r = float(np.mean(audio_rms[a_mask])) if np.any(a_mask) else 0.0

            # slice video
            v_mask = (v_times >= start_sec) & (v_times < end_sec)
            v_m = float(np.mean(v_motion[v_mask])) if np.any(v_mask) else 0.0
            v_s = float(np.mean(v_sharp[v_mask])) if np.any(v_mask) else 0.0

            v_motion_means.append(v_m)
            v_sharp_means.append(v_s)
            a_rms_means.append(a_r)

            # placeholder raw; computed later after normalization
            window_scores_raw.append(0.0)
            window_scores.append(
                {
                    "startSec": float(start_sec),
                    "endSec": float(end_sec),
                    "score": 0.0,
                    "prob_fake": 0.0,
                }
            )

        v_motion_means = np.asarray(v_motion_means, dtype=np.float32)
        v_sharp_means = np.asarray(v_sharp_means, dtype=np.float32)
        a_rms_means = np.asarray(a_rms_means, dtype=np.float32)

        v_motion_norm = minmax_norm(v_motion_means)
        v_sharp_norm = minmax_norm(v_sharp_means)
        a_rms_norm = minmax_norm(a_rms_means)

        sharp_inv = 1.0 - v_sharp_norm  # less sharp could indicate synthesis artifacts
        discrepancy = np.abs(v_motion_norm - a_rms_norm)  # audio-video mismatch proxy

        # Weighted fusion into a fake probability proxy in [0,1].
        prob_fake = np.clip(0.6 * discrepancy + 0.4 * sharp_inv, 0.0, 1.0)

        # Populate scores
        for idx, s in enumerate(prob_fake):
            window_scores[idx]["score"] = float(s)
            window_scores[idx]["prob_fake"] = float(s)

        # Localize with hysteresis thresholds.
        enter_thr = 0.65
        exit_thr = 0.55
        min_windows = 2
        segments = merge_segments_from_window_scores(
            window_scores=window_scores,
            enter_thr=enter_thr,
            exit_thr=exit_thr,
            min_windows_in_segment=min_windows,
        )

        if len(window_scores) == 0:
            prob = 0.5
        else:
            prob = float(np.median(np.asarray(prob_fake, dtype=np.float32)))

        prediction = "FAKE" if prob >= 0.5 else "REAL"

        return {
            "overall": {
                "prediction": prediction,
                "prob_fake": prob,
                "prob_real": float(1.0 - prob),
            },
            "segments": [
                {**seg, "label": "FAKE"} for seg in segments
            ],  # label for UI convenience
            "windowScores": window_scores,
            "heuristic": {
                "type": "audio_video_mismatch_proxy",
                "windowSec": window_sec,
                "overlapSec": overlap_sec,
                "sampleFps": sample_fps,
            },
        }
    finally:
        if audio_tmp and os.path.exists(audio_tmp):
            try:
                os.remove(audio_tmp)
            except OSError:
                pass


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    windowSec: float = Form(1.5),
    overlapSec: float = Form(0.5),
    sampleFps: float = Form(VIDEO_SAMPLE_FPS_DEFAULT),
) -> Dict[str, Any]:
    # Save upload to disk because both OpenCV and ffmpeg are filesystem-oriented.
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".mp4"
    req_id = str(uuid4())
    tmp_dir = tempfile.mkdtemp(prefix="verifai_")
    video_path = os.path.join(tmp_dir, f"upload{suffix}")

    try:
        with open(video_path, "wb") as f:
            f.write(await file.read())

        started = time.time()
        result = infer_heuristic(
            video_path=video_path,
            window_sec=windowSec,
            overlap_sec=overlapSec,
            sample_fps=sampleFps,
        )
        elapsed_ms = int((time.time() - started) * 1000)

        try:
            overall = result.get("overall", {})
            with open(INFER_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "requestId": req_id,
                            "filename": file.filename,
                            "windowSec": windowSec,
                            "overlapSec": overlapSec,
                            "sampleFps": sampleFps,
                            "overallPrediction": overall.get("prediction"),
                            "prob_fake": overall.get("prob_fake"),
                            "numSegments": len(result.get("segments") or []),
                            "elapsedMs": elapsed_ms,
                            "ts": int(time.time()),
                        }
                    )
                    + "\n"
                )
        except Exception:
            # Logging should never break inference.
            pass

        result["requestId"] = req_id
        result["timing"] = {"elapsedMs": elapsed_ms}
        return result
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    # For local development only.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))

