import os
from typing import List, Optional, Sequence, Tuple

import cv2
import ffmpeg
import librosa
import numpy as np


EPS = 1e-8
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def compute_aux_features(
    video_path: Optional[str] = None,
    audio_file_path: Optional[str] = None,
    video_frames_dir: Optional[str] = None,
    transcript_text: str = "",
) -> np.ndarray:
    lip_sync_score = compute_lip_sync_score(video_path, audio_file_path, video_frames_dir)
    fft_spike_score = compute_fft_spike_score(audio_file_path)
    metadata_suspicion = compute_metadata_suspicion(video_path, transcript_text)
    temporal_consistency = compute_temporal_consistency(video_frames_dir)
    return np.asarray(
        [lip_sync_score, fft_spike_score, metadata_suspicion, temporal_consistency],
        dtype=np.float32,
    )


def compute_lip_sync_score(
    video_path: Optional[str],
    audio_file_path: Optional[str],
    video_frames_dir: Optional[str],
) -> float:
    if not audio_file_path or not os.path.exists(audio_file_path):
        return 0.0

    frame_paths = _list_frame_paths(video_frames_dir)
    if len(frame_paths) < 2:
        return 0.0

    frame_times, fps = _frame_times(frame_paths, video_path)
    if fps <= 0 or len(frame_times) < 2:
        return 0.0

    mouth_motion = []
    motion_times = []
    previous_mouth = None

    for frame_path, frame_time in zip(frame_paths, frame_times):
        gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        face = _detect_face(gray)
        if face is None:
            continue

        mouth_roi = _extract_mouth_roi(gray, face)
        if mouth_roi is None:
            continue

        if previous_mouth is not None:
            mouth_motion.append(float(np.mean(np.abs(mouth_roi - previous_mouth))))
            motion_times.append(frame_time)

        previous_mouth = mouth_roi

    if len(mouth_motion) < 3:
        return 0.0

    y, sr = librosa.load(audio_file_path, sr=16000, mono=True)
    if y.size == 0:
        return 0.0

    hop_length = 512
    frame_length = 1024
    audio_rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if audio_rms.size < 3:
        return 0.0

    audio_times = librosa.times_like(audio_rms, sr=sr, hop_length=hop_length, n_fft=frame_length)
    aligned_rms = np.interp(motion_times, audio_times, audio_rms, left=audio_rms[0], right=audio_rms[-1])

    motion_arr = np.asarray(mouth_motion, dtype=np.float32)
    rms_arr = np.asarray(aligned_rms, dtype=np.float32)

    if np.std(motion_arr) < EPS or np.std(rms_arr) < EPS:
        return 0.0

    corr = float(np.corrcoef(motion_arr, rms_arr)[0, 1])
    if not np.isfinite(corr):
        return 0.0

    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def compute_fft_spike_score(audio_file_path: Optional[str]) -> float:
    if not audio_file_path or not os.path.exists(audio_file_path):
        return 0.0

    y, _ = librosa.load(audio_file_path, sr=16000, mono=True)
    if y.size == 0:
        return 0.0

    spectrum = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    if spectrum.size == 0:
        return 0.0

    mean_spectrum = np.mean(np.log1p(spectrum), axis=1)
    kernel = np.ones(9, dtype=np.float32) / 9.0
    smooth = np.convolve(mean_spectrum, kernel, mode="same")
    spikes = np.clip(mean_spectrum - smooth, 0.0, None)
    normalized_spikes = spikes / (np.mean(np.abs(smooth)) + EPS)
    spike_strength = float(np.percentile(normalized_spikes, 95))
    return float(np.clip(np.tanh(spike_strength / 3.0), 0.0, 1.0))


def compute_metadata_suspicion(video_path: Optional[str], transcript_text: str = "") -> float:
    score = 0.0

    if transcript_text.strip() == "":
        score += 0.1

    if not video_path or not os.path.exists(video_path):
        return float(np.clip(score, 0.0, 1.0))

    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error:
        return min(1.0, score + 0.2)

    streams = probe.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    format_info = probe.get("format", {})

    if video_stream is None:
        score += 0.4
    if audio_stream is None:
        score += 0.35

    fps = _parse_frame_rate(video_stream.get("avg_frame_rate") if video_stream else None)
    if fps <= 0.0 or fps > 120.0:
        score += 0.15

    bit_rate = _safe_float(format_info.get("bit_rate"))
    if bit_rate is None:
        score += 0.15
    elif bit_rate < 150_000 or bit_rate > 25_000_000:
        score += 0.1

    file_name = os.path.basename(video_path).lower()
    if any(token in file_name for token in ("fake", "deepfake", "synthetic", "generated", "clone")):
        score += 0.25

    return float(np.clip(score, 0.0, 1.0))


def compute_temporal_consistency(video_frames_dir: Optional[str]) -> float:
    frame_paths = _list_frame_paths(video_frames_dir)
    if len(frame_paths) < 2:
        return 0.0

    diffs = []
    previous_face = None

    for frame_path in frame_paths:
        gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        face = _detect_face(gray)
        if face is None:
            continue

        face_roi = _extract_face_roi(gray, face)
        if face_roi is None:
            continue

        if previous_face is not None:
            diffs.append(float(np.mean(np.abs(face_roi - previous_face))))

        previous_face = face_roi

    if not diffs:
        return 0.0

    inconsistency = float(np.median(diffs))
    return float(np.clip(1.0 - inconsistency, 0.0, 1.0))


def _list_frame_paths(video_frames_dir: Optional[str]) -> List[str]:
    if not video_frames_dir or not os.path.isdir(video_frames_dir):
        return []

    frame_paths = []
    for name in sorted(os.listdir(video_frames_dir)):
        lower_name = name.lower()
        if lower_name.endswith((".jpg", ".jpeg", ".png")):
            frame_paths.append(os.path.join(video_frames_dir, name))
    return frame_paths


def _frame_times(frame_paths: Sequence[str], video_path: Optional[str]) -> Tuple[np.ndarray, float]:
    fps = _video_fps(video_path)
    indices = np.asarray([_frame_index(path, idx) for idx, path in enumerate(frame_paths)], dtype=np.float32)

    if fps > 0:
        return indices / fps, fps

    if len(indices) <= 1:
        return np.zeros(len(indices), dtype=np.float32), 0.0

    return np.linspace(0.0, float(len(indices) - 1), num=len(indices), dtype=np.float32), 1.0


def _frame_index(frame_path: str, fallback_index: int) -> int:
    stem = os.path.splitext(os.path.basename(frame_path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else fallback_index


def _video_fps(video_path: Optional[str]) -> float:
    if not video_path or not os.path.exists(video_path):
        return 0.0

    capture = cv2.VideoCapture(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()

    return fps if np.isfinite(fps) else 0.0


def _detect_face(gray_frame: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    if gray_frame is None or gray_frame.size == 0 or FACE_CASCADE.empty():
        return None

    faces = FACE_CASCADE.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(48, 48),
    )
    if len(faces) == 0:
        return None

    return max(faces, key=lambda face: face[2] * face[3])


def _extract_mouth_roi(gray_frame: np.ndarray, face: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = face
    x1 = max(x + int(0.15 * w), 0)
    x2 = min(x + int(0.85 * w), gray_frame.shape[1])
    y1 = max(y + int(0.6 * h), 0)
    y2 = min(y + h, gray_frame.shape[0])
    roi = gray_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi = cv2.resize(roi, (64, 32), interpolation=cv2.INTER_AREA)
    return roi.astype(np.float32) / 255.0


def _extract_face_roi(gray_frame: np.ndarray, face: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = face
    roi = gray_frame[y : y + h, x : x + w]
    if roi.size == 0:
        return None

    roi = cv2.resize(roi, (96, 96), interpolation=cv2.INTER_AREA)
    return roi.astype(np.float32) / 255.0


def _parse_frame_rate(frame_rate: Optional[str]) -> float:
    if not frame_rate or frame_rate == "0/0":
        return 0.0
    if "/" in frame_rate:
        num, den = frame_rate.split("/", 1)
        den_value = float(den)
        return 0.0 if den_value == 0 else float(num) / den_value
    return float(frame_rate)


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
