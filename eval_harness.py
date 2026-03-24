import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

from main import infer_heuristic


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def load_labels(labels_csv: str) -> Dict[str, str]:
    """
    labels_csv format:
      filename,label
    where label is REAL or FAKE.
    """
    labels: Dict[str, str] = {}
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = (row.get("filename") or "").strip()
            lb = (row.get("label") or "").strip().upper()
            if not fn or not lb:
                continue
            labels[fn] = lb
    return labels


def infer_one(video_path: str, window_sec: float, overlap_sec: float, sample_fps: float) -> Dict:
    started = time.time()
    res = infer_heuristic(
        video_path=video_path,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
        sample_fps=sample_fps,
    )
    res["_timing"] = {"elapsedMs": int((time.time() - started) * 1000)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputDir", required=True, help="Folder containing videos")
    ap.add_argument("--labelsCsv", default=None, help="Optional CSV for accuracy metrics")
    ap.add_argument("--windowSec", type=float, default=1.5)
    ap.add_argument("--overlapSec", type=float, default=0.5)
    ap.add_argument("--sampleFps", type=float, default=8.0)
    ap.add_argument("--outDir", default="runs")
    args = ap.parse_args()

    labels = load_labels(args.labelsCsv) if args.labelsCsv else {}

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), args.outDir, f"eval_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.jsonl")

    videos = []
    for root, _, files in os.walk(args.inputDir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in VIDEO_EXTS:
                videos.append(os.path.join(root, fn))

    videos.sort()
    if not videos:
        print("No videos found in inputDir.")
        return

    correct = 0
    total = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for vp in videos:
            fn = os.path.basename(vp)
            print(f"Running: {fn}")

            try:
                res = infer_one(vp, args.windowSec, args.overlapSec, args.sampleFps)
                pred = res.get("overall", {}).get("prediction")
                label = labels.get(fn)
                if label:
                    total += 1
                    if str(pred).upper() == label.upper():
                        correct += 1
                record = {
                    "video": fn,
                    "path": vp,
                    "label": label,
                    "prediction": pred,
                    "result": res,
                }
                f.write(json.dumps(record) + "\n")
            except Exception as e:
                record = {"video": fn, "path": vp, "error": str(e)}
                f.write(json.dumps(record) + "\n")

    if total > 0:
        acc = correct / total
        summary = {"numLabeled": total, "correct": correct, "accuracy": acc}
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2)
        print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    else:
        print(f"Done. Results written to: {out_path}")


if __name__ == "__main__":
    main()

