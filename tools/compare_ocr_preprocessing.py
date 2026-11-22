"""Compare existing OCR crop preprocessing variants against benchmark labels.

Run after evaluate_benchmark.py so each result directory has saved plate crops:
python tools/compare_ocr_preprocessing.py --results-dir jobs/evaluations/run-01
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vtrace import ocr  # noqa: E402
from vtrace.metrics import edit_distance, normalize_plate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Existing evaluation output containing per-frame plate crops")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    result_root = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    truth = json.loads((ROOT / "test_clips" / "ground_truth.json").read_text(encoding="utf-8"))
    names = ("gray", "clahe", "sharpened", "confidence_selected")
    variants = {name: {"exact": 0, "errors": 0, "characters": 0, "reference_characters": 0,
                       "seconds": 0.0, "crops": 0,
                       "predictions": []} for name in names}

    for clip, labels in truth.items():
        result_path = result_root / clip / "result.json"
        if not result_path.is_file():
            raise SystemExit(f"Missing pipeline output: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        frames = [f for f in result.get("frames", [])
                  if f.get("used_in_fusion") and f.get("weight", 0) > 0]
        if not frames:
            frames = [f for f in result.get("frames", []) if f.get("plate_img")]
        for frame in frames:
            image_path = result_path.parent / frame["plate_img"]
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if gray.shape[0] < 48:
                scale = 48 / gray.shape[0]
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            expected = labels["plate"]
            hypotheses = []
            for name, processed in ocr._variants(gray):
                started = time.perf_counter()
                parts = ocr._split_lines(processed) if frame.get("two_line") else [processed]
                chars = []
                for line_index, part in enumerate(parts):
                    padded = ocr._clean_edges(part)
                    decoded = ocr._decode(ocr._posteriors(padded, detector=False), padded.shape[1])
                    for char in decoded:
                        char["line"] = line_index
                    chars.extend(ocr._mark_gaps(decoded))
                prediction = "".join(char["char"] for char in chars)
                elapsed = time.perf_counter() - started
                known = [char for char in chars if char["char"] != "?"]
                score = ((sum(char["conf"] for char in known) / len(known)) if known else 0.0)
                score *= min(1.0, len(known) / 6)
                if len(known) > 10:
                    score *= max(0.5, 1.0 - 0.12 * (len(known) - 10))
                hypotheses.append((score, name, prediction, elapsed))
                row = variants[name]
                _record(row, clip, frame["frame"], expected, prediction, elapsed)
            # Reproduce the current recognizer's confidence-based variant choice
            # using exactly the same three hypotheses from each saved crop.
            chosen = max(hypotheses, key=lambda item: item[0])
            _record(variants["confidence_selected"], clip, frame["frame"], expected,
                    chosen[2], sum(item[3] for item in hypotheses))

    summary = {}
    for name, row in variants.items():
        summary[name] = {
            "recognizer": "EasyOCR English CRNN (existing pretrained weights)",
            "crop_count": row["crops"], "exact_crop_matches": row["exact"],
            "normalized_edit_similarity": round(1 - row["errors"] / max(row["characters"], 1), 4),
            "character_error_rate": round(row["errors"] / max(row["reference_characters"], 1), 4),
            "mean_seconds_per_crop": round(row["seconds"] / max(row["crops"], 1), 4),
            "predictions": row["predictions"],
        }
    print(json.dumps(summary, indent=2))
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _record(row, clip, frame, expected, prediction, elapsed):
    row["seconds"] += elapsed
    row["crops"] += 1
    prediction = normalize_plate(prediction)
    expected = normalize_plate(expected)
    row["exact"] += prediction == expected
    row["errors"] += edit_distance(prediction, expected)
    row["characters"] += max(len(prediction), len(expected))
    row["reference_characters"] += len(expected)
    row["predictions"].append({"clip": clip, "frame": frame,
                               "expected": expected, "prediction": prediction})


if __name__ == "__main__":
    main()
