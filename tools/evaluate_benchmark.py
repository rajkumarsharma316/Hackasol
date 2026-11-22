"""Run the preserved benchmark without overwriting the original reports.

Example: python tools/evaluate_benchmark.py --output jobs/evaluations/run-01
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
try:
    import resource
except ImportError:  # Windows
    resource = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vtrace import engines, matcher  # noqa: E402
from vtrace.metrics import edit_distance, normalize_plate, plate_metrics  # noqa: E402
from vtrace.pipeline import VideoAnalysis  # noqa: E402


def decision_correct(result: dict, truth: dict) -> bool | None:
    if truth.get("in_registry") not in (True, False):
        return None
    if truth.get("in_registry") is True and not normalize_plate(truth.get("plate")):
        return None
    match = result.get("match")
    if match is None:  # summarized row from evaluation.json
        status = result.get("system_decision")
        registry_status = result.get("registry_decision", status)
        top_plate = result.get("top_candidate")
    else:
        status = match.get("status")
        registry_status = match.get("registry_status", status)
        top = (match.get("candidates") or [None])[0]
        top_plate = (top or {}).get("vehicle", {}).get("reg_no")
    if truth.get("in_registry"):
        return status == "match" and normalize_plate(top_plate) == normalize_plate(truth.get("plate"))
    return status == "no_confident_match" and registry_status != "match"


def measure(rows: list[dict], labeled: dict) -> dict:
    registered = [r for r in rows if labeled.get(r["clip"], {}).get("in_registry") is True]
    unregistered = [r for r in rows if labeled.get(r["clip"], {}).get("in_registry") is False]
    decision_rows = registered + unregistered
    plate = plate_metrics([(r.get("plate"), labeled.get(r["clip"], {}).get("plate")) for r in rows])
    top_candidate_correct = 0
    for row in rows:
        truth = labeled.get(row["clip"], {})
        if truth.get("in_registry") is True and normalize_plate(truth.get("plate")):
            top_candidate_correct += normalize_plate(row.get("top_candidate")) == normalize_plate(truth["plate"])
    correct = [decision_correct(row, labeled.get(row["clip"], {})) for row in decision_rows]
    decision_count = sum(value is not None for value in correct)
    decision_correct_count = sum(value is True for value in correct)
    fps = sum(r.get("system_decision") == "match" for r in unregistered)
    fns = sum(r.get("system_decision") != "match" for r in registered)
    automatic_matches = sum(r.get("system_decision") == "match" for r in rows)
    abstentions = len(rows) - automatic_matches
    valid_coverage = [r["track_coverage"] for r in rows if r.get("track_coverage") is not None]
    valid_runtime = [r["seconds"] for r in rows if r.get("seconds") is not None]
    return {
        "clips": len(rows),
        "decision_labels_available": decision_count,
        "correct_automatic_decisions": decision_correct_count,
        "automatic_decision_accuracy": decision_correct_count / decision_count if decision_count else None,
        "decision_accuracy": f"{decision_correct_count}/{decision_count}",
        "automatic_matches": automatic_matches,
        "automatic_match_rate": automatic_matches / len(rows) if rows else None,
        "abstentions": abstentions,
        "abstention_rate": abstentions / len(rows) if rows else None,
        "plate_ocr_exact_match": f"{plate['exact_matches']}/{plate['samples']}",
        "exact_plate_match_rate": plate["exact_match_rate"],
        "character_accuracy": plate["normalized_edit_similarity"],  # legacy name retained for old reports
        "normalized_edit_similarity": plate["normalized_edit_similarity"],
        "macro_normalized_edit_similarity": plate["macro_normalized_edit_similarity"],
        "character_error_rate": plate["character_error_rate"],
        "macro_character_error_rate": plate["macro_character_error_rate"],
        "character_errors": plate["character_errors"],
        "reference_characters": plate["reference_characters"],
        "plate_samples_evaluated": plate["samples"],
        "clips_missing_plate_ground_truth": plate["missing_ground_truth"],
        "normalized_edit_similarity_denominator": plate["normalized_edit_similarity_denominator"],
        "top_candidate_accuracy_registered": (f"{top_candidate_correct}/{len(registered)}" if registered else "n/a"),
        "review_required": sum(r.get("system_decision") == "review_required" for r in rows),
        "registry_false_positives": fps,
        "registry_false_positive_rate": fps / len(unregistered) if unregistered else None,
        "registry_false_negatives": fns,
        "registry_false_negative_rate": fns / len(registered) if registered else None,
        "vehicle_detection_precision_recall": None,
        "plate_localization_iou": None,
        "tracking_identity_consistency": None,
        "tracking_coverage_mean": round(sum(valid_coverage) / len(valid_coverage), 4) if valid_coverage else None,
        "processing_seconds_mean": round(sum(valid_runtime) / len(valid_runtime), 2) if valid_runtime else None,
        "metric_limits": "No vehicle boxes, plate boxes, or track identity labels exist in ground_truth.json; detection, localization IoU, and ID consistency are not measurable.",
    }


def _file_info(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "available": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None}


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory, e.g. jobs/evaluations/run-01")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace evaluation.json and clip outputs in the requested output directory")
    args = parser.parse_args()
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out = out.resolve()
    if out == (ROOT / "reports").resolve() or (ROOT / "reports").resolve() in out.parents:
        raise SystemExit("Refusing to write evaluation outputs inside preserved reports/.")
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit("Output directory is not empty; choose a new path or pass --overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    clips_dir = ROOT / "test_clips"
    gt_path = clips_dir / "ground_truth.json"
    gt_bytes = gt_path.read_bytes()
    ground_truth = json.loads(gt_bytes)
    clip_files = sorted(clips_dir.glob("*.mp4"))
    expected_clips = set(ground_truth)
    actual_clips = {clip.stem for clip in clip_files}
    if actual_clips != expected_clips:
        raise SystemExit(f"Ground-truth/video mismatch. Unlabeled videos={sorted(actual_clips - expected_clips)}; "
                         f"videos missing={sorted(expected_clips - actual_clips)}")
    registry = matcher.load_registry(ROOT / "data" / "registry.csv")
    rows = []
    for clip in clip_files:
        started = time.perf_counter()
        result = VideoAnalysis(clip, out / clip.stem, registry).analyse({})
        truth = ground_truth.get(clip.stem, {})
        track = result.get("track", {})
        summary = {
            "clip": clip.stem,
            "plate": result.get("fusion", {}).get("text", ""),
            "registry_decision": result.get("match", {}).get("registry_status", result.get("match", {}).get("status")),
            "system_decision": result.get("match", {}).get("status"),
            "top_candidate": ((result.get("match", {}).get("candidates") or [None])[0] or {}).get("vehicle", {}).get("reg_no"),
            "correct": decision_correct(result, truth),
            "track_coverage": round(track.get("frames", 0) / max(result.get("video", {}).get("sampled", 1), 1), 4),
            "seconds": round(time.perf_counter() - started, 2),
        }
        rows.append(summary)
        print(json.dumps(summary), flush=True)

    baseline = []
    for row in rows:
        path = ROOT / "reports" / row["clip"] / "result.json"
        if not path.is_file():
            raise SystemExit(f"Missing preserved baseline result for {row['clip']}: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        baseline.append({
            "clip": row["clip"],
            "correct": decision_correct(result, ground_truth.get(row["clip"], {})),
            "plate": result.get("fusion", {}).get("text", ""),
            "registry_decision": result.get("match", {}).get("registry_status", result.get("match", {}).get("status")),
            "system_decision": result.get("match", {}).get("status"),
            "top_candidate": ((result.get("match", {}).get("candidates") or [None])[0] or {}).get("vehicle", {}).get("reg_no"),
            "track_coverage": result.get("track", {}).get("frames", 0) / max(result.get("video", {}).get("sampled", 1), 1),
            "seconds": result.get("elapsed_s", 0),
        })
    if {r["clip"] for r in baseline} != {r["clip"] for r in rows}:
        raise SystemExit("Baseline and updated results do not cover the same clips.")
    report = {
        "ground_truth_sha256": hashlib.sha256(gt_bytes).hexdigest(),
        "benchmark_clips": sorted(expected_clips),
        "metric_definitions": {
            "character_error_rate": "micro Levenshtein edits / total reference characters; may exceed 1",
            "normalized_edit_similarity": "1 - summed Levenshtein edits / sum(max(prediction length, reference length)); micro",
            "macro_metrics": "unweighted mean of per-clip metric values",
            "plate_normalization": "uppercase; remove spaces and hyphens only; '?' remains an error symbol",
            "missing_ground_truth": "excluded from OCR and decision accuracy denominators and reported separately",
        },
        "runtime": {
            "platform": platform.platform(),
            "device": engines.DEVICE,
            "mps_available": bool(getattr(getattr(engines.torch.backends, "mps", None),
                                           "is_available", lambda: False)()),
            "peak_process_rss_bytes": _peak_rss_bytes(),
            "models": {
                "vehicle_detector": {"name": "Ultralytics YOLO11n",
                                     **_file_info(ROOT / "models" / "yolo11n.pt"),
                                     "inference_resolution": "640 default"},
                "plate_detector": {"name": "Babblu2821/alpr-plate-detector YOLOv8s",
                                   **_file_info(engines.PLATE_DETECTOR_WEIGHTS),
                                   "inference_resolution": 640},
                "ocr": {"name": "EasyOCR English CRNN (english_g2)",
                        **_file_info(ROOT / "models" / "easyocr" / "english_g2.pth"),
                        "input_height": 64, "width_padding": "right-padded to imgH multiple"},
            },
        },
        "baseline": measure(baseline, ground_truth),
        "updated": measure(rows, ground_truth),
        "clips": rows,
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": report["baseline"], "updated": report["updated"]}, indent=2))


if __name__ == "__main__":
    main()
