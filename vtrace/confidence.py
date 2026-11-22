"""Evidence-component scores and conservative recognition review gating.

These are rule-based quality scores, not statistically calibrated probabilities.
The eight synthetic benchmark clips are too small to fit or validate calibration.
"""
from __future__ import annotations

import numpy as np


def assess(track: dict, frames: list[dict], fusion: dict, match: dict) -> dict:
    boxes = track.get("_boxes", [])
    vehicle = float(np.mean([b[2] for b in boxes])) if boxes else 0.0
    direct = [f for f in frames if f.get("plate_source") != "propagated"]
    localization = float(np.mean([f.get("loc_score", 0.0) for f in direct])) if direct else 0.0
    used = [f for f in frames if f.get("used_in_fusion") and f.get("weight", 0) > 0]
    ocr = float(np.average([f.get("mean_conf", 0.0) for f in used],
                           weights=[max(1e-4, f.get("weight", 0.0)) for f in used])) if used else 0.0
    positions = fusion.get("positions", [])
    known = [p for p in positions if p.get("known")]
    support = float(np.mean([p.get("conf", 0.0) for p in known])) if known else 0.0
    support *= min(1.0, fusion.get("n_frames", 0) / 4.0)
    top = (match.get("candidates") or [None])[0]
    registry = (top or {}).get("confidence", 0.0) / 100.0

    # Use multiple independent evidence gates; do not multiply raw model scores
    # or present their combination as a probability.
    if (match.get("status") == "match" and len(known) >= 6
            and fusion.get("n_frames", 0) >= 3 and localization >= 0.45
            and ocr >= 0.55 and support >= 0.55):
        tier = "high"
        recommendation = "Eligible for automatic recognition"
    elif len(known) >= 3 or (used and localization >= 0.25):
        tier = "medium"
        recommendation = "Needs human review; retain as a lead"
    else:
        tier = "low"
        recommendation = "Uncertain; do not treat as a confirmed identity"
    return {
        "vehicle_detection": round(float(np.clip(vehicle, 0, 1)), 3),
        "plate_localization": round(float(np.clip(localization, 0, 1)), 3),
        "ocr": round(float(np.clip(ocr, 0, 1)), 3),
        "multi_frame_agreement": round(float(np.clip(support, 0, 1)), 3),
        "registry_match_posterior": round(float(np.clip(registry, 0, 1)), 3),
        "known_characters": len(known),
        "frames_in_fusion": int(fusion.get("n_frames", 0)),
        "tier": tier,
        "recommendation": recommendation,
        "automatic_eligible": tier == "high" and match.get("status") == "match",
        "method": "conservative evidence gates; not statistically calibrated",
    }


def gate_registry_match(match: dict, confidence: dict) -> dict:
    """Preserve the ranked registry lead but abstain from an unqualified match."""
    out = dict(match)
    out["registry_status"] = match.get("registry_status", match.get("status"))
    if match.get("status") == "match" and not confidence.get("automatic_eligible"):
        out["status"] = "review_required"
        out["message"] = (
            "Registry candidate found, but recognition evidence is not strong enough "
            "for automatic confirmation. Human review is required."
        )
    return out


def reassess_registry(scores: dict, match: dict) -> dict:
    """Re-evaluate the automatic gate after a no-OCR investigator re-match."""
    out = dict(scores)
    eligible = (
        match.get("status") == "match"
        and out.get("known_characters", 0) >= 6
        and out.get("frames_in_fusion", 0) >= 3
        and out.get("plate_localization", 0) >= 0.45
        and out.get("ocr", 0) >= 0.55
        and out.get("multi_frame_agreement", 0) >= 0.55
    )
    out["automatic_eligible"] = bool(eligible)
    if eligible:
        out["tier"] = "high"
        out["recommendation"] = "Eligible for automatic recognition"
    elif match.get("status") == "match":
        out["tier"] = "medium" if out.get("known_characters", 0) >= 3 else "low"
        out["recommendation"] = "Needs human review; retain as a lead"
    return out
