"""Small, dependency-free plate recognition metrics used by benchmark tools."""
from __future__ import annotations

import re
from statistics import mean


def normalize_plate(value) -> str:
    """Normalize harmless formatting only; keep '?' and other errors visible."""
    if value is None:
        return ""
    return re.sub(r"[\s-]+", "", str(value).upper())


def edit_distance(prediction: str, reference: str) -> int:
    """Levenshtein distance in characters, with O(len(reference)) memory."""
    if len(prediction) < len(reference):
        prediction, reference = reference, prediction
    previous = list(range(len(reference) + 1))
    for i, pred_char in enumerate(prediction, 1):
        current = [i]
        for j, ref_char in enumerate(reference, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (pred_char != ref_char)))
        previous = current
    return previous[-1]


def plate_metrics(pairs: list[tuple[str | None, str | None]]) -> dict:
    """Compute consistent micro/macro OCR metrics; missing references are skipped.

    CER uses the total number of reference characters (and can exceed 1 when
    insertions outnumber reference length). Normalized edit similarity uses
    max(prediction length, reference length) per example and is a different metric.
    Empty predictions against nonempty references incur a full-reference error.
    """
    scored, skipped = [], 0
    for prediction, reference in pairs:
        ref = normalize_plate(reference)
        if not ref:
            skipped += 1
            continue
        pred = normalize_plate(prediction)
        distance = edit_distance(pred, ref)
        denom = max(len(pred), len(ref), 1)
        scored.append((pred, ref, distance, denom))

    count = len(scored)
    refs = sum(len(ref) for _, ref, _, _ in scored)
    errors = sum(distance for _, _, distance, _ in scored)
    normalizer = sum(denom for *_, denom in scored)
    exact = sum(pred == ref for pred, ref, _, _ in scored)
    macro_cer = mean(distance / len(ref) for _, ref, distance, _ in scored) if count else None
    macro_similarity = mean(1 - distance / denom for _, _, distance, denom in scored) if count else None
    return {
        "samples": count,
        "missing_ground_truth": skipped,
        "exact_matches": exact,
        "exact_match_rate": exact / count if count else None,
        "character_errors": errors,
        "reference_characters": refs,
        "character_error_rate": errors / refs if refs else None,
        "macro_character_error_rate": macro_cer,
        "normalized_edit_similarity": 1 - errors / normalizer if normalizer else None,
        "macro_normalized_edit_similarity": macro_similarity,
        "normalized_edit_similarity_denominator": normalizer,
    }
