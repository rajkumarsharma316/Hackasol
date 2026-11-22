"""Cross-camera vehicle re-identification and multi-clip evidence fusion.

Compares two analysed CCTV clips (Camera A and Camera B) to determine whether
they show the same vehicle, explains every supporting/conflicting cue
(partial plate overlap, CLIP visual appearance embedding, colour, body type,
make/model), and produces a joint registry shortlist combining evidence from
both cameras.
"""
from __future__ import annotations

import math

import numpy as np

from . import attributes, matcher
from .vocab import (BODY_CONFUSION, BODY_TYPES, CHAR_CONFUSION, CHARS,
                    COLOUR_CONFUSION, COLOURS, UNKNOWN, body_similarity,
                    char_similarity, colour_similarity)


def _char_posterior(token: matcher.PlateToken) -> np.ndarray:
    """P(true char = c | token) under uniform prior."""
    if token.kind != "dist":
        return np.full(len(CHARS), 1.0 / len(CHARS))
    lik = token.dist @ CHAR_CONFUSION
    return lik / lik.sum()


def align_two_plates(toks_a: list[matcher.PlateToken], toks_b: list[matcher.PlateToken]):
    """Symmetric semi-global alignment between two partial plate reads."""
    m, n = len(toks_a), len(toks_b)
    if m == 0 or n == 0:
        return 0.0, [], {"exact": 0, "similar": 0, "mismatch": 0, "unknown": 0, "overlap": 0}
    pa = [_char_posterior(t) for t in toks_a]
    pb = [_char_posterior(t) for t in toks_b]
    llr = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if toks_a[i].kind != "dist" or toks_b[j].kind != "dist":
                llr[i, j] = 0.0
            else:
                # LR = P(same true char) / (1/|CHARS|)
                lr = float(np.dot(pa[i], pb[j]) * len(CHARS))
                llr[i, j] = math.log(max(lr, 1e-4))

    END = 0.25
    GAP = 2.2
    NEG = -1e9
    D = np.full((m + 1, n + 1), NEG)
    B = np.zeros((m + 1, n + 1), dtype=np.int8)
    D[0, 0] = 0.0
    for i in range(1, m + 1):
        D[i, 0] = -END * i
        B[i, 0] = 2
    for j in range(1, n + 1):
        D[0, j] = -END * j
        B[0, j] = 3
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cands = [
                (D[i - 1, j - 1] + llr[i - 1, j - 1], 1),
                (D[i - 1, j] - (END if j in (0, n) else GAP), 2),
                (D[i, j - 1] - (END if i in (0, m) else GAP), 3),
            ]
            best, arg = max(cands, key=lambda x: x[0])
            D[i, j], B[i, j] = best, arg

    score = float(D[m, n])
    i, j = m, n
    rows = []
    exact = similar = mismatch = unknown = overlap = 0
    while i > 0 or j > 0:
        a = int(B[i, j])
        if a == 1:
            ta, tb = toks_a[i - 1], toks_b[j - 1]
            s = float(llr[i - 1, j - 1])
            if ta.kind != "dist" or tb.kind != "dist":
                rel = "unknown"
                unknown += 1
            else:
                overlap += 1
                if ta.label == tb.label:
                    rel, exact = "exact", exact + 1
                elif char_similarity(ta.label, tb.label) >= 0.3 or s > 0:
                    rel, similar = "similar", similar + 1
                else:
                    rel, mismatch = "mismatch", mismatch + 1
            rows.append({
                "cam_a": ta.label if ta.kind == "dist" else UNKNOWN,
                "cam_b": tb.label if tb.kind == "dist" else UNKNOWN,
                "rel": rel,
                "llr": round(s, 2),
            })
            i, j = i - 1, j - 1
        elif a == 2:
            ta = toks_a[i - 1]
            rows.append({
                "cam_a": ta.label if ta.kind == "dist" else UNKNOWN,
                "cam_b": "",
                "rel": "not-visible",
                "llr": round(-END, 2),
            })
            i -= 1
        else:
            tb = toks_b[j - 1]
            rows.append({
                "cam_a": "",
                "cam_b": tb.label if tb.kind == "dist" else UNKNOWN,
                "rel": "not-visible",
                "llr": round(-END, 2),
            })
            j -= 1
    rows.reverse()
    return score, rows, {
        "exact": exact, "similar": similar, "mismatch": mismatch,
        "unknown": unknown, "overlap": overlap,
    }


def _dist_compat_llr(da: dict | None, db: dict | None, labels: list[str], conf_m: np.ndarray) -> float:
    if not da or not db:
        return 0.0
    qa = np.array([da.get(l, 0.0) for l in labels], dtype=float)
    qb = np.array([db.get(l, 0.0) for l in labels], dtype=float)
    if qa.sum() <= 0 or qb.sum() <= 0:
        return 0.0
    qa /= qa.sum()
    qb /= qb.sum()
    la = qa @ conf_m
    lb = qb @ conf_m
    la /= la.sum()
    lb /= lb.sum()
    lr = float(np.dot(la, lb) * len(labels))
    return float(math.log(max(lr, 1e-3)))


def compare_clips(res_a: dict, res_b: dict, registry: list) -> dict:
    """Compare two clip analysis results and fuse their evidence against the registry."""
    pos_a = res_a["fusion"]["positions"]
    pos_b = res_b["fusion"]["positions"]
    toks_a = matcher.tokens_from_fused(pos_a) if pos_a else []
    toks_b = matcher.tokens_from_fused(pos_b) if pos_b else []

    plate_llr, align_rows, align_stats = align_two_plates(toks_a, toks_b)

    col_a = res_a["attributes"].get("colour") or {}
    col_b = res_b["attributes"].get("colour") or {}
    body_a = res_a["attributes"].get("body_type") or {}
    body_b = res_b["attributes"].get("body_type") or {}
    mm_a = res_a["attributes"].get("make_model") or {}
    mm_b = res_b["attributes"].get("make_model") or {}

    colour_llr = 0.8 * _dist_compat_llr(col_a, col_b, COLOURS, COLOUR_CONFUSION)
    body_llr = 0.9 * _dist_compat_llr(body_a, body_b, BODY_TYPES, BODY_CONFUSION)

    emb_a = (res_a.get("signature") or {}).get("embedding")
    emb_b = (res_b.get("signature") or {}).get("embedding")
    if emb_a and emb_b:
        va = np.array(emb_a, dtype=float)
        vb = np.array(emb_b, dtype=float)
        cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        # CLIP ViT-B/32 cosine between random road vehicles sits around ~0.72; same vehicle ~0.85+
        app_llr = float(np.clip((cos - 0.74) * 14.0, -3.0, 3.5))
    else:
        cos = None
        app_llr = 0.0

    total_llr = plate_llr + colour_llr + body_llr + app_llr
    # Convert log-LR to posterior probability with a conservative prior P(same) = 0.25
    log_prior_odds = math.log(0.25 / 0.75)
    prob_same = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, total_llr + log_prior_odds))))
    conf_pct = round(prob_same * 100, 1)

    # Appearance and coarse attributes can nominate a review candidate, but may
    # never establish identity by themselves. Require several readable plate
    # positions to agree before using the definitive same-vehicle verdict.
    plate_identity_supported = (
        align_stats["overlap"] >= 3
        and align_stats["exact"] + align_stats["similar"] >= 3
        and align_stats["mismatch"] == 0
        and plate_llr >= 3.0
    )
    if conf_pct >= 75 and plate_identity_supported:
        verdict = "same_vehicle"
        summary = f"High confidence ({conf_pct}%) that both clips capture the SAME vehicle."
    elif conf_pct >= 40:
        verdict = "possible_same_vehicle"
        summary = f"Moderate confidence ({conf_pct}%) — compatible cues, treat as possible same vehicle."
    else:
        verdict = "different_vehicle"
        summary = f"Low similarity ({conf_pct}%) — clips likely show DIFFERENT vehicles."

    reasons = []
    pa_txt = res_a["fusion"]["text"] or "none"
    pb_txt = res_b["fusion"]["text"] or "none"
    if align_stats["overlap"] > 0:
        rtxt = (f"Partial plate comparison ('{pa_txt}' vs '{pb_txt}'): "
                f"{align_stats['exact']}/{align_stats['overlap']} overlapping readable character(s) match exactly")
        if align_stats["similar"]:
            rtxt += f", {align_stats['similar']} visually similar"
        if align_stats["mismatch"]:
            rtxt += f", {align_stats['mismatch']} conflicting"
        reasons.append({"factor": "plate overlap", "llr": round(plate_llr, 2), "text": rtxt + "."})
    else:
        reasons.append({"factor": "plate overlap", "llr": 0.0,
                        "text": f"No overlapping readable plate characters ('{pa_txt}' vs '{pb_txt}')."})

    if cos is not None:
        reasons.append({
            "factor": "visual appearance (CLIP)",
            "llr": round(app_llr, 2),
            "text": f"CLIP visual embedding cosine similarity is {cos:.3f} "
                    f"({'strong visual match' if cos >= 0.82 else 'moderate similarity' if cos >= 0.75 else 'distinct appearance'}).",
        })
    if col_a and col_b:
        ca, cb = max(col_a, key=col_a.get), max(col_b, key=col_b.get)
        rel = "matches" if ca == cb else ("is compatible with" if colour_similarity(ca, cb) >= 0.3 else "differs from")
        reasons.append({
            "factor": "colour",
            "llr": round(colour_llr, 2),
            "text": f"Clip A colour '{ca}' ({col_a[ca]:.0%}) {rel} Clip B colour '{cb}' ({col_b[cb]:.0%}).",
        })
    if body_a and body_b:
        ba, bb = max(body_a, key=body_a.get), max(body_b, key=body_b.get)
        rel = "matches" if ba == bb else ("is compatible with" if body_similarity(ba, bb) >= 0.3 else "differs from")
        reasons.append({
            "factor": "body type",
            "llr": round(body_llr, 2),
            "text": f"Clip A body type '{ba}' ({body_a[ba]:.0%}) {rel} Clip B body type '{bb}' ({body_b[bb]:.0%}).",
        })

    # Joint registry matching using evidence from BOTH cameras
    ev = matcher.Evidence()
    if any(p["known"] for p in pos_a):
        ev.plates.append(toks_a)
        ev.plate_sources.append(f"Camera A ({res_a['video']['file']})")
    if any(p["known"] for p in pos_b):
        ev.plates.append(toks_b)
        ev.plate_sources.append(f"Camera B ({res_b['video']['file']})")
    ev.colour = attributes.merge_dists([col_a, col_b]) or None
    ev.body = attributes.merge_dists([body_a, body_b]) or None
    ev.model = attributes.merge_dists([mm_a, mm_b]) or None
    joint_match = matcher.match(ev, registry)

    return {
        "verdict": verdict,
        "confidence": conf_pct,
        "summary": summary,
        "scores": {
            "total_llr": round(total_llr, 2),
            "plate_llr": round(plate_llr, 2),
            "appearance_llr": round(app_llr, 2),
            "colour_llr": round(colour_llr, 2),
            "body_llr": round(body_llr, 2),
            "clip_cosine": round(cos, 4) if cos is not None else None,
        },
        "plate_alignment": {"rows": align_rows, "stats": align_stats},
        "reasons": reasons,
        "joint_match": joint_match,
    }
