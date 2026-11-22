"""Multi-frame plate recovery.

Each frame gives a sequence of characters, each with a 36-way probability
distribution (or an explicit '?' gap). We

1. pick a reference reading (weighted-mode length, best quality),
2. align every other reading to it with Needleman-Wunsch using the
   dot-product of character distributions as similarity,
3. pool aligned distributions per position with quality weights
   (linear opinion pool = weighted vote),
4. declare a position KNOWN only if the pooled winner is strong and seen in
   enough frames; otherwise the position is output as '?' (unknown) together
   with its candidate alternatives. Nothing is ever filled in by guessing.
"""
from __future__ import annotations

import numpy as np

from .vocab import CHARS, INDIAN_STATE_CODES

KNOWN_MIN_P = 0.55
KNOWN_MIN_MARGIN = 0.18
KNOWN_MIN_FRAMES = 2
_IS_DIGIT = np.array([c.isdigit() for c in CHARS], dtype=float)
_IS_ALPHA = 1.0 - _IS_DIGIT


def _apply_indian_syntax_prior(pooled_list: list) -> list:
    """Soft letter/digit positional prior for standard Indian registration plates.

    Only activates when the plate has >= 7 positions and the first two positions
    look like a 2-letter state code. Never invents characters at unobserved/low-signal
    positions; it only helps disambiguate letter/digit look-alikes (1/I, 2/Z, 0/O, 5/S, 8/B).
    """
    L = len(pooled_list)
    if L < 7 or pooled_list[0] is None or pooled_list[1] is None:
        return pooled_list
    if (pooled_list[0] * _IS_ALPHA).sum() < 0.45 or (pooled_list[1] * _IS_ALPHA).sum() < 0.45:
        return pooled_list
    c0 = CHARS[int(np.argmax(pooled_list[0] * _IS_ALPHA))]
    c1 = CHARS[int(np.argmax(pooled_list[1] * _IS_ALPHA))]
    is_dl = (c0 + c1) == "DL"
    out = []
    for i, p in enumerate(pooled_list):
        if p is None:
            out.append(None)
            continue
        kind = None
        if i in (0, 1):
            kind = "L"
        elif i == 2:
            kind = "D"
        elif i == 3 and not is_dl:
            kind = "D"
        elif L == 10 and i in (4, 5):
            kind = "L"
        elif L == 10 and i >= 6:
            kind = "D"
        elif L == 9 and i >= 5:
            kind = "D"
        if kind == "D":
            w = 1.0 + 1.5 * _IS_DIGIT
        elif kind == "L":
            w = 1.0 + 1.5 * _IS_ALPHA
        else:
            w = np.ones_like(p)
        q = p * w
        out.append(q / q.sum())
    return out


def _sim(a, b):
    # A stacked commercial plate has two independent text rows. Do not align a
    # character from one row to a glyph on the other row when frame OCR lengths
    # differ slightly; treating a cross-row pairing as a mismatch makes the
    # Needleman-Wunsch pass prefer separate insertion/deletion gaps instead.
    if a.get("line", 0) != b.get("line", 0):
        return -1.5
    if a["dist"] is None or b["dist"] is None:
        return 0.1
    return 2.2 * float(np.dot(a["dist"], b["dist"])) - 0.6


def _nw(ref, seq, gap=-0.7):
    n, m = len(ref), len(seq)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1) * gap
    D[0, :] = np.arange(m + 1) * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = max(D[i - 1, j - 1] + _sim(ref[i - 1], seq[j - 1]), D[i - 1, j] + gap, D[i, j - 1] + gap)
    i, j, pairs, inserts = n, m, {}, {}
    while i > 0 and j > 0:
        if np.isclose(D[i, j], D[i - 1, j - 1] + _sim(ref[i - 1], seq[j - 1])):
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif np.isclose(D[i, j], D[i - 1, j] + gap):
            i -= 1
        else:
            inserts.setdefault(i - 1, []).append(j - 1)  # seq char after ref position i-1
            j -= 1
    while j > 0:
        inserts.setdefault(-1, []).append(j - 1)
        j -= 1
    return pairs, inserts


def fuse(readings: list) -> dict:
    """readings: [{'frame': int, 'weight': float, 'chars': [{'char','conf','dist'(np|None)}]}]"""
    readings = [r for r in readings if r["chars"] and r["weight"] > 0]
    if not readings:
        return {"positions": [], "text": "", "n_frames": 0, "reference_frame": None}
    # weighted mode of length
    lw = {}
    for r in readings:
        lw[len(r["chars"])] = lw.get(len(r["chars"]), 0) + r["weight"]
    L = max(lw, key=lw.get)
    ref = max((r for r in readings if len(r["chars"]) == L), key=lambda r: r["weight"])
    total_w = sum(r["weight"] for r in readings)

    slots = [[] for _ in range(L)]          # per ref position: (frame, dist|None, weight)
    ins_w = {}
    for r in readings:
        pairs, inserts = _nw(ref["chars"], r["chars"]) if r is not ref else ({i: i for i in range(L)}, {})
        for i, j in pairs.items():
            c = r["chars"][j]
            slots[i].append((r["frame"], c["dist"], r["weight"], c["char"], c["conf"]))
        for i, js in inserts.items():
            ins_w[i] = ins_w.get(i, 0) + r["weight"]

    raw_Auto = []
    meta_slots = []
    for i, votes in enumerate(slots):
        dvotes = [(d, w) for _, d, w, _, _ in votes if d is not None]
        n_obs = len(dvotes)
        if dvotes:
            pooled = sum(d * w for d, w in dvotes) / sum(w for _, w in dvotes)
            obs_frac = sum(w for _, w in dvotes) / total_w
        else:
            pooled, obs_frac = None, 0.0
        raw_Auto.append(pooled)
        meta_slots.append((votes, dvotes, n_obs, obs_frac))

    refined = _apply_indian_syntax_prior(raw_Auto)

    positions = []
    for i, (pooled, (votes, dvotes, n_obs, obs_frac)) in enumerate(zip(refined, meta_slots)):
        if pooled is None:
            pooled = np.full(len(CHARS), 1 / len(CHARS))
        order = np.argsort(-pooled)
        p1, p2 = float(pooled[order[0]]), float(pooled[order[1]])
        # agreement-adjusted confidence: shrink toward uncertainty when few frames saw the position
        conf = p1 * min(1.0, 0.5 + obs_frac)
        need = min(KNOWN_MIN_FRAMES, len(readings))
        known = bool(dvotes) and n_obs >= need and conf >= KNOWN_MIN_P and (p1 - p2) >= KNOWN_MIN_MARGIN
        best = CHARS[order[0]]
        positions.append({
            "idx": i, "char": best if known else "?", "best_guess": best if dvotes else None,
            "conf": round(conf, 3), "known": bool(known), "n_frames": n_obs,
            "alts": [(CHARS[k], round(float(pooled[k]), 3)) for k in order[:3]] if dvotes else [],
            "dist": {CHARS[k]: round(float(pooled[k]), 4) for k in order[:8]} if dvotes else {},
            "votes": [{"frame": f, "char": ch, "conf": round(cf, 3)} for f, _, _, ch, cf in votes],
        })
    return {"positions": positions, "text": "".join(p["char"] for p in positions), "n_frames": len(readings),
            "reference_frame": ref["frame"],
            "insertions": {int(k): round(v / total_w, 3) for k, v in ins_w.items()}}
