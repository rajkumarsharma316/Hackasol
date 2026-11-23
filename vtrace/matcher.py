"""TRACE matcher - the team's vehicle matching & ranking algorithm.

Idea
----
Every piece of evidence (each recovered plate character, the colour, the body
type, optionally make/model) is turned into a *likelihood ratio* (LR):

    LR = P(evidence | this registry vehicle) / P(evidence | a random vehicle)

LR > 1 supports the candidate, LR < 1 counts against it, LR = 1 is neutral.
Unknown plate characters are therefore exactly neutral - they never help or
hurt a candidate, so nothing is ever "invented".

Plate evidence is aligned against each registry plate with a semi-global
dynamic-programming alignment (like DNA sequence alignment) so that

* a partial read ("12DE45") can sit anywhere inside a longer plate,
* a missing / extra character from OCR is tolerated with a penalty,
* visually similar glyphs (8/B, 0/O/D, 5/S, 1/I ...) earn partial credit via a
  confusion matrix instead of a hard mismatch.

The log-LRs are summed (naive-Bayes evidence combination) and converted into a
posterior over the registry *plus a "not in registry / no fit" hypothesis*.
That null hypothesis is what produces honest "no confident match" answers:
if no registry entry explains the evidence better than chance, or if many
entries explain it equally well, no single candidate gets a high posterior.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .vocab import (BODY_CONFUSION, BODY_TYPES, CHAR_CONFUSION, CHAR_IDX, CHARS,
                    COLOUR_CONFUSION, COLOURS, UNKNOWN, body_similarity,
                    char_similarity, colour_similarity)

# Alignment costs (natural-log units).
GAP_OBS = 2.5      # OCR produced a character that is not on the plate
GAP_REG = 2.5      # OCR missed a character in the middle of the plate
END_SKIP = 0.35    # plate characters outside the visible window (cropped/occluded edge)

W_COLOUR = 0.8     # attribute weights (<1 = trust less than plate characters)
W_BODY = 0.8
W_MODEL = 0.5

PRIOR_NULL = 0.20  # prior mass for "vehicle not in registry / plate misread"
CONFIDENT_POSTERIOR = 0.50
MIN_PLATE_LLR = 4.0  # a confident match needs real plate support, not just colour


# ---------------------------------------------------------------------------
# Evidence containers
# ---------------------------------------------------------------------------
@dataclass
class PlateToken:
    kind: str                       # "dist" | "unk" | "star"
    dist: Optional[np.ndarray] = None   # 36-vector of char probabilities
    label: str = UNKNOWN            # best char (display only)
    conf: float = 0.0               # max probability (display only)

    @property
    def lr_vec(self) -> np.ndarray:
        if self.kind != "dist":
            return np.ones(len(CHARS))
        lik = self.dist @ CHAR_CONFUSION       # l(c) = sum_k q_k P(read k | c)
        return lik / lik.mean()


@dataclass
class Evidence:
    plates: list = field(default_factory=list)      # list[list[PlateToken]] (video, hint ...)
    plate_sources: list = field(default_factory=list)
    colour: Optional[dict] = None                    # {colour: prob}
    body: Optional[dict] = None                      # {type: prob}
    model: Optional[dict] = None                     # {"Make Model": prob}


def tokens_from_pattern(pattern: str, conf: float = 0.95) -> list:
    """Investigator-typed pattern, e.g. 'MH12??45*' ('?' = one unknown, '*' = any run)."""
    toks = []
    for ch in pattern.upper().replace(" ", "").replace("-", ""):
        if ch == "*":
            toks.append(PlateToken("star"))
        elif ch == "?" or ch not in CHAR_IDX:
            toks.append(PlateToken("unk"))
        else:
            d = np.full(len(CHARS), (1 - conf) / (len(CHARS) - 1))
            d[CHAR_IDX[ch]] = conf
            toks.append(PlateToken("dist", d, ch, conf))
    return toks


def tokens_from_fused(positions: list) -> list:
    """Positions produced by multi-frame fusion: {'char', 'conf', 'dist' (dict) , 'known'}."""
    toks = []
    for p in positions:
        if not p.get("known") or not p.get("dist"):
            toks.append(PlateToken("unk"))
            continue
        d = np.array([p["dist"].get(c, 0.0) for c in CHARS], dtype=float) + 1e-4
        d /= d.sum()
        toks.append(PlateToken("dist", d, p["char"], p["conf"]))
    return toks


# ---------------------------------------------------------------------------
# Plate alignment
# ---------------------------------------------------------------------------
def align_plate(tokens: list, plate: str):
    """Semi-global alignment. Returns (log_lr, ops) where ops explain each step."""
    m, n = len(tokens), len(plate)
    if m == 0:
        return 0.0, []
    lr = np.ones((m, n))
    plate_idx = np.array([CHAR_IDX.get(ch, -1) for ch in plate], dtype=int)
    for i, t in enumerate(tokens):
        if t.kind == "dist":
            v = t.lr_vec
            valid = plate_idx >= 0
            lr[i, valid] = v[plate_idx[valid]]
    llr = np.log(lr)

    NEG = -1e9
    D = np.full((m + 1, n + 1), NEG)
    B = np.zeros((m + 1, n + 1), dtype=np.int8)  # 0 start,1 diag,2 up(obs gap),3 left(reg skip),4 star-up,5 star-left
    for j in range(n + 1):
        D[0, j] = -END_SKIP * j
    for i in range(1, m + 1):
        t = tokens[i - 1]
        for j in range(n + 1):
            best, arg = NEG, 0
            if t.kind == "star":
                if D[i - 1, j] > best:
                    best, arg = D[i - 1, j], 4
                if j > 0 and D[i, j - 1] > best:
                    best, arg = D[i, j - 1], 5
            else:
                if j > 0 and D[i - 1, j - 1] + llr[i - 1, j - 1] > best:
                    best, arg = D[i - 1, j - 1] + llr[i - 1, j - 1], 1
                if D[i - 1, j] - GAP_OBS > best:
                    best, arg = D[i - 1, j] - GAP_OBS, 2
                if j > 0 and i < m and D[i, j - 1] - GAP_REG > best:
                    best, arg = D[i, j - 1] - GAP_REG, 3
            D[i, j], B[i, j] = best, arg
    finals = [D[m, j] - END_SKIP * (n - j) for j in range(n + 1)]
    jend = int(np.argmax(finals))
    score = float(finals[jend])

    ops = [("skip_end", None, plate[j]) for j in range(n - 1, jend - 1, -1)]
    i, j = m, jend
    while i > 0:
        a = B[i, j]
        if a == 1:
            ops.append(("align", i - 1, plate[j - 1], float(llr[i - 1, j - 1])))
            i, j = i - 1, j - 1
        elif a == 2:
            ops.append(("obs_gap", i - 1, None))
            i -= 1
        elif a == 3:
            ops.append(("reg_gap", None, plate[j - 1]))
            j -= 1
        elif a == 4:
            i -= 1
        elif a == 5:
            ops.append(("star", None, plate[j - 1]))
            j -= 1
        else:
            break
    ops += [("skip_start", None, plate[k]) for k in range(j - 1, -1, -1)]
    ops.reverse()
    return score, ops


def describe_alignment(tokens, plate, ops):
    """Turn alignment ops into per-character rows and human-readable reasons."""
    rows, exact, similar, mismatch, unknown, known = [], 0, 0, 0, 0, 0
    for op in ops:
        kind = op[0]
        if kind == "align":
            t = tokens[op[1]]
            reg = op[2]
            if t.kind != "dist":
                rel = "unknown"
                unknown += 1
            else:
                known += 1
                if t.label == reg:
                    rel, exact = "exact", exact + 1
                elif char_similarity(t.label, reg) >= 0.3 or op[3] > 0:
                    rel, similar = "similar", similar + 1
                else:
                    rel, mismatch = "mismatch", mismatch + 1
            rows.append({"obs": t.label if t.kind == "dist" else UNKNOWN,
                         "conf": round(t.conf, 3), "reg": reg, "rel": rel,
                         "llr": round(op[3], 2)})
        elif kind == "obs_gap":
            t = tokens[op[1]]
            known += t.kind == "dist"
            rows.append({"obs": t.label if t.kind == "dist" else UNKNOWN,
                         "conf": round(t.conf, 3), "reg": "", "rel": "extra", "llr": -GAP_OBS})
        elif kind == "reg_gap":
            rows.append({"obs": "", "conf": 0, "reg": op[2], "rel": "missed", "llr": -GAP_REG})
        elif kind in ("skip_start", "skip_end", "star"):
            rows.append({"obs": "", "conf": 0, "reg": op[2], "rel": "not-visible",
                         "llr": 0.0 if kind == "star" else -END_SKIP})
    return rows, dict(exact=exact, similar=similar, mismatch=mismatch, unknown=unknown,
                      known=known)


# ---------------------------------------------------------------------------
# Attribute likelihood ratios
# ---------------------------------------------------------------------------
def _label_llr(obs: Optional[dict], labels, conf_m, value) -> float:
    if not obs or value not in labels:
        return 0.0
    q = np.array([obs.get(l, 0.0) for l in labels], dtype=float)
    if q.sum() <= 0:
        return 0.0
    q /= q.sum()
    lik = q @ conf_m
    return float(np.log(lik[labels.index(value)] / lik.mean()))


def _model_llr(obs: Optional[dict], make: str, model: str, all_models) -> float:
    if not obs:
        return 0.0
    key = f"{make} {model}"
    make_mass = sum(p for k, p in obs.items() if k.startswith(make + " "))

    def lik(k, mk):
        return 0.05 + obs.get(k, 0.0) + 0.3 * sum(p for kk, p in obs.items() if kk.startswith(mk + " "))

    bg = np.mean([lik(f"{a} {b}", a) for a, b in all_models]) if all_models else 1.0
    return float(np.log((0.05 + obs.get(key, 0.0) + 0.3 * make_mass) / bg))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def load_registry(path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["reg_no"] = r["reg_no"].upper().replace(" ", "")
    return rows


def _top(d: Optional[dict]):
    if not d:
        return None, 0.0
    k = max(d, key=d.get)
    return k, d[k]


def match(evidence: Evidence, registry: list, top_k: int = 5) -> dict:
    all_models = sorted({(r["make"], r["model"]) for r in registry})
    n = len(registry)
    if n == 0:
        return {
            "status": "no_confident_match",
            "message": "No vehicles are available in the registry to compare against.",
            "p_not_in_registry": 100.0,
            "n_registry": 0,
            "n_plausible": 0,
            "candidates": [],
        }
    scored = []
    for r in registry:
        plate_llr, plate_ops = 0.0, []
        for toks, src in zip(evidence.plates, evidence.plate_sources):
            s, ops = align_plate(toks, r["reg_no"])
            plate_llr += s
            # Rendering explanations for every registry row is costly and is
            # discarded immediately. Keep alignment paths, then expand only
            # the handful of candidates returned to the investigator.
            plate_ops.append((toks, src, s, ops))
        c_llr = W_COLOUR * _label_llr(evidence.colour, COLOURS, COLOUR_CONFUSION, r["colour"])
        b_llr = W_BODY * _label_llr(evidence.body, BODY_TYPES, BODY_CONFUSION, r["body_type"])
        m_llr = W_MODEL * _model_llr(evidence.model, r["make"], r["model"], all_models)
        total = plate_llr + c_llr + b_llr + m_llr
        scored.append((total, plate_llr, c_llr, b_llr, m_llr, plate_ops, r))

    # Posterior over registry + null hypothesis (log-sum-exp for stability).
    log_prior = math.log((1 - PRIOR_NULL) / n)
    logs = np.array([s[0] + log_prior for s in scored] + [math.log(PRIOR_NULL)])
    mx = logs.max()
    post = np.exp(logs - mx)
    post /= post.sum()
    p_null = float(post[-1])

    order = np.argsort(-logs[:-1])[:top_k]
    cands = []
    for rank, idx in enumerate(order, 1):
        total, p_llr, c_llr, b_llr, m_llr, plate_ops, r = scored[idx]
        pdet = []
        for toks, src, score, ops in plate_ops:
            rows, stats = describe_alignment(toks, r["reg_no"], ops)
            pdet.append({"source": src, "llr": round(score, 2), "rows": rows, "stats": stats})
        cands.append({
            "rank": rank,
            "vehicle": r,
            "confidence": round(float(post[idx]) * 100, 1),
            "score": {"total_llr": round(total, 2), "plate_llr": round(p_llr, 2),
                      "colour_llr": round(c_llr, 2), "body_llr": round(b_llr, 2),
                      "model_llr": round(m_llr, 2)},
            "plate_alignment": pdet,
            "reasons": explain(evidence, r, pdet, c_llr, b_llr, m_llr),
        })

    # Decision
    top = cands[0] if cands else None
    n_plausible = int(sum(1 for s in scored if s[0] > 0))
    any_plate_evidence = any(t.kind == "dist" for toks in evidence.plates for t in toks)
    if top and top["confidence"] >= CONFIDENT_POSTERIOR * 100 and top["score"]["plate_llr"] >= MIN_PLATE_LLR:
        status, msg = "match", (f"Confident match: {top['vehicle']['reg_no']} "
                                f"({top['confidence']}% posterior).")
    else:
        status = "no_confident_match"
        if not any_plate_evidence:
            why = "no plate characters could be recovered; colour/type alone cannot identify a vehicle"
        elif p_null >= 0.5 or (top and top["score"]["total_llr"] <= 0):
            why = "no registry entry fits the recovered evidence better than chance"
        elif top and top["score"]["plate_llr"] < MIN_PLATE_LLR:
            why = "plate evidence is too weak to single out a vehicle"
        else:
            share = sum(c["confidence"] for c in cands[:3])
            why = (f"evidence fits several registry vehicles similarly "
                   f"(top-3 share {share:.0f}%, {n_plausible} vehicles have positive support)")
        msg = f"No confident match - {why}. Candidates below are investigative leads only."
    return {"status": status, "message": msg, "p_not_in_registry": round(p_null * 100, 1),
            "n_registry": n, "n_plausible": n_plausible, "candidates": cands}


def explain(ev: Evidence, r: dict, pdet, c_llr, b_llr, m_llr) -> list:
    reasons = []
    for pd in pdet:
        st, rows = pd["stats"], pd["rows"]
        obs_str = "".join(x["obs"] or "" for x in rows if x["rel"] not in ("not-visible", "missed"))
        reg_aligned = "".join(x["reg"] for x in rows if x["rel"] not in ("extra",))
        txt = (f"Plate ({pd['source']}): read '{obs_str}' vs registry '{r['reg_no']}' - "
               f"{st['exact']}/{st['known']} readable characters match exactly")
        if st["similar"]:
            sims = [f"{x['obs']}~{x['reg']}" for x in rows if x["rel"] == "similar"]
            txt += f", {st['similar']} visually similar ({', '.join(sims)})"
        if st["mismatch"]:
            mm = [f"{x['obs']}!={x['reg']}" for x in rows if x["rel"] == "mismatch"]
            txt += f", {st['mismatch']} conflicting ({', '.join(mm)})"
        if st["unknown"]:
            txt += f", {st['unknown']} unreadable position(s) left neutral"
        hidden = sum(1 for x in rows if x["rel"] == "not-visible")
        if hidden:
            txt += f", {hidden} plate character(s) outside the visible part"
        gaps = sum(1 for x in rows if x["rel"] in ("extra", "missed"))
        if gaps:
            txt += f", {gaps} OCR insertion/deletion(s) tolerated"
        reasons.append({"factor": "plate", "llr": pd["llr"], "text": txt + "."})
    if ev.colour:
        oc, op = _top(ev.colour)
        rel = ("matches" if oc == r["colour"] else
               "is compatible with" if colour_similarity(oc, r["colour"]) >= 0.3 else
               "conflicts with")
        reasons.append({"factor": "colour", "llr": round(c_llr, 2),
                        "text": f"Observed colour '{oc}' ({op:.0%}) {rel} registered colour '{r['colour']}'."})
    if ev.body:
        ob, bp = _top(ev.body)
        rel = ("matches" if ob == r["body_type"] else
               "is compatible with" if body_similarity(ob, r["body_type"]) >= 0.3 else
               "conflicts with")
        reasons.append({"factor": "body type", "llr": round(b_llr, 2),
                        "text": f"Observed body type '{ob}' ({bp:.0%}) {rel} registered '{r['body_type']}'."})
    if ev.model:
        om, mp = _top(ev.model)
        reasons.append({"factor": "make/model", "llr": round(m_llr, 2),
                        "text": f"Make/model estimate '{om}' ({mp:.0%}) vs registered "
                                f"'{r['make']} {r['model']}'."})
    return reasons
