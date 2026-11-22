"""Plate character recognition with per-character probability distributions.

We use the pretrained EasyOCR CRNN recogniser, but call the network directly so
we get the full CTC posterior for every time-step instead of just a string.
That lets us report, for each recovered character, its probability and the
runner-up alternatives - which multi-frame fusion and the matcher need.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import engines
from .vocab import CHARS, CHAR_IDX

_map_cache = {}


def _char_map(reader):
    """Indices of model outputs that map to each of our 36 plate symbols (lower-case folded)."""
    key = id(reader)
    if key not in _map_cache:
        chars = ["[blank]"] + list(reader.character)
        groups = [[] for _ in CHARS]
        for i, c in enumerate(chars):
            u = c.upper()
            if i > 0 and u in CHAR_IDX:
                groups[CHAR_IDX[u]].append(i)
        _map_cache[key] = groups
    return _map_cache[key]


def _prep(gray: np.ndarray, imgH=64) -> torch.Tensor:
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("OCR input must have nonzero width and height")
    nw = max(1, int(imgH * w / h))
    # Match EasyOCR's AlignCollate: LANCZOS resize to model height, then right-pad
    # to the next imgH multiple by repeating the final edge column. Feeding an
    # unpadded width changes the CTC time grid and produced spurious glyphs.
    img = np.asarray(Image.fromarray(gray).resize((nw, imgH), Image.Resampling.LANCZOS)).copy()
    padded_w = int(math.ceil(nw / imgH) * imgH)
    if padded_w > nw:
        img = cv2.copyMakeBorder(img, 0, 0, 0, padded_w - nw, cv2.BORDER_REPLICATE)
    t = torch.from_numpy(img).float().div(255.0).sub(0.5).div(0.5)
    return t[None, None]


def _posteriors(gray: np.ndarray, detector: bool = True) -> np.ndarray:
    """Return (T, 37) array: column 0 = blank, 1..36 = CHARS probabilities."""
    reader = engines.ocr_reader(detector=detector)
    x = _prep(gray).to(engines.DEVICE)
    with torch.no_grad():
        text = torch.zeros(1, int(x.shape[-1] / 10) + 1, dtype=torch.long, device=engines.DEVICE)
        preds = reader.recognizer(x, text)
        prob = F.softmax(preds, dim=2)[0].cpu().numpy()  # (T, C)
    return _project_alphanumeric_posteriors(prob, reader)


def _project_alphanumeric_posteriors(prob: np.ndarray, reader) -> np.ndarray:
    """Map EasyOCR classes to plate symbols without turning punctuation into glyphs.

    Previously non-alphanumeric probability was dropped and the remaining mass
    renormalized to 1. That forced a punctuation/space peak to become the most
    likely letter or digit, systematically inventing plate characters. Treat
    unsupported symbols as blank evidence instead; keep the original total mass.
    """
    groups = _char_map(reader)
    out = np.zeros((prob.shape[0], 37), np.float32)
    mapped_indices = {0}
    out[:, 0] = prob[:, 0]
    for k, idxs in enumerate(groups):
        if idxs:
            out[:, k + 1] = prob[:, idxs].sum(1)
            mapped_indices.update(idxs)
    unsupported = [i for i in range(prob.shape[1]) if i not in mapped_indices]
    if unsupported:
        out[:, 0] += prob[:, unsupported].sum(1)
    return out


def _decode(post: np.ndarray, width: int):
    """Greedy CTC decode keeping a distribution per emitted character."""
    am = post.argmax(1)
    T = len(am)
    chars, t = [], 0
    while t < T:
        k = am[t]
        if k == 0:
            t += 1
            continue
        s = t
        while t < T and am[t] == k:
            t += 1
        seg = post[s:t, 1:]
        dist = seg.sum(0)
        dist = dist / dist.sum()
        peak = float(post[s:t, k].max())
        chars.append({"char": CHARS[k - 1], "conf": float(dist.max()) * min(1.0, peak / 0.5),
                      "dist": dist, "x": (s + t) / 2 / T * width})
    return chars


def _variants(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    if gray.shape[0] < 48:
        f = 48 / gray.shape[0]
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    yield "gray", gray
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(gray)
    yield "clahe", clahe
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.2)
    sharp = cv2.addWeighted(clahe, 1.7, blur, -0.7, 0)
    yield "sharpened", sharp


def _split_lines(gray: np.ndarray):
    """Split a two-line (square) plate at the darkest-ink valley near the middle."""
    h = gray.shape[0]
    inv = 255 - cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    inv -= np.median(inv)
    prof = np.clip(inv, 0, None).sum(1)
    lo, hi = int(h * 0.35), int(h * 0.65)
    cut = lo + int(np.argmin(prof[lo:hi])) if hi > lo else h // 2
    return [gray[:cut], gray[cut:]]


def _clean_edges(part: np.ndarray) -> np.ndarray:
    """Suppress dark plate-holder borders at the outer edges and pad with background tone."""
    h, w = part.shape[:2]
    bg = int(np.percentile(part, 75))
    thr = int(np.percentile(part, 35))
    out = part.copy()
    # Strip horizontal border lines in the top/bottom 12%
    band_y = max(2, int(0.12 * h))
    for y in list(range(band_y)) + list(range(h - band_y, h)):
        if (out[y] < thr).mean() > 0.55:
            out[y] = bg
    # Strip vertical frame lines in the left/right 5%
    band_x = max(2, int(0.05 * w))
    for x in list(range(band_x)) + list(range(w - band_x, w)):
        if (out[:, x] < thr).mean() > 0.60:
            out[:, x] = bg
    return cv2.copyMakeBorder(out, 4, 4, 8, 8, cv2.BORDER_CONSTANT, value=bg)


def read_plate(bgr: np.ndarray, two_line: bool | None = None, line_strips: list[np.ndarray] | None = None,
               detector: bool = False) -> dict:
    """Read one plate crop (or pre-rectified line strips).

    Returns {'chars': [...], 'text', 'mean_conf', 'variant', 'two_line'}.
    """
    if line_strips:
        strips = [s for s in line_strips if s.shape[0] >= 5 and s.shape[1] >= 10]
        if strips:
            # A detector supplies the outer plate quadrilateral, so stacked
            # commercial plates arrive as one rectified strip. Split those using
            # their near-square aspect ratio instead of treating both rows as one
            # sequence (which previously collapsed clips 04 and 06 to one glyph).
            if len(strips) == 1 and strips[0].shape[1] / max(strips[0].shape[0], 1) < 1.6:
                strips = _split_lines(cv2.cvtColor(strips[0], cv2.COLOR_BGR2GRAY)
                                      if strips[0].ndim == 3 else strips[0])
            two_line = len(strips) > 1
            best = None
            for vname in ("gray", "clahe", "sharpened"):
                chars = []
                for li, st in enumerate(strips):
                    g = dict(_variants(st))[vname]
                    pad = _clean_edges(g)
                    cs = _decode(_posteriors(pad, detector=detector), pad.shape[1])
                    for c in cs:
                        c["line"] = li
                    chars += _mark_gaps(cs)
                known = [c for c in chars if c["char"] != "?"]
                len_pen = 1.0 if len(known) <= 10 else max(0.5, 1.0 - 0.12 * (len(known) - 10))
                score = (np.mean([c["conf"] for c in known]) if known else 0) * min(1.0, len(known) / 6) * len_pen
                if best is None or score > best[0]:
                    best = (score, vname, chars)
            _, name, chars = best
            known = [c for c in chars if c["char"] != "?"]
            return {"chars": chars, "text": "".join(c["char"] for c in chars),
                    "mean_conf": float(np.mean([c["conf"] for c in known])) if known else 0.0,
                    "variant": name, "two_line": two_line}

    h, w = bgr.shape[:2]
    if h < 6 or w < 12:
        return {"chars": [], "text": "", "mean_conf": 0.0, "variant": None, "two_line": False}
    if two_line is None:
        two_line = (w / h) < 2.2
    best = None
    for name, g in _variants(bgr):
        parts = _split_lines(g) if two_line else [g]
        chars = []
        for li, part in enumerate(parts):
            if part.shape[0] < 6:
                continue
            pad = _clean_edges(part)
            cs = _decode(_posteriors(pad, detector=detector), pad.shape[1])
            for c in cs:
                c["line"] = li
            chars += _mark_gaps(cs)
        known = [c for c in chars if c["char"] != "?"]
        # mild preference for plausible plate lengths (6..10 chars)
        len_pen = 1.0 if len(known) <= 10 else max(0.5, 1.0 - 0.12 * (len(known) - 10))
        score = (np.mean([c["conf"] for c in known]) if known else 0.0) * min(1.0, len(known) / 6) * len_pen
        if best is None or score > best[0]:
            best = (score, name, chars)
    score, name, chars = best
    known = [c for c in chars if c["char"] != "?"]
    return {"chars": chars, "text": "".join(c["char"] for c in chars),
            "mean_conf": float(np.mean([c["conf"] for c in known])) if known else 0.0,
            "variant": name, "two_line": two_line}


def _mark_gaps(chars):
    """Insert explicit unknown ('?') placeholders where the spacing shows a character is missing
    (e.g. a mud-covered glyph that the recogniser silently skipped). Never guesses its value."""
    if len(chars) < 3:
        return chars
    xs = np.array([c["x"] for c in chars])
    d = np.diff(xs)
    pitch = float(np.median(d))
    out = [chars[0]]
    for i in range(1, len(chars)):
        gap = d[i - 1]
        missing = (int(round(gap / pitch)) - 1) if (pitch > 0 and gap > 1.85 * pitch) else 0
        for m in range(min(max(missing, 0), 4)):
            out.append({"char": "?", "conf": 0.0, "dist": None, "x": xs[i - 1] + pitch * (m + 1),
                        "line": chars[i].get("line", 0), "gap": True})
        out.append(chars[i])
    return out
