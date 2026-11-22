"""Vehicle attribute extraction: colour, body type, make/model, appearance embedding."""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from PIL import Image

from . import engines
from .vocab import BODY_TYPES, COLOURS

# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def _name_colour(lab):
    L = lab[0] * 100 / 255
    a, b = lab[1] - 128.0, lab[2] - 128.0
    C = math.hypot(a, b)
    if C < 13 or (L > 72 and C < 18) or (52 < L <= 72 and C < 15):
        return "white" if L > 74 else "silver" if L > 52 else "grey" if L > 28 else "black"
    h = (math.degrees(math.atan2(b, a)) + 360) % 360
    if h < 38 or h >= 330:
        return "maroon" if L < 32 else "red"
    if h < 75:
        if C >= 20 and L >= 34:
            return "orange"
        return "brown" if L < 48 else "beige"
    if h < 108:
        if L < 42:
            return "brown"
        return "yellow" if C > 24 else "beige"
    if h < 200:
        return "green"
    if h < 305:
        return "blue" if C > 15 else ("silver" if L > 55 else "grey")
    return "maroon" if L < 32 else "red"


def colour_distribution(frame, vbox, k=5):
    """Colour-name distribution for the painted vehicle body in one frame."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in vbox]
    bw, bh = x2 - x1, y2 - y1
    # Focus on body panels (lower-middle 35%..86% avoids roof/windows/rider torso and bottom road)
    crop = frame[y1 + int(0.32 * bh):y1 + int(0.86 * bh), x1 + int(0.14 * bw):x1 + int(0.86 * bw)]
    if crop.size < 300:
        return {}
    crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    # background reference: strips left/right/below the vehicle
    strips = []
    for sx1, sy1, sx2, sy2 in ((x1 - bw // 5, y1, x1, y2), (x2, y1, x2 + bw // 5, y2), (x1, y2, x2, y2 + bh // 6)):
        sx1, sy1, sx2, sy2 = max(0, sx1), max(0, sy1), min(W, sx2), min(H, sy2)
        if sx2 - sx1 > 3 and sy2 - sy1 > 3:
            strips.append(cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2LAB).reshape(-1, 3))
    bg = np.median(np.concatenate(strips), 0).astype(np.float32) if strips else None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k).astype(float)
    total_px = counts.sum() + 1e-6
    dark_frac = sum(n for c, n in zip(centers, counts) if (c[0] * 100 / 255) < 25 and math.hypot(c[1] - 128, c[2] - 128) < 12) / total_px
    dist = {}
    for c, n in zip(centers, counts):
        w = n
        if bg is not None and np.linalg.norm(c - bg) < 16:
            w *= 0.25  # looks like road / background
        L100 = c[0] * 100 / 255
        chroma = math.hypot(c[1] - 128, c[2] - 128)
        # Tires, window glass and shadows are dark achromatic; downweight unless the whole car is dark
        if L100 < 24 and chroma < 11 and dark_frac < 0.52:
            w *= 0.30
        if chroma > 18 and (n / total_px) > 0.12:
            w *= 1.35  # painted body panels (ignore tiny tail-light clusters < 12% area)
        name = _name_colour(c)
        dist[name] = dist.get(name, 0) + w
    s = sum(dist.values())
    return {k_: v / s for k_, v in dist.items()} if s else {}


def fuse_colour_with_clip(lab_dist: dict, img_feats) -> dict:
    """Combine LAB pixel clustering with CLIP semantic vehicle colour classification."""
    if img_feats is None or not lab_dist:
        return lab_dist
    prompts = [f"a traffic camera photo of a {c} colored vehicle" for c in COLOURS]
    tf = _text_feats(prompts)
    pr = torch.softmax(100.0 * img_feats @ tf.T, dim=-1).mean(0).cpu().numpy()
    clip_c = {c: float(pr[i]) for i, c in enumerate(COLOURS)}
    fused = {}
    for c in COLOURS:
        p_lab = lab_dist.get(c, 0.0)
        p_clip = clip_c.get(c, 0.0)
        v = 0.55 * p_lab + 0.45 * p_clip
        if v > 0.015:
            fused[c] = v
    s = sum(fused.values()) or 1.0
    return {k: round(v / s, 4) for k, v in sorted(fused.items(), key=lambda x: -x[1])}


def merge_dists(dists, weights=None):
    out = {}
    weights = weights or [1.0] * len(dists)
    for d, w in zip(dists, weights):
        for k, v in d.items():
            out[k] = out.get(k, 0) + v * w
    s = sum(out.values())
    return {k: round(v / s, 4) for k, v in sorted(out.items(), key=lambda x: -x[1])} if s else {}


# ---------------------------------------------------------------------------
# Body type / make-model / embedding (CLIP zero-shot)
# ---------------------------------------------------------------------------
YOLO_TO_BODY = {
    "car": {"car": 0.45, "suv": 0.35, "van": 0.10, "auto-rickshaw": 0.10},
    "motorcycle": {"two-wheeler": 0.85, "auto-rickshaw": 0.15},
    "bicycle": {"two-wheeler": 1.0},
    "bus": {"bus": 0.75, "truck": 0.15, "van": 0.10},
    "truck": {"truck": 0.45, "auto-rickshaw": 0.25, "suv": 0.18, "van": 0.12},
}
BODY_PROMPTS = {
    "car": ["a hatchback car", "a sedan car"],
    "suv": ["an SUV", "a sport utility vehicle"],
    "van": ["a van", "a minivan", "a multi-purpose vehicle"],
    "two-wheeler": ["a motorcycle", "a scooter", "a person riding a scooter"],
    "auto-rickshaw": ["an auto rickshaw", "a three-wheeler tuk-tuk"],
    "truck": ["a goods truck", "a lorry"],
    "bus": ["a bus"],
}
_text_cache = {}


def _text_feats(prompts):
    key = tuple(prompts)
    if key not in _text_cache:
        m, p = engines.clip()
        with torch.no_grad():
            t = p(text=list(prompts), return_tensors="pt", padding=True).to(engines.DEVICE)
            f = m.get_text_features(**t)
            if not isinstance(f, torch.Tensor):
                f = f.pooler_output if hasattr(f, "pooler_output") else f[0]
            _text_cache[key] = torch.nn.functional.normalize(f, dim=-1)
    return _text_cache[key]


def clip_image_feats(crops):
    c = engines.clip()
    if c is None or not crops:
        return None
    m, p = c
    ims = [Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)) for x in crops]
    with torch.no_grad():
        t = p(images=ims, return_tensors="pt").to(engines.DEVICE)
        f = m.get_image_features(**t)
        if not isinstance(f, torch.Tensor):
            f = f.pooler_output if hasattr(f, "pooler_output") else f[0]
        return torch.nn.functional.normalize(f, dim=-1)


def body_type(yolo_classes, img_feats):
    """Product-of-experts: YOLO class votes (prior) x CLIP zero-shot."""
    prior = {b: 0.02 for b in BODY_TYPES}
    for cls, w in yolo_classes.items():
        for b, p in YOLO_TO_BODY.get(cls, {}).items():
            prior[b] += p * w
    s = sum(prior.values())
    prior = {k: v / s for k, v in prior.items()}
    if img_feats is None:
        return prior, None
    prompts, owners = [], []
    for b, ps in BODY_PROMPTS.items():
        for q in ps:
            prompts.append(f"a CCTV photo of {q} on an Indian road")
            owners.append(b)
    tf = _text_feats(prompts)
    logits = 100.0 * img_feats @ tf.T
    pr = torch.softmax(logits, dim=-1).mean(0).cpu().numpy()
    clip_d = {b: 0.0 for b in BODY_TYPES}
    for o, v in zip(owners, pr):
        clip_d[o] += float(v)
    fused = {b: (prior[b] ** 0.35) * (clip_d[b] + 0.005) for b in BODY_TYPES}
    s = sum(fused.values())
    return ({k: round(v / s, 4) for k, v in sorted(fused.items(), key=lambda x: -x[1])},
            {k: round(v, 4) for k, v in sorted(clip_d.items(), key=lambda x: -x[1])})


def make_model(img_feats, registry, body_hint=None, top=5):
    if img_feats is None:
        return None
    models = sorted({(r["make"], r["model"], r["body_type"]) for r in registry})
    if body_hint:
        keep = [m for m in models if m[2] == body_hint]
        models = keep or models
    words = {"car": "car", "suv": "SUV", "van": "van", "two-wheeler": "scooter or motorcycle",
             "auto-rickshaw": "auto rickshaw", "truck": "truck", "bus": "bus"}
    prompts = [f"a photo of a {mk} {md} {words.get(bt, '')}" for mk, md, bt in models]
    tf = _text_feats(prompts)
    pr = torch.softmax(100.0 * img_feats @ tf.T, dim=-1).mean(0).cpu().numpy()
    order = np.argsort(-pr)[:top]
    return {f"{models[i][0]} {models[i][1]}": round(float(pr[i]), 4) for i in order}
