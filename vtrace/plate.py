"""Number-plate localisation inside a vehicle crop using oriented 4-point quads.

Two independent proposal sources are combined:
  1. CRAFT text detector (from EasyOCR, both horizontal_list and rotated free_list)
     - finds character stroke lines at any tilt angle.
  2. Plate-background colour/shape heuristic - white or yellow rectangles
     with dark character strokes inside.
Collinear horizontally-split boxes (e.g. a plate split into two halves by mud)
are merged into a single spanning quad; vertically-stacked boxes on two-line
plates are paired into [top_quad, bottom_quad].
"""
from __future__ import annotations

import cv2
import numpy as np
from easyocr.utils import four_point_transform

from . import engines


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    idx_y = np.argsort(pts[:, 1])
    top = pts[idx_y[:2]]
    bot = pts[idx_y[2:]]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bot[np.argsort(bot[:, 0])]
    return np.float32([tl, tr, br, bl])


def box_to_quad(box) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def quads_bbox(quads: list[np.ndarray]) -> tuple[int, int, int, int]:
    pts = np.vstack(quads)
    return (
        int(np.floor(pts[:, 0].min())),
        int(np.floor(pts[:, 1].min())),
        int(np.ceil(pts[:, 0].max())),
        int(np.ceil(pts[:, 1].max())),
    )


def pad_quad(q: np.ndarray, shape, px=0.05, py=0.10) -> np.ndarray:
    tl, tr, br, bl = q.astype(np.float32)
    u = (tr - tl + br - bl) * 0.5
    v = (bl - tl + br - tr) * 0.5
    H, W = shape[:2]
    out = np.float32([
        tl - px * u - py * v,
        tr + px * u - py * v,
        br + px * u + py * v,
        bl - px * u + py * v,
    ])
    out[:, 0] = np.clip(out[:, 0], 0, W - 1)
    out[:, 1] = np.clip(out[:, 1], 0, H - 1)
    return out


def warp_quads(img: np.ndarray, quads: list[np.ndarray]) -> list[np.ndarray]:
    strips = []
    for q in quads:
        pq = pad_quad(q, img.shape)
        w = four_point_transform(img, pq)
        if w.size > 0 and w.shape[0] >= 5 and w.shape[1] >= 10:
            strips.append(w)
    return strips


def _plate_colour_fraction(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 120), (180, 75, 255))
    yellow = cv2.inRange(hsv, (12, 70, 100), (42, 255, 255))
    # Also accept high local contrast grey plates in dusk/night scenes
    g = hsv[:, :, 2]
    bright_bg = (np.percentile(g, 80) - np.percentile(g, 20)) > 45
    base = float(((white > 0) | (yellow > 0)).mean())
    return max(base, 0.35 if bright_bg else 0.0)


def _stroke_score(crop: np.ndarray) -> float:
    """Measure horizontal distribution of vertical character strokes inside the strip."""
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 10:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    mid = g[int(h * 0.15):max(int(h * 0.85), int(h * 0.15) + 2), int(w * 0.06):max(int(w * 0.94), int(w * 0.06) + 4)]
    if mid.size == 0:
        return 0.0
    gx = np.abs(cv2.Sobel(mid, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(mid, cv2.CV_32F, 0, 1, ksize=3))
    # Characters have strong vertical strokes spread across multiple columns
    col_energy = gx.mean(0)
    active_cols = float((col_energy > max(18.0, col_energy.mean() * 0.45)).mean())
    vert_dom = float(gx.mean() / (gy.mean() + 10.0))
    return min(1.0, active_cols / 0.35) * min(1.0, vert_dom / 0.85)


def _craft_quads(crop: np.ndarray) -> list[np.ndarray]:
    reader = engines.ocr_reader()
    h, w = crop.shape[:2]
    mag = 2.0 if max(h, w) < 600 else 1.5
    try:
        hl, fl = reader.detect(
            crop, min_size=6, text_threshold=0.38, low_text=0.25, link_threshold=0.25,
            canvas_size=1280, mag_ratio=mag, width_ths=0.7, add_margin=0.06,
        )
    except Exception:
        return []
    quads = []
    for q in (fl[0] if fl else []):
        oq = order_quad(np.array(q, dtype=np.float32))
        bb = quads_bbox([oq])
        if bb[2] - bb[0] > 6 and bb[3] - bb[1] > 4:
            quads.append(oq)
    for b in (hl[0] if hl else []):
        x1, x2, y1, y2 = [int(v) for v in b]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 <= 6 or y2 - y1 <= 4:
            continue
        hq = box_to_quad((x1, y1, x2, y2))
        if all(iou(quads_bbox([hq]), quads_bbox([fq])) < 0.55 for fq in quads):
            quads.append(hq)
    return quads


def _q_metrics(q: np.ndarray):
    tl, tr, br, bl = q
    w = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    h = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    c = q.mean(0)
    ang = float(np.degrees(np.arctan2(tr[1] - tl[1], max(1e-3, tr[0] - tl[0]))))
    return float(w), float(h), float(c[0]), float(c[1]), ang


def _group_text_quads(quads: list[np.ndarray]) -> list[list[np.ndarray]]:
    """Produce candidate 1-line and 2-line plate quad groups."""
    if not quads:
        return []
    # 1. Horizontally collinear merge (e.g. plate split by mud in the middle)
    qs = sorted(quads, key=lambda q: q.mean(0)[0])
    merged_1 = list(qs)
    n = len(qs)
    for i in range(n):
        w1, h1, cx1, cy1, a1 = _q_metrics(qs[i])
        for j in range(i + 1, n):
            w2, h2, cx2, cy2, a2 = _q_metrics(qs[j])
            if not (0.5 <= h1 / max(h2, 1.0) <= 2.0) or abs(a1 - a2) > 15:
                continue
            # Project vertical offset perpendicular to line direction
            dx = cx2 - cx1
            dy = cy2 - cy1
            expected_dy = dx * np.tan(np.radians(0.5 * (a1 + a2)))
            perp_err = abs(dy - expected_dy)
            gap = qs[j][:, 0].min() - qs[i][:, 0].max()
            if perp_err < 0.65 * max(h1, h2) and -0.25 * min(w1, w2) <= gap <= 2.6 * max(h1, h2):
                span = np.float32([qs[i][0], qs[j][1], qs[j][2], qs[i][3]])
                merged_1.append(span)

    groups = [[q] for q in merged_1]
    # 2. Vertically stacked two-line plates
    by_y = sorted(merged_1, key=lambda q: q.mean(0)[1])
    for i in range(len(by_y)):
        w1, h1, cx1, cy1, a1 = _q_metrics(by_y[i])
        for j in range(i + 1, len(by_y)):
            w2, h2, cx2, cy2, a2 = _q_metrics(by_y[j])
            avg_h = 0.5 * (h1 + h2)
            if not (0.45 <= h1 / max(h2, 1.0) <= 2.2) or abs(a1 - a2) > 16:
                continue
            dy = cy2 - cy1
            b1, b2 = quads_bbox([by_y[i]]), quads_bbox([by_y[j]])
            x_ov = min(b1[2], b2[2]) - max(b1[0], b2[0])
            if 0.45 * avg_h <= dy <= 2.2 * avg_h and x_ov > 0.35 * min(b1[2] - b1[0], b2[2] - b2[0]):
                groups.append([by_y[i], by_y[j]])
    return groups


def _colour_quads(crop: np.ndarray) -> list[list[np.ndarray]]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = crop.shape[:2]
    out = []
    for lo, hi in (((0, 0, 145), (180, 62, 255)), ((12, 80, 110), (42, 255, 255))):
        m = cv2.inRange(hsv, lo, hi)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 16 or bh < 8 or bw > 0.55 * w or bh > 0.38 * h:
                continue
            rect_area = bw * bh
            if cv2.contourArea(c) / rect_area < 0.52:
                continue
            rbox = cv2.boxPoints(cv2.minAreaRect(c))
            oq = order_quad(rbox)
            out.append([oq])
    return out


def score_group(crop: np.ndarray, quads: list[np.ndarray], from_text: bool) -> float:
    h, w = crop.shape[:2]
    x1, y1, x2, y2 = quads_bbox(quads)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 6 or bh <= 4:
        return 0.0
    rw, rh = bw / w, bh / h
    if rw > 0.58 or rh > 0.40 or rw < 0.04:
        return 0.0
    a = bw / bh
    if len(quads) == 2:
        s_aspect = 1.15 if 0.9 <= a <= 3.0 else 0.2
    else:
        s_aspect = 1.0 if 1.1 <= a <= 6.5 else 0.2
    cy, cx = (y1 + y2) / 2 / h, (x1 + x2) / 2 / w
    s_pos = (1.0 if cy >= 0.32 else 0.25) * (1.0 if 0.12 <= cx <= 0.88 else 0.45)
    s_size = 1.0 if 0.06 <= rw <= 0.48 else 0.4
    strips = warp_quads(crop, quads)
    if not strips:
        return 0.0
    pc = float(np.mean([_plate_colour_fraction(s) for s in strips]))
    st = float(np.mean([_stroke_score(s) for s in strips]))
    if not from_text and st < 0.35:
        return 0.0
    s_col = 0.25 + 0.75 * min(1.0, pc / 0.45)
    s_str = 0.2 + 0.8 * st
    return s_aspect * s_pos * s_size * s_col * s_str * (1.45 if from_text else 0.85)


def propose(crop: np.ndarray) -> list[dict]:
    """Return candidate plate dicts sorted by score: [{'score', 'box', 'quads', 'from_text'}]."""
    t_groups = _group_text_quads(_craft_quads(crop))
    c_groups = _colour_quads(crop)
    cands = []
    for qs in t_groups:
        sc = score_group(crop, qs, True)
        if sc > 0.10:
            cands.append({"score": sc, "box": quads_bbox(qs), "quads": qs, "from_text": True})
    for qs in c_groups:
        sc = score_group(crop, qs, False)
        if sc > 0.12:
            cands.append({"score": sc, "box": quads_bbox(qs), "quads": qs, "from_text": False})
    cands.sort(key=lambda c: -c["score"])
    # Deduplicate heavily overlapping proposals
    dedup = []
    for c in cands:
        if all(iou(c["box"], d["box"]) < 0.75 or len(c["quads"]) != len(d["quads"]) for d in dedup):
            dedup.append(c)
        if len(dedup) >= 6:
            break
    return dedup


def shift_quads(quads: list[np.ndarray], dx: float, dy: float) -> list[np.ndarray]:
    off = np.float32([dx, dy])
    return [q + off for q in quads]


def rel_box(box, vbox):
    vx1, vy1, vx2, vy2 = vbox
    vw, vh = max(1.0, vx2 - vx1), max(1.0, vy2 - vy1)
    x1, y1, x2, y2 = box
    return ((x1 - vx1) / vw, (y1 - vy1) / vh, (x2 - vx1) / vw, (y2 - vy1) / vh)


def abs_box(rb, vbox):
    vx1, vy1, vx2, vy2 = vbox
    vw, vh = vx2 - vx1, vy2 - vy1
    return (vx1 + rb[0] * vw, vy1 + rb[1] * vh, vx1 + rb[2] * vw, vy1 + rb[3] * vh)


def rel_quads(quads: list[np.ndarray], vbox) -> list[np.ndarray]:
    vx1, vy1, vx2, vy2 = vbox
    vw, vh = max(1.0, vx2 - vx1), max(1.0, vy2 - vy1)
    out = []
    for q in quads:
        rq = q.copy().astype(np.float32)
        rq[:, 0] = (rq[:, 0] - vx1) / vw
        rq[:, 1] = (rq[:, 1] - vy1) / vh
        out.append(rq)
    return out


def abs_quads(rquads: list[np.ndarray], vbox) -> list[np.ndarray]:
    vx1, vy1, vx2, vy2 = vbox
    vw, vh = vx2 - vx1, vy2 - vy1
    out = []
    for rq in rquads:
        q = rq.copy().astype(np.float32)
        q[:, 0] = vx1 + q[:, 0] * vw
        q[:, 1] = vy1 + q[:, 1] * vh
        out.append(q)
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def pad_box(box, shape, px=0.06, py=0.12):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = shape[:2]
    return (int(max(0, x1 - px * w)), int(max(0, y1 - py * h)), int(min(W, x2 + px * w)), int(min(H, y2 + py * h)))
