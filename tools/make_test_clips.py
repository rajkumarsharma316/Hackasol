"""Build synthetic CCTV test clips with known ground truth.

Source stills (assets/vehicles/*.jpg) have blank plates. We render a plate with a
known registration, warp it onto the blank plate, then synthesise a short CCTV
clip: camera zoom/approach, jitter, motion blur, defocus, sensor noise, low light,
colour casts, JPEG compression, and plate damage/occlusion (mud, cropped edge,
passing occluder). Ground truth is written to test_clips/ground_truth.json.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "vehicles"
OUT = ROOT / "test_clips"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

# approx blank-plate boxes in the 1376x768 stills (refined automatically)
PLATE_BOX = {
    "white_hatchback": (589, 428, 678, 452), "red_suv": (687, 458, 750, 481),
    "black_sedan": (755, 420, 878, 465), "auto_rickshaw": (850, 444, 889, 474),
    "scooter": (808, 450, 857, 495), "truck": (727, 523, 776, 559),
    "silver_suv": (752, 494, 829, 533),
}

SCENARIOS = [
    dict(name="clip01_white_hatchback_camA", img="white_hatchback", plate="MH12DE4521",
         zoom=(2.0, 3.6), blur=(0.6, 1.5), motion=(1, 6), noise=5, jpeg=(40, 75),
         note="Daylight, moderate blur, approaching vehicle"),
    dict(name="clip02_red_suv_mud", img="red_suv", plate="KA05MN7788", zoom=(2.8, 4.8),
         blur=(0.6, 1.4), motion=(1, 6), noise=5, jpeg=(40, 70), mud=(4, 5),
         note="Mud smear hides characters 5-6 (series letters MN)"),
    dict(name="clip03_black_sedan_lowlight", img="black_sedan", plate="DL3CAF0912",
         zoom=(1.8, 3.2), blur=(0.6, 1.4), motion=(1, 5), noise=10, jpeg=(35, 65), night=True,
         note="Dusk / low light, strong sensor noise"),
    dict(name="clip04_auto_rickshaw", img="auto_rickshaw", plate="TN09BX3307", two_line=True,
         yellow=True, zoom=(3.2, 5.6), blur=(0.5, 1.2), motion=(1, 5), noise=4, jpeg=(40, 75),
         note="Commercial yellow two-line plate"),
    dict(name="clip05_scooter_motionblur", img="scooter", plate="GJ01KQ5604", two_line=True,
         zoom=(3.0, 5.2), blur=(0.6, 1.3), motion=(2, 8), noise=5, jpeg=(38, 70), occluder=True,
         note="Fast two-wheeler, motion blur, passing occluder in middle frames"),
    dict(name="clip06_truck", img="truck", plate="HR55AJ8016", two_line=True, yellow=True,
         zoom=(2.8, 5.0), blur=(0.6, 1.3), motion=(1, 5), noise=5, jpeg=(38, 70), dirt=True,
         note="Goods truck, dirty yellow two-line plate"),
    dict(name="clip07_silver_suv_unregistered", img="silver_suv", plate="UP32ZZ9087",
         zoom=(2.2, 4.0), blur=(0.6, 1.3), motion=(1, 5), noise=5, jpeg=(40, 75),
         note="Vehicle NOT in registry - expected 'no confident match'"),
    dict(name="clip08_white_hatchback_camB", img="white_hatchback", plate="MH12DE4521",
         zoom=(2.0, 3.5), blur=(0.8, 1.6), motion=(1, 7), noise=8, jpeg=(30, 55), cast=(1.0, 1.0, 1.22),
         hide_end=2, flip_pan=True, out=(800, 450),
         note="Second camera: lower resolution, blue cast, last 2 characters cut off by a tow-hook/bracket"),
]


def font(size):
    for name, var in (("bahnschrift.ttf", b"Bold SemiCondensed"), ("arialbd.ttf", None)):
        try:
            f = ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
            if var:
                try:
                    f.set_variation_by_name(var)
                except Exception:
                    pass
            return f
        except OSError:
            continue
    return ImageFont.load_default()


def render_plate(text, two_line=False, yellow=False):
    """Returns RGB plate image and list of per-character x-ranges (for occlusion)."""
    bg = (245, 200, 20) if yellow else (246, 246, 242)
    if two_line:
        W, H = 340, 240
        lines = [text[:4], text[4:]]
        fsize = 86
    else:
        W, H = 640, 140
        lines = [text]
        fsize = 92
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, W - 3, H - 3], outline=(70, 70, 70), width=2)
    f = font(fsize)
    boxes = []
    line_h = H / len(lines)
    for li, line in enumerate(lines):
        widths = [d.textlength(c, font=f) for c in line]
        gap = 8
        total = sum(widths) + gap * (len(line) - 1)
        x = (W - total) / 2
        tb = d.textbbox((0, 0), line, font=f)
        gh = tb[3] - tb[1]
        y = li * line_h + (line_h - gh) / 2 - tb[1]
        for c, w in zip(line, widths):
            d.text((x, y), c, font=f, fill=(18, 18, 18))
            boxes.append((x, li * line_h + 6, x + w, (li + 1) * line_h - 6))
            x += w + gap
    return np.array(img), boxes


def refine_quad(img, box, yellow):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1, Y1 = max(0, int(x1 - 0.15 * w)), max(0, int(y1 - 0.20 * h))
    X2, Y2 = min(img.shape[1], int(x2 + 0.15 * w)), min(img.shape[0], int(y2 + 0.20 * h))
    roi = img[Y1:Y2, X1:X2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if yellow:
        mask = cv2.inRange(hsv, (15, 90, 120), (40, 255, 255))
    else:
        mask = cv2.inRange(hsv, (0, 0, 175), (180, 60, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 0.35 * w * h]
    if not cnts:
        return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    # pick contour whose bounding box overlaps the nominal plate box best
    target = (x1 - X1, y1 - Y1, x2 - X1, y2 - Y1)

    def _ov(c):
        bx, by, bw, bh = cv2.boundingRect(c)
        ix = max(0, min(bx + bw, target[2]) - max(bx, target[0]))
        iy = max(0, min(by + bh, target[3]) - max(by, target[1]))
        return ix * iy

    c = max(cnts, key=_ov)
    hull = cv2.convexHull(c).reshape(-1, 2).astype(np.float32)
    s, dfn = hull.sum(1), np.diff(hull, axis=1).ravel()
    quad = np.float32([hull[np.argmin(s)], hull[np.argmin(dfn)], hull[np.argmax(s)], hull[np.argmax(dfn)]])
    return quad + np.float32([X1, Y1])


def paste_plate(img, quad, plate_rgb):
    h, w = plate_rgb.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2BGR), M, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (img.shape[1], img.shape[0]))
    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)[..., None] / 255
    return (img * (1 - mask) + warped * mask).astype(np.uint8)


def motion_kernel(length, angle):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
    k = cv2.warpAffine(k, M, (length, length))
    return k / max(k.sum(), 1e-6)


def build(sc):
    img = cv2.imread(str(SRC / f"{sc['img']}.jpg"))
    S = 2  # work at 2x so the pasted plate has real detail before degradation
    img = cv2.resize(img, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    quad = refine_quad(img, [v * S for v in PLATE_BOX[sc["img"]]], sc.get("yellow", False))
    plate, boxes = render_plate(sc["plate"], sc.get("two_line", False), sc.get("yellow", False))
    plate = plate.copy()
    if "mud" in sc:  # brown smear over characters a..b (0-based, inclusive)
        a, b = sc["mud"]
        xa, ya = boxes[a][0], boxes[a][1]
        xb, yb = boxes[b][2], boxes[b][3]
        ov = plate.copy()
        cv2.ellipse(ov, (int((xa + xb) / 2), int((ya + yb) / 2)), (int((xb - xa) / 2 + 10), int((yb - ya) / 2)),
                    -8, 0, 360, (85, 62, 40), -1)
        for _ in range(25):
            cx = int(rng.uniform(xa - 15, xb + 15)); cy = int(rng.uniform(ya, yb))
            cv2.circle(ov, (cx, cy), int(rng.uniform(4, 14)), (95, 70, 45), -1)
        plate = cv2.addWeighted(ov, 0.93, plate, 0.07, 0)
    if sc.get("dirt"):
        noise = rng.normal(0, 1, plate.shape[:2])
        noise = cv2.GaussianBlur(noise, (0, 0), 9)
        dirt = (noise > 0.15)[..., None]
        plate = np.where(dirt, (plate * 0.62 + np.array([60, 45, 25]) * 0.38).astype(np.uint8), plate)
    img = paste_plate(img, quad, plate)
    if sc.get("hide_end"):  # bracket covering the last n characters
        n = sc["hide_end"]
        xa = boxes[-n][0] - 4
        W = plate.shape[1]
        t0, t1 = xa / W, 1.04
        tl, tr, br, bl = quad
        p = lambda t, a, b: a + (b - a) * t
        poly = np.int32([p(t0, tl, tr) - [0, 6], p(t1, tl, tr) - [0, 6], p(t1, bl, br) + [0, 6], p(t0, bl, br) + [0, 6]])
        cv2.fillPoly(img, [poly], (38, 38, 42))

    H, W = img.shape[:2]
    out_w, out_h = sc.get("out", (960, 540))
    pc = quad.mean(0)
    n = 40
    frames = []
    for t in range(n):
        u = t / (n - 1)
        z = sc["zoom"][0] + (sc["zoom"][1] - sc["zoom"][0]) * (u ** 1.3)
        cw, ch = W / z, H / z
        cx = W / 2 + (pc[0] - W / 2) * min(1, u * 1.2 + 0.3) * (0.9 if not sc.get("flip_pan") else 1.0)
        cy = H / 2 + (pc[1] - H / 2) * min(1, u * 1.2 + 0.3)
        cx += rng.normal(0, 4); cy += rng.normal(0, 3)
        x0 = int(np.clip(cx - cw / 2, 0, W - cw)); y0 = int(np.clip(cy - ch / 2, 0, H - ch))
        fr = cv2.resize(img[y0:y0 + int(ch), x0:x0 + int(cw)], (out_w, out_h), interpolation=cv2.INTER_AREA)
        # defocus + motion blur (some frames much worse)
        sig = rng.uniform(*sc["blur"]) * (1.8 if rng.random() < 0.2 else 1.0)
        fr = cv2.GaussianBlur(fr, (0, 0), sig)
        L = int(rng.integers(sc["motion"][0], sc["motion"][1] + 1))
        if L > 1:
            fr = cv2.filter2D(fr, -1, motion_kernel(L, rng.uniform(-20, 20)))
        f = fr.astype(np.float32)
        if sc.get("night"):
            f = 255 * (f / 255) ** 1.8 * 0.55
            f[..., 0] *= 1.1
        if sc.get("cast"):
            f *= np.array(sc["cast"], np.float32)
        f += rng.normal(0, sc["noise"], f.shape)
        fr = np.clip(f, 0, 255).astype(np.uint8)
        if sc.get("occluder") and 14 <= t <= 22:  # pedestrian / pole passing in front
            px = int(out_w * (0.25 + 0.06 * (t - 14)))
            cv2.rectangle(fr, (px, 0), (px + int(out_w * 0.09), out_h), (40, 45, 50), -1)
        q = int(rng.integers(*sc["jpeg"]))
        fr = cv2.imdecode(cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
        frames.append(fr)

    path = OUT / f"{sc['name']}.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (out_w, out_h))
    for fr in frames:
        vw.write(fr)
    vw.release()
    cv2.imwrite(str(OUT / f"{sc['name']}_preview.jpg"), frames[-1])
    return path


if __name__ == "__main__":
    gt = {}
    for sc in SCENARIOS:
        p = build(sc)
        gt[sc["name"]] = {"plate": sc["plate"], "source_image": sc["img"], "note": sc["note"],
                          "in_registry": sc["plate"] != "UP32ZZ9087"}
        print("built", p.name)
    (OUT / "ground_truth.json").write_text(json.dumps(gt, indent=2))
