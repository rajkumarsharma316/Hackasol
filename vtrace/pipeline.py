"""End-to-end video analysis pipeline."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np

from . import attributes, confidence, engines, fusion, matcher, ocr, plate
from .vocab import BODY_TYPES, COLOURS

MAX_FRAMES = 60
MAX_WIDTH = 1280
VEHICLE_CLASSES = [1, 2, 3, 5, 7]   # COCO: bicycle, car, motorcycle, bus, truck
MAX_FUSION_FRAMES = 15


def _np(o):
    if isinstance(o, dict):
        return {str(k): _np(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_np(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def _sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if g.shape[0] < 4:
        return 0.0
    g = cv2.resize(g, (int(48 * g.shape[1] / g.shape[0]) or 1, 48))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _predict_box(track, frame_idx):
    """Constant-velocity box extrapolation across sampled-frame gaps."""
    history = track["boxes"]
    fi, last, _ = history[-1]
    gap = max(1, frame_idx - fi)
    pred = np.asarray(last, dtype=np.float32)
    if len(history) >= 2:
        pfi, prev, _ = history[-2]
        span = max(1, fi - pfi)
        pred += (pred - np.asarray(prev, dtype=np.float32)) * (gap / span)
    last_w, last_h = max(2.0, history[-1][1][2] - history[-1][1][0]), max(2.0, history[-1][1][3] - history[-1][1][1])
    cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
    pw = float(np.clip(pred[2] - pred[0], 0.5 * last_w, 2.0 * last_w))
    ph = float(np.clip(pred[3] - pred[1], 0.5 * last_h, 2.0 * last_h))
    pred = np.asarray([cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2], dtype=np.float32)
    return pred.tolist()


def _center_distance(a, b):
    ac = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2], dtype=float)
    bc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], dtype=float)
    diag = max(1.0, float(np.hypot(a[2] - a[0], a[3] - a[1])))
    return float(np.linalg.norm(ac - bc) / diag)


class VideoAnalysis:
    def __init__(self, video_path, job_dir, registry, progress=None):
        self.video_path = str(video_path)
        self.dir = Path(job_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "frames").mkdir(exist_ok=True)
        (self.dir / "plates").mkdir(exist_ok=True)
        (self.dir / "crops").mkdir(exist_ok=True)
        self.registry = registry
        self.progress = progress or (lambda p, msg: None)
        self.frames, self.frame_ids, self.times = [], [], []
        self.tracks = None
        self.meta = {}

    # ------------------------------------------------------------------ load
    def load(self):
        if self.tracks is not None:
            return
        self.progress(2, "Reading video")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError("Could not open the uploaded video. Check that it is a valid, supported video file.")
        reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        picked, i = [], 0
        if reported_total > 0:
            idxs = set(np.linspace(0, reported_total - 1, min(reported_total, MAX_FRAMES)).astype(int).tolist())
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if i in idxs:
                    picked.append((i, fr))
                i += 1
            total = i
        else:   # Unknown-length videos: reservoir-sample without retaining every decoded frame.
            rng = random.Random(0)
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if len(picked) < MAX_FRAMES:
                    picked.append((i, fr))
                else:
                    slot = rng.randrange(i + 1)
                    if slot < MAX_FRAMES:
                        picked[slot] = (i, fr)
                i += 1
            total = i
            picked.sort(key=lambda item: item[0])
        cap.release()
        if not picked:
            raise ValueError("Could not decode any frames from the uploaded file.")
        for i, fr in picked:
            if fr.shape[1] > MAX_WIDTH:
                s = MAX_WIDTH / fr.shape[1]
                fr = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            self.frames.append(fr)
            self.frame_ids.append(int(i))
            self.times.append(round(i / fps, 2))
        for k, fr in enumerate(self.frames):
            cv2.imwrite(str(self.dir / "frames" / f"f{k:03d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        self.meta = {"source_frames": total, "fps": round(fps, 2), "sampled": len(self.frames),
                     "resolution": f"{self.frames[0].shape[1]}x{self.frames[0].shape[0]}",
                     "duration_s": round(total / fps, 2)}
        self.progress(10, f"Detecting vehicles in {len(self.frames)} frames")
        self._detect_and_track()

    def _detect_and_track(self):
        model = engines.yolo()
        names = model.names
        dets = []
        for s in range(0, len(self.frames), 16):
            res = model.predict(self.frames[s:s + 16], classes=VEHICLE_CLASSES, conf=0.25, verbose=False,
                                device=engines.YOLO_DEVICE)
            for r in res:
                d = []
                for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    d.append({"box": [float(v) for v in b], "cls": names[int(c)], "conf": float(cf)})
                dets.append(d)
        tracks = []
        for fi, d in enumerate(dets):
            # class-agnostic NMS (a vehicle detected both as car and truck)
            d = sorted(d, key=lambda x: -x["conf"])
            keep = []
            for x in d:
                if all(plate.iou(x["box"], k["box"]) < 0.6 for k in keep):
                    keep.append(x)
            # Match globally rather than letting the first high-confidence box
            # greedily claim the best track. A constant-velocity prediction helps
            # maintain identity through sampled-frame gaps and brief occlusions.
            edges = []
            for di, x in enumerate(keep):
                for ti, t in enumerate(tracks):
                    if fi - t["last"] > 6:
                        continue
                    prev_box = t["boxes"][-1][1]
                    pred_box = _predict_box(t, fi)
                    pred_iou = plate.iou(x["box"], pred_box)
                    if pred_iou >= 0.12 or _center_distance(x["box"], pred_box) <= 0.35:
                        edge_score = 0.7 * pred_iou + 0.3 * plate.iou(x["box"], prev_box)
                        edges.append((edge_score, di, ti))
            assigned_dets, assigned_tracks = set(), set()
            for edge_score, di, ti in sorted(edges, reverse=True):
                if edge_score < 0.10 or di in assigned_dets or ti in assigned_tracks:
                    continue
                x = keep[di]
                t = tracks[ti]
                assigned_dets.add(di)
                assigned_tracks.add(ti)
                t["boxes"].append((fi, x["box"], x["conf"]))
                t["cls"][x["cls"]] = t["cls"].get(x["cls"], 0) + x["conf"]
                t["last"] = fi
                H, W = self.frames[fi].shape[:2]
                b = x["box"]
                t["score"] += x["conf"] * ((b[2] - b[0]) * (b[3] - b[1]) / (W * H)) ** 0.5
            for di, x in enumerate(keep):
                if di in assigned_dets:
                    continue
                t = {"id": len(tracks) + 1, "boxes": [(fi, x["box"], x["conf"])],
                     "cls": {x["cls"]: x["conf"]}, "last": fi, "score": 0.0}
                H, W = self.frames[fi].shape[:2]
                b = x["box"]
                t["score"] = x["conf"] * ((b[2] - b[0]) * (b[3] - b[1]) / (W * H)) ** 0.5
                tracks.append(t)
        tracks = [t for t in tracks if len(t["boxes"]) >= 2] or tracks
        if not tracks:  # nothing detected - treat the whole frame as the vehicle
            tracks = [{"id": 1, "boxes": [(fi, [0, 0, f.shape[1], f.shape[0]], 0.0) for fi, f in enumerate(self.frames)],
                       "cls": {}, "last": len(self.frames) - 1, "score": 0.0, "fallback": True}]
        tracks.sort(key=lambda t: -t["score"])
        for t in tracks:
            fi, b, _ = max(t["boxes"], key=lambda x: (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]))
            x1, y1, x2, y2 = [int(v) for v in b]
            th = self.frames[fi][max(0, y1):y2, max(0, x1):x2]
            if th.size:
                th = cv2.resize(th, (160, int(160 * th.shape[0] / max(th.shape[1], 1)) or 1))
                cv2.imwrite(str(self.dir / "crops" / f"track{t['id']}.jpg"), th)
        self.tracks = tracks

    def tracks_summary(self):
        tot = sum(t["score"] for t in self.tracks) or 1
        return [{"id": t["id"], "frames": len(t["boxes"]), "classes": {k: round(v, 2) for k, v in t["cls"].items()},
                 "mean_detection_confidence": round(float(np.mean([x[2] for x in t["boxes"]])), 3),
                 "max_sample_gap": max(0, max((b[0] - a[0] for a, b in zip(t["boxes"], t["boxes"][1:])), default=1) - 1),
                 "prominence": round(t["score"] / tot, 3), "thumb": f"crops/track{t['id']}.jpg"}
                for t in self.tracks[:8]]

    # --------------------------------------------------------------- analyse
    def analyse(self, hints=None):
        hints = hints or {}
        t0 = time.time()
        self.load()
        track = self.tracks[0]
        if hints.get("track_id"):
            track = next((t for t in self.tracks if t["id"] == int(hints["track_id"])), track)
        boxes = {fi: b for fi, b, _ in track["boxes"]}
        self.progress(25, f"Locating number plate (vehicle track #{track['id']}, {len(boxes)} frames)")

        # ---- plate localisation per frame
        roi = hints.get("roi")
        per_frame = {}
        if roi:
            rf = int(roi["frame"])
            near = min(boxes, key=lambda f: abs(f - rf))
            rel = plate.rel_box((roi["x1"], roi["y1"], roi["x2"], roi["y2"]), boxes[near])
            for fi, vb in boxes.items():
                ab = plate.abs_box(rel, vb)
                per_frame[fi] = {"box": ab, "quads": [plate.box_to_quad(ab)], "score": 1.0, "source": "manual ROI"}
        else:
            props = {}
            plate_model = engines.plate_detector()
            if plate_model is not None:
                # Run the specialized plate detector on vehicle crops in batches.
                # This avoids treating arbitrary CRAFT text (logos, lamps, trim) as
                # a plate and avoids loading CRAFT weights on the model-backed path.
                crops, crop_meta = [], []
                for fi, vb in boxes.items():
                    fr = self.frames[fi]
                    x1, y1, x2, y2 = plate.pad_box(vb, fr.shape, 0.08, 0.08)
                    crop = fr[y1:y2, x1:x2]
                    if crop.size:
                        crops.append(crop)
                        crop_meta.append((fi, vb, x1, y1, x2, y2))
                detections = []
                for start in range(0, len(crops), 16):
                    batch = plate_model.predict(
                        crops[start:start + 16], conf=0.25, imgsz=640, max_det=20,
                        verbose=False, device=engines.YOLO_DEVICE,
                    )
                    detections.extend(batch)
                for k, ((fi, vb, x1, y1, x2, y2), result) in enumerate(zip(crop_meta, detections)):
                    fr_h, fr_w = self.frames[fi].shape[:2]
                    vx1, vy1, vx2, vy2 = vb
                    vw, vh = max(1.0, vx2 - vx1), max(1.0, vy2 - vy1)
                    candidates = []
                    if result.boxes is not None:
                        for b, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                            bx1, by1, bx2, by2 = [float(v) for v in b]
                            bw, bh = bx2 - bx1, by2 - by1
                            ratio = bw / max(bh, 1.0)
                            # Reject tiny, implausibly large, or non-plate-shaped detections.
                            if (bw < max(10.0, 0.025 * vw) or bh < max(6.0, 0.015 * vh)
                                    or not 0.60 <= ratio <= 8.5
                                    or bw * bh > 0.24 * vw * vh):
                                continue
                            ab = (max(0.0, bx1 + x1), max(0.0, by1 + y1),
                                  min(float(fr_w), bx2 + x1), min(float(fr_h), by2 + y1))
                            candidates.append({
                                "score": float(conf), "box": ab,
                                "quads": [plate.box_to_quad(ab)], "source": "plate model",
                            })
                    candidates.sort(key=lambda p: -p["score"])
                    props[fi] = candidates[:6]
                    if k % 5 == 0:
                        self.progress(25 + int(30 * k / max(len(crop_meta), 1)), "Detecting number plates")
            else:
                # Keep the original CRAFT + colour proposals available when the
                # optional plate detector has not been installed.
                for k, (fi, vb) in enumerate(boxes.items()):
                    fr = self.frames[fi]
                    x1, y1, x2, y2 = plate.pad_box(vb, fr.shape, 0.04, 0.04)
                    crop = fr[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    shifted = []
                    for p in plate.propose(crop):
                        b = p["box"]
                        shifted.append({
                            "score": p["score"],
                            "box": (b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1),
                            "quads": plate.shift_quads(p["quads"], x1, y1),
                            "from_text": p["from_text"], "source": "detected",
                        })
                    props[fi] = shifted
                    if k % 5 == 0:
                        self.progress(25 + int(30 * k / max(len(boxes), 1)), "Locating number plate")
            # Reference plate location from high-confidence (preferably text-backed) frames
            text_good = [
                (fi, p[0]) for fi, p in props.items()
                if p and (p[0].get("from_text") or p[0].get("source") == "plate model")
                and p[0]["score"] >= 0.25
            ]
            pool = text_good if len(text_good) >= 2 else [(fi, p[0]) for fi, p in props.items() if p]
            med, ref_rquads = None, None
            if len(pool) >= 2:
                pool_sorted = sorted(pool, key=lambda x: -x[1]["score"])[:max(3, len(pool) // 2)]
                arr = np.array([plate.rel_box(p["box"], boxes[fi]) for fi, p in pool_sorted])
                med = tuple(np.median(arr, 0))
                best_fi_loc, best_p = pool_sorted[0]
                ref_rquads = plate.rel_quads(best_p["quads"], boxes[best_fi_loc])
            for fi, plist in props.items():
                best = None
                for p in plist:
                    agree = plate.iou(plate.rel_box(p["box"], boxes[fi]), med) if med else 1.0
                    sc = p["score"] * (0.20 + 0.80 * agree) if med else p["score"]
                    if best is None or sc > best["score"]:
                        best = {"box": p["box"], "quads": p["quads"], "score": sc,
                                "source": p.get("source", "detected")}
                if (best is None or best["score"] < 0.14) and med is not None:
                    ab = plate.abs_box(med, boxes[fi])
                    aq = plate.abs_quads(ref_rquads, boxes[fi]) if ref_rquads else [plate.box_to_quad(ab)]
                    best = {"box": ab, "quads": aq, "score": 0.12, "source": "propagated"}
                if best:
                    per_frame[fi] = best

        # ---- OCR per frame
        self.progress(55, "Reading plate characters frame by frame")
        frame_rows, readings = [], []
        for k, (fi, info) in enumerate(sorted(per_frame.items())):
            fr = self.frames[fi]
            pb = plate.pad_box(info["box"], fr.shape)
            pc = fr[pb[1]:pb[3], pb[0]:pb[2]]
            if pc.size == 0 or pc.shape[0] < 6:
                continue
            strips = plate.warp_quads(fr, info.get("quads") or [])
            # OCR reads a known plate crop directly; CRAFT is only needed by the
            # legacy plate-proposal path, never by the CRNN recognizer itself.
            rd = ocr.read_plate(pc, line_strips=strips, detector=False)
            known = [c for c in rd["chars"] if c["char"] != "?"]
            sharp = _sharpness(pc)
            s_sharp = min(1.0, sharp / 600.0) ** 0.5
            ph = pb[3] - pb[1]
            s_size = min(1.0, ph / (40.0 if rd["two_line"] else 24.0))
            gray_pc = cv2.cvtColor(pc, cv2.COLOR_BGR2GRAY) if pc.ndim == 3 else pc
            contrast = float(np.std(gray_pc))
            s_contrast = min(1.0, contrast / 50.0)
            quality_crop = strips[0] if strips else pc
            aspect = quality_crop.shape[1] / max(quality_crop.shape[0], 1)
            plausible_aspect = (0.75 <= aspect <= 1.75) if rd["two_line"] else (1.7 <= aspect <= 8.5)
            s_geometry = 1.0 if plausible_aspect else 0.25
            # OCR can be confidently wrong when the text detector selected a lamp,
            # badge, or other text on the vehicle. Keep localization confidence in
            # the frame vote so a weak/propagated crop cannot dominate fusion.
            s_loc = float(np.clip(info.get("score", 0.0), 0.0, 1.0))
            quality_score = 0.30 * s_loc + 0.20 * s_sharp + 0.20 * s_size + 0.15 * s_contrast + 0.15 * s_geometry
            # Mean OCR confidence is computed only over emitted glyphs. Without a
            # completeness term, a tiny high-confidence fragment can outweigh a
            # longer reading that carries more plate evidence. Use a gentle
            # square-root penalty so partial/occluded views still contribute.
            s_coverage = min(1.0, len(known) / 6.0) ** 0.5
            w = (rd["mean_conf"] ** 2) * s_coverage * (0.4 + 0.6 * s_sharp) * (0.4 + 0.6 * s_size) * \
                (0.25 + 0.75 * s_loc) * (0.6 if info["source"] == "propagated" else 1.0)
            excluded = None
            if len(known) < 3:
                excluded, w = "fewer than 3 legible characters", 0.0
            big = cv2.resize(pc, (int(64 * pc.shape[1] / pc.shape[0]), 64), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(self.dir / "plates" / f"p{fi:03d}.png"), big)
            row = {"frame": fi, "source_frame": self.frame_ids[fi], "time_s": self.times[fi],
                   "vehicle_box": [round(v) for v in boxes[fi]], "plate_box": [int(v) for v in pb],
                   "plate_source": info["source"], "loc_score": round(float(info["score"]), 3),
                   "text": rd["text"], "mean_conf": round(rd["mean_conf"], 3), "two_line": rd["two_line"],
                   "quality": {"score": round(float(quality_score), 3), "contrast": round(contrast, 1),
                               "geometry": round(s_geometry, 3), "sharpness_score": round(s_sharp, 3),
                               "size_score": round(s_size, 3), "ocr_coverage_score": round(s_coverage, 3)},
                   "sharpness": round(sharp, 1), "plate_height_px": int(ph), "weight": round(w, 4),
                   "excluded": excluded, "variant": rd["variant"],
                   "chars": [{"char": c["char"], "conf": round(c["conf"], 3),
                              "alts": ([(attributes_char, round(float(p), 3)) for attributes_char, p in
                                        _top_alts(c["dist"])] if c["dist"] is not None else [])}
                             for c in rd["chars"]],
                   "plate_img": f"plates/p{fi:03d}.png", "frame_img": f"frames/f{fi:03d}.jpg"}
            frame_rows.append(row)
            if w > 0:
                readings.append({"frame": fi, "weight": w, "chars": rd["chars"]})
            if k % 5 == 0:
                self.progress(55 + int(20 * k / max(len(per_frame), 1)), "Reading plate characters")
        readings.sort(key=lambda r: -r["weight"])
        used = readings[:MAX_FUSION_FRAMES]
        used_ids = {r["frame"] for r in used}
        for row in frame_rows:
            row["used_in_fusion"] = row["frame"] in used_ids
            if not row["used_in_fusion"] and not row["excluded"]:
                row["excluded"] = "lower quality than the frames selected for fusion"
        fused = fusion.fuse(used)

        # ---- attributes
        self.progress(78, "Extracting colour, body type and make/model")
        H0, W0 = self.frames[0].shape[:2]
        cand = sorted(boxes.items(), key=lambda kv: -((kv[1][2] - kv[1][0]) * (kv[1][3] - kv[1][1])))
        attr_frames = cand[: min(8, len(cand))]
        cols, cw, crops = [], [], []
        for fi, vb in attr_frames:
            d = attributes.colour_distribution(self.frames[fi], vb)
            if d:
                cols.append(d)
                cw.append(1.0)
            x1, y1, x2, y2 = [int(v) for v in vb]
            c = self.frames[fi][max(0, y1):y2, max(0, x1):x2]
            if c.size:
                crops.append(c)
        colour = attributes.merge_dists(cols, cw)
        feats = attributes.clip_image_feats(crops[:6])
        colour = attributes.fuse_colour_with_clip(colour, feats)
        body, body_clip = attributes.body_type(track["cls"], feats)
        top_body = max(body, key=body.get) if body else None
        mm = attributes.make_model(feats, self.registry, top_body)
        signature = {"embedding": feats.mean(0).cpu().numpy().round(5).tolist() if feats is not None else None,
                     "colour": colour, "body": body, "plate_positions": fused["positions"]}

        # ---- evidence + matching
        self.progress(90, "Matching against registry")
        ev = matcher.Evidence()
        if any(p["known"] for p in fused["positions"]) and not hints.get("ignore_video_plate"):
            ev.plates.append(matcher.tokens_from_fused(fused["positions"]))
            ev.plate_sources.append("video, multi-frame")
        if hints.get("plate_pattern"):
            ev.plates.append(matcher.tokens_from_pattern(hints["plate_pattern"]))
            ev.plate_sources.append("investigator hint")
        ev.colour = {hints["colour"]: 1.0} if hints.get("colour") in COLOURS else (colour or None)
        ev.body = {hints["body_type"]: 1.0} if hints.get("body_type") in BODY_TYPES else (body or None)
        ev.model = mm if hints.get("use_make_model", True) else None
        result = matcher.match(ev, self.registry)
        quality = confidence.assess(
            {"_boxes": track["boxes"]}, frame_rows, fused, result,
        )
        result = confidence.gate_registry_match(result, quality)

        # ---- key frame annotation
        best_fi = fused.get("reference_frame")
        if best_fi is None:
            best_fi = cand[0][0] if cand else 0
        kf = self.frames[best_fi].copy()
        vb = [int(v) for v in boxes.get(best_fi, (0, 0, 0, 0))]
        cv2.rectangle(kf, (vb[0], vb[1]), (vb[2], vb[3]), (0, 200, 255), 2)
        if best_fi in per_frame:
            pb = [int(v) for v in per_frame[best_fi]["box"]]
            cv2.rectangle(kf, (pb[0], pb[1]), (pb[2], pb[3]), (0, 255, 0), 2)
        label = f"plate: {fused['text'] or 'n/a'}"
        cv2.putText(kf, label, (vb[0], max(20, vb[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(kf, label, (vb[0], max(20, vb[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(self.dir / "keyframe.jpg"), kf, [cv2.IMWRITE_JPEG_QUALITY, 90])

        out = {
            "video": {**self.meta, "file": Path(self.video_path).name},
            "track": {"id": track["id"], "frames": len(boxes), "yolo_classes": track["cls"],
                      "fallback_whole_frame": bool(track.get("fallback"))},
            "tracks": self.tracks_summary(),
            "hints": hints,
            "frames": frame_rows,
            "fusion": fused,
            "attributes": {"colour": colour, "body_type": body, "body_type_clip": body_clip,
                           "make_model": mm, "colour_hint": hints.get("colour"),
                           "body_hint": hints.get("body_type")},
            "confidence": quality,
            "match": result,
            "signature": signature,
            "keyframe": "keyframe.jpg",
            "elapsed_s": round(time.time() - t0, 1),
        }
        out = _np(out)
        (self.dir / "result.json").write_text(json.dumps(out, indent=1))
        self.progress(100, "Done")
        return out


def _top_alts(dist, k=3):
    if dist is None:
        return []
    from .vocab import CHARS
    order = np.argsort(-dist)[:k]
    return [(CHARS[i], dist[i]) for i in order]
