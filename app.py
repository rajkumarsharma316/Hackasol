"""FastAPI web application for VTrace - Vehicle Identification & Traceability from CCTV Footage."""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import uuid
import csv
import hmac
import io
import os
import mimetypes
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from vtrace import confidence, matcher, reid, report
from vtrace.pipeline import MAX_FRAMES, VideoAnalysis
from vtrace.vocab import BODY_TYPES, COLOURS, INDIAN_STATE_CODES

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TEST_CLIPS_DIR = ROOT / "test_clips"
REPORTS_DIR = ROOT / "reports"
JOBS_DIR = ROOT / "jobs"
STATIC_DIR = ROOT / "static"

JOBS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="VTrace - CCTV Vehicle Identification & Traceability")
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")

_registry_lock = threading.Lock()
_registry: list = matcher.load_registry(DATA_DIR / "registry.csv")
_jobs: dict[str, dict] = {}
_va_cache: OrderedDict[str, VideoAnalysis] = OrderedDict()
_analysis_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vtrace-analysis")
_analysis_slots = threading.BoundedSemaphore(2)  # one running inference plus one queued job


def _queue_analysis(job_id: str, hints: dict) -> bool:
    if not _analysis_slots.acquire(blocking=False):
        return False
    future = _analysis_executor.submit(_run_analysis_bg, job_id, hints)
    future.add_done_callback(lambda _: _analysis_slots.release())
    return True


def get_registry() -> list:
    with _registry_lock:
        return list(_registry)


def _clip_path(clip_id: str, suffix: str) -> Path:
    """Resolve a sample clip by its simple ID, never by an arbitrary path."""
    if not CLIP_ID_RE.fullmatch(clip_id):
        raise HTTPException(404, "Test clip not found")
    path = (TEST_CLIPS_DIR / f"{clip_id}{suffix}").resolve()
    if not path.is_relative_to(TEST_CLIPS_DIR.resolve()):
        raise HTTPException(404, "Test clip not found")
    return path


def _public_job(job: dict) -> dict:
    """Return job state without exposing local filesystem paths."""
    return {key: value for key, value in job.items() if key not in {"video_path", "job_dir"}}


def _seed_precomputed_jobs():
    """Preload any batch-analysed test clip directories from reports/ into _jobs so they open instantly."""
    gt_path = TEST_CLIPS_DIR / "ground_truth.json"
    gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}
    for clip in sorted(TEST_CLIPS_DIR.glob("*.mp4")):
        jid = f"demo_{clip.stem}"
        # Keep in-flight work and updated results in memory instead of resetting
        # them to the original precomputed snapshot on every polling request.
        if jid in _jobs:
            continue
        rdir = REPORTS_DIR / clip.stem
        res_file = rdir / "result.json"
        if res_file.exists():
            try:
                res = json.loads(res_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping invalid precomputed result: %s", res_file)
                continue
            _jobs[jid] = {
                "job_id": jid,
                "status": "done",
                "progress": 100,
                "message": "Ready",
                "video_path": str(clip),
                "job_dir": str(rdir),
                "filename": clip.name,
                "ground_truth": gt.get(clip.stem),
                "camera_id": _camera_from_name(clip.stem),
                "recorded_at": None,
                "result": res,
            }
            res.setdefault("observation", {"job_id": jid, "camera_id": _camera_from_name(clip.stem),
                                           "recorded_at": None, "time_basis": "video-relative offset only"})
            res.setdefault("review", {"status": "pending", "machine_prediction": res.get("fusion", {}).get("text", ""),
                                       "corrected_plate": None, "events": []})

    # Rehydrate completed upload jobs so observation history survives restarts.
    for result_file in sorted(JOBS_DIR.glob("*/result.json")):
        try:
            res = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping invalid saved job result: %s", result_file)
            continue
        folder = result_file.parent
        obs = res.get("observation") or {}
        jid = obs.get("job_id") or folder.name
        if jid in _jobs:
            continue
        video = res.get("video", {}).get("file", "")
        vpath = folder / Path(video).name if video else None
        _jobs[jid] = {
            "job_id": jid, "status": "done", "progress": 100, "message": "Ready",
            "video_path": str(vpath) if vpath and vpath.is_file() else "",
            "job_dir": str(folder), "filename": video or folder.name,
            "camera_id": obs.get("camera_id", "unspecified"),
            "recorded_at": obs.get("recorded_at"), "result": res,
        }


def _camera_from_name(name: str) -> str:
    m = re.search(r"cam([A-Za-z0-9_-]+)", name, re.I)
    return f"camera-{m.group(1)}" if m else "benchmark"


def _decorate_result(job: dict, result: dict):
    """Attach source and time context while keeping frame timestamps as offsets."""
    analyzed_at = _utc_now()
    recorded_at = job.get("recorded_at")
    result["observation"] = {
        "job_id": job["job_id"],
        "camera_id": job.get("camera_id") or "unspecified",
        "recorded_at": recorded_at,
        "analyzed_at": analyzed_at,
        "time_basis": "recorded_at + video offset" if recorded_at else "video-relative offset only",
    }
    if recorded_at:
        base = datetime.fromisoformat(recorded_at)
        for frame in result.get("frames", []):
            frame["timestamp"] = (base.timestamp() + float(frame.get("time_s", 0)))
            frame["timestamp_utc"] = datetime.fromtimestamp(frame["timestamp"], timezone.utc).isoformat(timespec="seconds")
    else:
        for frame in result.get("frames", []):
            frame["timestamp"] = None
            frame["timestamp_utc"] = None
    current = job.get("result") or {}
    previous = current.get("review") or {}
    result["review"] = {
        "status": "pending",
        "machine_prediction": result.get("fusion", {}).get("text", ""),
        "corrected_plate": None,
        "events": previous.get("events", []),
    }
    return result


_seed_precomputed_jobs()


class ROIRequest(BaseModel):
    frame: int = Field(ge=0, le=MAX_FRAMES - 1)
    x1: float = Field(ge=0, le=10000)
    y1: float = Field(ge=0, le=10000)
    x2: float = Field(gt=0, le=10000)
    y2: float = Field(gt=0, le=10000)

    @model_validator(mode="after")
    def check_bounds(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("ROI must have positive width and height")
        return self


class ReanalyseRequest(BaseModel):
    plate_pattern: str | None = Field(default=None, max_length=24)
    colour: str | None = Field(default=None, max_length=32)
    body_type: str | None = Field(default=None, max_length=32)
    track_id: int | None = Field(default=None, gt=0)
    ignore_video_plate: bool = False
    use_make_model: bool = True
    roi: ROIRequest | None = None


class QuickMatchRequest(BaseModel):
    plate_pattern: str = Field(default="", max_length=24)
    colour: str | None = Field(default=None, max_length=32)
    body_type: str | None = Field(default=None, max_length=32)
    make_model: str | None = Field(default=None, max_length=100)


class CompareRequest(BaseModel):
    job_a: str = Field(min_length=1, max_length=80)
    job_b: str = Field(min_length=1, max_length=80)


class ReviewRequest(BaseModel):
    action: Literal["accepted", "rejected", "corrected"]
    corrected_plate: str | None = Field(default=None, max_length=16)
    reviewer: str = Field(min_length=1, max_length=80)
    note: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def review_fields(self):
        if self.action == "corrected" and not self.corrected_plate:
            raise ValueError("A corrected registration is required for a correction.")
        if self.action != "corrected" and self.corrected_plate:
            raise ValueError("Only a corrected review may include corrected_plate.")
        return self


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_recorded_at(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(422, "recorded_at must be an ISO-8601 date/time.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def _valid_registration(value: str) -> bool:
    plate = re.sub(r"[ -]", "", value.upper())
    match = re.fullmatch(r"([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})", plate)
    if match and match.group(1) in INDIAN_STATE_CODES:
        return True
    return bool(re.fullmatch(r"\d{2}BH\d{4}[A-Z]{1,2}", plate))


def _validate_hints(plate_pattern: str | None, colour: str | None, body_type: str | None):
    if plate_pattern and not re.fullmatch(r"[A-Za-z0-9?* -]{1,24}", plate_pattern):
        raise HTTPException(422, "Plate pattern may contain letters, numbers, spaces, hyphens, ? and * only.")
    if colour and colour not in COLOURS:
        raise HTTPException(422, "Unsupported vehicle colour.")
    if body_type and body_type not in BODY_TYPES:
        raise HTTPException(422, "Unsupported vehicle body type.")


def _run_analysis_bg(jid: str, hints: dict):
    job = _jobs[jid]
    try:
        def _prog(p: int, msg: str):
            job["progress"] = p
            job["message"] = msg

        va = _va_cache.get(jid)
        if va is None:
            va = VideoAnalysis(job["video_path"], job["job_dir"], get_registry(), progress=_prog)
            _va_cache[jid] = va
        else:
            _va_cache.move_to_end(jid)
            va.progress = _prog
            va.registry = get_registry()
        while len(_va_cache) > 1:
            _va_cache.popitem(last=False)
        res = va.analyse(hints)
        res = _decorate_result(job, res)
        (Path(job["job_dir"]) / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        html_str = report.render(res, Path(job["job_dir"]), embed=True)
        (Path(job["job_dir"]) / "report.html").write_text(html_str, encoding="utf-8")
        job["result"] = res
        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Analysis complete"
    except Exception:
        _va_cache.pop(jid, None)
        logger.exception("Analysis failed for job %s", jid)
        job["status"] = "error"
        job["message"] = "Analysis failed. Check the server log for details."


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/meta")
def api_meta():
    _seed_precomputed_jobs()
    gt_path = TEST_CLIPS_DIR / "ground_truth.json"
    gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}
    clips = []
    for clip in sorted(TEST_CLIPS_DIR.glob("*.mp4")):
        g = gt.get(clip.stem, {})
        jid = f"demo_{clip.stem}"
        clips.append({
            "id": clip.stem,
            "filename": clip.name,
            "precomputed_job_id": jid if jid in _jobs else None,
            "note": g.get("note", ""),
            "ground_truth_plate": g.get("plate"),
            "in_registry": g.get("in_registry", True),
            "preview_url": f"/api/test-clips/{clip.stem}/preview",
            "video_url": f"/api/test-clips/{clip.stem}/video",
        })
    reg = get_registry()
    models = sorted({f"{r['make']} {r['model']}" for r in reg})
    return {
        "colours": COLOURS,
        "body_types": BODY_TYPES,
        "models": models,
        "registry_size": len(reg),
        "test_clips": clips,
        "jobs": [
            {
                "job_id": j["job_id"],
                "filename": j["filename"],
                "status": j["status"],
                "plate": (j.get("result") or {}).get("fusion", {}).get("text"),
                "camera_id": j.get("camera_id", "unspecified"),
            }
            for j in _jobs.values()
            if j["status"] == "done"
        ],
    }


@app.get("/api/test-clips/{clip_id}/preview")
def api_clip_preview(clip_id: str):
    p = _clip_path(clip_id, "_preview.jpg")
    if not p.exists():
        raise HTTPException(404, "Preview not found")
    return FileResponse(p)


@app.get("/api/test-clips/{clip_id}/video")
def api_clip_video(clip_id: str):
    p = _clip_path(clip_id, ".mp4")
    if not p.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(p, media_type="video/mp4")


@app.post("/api/analyse/sample/{clip_id}")
def api_analyse_sample(clip_id: str, force: bool = False):
    _seed_precomputed_jobs()
    # Keep immutable reports/ benchmark outputs untouched; live analyses belong
    # under jobs/ and use a separate ID from the precomputed baseline snapshot.
    jid = f"live_demo_{clip_id}"
    if jid in _jobs and _jobs[jid]["status"] == "running":
        raise HTTPException(409, "This clip is already being analysed.")
    if not force and jid in _jobs and _jobs[jid]["status"] == "done":
        return {"job_id": jid, "cached": True}
    vpath = _clip_path(clip_id, ".mp4")
    if not vpath.exists():
        raise HTTPException(404, "Test clip not found")
    previous_job = _jobs.get(jid)
    jdir = JOBS_DIR / jid
    jdir.mkdir(parents=True, exist_ok=True)
    _jobs[jid] = {
        "job_id": jid,
        "status": "running",
        "progress": 1,
        "message": "Starting analysis...",
        "video_path": str(vpath),
        "job_dir": str(jdir),
        "filename": vpath.name,
        "camera_id": _camera_from_name(clip_id),
        "recorded_at": None,
        "result": None,
    }
    if not _queue_analysis(jid, {}):
        if previous_job:
            _jobs[jid] = previous_job
        else:
            _jobs.pop(jid, None)
        raise HTTPException(429, "Analysis queue is full. Wait for a running clip to finish, then retry.")
    return {"job_id": jid, "cached": False}


@app.post("/api/analyse/upload")
async def api_analyse_upload(
    file: UploadFile = File(...),
    plate_pattern: str = Form(""),
    colour: str = Form(""),
    body_type: str = Form(""),
    camera_id: str = Form("unspecified"),
    recorded_at: str = Form(""),
):
    _validate_hints(plate_pattern.strip() or None, colour.strip() or None, body_type.strip() or None)
    camera_id = camera_id.strip()[:80] or "unspecified"
    if not re.fullmatch(r"[\w .:-]{1,80}", camera_id, re.UNICODE):
        raise HTTPException(422, "Camera ID contains unsupported characters.")
    recorded_at_utc = _parse_recorded_at(recorded_at.strip() or None)
    jid = uuid.uuid4().hex[:10]
    jdir = JOBS_DIR / jid
    jdir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload.mp4").name
    if Path(safe_name).suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        await file.close()
        shutil.rmtree(jdir, ignore_errors=True)
        raise HTTPException(415, "Upload a supported video file (MP4, MOV, M4V, AVI, MKV, or WebM).")
    vpath = jdir / safe_name
    size = 0
    try:
        with open(vpath, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Video exceeds the 500 MB upload limit.")
                f.write(chunk)
    except Exception:
        shutil.rmtree(jdir, ignore_errors=True)
        raise
    finally:
        await file.close()
    if size == 0:
        shutil.rmtree(jdir, ignore_errors=True)
        raise HTTPException(400, "The uploaded video is empty.")
    hints = {}
    if plate_pattern.strip():
        hints["plate_pattern"] = plate_pattern.strip()
    if colour.strip() in COLOURS:
        hints["colour"] = colour.strip()
    if body_type.strip() in BODY_TYPES:
        hints["body_type"] = body_type.strip()
    _jobs[jid] = {
        "job_id": jid,
        "status": "running",
        "progress": 1,
        "message": "Uploaded clip, starting pipeline...",
        "video_path": str(vpath),
        "job_dir": str(jdir),
        "filename": safe_name,
        "camera_id": camera_id,
        "recorded_at": recorded_at_utc,
        "result": None,
    }
    if not _queue_analysis(jid, hints):
        _jobs.pop(jid, None)
        shutil.rmtree(jdir, ignore_errors=True)
        raise HTTPException(429, "Analysis queue is full. Wait for a running clip to finish, then retry.")
    return {"job_id": jid}


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str):
    _seed_precomputed_jobs()
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return _public_job(_jobs[job_id])


@app.post("/api/jobs/{job_id}/reanalyse")
def api_reanalyse(job_id: str, req: ReanalyseRequest):
    _seed_precomputed_jobs()
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    if job["status"] == "running":
        raise HTTPException(409, "This job is already being analysed.")
    _validate_hints(req.plate_pattern, req.colour, req.body_type)
    hints = {k: v for k, v in req.model_dump().items() if v is not None and v != ""}
    # If only matching hints changed (no new ROI or track_id), re-match instantaneously without re-running OCR
    prev_res = job.get("result")
    prev_hints = (prev_res or {}).get("hints") or {}
    needs_full_ocr = (
        prev_res is None
        or hints.get("track_id") != prev_hints.get("track_id")
        or hints.get("roi") != prev_hints.get("roi")
    )
    if not needs_full_ocr:
        fused = prev_res["fusion"]
        at = prev_res["attributes"]
        ev = matcher.Evidence()
        if any(p["known"] for p in fused["positions"]) and not hints.get("ignore_video_plate"):
            ev.plates.append(matcher.tokens_from_fused(fused["positions"]))
            ev.plate_sources.append("video, multi-frame")
        if hints.get("plate_pattern"):
            ev.plates.append(matcher.tokens_from_pattern(hints["plate_pattern"]))
            ev.plate_sources.append("investigator hint")
        ev.colour = {hints["colour"]: 1.0} if hints.get("colour") in COLOURS else (at["colour"] or None)
        ev.body = {hints["body_type"]: 1.0} if hints.get("body_type") in BODY_TYPES else (at["body_type"] or None)
        ev.model = at.get("make_model") if hints.get("use_make_model", True) else None
        prev_res["hints"] = hints
        prev_res["attributes"]["colour_hint"] = hints.get("colour")
        prev_res["attributes"]["body_hint"] = hints.get("body_type")
        matched = matcher.match(ev, get_registry())
        quality = confidence.reassess_registry(prev_res.get("confidence", {}), matched)
        prev_res["confidence"] = quality
        prev_res["match"] = confidence.gate_registry_match(matched, quality)
        jdir = Path(job["job_dir"])
        (jdir / "result.json").write_text(json.dumps(prev_res, indent=1), encoding="utf-8")
        (jdir / "report.html").write_text(report.render(prev_res, jdir, embed=True), encoding="utf-8")
        return {"job_id": job_id, "instant": True, "result": prev_res}

    previous_job = dict(job)
    job["status"] = "running"
    job["progress"] = 10
    job["message"] = "Re-running pipeline with updated investigator hints..."
    if not _queue_analysis(job_id, hints):
        _jobs[job_id] = previous_job
        raise HTTPException(429, "Analysis queue is full. Wait for a running clip to finish, then retry.")
    return {"job_id": job_id, "instant": False}


@app.get("/api/jobs/{job_id}/asset/{rel_path:path}")
def api_job_asset(job_id: str, rel_path: str):
    _seed_precomputed_jobs()
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    base = Path(_jobs[job_id]["job_dir"]).resolve()
    target = (base / rel_path).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        raise HTTPException(404, "Asset not found")
    return FileResponse(target)


@app.get("/api/jobs/{job_id}/video")
def api_job_video(job_id: str):
    _seed_precomputed_jobs()
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    p = Path(_jobs[job_id]["video_path"])
    if not p.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(p, media_type=mimetypes.guess_type(p.name)[0] or "application/octet-stream")


@app.get("/api/jobs/{job_id}/report", response_class=HTMLResponse)
def api_job_report(job_id: str, download: bool = False):
    _seed_precomputed_jobs()
    if job_id not in _jobs or not _jobs[job_id].get("result"):
        raise HTTPException(404, "Result not ready")
    job = _jobs[job_id]
    html_str = report.render(job["result"], Path(job["job_dir"]), embed=True)
    headers = {}
    if download:
        fname = f"VTrace_Report_{job_id}.html"
        headers["Content-Disposition"] = f'attachment; filename="{fname}"'
    return HTMLResponse(content=html_str, headers=headers)


@app.post("/api/match")
def api_quick_match(req: QuickMatchRequest):
    _validate_hints(req.plate_pattern.strip() or None, req.colour, req.body_type)
    if req.make_model and req.make_model not in {f"{r['make']} {r['model']}" for r in get_registry()}:
        raise HTTPException(422, "Make/model is not present in the registry.")
    ev = matcher.Evidence()
    if req.plate_pattern.strip():
        ev.plates.append(matcher.tokens_from_pattern(req.plate_pattern.strip()))
        ev.plate_sources.append("investigator query")
    if req.colour in COLOURS:
        ev.colour = {req.colour: 1.0}
    if req.body_type in BODY_TYPES:
        ev.body = {req.body_type: 1.0}
    if req.make_model:
        ev.model = {req.make_model: 1.0}
    return matcher.match(ev, get_registry())


@app.post("/api/compare")
def api_compare(req: CompareRequest):
    _seed_precomputed_jobs()
    if req.job_a not in _jobs or req.job_b not in _jobs:
        raise HTTPException(404, "One or both jobs not found")
    ja, jb = _jobs[req.job_a], _jobs[req.job_b]
    if not ja.get("result") or not jb.get("result"):
        raise HTTPException(400, "Both clips must finish analysis before comparison")
    return reid.compare_clips(ja["result"], jb["result"], get_registry())


def _observation_rows():
    _seed_precomputed_jobs()
    rows = []
    for jid, job in _jobs.items():
        res = job.get("result")
        if job.get("status") != "done" or not res:
            continue
        obs = res.get("observation", {})
        review = res.get("review", {})
        match = res.get("match", {})
        confidence_data = res.get("confidence", {})
        registry_status = match.get("registry_status", match.get("status", "unknown"))
        top = (match.get("candidates") or [None])[0]
        machine_plate = review.get("machine_prediction", res.get("fusion", {}).get("text", ""))
        effective_plate = review.get("corrected_plate") or (
            machine_plate if review.get("status") == "accepted" else ""
        )
        rows.append({
            "job_id": jid,
            "filename": job.get("filename", ""),
            "camera_id": obs.get("camera_id", job.get("camera_id", "unspecified")),
            "recorded_at": obs.get("recorded_at", job.get("recorded_at")),
            "analyzed_at": obs.get("analyzed_at"),
            "timestamp_basis": obs.get("time_basis", "analysis time only"),
            "video_offset_s": None,
            "machine_prediction": machine_plate,
            "plate": effective_plate or machine_plate,
            "review_status": review.get("status", "pending"),
            "reviewed_plate": review.get("corrected_plate"),
            "registry_status": registry_status,
            "system_status": match.get("status", "unknown"),
            "registry_candidate": (top or {}).get("vehicle", {}).get("reg_no"),
            "registry_posterior": confidence_data.get("registry_match_posterior",
                                                       ((top or {}).get("confidence", 0) / 100)),
            "recognition_tier": confidence_data.get("tier", "unscored"),
            "processing_seconds": res.get("elapsed_s", 0),
            "track_count": len(res.get("tracks", [])),
            "frame_count": res.get("video", {}).get("sampled", 0),
            "ground_truth": (job.get("ground_truth") or {}).get("plate"),
            "ground_truth_in_registry": (job.get("ground_truth") or {}).get("in_registry"),
        })
    rows.sort(key=lambda x: x.get("recorded_at") or x.get("analyzed_at") or "", reverse=True)
    # Mark possible duplicate observations only where explicit capture timestamps
    # and plate evidence exist. Appearance and colour are never identity keys.
    for i, row in enumerate(rows):
        row["possible_duplicate_of"] = None
        if not row["recorded_at"] or not row["plate"] or "?" in row["plate"]:
            continue
        for earlier in rows[:i]:
            if (earlier["recorded_at"] and earlier["plate"].replace(" ", "") == row["plate"].replace(" ", "")
                    and earlier["camera_id"] == row["camera_id"]):
                try:
                    dt = abs(datetime.fromisoformat(row["recorded_at"]).timestamp()
                             - datetime.fromisoformat(earlier["recorded_at"]).timestamp())
                except ValueError:
                    continue
                if dt <= 30:
                    row["possible_duplicate_of"] = earlier["job_id"]
                    break
    return rows


@app.get("/api/observations")
def api_observations(q: str = "", camera_id: str = "", start: str = "", end: str = "",
                     review_status: str = "", limit: int = 500):
    limit = max(1, min(limit, 2000))
    start_utc = _parse_recorded_at(start + "T00:00:00+00:00" if len(start) == 10 else start) if start else None
    end_utc = _parse_recorded_at(end + "T23:59:59+00:00" if len(end) == 10 else end) if end else None
    if review_status and review_status not in {"pending", "accepted", "rejected", "corrected"}:
        raise HTTPException(422, "Unsupported review status.")
    q = q.strip().upper()
    out = []
    for row in _observation_rows():
        stamp = row["recorded_at"] or row["analyzed_at"] or ""
        if q and q not in (row["plate"] + " " + row["filename"]).upper():
            continue
        if camera_id and camera_id != row["camera_id"]:
            continue
        if start_utc and stamp and stamp < start_utc:
            continue
        if end_utc and stamp and stamp > end_utc:
            continue
        if review_status and row["review_status"] != review_status:
            continue
        out.append(row)
    return {"total": len(out), "rows": out[:limit]}


@app.get("/api/analytics")
def api_analytics():
    rows = _observation_rows()
    by_camera = {}
    by_day = {}
    for row in rows:
        cam = row["camera_id"]
        stats = by_camera.setdefault(cam, {"observations": 0, "registry_matches": 0,
                                           "uncertain": 0, "review_pending": 0})
        stats["observations"] += 1
        stats["registry_matches"] += row["registry_status"] == "match"
        stats["uncertain"] += row["recognition_tier"] in {"low", "medium", "unscored"}
        stats["review_pending"] += row["review_status"] == "pending"
        day = (row.get("analyzed_at") or "")[:10] or "unknown"
        trend = by_day.setdefault(day, {"observations": 0, "automatic_matches": 0, "uncertain": 0})
        trend["observations"] += 1
        trend["automatic_matches"] += row["system_status"] == "match"
        trend["uncertain"] += row["recognition_tier"] in {"low", "medium", "unscored"}
    labeled = [r for r in rows if r["ground_truth_in_registry"] is not None]
    registered = [r for r in labeled if r["ground_truth_in_registry"]]
    unregistered = [r for r in labeled if not r["ground_truth_in_registry"]]
    fp = sum(r["system_status"] == "match" for r in unregistered)
    fn = sum(r["system_status"] != "match" for r in registered)
    candidate_fp = sum(r["registry_status"] == "match" for r in unregistered)
    runtimes = [float(r["processing_seconds"] or 0) for r in rows]
    return {
        "observations": len(rows),
        "vehicle_tracks_detected": sum(r["track_count"] for r in rows),
        "successful_plate_recognitions": sum(r["recognition_tier"] == "high" for r in rows),
        "uncertain_predictions": sum(r["recognition_tier"] in {"low", "medium", "unscored"} for r in rows),
        "registry_matches": sum(r["registry_status"] == "match" for r in rows),
        "unmatched": sum(r["registry_status"] != "match" for r in rows),
        "review_pending": sum(r["review_status"] == "pending" for r in rows),
        "reviewed_unique_plates": len({r["plate"] for r in rows if r["review_status"] in {"accepted", "corrected"} and r["plate"]}),
        "benchmark": {
            "labeled_clips": len(labeled),
            "false_positive_rate": round(fp / len(unregistered), 4) if unregistered else None,
            "false_negative_rate": round(fn / len(registered), 4) if registered else None,
            "registry_candidate_false_positive_rate": round(candidate_fp / len(unregistered), 4) if unregistered else None,
            "limitation": "Only present for stored observations with benchmark ground truth.",
        },
        "processing_seconds_mean": round(sum(runtimes) / len(runtimes), 2) if runtimes else None,
        "by_camera": by_camera,
        "by_analysis_day": by_day,
        "time_basis": "Capture time when supplied; otherwise analyzed_at. Sample clips have no capture date.",
    }


@app.get("/api/observations.csv")
def api_observations_csv(q: str = "", camera_id: str = "", start: str = "", end: str = "",
                         review_status: str = ""):
    data = api_observations(q=q, camera_id=camera_id, start=start, end=end,
                            review_status=review_status, limit=2000)
    fields = ["job_id", "filename", "camera_id", "recorded_at", "analyzed_at", "plate",
              "machine_prediction", "reviewed_plate", "review_status", "recognition_tier",
              "system_status", "registry_status", "registry_candidate", "registry_posterior", "processing_seconds",
              "possible_duplicate_of"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(data["rows"])
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=vtrace_observations.csv"})


@app.post("/api/jobs/{job_id}/review")
def api_review(job_id: str, req: ReviewRequest,
               x_vtrace_review_token: str | None = Header(default=None)):
    expected = os.getenv("VTRACE_REVIEW_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Authorized review is disabled. Configure VTRACE_REVIEW_TOKEN on the server.")
    if not x_vtrace_review_token or not hmac.compare_digest(x_vtrace_review_token, expected):
        raise HTTPException(403, "Review authorization failed.")
    _seed_precomputed_jobs()
    job = _jobs.get(job_id)
    if not job or job.get("status") != "done" or not job.get("result"):
        raise HTTPException(404, "Completed observation not found.")
    result = job["result"]
    review = result.setdefault("review", {"events": [], "machine_prediction": result.get("fusion", {}).get("text", "")})
    machine = review.get("machine_prediction", result.get("fusion", {}).get("text", ""))
    corrected = req.corrected_plate.upper().replace(" ", "").replace("-", "") if req.corrected_plate else None
    if req.action == "accepted" and not _valid_registration(machine):
        raise HTTPException(422, "The machine output is partial or invalid. Correct it to a full supported registration instead.")
    if corrected and not _valid_registration(corrected):
        raise HTTPException(422, "Corrected plate does not match a supported Indian registration format.")
    event = {"action": req.action, "machine_prediction": machine,
             "corrected_plate": corrected, "reviewer": req.reviewer.strip(),
             "note": req.note.strip(), "reviewed_at": _utc_now()}
    review.setdefault("events", []).append(event)
    review.update({"status": req.action, "corrected_plate": corrected,
                   "reviewer": event["reviewer"], "reviewed_at": event["reviewed_at"]})
    result_path = Path(job["job_dir"]) / "result.json"
    tmp = result_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    os.replace(tmp, result_path)
    html_str = report.render(result, Path(job["job_dir"]), embed=True)
    (Path(job["job_dir"]) / "report.html").write_text(html_str, encoding="utf-8")
    return {"job_id": job_id, "review": review}


@app.get("/api/registry")
def api_registry(q: str = "", colour: str = "", body_type: str = "", limit: int = 100):
    limit = max(1, min(limit, 500))
    if colour and colour not in COLOURS:
        raise HTTPException(422, "Unsupported vehicle colour.")
    if body_type and body_type not in BODY_TYPES:
        raise HTTPException(422, "Unsupported vehicle body type.")
    rows = get_registry()
    q = q.strip().upper()
    out = []
    for r in rows:
        if colour and r["colour"] != colour:
            continue
        if body_type and r["body_type"] != body_type:
            continue
        if q and not (
            q in r["reg_no"].upper()
            or q in r["make"].upper()
            or q in r["model"].upper()
            or q in r.get("owner", "").upper()
            or q in r.get("city", "").upper()
        ):
            continue
        out.append(r)
        if len(out) >= limit:
            break
    return {"total": len(rows), "matched": len(out), "rows": out}
