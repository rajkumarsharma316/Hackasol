"""FastAPI web application for VTrace - Vehicle Identification & Traceability from CCTV Footage."""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vtrace import matcher, reid, report
from vtrace.pipeline import VideoAnalysis
from vtrace.vocab import BODY_TYPES, COLOURS

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

_registry_lock = threading.Lock()
_registry: list = matcher.load_registry(DATA_DIR / "registry.csv")
_jobs: dict[str, dict] = {}
_va_cache: dict[str, VideoAnalysis] = {}


def get_registry() -> list:
    with _registry_lock:
        return list(_registry)


def _seed_precomputed_jobs():
    """Preload any batch-analysed test clip directories from reports/ into _jobs so they open instantly."""
    gt_path = TEST_CLIPS_DIR / "ground_truth.json"
    gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}
    for clip in sorted(TEST_CLIPS_DIR.glob("*.mp4")):
        rdir = REPORTS_DIR / clip.stem
        res_file = rdir / "result.json"
        if res_file.exists():
            jid = f"demo_{clip.stem}"
            res = json.loads(res_file.read_text(encoding="utf-8"))
            _jobs[jid] = {
                "job_id": jid,
                "status": "done",
                "progress": 100,
                "message": "Ready",
                "video_path": str(clip),
                "job_dir": str(rdir),
                "filename": clip.name,
                "ground_truth": gt.get(clip.stem),
                "result": res,
            }


_seed_precomputed_jobs()


class ReanalyseRequest(BaseModel):
    plate_pattern: str | None = None
    colour: str | None = None
    body_type: str | None = None
    track_id: int | None = None
    ignore_video_plate: bool = False
    use_make_model: bool = True
    roi: dict | None = None


class QuickMatchRequest(BaseModel):
    plate_pattern: str = ""
    colour: str | None = None
    body_type: str | None = None
    make_model: str | None = None


class CompareRequest(BaseModel):
    job_a: str
    job_b: str


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
            va.progress = _prog
            va.registry = get_registry()
        res = va.analyse(hints)
        html_str = report.render(res, Path(job["job_dir"]), embed=True)
        (Path(job["job_dir"]) / "report.html").write_text(html_str, encoding="utf-8")
        job["result"] = res
        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Analysis complete"
    except Exception as e:
        job["status"] = "error"
        job["message"] = f"Error: {e}"


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
            }
            for j in _jobs.values()
            if j["status"] == "done"
        ],
    }


@app.get("/api/test-clips/{clip_id}/preview")
def api_clip_preview(clip_id: str):
    p = TEST_CLIPS_DIR / f"{clip_id}_preview.jpg"
    if not p.exists():
        raise HTTPException(404, "Preview not found")
    return FileResponse(p)


@app.get("/api/test-clips/{clip_id}/video")
def api_clip_video(clip_id: str):
    p = TEST_CLIPS_DIR / f"{clip_id}.mp4"
    if not p.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(p, media_type="video/mp4")


@app.post("/api/analyse/sample/{clip_id}")
def api_analyse_sample(clip_id: str, force: bool = False):
    _seed_precomputed_jobs()
    jid = f"demo_{clip_id}"
    if not force and jid in _jobs and _jobs[jid]["status"] == "done":
        return {"job_id": jid, "cached": True}
    vpath = TEST_CLIPS_DIR / f"{clip_id}.mp4"
    if not vpath.exists():
        raise HTTPException(404, "Test clip not found")
    jdir = REPORTS_DIR / clip_id
    jdir.mkdir(parents=True, exist_ok=True)
    _jobs[jid] = {
        "job_id": jid,
        "status": "running",
        "progress": 1,
        "message": "Starting analysis...",
        "video_path": str(vpath),
        "job_dir": str(jdir),
        "filename": vpath.name,
        "result": None,
    }
    threading.Thread(target=_run_analysis_bg, args=(jid, {}), daemon=True).start()
    return {"job_id": jid, "cached": False}


@app.post("/api/analyse/upload")
async def api_analyse_upload(
    file: UploadFile = File(...),
    plate_pattern: str = Form(""),
    colour: str = Form(""),
    body_type: str = Form(""),
):
    jid = uuid.uuid4().hex[:10]
    jdir = JOBS_DIR / jid
    jdir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload.mp4").name
    vpath = jdir / safe_name
    with open(vpath, "wb") as f:
        shutil.copyfileobj(file.file, f)
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
        "result": None,
    }
    threading.Thread(target=_run_analysis_bg, args=(jid, hints), daemon=True).start()
    return {"job_id": jid}


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str):
    _seed_precomputed_jobs()
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return _jobs[job_id]


@app.post("/api/jobs/{job_id}/reanalyse")
def api_reanalyse(job_id: str, req: ReanalyseRequest):
    _seed_precomputed_jobs()
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
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
        prev_res["match"] = matcher.match(ev, get_registry())
        jdir = Path(job["job_dir"])
        (jdir / "result.json").write_text(json.dumps(prev_res, indent=1), encoding="utf-8")
        (jdir / "report.html").write_text(report.render(prev_res, jdir, embed=True), encoding="utf-8")
        return {"job_id": job_id, "instant": True, "result": prev_res}

    job["status"] = "running"
    job["progress"] = 10
    job["message"] = "Re-running pipeline with updated investigator hints..."
    threading.Thread(target=_run_analysis_bg, args=(job_id, hints), daemon=True).start()
    return {"job_id": job_id, "instant": False}


@app.get("/api/jobs/{job_id}/asset/{rel_path:path}")
def api_job_asset(job_id: str, rel_path: str):
    _seed_precomputed_jobs()
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    base = Path(_jobs[job_id]["job_dir"]).resolve()
    target = (base / rel_path).resolve()
    if not str(target).startswith(str(base)) or not target.exists():
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
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/report", response_class=HTMLResponse)
def api_job_report(job_id: str, download: bool = False):
    _seed_precomputed_jobs()
    if job_id not in _jobs or not _jobs[job_id].get("result"):
        raise HTTPException(404, "Result not ready")
    job = _jobs[job_id]
    html_str = report.render(job["result"], Path(job["job_dir"]), embed=True)
    headers = {}
    if download:
        fname = f"VTrace_Report_{Path(job['filename']).stem}.html"
        headers["Content-Disposition"] = f'attachment; filename="{fname}"'
    return HTMLResponse(content=html_str, headers=headers)


@app.post("/api/match")
def api_quick_match(req: QuickMatchRequest):
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


@app.get("/api/registry")
def api_registry(q: str = "", colour: str = "", body_type: str = "", limit: int = 100):
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
