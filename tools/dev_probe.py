import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vtrace import matcher
from vtrace.pipeline import VideoAnalysis

reg = matcher.load_registry(ROOT / "data" / "registry.csv")
name = sys.argv[1]
t = time.time()
va = VideoAnalysis(ROOT / "test_clips" / f"{name}.mp4", ROOT / "scratch_jobs" / name, reg,
                   progress=lambda p, m: None)
r = va.analyse({})
print("tracks", r["tracks"][:3])
for f in r["frames"]:
    print(f["frame"], f["plate_source"], f["loc_score"], f["plate_box"], repr(f["text"]), f["mean_conf"], f["sharpness"], f["plate_height_px"], f["weight"])
print("FUSED", r["fusion"]["text"], [(p["best_guess"], p["conf"], p["n_frames"]) for p in r["fusion"]["positions"]])
print("ATTR", r["attributes"]["colour"], r["attributes"]["body_type"], r["attributes"]["make_model"])
m = r["match"]
print(m["status"], m["message"])
for c in m["candidates"]:
    print(c["rank"], c["vehicle"]["reg_no"], c["vehicle"]["colour"], c["vehicle"]["body_type"], c["confidence"], c["score"])
print("time", time.time() - t)
