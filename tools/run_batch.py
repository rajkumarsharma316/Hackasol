"""Batch-analyse every clip in test_clips/ and write per-clip HTML reports + a summary.

usage: python tools/run_batch.py [clip-name-substring ...]
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vtrace import matcher, report  # noqa: E402
from vtrace.pipeline import VideoAnalysis  # noqa: E402

registry = matcher.load_registry(ROOT / "data" / "registry.csv")
gt = json.loads((ROOT / "test_clips" / "ground_truth.json").read_text())
out_dir = ROOT / "reports"
out_dir.mkdir(exist_ok=True)
filters = sys.argv[1:]
rows = []
for clip in sorted((ROOT / "test_clips").glob("*.mp4")):
    if filters and not any(f in clip.stem for f in filters):
        continue
    t = time.time()
    job = out_dir / clip.stem
    va = VideoAnalysis(clip, job, registry)
    res = va.analyse({})
    html = report.render(res, job, embed=True)
    (out_dir / f"{clip.stem}.html").write_text(html, encoding="utf-8")
    g = gt.get(clip.stem, {})
    m = res["match"]
    top = m["candidates"][0] if m["candidates"] else None
    ok = (m["status"] == "match" and top and top["vehicle"]["reg_no"] == g.get("plate")) if g.get("in_registry") \
        else m["status"] == "no_confident_match"
    rows.append({"clip": clip.stem, "truth": g.get("plate"), "recovered": res["fusion"]["text"],
                 "colour": next(iter(res["attributes"]["colour"]), None),
                 "body": max(res["attributes"]["body_type"], key=res["attributes"]["body_type"].get),
                 "status": m["status"], "top": top["vehicle"]["reg_no"] if top else None,
                 "conf": top["confidence"] if top else None, "correct": bool(ok),
                 "secs": round(time.time() - t, 1)})
    print(json.dumps(rows[-1]))

if not filters:
    lines = ["# Batch evaluation on synthetic test clips", "",
             "| Clip | Ground truth | Recovered plate | Colour | Body | Decision | Top-1 | Confidence | Correct |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| [{r['clip']}]({r['clip']}.html) | {r['truth']} | `{r['recovered']}` | {r['colour']} | "
                     f"{r['body']} | {r['status']} | {r['top']} | {r['conf']}% | {'yes' if r['correct'] else 'NO'} |")
    acc = sum(r["correct"] for r in rows)
    lines += ["", f"**{acc}/{len(rows)} clips handled correctly** (correct = confident match to the true vehicle, "
              "or 'no confident match' for the unregistered vehicle)."]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
