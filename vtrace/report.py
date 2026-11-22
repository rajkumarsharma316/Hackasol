"""HTML analysis report (self-contained; print to PDF from the browser)."""
from __future__ import annotations

import base64
import datetime as dt
import html
from pathlib import Path

E = html.escape


def _img(job_dir: Path, rel: str, embed: bool, prefix: str = "", style: str = ""):
    p = Path(job_dir) / rel
    if embed and p.exists():
        mime = "image/png" if p.suffix == ".png" else "image/jpeg"
        src = f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    else:
        src = prefix + rel
    return f'<img src="{src}" style="{style}" loading="lazy">'


def _conf_colour(c):
    if c >= 0.8:
        return "#1b8a3a"
    if c >= 0.55:
        return "#b88a00"
    return "#b3261e"


def _bars(d, n=6):
    if not d:
        return "<i>n/a</i>"
    rows = []
    for k, v in list(d.items())[:n]:
        rows.append(f'<div class="bar"><span class="bl">{E(str(k))}</span><span class="bt"><span class="bf" '
                    f'style="width:{v * 100:.0f}%"></span></span><span class="bv">{v * 100:.0f}%</span></div>')
    return "".join(rows)


CSS = """
body{font-family:Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#f4f6f9;color:#1d2433}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 10px;border-bottom:2px solid #d9dee7;padding-bottom:4px}
.muted{color:#5b6475;font-size:13px}
.card{background:#fff;border:1px solid #dde2ea;border-radius:10px;padding:16px;margin:10px 0}
.banner{padding:14px 18px;border-radius:10px;font-weight:600;font-size:16px}
.ok{background:#e3f5e8;border:1px solid #87c99a;color:#145c2a}
.warn{background:#fff3e0;border:1px solid #f0b766;color:#7a4b00}
.grid{display:grid;grid-template-columns:1.3fr 1fr;gap:16px}
.plate{display:flex;gap:4px;flex-wrap:wrap;margin:8px 0}
.pc{width:38px;text-align:center;border:2px solid #222;border-radius:6px;background:#fffbe6;padding:4px 0}
.pc b{display:block;font-size:24px;font-family:Consolas,monospace}
.pc small{font-size:10px}
.pc.unk{background:#e8e8e8;border-style:dashed;color:#777}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid #e3e7ee;padding:6px;text-align:left;vertical-align:middle}
th{background:#f0f3f7}
.ch{display:inline-block;min-width:22px;text-align:center;margin:1px;padding:2px 3px;border-radius:4px;color:#fff;
font-family:Consolas,monospace;font-weight:700}
.ch sub{font-weight:400;font-size:9px;display:block}
.bar{display:flex;align-items:center;gap:6px;font-size:12px;margin:2px 0}
.bl{width:110px}.bt{flex:1;background:#e9edf3;height:10px;border-radius:5px;overflow:hidden}
.bf{display:block;height:100%;background:#3a6ff7}.bv{width:40px;text-align:right}
.cand{border-left:6px solid #3a6ff7}
.cand.top{border-left-color:#1b8a3a}
.rel-exact{background:#d8f2df}.rel-similar{background:#fff1c2}.rel-mismatch{background:#f9d4d2}
.rel-unknown{background:#ececec}.rel-not-visible{background:#f5f5f5;color:#999}.rel-extra,.rel-missed{background:#ffe0cc}
.llr-pos{color:#1b8a3a;font-weight:600}.llr-neg{color:#b3261e;font-weight:600}
.btn{background:#3a6ff7;color:#fff;border:0;border-radius:6px;padding:8px 14px;cursor:pointer}
@media print{.noprint{display:none}body{background:#fff}.card{break-inside:avoid}}
"""


def render(res: dict, job_dir, embed=True, prefix="") -> str:
    v, m, fu, at = res["video"], res["match"], res["fusion"], res["attributes"]
    I = lambda rel, style="": _img(job_dir, rel, embed, prefix, style)  # noqa: E731
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>VTrace report - {E(v['file'])}</title>"
           f"<style>{CSS}</style></head><body><div class='wrap'>"]
    out.append("<div class='noprint' style='float:right'><button class='btn' onclick='window.print()'>"
               "Print / Save as PDF</button></div>")
    out.append(f"<h1>Vehicle identification report</h1><div class='muted'>Clip <b>{E(v['file'])}</b> &middot; "
               f"{v['resolution']} &middot; {v['duration_s']} s @ {v['fps']} fps &middot; {v['sampled']} frames sampled "
               f"&middot; generated {dt.datetime.now():%Y-%m-%d %H:%M} &middot; analysis {res['elapsed_s']} s</div>")
    cls = "ok" if m["status"] == "match" else "warn"
    out.append(f"<div class='card banner {cls}'>{E(m['message'])}</div>")

    # ---- extracted info
    out.append("<h2>1. Extracted information</h2><div class='grid'><div class='card'>")
    out.append(I(res["keyframe"], "width:100%;border-radius:6px"))
    out.append(f"<div class='muted'>Key frame #{fu.get('reference_frame')} (reference reading). Orange = tracked "
               f"vehicle #{res['track']['id']}, green = plate.</div></div><div class='card'>")
    out.append("<b>Recovered plate (multi-frame)</b><div class='plate'>")
    for p in fu["positions"]:
        alts = ", ".join(f"{c} {pp * 100:.0f}%" for c, pp in p["alts"])
        if p["known"]:
            out.append(f"<div class='pc' title='{E(alts)}'><b>{E(p['char'])}</b><small style='color:"
                       f"{_conf_colour(p['conf'])}'>{p['conf'] * 100:.0f}%</small></div>")
        else:
            out.append(f"<div class='pc unk' title='candidates: {E(alts) or 'none'}'><b>?</b><small>unknown</small></div>")
    if not fu["positions"]:
        out.append("<i>No plate characters could be recovered.</i>")
    out.append("</div>")
    out.append(f"<div class='muted'>Pattern: <code>{E(fu['text'] or '-')}</code> &middot; fused from {fu['n_frames']} "
               "frames. '?' = unreadable, never guessed.</div>")
    out.append("<p><b>Colour</b>" + (f" <i>(investigator override: {E(at['colour_hint'])})</i>" if at.get("colour_hint") else "")
               + _bars(at["colour"]) + "</p>")
    out.append("<p><b>Body type</b>" + (f" <i>(investigator override: {E(at['body_hint'])})</i>" if at.get("body_hint") else "")
               + _bars(at["body_type"]) + "</p>")
    out.append("<p><b>Make / model (zero-shot estimate)</b>" + _bars(at.get("make_model") or {}, 3) + "</p>")
    out.append("</div></div>")

    # ---- frames
    used = [f for f in res["frames"] if f.get("used_in_fusion")]
    other = [f for f in res["frames"] if not f.get("used_in_fusion")]
    out.append(f"<h2>2. Frames used and characters recovered per frame</h2><div class='card'>"
               f"<div class='muted'>{len(used)} frames used for fusion, {len(other)} examined but not used. Quality "
               "weight = OCR confidence&sup2; &times; sharpness &times; plate size.</div><table><tr><th>Frame</th><th>Time</th>"
               "<th>Plate crop</th><th>Characters (confidence)</th><th>Weight</th><th>Notes</th></tr>")
    for f in used + other[:12]:
        chars = "".join(
            f"<span class='ch' style='background:{_conf_colour(c['conf']) if c['char'] != '?' else '#888'}' "
            f"title='{E(', '.join(a + ' ' + str(round(p * 100)) + '%' for a, p in c['alts']))}'>{E(c['char'])}"
            f"<sub>{c['conf'] * 100:.0f}%</sub></span>" for c in f["chars"])
        note = "used" if f.get("used_in_fusion") else E(f.get("excluded") or "")
        out.append(f"<tr><td>#{f['frame']}<br><span class='muted'>src {f['source_frame']}</span></td><td>{f['time_s']} s</td>"
                   f"<td>{I(f['plate_img'], 'height:34px;max-width:220px')}</td><td>{chars or '<i>none</i>'}</td>"
                   f"<td>{f['weight']:.3f}</td><td>{note}<br><span class='muted'>{E(f['plate_source'])}, "
                   f"{f['plate_height_px']}px, sharp {f['sharpness']:.0f}</span></td></tr>")
    out.append("</table></div>")

    # ---- fusion per position
    out.append("<h2>3. Multi-frame fusion per plate position</h2><div class='card'><table><tr><th>Pos</th><th>Result</th>"
               "<th>Confidence</th><th>Frames seeing it</th><th>Top alternatives</th><th>Per-frame votes</th></tr>")
    for p in fu["positions"]:
        votes = " ".join(f"{E(x['char'])}<sub>{x['conf'] * 100:.0f}</sub>" for x in p["votes"][:15])
        res_txt = f"<b>{E(p['char'])}</b>" if p["known"] else "<b>?</b> unknown"
        out.append(f"<tr><td>{p['idx'] + 1}</td><td>{res_txt}</td><td>{p['conf'] * 100:.0f}%</td><td>{p['n_frames']}</td>"
                   f"<td>{E(', '.join(f'{c} {pp * 100:.0f}%' for c, pp in p['alts']))}</td><td style='font-family:Consolas'>{votes}</td></tr>")
    out.append("</table><div class='muted'>A position is reported only if its pooled probability &ge; 55%, it leads the "
               "runner-up by &ge; 20 points and at least 2 frames observed it; otherwise it is marked unknown.</div></div>")

    # ---- candidates
    out.append(f"<h2>4. Ranked shortlist (registry of {m['n_registry']} vehicles)</h2>")
    out.append(f"<div class='muted'>Probability that the vehicle is not in the registry / no entry fits: "
               f"<b>{m['p_not_in_registry']}%</b>. {m['n_plausible']} registry vehicles have net positive evidence.</div>")
    for c in m["candidates"]:
        veh, sc = c["vehicle"], c["score"]
        top = c["rank"] == 1 and m["status"] == "match"
        out.append(f"<div class='card cand {'top' if top else ''}'><div style='display:flex;justify-content:space-between'>"
                   f"<div><b style='font-size:18px'>#{c['rank']} &nbsp;{E(veh['reg_no'])}</b> &nbsp; {E(veh['make'])} "
                   f"{E(veh['model'])} &middot; {E(veh['colour'])} &middot; {E(veh['body_type'])} &middot; {E(str(veh.get('year', '')))}"
                   f"<div class='muted'>Owner {E(veh.get('owner', ''))}, {E(veh.get('city', ''))} &middot; status: "
                   f"{E(veh.get('status', ''))}</div></div><div style='text-align:right'><div style='font-size:22px;"
                   f"font-weight:700'>{c['confidence']}%</div><div class='muted'>posterior confidence</div></div></div>")
        out.append("<div class='muted'>Evidence (log-likelihood ratio, + supports / &minus; opposes): " + " &middot; ".join(
            f"{k.replace('_llr', '')} <span class='{'llr-pos' if val > 0 else 'llr-neg' if val < 0 else ''}'>{val:+.2f}</span>"
            for k, val in sc.items()) + "</div><ul>")
        for r in c["reasons"]:
            cls_ = "llr-pos" if r["llr"] > 0 else "llr-neg" if r["llr"] < 0 else ""
            out.append(f"<li><span class='{cls_}'>[{r['llr']:+.2f}]</span> {E(r['text'])}</li>")
        out.append("</ul>")
        for pa in c["plate_alignment"]:
            out.append(f"<table style='width:auto'><tr><th>{E(pa['source'])}</th>" + "".join(
                f"<td class='rel-{x['rel']}' style='text-align:center'>{E(x['obs'] or '&middot;') if x['obs'] else '&middot;'}</td>"
                for x in pa["rows"]) + "</tr><tr><th>registry</th>" + "".join(
                f"<td class='rel-{x['rel']}' style='text-align:center'><b>{E(x['reg'] or '-')}</b></td>" for x in pa["rows"])
                + "</tr></table>")
        out.append("</div>")
    if not m["candidates"]:
        out.append("<div class='card'>Registry empty.</div>")

    out.append("<h2>5. Method and limitations</h2><div class='card muted'>"
               "<p><b>Pipeline.</b> Frames are sampled evenly; vehicles detected with YOLO11 and tracked by IoU; the plate is "
               "localised with a CRAFT text detector plus a plate-colour heuristic and a temporal-consistency filter; each plate "
               "crop is read by a CRNN recogniser whose full per-character CTC probabilities are kept; readings are aligned "
               "across frames and pooled with quality weights. Colour comes from k-means clustering of the vehicle body; body "
               "type fuses YOLO class votes with CLIP zero-shot classification; make/model is a CLIP zero-shot estimate.</p>"
               "<p><b>Matching (TRACE).</b> Every recovered character, colour and body type becomes a likelihood ratio against "
               "a random vehicle using visual-confusion models (8&harr;B, 0&harr;O&harr;D, 5&harr;S, white&harr;silver, car&harr;SUV ...). "
               "Plates are aligned with semi-global dynamic programming so partial and shifted reads work. Unknown characters "
               "are neutral (LR = 1). Scores are converted to posteriors including a 'not in registry' hypothesis; a match "
               "is declared only when the top posterior &ge; 50% and the plate evidence alone is substantial.</p>"
               "<p><b>Limitations.</b> Posteriors assume the sample registry is the population; colour is affected by lighting "
               "and casts; make/model is indicative only. Results are investigative leads and must be verified by an officer.</p>"
               "</div></div></body></html>")
    return "".join(out)
