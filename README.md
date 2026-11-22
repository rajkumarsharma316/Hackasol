# VTrace — Vehicle Identification & Traceability from CCTV Footage

Turn blurry CCTV footage and a partial number plate into a ranked, explainable shortlist of possible vehicles from a sample vehicle registry.

---

## Key Features

1. **Multi-Frame Plate Recovery (`vtrace/plate.py`, `vtrace/ocr.py`, `vtrace/fusion.py`)**
   - Detects and tracks vehicles across sampled frames using **YOLO11** (`yolo11n`) + IoU tracking.
   - Localises single-line and two-line number plates at arbitrary camera tilt angles using oriented 4-point quadrilaterals from **CRAFT** (`horizontal_list` + rotated `free_list`) fused with an HSV plate-colour / vertical-stroke heuristic and temporal median consistency.
   - Horizontally collinear text boxes separated by mud/occlusion are automatically merged into a single spanning plate quad; vertically stacked boxes on commercial/two-wheeler two-line plates are paired as `[top_line, bottom_line]`.
   - Extracts full 36-symbol CTC posterior probability distributions per character from the **CRNN** recogniser (`vtrace/ocr.py`), aligns per-frame readings via **Needleman–Wunsch** sequence alignment (`vtrace/fusion.py`), and pools character distributions across frames weighted by $\text{conf}^2 \times \text{sharpness} \times \text{plate\_size}$.
   - **Strict Unknown Policy**: Any character slot covered by mud/occlusion (detected via inter-character pitch gap $> 1.85\times$ median pitch) or failing the multi-frame consensus threshold ($P < 55\%$, margin $< 18\%$, or $< 2$ frames) is marked **`?` (unknown)** — never invented.

2. **Vehicle Attribute Extraction (`vtrace/attributes.py`)**
   - **Colour**: LAB $k$-means clustering on the vehicle body panel region (suppressing road background and dark tire/window glass clusters) fused with **CLIP ViT-B/32** zero-shot vehicle colour probabilities across 12 standard colours.
   - **Body Type**: Product-of-experts combining YOLO class votes with **CLIP ViT-B/32** zero-shot classification across 7 Indian road body types (`car`, `suv`, `van`, `two-wheeler`, `auto-rickshaw`, `truck`, `bus`).
   - **Make & Model Recognition (Bonus)**: Zero-shot **CLIP ViT-B/32** distribution over registry makes and models conditioned on body type.

3. **TRACE Vehicle Matching & Ranking Algorithm (`vtrace/matcher.py`, `vtrace/vocab.py`)**
   - Converts every piece of evidence into a **log-likelihood ratio (LLR)** against a random baseline vehicle:
     $$\text{LLR}(e \mid v) = \ln \frac{P(e \mid v)}{P(e \mid H_0)}$$
   - **Semi-Global Dynamic Programming Plate Alignment**: Aligns partial/cropped plate reads against full registry plates with confusion-matrix partial credit for visually similar glyphs (`8↔B`, `0↔O↔D↔Q`, `5↔S`, `1↔I↔7↔T`, `2↔Z`, `6↔G`), small end-skip penalty (`0.35`) for characters outside the visible window, and exact neutrality ($\text{LR} = 1, \text{LLR} = 0$) for unknown (`?`) characters.
   - **Soft Attribute Confusion Models**: Compatible colours (`white↔silver`, `red↔maroon`, `orange↔yellow`) and body types (`car↔suv`, `suv↔van`) earn partial positive LLRs instead of brittle hard filters.
   - **Calibrated Posterior & Honest No-Match Handling**: Combines plate, colour, body-type, and make/model LLRs with an explicit **"not in registry / no fit" null hypothesis** ($P(H_{\text{null}}) = 0.20$). Declares **`no_confident_match`** whenever no single candidate reaches $\ge 50\%$ posterior confidence with strong plate support ($\text{plate\_llr} \ge 4.0$).

4. **Cross-Camera Vehicle Re-Identification & Multi-Clip Fusion (Bonus — `vtrace/reid.py`)**
   - Compares two CCTV clips from different cameras using symmetric semi-global partial-plate alignment, **CLIP ViT-B/32** visual appearance cosine similarity, colour compatibility LLR, and body-type compatibility LLR.
   - Produces a **Same-Vehicle Confidence Score** and a **Joint Registry Shortlist** fusing evidence from both cameras.

5. **Investigator Web Interface & Self-Contained HTML/PDF Reports (`app.py`, `static/index.html`, `vtrace/report.py`)**
   - Upload any video clip or inspect the 8 synthetic benchmark clips.
   - Interactive frame scrubber, manual plate ROI drawing canvas, instant witness/investigator hint re-matching, per-frame OCR evidence table, position-by-position fusion breakdown, and one-click printable HTML/PDF reports.

---

## Project Structure

```text
Hackasol/
├── app.py                  # FastAPI server & REST endpoints
├── static/index.html       # Investigator web application UI
├── vtrace/
│   ├── vocab.py            # Character, colour & body-type confusion matrices
│   ├── matcher.py          # TRACE Bayesian LLR & semi-global DP plate matcher
│   ├── engines.py          # Lazy loader for YOLO11n, EasyOCR CRNN/CRAFT, CLIP ViT-B/32
│   ├── plate.py            # Oriented 4-point quad plate localisation & grouping
│   ├── ocr.py              # CTC probability distribution extraction & gap detection
│   ├── fusion.py           # Needleman-Wunsch multi-frame plate alignment & pooling
│   ├── attributes.py       # Vehicle colour, body type & make/model extraction
│   ├── pipeline.py         # End-to-end CCTV clip analysis pipeline
│   ├── reid.py             # Cross-camera vehicle tracking & multi-clip fusion
│   └── report.py           # Self-contained HTML/PDF analysis report generator
├── data/registry.csv       # 600-vehicle sample Indian vehicle registry
├── test_clips/             # 8 synthetic CCTV benchmark clips + ground_truth.json
├── reports/                # Generated self-contained HTML reports for all 8 test clips
└── tools/
    ├── setup_models.py     # Downloads pretrained weights into models/
    ├── make_registry.py    # Generates data/registry.csv
    ├── make_test_clips.py  # Synthesises the 8 degraded CCTV test clips
    └── run_batch.py        # Batch-evaluates all test clips & generates reports/
```

---

## Quick Start

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Download pretrained weights once (saved in models/)
python tools/setup_models.py --clip

# 3. Start the Investigator Web Application
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.
