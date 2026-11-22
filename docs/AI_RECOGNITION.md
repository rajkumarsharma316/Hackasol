# Recognition pipeline notes

## Pipeline and failure analysis

Video frames are sampled up to 60 per clip. Ultralytics YOLO11n detects COCO
vehicle classes and a lightweight association tracker selects a vehicle track. Plate crops
were previously proposed using EasyOCR/CRAFT text detections plus plate-like
colour and shape heuristics. That makes bright lamps, badges, and trim plausible
false plate candidates. Crops are rectified, read with EasyOCR's English CRNN,
then fused over frames and matched to registry evidence. The vehicle tracker now
associates detections globally against constant-velocity predicted boxes and
allows a short missed-detection gap, reducing fragmentation from occasional
missed detections without adding a tracking dependency.

Inspection of the preserved baseline reports (2/8 benchmark decisions correct)
showed localization errors on at least the auto-rickshaw and camera-B hatchback:
the selected regions were a tail lamp rather than the plate. Other hard cases
include mud/partial occlusion, low light, motion blur, and small plate crops.
The vehicle tracker itself was reasonably persistent on most clips; plate
localization and OCR crop quality remain the main observed bottlenecks. The
lightweight tracker may still fragment fast motion, long occlusions, or camera
cuts.

## Optional pretrained plate detector

When `models/plate_detector/best.pt` exists, the pipeline uses the specialized
detector on vehicle crops in batches. It filters implausible geometry, uses
relative plate position across frames to reject outliers, and weights OCR frame
votes by localization confidence and crop quality. It avoids CRAFT detection
weights on this path while retaining EasyOCR's CRNN recognizer. Without the
optional file, the prior CRAFT/colour proposal path remains available.

Install the optional weights with:

```sh
python tools/setup_plate_detector.py
```

The Hugging Face model card declares MIT for the model and reports a 22.5 MB
YOLOv8s detector trained on Indian and European plate imagery. The card's
reported evaluation is uploader-provided and small; it is not independent
validation on this project's videos. Its source datasets are Roboflow Universe
NIVU Indian plates and European License Plates, both declared CC BY 4.0 by their
dataset pages. No dataset was downloaded or used for additional training here.
Sources: [model card](https://huggingface.co/Babblu2821/alpr-plate-detector),
[NIVU dataset](https://universe.roboflow.com/nivu/indian-license-plate-knte7),
[European dataset](https://universe.roboflow.com/e-hh49k/european-license-plates-tjviy).

The selected checkpoint is a YOLOv8s single-class detector with 640-pixel
inference inputs and axis-aligned boxes. The local weight file is 22,524,970 bytes
(about 21.5 MiB). The model card's reported T4/M4 speed and held-out test results
are uploader-provided, and its own limitations note that oblique plates remain
axis-aligned and difficult night/blur cases are not characterized. The underlying
Ultralytics inference package is separately AGPL-3.0 or Enterprise licensed; the
checkpoint's MIT declaration does not change the dependency license. Review the
Ultralytics terms before a proprietary deployment.

### Other researched candidates

- [Awiros ANPR OCR](https://huggingface.co/Awiros/anpr-ocr) declares Apache-2.0,
  uses PP-OCRv5 / SVTR-HGNet with a 3x48x320 input, and provides Safetensors
  weights (about 149 MB). The card reports 37.3M parameters and NVIDIA RTX 3090
  inference results, which are not CPU benchmarks. Its card says its data
  expansion uses synthetic synthesis and pseudo-labeling. I did not integrate it
  because that data provenance conflicts with the requirement to avoid synthetic
  datasets, and it would add Paddle runtime/dependency weight.
- [Intel License Plate Recognition](https://huggingface.co/Intel/license-plate-recognition)
  declares MIT and combines an oriented YOLOv8 plate detector with PP-OCRv4 in an
  OpenVINO pipeline (CPU/GPU/NPU). Its card says its detector is trained primarily
  on European and US plates and recommends fine-tuning for other regions, which
  is outside this task's constraints; no weights were downloaded.
- An Indian-specific [TrOCR-small model](https://huggingface.co/subash1652007/indian-plate-ocr)
  is publicly available at about 257 MB for the repository, including its plate
  detector. Its model card reports 67% exact match on 190 held-out crops, but
  this is the best of 15 checkpoints selected on that same small validation set;
  the card also says blur, low resolution, occlusion, and two-line plates are
  weak cases. Its weights are explicitly non-commercial research/education only
  and derive from a Kaggle dataset the card lists as CC BY-NC-ND. I did not
  download or integrate it: that restriction may not fit deployment, and the
  heavyweight sequence model cannot be benchmarked here because `transformers`
  is not installed. [Model card](https://huggingface.co/subash1652007/indian-plate-ocr)
  | [pipeline and limitations](https://github.com/subash9940/indian-number-plate-recognition).
- [PP-OCRv5 English mobile recognition](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md)
  has an official 7.5 MB model and general English text benchmarks, but it is
  not trained specifically for Indian plates; general OCR metrics cannot
  establish an ANPR improvement. I did not add PaddleOCR or another runtime.
- The [Indian Licence Plate Dataset in the Wild paper](https://arxiv.org/abs/2111.06054)
  describes 16,192 varied-view images, plate corner annotations, and plate text.
  It is promising for future evaluation, but this project's benchmark lacks
  localization labels and the paper alone does not settle current dataset
  redistribution terms. No dataset was downloaded.
- EasyOCR's Python package declares Apache-2.0, but its downloaded model files
  have separate distribution terms. The package's public model metadata does not
  make every weight's redistribution rights equally clear. Existing EasyOCR
  weights are used as-is here and are not redistributed by this project.

The existing YOLO vehicle detector and EasyOCR English recognizer remain
pretrained models. No training, fine-tuning, or synthetic dataset generation
was performed. The project benchmark clips are preserved and used only for
evaluation. The custom plate detector localizes; it does not itself recognize
characters.

## Phase I experiments

`tools/compare_ocr_preprocessing.py` compares grayscale, CLAHE, sharpened, and
the current confidence-selected preprocessing on every selected plate crop from
a saved pipeline run. On the 100 selected crops in `jobs/evaluations/v2-final`,
crop-level normalized edit similarity was 6.95% grayscale, 6.91% CLAHE,
7.95% sharpened, and 7.92% for confidence selection. Reference-normalized
character error rates were 97.8%, 98.3%, 98.4%, and 96.5%, respectively; exact
crop matches were 0/100 for every variant. Confidence selection takes about
0.030 seconds per crop for three hypotheses versus about 0.010 seconds for one
fixed variant. The metrics do not agree on the best option, so this experiment
does not justify changing production preprocessing. These crop-level results
compare against the full plate string (not independent samples); crops can be
mislocalized.

A temporal-diversity ablation selected at most one crop in nearby sampled-frame
windows to reduce duplicate votes. On the full eight-clip benchmark this gave
1/8 correct decisions, 0/8 exact plate reads, and 6.10% normalized edit similarity,
below the recorded updated run's 1/8, 0/8, and 9.41%. The change was reverted.
Frame-level OCR errors and localization failures dominate this set. The ablation
output is preserved in the ignored `jobs/evaluations/phase1-temporal/` folder.

The benchmark evaluator now reports both normalized edit similarity and
reference-normalized character error rate so insertions in noisy OCR strings
cannot be hidden by normalizing each example against the longer string.

No alternative detector or recognizer was integrated in this phase. The
available Indian-specific OCR candidates either use synthetic-data-derived
training (Awiros), have a restrictive non-commercial license and a large
distribution (TrOCR), or are general scene-text models rather than plate OCR.
The existing optional detector remains the only compatible dedicated Indian
plate detector already installed in this workspace. Vehicle detection input
size and architecture were not changed: the eight-clip set has no annotated
vehicle boxes to compare detector quality, and changing YOLO model size without
that evidence would be guesswork.

Reproduce the crop preprocessing experiment with:

```sh
python tools/compare_ocr_preprocessing.py \
  --results-dir jobs/evaluations/v2-final \
  --output jobs/evaluations/v2-final/ocr_preprocessing.json
```

## Phase II inference audit

Visual review of the highest-weight plate crops saved by the pipeline shows
that localization is usually finding a plate-shaped region in these eight
clips: the two-line auto-rickshaw and truck plates are present, the muddy SUV
crop contains the occluding mud, the scooter crop is oblique/blurred, and the
second-camera hatchback crop is very low resolution. This is a qualitative
inspection, not a detection metric; there are no box labels to calculate IoU,
precision, or recall. It points to recognition quality and difficult image
quality as the dominant measured failure, while incorrect crops remain possible.

| Clip | Best saved crop observation | Recognition result | Main visible failure factor |
|---|---|---|---|
| clip01 | Plate localized and characters visually legible | `?IXZOE45Z?` | CRNN glyph confusion on plate font despite a clear crop |
| clip02 | Plate localized; mud covers central glyphs | `P?OS??1?2` | Occlusion removes character evidence |
| clip03 | Plate localized, dark/noisy view | `???` | Low light/noise leaves no fused characters |
| clip04 | Two-line yellow plate localized | `?OLEX?ZEZ` | Two-line row ordering / non-plate OCR errors |
| clip05 | Oblique scooter plate crop, some blur | `EEL` | Motion blur, skew, and track coverage only 65% |
| clip06 | Two-line truck plate crop visibly present | `L??` | Recognition failure on commercial two-line text |
| clip07 | Plate crop legible | `?ZZ` | Partial OCR, though the unregistered abstention is correct |
| clip08 | Plate localized but very small/soft | `?TT` | Second-camera blur, low resolution, and cut-off tail |

The direct EasyOCR inference path had two implementation mismatches. It resized
strips but did not pad them to EasyOCR's expected right-padded width, changing
the CTC time grid. It also dropped punctuation/space class probability and
renormalized the remaining classes, forcing unsupported-symbol peaks into
letters or digits. The path now uses the recognizer's LANCZOS resize and
right-edge padding, and transfers unsupported-symbol probability to blank
evidence rather than fabricating a character. Unit tests cover both behaviors.

This change reduced edit errors from 77 to 74 over 80 reference characters on
the eight clips (CER 96.25% to 92.50%); exact OCR stayed **0/8** and automatic
decisions stayed **1/8**. The result is a small-sample improvement in CER, not
statistically meaningful evidence of general accuracy improvement. Normalized
edit similarity moved from **9.41% down to 7.50%** because that metric divides
each distance by the longer string: the updated predictions are shorter, so
the denominator fell from 85 to 80 while edit errors fell by only three. CER
uses the fixed 80 reference characters and shows that error reduction directly.

The evaluator now normalizes only case, spaces, and hyphens, preserves `?` as
an error, treats empty predictions as full-reference errors, excludes and
counts missing labels, checks the baseline/current clip sets, and records a
ground-truth SHA-256. It reports micro and macro CER and normalized edit
similarity. The previous 8.43% vs 9.41% similarity values and 95.00% vs 96.25%
CER values are reproducible with the old run outputs; the mismatch came from
the metrics' distinct denominators, not different benchmark labels.

### Phase II benchmark

The original benchmark is unchanged. Full results are saved in the ignored
`jobs/evaluations/phase2-final/evaluation.json` directory.

| Metric | Original `reports/` | Previous updated run | Phase II OCR fix |
|---|---:|---:|---:|
| Exact plate OCR | 0/8 | 0/8 | 0/8 |
| CER (micro) | 95.00% | 96.25% | 92.50% |
| Normalized edit similarity (micro) | 8.43% | 9.41% | 7.50% |
| Correct automatic decisions | 2/8 | 1/8 | 1/8 |
| Automatic match rate | 1/8 | 0/8 | 0/8 |
| Abstention rate (includes review) | 7/8 | 8/8 | 8/8 |
| Registry false positives | 0/1 | 0/1 | 0/1 |
| Registry false negatives | 6/7 | 7/7 | 7/7 |
| Mean time per clip | 12.11s* | 6.53s | 7.17s |
| Peak process RSS | unavailable | unavailable | 2.30 GB** |
| Plate detection precision / recall | unavailable | unavailable | unavailable |

*The original mean includes a 44.3-second cold-start first clip and is not a
controlled latency baseline. **macOS `ru_maxrss` reports bytes; the measured
process peak is 2,304,475,136 bytes (about 2.15 GiB). It includes Python,
PyTorch, OpenCV, and loaded models, not just model weights.

Only the registered/unregistered result on clip07 is automatically correct in
Phase II. The system did not improve exact recognition or automatic decision
accuracy. It made three fewer character edits on this eight-clip set.

After fixing OCR preprocessing, the 95 frame crops selected in the Phase II run
were rechecked with the same four preprocessing candidates. None produced an
exact crop match. Sharpening was best at 9.70% normalized edit similarity and
91.16% CER, versus 9.01% and 91.47% for the current confidence-selected
variant. The 0.31 percentage-point CER gap on repeated crops from eight videos
is too small and correlated to establish a general benefit; production
preprocessing remains unchanged.

Reproduce this crop ablation with:

```sh
python tools/compare_ocr_preprocessing.py \
  --results-dir jobs/evaluations/phase2-final \
  --output jobs/evaluations/phase2-final/ocr_preprocessing.json
```

### Dataset and alternative model audit

- [DataCluster Labs' Indian Number Plates sample](https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset)
  matches the evaluation need: the owner describes about 200 real images with
  Pascal VOC plate boxes and plate text, collected from mobile captures with
  varied locations and conditions. The sample is 156 MB and explicitly
  `CC BY-NC-ND 4.0`: the card permits free evaluation/non-commercial research,
  prohibits commercial use and redistribution/derivatives. The owner says the
  images are theirs and not scraped. I attempted a local temporary download for
  a non-commercial evaluation, but DNS/network access failed; no dataset was
  downloaded. Confirm commercial compatibility before using this data for any
  deployment.
- The [Indian Licence Plate Dataset in the Wild paper](https://arxiv.org/abs/2111.06054)
  describes 16,192 images, but its [maintainer says the images cannot be made
  public due to legalities](https://github.com/sanchit2843/Indian_LPR); I
  excluded it. The Kaggle link named by the maintainer has no readable license
  metadata in this environment, so it was not downloaded.
- The [3,742-image Hugging Face detection dataset](https://huggingface.co/datasets/thundarstrom/indian-license-plate-detection)
  declares CC BY 4.0 but the repository reports only 2.41 kB total and the
  dataset viewer has no supported data files. Its card lists merged source
  sets without enough source-specific terms to verify reuse. It was excluded.
- A separate [Indian PARSeq OCR dataset page](https://huggingface.co/datasets/thundarstrom/indian-anpr-ocr-corpus)
  declares CC BY 4.0 but states that the repository is currently empty. It was
  not usable as benchmark data or a model source.
- The [Indian TrOCR weights](https://huggingface.co/subash1652007/indian-plate-ocr)
  remain non-commercial research/education-only, about 257 MB, and report 67%
  exact on a small validation set chosen from 15 checkpoints. I did not
  integrate/download them: project deployment intent is unspecified,
  `transformers` is missing, and network access failed. Awiros ANPR OCR uses
  synthetic/pseudo-labeled expansion and was excluded under the no-synthetic
  data rule. PP-OCRv5 is general English text OCR, not a proven Indian plate
  recognizer. No alternative was tested on the same crops.

The current model artifacts were already present: YOLO11n vehicle detector
(5,613,764 bytes), Babblu YOLOv8s plate detector (22,524,970 bytes), and
EasyOCR English CRNN weights (15,143,997 bytes). This run used CPU; CUDA was
unavailable and PyTorch reported MPS unavailable on this arm64 macOS host.
The project does not currently enable MPS in its engine selection, and no MPS
performance claim is made. The model files and Ultralytics/EasyOCR licensing
notes above still apply.

Application startup was not tested because FastAPI, Uvicorn, and
`python-multipart` are absent from the environment, although listed in
`requirements.txt`. The inference benchmark and tests ran independently of the
web server.

Run a fresh, non-destructive benchmark to a new output directory:

```sh
python tools/evaluate_benchmark.py --output jobs/evaluations/run-02
```

The evaluator refuses to write under `reports/` and refuses to overwrite a
non-empty destination unless `--overwrite` is passed.

## Evaluation limits

Benchmark clips are synthetic and only eight examples. A result on them is a
smoke check, not a general accuracy estimate. Keep baseline reports before
re-running `tools/run_batch.py`, since that script writes into `reports/`.
The pipeline's experimental reruns should use an output directory outside the
preserved benchmark reports when comparing changes.

### Recorded comparison

The original `reports/` results were preserved as the baseline. The latest run is
in the ignored `jobs/evaluations/v2-final/` folder:

| Clip | Baseline outcome | Updated outcome | Updated fused OCR |
|---|---|---|---|
| clip01 | Correct registry match | Review required; top candidate correct | `???IZOE?SZI?` |
| clip02 | Incorrect / abstain | Incorrect / abstain | `???` |
| clip03 | Incorrect / abstain | Incorrect / abstain | `??C???7` |
| clip04 | Incorrect / abstain | Incorrect / abstain | `??7LEX?ZEZ` |
| clip05 | Incorrect / abstain | Incorrect / abstain | `E?E?` |
| clip06 | Incorrect / abstain | Incorrect / abstain | `E?RSS7LA?EO??` |
| clip07 | Correct unregistered abstention | Correct unregistered abstention | `?ZZ` |
| clip08 | Incorrect / abstain | Incorrect / abstain | `57?` |

Automatic decision accuracy is **2/8 baseline, 1/8 updated**. The updated
system retained the correct top registry candidate on clip01 but abstained because
its localization evidence did not meet the automatic gate. Registered top-candidate
accuracy is **1/7 in both runs**, OCR exact match is **0/8 in both**, and character
edit similarity changed from **8.43% to 9.41%** under the evaluator's per-clip
normalized edit-distance metric (`1 - distance/max(reference length, prediction
length)`). Standard reference-normalized character error rate moved from **95.00%
to 96.25%**, so the modest edit-similarity rise does not mean character accuracy
improved. Automatic
false positives were **0/1** on the single unregistered clip in both runs; automatic
false negatives changed from **6/7 to 7/7** because clip01 now requires review.
This is a safety/abstention tradeoff, not an accuracy improvement.

The current run averaged **6.53 seconds per clip** on this CPU environment
(5.08–7.47 seconds). The saved baseline averaged 12.11 seconds, but it included a
44-second first clip/model startup, so the timing comparison is not controlled and
does not establish a throughput improvement. Track coverage stayed at 95.63%.
Vehicle detection precision/recall, plate IoU, and tracking ID consistency remain
unavailable because the ground truth contains no boxes or persistent IDs.

Known limitations include plate text too small or blurred for the CRNN, severe
occlusion, night glare, oblique views, and plate formats/layouts not represented
by the recognizer. Improving these reliably requires a diverse, legally
usable real-world evaluation set with ground truth and a stronger Indian-plate
OCR model, with its runtime and license reviewed before adoption.

## VEHITRACE AI 2.0 reliability and operations

The selected vehicle tracker now uses global greedy assignment over predicted
boxes, constant-velocity extrapolation, and a short missed-detection window. It
remains a lightweight heuristic tracker, not a learned MOT system; ID-switch
accuracy cannot be measured because the benchmark has no persistent vehicle IDs.

Each result exposes separate vehicle, localization, OCR, multi-frame agreement,
and registry-posterior components. These are evidence quality scores and matcher
posteriors, not calibrated real-world probabilities. High/medium/low review tiers
use multiple gates; weak registry candidates are labeled `review_required` rather
than automatically confirmed. Eight synthetic clips do not support statistical
calibration.

Cross-camera comparison now requires readable plate evidence for a definitive
`same_vehicle` verdict. Similar appearance, colour, or body style can still yield
`possible_same_vehicle`, but cannot establish identity alone. Uploads accept a
camera ID and optional ISO-8601 capture time. When capture time is absent, the UI
shows processing time or video-relative offset rather than guessing.

Completed uploads can be reloaded from `jobs/` result files after restart. The API
provides filtered observation history, summary analytics, and CSV export. Metrics
use ground truth only where it exists. Track counts and OCR strings are not claimed
to identify unique real vehicles. Duplicate indications require an exact plate,
the same camera, and supplied timestamps, and are suggestions only.

Review supports accept, reject, and correction actions while preserving the
machine prediction and an append-only review history. It is disabled unless the
server has `VTRACE_REVIEW_TOKEN` configured. The app does not provide user
accounts; add authentication and access control before exposing it to an untrusted
network.

Analysis runs through one inference worker with one additional pending slot;
requests beyond that bounded queue receive HTTP 429. Only one `VideoAnalysis`
object is retained in the in-memory cache to limit memory use. Job videos,
results, and review events are not automatically deleted, so deployments should
configure their own retention and storage monitoring.

Use this command for a repeatable current-vs-saved-baseline comparison without
overwriting the original `reports/` results:

```sh
python tools/evaluate_benchmark.py --output jobs/evaluations/run-01
```

`ground_truth.json` has plate and registry status but no annotated vehicle boxes,
plate boxes, or persistent track IDs. Detection precision/recall, plate IoU, and
tracking identity consistency are unavailable. The evaluator computes exact
plate accuracy, character accuracy, registry false-positive/negative rates, track
coverage, and processing time. Results remain limited to this small synthetic set.

## Phase III frame-vote refinement

Frame OCR confidence is averaged only over characters the recognizer emitted.
That allowed a short, high-confidence fragment to receive a similar vote to a
longer reading with more plate evidence. `vtrace/pipeline.py` now applies a
gentle square-root completeness factor to each frame's vote and records it in
the frame quality diagnostics. It does not suppress partially occluded reads.

On the same eight clips, compared with the Phase II run, this reduced OCR edits
from 74 to 73 of 80 reference characters (CER 92.50% to 91.25%). Normalized edit
similarity moved from 7.50% to 8.75%. Exact OCR remained 0/8, correct automatic
decisions remained 1/8, and registered automatic matches remained 0/8. The
change is small and is not evidence of general accuracy improvement. Full output
is in `jobs/evaluations/phase3-completeness/evaluation.json`.

A sharpened-only OCR experiment was also tested. On 95 correlated crops, the
crop ablation favored sharpened preprocessing over confidence-selected
preprocessing by 0.31 CER percentage points. But on the full clip benchmark it
increased errors to 75/80 and lowered normalized edit similarity to 7.41%, even
though top registry candidate accuracy changed from 1/7 to 2/7 and mean clip
time fell from 7.16s to 6.29s. Because character recognition regressed and the
small benchmark gives conflicting signals, the fixed preprocessing change was
reverted; production still chooses among the three existing variants. The
experimental result remains in `jobs/evaluations/phase5-sharpened/`.

## Phase IV row-aware sequence alignment

The CRNN output tags glyphs with their row for two-line plates, but multi-frame
alignment previously compared glyph distributions without considering that row.
The fusion similarity now penalizes cross-row character pairing more than the
cost of separate insertions/deletions. This is intended to prevent top- and
bottom-row characters from being fused as if they occupied the same position.

The eight-clip run (`jobs/evaluations/phase6-row-aware/evaluation.json`) was
neutral on aggregate OCR: 73/80 edits, 91.25% CER, 8.75% normalized edit
similarity, 0/8 exact reads, and 1/8 correct automatic decisions—unchanged from
Phase III. The truck fused string changed by one character, but its edit distance
did not. Keep this as a structural alignment safeguard, not a measured accuracy
gain; the tiny benchmark has no annotated row boundaries for direct validation.
