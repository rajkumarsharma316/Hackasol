# VTrace

VTrace analyses CCTV video for vehicle tracks, number plates, visual attributes, and possible registry matches. It includes a FastAPI web app, sample clips, precomputed reports, and batch-analysis tools.

## Run locally

Use Python 3.10 or newer:

```sh
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python tools/setup_models.py
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

### Optional specialized number-plate detector

The pipeline falls back to OCR-text and colour proposals when no dedicated plate detector is installed. For stronger plate localization on supported Indian and European plate imagery, install the optional checkpoint:

```sh
python tools/setup_plate_detector.py
```

The weights are saved outside version control at `models/plate_detector/best.pt`. See [recognition pipeline notes](docs/AI_RECOGNITION.md) for model provenance, limitations, and benchmark notes.

## Useful commands

```sh
python tools/run_batch.py
python tools/run_batch.py clip01
python tools/evaluate_benchmark.py --output jobs/evaluations/run-01
```

`tools/run_batch.py` refreshes `reports/`. Use `evaluate_benchmark.py` for comparisons that leave saved reports untouched.

## Project map

- `app.py` — FastAPI routes, uploads, and background analysis jobs.
- `vtrace/` — vehicle tracking, plate localization/OCR, matching, attributes, and reporting.
- `data/registry.csv` — example vehicle registry.
- `test_clips/` — synthetic clips and associated ground truth.
- `tools/` — setup, evaluation, and batch-analysis utilities.

Registry matches are investigative leads, not identity confirmation. Validate against original footage and authoritative records.
