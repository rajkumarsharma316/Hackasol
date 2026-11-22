"""Download the optional pretrained plate detector used by VTrace.

Run from the project root with an installed ``huggingface_hub`` package:
    python tools/setup_plate_detector.py
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "plate_detector"
DEST.mkdir(parents=True, exist_ok=True)

path = hf_hub_download(
    repo_id="Babblu2821/alpr-plate-detector",
    filename="best.pt",
    local_dir=DEST,
)
target = DEST / "best.pt"
if Path(path) != target:
    target.parent.mkdir(parents=True, exist_ok=True)
    Path(path).replace(target)
print(f"Plate detector ready: {target} ({target.stat().st_size:,} bytes)")
