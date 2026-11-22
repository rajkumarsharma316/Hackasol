"""Lazy-loaded pretrained models (loaded once per process)."""
from __future__ import annotations

import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")

import torch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _yolo_device():
    """torchvision wheels are sometimes CPU-only even when torch has CUDA; YOLO's NMS then fails on GPU."""
    if DEVICE != "cuda":
        return "cpu"
    try:
        import torchvision
        b = torch.zeros((1, 4), device="cuda")
        torchvision.ops.nms(b, torch.zeros(1, device="cuda"), 0.5)
        return 0
    except Exception:
        return "cpu"


YOLO_DEVICE = _yolo_device()
_lock = threading.Lock()
_cache: dict = {}
PLATE_DETECTOR_WEIGHTS = MODELS / "plate_detector" / "best.pt"


def yolo():
    with _lock:
        if "yolo" not in _cache:
            from ultralytics import YOLO
            _cache["yolo"] = YOLO(str(MODELS / "yolo11n.pt"))
        return _cache["yolo"]


def ocr_reader(detector: bool = True):
    """Return EasyOCR with/without CRAFT text detection, cached separately."""
    key = "ocr_with_detector" if detector else "ocr_recognizer_only"
    with _lock:
        if key not in _cache:
            import easyocr
            storage = MODELS / "easyocr"
            storage.mkdir(parents=True, exist_ok=True)
            _cache[key] = easyocr.Reader(["en"], gpu=DEVICE == "cuda", detector=detector,
                                         model_storage_directory=str(storage),
                                         user_network_directory=str(storage / "user_network"), verbose=False)
        return _cache[key]


def plate_detector():
    """Return the optional Indian/European plate detector when weights are installed."""
    if not PLATE_DETECTOR_WEIGHTS.is_file():
        return None
    with _lock:
        if "plate_detector" not in _cache:
            from ultralytics import YOLO
            _cache["plate_detector"] = YOLO(str(PLATE_DETECTOR_WEIGHTS))
        return _cache["plate_detector"]


def clip():
    """Returns (model, processor) or None if CLIP weights are unavailable."""
    with _lock:
        if "clip" not in _cache:
            try:
                from transformers import CLIPModel, CLIPProcessor
                name = "openai/clip-vit-base-patch32"
                m = CLIPModel.from_pretrained(name, cache_dir=str(MODELS / "hf")).to(DEVICE).eval()
                p = CLIPProcessor.from_pretrained(name, cache_dir=str(MODELS / "hf"))
                _cache["clip"] = (m, p)
            except Exception as e:  # offline / not installed -> feature disabled
                print("CLIP unavailable:", e)
                _cache["clip"] = None
        return _cache["clip"]


def warmup():
    yolo(); ocr_reader(); clip()
