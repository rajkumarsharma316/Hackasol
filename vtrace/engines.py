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


def yolo():
    with _lock:
        if "yolo" not in _cache:
            from ultralytics import YOLO
            _cache["yolo"] = YOLO(str(MODELS / "yolo11n.pt"))
        return _cache["yolo"]


def ocr_reader():
    with _lock:
        if "ocr" not in _cache:
            import easyocr
            _cache["ocr"] = easyocr.Reader(["en"], gpu=DEVICE == "cuda",
                                           model_storage_directory=str(MODELS / "easyocr"), verbose=False)
        return _cache["ocr"]


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
