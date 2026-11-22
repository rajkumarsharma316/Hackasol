"""Download / warm up all pretrained weights once (needs internet the first time)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))

import torch  # noqa: E402

print("torch", torch.__version__, "cuda:", torch.cuda.is_available())

from ultralytics import YOLO  # noqa: E402

YOLO(str(MODELS / "yolo11n.pt"))  # auto-downloads to that path
print("YOLO ok")

import easyocr  # noqa: E402

easyocr.Reader(["en"], gpu=torch.cuda.is_available(), model_storage_directory=str(MODELS / "easyocr"))
print("EasyOCR ok")

import torchvision  # noqa: E402

torchvision.models.efficientnet_b0(weights="DEFAULT")
print("EfficientNet ok")

if "--clip" in sys.argv:
    from transformers import CLIPModel, CLIPProcessor  # noqa: E402
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=str(MODELS / "hf"))
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=str(MODELS / "hf"))
    print("CLIP ok")
