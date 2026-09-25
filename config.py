"""Centralized configuration for the retail product detection pipeline.

All paths, thresholds, and model selection live here so the rest of the
codebase never hard-codes a magic string or path. Every value can be
overridden via environment variables, which keeps the same code portable
between a developer's laptop and a server/container deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
CROPS_DIR = OUTPUT_DIR / "crops"
JSON_DIR = OUTPUT_DIR / "json"
TEST_IMAGES_DIR = BASE_DIR / "test_images"
UPLOADS_DIR = BASE_DIR / "uploads"

for _directory in (
    MODELS_DIR,
    OUTPUT_DIR,
    ANNOTATED_DIR,
    CROPS_DIR,
    JSON_DIR,
    TEST_IMAGES_DIR,
    UPLOADS_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)


@dataclass
class RoboflowConfig:
    """Optional Roboflow Universe integration for retail-specific weights.

    Roboflow Universe hosts community-trained grocery/shelf/retail product
    detection models. Populate these fields (or the matching environment
    variables) to pull a purpose-trained model automatically. If left
    unconfigured, the pipeline transparently falls back to the official
    COCO-pretrained YOLO checkpoint, so the app still works out of the box.
    """

    api_key: str = os.getenv("ROBOFLOW_API_KEY", "")
    workspace: str = os.getenv("ROBOFLOW_WORKSPACE", "")
    project: str = os.getenv("ROBOFLOW_PROJECT", "grocery-dataset")
    version: int = int(os.getenv("ROBOFLOW_VERSION", "1"))
    # Export format Roboflow should package the weights in.
    export_format: str = os.getenv("ROBOFLOW_FORMAT", "yolov8")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.workspace)


@dataclass
class DetectorConfig:
    """Runtime configuration consumed by `RetailProductDetector`."""

    # Fallback COCO-pretrained checkpoint used when no retail-specific
    # weights are configured/available. Ultralytics auto-downloads this
    # file on first use if it isn't already cached locally.
    fallback_model_name: str = os.getenv("FALLBACK_MODEL", "yolo11m.pt")

    confidence_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.35"))
    iou_threshold: float = float(os.getenv("IOU_THRESHOLD", "0.45"))

    # "auto" selects CUDA if available, otherwise CPU.
    device: str = os.getenv("DEVICE", "auto")

    save_crops: bool = os.getenv("SAVE_CROPS", "true").lower() == "true"
    save_annotated: bool = os.getenv("SAVE_ANNOTATED", "true").lower() == "true"

    roboflow: RoboflowConfig = field(default_factory=RoboflowConfig)


DETECTOR_CONFIG = DetectorConfig()
