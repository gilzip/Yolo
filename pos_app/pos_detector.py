"""Lightweight YOLO inference wrapper optimized for real-time POS video.

Unlike `detector.RetailProductDetector` (which writes JSON/crops/annotated
images to disk on every call — convenient for the batch CLI/API, too slow
for a live camera loop), this module keeps everything in memory and never
touches disk during inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from config import DETECTOR_CONFIG
from pos_app.config import CUSTOM_MODEL_PATH
from retail_yolo_pipeline import resolve_model_path

logger = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


@dataclass
class LiveDetection:
    """One detection from a single real-time inference pass."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


class POSDetector:
    """Real-time detector used by the Smart Checkout screen (and, for the
    generic fallback model, by the onboarding pre-processing pipeline)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.35,
    ) -> None:
        self.device = _resolve_device(DETECTOR_CONFIG.device)
        self.confidence_threshold = confidence_threshold
        resolved = Path(model_path) if model_path else resolve_model_path(None)

        logger.info("POSDetector loading '%s' on device '%s'", resolved, self.device)
        self.model = YOLO(str(resolved))
        try:
            self.model.to(self.device)
        except Exception as exc:  # noqa: BLE001 - device placement is best-effort
            logger.warning("Could not move model to device '%s' (%s); using per-call device arg.", self.device, exc)

        self.class_names: dict[int, str] = self.model.names
        self.using_custom_model = model_path is not None

    def infer(self, frame: np.ndarray) -> list[LiveDetection]:
        """Run one detection pass on a BGR frame; returns all boxes above the confidence threshold."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        detections: list[LiveDetection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].round().tolist())
            class_id = int(box.cls[0])
            label = self.class_names.get(class_id, f"class_{class_id}")
            confidence = float(box.conf[0])
            detections.append(LiveDetection(label=label, confidence=confidence, bbox=(x1, y1, x2, y2)))
        return detections


_generic_detector_cache: POSDetector | None = None


def get_generic_detector() -> POSDetector | None:
    """Return a cached generic (COCO/Roboflow fallback) detector for pseudo-labeling.

    Used only by the dataset preprocessing utility, never the live checkout
    loop. Returns None instead of raising if the model can't be loaded, so
    preprocessing can still fall back to centered-box labels.
    """
    global _generic_detector_cache
    if _generic_detector_cache is None:
        try:
            _generic_detector_cache = POSDetector()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load generic detector for pseudo-labeling (%s).", exc)
            return None
    return _generic_detector_cache


def load_checkout_detector() -> POSDetector:
    """Load the detector used by the Smart Checkout screen.

    Prefers a fine-tuned custom model (trained on onboarded products, whose
    class names are barcodes — see train_yolo.py) if one exists at
    CUSTOM_MODEL_PATH; otherwise falls back to the generic model, in which
    case detected labels will be generic object classes rather than
    barcodes until a custom model is trained.
    """
    if CUSTOM_MODEL_PATH.exists():
        return POSDetector(model_path=CUSTOM_MODEL_PATH, confidence_threshold=DETECTOR_CONFIG.confidence_threshold)

    logger.warning(
        "No fine-tuned model found at %s — falling back to the generic model. "
        "Detected labels will be generic object classes, not barcodes, until "
        "you onboard products and run pos_app/train_yolo.py.",
        CUSTOM_MODEL_PATH,
    )
    return POSDetector(model_path=None, confidence_threshold=DETECTOR_CONFIG.confidence_threshold)
