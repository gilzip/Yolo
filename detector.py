"""Core inference engine: `RetailProductDetector`.

Wraps an Ultralytics YOLO model with retail-oriented conveniences: device
auto-selection, confidence/NMS filtering, structured JSON output, per-product
bounding-box crop extraction, and annotated-image rendering via OpenCV.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import ANNOTATED_DIR, CROPS_DIR, DETECTOR_CONFIG, JSON_DIR
from retail_yolo_pipeline import ModelDownloadError, resolve_model_path

logger = logging.getLogger(__name__)

ImageSource = Union[str, Path, np.ndarray]


class DetectorInitError(RuntimeError):
    """Raised when the model fails to load."""


class DetectorInferenceError(RuntimeError):
    """Raised when inference fails on a given source."""


@dataclass
class ProductDetection:
    """A single detected product, ready for JSON serialization."""

    id: int
    label: str
    confidence: float
    bbox: list[int]  # [x1, y1, x2, y2]
    crop_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetailProductDetector:
    """High-level detector for retail shelf photos, video files, or webcam feeds."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        device: str | None = None,
    ) -> None:
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else DETECTOR_CONFIG.confidence_threshold
        )
        self.iou_threshold = iou_threshold if iou_threshold is not None else DETECTOR_CONFIG.iou_threshold
        self.device = self._resolve_device(device or DETECTOR_CONFIG.device)
        self.is_ready = False

        try:
            resolved_path = resolve_model_path(model_path)
            logger.info("Loading YOLO weights from '%s' on device '%s'", resolved_path, self.device)
            self.model = YOLO(str(resolved_path))
            try:
                self.model.to(self.device)
            except Exception as exc:  # noqa: BLE001 - device placement is best-effort here
                logger.warning("Could not move model to device '%s' up front (%s); will pass device per-call.", self.device, exc)
        except ModelDownloadError as exc:
            raise DetectorInitError(f"Could not resolve model weights: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - surface any load failure as a clean, typed error
            raise DetectorInitError(f"Failed to load YOLO model: {exc}") from exc

        self.class_names: dict[int, str] = self.model.names
        self.is_ready = True

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Pick CUDA if available and requested/auto, otherwise CPU."""
        if requested == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available on this machine; falling back to CPU.")
            return "cpu"
        return requested

    def health(self) -> dict[str, Any]:
        """Report GPU/CPU status and model load state for the `/health` endpoint."""
        return {
            "model_loaded": self.is_ready,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "num_classes": len(self.class_names) if self.is_ready else 0,
        }

    def _load_image(self, source: ImageSource) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source
        path = Path(source)
        if not path.exists():
            raise DetectorInferenceError(f"Image source does not exist: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise DetectorInferenceError(f"OpenCV failed to decode image: {path}")
        return image

    def detect_image(
        self,
        source: ImageSource,
        save_crops: bool | None = None,
        save_annotated: bool | None = None,
        output_stem: str | None = None,
    ) -> dict[str, Any]:
        """Run detection on a single image and return the structured payload."""
        save_crops = DETECTOR_CONFIG.save_crops if save_crops is None else save_crops
        save_annotated = DETECTOR_CONFIG.save_annotated if save_annotated is None else save_annotated

        image = self._load_image(source)
        stem = output_stem or (
            Path(source).stem if not isinstance(source, np.ndarray) else f"frame_{int(datetime.now().timestamp() * 1000)}"
        )

        try:
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - any Ultralytics/torch failure becomes a typed error
            raise DetectorInferenceError(f"YOLO inference failed: {exc}") from exc

        result = results[0]
        products: list[ProductDetection] = []

        for idx, box in enumerate(result.boxes):
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].round().tolist())
            class_id = int(box.cls[0])
            label = self.class_names.get(class_id, f"class_{class_id}")
            confidence = float(box.conf[0])

            crop_path = None
            if save_crops:
                crop_path = self._save_crop(image, (x1, y1, x2, y2), stem, idx)

            products.append(
                ProductDetection(
                    id=idx,
                    label=label,
                    confidence=round(confidence, 4),
                    bbox=[x1, y1, x2, y2],
                    crop_path=str(crop_path) if crop_path else None,
                )
            )

        annotated_path = self._save_annotated(image, products, stem) if save_annotated else None

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": str(source) if not isinstance(source, np.ndarray) else "in-memory-frame",
            "total_products_detected": len(products),
            "products": [p.to_dict() for p in products],
            "annotated_image_path": str(annotated_path) if annotated_path else None,
        }

        self._write_json(payload, stem)
        return payload

    def _write_json(self, payload: dict[str, Any], stem: str) -> None:
        json_path = JSON_DIR / f"{stem}.json"
        try:
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write JSON output to %s: %s", json_path, exc)

    def _save_crop(self, image: np.ndarray, bbox: tuple[int, int, int, int], stem: str, idx: int) -> Path | None:
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            logger.warning("Skipping degenerate bbox %s for crop %d", bbox, idx)
            return None

        crop = image[y1:y2, x1:x2]
        crop_path = CROPS_DIR / f"{stem}_product_{idx}.jpg"
        if not cv2.imwrite(str(crop_path), crop):
            logger.warning("Failed to write crop to %s", crop_path)
            return None
        return crop_path

    def _save_annotated(self, image: np.ndarray, products: list[ProductDetection], stem: str) -> Path | None:
        annotated = image.copy()
        rng = np.random.default_rng(42)
        color_map: dict[str, tuple[int, int, int]] = {}

        for product in products:
            if product.label not in color_map:
                color_map[product.label] = tuple(int(c) for c in rng.integers(64, 256, size=3))
            color = color_map[product.label]

            x1, y1, x2, y2 = product.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            caption = f"{product.label} {product.confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - text_h - 8)), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(
                annotated,
                caption,
                (x1 + 2, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        annotated_path = ANNOTATED_DIR / f"{stem}_annotated.jpg"
        if not cv2.imwrite(str(annotated_path), annotated):
            logger.warning("Failed to write annotated image to %s", annotated_path)
            return None
        return annotated_path

    def detect_video(
        self,
        source: ImageSource,
        save_crops: bool = False,
        save_annotated: bool = True,
        frame_stride: int = 1,
    ) -> list[dict[str, Any]]:
        """Run detection across a video file or webcam index, frame by frame.

        `source` may be a file path, or a numeric string (e.g. "0") to open a
        webcam by index. Returns one JSON payload per processed frame.
        """
        cam_index: int | None = None
        if isinstance(source, (str, Path)) and str(source).isdigit():
            cam_index = int(str(source))

        capture = cv2.VideoCapture(cam_index if cam_index is not None else str(source))
        if not capture.isOpened():
            raise DetectorInferenceError(f"Could not open video source: {source}")

        frame_results: list[dict[str, Any]] = []
        frame_idx = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_idx % max(1, frame_stride) == 0:
                    payload = self.detect_image(
                        frame,
                        save_crops=save_crops,
                        save_annotated=save_annotated,
                        output_stem=f"frame_{frame_idx:06d}",
                    )
                    frame_results.append(payload)
                frame_idx += 1
        finally:
            capture.release()

        return frame_results
