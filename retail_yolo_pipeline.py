"""Model acquisition pipeline: resolves which YOLO weights to use.

Resolution order:
1. An explicit local weights path the caller points at (highest priority).
2. A Roboflow Universe retail/grocery detection model, if credentials are
   configured (`ROBOFLOW_API_KEY` / `ROBOFLOW_WORKSPACE`).
3. The official Ultralytics COCO-pretrained checkpoint (yolo11m.pt /
   yolov8m.pt), which Ultralytics downloads automatically on first load.

This means the application works "out of the box" with zero configuration
(COCO weights already recognize many shelf-relevant classes such as bottle,
book, box, cup, etc.) while still supporting a drop-in upgrade to a
purpose-trained retail/grocery model from Roboflow Universe.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import DETECTOR_CONFIG, MODELS_DIR, RoboflowConfig

logger = logging.getLogger(__name__)


class ModelDownloadError(RuntimeError):
    """Raised when no usable model weights could be resolved."""


def _download_from_roboflow(cfg: RoboflowConfig) -> Path | None:
    """Attempt to download a retail-specific model from Roboflow Universe.

    Returns the path to the downloaded weights file, or None if Roboflow is
    not configured or the download fails — callers should treat None as
    "fall back to the COCO-pretrained checkpoint" rather than a hard error.
    """
    if not cfg.is_configured:
        logger.info("Roboflow not configured (missing API key/workspace); skipping.")
        return None

    try:
        from roboflow import Roboflow  # optional dependency, imported lazily
    except ImportError:
        logger.warning("`roboflow` package not installed; skipping Roboflow download.")
        return None

    try:
        rf = Roboflow(api_key=cfg.api_key)
        project = rf.workspace(cfg.workspace).project(cfg.project)
        version = project.version(cfg.version)

        dest_dir = MODELS_DIR / f"roboflow_{cfg.project}_v{cfg.version}"
        dataset = version.model.download(cfg.export_format, location=str(dest_dir))

        weights_path = Path(dataset.location) / "weights.pt"
        if weights_path.exists():
            logger.info("Downloaded Roboflow model to %s", weights_path)
            return weights_path

        logger.warning(
            "Roboflow download completed but no weights.pt found at %s",
            weights_path,
        )
        return None
    except Exception as exc:  # noqa: BLE001 - any network/SDK failure should degrade gracefully
        logger.warning(
            "Roboflow model download failed (%s); falling back to COCO weights.",
            exc,
        )
        return None


def resolve_model_path(explicit_path: str | Path | None = None) -> Path:
    """Resolve the weights file to load, downloading it if necessary.

    Raises `ModelDownloadError` only if an explicit path was given and does
    not exist — the Roboflow/COCO fallback chain never raises, since a
    missing Roboflow model is expected to fall through to COCO weights.
    """
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise ModelDownloadError(f"Explicit model path does not exist: {path}")
        return path

    roboflow_path = _download_from_roboflow(DETECTOR_CONFIG.roboflow)
    if roboflow_path is not None:
        return roboflow_path

    # Ultralytics resolves bare checkpoint names (e.g. "yolo11m.pt") by
    # downloading them from the official release assets on first load, so we
    # hand back a cached copy if we have one, otherwise the bare name and let
    # `YOLO()` perform (and cache) the download itself.
    fallback_name = DETECTOR_CONFIG.fallback_model_name
    cached = MODELS_DIR / fallback_name
    if cached.exists():
        return cached

    logger.info(
        "No cached weights found; Ultralytics will auto-download '%s' on first load.",
        fallback_name,
    )
    return Path(fallback_name)
