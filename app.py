"""FastAPI REST server exposing `RetailProductDetector` over HTTP.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config import UPLOADS_DIR
from detector import DetectorInferenceError, DetectorInitError, RetailProductDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Retail Product Detector",
    description="YOLO11/YOLOv8-based retail product detection for supermarket/pharmacy shelves.",
    version="1.0.0",
)

detector: RetailProductDetector | None = None


@app.on_event("startup")
def load_model() -> None:
    """Load the YOLO model once at startup so `/detect` requests stay fast."""
    global detector
    try:
        detector = RetailProductDetector()
        logger.info("Model loaded successfully: %s", detector.health())
    except DetectorInitError as exc:
        logger.error("Model failed to load at startup: %s", exc)
        detector = None


@app.get("/health")
def health() -> dict:
    """Return GPU/CPU status and model load state."""
    if detector is None:
        return JSONResponse(
            status_code=503,
            content={"model_loaded": False, "error": "Model is not loaded. Check server logs."},
        )
    return detector.health()


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict:
    """Accept an uploaded image, run detection, and return the structured JSON payload."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is not loaded; check /health for details.")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    tmp_path = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"

    try:
        with tmp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded file: {exc}") from exc
    finally:
        await file.close()

    try:
        stem = Path(file.filename or tmp_path.name).stem
        payload = detector.detect_image(tmp_path, output_stem=stem)
    except DetectorInferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return payload
