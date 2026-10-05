"""Configuration for the Smart Retail POS PyQt5 application.

Reuses the root project's `config.py` (paths, model settings) and adds the
onboarding-capture and checkout/cart behavior settings that are specific to
this GUI app. All values can be overridden via environment variables.
"""

from __future__ import annotations

import os

from config import BASE_DIR, MODELS_DIR

# ---------------------------------------------------------------------------
# Dataset paths (onboarding capture output)
# ---------------------------------------------------------------------------
DATASET_DIR = BASE_DIR / "dataset"
DATASET_IMAGES_DIR = DATASET_DIR / "images"
PRODUCTS_JSON_PATH = DATASET_DIR / "products.json"
YOLO_DATASET_DIR = DATASET_DIR / "yolo"

# Completed checkout transactions (receipts), separate from training data.
ORDERS_DIR = BASE_DIR / "orders"

for _directory in (DATASET_DIR, DATASET_IMAGES_DIR, YOLO_DATASET_DIR, ORDERS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX = int(os.getenv("POS_CAMERA_INDEX", "0"))

# ---------------------------------------------------------------------------
# Onboarding / data-collection capture
# ---------------------------------------------------------------------------
CAPTURE_COUNT = int(os.getenv("POS_CAPTURE_COUNT", "18"))
CAPTURE_INTERVAL_MS = int(os.getenv("POS_CAPTURE_INTERVAL_MS", "350"))

# ---------------------------------------------------------------------------
# Checkout / cart behavior
# ---------------------------------------------------------------------------
POS_CONFIDENCE_THRESHOLD = float(os.getenv("POS_CONFIDENCE_THRESHOLD", "0.80"))
CONSISTENCY_FRAMES = int(os.getenv("POS_CONSISTENCY_FRAMES", "5"))
COOLDOWN_SECONDS = float(os.getenv("POS_COOLDOWN_SECONDS", "3.0"))
THUMBNAIL_SIZE = 56

# Detections below the cart-adding threshold but above this floor are shown
# as clickable "possible match" suggestions instead of being discarded.
# Shared between the desktop Checkout screen and the browser UI.
SUGGESTION_MIN_CONFIDENCE = float(os.getenv("POS_SUGGESTION_MIN_CONFIDENCE", "0.10"))
MAX_SUGGESTIONS = int(os.getenv("POS_MAX_SUGGESTIONS", "3"))

# Custom fine-tuned model (produced by train_yolo.py). Its class names are
# expected to be product barcodes. Falls back to the root project's generic
# COCO/Roboflow model when this file doesn't exist yet.
CUSTOM_MODEL_PATH = MODELS_DIR / os.getenv("POS_CUSTOM_MODEL", "pos_custom.pt")

# ---------------------------------------------------------------------------
# Vector (embedding) recognition — an alternative to the trained YOLO
# classifier that matches a scanned photo against the catalog by nearest
# neighbor instead of a fixed set of learned classes. No training/fine-tuning
# step is needed: a barcode is searchable as soon as it has at least one
# photo, which is the main thing the classifier struggles with.
# ---------------------------------------------------------------------------
EMBEDDINGS_PATH = DATASET_DIR / "embeddings.npz"
VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("POS_VECTOR_SIMILARITY_THRESHOLD", "0.65"))
VECTOR_TOP_K = int(os.getenv("POS_VECTOR_TOP_K", "5"))
