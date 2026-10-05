"""Vector (embedding) based product recognition.

An alternative to the trained YOLO classifier in `pos_detector.py`: instead
of learning a fixed set of classes, this embeds every catalog photo with a
generic pretrained feature extractor (MobileNetV3) and recognizes a scanned
frame by nearest-neighbor similarity against those embeddings. A barcode
becomes searchable as soon as it has one photo — no fine-tuning run needed,
which is the main thing the classifier struggles with for a catalog where
most products only have a single reference photo.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
from torchvision import models
from torchvision.transforms import functional as TF

from pos_app.config import EMBEDDINGS_PATH
from pos_app.dataset_utils import get_catalog

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_INPUT_SIZE = 224

_model: torch.nn.Module | None = None
_catalog_barcodes: np.ndarray | None = None
_catalog_vectors: np.ndarray | None = None


def _get_model() -> torch.nn.Module:
    global _model
    if _model is None:
        logger.info("Loading MobileNetV3-Small for vector recognition (one-time download if not cached)")
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        net.classifier = torch.nn.Identity()  # keep the pooled feature vector, drop the 1000-class head
        net.eval()
        _model = net
    return _model


def compute_embedding(image_bgr: np.ndarray) -> np.ndarray:
    """Return an L2-normalized feature vector for one BGR image (as from cv2)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = TF.normalize(tensor, mean=_IMAGENET_MEAN, std=_IMAGENET_STD).unsqueeze(0)

    with torch.no_grad():
        features = _get_model()(tensor)

    vector = features.squeeze(0).numpy().astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def build_catalog_embeddings() -> dict:
    """Recompute embeddings for every photo of every catalog product and save them.

    Cheap enough (a few hundred ms per image on CPU) to just rerun in full
    after any catalog change, rather than tracking incremental updates.
    """
    catalog = get_catalog()
    barcodes: list[str] = []
    vectors: list[np.ndarray] = []
    skipped: list[str] = []

    for barcode, product in catalog.items():
        for image_path in product.get("image_paths", []):
            image = cv2.imread(str(image_path))
            if image is None or image.size == 0:
                skipped.append(str(image_path))
                continue
            barcodes.append(barcode)
            vectors.append(compute_embedding(image))

    if vectors:
        matrix = np.stack(vectors).astype(np.float32)
    else:
        matrix = np.empty((0, 0), dtype=np.float32)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(EMBEDDINGS_PATH, barcodes=np.array(barcodes, dtype=object), vectors=matrix)

    global _catalog_barcodes, _catalog_vectors
    _catalog_barcodes = np.array(barcodes, dtype=object)
    _catalog_vectors = matrix

    return {
        "products": len(catalog),
        "images_embedded": len(barcodes),
        "images_skipped": len(skipped),
        "skipped_paths": skipped,
    }


def _load_cached_embeddings() -> tuple[np.ndarray, np.ndarray]:
    global _catalog_barcodes, _catalog_vectors
    if _catalog_barcodes is None or _catalog_vectors is None:
        if not EMBEDDINGS_PATH.exists():
            return np.array([], dtype=object), np.empty((0, 0), dtype=np.float32)
        with np.load(EMBEDDINGS_PATH, allow_pickle=True) as data:
            _catalog_barcodes = data["barcodes"]
            _catalog_vectors = data["vectors"]
    return _catalog_barcodes, _catalog_vectors


def invalidate_cache() -> None:
    """Force the next search to reload embeddings.npz from disk."""
    global _catalog_barcodes, _catalog_vectors
    _catalog_barcodes, _catalog_vectors = None, None


def find_nearest(query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    """Return up to `top_k` catalog products ranked by cosine similarity.

    Each catalog photo votes with its own similarity score; a product is
    represented by its single best-matching photo (not an average), since a
    product's photos can cover quite different angles.
    """
    barcodes, vectors = _load_cached_embeddings()
    if vectors.size == 0:
        return []

    similarities = vectors @ query_embedding  # vectors are rows, both L2-normalized
    best_per_barcode: dict[str, float] = {}
    for barcode, score in zip(barcodes, similarities):
        barcode = str(barcode)
        if score > best_per_barcode.get(barcode, -1.0):
            best_per_barcode[barcode] = float(score)

    ranked = sorted(best_per_barcode.items(), key=lambda item: item[1], reverse=True)
    return [{"barcode": barcode, "similarity": score} for barcode, score in ranked[:top_k]]
