"""Dataset utilities for the POS onboarding pipeline.

Handles saving raw product-capture images plus a JSON catalog
(`products.json`), and converting that raw capture set into a YOLO
detection-format dataset ready for fine-tuning.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pos_app.config import DATASET_IMAGES_DIR, PRODUCTS_JSON_PATH, YOLO_DATASET_DIR

logger = logging.getLogger(__name__)


def _load_catalog() -> dict[str, Any]:
    if PRODUCTS_JSON_PATH.exists():
        try:
            return json.loads(PRODUCTS_JSON_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("products.json is corrupt; starting a fresh catalog.")
    return {}


def _save_catalog(catalog: dict[str, Any]) -> None:
    PRODUCTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRODUCTS_JSON_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")


def next_capture_index(barcode: str) -> int:
    """Return the next free img_N index for a barcode, continuing across capture sessions.

    Re-onboarding an existing barcode must not reuse `img_1.jpg..img_N.jpg`
    from a prior session — that would silently overwrite those files while
    the catalog still lists them, inflating `image_count` beyond what's
    actually on disk. Basing the next index on files already present (not
    just the catalog's `image_count`) keeps this correct even if the catalog
    and folder ever drift apart.
    """
    product_dir = DATASET_IMAGES_DIR / barcode
    if not product_dir.exists():
        return 1
    existing_indexes = []
    for path in product_dir.glob("img_*.jpg"):
        try:
            existing_indexes.append(int(path.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return (max(existing_indexes) + 1) if existing_indexes else 1


def save_capture_frame(barcode: str, frame: np.ndarray, index: int) -> Path:
    """Save one captured frame under dataset/images/<barcode>/img_N.jpg."""
    product_dir = DATASET_IMAGES_DIR / barcode
    product_dir.mkdir(parents=True, exist_ok=True)
    image_path = product_dir / f"img_{index}.jpg"
    if not cv2.imwrite(str(image_path), frame):
        raise IOError(f"Failed to write capture frame to {image_path}")
    return image_path


def save_front_image(barcode: str, frame: np.ndarray) -> Path:
    """Save a single reference photo of the product's front (label/name side).

    Always written to dataset/images/<barcode>/front.jpg — a fixed name, so
    re-capturing it simply replaces the previous one rather than piling up
    copies like the rotation burst does.
    """
    product_dir = DATASET_IMAGES_DIR / barcode
    product_dir.mkdir(parents=True, exist_ok=True)
    image_path = product_dir / "front.jpg"
    if not cv2.imwrite(str(image_path), frame):
        raise IOError(f"Failed to write front image to {image_path}")
    return image_path


def save_checkout_confirmation_frame(barcode: str, frame: np.ndarray) -> Path:
    """Save a real checkout-camera frame as additional training data for `barcode`.

    Unlike the onboarding burst's sequential img_N.jpg naming, these accrue
    continuously during real operation, so each gets a timestamped filename
    instead of colliding.
    """
    product_dir = DATASET_IMAGES_DIR / barcode
    product_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    image_path = product_dir / f"checkout_{timestamp}.jpg"
    if not cv2.imwrite(str(image_path), frame):
        raise IOError(f"Failed to write checkout confirmation frame to {image_path}")
    return image_path


def record_checkout_training_sample(barcode: str, frame: np.ndarray) -> None:
    """Fold a human-confirmed checkout frame back into the product's training set.

    Called only when an operator *manually* confirms a match at checkout (a
    weak-detection suggestion, a catalog pick, or a typed/scanned barcode) —
    never for fully-automatic high-confidence scans, since those aren't
    human-verified and feeding likely-correct-but-unverified guesses back in
    would risk reinforcing the model's own mistakes. Manual confirmations are
    trustworthy ground truth, and they come from the actual deployed camera
    angle/lighting — exactly the real-world conditions the model needs more
    of — so the model keeps improving from genuine use, not just onboarding.

    Best-effort: swallows failures so a disk hiccup never disrupts an
    active checkout flow.
    """
    try:
        image_path = save_checkout_confirmation_frame(barcode, frame)
        register_product(barcode=barcode, name="", price=0.0, image_paths=[image_path])
    except IOError as exc:
        logger.warning("Could not save checkout training sample for %s: %s", barcode, exc)


def register_product(
    barcode: str,
    name: str,
    price: float,
    image_paths: list[Path],
    front_image_path: Path | None = None,
) -> None:
    """Create or update a product's entry in products.json with new capture paths."""
    catalog = _load_catalog()
    entry = catalog.get(barcode, {})
    existing_paths = entry.get("image_paths", [])
    # dict.fromkeys dedupes while preserving order, guarding against any
    # already-duplicated catalog entries from before this fix.
    all_paths = list(dict.fromkeys(existing_paths + [str(p) for p in image_paths]))
    now = datetime.now(timezone.utc).isoformat()

    catalog[barcode] = {
        "barcode": barcode,
        "name": name or entry.get("name", barcode),
        "price": price if price else entry.get("price", 0.0),
        "image_paths": all_paths,
        "image_count": len(all_paths),
        "front_image_path": str(front_image_path) if front_image_path else entry.get("front_image_path"),
        "created_at": entry.get("created_at", now),
        "updated_at": now,
    }
    _save_catalog(catalog)


def get_catalog() -> dict[str, Any]:
    """Return the full barcode -> product metadata catalog."""
    return _load_catalog()


def get_product(barcode: str) -> dict[str, Any] | None:
    """Look up a single product's metadata by barcode."""
    return _load_catalog().get(barcode)


def _grabcut_refine(image: np.ndarray, prior_box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Tighten a prior box to the actual foreground object using GrabCut segmentation.

    Onboarding photos are typically a hand holding a product against a room
    background (and often the operator's face/torso). GrabCut, seeded with a
    prior rectangle, segments foreground from background by color/texture —
    letting us carve out just the product without any manual annotation.
    Falls back to the prior box unchanged if segmentation fails or produces
    a degenerate result.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in prior_box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return prior_box

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (x1, y1, x2 - x1, y2 - y1)

    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error as exc:
        logger.warning("GrabCut refinement failed (%s); using the prior box unchanged.", exc)
        return prior_box

    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
    ys, xs = np.where(foreground)
    if xs.size == 0 or ys.size == 0:
        return prior_box

    fx1, fx2 = int(xs.min()), int(xs.max())
    fy1, fy2 = int(ys.min()), int(ys.max())
    if (fx2 - fx1) < 0.05 * w or (fy2 - fy1) < 0.05 * h:
        return prior_box  # degenerate sliver — not a trustworthy segmentation

    # A uniform-colored shirt (low internal contrast) or a frame where the
    # product isn't actually in view yet can make GrabCut latch onto a tiny
    # unrelated texture (e.g. a background object) instead of failing
    # outright. If the found foreground is much smaller than the region we
    # asked it to search, that's a sign the segmentation isn't trustworthy —
    # better to keep the (looser but sane) prior box than a confident-looking
    # but wrong tight one.
    prior_area = (x2 - x1) * (y2 - y1)
    foreground_area = (fx2 - fx1) * (fy2 - fy1)
    if prior_area > 0 and foreground_area < 0.15 * prior_area:
        return prior_box

    return (fx1, fy1, fx2, fy2)


def _detect_primary_bbox(image: np.ndarray, generic_detector: Any) -> tuple[float, float, float, float]:
    """Return a normalized (x_center, y_center, w, h) pseudo-label bbox for one product image.

    Captured onboarding photos are close-up, single-product shots with no
    hand-labeled bounding boxes. We approximate one in three steps:

    1. Run the generic COCO/Roboflow detector and keep its largest
       non-person box that's also large enough to plausibly be a held
       product (>=15% of the frame) — grocery/pharmacy products aren't
       COCO classes, so this usually finds nothing, but it also means a
       detected "person" (or an incidental small background object like a
       clock) is never mistaken for the product.
    2. If nothing else was found, use a centered prior box positioned below
       any detected person's head/shoulders (products are typically held at
       chest height), rather than one spanning almost the entire frame.
    3. Refine that prior with GrabCut foreground segmentation, so the final
       label hugs the actual product instead of the surrounding room/body.

    This is a heuristic, not ground truth: spot-check a few generated labels
    before a long training run.
    """
    h, w = image.shape[:2]
    best_box: tuple[float, float, float, float] | None = None
    best_area = 0.0
    person_boxes: list[tuple[float, float, float, float]] = []

    if generic_detector is not None:
        try:
            results = generic_detector.model.predict(source=image, conf=0.25, verbose=False)
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                label = generic_detector.class_names.get(class_id, "")
                if label == "person":
                    person_boxes.append((x1, y1, x2, y2))
                    continue
                area = (x2 - x1) * (y2 - y1)
                # A product held up to the camera fills a substantial part
                # of the frame. Ignore small non-person detections (e.g. a
                # background clock or picture frame) that COCO happens to
                # recognize but that clearly aren't the item being onboarded.
                if area > best_area and area >= 0.15 * (w * h):
                    best_area = area
                    best_box = (x1, y1, x2, y2)
        except Exception as exc:  # noqa: BLE001 - pseudo-labeling must never hard-fail the pipeline
            logger.warning("Generic detector failed during pseudo-labeling (%s); using centered box.", exc)

    if best_box is None:
        # A held product usually sits below the face/shoulders. If a person
        # was detected, start the prior box partway down their bounding box
        # (past the face, into the neck/chest area); otherwise fall back to
        # a moderate centered box (not the whole frame).
        if person_boxes:
            top = max(py1 + (py2 - py1) * 0.35 for (_, py1, _, py2) in person_boxes)
        else:
            top = h * 0.25
        prior = (w * 0.20, top, w * 0.80, h * 0.95)
        best_box = _grabcut_refine(image, prior)

    x1, y1, x2, y2 = best_box
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    box_w = (x2 - x1) / w
    box_h = (y2 - y1) / h
    return x_center, y_center, box_w, box_h


def prepare_yolo_dataset(val_split: float = 0.2, seed: int = 42) -> Path:
    """Convert dataset/images/<barcode>/*.jpg into a YOLO detection dataset.

    Produces dataset/yolo/{images,labels}/{train,val}/ plus a data.yaml, with
    each barcode as its own class (so a fine-tuned model's detected class
    name IS the barcode). Returns the path to data.yaml.
    """
    from pos_app.pos_detector import get_generic_detector  # lazy: avoid loading YOLO at import time

    catalog = _load_catalog()
    barcodes = sorted(catalog.keys())
    if not barcodes:
        raise ValueError("No products found in products.json — run onboarding capture first.")

    generic_detector = get_generic_detector()

    for split in ("train", "val"):
        (YOLO_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    class_index = {barcode: idx for idx, barcode in enumerate(barcodes)}

    for barcode in barcodes:
        image_paths = [Path(p) for p in catalog[barcode].get("image_paths", []) if Path(p).exists()]
        if not image_paths:
            logger.warning("No existing images for barcode %s; skipping.", barcode)
            continue
        rng.shuffle(image_paths)

        val_count = max(1, int(len(image_paths) * val_split)) if len(image_paths) >= 5 else 0
        val_set = set(image_paths[:val_count])

        for image_path in image_paths:
            split = "val" if image_path in val_set else "train"
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Skipping unreadable image: %s", image_path)
                continue

            x_center, y_center, box_w, box_h = _detect_primary_bbox(image, generic_detector)

            dest_image = YOLO_DATASET_DIR / "images" / split / f"{barcode}_{image_path.stem}.jpg"
            shutil.copy2(image_path, dest_image)

            label_path = YOLO_DATASET_DIR / "labels" / split / f"{barcode}_{image_path.stem}.txt"
            class_id = class_index[barcode]
            label_path.write_text(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n",
                encoding="utf-8",
            )

    names_block = "\n".join(f"  {idx}: {barcode}" for barcode, idx in class_index.items())
    data_yaml_path = YOLO_DATASET_DIR / "data.yaml"
    data_yaml_path.write_text(
        "path: {path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n{names}\n".format(path=str(YOLO_DATASET_DIR).replace("\\", "/"), names=names_block),
        encoding="utf-8",
    )
    logger.info("YOLO dataset prepared at %s (%d classes).", data_yaml_path, len(barcodes))
    return data_yaml_path
