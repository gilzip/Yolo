"""Evaluate vector recognition accuracy without a physical camera.

For every product photo already in the catalog, synthesizes a plausible
"different camera capture" of the same product (rotation, crop, brightness/
contrast jitter, light blur) and checks whether `find_nearest` still ranks
the correct barcode first against the *full* catalog index — i.e. a
realistic rehearsal of live checkout scanning, usable before any real photos
exist from an actual camera.

Run with:
    python -m pos_app.simulate_scans
    python -m pos_app.simulate_scans --threshold 0.75 --samples-per-product 3
"""

from __future__ import annotations

import argparse
import logging
import random

import cv2
import numpy as np

from pos_app.config import VECTOR_SIMILARITY_THRESHOLD
from pos_app.dataset_utils import get_catalog
from pos_app.embeddings import build_catalog_embeddings, compute_embedding, find_nearest

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def simulate_capture(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply a random, realistic-looking distortion to stand in for a fresh photo."""
    h, w = image.shape[:2]

    angle = rng.uniform(-15, 15)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, rng.uniform(0.92, 1.08))
    rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    margin_x, margin_y = rng.uniform(0.0, 0.12), rng.uniform(0.0, 0.12)
    x0, y0 = int(w * margin_x), int(h * margin_y)
    x1, y1 = int(w * (1 - rng.uniform(0.0, 0.12))), int(h * (1 - rng.uniform(0.0, 0.12)))
    cropped = rotated[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else rotated

    alpha = rng.uniform(0.85, 1.2)  # contrast
    beta = rng.uniform(-20, 20)  # brightness
    adjusted = cv2.convertScaleAbs(cropped, alpha=alpha, beta=beta)

    if rng.random() < 0.5:
        adjusted = cv2.GaussianBlur(adjusted, (3, 3), 0)

    return adjusted


def run_simulation(samples_per_product: int, threshold: float, seed: int) -> dict:
    rng = random.Random(seed)

    logger.info("Building the vector index from the current catalog...")
    build_stats = build_catalog_embeddings()
    logger.info(
        "Indexed %d images across %d products.", build_stats["images_embedded"], build_stats["products"]
    )

    catalog = get_catalog()
    results = []

    for barcode, product in catalog.items():
        image_paths = product.get("image_paths", [])
        if not image_paths:
            continue
        sample_paths = rng.sample(image_paths, k=min(samples_per_product, len(image_paths)))

        for path in sample_paths:
            original = cv2.imread(str(path))
            if original is None:
                continue
            simulated = simulate_capture(original, rng)
            query = compute_embedding(simulated)
            matches = find_nearest(query, top_k=5)
            if not matches:
                continue

            top1 = matches[0]
            correct_rank = next((i for i, m in enumerate(matches) if m["barcode"] == barcode), None)
            best_wrong = next((m["similarity"] for m in matches if m["barcode"] != barcode), None)

            results.append(
                {
                    "barcode": barcode,
                    "num_training_images": len(image_paths),
                    "top1_barcode": top1["barcode"],
                    "top1_similarity": top1["similarity"],
                    "correct_in_top5": correct_rank is not None,
                    "correct_rank": correct_rank,
                    "top1_correct": top1["barcode"] == barcode,
                    "best_wrong_similarity": best_wrong,
                }
            )

    total = len(results)
    top1_correct = sum(r["top1_correct"] for r in results)
    top5_correct = sum(r["correct_in_top5"] for r in results)

    single_image = [r for r in results if r["num_training_images"] == 1]
    multi_image = [r for r in results if r["num_training_images"] > 1]

    would_accept = [r for r in results if r["top1_similarity"] >= threshold]
    would_accept_and_correct = [r for r in would_accept if r["top1_correct"]]

    summary = {
        "total_simulated_scans": total,
        "top1_accuracy": top1_correct / total if total else 0.0,
        "top5_accuracy": top5_correct / total if total else 0.0,
        "single_image_products": {
            "count": len(single_image),
            "top1_accuracy": sum(r["top1_correct"] for r in single_image) / len(single_image) if single_image else 0.0,
        },
        "multi_image_products": {
            "count": len(multi_image),
            "top1_accuracy": sum(r["top1_correct"] for r in multi_image) / len(multi_image) if multi_image else 0.0,
        },
        "at_current_threshold": {
            "threshold": threshold,
            "scans_accepted": len(would_accept),
            "accepted_and_correct": len(would_accept_and_correct),
            "precision_if_accepted": (len(would_accept_and_correct) / len(would_accept)) if would_accept else None,
            "recall": (len(would_accept_and_correct) / total) if total else None,
        },
    }
    return {"summary": summary, "details": results}


def _print_report(report: dict) -> None:
    s = report["summary"]
    logger.info("")
    logger.info("=== Simulated-scan results (%d scans) ===", s["total_simulated_scans"])
    logger.info("Top-1 accuracy (any training-image count): %.1f%%", s["top1_accuracy"] * 100)
    logger.info("Top-5 accuracy: %.1f%%", s["top5_accuracy"] * 100)
    logger.info(
        "Single-image products (%d): top-1 accuracy %.1f%%",
        s["single_image_products"]["count"],
        s["single_image_products"]["top1_accuracy"] * 100,
    )
    logger.info(
        "Multi-image products (%d): top-1 accuracy %.1f%%",
        s["multi_image_products"]["count"],
        s["multi_image_products"]["top1_accuracy"] * 100,
    )
    at = s["at_current_threshold"]
    logger.info("")
    logger.info("At similarity threshold %.2f:", at["threshold"])
    logger.info("  Scans that would be auto-accepted: %d / %d", at["scans_accepted"], s["total_simulated_scans"])
    if at["precision_if_accepted"] is not None:
        logger.info("  Of those, actually correct: %.1f%% (precision)", at["precision_if_accepted"] * 100)
    logger.info("  Recall (correct scans that clear the bar): %.1f%%", (at["recall"] or 0) * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-product", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=VECTOR_SIMILARITY_THRESHOLD)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = run_simulation(args.samples_per_product, args.threshold, args.seed)
    _print_report(report)
