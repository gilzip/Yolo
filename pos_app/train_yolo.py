"""Fine-tune YOLO on the onboarded product dataset.

Run after capturing products via the Onboarding screen:

    python -m pos_app.train_yolo --epochs 50

This prepares the YOLO-format dataset (see dataset_utils.prepare_yolo_dataset),
fine-tunes from the configured base checkpoint, and copies the resulting
best.pt to the path the Smart Checkout screen loads automatically.
"""

from __future__ import annotations

import argparse
import logging

from ultralytics import YOLO

from config import DETECTOR_CONFIG, MODELS_DIR
from pos_app.config import CUSTOM_MODEL_PATH
from pos_app.dataset_utils import prepare_yolo_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on onboarded retail products.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base-model", default=DETECTOR_CONFIG.fallback_model_name, help="Checkpoint to fine-tune from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = prepare_yolo_dataset()
    logger.info("Dataset ready at %s; starting fine-tuning from %s", data_yaml, args.base_model)

    model = YOLO(args.base_model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(MODELS_DIR),
        name="pos_finetune",
        # Without this, Ultralytics auto-versions the run dir on a rerun
        # (pos_finetune-2, -3, ...) instead of overwriting it, silently
        # breaking the hardcoded best.pt path below.
        exist_ok=True,
    )

    best_weights = MODELS_DIR / "pos_finetune" / "weights" / "best.pt"
    if best_weights.exists():
        CUSTOM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        best_weights.replace(CUSTOM_MODEL_PATH)
        logger.info("Fine-tuned model saved to %s — the Smart Checkout screen will use it automatically.", CUSTOM_MODEL_PATH)
    else:
        logger.warning("Training finished but best.pt was not found at %s", best_weights)


if __name__ == "__main__":
    main()
