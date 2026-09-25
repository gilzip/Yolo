"""Command-line entry point for running retail product detection.

Examples:
    python cli.py --source ./test_images/shelf.jpg --conf 0.4 --save-crops
    python cli.py --source ./test_images/aisle.mp4 --video --frame-stride 10
    python cli.py --source 0 --video --no-annotated
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from detector import DetectorInferenceError, DetectorInitError, RetailProductDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retail product detection CLI (YOLO11/YOLOv8).")
    parser.add_argument("--source", required=True, help="Path to an image/video file, or a webcam index (e.g. '0').")
    parser.add_argument("--model", default=None, help="Optional path to custom YOLO weights. Defaults to the configured/fallback model.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold (default from config: 0.35).")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold for NMS (default from config: 0.45).")
    parser.add_argument("--device", default=None, help="'auto', 'cpu', or 'cuda:0'. Default: auto.")
    parser.add_argument("--save-crops", action="store_true", help="Save a cropped image per detected product.")
    parser.add_argument("--no-annotated", action="store_true", help="Skip saving the annotated overview image.")
    parser.add_argument("--video", action="store_true", help="Treat --source as a video file or webcam index instead of a still image.")
    parser.add_argument("--frame-stride", type=int, default=5, help="Process every Nth frame in video mode (default: 5).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        detector = RetailProductDetector(
            model_path=args.model,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
        )
    except DetectorInitError as exc:
        logger.error("Failed to initialize detector: %s", exc)
        return 1

    try:
        if args.video:
            frame_results = detector.detect_video(
                args.source,
                save_crops=args.save_crops,
                save_annotated=not args.no_annotated,
                frame_stride=args.frame_stride,
            )
            total_products = sum(r["total_products_detected"] for r in frame_results)
            summary = {
                "frames_processed": len(frame_results),
                "total_products_detected": total_products,
            }
            logger.info("Processed %d frames, %d total product detections.", len(frame_results), total_products)
            print(json.dumps(summary, indent=2))
        else:
            payload = detector.detect_image(
                args.source,
                save_crops=args.save_crops,
                save_annotated=not args.no_annotated,
            )
            print(json.dumps(payload, indent=2))
    except DetectorInferenceError as exc:
        logger.error("Detection failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
