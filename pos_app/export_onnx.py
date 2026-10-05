"""Export the MobileNetV3-Small embedder to ONNX for the browser.

Produces a model file that takes a (1, 3, 224, 224) float32 tensor —
already resized and ImageNet-normalized, matching `embeddings.compute_embedding`
— and returns the 576-d pooled feature vector (not yet L2-normalized; the
browser side normalizes it, same as `compute_embedding` does on the server).

Run with:
    python -m pos_app.export_onnx
"""

from __future__ import annotations

import logging

import torch

from pos_app.config import BASE_DIR
from pos_app.embeddings import get_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ONNX_OUTPUT_PATH = BASE_DIR / "static" / "embedder.onnx"


def export() -> None:
    model = get_model()
    model.eval()

    dummy_input = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    ONNX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_OUTPUT_PATH),
        input_names=["image"],
        output_names=["embedding"],
        opset_version=17,
        dynamo=False,
    )
    size_kb = ONNX_OUTPUT_PATH.stat().st_size / 1024
    logger.info("Exported ONNX embedder to %s (%.0f KB)", ONNX_OUTPUT_PATH, size_kb)


if __name__ == "__main__":
    export()
