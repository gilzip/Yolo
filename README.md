# Retail Product Detector

Production-ready, end-to-end retail product detection for supermarket and
pharmacy shelves, built on **Ultralytics YOLO11 / YOLOv8**. Works out of the
box with pre-trained COCO weights (auto-downloaded) and transparently
upgrades to a purpose-trained retail/grocery model from **Roboflow
Universe** when API credentials are supplied — no manual dataset
generation or custom training required to get started.

## Features

- **`RetailProductDetector`** (`detector.py`) — inference wrapper with
  automatic CUDA/CPU selection, confidence + NMS filtering, per-product
  bounding-box crops, annotated-image rendering, and structured JSON output.
- **Model pipeline** (`retail_yolo_pipeline.py`) — resolves weights in
  order: explicit path → Roboflow Universe model → COCO-pretrained fallback
  (`yolo11m.pt` / `yolov8m.pt`).
- **REST API** (`app.py`) — FastAPI server with `POST /detect` and
  `GET /health`.
- **CLI** (`cli.py`) — run detection on an image, video file, or webcam
  directly from the terminal.

## Project layout

```
yolo/
├── config.py                  # Centralized configuration
├── retail_yolo_pipeline.py    # Model resolution / download logic
├── detector.py                # RetailProductDetector core class
├── app.py                     # FastAPI REST server
├── cli.py                     # Command-line tool
├── requirements.txt
├── setup_env.sh / setup_env.bat
├── models/                    # Cached/downloaded weights (git-ignored)
├── test_images/               # Put sample shelf photos here
└── outputs/
    ├── annotated/             # Annotated overview images
    ├── crops/                 # Per-product bounding-box crops
    └── json/                  # Structured detection payloads
```

## 1. Setup

**Windows:**
```bash
setup_env.bat
```

**macOS/Linux:**
```bash
chmod +x setup_env.sh
./setup_env.sh
```

This creates a `.venv` virtual environment and installs everything in
`requirements.txt`. Python 3.10+ is required.

### Optional: Roboflow retail model

By default the app uses the COCO-pretrained `yolo11m.pt` checkpoint. To use
a purpose-trained retail/grocery/shelf detection model from [Roboflow
Universe](https://universe.roboflow.com/) instead, set these environment
variables before running:

```bash
# Windows (PowerShell)
$env:ROBOFLOW_API_KEY = "your-api-key"
$env:ROBOFLOW_WORKSPACE = "your-workspace"
$env:ROBOFLOW_PROJECT = "grocery-dataset"
$env:ROBOFLOW_VERSION = "1"

# macOS/Linux
export ROBOFLOW_API_KEY="your-api-key"
export ROBOFLOW_WORKSPACE="your-workspace"
export ROBOFLOW_PROJECT="grocery-dataset"
export ROBOFLOW_VERSION="1"
```

If these are left unset (or the download fails for any reason), the
pipeline logs a warning and falls back to the COCO checkpoint automatically.

## 2. Run detection from the CLI

Place a test image in `test_images/`, then:

```bash
python cli.py --source ./test_images/shelf.jpg --conf 0.4 --save-crops
```

Video file or webcam:

```bash
python cli.py --source ./test_images/aisle.mp4 --video --frame-stride 10
python cli.py --source 0 --video
```

Useful flags:

| Flag              | Description                                         |
|-------------------|------------------------------------------------------|
| `--source`        | Image/video path, or webcam index (e.g. `0`)         |
| `--model`         | Path to custom `.pt` weights                         |
| `--conf`          | Confidence threshold (default `0.35`)                |
| `--iou`           | NMS IoU threshold (default `0.45`)                   |
| `--device`        | `auto` / `cpu` / `cuda:0`                            |
| `--save-crops`    | Save a cropped image per detected product            |
| `--no-annotated`  | Skip saving the annotated overview image              |
| `--video`         | Treat `--source` as video/webcam instead of an image  |
| `--frame-stride`  | Process every Nth frame in video mode (default `5`)  |

Output: a JSON payload printed to stdout and saved under
`outputs/json/`, plus annotated images (`outputs/annotated/`) and crops
(`outputs/crops/`) unless disabled.

## 3. Run the REST API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Check health / model status:**
```bash
curl http://localhost:8000/health
```

**Run detection on an uploaded image:**
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@./test_images/shelf.jpg"
```

Response shape:
```json
{
  "timestamp": "2026-09-25T12:00:00+00:00",
  "source": "shelf.jpg",
  "total_products_detected": 12,
  "products": [
    {
      "id": 0,
      "label": "bottle",
      "confidence": 0.87,
      "bbox": [120, 45, 210, 340],
      "crop_path": "outputs/crops/shelf_product_0.jpg"
    }
  ],
  "annotated_image_path": "outputs/annotated/shelf_annotated.jpg"
}
```

## Smart Retail POS desktop app

A separate PyQt5 desktop application — with webcam-based product onboarding
and a real-time smart-checkout/cart screen — lives in [`pos_app/`](pos_app/).
See [`pos_app/README.md`](pos_app/README.md) for details; run it with:

```bash
python -m pos_app.main
```

## Notes

- First run downloads the selected model checkpoint automatically (requires
  internet access once); subsequent runs use the cached copy.
- GPU is used automatically if CUDA is available; otherwise the app falls
  back to CPU without any configuration changes.
- To plug in your own trained weights, pass `--model path/to/weights.pt`
  (CLI) or set `model_path` when constructing `RetailProductDetector`.
