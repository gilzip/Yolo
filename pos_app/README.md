# Smart Retail POS (PyQt5 + YOLO)

A two-screen desktop application for onboarding retail products via webcam
and then recognizing them at checkout in real time.

Built on top of the root project's YOLO model pipeline (`config.py`,
`retail_yolo_pipeline.py`) — it starts working immediately with the
COCO-pretrained fallback model, then gets barcode-accurate once you onboard
your own products and fine-tune.

## Screens

### 1. Product Onboarding (Training Mode)

- Live webcam feed with a barcode / name / price form.
- **Automatic barcode detection**: hold the product's barcode up to the
  camera and the Barcode field fills in by itself (via `pyzbar`) — no
  typing needed. It only scans while idle (no active capture session) and
  never overwrites a field that already has a value.
- **Start Capture** grabs 18 frames (configurable) at a timed interval while
  you rotate the product in front of the camera — or, with auto-capture
  unchecked, click the video area and press **Spacebar** to capture each
  frame manually.
- **Capture Front Photo** saves one extra reference photo of the product's
  front/label side (usually where the printed name is), separate from the
  rotation burst, stored at `dataset/images/<barcode>/front.jpg` and linked
  in the catalog as `front_image_path`.
- **Finish & Next Product** ends the current capture session at any point
  (even before the full 18 frames), saves whatever was captured, pops up a
  confirmation with the count, and clears the form — so it's always obvious
  when one product is done and the app is ready for the next barcode.
- Images are saved to `dataset/images/<barcode>/img_N.jpg`; `dataset/products.json`
  is updated with the barcode → name/price/image-paths mapping. Re-onboarding
  an existing barcode continues the file numbering instead of overwriting
  earlier captures.

### 2. Smart Checkout (Inference Mode)

- Runs YOLO on the live feed, drawing a bounding box + confidence score for
  every detection (green if above the cart threshold, orange otherwise).
- A product is added to the cart once it's detected above **80% confidence**
  for **5 consecutive frames** (both configurable). A **3-second cooldown**
  per barcode then prevents the same physical item from being scanned twice
  in a row, with a beep + green flash as feedback; a repeated scan after the
  cooldown increments quantity instead of adding a new row.
- **Possible-match suggestions**: a detection too weak to auto-add (above
  10% confidence but below the 80% cart threshold) shows up as a clickable
  thumbnail button — "click to confirm" — instead of being silently dropped.
- **+ Add Manually**: opens the full product catalog (with photos) so the
  operator can add an item by hand when the camera can't identify it at all.
- **Scan or type barcode**: a last-resort text field that also doubles as
  the input for a physical USB barcode scanner (those act as a keyboard —
  scanning into this focused field just types the digits + Enter).
- **Continuous learning loop**: every *manual* confirmation (a suggestion
  click, a catalog pick, or a typed/scanned barcode) saves the current
  checkout-camera frame as a new training image for that barcode and folds
  it into the catalog. Automatic high-confidence scans are **not** recorded
  this way (they're not human-verified). Re-running `train_yolo.py`
  incorporates these real checkout-condition photos, so accuracy keeps
  improving from actual use — not just the initial onboarding session.
- The cart table shows a cropped thumbnail, barcode, item name, quantity,
  and price, with a running total.
- **Complete Order**: shows an itemized confirmation (like an e-commerce
  checkout), then saves a JSON receipt to `orders/` and clears the cart for
  the next customer. **Cancel Order** clears the cart without saving a receipt.

## Setup

Uses the same virtual environment as the rest of the project (`PyQt5` was
added to the root `requirements.txt`):

```bash
cd C:\yolo
setup_env.bat            # or ./setup_env.sh on macOS/Linux
```

## Running

Always run from the project root so both `pos_app` and the root `config`/
`detector` modules resolve correctly:

```bash
python -m pos_app.main
```

On first launch, Checkout mode falls back to the generic COCO/Roboflow
model — it will detect generic object classes (e.g. "bottle"), not
barcodes, until you onboard products and fine-tune.

## End-to-end workflow

1. **Onboard your products** — launch the app, go to *Product Onboarding*,
   and capture 15–20 images per product (repeat for every SKU you want
   recognized).
2. **Fine-tune YOLO on your catalog:**
   ```bash
   python -m pos_app.train_yolo --epochs 50
   ```
   This converts `dataset/images/` into a YOLO detection dataset
   (`dataset/yolo/`), using each barcode as its own class, then fine-tunes
   from the configured base checkpoint and saves the result to
   `models/pos_custom.pt`.
3. **Switch to Smart Checkout** — it automatically picks up
   `models/pos_custom.pt` if present, so detected labels are now your
   barcodes, matched against `dataset/products.json` for name/price.

### About the auto-generated bounding boxes

Onboarding photos are single-product close-ups with no hand-drawn bounding
boxes. `dataset_utils.prepare_yolo_dataset()` approximates one per image by
running the generic detector and keeping its largest box, falling back to a
centered box covering most of the frame when nothing is detected. This is a
heuristic, not ground truth — spot-check a few generated `.txt` labels in
`dataset/yolo/labels/` before a long training run, and re-capture/relabel
manually if a product's boxes look off.

## Configuration

All defaults live in `pos_app/config.py` and can be overridden via
environment variables:

| Variable                     | Default | Meaning                                   |
|-------------------------------|---------|--------------------------------------------|
| `POS_CAMERA_INDEX`            | `0`     | Webcam device index                        |
| `POS_CAPTURE_COUNT`           | `18`    | Frames captured per onboarding session     |
| `POS_CAPTURE_INTERVAL_MS`     | `350`   | Delay between auto-captured frames         |
| `POS_CONFIDENCE_THRESHOLD`    | `0.80`  | Minimum confidence to count toward the cart|
| `POS_CONSISTENCY_FRAMES`      | `5`     | Consecutive frames required before adding  |
| `POS_COOLDOWN_SECONDS`        | `3.0`   | Duplicate-scan cooldown per barcode        |
| `POS_CUSTOM_MODEL`            | `pos_custom.pt` | Fine-tuned weights filename in `models/` |

## Files

```
pos_app/
├── config.py            # POS-specific settings (paths, thresholds, camera)
├── camera_worker.py      # QThread webcam capture
├── dataset_utils.py      # Save captures, products.json, YOLO dataset prep,
│                         # + continuous-learning feedback from checkout
├── pos_detector.py        # Real-time YOLO wrapper (no per-frame disk I/O)
├── cart.py                # Consistency/cooldown tracking + cart model
├── orders.py               # Completed-order JSON receipts
├── onboarding_screen.py   # Product Onboarding screen widget
├── checkout_screen.py     # Smart Checkout screen widget
├── main_window.py         # Tab container, camera lifecycle switching
├── main.py                # Entry point
└── train_yolo.py          # Fine-tuning CLI
```
