"""Data Collection & Product Onboarding screen (Training Mode).

Lets an operator point the webcam at a real product, tag it with a barcode
(plus a name and price so the checkout cart can display them later), and
capture a burst of multi-angle images while rotating the item in front of
the camera.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

from pos_app.camera_worker import CameraWorker
from pos_app.config import CAMERA_INDEX, CAPTURE_COUNT, CAPTURE_INTERVAL_MS
from pos_app.dataset_utils import (
    next_capture_index,
    register_product,
    save_capture_frame,
    save_front_image,
)

logger = logging.getLogger(__name__)

try:
    from pyzbar.pyzbar import decode as decode_barcodes

    _BARCODE_SCAN_AVAILABLE = True
except ImportError:  # pragma: no cover - degrade gracefully if pyzbar/its native lib is missing
    _BARCODE_SCAN_AVAILABLE = False
    logger.warning("pyzbar not available; automatic barcode detection is disabled (pip install pyzbar).")

# Only re-run the (relatively expensive) barcode decode every Nth frame while idle.
_BARCODE_SCAN_FRAME_STRIDE = 5


class OnboardingScreen(QWidget):
    """Captures labeled multi-angle product photos for later YOLO fine-tuning."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._camera: CameraWorker | None = None
        self._latest_frame: np.ndarray | None = None
        self._capture_index = 0
        self._file_index_offset = 0
        self._capture_paths: list = []
        self._barcode_scan_counter = 0
        self._barcode_highlight: tuple[int, int, int, int] | None = None
        self._barcode_highlight_until = 0.0

        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._capture_one_frame)

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.video_label = QLabel("Camera feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #111; color: white;")
        # Space-to-capture only fires while the video area itself has focus,
        # so typing a space into the name field doesn't accidentally trigger it.
        self.video_label.setFocusPolicy(Qt.StrongFocus)
        space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self.video_label)
        space_shortcut.setContext(Qt.WidgetShortcut)
        space_shortcut.activated.connect(self._on_spacebar)

        self.barcode_input = QLineEdit()
        barcode_hint = (
            "Barcode (e.g. 7290000000001) — auto-filled from the camera"
            if _BARCODE_SCAN_AVAILABLE
            else "Barcode (e.g. 7290000000001)"
        )
        self.barcode_input.setPlaceholderText(barcode_hint)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Product name")

        self.price_input = QDoubleSpinBox()
        self.price_input.setRange(0.0, 100000.0)
        self.price_input.setDecimals(2)
        self.price_input.setPrefix("$ ")

        self.front_photo_button = QPushButton("Capture Front Photo")
        self.front_photo_button.setToolTip(
            "Saves one reference photo of the product's front (label side), "
            "which usually shows the printed product name."
        )
        self.front_photo_button.clicked.connect(self._capture_front_photo)

        self.auto_checkbox = QCheckBox("Auto-capture (timed)")
        self.auto_checkbox.setChecked(True)

        self.start_button = QPushButton("Start Capture")
        self.start_button.clicked.connect(self._start_capture)

        self.finish_button = QPushButton("Finish && Next Product")
        self.finish_button.setEnabled(False)
        self.finish_button.clicked.connect(self._finish_capture)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, CAPTURE_COUNT)

        self.status_label = QLabel(
            "Point the camera at the barcode to auto-fill it (or type it manually), "
            "add a name and price, then click Start Capture and rotate the product "
            "in front of the camera.\nClick the video area, then press Spacebar "
            "any time to capture an extra frame manually."
        )
        self.status_label.setWordWrap(True)

        form_layout = QVBoxLayout()
        form_layout.addWidget(QLabel("Barcode"))
        form_layout.addWidget(self.barcode_input)
        form_layout.addWidget(QLabel("Product name"))
        form_layout.addWidget(self.name_input)
        form_layout.addWidget(QLabel("Price"))
        form_layout.addWidget(self.price_input)
        form_layout.addWidget(self.front_photo_button)
        form_layout.addWidget(self.auto_checkbox)
        form_layout.addWidget(self.start_button)
        form_layout.addWidget(self.finish_button)
        form_layout.addWidget(self.progress_bar)
        form_layout.addWidget(self.status_label)
        form_layout.addStretch(1)

        side_panel = QWidget()
        side_panel.setLayout(form_layout)
        side_panel.setFixedWidth(300)

        root_layout = QHBoxLayout(self)
        root_layout.addWidget(self.video_label, stretch=2)
        root_layout.addWidget(side_panel, stretch=0)

    # --------------------------------------------------------------- camera
    def start_camera(self) -> None:
        if self._camera is not None:
            return
        self._camera = CameraWorker(CAMERA_INDEX)
        self._camera.frame_ready.connect(self._on_frame)
        self._camera.error.connect(self._on_camera_error)
        self._camera.start()

    def stop_camera(self) -> None:
        if self._camera is not None:
            self._camera.frame_ready.disconnect(self._on_frame)
            self._camera.error.disconnect(self._on_camera_error)
            self._camera.stop()
            self._camera = None

    def _on_camera_error(self, message: str) -> None:
        self.video_label.setText(message)
        logger.warning(message)

    def _on_frame(self, frame: np.ndarray) -> None:
        self._latest_frame = frame
        self._maybe_scan_barcode(frame)
        self._render_frame(frame)

    def _maybe_scan_barcode(self, frame: np.ndarray) -> None:
        """Auto-fill the barcode field by decoding a barcode held up to the camera.

        Only runs while idle (no capture session active) and the field is
        still empty, so it never fights a manually typed barcode or interferes
        with an in-progress rotation burst.
        """
        if not _BARCODE_SCAN_AVAILABLE:
            return
        if not self.start_button.isEnabled():
            return  # a capture session is active
        if self.barcode_input.text().strip():
            return  # already have a barcode

        self._barcode_scan_counter = (self._barcode_scan_counter + 1) % _BARCODE_SCAN_FRAME_STRIDE
        if self._barcode_scan_counter != 0:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            results = decode_barcodes(gray)
        except Exception as exc:  # noqa: BLE001 - a decode hiccup must never crash the camera loop
            logger.warning("Barcode decode failed: %s", exc)
            return

        if results:
            value = results[0].data.decode("utf-8", errors="ignore").strip()
            if value:
                self.barcode_input.setText(value)
                self.status_label.setText(f"Barcode detected automatically: {value}")
                # Visual "focus" feedback: highlight where the barcode was
                # found for a moment, so it's clear the camera actually
                # locked onto it rather than the field just silently filling.
                rect = results[0].rect
                self._barcode_highlight = (rect.left, rect.top, rect.left + rect.width, rect.top + rect.height)
                self._barcode_highlight_until = time.monotonic() + 1.5

    def _render_frame(self, frame: np.ndarray) -> None:
        display_frame = frame
        if self._barcode_highlight is not None and time.monotonic() < self._barcode_highlight_until:
            display_frame = frame.copy()
            x1, y1, x2, y2 = self._barcode_highlight
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                display_frame, "Barcode detected", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pixmap)

    # ------------------------------------------------------------- capture
    def _start_capture(self) -> None:
        barcode = self.barcode_input.text().strip()
        if not barcode:
            QMessageBox.warning(self, "Missing barcode", "Enter a barcode before starting capture.")
            return
        if self._latest_frame is None:
            QMessageBox.warning(self, "Camera not ready", "Waiting for the camera feed to start.")
            return

        self._capture_index = 0
        self._capture_paths = []
        # Continue file numbering from what's already on disk for this
        # barcode, so re-onboarding it doesn't overwrite earlier captures.
        self._file_index_offset = next_capture_index(barcode) - 1
        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)
        self.finish_button.setEnabled(True)

        if self.auto_checkbox.isChecked():
            self.status_label.setText(f"Capturing images for barcode {barcode}... rotate the product slowly.")
            self._auto_timer.start(CAPTURE_INTERVAL_MS)
        else:
            self.status_label.setText(
                f"Capturing images for barcode {barcode} — click the video area and "
                "press Spacebar to capture each frame."
            )

    def _on_spacebar(self) -> None:
        if self.start_button.isEnabled():
            return  # no active capture session
        self._capture_one_frame()

    def _capture_one_frame(self) -> None:
        if self._latest_frame is None or self._capture_index >= CAPTURE_COUNT:
            self._finish_capture()
            return

        barcode = self.barcode_input.text().strip()
        try:
            path = save_capture_frame(barcode, self._latest_frame, self._file_index_offset + self._capture_index + 1)
            self._capture_paths.append(path)
        except IOError as exc:
            logger.error("Failed to save capture frame: %s", exc)

        self._capture_index += 1
        self.progress_bar.setValue(self._capture_index)

        if self._capture_index >= CAPTURE_COUNT:
            self._finish_capture()

    def _capture_front_photo(self) -> None:
        """Save one reference photo of the product's front (label/name side).

        This is separate from the rotation burst — it's meant to capture the
        printed product name/branding clearly, which is often useful later
        for identifying a product (e.g. one onboarded without a typed name).
        """
        barcode = self.barcode_input.text().strip()
        if not barcode:
            QMessageBox.warning(self, "Missing barcode", "Enter or scan a barcode first.")
            return
        if self._latest_frame is None:
            QMessageBox.warning(self, "Camera not ready", "Waiting for the camera feed to start.")
            return

        try:
            front_path = save_front_image(barcode, self._latest_frame)
        except IOError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        register_product(
            barcode=barcode,
            name=self.name_input.text().strip(),
            price=self.price_input.value(),
            image_paths=[front_path],
            front_image_path=front_path,
        )
        self.status_label.setText(f"Saved front-of-package photo for barcode {barcode}.")

    def _finish_capture(self) -> None:
        """Stop the current capture session (whether it ran to completion or
        was ended early via the Finish button), save what was captured, and
        reset the form so the operator can clearly move on to the next product.
        """
        self._auto_timer.stop()
        barcode = self.barcode_input.text().strip()
        captured_count = len(self._capture_paths)

        if self._capture_paths:
            register_product(
                barcode=barcode,
                name=self.name_input.text().strip(),
                price=self.price_input.value(),
                image_paths=self._capture_paths,
            )

        self.start_button.setEnabled(True)
        self.finish_button.setEnabled(False)

        if captured_count:
            QMessageBox.information(
                self,
                "Capture complete",
                f"Saved {captured_count} images for barcode {barcode}.\n"
                "The form is now cleared for the next product.",
            )

        self._reset_form_for_next_product()

    def _reset_form_for_next_product(self) -> None:
        self.barcode_input.clear()
        self.name_input.clear()
        self.price_input.setValue(0.0)
        self.progress_bar.setValue(0)
        self._capture_index = 0
        self._file_index_offset = 0
        self._capture_paths = []
        self.status_label.setText(
            "Point the camera at the barcode to auto-fill it (or type it manually), "
            "then click Start Capture for the next product."
        )
