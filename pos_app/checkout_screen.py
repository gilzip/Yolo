"""Smart Checkout screen (Inference / POS Mode).

Runs live YOLO inference on the webcam feed, draws bounding boxes with
confidence scores, and automatically adds a product to the shopping cart
once it's been detected consistently, above the confidence threshold, over
several consecutive frames — with a cooldown to prevent duplicate scans.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pos_app.camera_worker import CameraWorker
from pos_app.cart import ShoppingCart
from pos_app.config import (
    CAMERA_INDEX,
    MAX_SUGGESTIONS,
    POS_CONFIDENCE_THRESHOLD,
    SUGGESTION_MIN_CONFIDENCE,
    THUMBNAIL_SIZE,
)
from pos_app.dataset_utils import get_catalog, get_product, record_checkout_training_sample
from pos_app.orders import save_order
from pos_app.pos_detector import load_checkout_detector

logger = logging.getLogger(__name__)

try:
    import winsound  # Windows-only; provides the duplicate-scan-prevention beep

    def _beep() -> None:
        winsound.Beep(1200, 120)

except ImportError:  # pragma: no cover - non-Windows platforms get a silent no-op

    def _beep() -> None:
        pass


def _bgr_to_pixmap(image: np.ndarray, size: int | None = None) -> QPixmap:
    """Convert a BGR numpy image (e.g. from OpenCV) to a QPixmap, optionally scaled to a square icon size."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qt_image)
    if size is not None:
        pixmap = pixmap.scaled(size, size, Qt.KeepAspectRatio)
    return pixmap


class CheckoutScreen(QWidget):
    """Runs live YOLO inference and builds a shopping cart from consistent detections."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._camera: CameraWorker | None = None
        self._detector = None
        self._latest_frame: np.ndarray | None = None
        self.cart = ShoppingCart()
        self._flash_until = 0.0
        self._suggestion_buttons: list[QPushButton] = []
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.video_label = QLabel("Camera feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #111; color: white;")

        self.model_status_label = QLabel("Loading detection model...")
        self.model_status_label.setWordWrap(True)

        self.suggestions_label = QLabel("")
        self.suggestions_label.setWordWrap(True)
        self.suggestions_layout = QHBoxLayout()
        self.suggestions_layout.addStretch(1)

        self.manual_add_button = QPushButton("+ Add Manually")
        self.manual_add_button.setToolTip("Pick a product from the catalog by hand if the camera can't identify it.")
        self.manual_add_button.clicked.connect(self._open_manual_add_dialog)

        self.barcode_entry_input = QLineEdit()
        self.barcode_entry_input.setPlaceholderText("Scan or type barcode...")
        self.barcode_entry_input.returnPressed.connect(self._add_by_barcode)

        self.barcode_entry_button = QPushButton("Add")
        self.barcode_entry_button.clicked.connect(self._add_by_barcode)

        barcode_entry_layout = QHBoxLayout()
        barcode_entry_layout.addWidget(self.barcode_entry_input)
        barcode_entry_layout.addWidget(self.barcode_entry_button)

        self.cart_table = QTableWidget(0, 5)
        self.cart_table.setHorizontalHeaderLabels(["Thumbnail", "Barcode", "Item Name", "Qty", "Price"])
        self.cart_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cart_table.verticalHeader().setVisible(False)
        self.cart_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.total_label = QLabel("Total: $0.00")
        self.total_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        self.complete_order_button = QPushButton("Complete Order")
        self.complete_order_button.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold; padding: 8px;"
        )
        self.complete_order_button.clicked.connect(self._complete_order)

        self.clear_button = QPushButton("Cancel Order")
        self.clear_button.clicked.connect(self._clear_cart)

        cart_header = QLabel("Shopping Cart")
        cart_header.setStyleSheet("font-size: 16px; font-weight: bold;")

        side_layout = QVBoxLayout()
        side_layout.addWidget(self.model_status_label)
        side_layout.addWidget(self.suggestions_label)
        side_layout.addLayout(self.suggestions_layout)
        side_layout.addWidget(self.manual_add_button)
        side_layout.addLayout(barcode_entry_layout)
        side_layout.addWidget(cart_header)
        side_layout.addWidget(self.cart_table, stretch=1)
        side_layout.addWidget(self.total_label)
        side_layout.addWidget(self.complete_order_button)
        side_layout.addWidget(self.clear_button)

        side_widget = QWidget()
        side_widget.setLayout(side_layout)
        side_widget.setFixedWidth(420)

        root_layout = QHBoxLayout(self)
        root_layout.addWidget(self.video_label, stretch=2)
        root_layout.addWidget(side_widget, stretch=0)

    # -------------------------------------------------------------- lifecycle
    def start_camera(self) -> None:
        if self._detector is None:
            try:
                self._detector = load_checkout_detector()
                model_kind = (
                    "custom fine-tuned model (recognizes onboarded barcodes)"
                    if self._detector.using_custom_model
                    else "generic fallback model — train a custom model (pos_app/train_yolo.py) to recognize barcodes"
                )
                self.model_status_label.setText(f"Model: {model_kind}")
            except Exception as exc:  # noqa: BLE001 - surface load failure in the UI instead of crashing
                self.model_status_label.setText(f"Model failed to load: {exc}")
                logger.error("Failed to load checkout detector: %s", exc)
                return

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

    # ---------------------------------------------------------------- frames
    def _on_frame(self, frame: np.ndarray) -> None:
        self._latest_frame = frame
        if self._detector is None:
            return

        detections = self._detector.infer(frame)
        seen_barcodes: set[str] = set()
        weak_candidates: list[tuple[str, float, tuple[int, int, int, int]]] = []
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            is_confident = det.confidence >= POS_CONFIDENCE_THRESHOLD
            color = (0, 200, 0) if is_confident else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            caption = f"{det.label} {det.confidence:.2f}"
            cv2.putText(annotated, caption, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if not is_confident:
                if det.confidence >= SUGGESTION_MIN_CONFIDENCE:
                    weak_candidates.append((det.label, det.confidence, det.bbox))
                continue

            barcode = det.label
            seen_barcodes.add(barcode)
            if self.cart.observe(barcode):
                self._scan_to_cart(barcode, frame, det.bbox)

        self.cart.decay_missing(seen_barcodes)
        self._update_suggestions(frame, weak_candidates)

        if time.monotonic() < self._flash_until:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1] - 1, annotated.shape[0] - 1), (0, 255, 0), 12)

        self._render_frame(annotated)

    def _render_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pixmap)

    # ------------------------------------------------------------ suggestions
    def _update_suggestions(
        self,
        frame: np.ndarray,
        weak_candidates: list[tuple[str, float, tuple[int, int, int, int]]],
    ) -> None:
        """Show clickable "possible match" buttons for detections too weak to auto-add.

        Lets the operator manually confirm a match the camera noticed but
        wasn't confident enough to add on its own — instead of silently
        discarding it.
        """
        for button in self._suggestion_buttons:
            self.suggestions_layout.removeWidget(button)
            button.deleteLater()
        self._suggestion_buttons = []

        if not weak_candidates:
            self.suggestions_label.setText("")
            return

        self.suggestions_label.setText("Possible match — click to confirm:")

        best_per_barcode: dict[str, tuple[float, tuple[int, int, int, int]]] = {}
        for barcode, confidence, bbox in weak_candidates:
            if barcode not in best_per_barcode or confidence > best_per_barcode[barcode][0]:
                best_per_barcode[barcode] = (confidence, bbox)
        top_candidates = sorted(best_per_barcode.items(), key=lambda kv: kv[1][0], reverse=True)[:MAX_SUGGESTIONS]

        h, w = frame.shape[:2]
        for barcode, (confidence, bbox) in top_candidates:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None

            product = get_product(barcode)
            label = product["name"] if product else barcode

            button = QPushButton(f"{label}\n{confidence:.0%}")
            if crop is not None and crop.size > 0:
                button.setIcon(QIcon(_bgr_to_pixmap(crop, THUMBNAIL_SIZE)))
                button.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
            button.clicked.connect(
                lambda checked=False, b=barcode, f=frame, bb=bbox: self._confirm_suggestion(b, f, bb)
            )
            self.suggestions_layout.insertWidget(self.suggestions_layout.count() - 1, button)
            self._suggestion_buttons.append(button)

    def _confirm_suggestion(self, barcode: str, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        self.cart.mark_scanned(barcode)
        self._scan_to_cart(barcode, frame, bbox)

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        training_crop = frame[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else frame
        record_checkout_training_sample(barcode, training_crop)

    def _open_manual_add_dialog(self) -> None:
        """Let the operator pick a product from the full catalog by hand.

        A fallback for when the camera can't identify the item at all (poor
        lighting, an unusual angle, or a product the model hasn't learned
        well yet).
        """
        catalog = get_catalog()
        if not catalog:
            QMessageBox.information(self, "No products yet", "Onboard at least one product first.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Add Product Manually")
        dialog.resize(420, 400)
        layout = QVBoxLayout(dialog)

        list_widget = QListWidget()
        list_widget.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        for barcode in sorted(catalog.keys()):
            product = catalog[barcode]
            item = QListWidgetItem(f"{product.get('name', barcode)}  —  {barcode}  (${product.get('price', 0.0):.2f})")
            front_path = product.get("front_image_path")
            if front_path and Path(front_path).exists():
                item.setIcon(QIcon(front_path))
            item.setData(Qt.UserRole, barcode)
            list_widget.addItem(item)
        layout.addWidget(list_widget)

        button_row = QHBoxLayout()
        add_button = QPushButton("Add to Cart")
        cancel_button = QPushButton("Cancel")
        button_row.addWidget(add_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

        def confirm_selection() -> None:
            item = list_widget.currentItem()
            if item is None:
                return
            barcode = item.data(Qt.UserRole)
            product = catalog[barcode]
            self.cart.mark_scanned(barcode)
            self.cart.add_or_increment(barcode, product.get("name", barcode), product.get("price", 0.0))
            self._refresh_cart_table()
            if self._latest_frame is not None:
                record_checkout_training_sample(barcode, self._latest_frame)
            dialog.accept()

        add_button.clicked.connect(confirm_selection)
        cancel_button.clicked.connect(dialog.reject)
        list_widget.itemDoubleClicked.connect(lambda _: confirm_selection())
        dialog.exec_()

    def _add_by_barcode(self) -> None:
        """Add a product by typed or scanned barcode — the last-resort fallback.

        Works with a physical USB barcode scanner too: those act as a
        keyboard (type the digits + Enter), so scanning into this focused
        field behaves exactly like typing it manually. Known or not, the
        current frame is saved as a human-confirmed training sample.
        """
        barcode = self.barcode_entry_input.text().strip()
        if not barcode:
            return

        product = get_product(barcode)
        name = product["name"] if product else barcode
        price = product["price"] if product else 0.0

        self.cart.mark_scanned(barcode)
        self.cart.add_or_increment(barcode, name, price)
        self._refresh_cart_table()

        if self._latest_frame is not None:
            record_checkout_training_sample(barcode, self._latest_frame)

        self.barcode_entry_input.clear()

    # ----------------------------------------------------------------- cart
    def _scan_to_cart(self, barcode: str, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        product = get_product(barcode)
        name = product["name"] if product else f"Unknown ({barcode})"
        price = product["price"] if product else 0.0

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        thumbnail = frame[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None

        self.cart.add_or_increment(barcode, name, price, thumbnail)
        self._flash_until = time.monotonic() + 0.4
        _beep()
        self._refresh_cart_table()

    def _refresh_cart_table(self) -> None:
        self.cart_table.setRowCount(len(self.cart.items))
        for row, item in enumerate(self.cart.items.values()):
            thumb_label = QLabel()
            thumb_label.setAlignment(Qt.AlignCenter)
            if item.thumbnail is not None:
                thumb_label.setPixmap(_bgr_to_pixmap(item.thumbnail, THUMBNAIL_SIZE))
            self.cart_table.setCellWidget(row, 0, thumb_label)

            self.cart_table.setItem(row, 1, QTableWidgetItem(item.barcode))
            self.cart_table.setItem(row, 2, QTableWidgetItem(item.name))
            self.cart_table.setItem(row, 3, QTableWidgetItem(str(item.quantity)))
            self.cart_table.setItem(row, 4, QTableWidgetItem(f"${item.price:.2f}"))

        self.total_label.setText(f"Total: ${self.cart.total():.2f}")

    def _clear_cart(self) -> None:
        self.cart.clear()
        self._refresh_cart_table()

    def _complete_order(self) -> None:
        """Finalize the sale: confirm the itemized total, save a receipt, and reset the cart."""
        if not self.cart.items:
            QMessageBox.information(self, "Cart is empty", "Add at least one product before completing the order.")
            return

        items_payload = [
            {
                "barcode": item.barcode,
                "name": item.name,
                "quantity": item.quantity,
                "price": item.price,
                "subtotal": round(item.price * item.quantity, 2),
            }
            for item in self.cart.items.values()
        ]
        total = self.cart.total()
        summary = "\n".join(f"{it['quantity']}x {it['name']} — ${it['subtotal']:.2f}" for it in items_payload)

        confirmed = QMessageBox.question(
            self,
            "Complete Order",
            f"{summary}\n\nTotal: ${total:.2f}\n\nComplete this order?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirmed != QMessageBox.Yes:
            return

        try:
            save_order(items_payload, total)
        except OSError as exc:
            logger.warning("Could not save order receipt: %s", exc)

        QMessageBox.information(self, "Order Complete", f"Order complete! Total charged: ${total:.2f}")
        self._clear_cart()
