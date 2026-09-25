"""Background webcam capture thread for the PyQt5 UI.

Runs `cv2.VideoCapture.read()` on a dedicated QThread so the GUI event loop
never blocks on frame grabs, and emits each captured frame as a numpy array
via a Qt signal for main-thread UI code to render.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class CameraWorker(QThread):
    """Continuously reads frames from a webcam and emits them as they arrive."""

    frame_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, camera_index: int = 0, parent=None) -> None:
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False

    def run(self) -> None:
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            self.error.emit(f"Could not open camera index {self.camera_index}.")
            return

        self._running = True
        try:
            while self._running:
                ok, frame = capture.read()
                if not ok:
                    self.error.emit("Camera read failed; stopping feed.")
                    break
                self.frame_ready.emit(frame)
                self.msleep(15)  # yields to the event loop; actual rate is bound by camera + inference
        finally:
            capture.release()

    def stop(self) -> None:
        self._running = False
        self.wait(2000)
