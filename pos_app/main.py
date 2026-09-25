"""Entry point for the Smart Retail POS PyQt5 application.

Run from the project root (C:\\yolo) so both this package and the root
`config`/`detector` modules are importable:

    python -m pos_app.main
"""

from __future__ import annotations

import logging
import sys

# Imported before PyQt5: on Windows, loading PyQt5's bundled Qt5 DLLs first
# causes torch's c10.dll to fail its init routine (WinError 1114). Importing
# torch first avoids the conflict for every later import in the process.
import torch  # noqa: F401

from PyQt5.QtWidgets import QApplication

from pos_app.main_window import MainWindow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
