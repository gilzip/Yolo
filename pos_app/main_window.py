"""Main application window: tabs between Onboarding and Checkout screens.

Only one screen's webcam is active at a time — switching tabs stops the
outgoing screen's camera and starts the incoming one's, so both screens
never fight over the same camera device simultaneously.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QMainWindow, QTabWidget

from pos_app.checkout_screen import CheckoutScreen
from pos_app.onboarding_screen import OnboardingScreen


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Smart Retail POS")
        self.resize(1150, 720)

        self.onboarding_screen = OnboardingScreen()
        self.checkout_screen = CheckoutScreen()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.onboarding_screen, "Product Onboarding")
        self.tabs.addTab(self.checkout_screen, "Smart Checkout")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        self.onboarding_screen.start_camera()

    def _on_tab_changed(self, index: int) -> None:
        self.onboarding_screen.stop_camera()
        self.checkout_screen.stop_camera()
        if index == 0:
            self.onboarding_screen.start_camera()
        else:
            self.checkout_screen.start_camera()

    def closeEvent(self, event) -> None:
        self.onboarding_screen.stop_camera()
        self.checkout_screen.stop_camera()
        super().closeEvent(event)
