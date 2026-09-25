"""Shopping cart model: tracks per-barcode detection streaks, cooldowns, and cart contents."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pos_app.config import CONSISTENCY_FRAMES, COOLDOWN_SECONDS


@dataclass
class CartItem:
    """One line in the shopping cart."""

    barcode: str
    name: str
    price: float
    quantity: int = 1
    thumbnail: Any = None  # BGR numpy crop for the cart-table thumbnail


class ShoppingCart:
    """Tracks detection consistency/cooldown per barcode and the resulting cart lines."""

    def __init__(self) -> None:
        self._streaks: dict[str, int] = {}
        self._last_added: dict[str, float] = {}
        self.items: dict[str, CartItem] = {}

    def observe(self, barcode: str) -> bool:
        """Register one more consistent high-confidence detection frame for `barcode`.

        Returns True exactly when this observation should trigger a
        scan-to-cart event: the detection streak is long enough AND the
        per-barcode cooldown (duplicate-scan prevention) has elapsed.
        """
        self._streaks[barcode] = self._streaks.get(barcode, 0) + 1
        if self._streaks[barcode] < CONSISTENCY_FRAMES:
            return False

        if not self._cooldown_elapsed(barcode):
            return False

        self.mark_scanned(barcode)
        return True

    def _cooldown_elapsed(self, barcode: str) -> bool:
        last_added = self._last_added.get(barcode, 0.0)
        return time.monotonic() - last_added >= COOLDOWN_SECONDS

    def mark_scanned(self, barcode: str) -> None:
        """Reset the streak and (re)start the cooldown for `barcode`.

        Called both after an automatic scan-to-cart event and after a
        manual confirmation (e.g. clicking a suggested-match button), so
        the two paths can't double-add the same physical item moments apart.
        """
        self._streaks[barcode] = 0
        self._last_added[barcode] = time.monotonic()

    def decay_missing(self, seen_barcodes: set[str]) -> None:
        """Reset streaks for barcodes not observed in the current frame."""
        for barcode in list(self._streaks):
            if barcode not in seen_barcodes:
                self._streaks[barcode] = 0

    def add_or_increment(self, barcode: str, name: str, price: float, thumbnail: Any = None) -> CartItem:
        """Add a new cart line, or increment quantity if the barcode is already in the cart."""
        if barcode in self.items:
            self.items[barcode].quantity += 1
        else:
            self.items[barcode] = CartItem(barcode=barcode, name=name, price=price, quantity=1, thumbnail=thumbnail)
        return self.items[barcode]

    def total(self) -> float:
        return sum(item.price * item.quantity for item in self.items.values())

    def clear(self) -> None:
        self.items.clear()
        self._streaks.clear()
        self._last_added.clear()
