"""Persists completed checkout transactions as simple JSON receipts.

Kept separate from `dataset_utils.py` — orders are point-of-sale records,
not training data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pos_app.config import ORDERS_DIR


def save_order(items: list[dict[str, Any]], total: float) -> Path:
    """Write a timestamped JSON receipt for a completed checkout and return its path."""
    timestamp = datetime.now(timezone.utc)
    order_id = timestamp.strftime("%Y%m%dT%H%M%S%f")
    receipt = {
        "order_id": order_id,
        "timestamp": timestamp.isoformat(),
        "items": items,
        "total": round(total, 2),
    }
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    path = ORDERS_DIR / f"order_{order_id}.json"
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
