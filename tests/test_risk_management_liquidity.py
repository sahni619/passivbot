import math
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from risk_management.liquidity import (  # noqa: E402
    calculate_position_liquidity,
    normalise_order_book,
)


def test_normalise_order_book_trims_levels():
    raw = {
        "bids": [["1000", "1"], ["bad", "data"], [995, 2]],
        "asks": [[1010, 3]],
        "timestamp": 111,
    }
    snapshot = normalise_order_book(raw, depth=2)
    assert snapshot is not None
    assert snapshot["bids"] == [[1000.0, 1.0], [995.0, 2.0]]
    assert snapshot["best_bid"] == 1000.0
    assert snapshot["best_ask"] == 1010.0
    assert snapshot["depth"] == {"bids": 2, "asks": 1}


def test_calculate_liquidity_warns_for_thin_books():
    position = {
        "symbol": "BTC/USDT",
        "side": "LONG",
        "size": 5,
        "notional": 5000,
        "mark_price": 1000,
    }
    order_book = {
        "bids": [[920, 2], [900, 1]],
        "asks": [[1080, 5]],
    }
    metrics = calculate_position_liquidity(position, order_book, warning_threshold=0.01)
    assert math.isclose(metrics["filled_size"], 3.0)
    assert math.isclose(metrics["coverage_pct"], 0.6)
    assert "insufficient_depth" in metrics["warnings"]
    assert "slippage_threshold_exceeded" in metrics["warnings"]
    assert metrics["slippage_pct"] < 0
    assert metrics["average_price"] == pytest.approx((920 * 2 + 900) / 3)
