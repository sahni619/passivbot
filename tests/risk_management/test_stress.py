from __future__ import annotations

import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_management.domain.models import Scenario, ScenarioShock
from risk_management.stress import simulate_scenarios


def _snapshot_with_position(mark_price: float = 100.0) -> dict[str, object]:
    return {
        "generated_at": "2024-01-01T00:00:00Z",
        "accounts": [
            {
                "name": "Test account",
                "balance": 1000.0,
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "side": "long",
                        "notional": 500.0,
                        "entry_price": 100.0,
                        "mark_price": mark_price,
                        "unrealized_pnl": 0.0,
                        "size": 5.0,
                    }
                ],
            }
        ],
        "alert_thresholds": {},
    }


def test_simulate_scenarios_long_position_loss() -> None:
    snapshot = _snapshot_with_position()
    scenario = Scenario(
        id="btc_down",
        name="BTC -10%",
        description="Bitcoin drops by ten percent",
        shocks=(ScenarioShock(symbol="BTCUSDT", price_pct=-0.1),),
    )

    results = simulate_scenarios(snapshot, [scenario])

    assert len(results) == 1
    result = results[0]
    assert result.scenario.name == "BTC -10%"
    portfolio = result.portfolio
    assert portfolio.balance_before == pytest.approx(1000.0)
    assert portfolio.balance_after == pytest.approx(950.0)
    assert portfolio.pnl == pytest.approx(-50.0)
    assert portfolio.gross_exposure == pytest.approx(450.0)
    assert portfolio.net_exposure == pytest.approx(450.0)
    assert portfolio.gross_exposure_pct == pytest.approx(450.0 / 950.0)
    assert portfolio.symbols
    symbol = portfolio.symbols[0]
    assert symbol.symbol == "BTCUSDT"
    assert symbol.gross_notional == pytest.approx(450.0)
    assert symbol.pnl == pytest.approx(-50.0)


def test_simulate_scenarios_no_scenarios_returns_empty() -> None:
    snapshot = _snapshot_with_position()
    assert simulate_scenarios(snapshot, []) == []
