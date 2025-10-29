from __future__ import annotations

from datetime import datetime, timezone

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_management.dashboard import render_dashboard
from risk_management.domain.models import (
    Account,
    Position,
    Scenario,
    ScenarioAccountImpact,
    ScenarioResult,
    ScenarioShock,
    ScenarioSymbolExposure,
)


def _sample_account() -> Account:
    position = Position(
        symbol="BTCUSDT",
        side="long",
        notional=100.0,
        entry_price=100.0,
        mark_price=100.0,
        liquidation_price=None,
        wallet_exposure_pct=None,
        unrealized_pnl=0.0,
        max_drawdown_pct=None,
    )
    return Account(name="Test", balance=1000.0, positions=[position])


def test_render_dashboard_includes_scenario_results() -> None:
    account = _sample_account()
    scenario = Scenario(
        id="stress_1",
        name="Stress Test",
        description="BTC tumbles",
        shocks=(ScenarioShock(symbol="BTCUSDT", price_pct=-0.1),),
    )
    symbol_exposure = ScenarioSymbolExposure(
        symbol="BTCUSDT",
        gross_notional=450.0,
        net_notional=450.0,
        gross_pct=0.45,
        net_pct=0.45,
        pnl=-50.0,
    )
    account_impact = ScenarioAccountImpact(
        name="Portfolio",
        balance_before=1000.0,
        balance_after=950.0,
        pnl=-50.0,
        gross_exposure=450.0,
        gross_exposure_pct=0.45,
        net_exposure=450.0,
        net_exposure_pct=0.45,
        symbols=[symbol_exposure],
    )
    scenario_result = ScenarioResult(
        scenario=scenario,
        portfolio=account_impact,
        accounts=[account_impact],
    )

    output = render_dashboard(
        datetime.now(timezone.utc),
        [account],
        alerts=[],
        notifications=[],
        scenario_results=[scenario_result],
    )

    assert "Scenario analysis" in output
    assert "Stress Test" in output
    assert "BTCUSDT: -10.00%" in output
    assert "Portfolio PnL: $-50.00" in output
