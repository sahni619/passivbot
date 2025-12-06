import datetime

import pytest

pytest.importorskip("pydantic")

from risk_management import snapshot_utils
from risk_management.dashboard import Account, AlertThresholds, Order, Position
from risk_management.models import Balance


def build_sample_account(name: str, *, message: str | None = None) -> tuple[Account, dict[str, str]]:
    positions = [
        Position(
            symbol="BTCUSDT",
            side="long",
            notional=200.0,
            entry_price=20_000,
            mark_price=21_000,
            liquidation_price=10_000,
            wallet_exposure_pct=0.2,
            unrealized_pnl=25.0,
            max_drawdown_pct=0.05,
            take_profit_price=22_000,
            stop_loss_price=18_000,
            size=0.01,
            signed_notional=200.0,
            volatility={"1h": 0.3, "4h": 0.5},
            funding_rates={"1h": 0.001, "4h": 0.002},
            daily_realized_pnl=5.0,
        ),
        Position(
            symbol="ETHUSDT",
            side="short",
            notional=120.0,
            entry_price=2_000,
            mark_price=1_950,
            liquidation_price=2_400,
            wallet_exposure_pct=0.1,
            unrealized_pnl=-15.0,
            max_drawdown_pct=0.02,
            take_profit_price=1_800,
            stop_loss_price=2_200,
            size=0.06,
            signed_notional=-120.0,
            volatility={"1h": 0.2},
            funding_rates={"1h": -0.001},
            daily_realized_pnl=-3.0,
        ),
    ]
    orders = [
        Order(
            symbol="BTCUSDT",
            side="sell",
            type="limit",
            price=21_500,
            amount=0.01,
            remaining=0.01,
            status="open",
            reduce_only=False,
            stop_price=None,
            notional=215.0,
            order_id="abc",
            created_at="2024-01-01T00:00:00Z",
        )
    ]
    account = Account(name=name, balance=Balance(balance=1000.0), positions=positions, orders=orders)
    message_mapping = {name: message} if message else {}
    return account, message_mapping


def test_build_presentable_snapshot_splits_hidden_accounts():
    visible_account, visible_message = build_sample_account("Visible")
    hidden_account, hidden_message = build_sample_account("Hidden", message="Maintenance")

    snapshot = {
        "generated_at": datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc).isoformat(),
        "accounts": [visible_account, hidden_account],
        "alert_thresholds": AlertThresholds(),
        "notification_channels": ["email"],
        "account_messages": {**visible_message, **hidden_message},
        "portfolio_stop_loss": {"enabled": True, "threshold": -0.15},
        "conditional_stop_losses": [{"symbol": "BTCUSDT", "threshold": -0.1}],
        "policy_violations": ["drawdown"],
    }

    payload = snapshot_utils.build_presentable_snapshot(snapshot)

    assert payload["generated_at"].startswith("2024-01-01")
    assert payload["alerts"] == []
    assert payload["accounts"][0]["name"] == "Visible"
    assert payload["hidden_accounts"] == [{"name": "Hidden", "message": "Maintenance"}]
    assert payload["notifications"] == ["email"]
    assert payload["portfolio_stop_loss"] == {"enabled": True, "threshold": -0.15}
    assert payload["conditional_stop_losses"] == [{"symbol": "BTCUSDT", "threshold": -0.1}]
    assert payload["policy_violations"] == ["drawdown"]

    portfolio = payload["portfolio"]
    assert portfolio["balance"] == 2000.0
    assert portfolio["gross_exposure"] == 320.0
    assert portfolio["net_exposure"] == 80.0
    assert portfolio["volatility"]["1h"] == pytest.approx((0.3 * 200 + 0.2 * 120) / 320)
    assert portfolio["funding_rates"]["1h"] == pytest.approx((0.001 * 200 - 0.001 * 120) / 320)


def test_symbol_and_position_views_are_sorted_and_formatted():
    account, messages = build_sample_account("Trader")
    payload = snapshot_utils.build_presentable_snapshot(
        {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "accounts": [account],
            "alert_thresholds": AlertThresholds(loss_threshold_pct=-1.0),
            "account_messages": messages,
        }
    )

    account_view = payload["accounts"][0]
    exposures = account_view["symbol_exposures"]
    assert [entry["symbol"] for entry in exposures] == ["BTCUSDT", "ETHUSDT"]
    assert account_view["positions"][0]["volatility"] == {"1h": 0.3, "4h": 0.5}
    assert account_view["orders"][0]["order_id"] == "abc"
    assert account_view["volatility"]["1h"] == pytest.approx((0.3 * 200 + 0.2 * 120) / 320)
    assert account_view["funding_rates"]["1h"] == pytest.approx((0.001 * 200 - 0.001 * 120) / 320)

    portfolio_symbols = payload["portfolio"]["symbols"]
    assert portfolio_symbols[0]["symbol"] == "BTCUSDT"
    assert portfolio_symbols[1]["symbol"] == "ETHUSDT"
