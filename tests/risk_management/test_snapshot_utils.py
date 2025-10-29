import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from risk_management.presentation.snapshot_builder import build_presentable_snapshot


def test_snapshot_utils_preserves_position_fields() -> None:
    snapshot = {
        "generated_at": "2024-03-02T00:00:00+00:00",
        "accounts": [
            {
                "name": "Demo",
                "balance": 1_000,
                "positions": [
                    {
                        "symbol": "BTC/USDT",
                        "side": "long",
                        "notional": 500,
                        "entry_price": 100,
                        "mark_price": 110,
                        "liquidation_price": 80,
                        "wallet_exposure_pct": 0.5,
                        "unrealized_pnl": 50,
                        "daily_realized_pnl": 10,
                        "max_drawdown_pct": 0.2,
                        "take_profit_price": 120,
                        "stop_loss_price": 90,
                    }
                ],
                "daily_realized_pnl": 10,
                "open_orders": [],
                "metadata": {
                    "counterparty_rating": "Tier 1",
                    "concentration": {
                        "venue_concentration_pct": 0.4,
                        "asset_concentration_pct": 0.6,
                        "top_asset": "BTC/USDT",
                    },
                    "exposure_limits": {"asset_concentration_pct": 0.5},
                    "limit_breaches": {
                        "asset_concentration_pct": {
                            "breached": True,
                            "value": 0.6,
                            "limit": 0.5,
                        }
                    },
                    "scores": {
                        "counterparty_rating": "Tier 1",
                        "asset_concentration_pct": 0.6,
                        "venue_concentration_pct": 0.4,
                    },
                },
            }
        ],
        "alert_thresholds": {},
        "notification_channels": [],
        "account_stop_losses": {
            "Demo": {
                "threshold_pct": 5.0,
                "baseline_balance": 1_000.0,
                "current_balance": 950.0,
                "current_drawdown_pct": 0.05,
                "triggered": False,
                "active": True,
                "triggered_at": None,
            }
        },
        "performance": {
            "portfolio": {
                "current_balance": 1_000.0,
                "latest_snapshot": {
                    "date": "2024-03-01",
                    "balance": 980.0,
                    "timestamp": "2024-03-01T21:00:00+00:00",
                },
                "daily": {
                    "pnl": 20.0,
                    "since": "2024-03-01",
                    "reference_balance": 960.0,
                },
            },
            "accounts": {
                "Demo": {
                    "current_balance": 1_000.0,
                    "latest_snapshot": {
                        "date": "2024-03-01",
                        "balance": 985.0,
                        "timestamp": "2024-03-01T21:00:00+00:00",
                    },
                    "daily": {
                        "pnl": 15.0,
                        "since": "2024-03-01",
                        "reference_balance": 985.0,
                    },
                }
            },
        },
        "concentration": {
            "venues": {"Demo": 0.4},
            "assets": {"BTC/USDT": 0.6, "ETH/USDT": 0.4},
        },
    }

    view = build_presentable_snapshot(snapshot)

    assert view["accounts"][0]["positions"][0]["daily_realized_pnl"] == 10
    assert view["accounts"][0]["positions"][0]["liquidation_price"] == 80
    assert view["accounts"][0]["positions"][0]["take_profit_price"] == 120
    assert view["accounts"][0]["positions"][0]["stop_loss_price"] == 90
    assert view["accounts"][0]["positions"][0]["max_drawdown_pct"] == 0.2

    stop_loss = view["accounts"][0]["stop_loss"]
    assert stop_loss["threshold_pct"] == 5.0
    assert stop_loss["current_balance"] == 950.0

    performance = view["accounts"][0]["performance"]
    assert performance["daily"] == 15.0
    assert performance["since"]["daily"] == "2024-03-01"

    portfolio_perf = view["portfolio"]["performance"]
    assert portfolio_perf["daily"] == 20.0
    assert portfolio_perf["latest_snapshot"]["balance"] == 980.0

    assert view["account_stop_losses"]["Demo"]["current_balance"] == 950.0
    assert view["accounts"][0]["counterparty_rating"] == "Tier 1"
    assert (
        view["accounts"][0]["limit_breaches"]["asset_concentration_pct"]["breached"]
        is True
    )
    assert view["concentration"]["venues"]["Demo"] == pytest.approx(0.4)
    assert view["concentration"]["assets"]["BTC/USDT"] == pytest.approx(0.6)


def test_portfolio_daily_realized_aggregates_account_values() -> None:
    snapshot = {
        "generated_at": "2024-03-02T00:00:00+00:00",
        "accounts": [
            {
                "name": "Primary",
                "balance": 20_000,
                "daily_realized_pnl": 150.75,
                "positions": [
                    {
                        "symbol": "ETH/USDT",
                        "side": "long",
                        "notional": 0,
                        "entry_price": 0,
                        "mark_price": 0,
                        "unrealized_pnl": 0,
                        "daily_realized_pnl": 0,
                    }
                ],
            },
            {
                "name": "Secondary",
                "balance": 5_000,
                "daily_realized_pnl": -25.5,
                "positions": [],
            },
        ],
        "alert_thresholds": {},
        "notification_channels": [],
    }

    view = build_presentable_snapshot(snapshot)

    assert view["portfolio"]["daily_realized_pnl"] == pytest.approx(125.25)
    realised_values = [account["daily_realized_pnl"] for account in view["accounts"]]
    assert realised_values == [pytest.approx(150.75), pytest.approx(-25.5)]


def test_snapshot_utils_includes_policies() -> None:
    snapshot = {
        "generated_at": "2024-03-02T00:00:00+00:00",
        "accounts": [
            {"name": "Alpha", "balance": 50_000, "positions": []},
        ],
        "alert_thresholds": {},
        "notification_channels": [],
        "policies": {
            "evaluations": [
                {
                    "name": "Balance floor",
                    "metric": "account.Alpha.balance",
                    "operator": "<",
                    "threshold": 50_000,
                    "value": 45_000,
                    "triggered": True,
                    "actions": [
                        {"type": "notify", "status": "triggered", "message": "Balance low"},
                    ],
                }
            ],
            "active": [
                {"name": "Balance floor", "value": 45_000, "threshold": 50_000, "operator": "<"}
            ],
            "pending_actions": [
                {"policy": "Balance floor", "message": "Confirm trading continuation"}
            ],
        },
    }

    view = build_presentable_snapshot(snapshot)

    policies = view.get("policies")
    assert isinstance(policies, dict)
    assert policies["evaluations"][0]["name"] == "Balance floor"
    assert policies["pending_actions"][0]["policy"] == "Balance floor"
