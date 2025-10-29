from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from risk_management.snapshot_utils import DEFAULT_ACCOUNTS_PAGE_SIZE, build_presentable_snapshot


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


def _complex_snapshot() -> dict:
    return {
        "generated_at": "2024-05-01T12:00:00+00:00",
        "accounts": [
            {
                "name": "Alpha",
                "balance": 100_000,
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "side": "long",
                        "notional": 40_000,
                        "entry_price": 40_000,
                        "mark_price": 41_000,
                        "liquidation_price": 30_000,
                        "wallet_exposure_pct": 0.4,
                        "unrealized_pnl": 1_000,
                        "daily_realized_pnl": 200,
                        "max_drawdown_pct": 0.12,
                        "take_profit_price": 44_000,
                        "stop_loss_price": 38_000,
                        "signed_notional": 40_000,
                        "volatility": {"1d": 0.15},
                        "funding_rates": {"8h": 0.004},
                        "liquidity": {"source": "order_book", "warnings": ["depth_unavailable"]},
                        "liquidity_warnings": ["depth_unavailable"],
                    },
                    {
                        "symbol": "ETHUSDT",
                        "side": "short",
                        "notional": 20_000,
                        "entry_price": 2_000,
                        "mark_price": 1_900,
                        "liquidation_price": 2_300,
                        "wallet_exposure_pct": 0.2,
                        "unrealized_pnl": -500,
                        "daily_realized_pnl": -75,
                        "max_drawdown_pct": 0.08,
                        "take_profit_price": 1_700,
                        "stop_loss_price": 2_200,
                        "signed_notional": -20_000,
                        "volatility": {"1d": 0.2},
                        "funding_rates": {"8h": -0.001},
                    },
                ],
                "open_orders": [
                    {
                        "symbol": "BTCUSDT",
                        "side": "buy",
                        "type": "limit",
                        "price": 39_500,
                        "amount": 0.5,
                        "remaining": 0.25,
                        "status": "open",
                        "reduce_only": False,
                        "stop_price": 39_000,
                        "notional": 19_750,
                        "order_id": "1",
                        "created_at": "2024-05-01T10:00:00Z",
                    }
                ],
                "daily_realized_pnl": 125,
                "metadata": {
                    "scores": {"counterparty_rating": "AA"},
                    "counterparty_rating": "A",
                    "concentration": {"venue_concentration_pct": 0.5},
                    "exposure_limits": {"net": 0.4},
                    "limit_breaches": {
                        "venue_concentration_pct": {"breached": True, "value": 0.55, "limit": 0.4}
                    },
                },
            },
            {
                "name": "Bravo",
                "balance": 50_000,
                "positions": [
                    {
                        "symbol": "ETHUSDT",
                        "side": "short",
                        "notional": 25_000,
                        "entry_price": 2_100,
                        "mark_price": 2_050,
                        "liquidation_price": 2_400,
                        "wallet_exposure_pct": 0.5,
                        "unrealized_pnl": -150,
                        "daily_realized_pnl": 0,
                        "max_drawdown_pct": 0.05,
                        "signed_notional": -25_000,
                    }
                ],
                "limit_breaches": {
                    "asset_concentration_pct": {"breached": True, "value": 0.65, "limit": 0.5}
                },
            },
            {
                "name": "Charlie",
                "balance": 10_000,
                "positions": [],
            },
        ],
        "alert_thresholds": {
            "wallet_exposure_pct": 0.6,
            "position_wallet_exposure_pct": 0.3,
            "max_drawdown_pct": 0.2,
            "loss_threshold_pct": -0.1,
        },
        "notification_channels": ["email:alerts@example.com"],
        "account_messages": {"Charlie": "Account paused"},
        "account_stop_losses": {
            "Alpha": {
                "threshold_pct": "5",
                "baseline_balance": "100000",
                "current_balance": "95000",
                "current_drawdown_pct": "0.05",
                "triggered": False,
                "active": True,
                "triggered_at": "2024-05-01T09:00:00Z",
            },
            "Bravo": ["invalid"],
        },
        "portfolio_stop_loss": {
            "threshold_pct": "4.5",
            "baseline_balance": "150000",
            "current_balance": "143000",
            "current_drawdown_pct": "0.046",
            "triggered": True,
            "active": False,
            "triggered_at": "2024-05-01T08:00:00Z",
        },
        "performance": {
            "portfolio": {
                "current_balance": "150000",
                "latest_snapshot": {
                    "date": "2024-05-01",
                    "balance": "148000",
                    "timestamp": "2024-05-01T12:00:00Z",
                },
                "daily": {
                    "pnl": "500",
                    "pct_change": "0.5",
                    "since": "2024-05-01",
                    "reference_balance": "147000",
                },
                "weekly": 1_200,
            },
            "accounts": {
                "Alpha": {
                    "current_balance": "100000",
                    "latest_snapshot": {
                        "date": "2024-05-01",
                        "balance": "98000",
                        "timestamp": "2024-05-01T12:00:00Z",
                    },
                    "monthly": {
                        "pnl": "1000",
                        "pct_change": "1.5",
                        "since": "2024-04-01",
                        "reference_balance": "90000",
                    },
                }
            },
        },
        "concentration": {
            "venues": {"Alpha": 0.5, "Bravo": "0.3"},
            "assets": {"BTCUSDT": 0.6, "ETHUSDT": "0.25"},
        },
    }


def test_snapshot_sorting_and_pagination_defaults() -> None:
    snapshot = _complex_snapshot()
    view = build_presentable_snapshot(
        snapshot,
        sort_key="unknown",
        sort_order="ascending",
        page=0,
        page_size=0,
    )

    names = [account["name"] for account in view["accounts"]]
    assert names[0] == "Alpha"

    meta = view["accounts_meta"]
    assert meta["sort_key"] == "balance"
    assert meta["sort_order"] == "desc"
    assert meta["page_size"] == DEFAULT_ACCOUNTS_PAGE_SIZE
    assert meta["page"] == 1
    assert meta["pages"] == 1
    assert meta["filtered"] == 2
    assert meta["has_previous"] is False
    assert meta["has_next"] is False


def test_snapshot_filtering_and_search() -> None:
    snapshot = _complex_snapshot()
    view = build_presentable_snapshot(
        snapshot,
        exposure_filter="net_short",
        search="eth",
        sort_key="name",
        sort_order="asc",
    )

    names = [account["name"] for account in view["accounts"]]
    assert names == ["Bravo"]

    meta = view["accounts_meta"]
    assert meta["filtered"] == 1
    assert meta["exposure_filter"] == "net_short"
    assert meta["search"] == "eth"


def test_stop_loss_and_portfolio_normalisation() -> None:
    snapshot = _complex_snapshot()
    view = build_presentable_snapshot(snapshot)

    portfolio_stop = view["portfolio_stop_loss"]
    assert portfolio_stop["threshold_pct"] == "4.5"
    assert portfolio_stop["current_balance"] == "143000"
    assert portfolio_stop["triggered"] is True
    assert portfolio_stop["active"] is False

    account_stops = view["account_stop_losses"]
    assert set(account_stops) == {"Alpha"}
    alpha_stop = account_stops["Alpha"]
    assert alpha_stop["threshold_pct"] == pytest.approx(5.0)
    assert alpha_stop["current_drawdown_pct"] == pytest.approx(0.05)

    hidden = view.get("hidden_accounts", [])
    assert hidden == [{"name": "Charlie", "message": "Account paused"}]

    thresholds = view["thresholds"]
    assert thresholds["wallet_exposure_pct"] == pytest.approx(0.6)

    concentration = view["concentration"]
    assert concentration["venues"]["Alpha"] == pytest.approx(0.5)
    assert concentration["assets"]["BTCUSDT"] == pytest.approx(0.6)
