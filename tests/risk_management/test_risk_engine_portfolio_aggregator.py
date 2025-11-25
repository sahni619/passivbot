import pytest

from risk_management.risk_engine.portfolio_aggregator import PortfolioAggregator, PortfolioView


def test_aggregate_multiple_exchanges():
    aggregator = PortfolioAggregator()
    accounts = [
        {
            "name": "binance",
            "balance": 1000,
            "positions": [
                {"symbol": "BTC/USDT", "signed_notional": 500, "unrealized_pnl": 50},
                {"symbol": "ETH/USDT", "signed_notional": -200, "unrealized_pnl": -10},
            ],
            "realized_pnl": 20,
        },
        {
            "name": "bybit",
            "balance": 800,
            "positions": [
                {"symbol": "BTC/USDT", "signed_notional": 300, "unrealized_pnl": 30},
            ],
            "realized_pnl": -5,
        },
    ]
    cashflows = [
        {"type": "deposit", "amount": 100},
        {"type": "withdrawal", "amount": 50},
    ]

    view = aggregator.aggregate(accounts, cashflows)

    assert isinstance(view, PortfolioView)
    assert view.total_balance == pytest.approx(1800)
    assert view.total_unrealized_pnl == pytest.approx(70)
    assert view.total_realized_pnl == pytest.approx(15)
    assert view.total_equity == pytest.approx(1885)
    assert view.net_cashflow == pytest.approx(50)
    assert len(view.accounts) == 2
    assert view.accounts[0].equity == pytest.approx(1060)
    assert view.accounts[1].equity == pytest.approx(825)


def test_aggregate_handles_missing_or_down_accounts():
    aggregator = PortfolioAggregator()
    accounts = [
        None,
        {
            "name": "okx",
            "balance": 500,
            "positions": [],
        },
    ]

    view = aggregator.aggregate(accounts, cashflow_events=[])

    assert view.total_balance == pytest.approx(500)
    assert view.total_unrealized_pnl == pytest.approx(0)
    assert view.total_equity == pytest.approx(500)
    assert view.net_cashflow == pytest.approx(0)
    assert len(view.accounts) == 1


def test_aggregate_negative_unrealized_on_single_exchange():
    aggregator = PortfolioAggregator()
    accounts = [
        {
            "name": "binance",
            "balance": 1000,
            "positions": [
                {"symbol": "SOL/USDT", "signed_notional": 400, "unrealized_pnl": -100},
            ],
        },
        {
            "name": "bybit",
            "balance": 1200,
            "positions": [],
        },
    ]

    view = aggregator.aggregate(accounts, cashflow_events=[])

    assert view.total_balance == pytest.approx(2200)
    assert view.total_unrealized_pnl == pytest.approx(-100)
    assert view.total_equity == pytest.approx(2100)
    assert any(pos.symbol == "SOL/USDT" for pos in view.accounts[0].positions)
