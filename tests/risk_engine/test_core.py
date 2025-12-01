import pytest

from risk_management.risk_engine.core import (
    CashFlowEvent,
    ExchangeDrawdown,
    LimitBreach,
    Position,
    evaluate_exchange_drawdowns,
    evaluate_position_limits,
    separate_pnl_from_cash_flow,
    total_unrealized_pnl,
    validate_one_way_mode,
)


def test_total_unrealized_pnl_with_zero_positions():
    assert total_unrealized_pnl([]) == 0


def test_total_unrealized_pnl_mixed_exchanges():
    positions = [
        Position(exchange="binance", symbol="BTCUSDT", quantity=0.5, entry_price=20000, mark_price=21000),
        Position(exchange="bybit", symbol="ETHUSDT", quantity=-2, entry_price=1500, mark_price=1400),
    ]
    expected = (21000 - 20000) * 0.5 + (1400 - 1500) * -2
    assert total_unrealized_pnl(positions) == expected


def test_evaluate_exchange_drawdowns_handles_missing_peaks():
    current_equity = {"binance": 9000, "bybit": 5000}
    peak_equity = {"binance": 10000}
    limits = {"binance": 0.1, "bybit": 0.2}

    results = evaluate_exchange_drawdowns(current_equity, peak_equity, limits)
    result_map = {res.exchange: res for res in results}

    assert pytest.approx(result_map["binance"].drawdown) == 0.1
    assert result_map["binance"].breached is False
    assert result_map["bybit"].drawdown == 0
    assert result_map["bybit"].breached is False


def test_evaluate_position_limits_detects_notional_and_leverage():
    positions = [
        Position(
            exchange="binance",
            symbol="BTCUSDT",
            quantity=1,
            entry_price=20000,
            mark_price=22000,
            leverage=11,
        ),
        Position(
            exchange="bybit",
            symbol="ETHUSDT",
            quantity=0.1,
            entry_price=1500,
            mark_price=1600,
            leverage=2,
        ),
    ]

    breaches = evaluate_position_limits(
        positions,
        notional_limits={"binance": 15000},
        leverage_limits={"binance": 10, "bybit": 1},
    )

    kinds = {(b.exchange, b.kind) for b in breaches}
    assert ("binance", "notional") in kinds
    assert ("binance", "leverage") in kinds
    assert ("bybit", "leverage") in kinds


def test_validate_one_way_mode_conflict_detection():
    positions = [
        Position("binance", "BTCUSDT", 0.5, 20000, 21000),
        Position("binance", "BTCUSDT", -0.3, 20000, 19000),
        Position("bybit", "ETHUSDT", 2, 1500, 1525),
    ]

    conflicts = validate_one_way_mode(positions)
    assert ("binance", "BTCUSDT") in conflicts
    assert ("bybit", "ETHUSDT") not in conflicts


def test_separate_pnl_from_cash_flow_handles_deposits_and_withdrawals():
    events = [
        CashFlowEvent(timestamp=1, amount=1000, description="deposit"),
        CashFlowEvent(timestamp=2, amount=-200, description="withdrawal"),
    ]

    net_pnl, net_cash_flow = separate_pnl_from_cash_flow(5000, 6200, events)

    assert net_cash_flow == 800
    assert net_pnl == 400


def test_exchange_drawdown_breach_flag():
    results = evaluate_exchange_drawdowns(
        current_equity={"binance": 5000},
        peak_equity={"binance": 10000},
        limits={"binance": 0.49},
    )
    assert results[0].breached is True

