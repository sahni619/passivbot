from risk_engine import RiskPolicy, RiskViolation, evaluate_policies


def _sample_snapshot() -> dict:
    return {
        "accounts": [
            {
                "name": "Alpha",
                "exchange": "binance",
                "balance": 10_000,
                "daily_realized_pnl": -150.0,
                "positions": [
                    {"symbol": "BTCUSDT", "side": "long", "notional": 3_000, "unrealized_pnl": -250.0},
                    {"symbol": "ETHUSDT", "side": "short", "notional": 1_500, "unrealized_pnl": -50.0},
                ],
            },
            {
                "name": "Beta",
                "exchange": "okx",
                "balance": 5_000,
                "positions": [
                    {"symbol": "BTCUSDT", "side": "short", "notional": 4_500, "unrealized_pnl": 120.0},
                ],
            },
        ]
    }


def test_combined_pnl_policy_flags_losses() -> None:
    snapshot = _sample_snapshot()
    policy = RiskPolicy(type="combined_pnl", threshold=-200.0, name="portfolio_pnl")

    violations = evaluate_policies([policy], snapshot)

    assert violations == [
        RiskViolation(
            "portfolio_pnl",
            "Combined PnL -330.00 breached limit -200.00",
            severity="warning",
            data={"pnl": -330.0},
        )
    ]


def test_exchange_drawdown_policy_respects_exchange_scope() -> None:
    snapshot = _sample_snapshot()
    policy = RiskPolicy(type="exchange_drawdown", threshold=0.04, exchange="binance")

    violations = evaluate_policies([policy], snapshot)

    assert len(violations) == 1
    violation = violations[0]
    assert violation.policy == "exchange_drawdown"
    assert violation.subject == "binance"
    assert "drawdown" in violation.message


def test_notional_cap_policy_sums_matching_positions() -> None:
    snapshot = _sample_snapshot()
    policy = RiskPolicy(type="notional_cap", threshold=6_000, symbol="BTCUSDT", severity="critical")

    violations = evaluate_policies([policy], snapshot)

    assert violations == [
        RiskViolation(
            "notional_cap",
            "Notional 7500.00 exceeds cap 6000.00",
            severity="critical",
            subject="BTCUSDT",
            data={"notional": 7500.0},
        )
    ]


def test_one_way_policy_detects_opposite_netting() -> None:
    snapshot = _sample_snapshot()
    policy = RiskPolicy(type="one_way", threshold=0, symbol="BTCUSDT", side="long")

    violations = evaluate_policies([policy], snapshot)

    assert violations
    violation = violations[0]
    assert violation.policy == "one_way"
    assert violation.subject == "BTCUSDT"
    assert "violates one-way" in violation.message
