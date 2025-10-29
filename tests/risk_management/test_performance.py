import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from risk_management.performance import PerformanceTracker


def test_performance_tracker_records_daily_snapshots(tmp_path: Path) -> None:
    tracker = PerformanceTracker(tmp_path)

    first_day = datetime(2024, 3, 1, 21, 0, tzinfo=timezone.utc)
    summary = tracker.record(
        generated_at=first_day,
        portfolio_balance=10_000.0,
        account_balances={"Demo": 10_000.0},
    )
    assert summary["portfolio"]["latest_snapshot"]["balance"] == 10_000.0
    assert summary["accounts"]["Demo"]["latest_snapshot"]["balance"] == 10_000.0

    # A call before the next recording window should not add a new snapshot
    midday = datetime(2024, 3, 2, 15, 0, tzinfo=timezone.utc)
    summary_mid = tracker.record(
        generated_at=midday,
        portfolio_balance=10_500.0,
        account_balances={"Demo": 10_500.0},
    )
    assert summary_mid["portfolio"]["daily"] is None

    second_day = datetime(2024, 3, 2, 21, 5, tzinfo=timezone.utc)
    summary_second = tracker.record(
        generated_at=second_day,
        portfolio_balance=11_200.0,
        account_balances={"Demo": 11_200.0},
    )
    daily_change = summary_second["accounts"]["Demo"]["daily"]
    assert daily_change is not None
    assert pytest.approx(daily_change["pnl"]) == 1_200.0
    assert daily_change["since"] == "2024-03-01"
    references = summary_second["accounts"]["Demo"]["reference_balances"]
    periods_since = summary_second["accounts"]["Demo"]["since"]
    assert references["daily"] == pytest.approx(10_000.0)
    assert periods_since["daily"] == "2024-03-01"


def test_performance_tracker_handles_missing_history(tmp_path: Path) -> None:
    tracker = PerformanceTracker(tmp_path)
    summary = tracker.record(
        generated_at=datetime(2024, 3, 3, 21, 0, tzinfo=timezone.utc),
        portfolio_balance=5_000.0,
        account_balances={"Demo": 5_000.0},
    )
    assert summary["portfolio"]["daily"] is None
    assert summary["accounts"]["Demo"]["weekly"] is None


def test_performance_tracker_exposes_history(tmp_path: Path) -> None:
    tracker = PerformanceTracker(tmp_path)

    tracker.record(
        generated_at=datetime(2024, 3, 1, 21, 0, tzinfo=timezone.utc),
        portfolio_balance=10_000.0,
        account_balances={"Alpha": 6_000.0, "Beta": 4_000.0},
    )
    tracker.record(
        generated_at=datetime(2024, 3, 2, 21, 0, tzinfo=timezone.utc),
        portfolio_balance=10_800.0,
        account_balances={"Alpha": 6_500.0, "Beta": 4_300.0},
    )

    portfolio_history = tracker.get_history()
    assert portfolio_history["scope"] == "portfolio"
    assert portfolio_history["count"] == 2
    assert portfolio_history["history"][0]["date"] == "2024-03-01"
    assert portfolio_history["history"][-1]["balance"] == pytest.approx(10_800.0)
    assert portfolio_history["range"] == {"from": "2024-03-01", "to": "2024-03-02"}

    alpha_history = tracker.get_history("Alpha")
    assert alpha_history["account"] == "Alpha"
    assert alpha_history["count"] == 2
    assert alpha_history["history"][0]["balance"] == pytest.approx(6_000.0)

    missing_history = tracker.get_history("Gamma")
    assert missing_history["account"] == "Gamma"
    assert missing_history["history"] == []
    assert missing_history["count"] == 0
