from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from risk_management.performance_metrics import build_performance_metrics


def test_build_performance_metrics_computes_statistics() -> None:
    series = [
        {"date": "2024-01-01", "balance": 1000.0, "timestamp": "2024-01-01T21:00:00Z"},
        {"date": "2024-01-02", "balance": 1100.0, "timestamp": "2024-01-02T21:00:00Z"},
        {"date": "2024-01-03", "balance": 900.0, "timestamp": "2024-01-03T21:00:00Z"},
        {"date": "2024-01-04", "balance": 950.0, "timestamp": "2024-01-04T21:00:00Z"},
    ]

    metrics = build_performance_metrics(series)

    assert metrics["statistics"]["num_points"] == 4
    assert metrics["latest_snapshot"]["date"] == "2024-01-04"
    assert pytest.approx(metrics["statistics"]["total_return"]) == -50.0
    assert pytest.approx(metrics["statistics"]["total_return_pct"], rel=1e-5) == -0.05

    drawdown = metrics["max_drawdown"]
    assert pytest.approx(drawdown["amount"], rel=1e-5) == 200.0
    assert pytest.approx(drawdown["percentage"], rel=1e-5) == 200.0 / 1100.0
    assert drawdown["peak_date"] == "2024-01-02"
    assert drawdown["trough_date"] == "2024-01-03"

    sharpe = metrics["sharpe_ratio"]
    assert sharpe is not None
    assert pytest.approx(sharpe, rel=1e-5) == -1.123320066865463


def test_build_performance_metrics_handles_short_history() -> None:
    series = [{"date": "2024-05-01", "balance": 5000.0, "timestamp": "2024-05-01T21:00:00Z"}]

    metrics = build_performance_metrics(series)

    assert metrics["sharpe_ratio"] is None
    assert metrics["max_drawdown"]["amount"] == 0.0
    assert metrics["statistics"]["total_return_pct"] == 0.0


def test_build_performance_metrics_handles_empty_series() -> None:
    metrics = build_performance_metrics([])

    assert metrics["equity_curve"] == []
    assert metrics["latest_snapshot"] is None
    assert metrics["sharpe_ratio"] is None
    assert metrics["max_drawdown"]["amount"] == 0.0
