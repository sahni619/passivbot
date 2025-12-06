import asyncio
from pathlib import Path

import pytest

from risk_management.reporting import ReportManager


def sample_snapshot() -> dict:
    return {
        "generated_at": "2024-01-01T00:00:00+00:00",
        "portfolio": {"balance": 2000.0},
        "alerts": ["breach"],
        "accounts": [
            {
                "name": "Main",
                "balance": 1500.0,
                "gross_exposure_notional": 600.0,
                "gross_exposure": 0.4,
                "net_exposure_notional": 200.0,
                "net_exposure": 0.1333,
                "unrealized_pnl": 50.0,
                "symbol_exposures": [
                    {
                        "symbol": "BTCUSDT",
                        "gross_notional": 400.0,
                        "gross_pct": 0.2666,
                        "net_notional": 100.0,
                        "net_pct": 0.0666,
                    }
                ],
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "side": "long",
                        "notional": 400.0,
                        "exposure": 0.2,
                        "unrealized_pnl": 25.0,
                        "pnl_pct": 0.05,
                        "entry_price": 20000,
                        "mark_price": 21000,
                        "liquidation_price": 15000,
                        "take_profit_price": 22000,
                        "stop_loss_price": 18000,
                    }
                ],
                "orders": [
                    {
                        "order_id": "1",
                        "symbol": "BTCUSDT",
                        "side": "buy",
                        "type": "limit",
                        "price": 21000,
                        "amount": 0.01,
                        "remaining": 0.01,
                        "status": "open",
                        "reduce_only": False,
                        "notional": 210.0,
                        "stop_price": None,
                        "created_at": "2024-01-01T00:00:00Z",
                    }
                ],
            }
        ],
    }


def test_report_creation_and_listing(tmp_path: Path):
    manager = ReportManager(tmp_path)
    report = asyncio.run(manager.create_account_report("Main", sample_snapshot()))

    assert report.path.exists()
    assert report.account == "Main"
    assert report.report_id
    listed = asyncio.run(manager.list_reports("Main"))
    assert len(listed) == 1
    assert listed[0].report_id == report.report_id

    path = asyncio.run(manager.get_report_path("Main", report.report_id))
    assert path == report.path


def test_report_sync_helpers_cover_edge_cases(tmp_path: Path):
    manager = ReportManager(tmp_path)

    with pytest.raises(ValueError):
        manager._create_account_report_sync("Unknown", sample_snapshot())

    empty_snapshot = {"accounts": "not-a-sequence"}
    with pytest.raises(ValueError):
        manager._create_account_report_sync("Unknown", empty_snapshot)

    valid_name = manager._account_directory("Spaced Name")
    assert valid_name.name == "Spaced_Name"

    old_file = valid_name / "20240101T010101000000Z.csv"
    new_file = valid_name / "20240101T020202Z.csv"
    valid_name.mkdir(parents=True, exist_ok=True)
    old_file.write_text("old")
    new_file.write_text("new")

    reports = manager._list_reports_sync("Spaced Name")
    assert [r.report_id for r in reports] == ["20240101T020202Z", "20240101T010101000000Z"]
    assert manager._get_report_path_sync("Spaced Name", "missing") is None
    assert manager._get_report_path_sync("Spaced Name", "20240101T010101000000Z") == old_file

    assert manager._format_currency("oops") == "-"
    assert manager._format_pct("oops") == "0.00%"
    assert manager._format_price("oops") == "-"


def test_report_rows_cover_empty_states(tmp_path: Path):
    manager = ReportManager(tmp_path)
    snapshot = sample_snapshot()
    snapshot["accounts"][0].update({"symbol_exposures": None, "positions": None, "orders": None})
    report = manager._create_account_report_sync("Main", snapshot)
    rows = report.path.read_text().splitlines()
    assert any("No symbol exposure" in row for row in rows)
    assert any("No open positions" in row for row in rows)
    assert any("No open orders" in row for row in rows)


