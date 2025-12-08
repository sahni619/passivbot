from datetime import datetime, timezone

from services.persistence import PortfolioHistoryStore


def _snapshot(now: datetime) -> dict:
    return {
        "generated_at": now.isoformat(),
        "portfolio": {
            "balance": 1000,
            "gross_exposure": 500,
            "net_exposure": 200,
            "daily_realized_pnl": 10,
        },
        "accounts": [
            {"unrealized_pnl": 15},
            {"unrealized_pnl": -5},
        ],
    }


def test_record_and_fetch_range(tmp_path):
    store = PortfolioHistoryStore(tmp_path, min_interval_seconds=1)
    now = datetime.now(timezone.utc)
    store.record(_snapshot(now))

    result = store.fetch_range("1h")
    assert result["range"] == "1h"
    assert len(result["series"]) == 1
    entry = result["series"][0]
    assert entry["nav"] == 1000
    assert entry["equity"] == 1010
    assert entry["gross_exposure"] == 500
    assert entry["net_exposure"] == 200
    summary = result["summary"]
    assert summary["nav_start"] == 1000
    assert summary["equity_end"] == 1010
    assert summary["cashflow_net"] == 0


def test_cashflow_round_trip(tmp_path):
    store = PortfolioHistoryStore(tmp_path, min_interval_seconds=1)
    ts = datetime.now(timezone.utc)

    deposit = store.add_cashflow(flow_type="deposit", amount=50, currency="usdt", timestamp=ts)
    withdrawal = store.add_cashflow(
        flow_type="withdrawal", amount=20, currency="usdt", timestamp=ts
    )

    entries = store.list_cashflows(limit=10)
    assert len(entries) == 2
    ids = {entry["id"] for entry in entries}
    assert ids == {deposit["id"], withdrawal["id"]}
    net = sum(entry["signed_amount"] for entry in entries)
    assert net == 30

    summary = store.fetch_range("1h")["summary"]
    assert summary["cashflow_net"] == 30
