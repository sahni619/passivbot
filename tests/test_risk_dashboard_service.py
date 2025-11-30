import pytest

pytest.importorskip("fastapi")

from risk_management.web import RiskDashboardService


class _RecordingService:
    def __init__(self) -> None:
        self.calls = []

    async def fetch_snapshot(self):
        self.calls.append(("fetch_snapshot",))
        return {"snapshot": True}

    async def trigger_kill_switch(self, account_name=None, symbol=None):
        self.calls.append(("trigger_kill_switch", account_name, symbol))
        return {"kill": True}

    async def cancel_all_orders(self, account_name, symbol=None):
        self.calls.append(("cancel_all_orders", account_name, symbol))
        return {"cancelled": True}

    async def close_all_positions(self, account_name, symbol=None):
        self.calls.append(("close_all_positions", account_name, symbol))
        return {"closed": True}

    async def place_order(self, account_name, **kwargs):
        self.calls.append(("place_order", account_name, kwargs))
        return {"placed": True}

    async def cancel_order(self, account_name, order_id, *, symbol=None, params=None):
        self.calls.append(("cancel_order", account_name, order_id, symbol, params))
        return {"cancelled_order": True}

    async def close_position(self, account_name, symbol):
        self.calls.append(("close_position", account_name, symbol))
        return {"closed_position": True}

    async def list_order_types(self, account_name):
        self.calls.append(("list_order_types", account_name))
        return ["limit"]

    def get_portfolio_stop_loss(self):
        self.calls.append(("get_portfolio_stop_loss",))
        return {"active": False}

    async def set_portfolio_stop_loss(self, threshold_pct):
        self.calls.append(("set_portfolio_stop_loss", threshold_pct))
        return {"threshold_pct": threshold_pct}

    async def clear_portfolio_stop_loss(self):
        self.calls.append(("clear_portfolio_stop_loss",))

    def get_account_stop_loss(self, account_name):
        self.calls.append(("get_account_stop_loss", account_name))
        return None

    async def set_account_stop_loss(self, account_name, threshold_pct):
        self.calls.append(("set_account_stop_loss", account_name, threshold_pct))
        return {"account": account_name, "threshold_pct": threshold_pct}

    async def clear_account_stop_loss(self, account_name):
        self.calls.append(("clear_account_stop_loss", account_name))

    async def close(self):
        self.calls.append(("close",))


@pytest.mark.asyncio
async def test_dashboard_service_delegates_calls():
    underlying = _RecordingService()
    dashboard = RiskDashboardService(underlying)

    assert await dashboard.fetch_snapshot() == {"snapshot": True}
    assert await dashboard.trigger_kill_switch("acc", "BTCUSDT") == {"kill": True}
    assert await dashboard.cancel_all_orders("acc", "ETHUSDT") == {"cancelled": True}
    assert await dashboard.close_all_positions("acc", None) == {"closed": True}
    assert await dashboard.place_order(
        "acc",
        symbol="BTCUSDT",
        order_type="limit",
        side="buy",
        amount=1.0,
    ) == {"placed": True}
    assert await dashboard.cancel_order("acc", "123", symbol="BTCUSDT", params=None) == {
        "cancelled_order": True
    }
    assert await dashboard.close_position("acc", "BTCUSDT") == {"closed_position": True}
    assert await dashboard.list_order_types("acc") == ["limit"]
    assert dashboard.get_portfolio_stop_loss() == {"active": False}
    assert await dashboard.set_portfolio_stop_loss(5.0) == {"threshold_pct": 5.0}
    await dashboard.clear_portfolio_stop_loss()
    assert dashboard.get_account_stop_loss("acc") is None
    assert await dashboard.set_account_stop_loss("acc", 3.0) == {
        "account": "acc",
        "threshold_pct": 3.0,
    }
    await dashboard.clear_account_stop_loss("acc")
    await dashboard.close()

    expected_calls = [
        ("fetch_snapshot",),
        ("trigger_kill_switch", "acc", "BTCUSDT"),
        ("cancel_all_orders", "acc", "ETHUSDT"),
        ("close_all_positions", "acc", None),
        (
            "place_order",
            "acc",
            {"symbol": "BTCUSDT", "order_type": "limit", "side": "buy", "amount": 1.0},
        ),
        ("cancel_order", "acc", "123", "BTCUSDT", None),
        ("close_position", "acc", "BTCUSDT"),
        ("list_order_types", "acc"),
        ("get_portfolio_stop_loss",),
        ("set_portfolio_stop_loss", 5.0),
        ("clear_portfolio_stop_loss",),
        ("get_account_stop_loss", "acc"),
        ("set_account_stop_loss", "acc", 3.0),
        ("clear_account_stop_loss", "acc"),
        ("close",),
    ]

    assert underlying.calls == expected_calls
