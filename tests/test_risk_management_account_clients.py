import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from risk_management.account_clients import CCXTAccountClient, BaseError  # noqa: E402
from risk_management.configuration import LiquiditySettings  # noqa: E402


class StubExchange:
    def __init__(
        self,
        *,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        last: Optional[float] = None,
        position_info: Optional[dict] = None,
        position_overrides: Optional[dict] = None,

    ) -> None:
        self._bid = bid
        self._ask = ask
        self._last = last
        self._cancel_calls = []
        self._orders = []
        self.markets = True
        self._position_info = dict(position_info or {})
        self._position_overrides = dict(position_overrides or {})

    async def cancel_all_orders(self, symbol=None, params=None):
        self._cancel_calls.append({"symbol": symbol, "params": params})

    async def fetch_positions(self, params=None):
        return [
            {
                **self._position_overrides,
                "symbol": "BTC/USDT",
                "contracts": 1,
                "info": dict(self._position_info),
            }
        ]

    async def fetch_order_book(self, symbol):
        raise BaseError("order book unavailable")

    async def fetch_ticker(self, symbol):
        payload = {"info": {}}
        if self._bid is not None:
            payload["bid"] = self._bid
        if self._ask is not None:
            payload["ask"] = self._ask
        if self._last is not None:
            payload["last"] = self._last
        return payload

    async def create_order(self, symbol, order_type, side, amount, price, params=None):
        self._orders.append(
            {
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "amount": amount,
                "price": price,
                "params": params,
            }
        )


class StubDepthExchange:
    def __init__(self) -> None:
        self.markets = True
        self._order_book_calls = []

    async def fetch_balance(self, params=None):
        return {"total": {"USDT": 10_000}}

    async def fetch_positions(self, params=None):
        return [
            {
                "symbol": "BTC/USDT",
                "contracts": 5,
                "entryPrice": 950,
                "markPrice": 1000,
            }
        ]

    async def fetch_order_book(self, symbol, limit=None, params=None):
        self._order_book_calls.append({"symbol": symbol, "limit": limit, "params": params})
        return {
            "bids": [[920, 2], [900, 1]],
            "asks": [[1080, 5]],
            "timestamp": 1234567890,
        }

    async def fetch_open_orders(self, symbol=None, params=None):
        return []


class SymbolRequiredCancelExchange:
    def __init__(self) -> None:
        self.calls = []

    async def cancel_all_orders(self, symbol=None, params=None):
        if symbol is None:
            raise BaseError("symbol required")
        self.calls.append({"symbol": symbol, "params": params})
        return {"symbol": symbol}

    async def fetch_open_orders(self, symbol=None, params=None):
        return [{"symbol": "BTC/USDT"}, {"symbol": "ETH/USDT"}]


class SymbolRequiredNoFetchExchange:
    def __init__(self) -> None:
        self.calls = []

    async def cancel_all_orders(self, symbol=None, params=None):
        if symbol is None:
            raise BaseError("symbol required")
        self.calls.append({"symbol": symbol, "params": params})



class MarketOrderTypesExchange:
    def __init__(self) -> None:
        self.markets = {
            "BTC/USDT": {"orderTypes": ["LIMIT", "STOP_MARKET"]},
            "ETH/USDT": {"info": {"orderTypes": ["MARKET"]}},
        }


class HasOrderTypesExchange:
    def __init__(self) -> None:
        self.has = {
            "createLimitOrder": True,
            "createStopMarketOrder": "emulated",
            "createOrder": True,
        }
        self.markets = True


class FailingOrderTypesExchange:
    def __init__(self) -> None:
        self.markets = None

    async def load_markets(self):
        raise BaseError("load_markets unavailable")



def test_kill_switch_falls_back_to_ticker_price(caplog):
    exchange = StubExchange(bid=101.2)
    client = CCXTAccountClient.__new__(CCXTAccountClient)
    client.config = SimpleNamespace(name="Demo", symbols=None)
    client.client = exchange
    client._balance_params = {}
    client._positions_params = {}
    client._orders_params = {}
    client._close_params = {}
    client._markets_loaded = None
    client._debug_api_payloads = False

    caplog.set_level(logging.INFO, "risk_management")

    summary = asyncio.run(client.kill_switch("BTC/USDT"))

    assert summary["closed_positions"], "Position should be closed when ticker price is available"
    order = exchange._orders[0]
    assert order["price"] == pytest.approx(101.2)
    assert order["params"].get("reduceOnly") is True
    assert any("Executing kill switch" in record.message for record in caplog.records)
    assert any("Kill switch completed" in record.message for record in caplog.records)


def test_kill_switch_logs_failures_when_price_missing(caplog):
    exchange = StubExchange()
    client = CCXTAccountClient.__new__(CCXTAccountClient)
    client.config = SimpleNamespace(name="Demo", symbols=None)
    client.client = exchange
    client._balance_params = {}
    client._positions_params = {}
    client._orders_params = {}
    client._close_params = {}
    client._markets_loaded = None
    client._debug_api_payloads = False

    caplog.set_level(logging.DEBUG, "risk_management")

    summary = asyncio.run(client.kill_switch("BTC/USDT"))

    assert not exchange._orders, "Order should not be placed when price cannot be determined"
    assert summary["failed_position_closures"], "Failure should be recorded when price is missing"
    assert any("Kill switch completed" in record.message for record in caplog.records)
    # Debug details are only emitted when failures occur
    assert any("Kill switch details" in record.message for record in caplog.records)


def test_kill_switch_uses_position_side_from_exchange_payload():
    exchange = StubExchange(bid=99.5, position_info={"positionSide": "LONG"})
    client = CCXTAccountClient.__new__(CCXTAccountClient)
    client.config = SimpleNamespace(name="Demo", symbols=None)
    client.client = exchange
    client._balance_params = {}
    client._positions_params = {}
    client._orders_params = {}
    client._close_params = {}
    client._markets_loaded = None
    client._debug_api_payloads = False

    summary = asyncio.run(client.kill_switch("BTC/USDT"))

    assert summary["closed_positions"], "Kill switch should attempt to close the position"
    order = exchange._orders[0]
    assert order["params"]["positionSide"] == "LONG"
    assert "reduceOnly" not in order["params"]
    assert "reduceonly" not in order["params"]


def test_kill_switch_drops_reduce_only_from_configured_close_params():
    exchange = StubExchange(bid=100.4, position_info={"positionSide": "SHORT"})
    client = CCXTAccountClient.__new__(CCXTAccountClient)
    client.config = SimpleNamespace(name="Demo", symbols=None)
    client.client = exchange
    client._balance_params = {}
    client._positions_params = {}
    client._orders_params = {}
    client._close_params = {"reduceOnly": True, "foo": "bar"}
    client._markets_loaded = None
    client._debug_api_payloads = False

    summary = asyncio.run(client.kill_switch("BTC/USDT"))

    assert summary["closed_positions"], "Kill switch should attempt to close the position"
    order = exchange._orders[0]
    assert order["params"]["positionSide"] == "SHORT"
    assert order["params"].get("foo") == "bar"
    assert "reduceOnly" not in order["params"]
    assert "reduceonly" not in order["params"]


def test_kill_switch_includes_position_idx_from_info():
    exchange = StubExchange(bid=101.0, position_info={"positionIdx": "2"})
    client = CCXTAccountClient.__new__(CCXTAccountClient)
    client.config = SimpleNamespace(name="Demo", symbols=None)
    client.client = exchange
    client._balance_params = {}
    client._positions_params = {}
    client._orders_params = {}
    client._close_params = {}
    client._markets_loaded = None
    client._debug_api_payloads = False

    summary = asyncio.run(client.kill_switch("BTC/USDT"))

    assert summary["closed_positions"], "Kill switch should attempt to close the position"
    order = exchange._orders[0]
    assert order["params"]["positionSide"] == "SHORT"
    assert order["params"]["positionIdx"] == 2
    assert order["params"]["reduceOnly"] is True
    assert "reduceonly" not in order["params"]


def test_kill_switch_uses_position_idx_from_position_payload():
    exchange = StubExchange(
        bid=102.5,
        position_overrides={"positionIdx": 1, "positionSide": None},
        position_info={},
    )
    client = CCXTAccountClient.__new__(CCXTAccountClient)
    client.config = SimpleNamespace(name="Demo", symbols=None)
    client.client = exchange
    client._balance_params = {}
    client._positions_params = {}
    client._orders_params = {}
    client._close_params = {}
    client._markets_loaded = None
    client._debug_api_payloads = False

    summary = asyncio.run(client.kill_switch("BTC/USDT"))

    assert summary["closed_positions"], "Kill switch should attempt to close the position"
    order = exchange._orders[0]
    assert order["params"]["positionSide"] == "LONG"
    assert order["params"]["positionIdx"] == 1
    assert order["params"]["reduceOnly"] is True
    assert "reduceonly" not in order["params"]


def test_fetch_includes_liquidity_metrics_for_thin_books():
    exchange = StubDepthExchange()
    client = CCXTAccountClient.__new__(CCXTAccountClient)
    liquidity_settings = LiquiditySettings(
        fetch_order_book=True,
        depth=2,
        fallback_mode="none",
        slippage_warning_pct=0.01,
    )
    client.config = SimpleNamespace(
        name="Demo",
        symbols=["BTC/USDT"],
        settle_currency="USDT",
        params={"balance": {}, "positions": {}, "orders": {}},
        liquidity=liquidity_settings,
    )
    client.client = exchange
    client._balance_params = {}
    client._positions_params = {}
    client._orders_params = {}
    client._close_params = {}
    client._markets_loaded = None
    client._debug_api_payloads = False
    client._liquidity_settings = liquidity_settings

    summary = asyncio.run(client.fetch())

    assert summary["order_books"], "order book snapshots should be included when available"
    positions = summary["positions"]
    assert positions, "positions should be populated"
    liquidity = positions[0].get("liquidity")
    assert liquidity, "liquidity metrics should be computed for positions"
    warnings = liquidity.get("warnings") if isinstance(liquidity, dict) else None
    assert warnings and "insufficient_depth" in warnings
    assert "slippage_threshold_exceeded" in warnings


def test_cancel_all_orders_falls_back_to_per_symbol_cancellations():
    exchange = SymbolRequiredCancelExchange()
    client = CCXTAccountClient.__new__(CCXTAccountClient)
    client.config = SimpleNamespace(name="Demo", symbols=None)
    client.client = exchange
    client._balance_params = {}
    client._positions_params = {}
    client._orders_params = {}
    client._close_params = {"foo": "bar"}
    client._markets_loaded = None
    client._debug_api_payloads = False

    result = asyncio.run(client.cancel_all_orders())

    assert [call["symbol"] for call in exchange.calls] == ["BTC/USDT", "ETH/USDT"]
    assert all(call["params"] == {"foo": "bar"} for call in exchange.calls)
    cancelled = result.get("cancelled_orders") or []
    assert [entry.get("symbol") for entry in cancelled] == ["BTC/USDT", "ETH/USDT"]
    assert "failed_order_cancellations" not in result or not result["failed_order_cancellations"]


def test_cancel_all_orders_requires_symbol_when_open_orders_unavailable():
    exchange = SymbolRequiredNoFetchExchange()
    client = CCXTAccountClient.__new__(CCXTAccountClient)
    client.config = SimpleNamespace(name="Demo", symbols=None)
    client.client = exchange
    client._balance_params = {}
    client._positions_params = {}
    client._orders_params = {}
    client._close_params = {}
    client._markets_loaded = None
    client._debug_api_payloads = False

    with pytest.raises(RuntimeError):
        asyncio.run(client.cancel_all_orders())



def test_list_order_types_uses_market_metadata():
    exchange = MarketOrderTypesExchange()
    client = CCXTAccountClient.__new__(CCXTAccountClient)
    client.config = SimpleNamespace(name="Demo", symbols=None)
    client.client = exchange
    client._balance_params = {}
    client._positions_params = {}
    client._orders_params = {}
    client._close_params = {}
    client._markets_loaded = None
    client._debug_api_payloads = False

    order_types = asyncio.run(client.list_order_types())

    assert set(order_types) == {"limit", "market", "stop_market"}


def test_list_order_types_from_has_map():
    exchange = HasOrderTypesExchange()
    client = CCXTAccountClient.__new__(CCXTAccountClient)
    client.config = SimpleNamespace(name="Demo", symbols=None)
    client.client = exchange
    client._balance_params = {}
    client._positions_params = {}
    client._orders_params = {}
    client._close_params = {}
    client._markets_loaded = None
    client._debug_api_payloads = False

    order_types = asyncio.run(client.list_order_types())

    assert order_types == ("limit", "stop_market")


def test_list_order_types_falls_back_on_failure(caplog):
    exchange = FailingOrderTypesExchange()
    client = CCXTAccountClient.__new__(CCXTAccountClient)
    client.config = SimpleNamespace(name="Demo", symbols=None)
    client.client = exchange
    client._balance_params = {}
    client._positions_params = {}
    client._orders_params = {}
    client._close_params = {}
    client._markets_loaded = None
    client._debug_api_payloads = False

    caplog.set_level(logging.INFO, "risk_management")

    order_types = asyncio.run(client.list_order_types())

    assert order_types == ("limit", "market")
    assert any("Falling back to default order types" in record.message for record in caplog.records)
