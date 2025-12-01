import json
from pathlib import Path

import pytest

from services.exchanges import (
    BinanceExchangeAdapter,
    BybitExchangeAdapter,
    OKXExchangeAdapter,
)
from services.exchanges.base import HedgedPositionViolation
from risk_management.models import ExchangeId


class DummyCcxtClient:
    def __init__(self, balance_payload, positions_payload):
        self._balance_payload = balance_payload
        self._positions_payload = positions_payload
        self.markets = None
        self.orders = []
        self.leverage_updates = []
        self.balance_params = None
        self.positions_params = None

    async def load_markets(self):
        self.markets = True

    async def fetch_balance(self, params=None):
        self.balance_params = params or {}
        return self._balance_payload

    async def fetch_positions(self, params=None):
        self.positions_params = params or {}
        return self._positions_payload

    async def create_order(self, symbol, order_type, side, amount, price=None, params=None):
        self.orders.append(
            {
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "amount": amount,
                "price": price,
                "params": params or {},
            }
        )
        return {"id": "order-1"}

    async def set_leverage(self, leverage, symbol=None, params=None):
        self.leverage_updates.append({"symbol": symbol, "leverage": leverage, "params": params or {}})
        return {"symbol": symbol, "leverage": leverage}


@pytest.fixture
def fixture_path() -> Path:
    return Path(__file__).parent / "fixtures"


def _load_fixture(path: Path, name: str) -> dict:
    payload = path.joinpath(f"{name}.json").read_text(encoding="utf-8")
    return json.loads(payload)


@pytest.mark.asyncio
async def test_binance_adapter_fetch_and_close(fixture_path: Path):
    payload = _load_fixture(fixture_path, "binance_account")
    client = DummyCcxtClient(payload["balance"], payload["positions"])
    adapter = BinanceExchangeAdapter("Binance", client)

    state = await adapter.fetch_account_state()

    assert state.exchange == ExchangeId.BINANCE
    assert state.balance.total == pytest.approx(1250.5)
    assert len(state.positions) == 1
    position = state.positions[0]
    assert position.symbol == "BTCUSDT"
    assert position.side == "long"
    assert position.position_side == "BOTH"

    closed = await adapter.close_positions()
    assert closed == ["BTCUSDT"]
    assert client.orders[0]["params"]["reduceOnly"] is True

    await adapter.set_leverage("BTCUSDT", 7)
    assert client.leverage_updates == [
        {"symbol": "BTCUSDT", "leverage": 7, "params": {}},
    ]


@pytest.mark.asyncio
async def test_bybit_adapter_closes_shorts_with_params(fixture_path: Path):
    payload = _load_fixture(fixture_path, "bybit_account")
    client = DummyCcxtClient(payload["balance"], payload["positions"])
    adapter = BybitExchangeAdapter("Bybit", client)

    state = await adapter.fetch_account_state()
    assert state.exchange == ExchangeId.BYBIT
    assert state.balance.total == pytest.approx(2200.0)
    assert state.positions[0].side == "short"

    closed = await adapter.close_positions()
    assert closed == ["ETH/USDT:USDT"]
    params = client.orders[0]["params"]
    assert params["positionSide"] == "BOTH"
    assert params["reduceOnly"] is True

    await adapter.set_leverage("ETH/USDT:USDT", 5)
    assert client.leverage_updates[0]["params"]["buyLeverage"] == 5
    assert client.positions_params.get("type") == "swap"


@pytest.mark.asyncio
async def test_okx_adapter_fetches_positions_with_inst_type(fixture_path: Path):
    payload = _load_fixture(fixture_path, "okx_account")
    client = DummyCcxtClient(payload["balance"], payload["positions"])
    adapter = OKXExchangeAdapter("OKX", client)

    state = await adapter.fetch_account_state()
    assert state.exchange == ExchangeId.OKX
    assert state.balance.total == pytest.approx(980.25)
    assert state.positions[0].symbol == "LTC/USDT:USDT"
    assert state.positions[0].side == "long"

    closed = await adapter.close_positions()
    assert closed == ["LTC/USDT:USDT"]
    assert client.positions_params["instType"] == "SWAP"


@pytest.mark.asyncio
async def test_rejects_hedged_positions(fixture_path: Path, caplog: pytest.LogCaptureFixture):
    payload = _load_fixture(fixture_path, "hedged_position")
    client = DummyCcxtClient(payload["balance"], payload["positions"])
    adapter = BinanceExchangeAdapter("Binance", client)

    caplog.set_level("ERROR")
    with pytest.raises(HedgedPositionViolation):
        await adapter.fetch_account_state()

    assert any("hedged positions" in record.message for record in caplog.records)
