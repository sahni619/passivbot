import asyncio
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import pytest

from risk_management.risk_engine.exchange_client import ExchangeClientAdapter, ONE_WAY_ERROR_CODES


class DummyClient:
    def __init__(self):
        self.config = SimpleNamespace(name="dummy")
        self.calls = []
        self.side_effects = []

    async def fetch(self) -> Mapping[str, Any]:  # pragma: no cover - unused
        return {}

    async def close(self) -> None:  # pragma: no cover - unused
        return None

    async def kill_switch(self, symbol: Optional[str] = None) -> Mapping[str, Any]:  # pragma: no cover - unused
        return {"symbol": symbol}

    async def close_all_positions(self, symbol: Optional[str] = None) -> Mapping[str, Any]:  # pragma: no cover - unused
        return {"closed": symbol}

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Mapping[str, Any]:  # pragma: no cover - unused
        return {"cancelled": symbol}

    async def list_order_types(self) -> Sequence[str]:  # pragma: no cover - unused
        return ["market"]

    async def close_position(self, symbol: str) -> Mapping[str, Any]:  # pragma: no cover - unused
        return {"closed": symbol}

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        self.calls.append({"symbol": symbol, "params": params})
        if self.side_effects:
            effect = self.side_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
        return {"symbol": symbol, "params": params}


def test_one_way_strips_position_side():
    client = DummyClient()
    adapter = ExchangeClientAdapter(client, one_way_mode=True)

    result = asyncio.run(
        adapter.create_order(
            "BTC/USDT", "market", "buy", 1, params={"positionSide": "LONG", "foo": "bar"}
        )
    )

    assert client.calls[-1]["params"].get("positionSide") is None
    assert result["params"].get("positionSide") is None
    assert client.calls[-1]["params"].get("foo") == "bar"


def test_one_way_retries_on_position_side_errors():
    client = DummyClient()
    client.side_effects.append(Exception("-4061 position side does not match"))
    adapter = ExchangeClientAdapter(client, one_way_mode=True)

    result = asyncio.run(
        adapter.create_order(
            "ETH/USDT", "limit", "sell", 2, price=1000, params={"positionSide": "SHORT"}
        )
    )

    assert len(client.calls) == 2  # retry once
    for call in client.calls:
        assert call["params"].get("positionSide") is None
    assert result["params"].get("positionSide") is None


def test_hedge_mode_preserves_position_side_and_does_not_retry():
    client = DummyClient()
    adapter = ExchangeClientAdapter(client, one_way_mode=False)

    asyncio.run(
        adapter.create_order(
            "ETH/USDT", "limit", "sell", 1, price=900, params={"positionSide": "SHORT"}
        )
    )

    assert len(client.calls) == 1
    assert client.calls[0]["params"].get("positionSide") == "SHORT"
