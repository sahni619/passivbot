import asyncio
from typing import Any, Dict, List, Mapping, Optional

from risk_management.configuration import AccountConfig, RealtimeConfig
from risk_management.realtime import RealtimeDataFetcher


class RecordingKillSwitchClient:
    def __init__(
        self,
        name: str,
        cancel_payload: Mapping[str, Any],
        close_payload: Mapping[str, Any],
        *,
        cancel_exception: Optional[BaseException] = None,
    ) -> None:
        self.config = AccountConfig(name=name, exchange="test")
        self.cancel_payload = cancel_payload
        self.close_payload = close_payload
        self.cancel_exception = cancel_exception
        self.cancel_all_orders_calls: List[Optional[str]] = []
        self.close_all_positions_calls: List[Optional[str]] = []
        self.kill_switch_calls: List[Optional[str]] = []

    async def fetch(self) -> Mapping[str, Any]:  # pragma: no cover - unused in this suite
        return {"name": self.config.name, "balance": 0.0, "positions": []}

    async def close(self) -> None:  # pragma: no cover - unused in this suite
        return None

    async def list_order_types(self) -> tuple[str, ...]:  # pragma: no cover - unused
        return ("market", "limit")

    async def create_order(  # pragma: no cover - unused
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    async def cancel_order(  # pragma: no cover - unused
        self, order_id: str, symbol: Optional[str] = None, params: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    async def close_position(self, symbol: str) -> Mapping[str, Any]:  # pragma: no cover - unused
        raise NotImplementedError

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Mapping[str, Any]:
        self.cancel_all_orders_calls.append(symbol)
        if self.cancel_exception is not None:
            raise self.cancel_exception
        return dict(self.cancel_payload)

    async def close_all_positions(self, symbol: Optional[str] = None) -> Mapping[str, Any]:
        self.close_all_positions_calls.append(symbol)
        return dict(self.close_payload)

    async def kill_switch(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        self.kill_switch_calls.append(symbol)
        result: Dict[str, Any] = {}
        cancel_result = await self.cancel_all_orders(symbol)
        result["cancelled_orders"] = cancel_result

        close_result = await self.close_all_positions(symbol)
        if isinstance(close_result, Mapping):
            result["closed_positions"] = close_result.get("closed_positions", [])
            if "failed_position_closures" in close_result:
                result["failed_position_closures"] = close_result["failed_position_closures"]
        else:
            result["closed_positions"] = close_result

        if isinstance(cancel_result, Mapping) and "failed_order_cancellations" in cancel_result:
            result["failed_order_cancellations"] = cancel_result["failed_order_cancellations"]
        return result


def test_execute_kill_switch_handles_mixed_client_results() -> None:
    async def run_scenario() -> Mapping[str, Any]:
        success_client = RecordingKillSwitchClient(
            "Alpha",
            cancel_payload={"cancelled_orders": ["o-1"]},
            close_payload={
                "closed_positions": ["BTCUSDT"],
                "failed_position_closures": ["ETHUSDT"],
            },
        )
        failure_client = RecordingKillSwitchClient(
            "Beta",
            cancel_payload={"cancelled_orders": ["o-2"]},
            close_payload={"closed_positions": ["BTCUSDT"]},
            cancel_exception=RuntimeError("cancel failed"),
        )
        config = RealtimeConfig(accounts=[success_client.config, failure_client.config])
        fetcher = RealtimeDataFetcher(config, account_clients=[success_client, failure_client])

        try:
            result = await fetcher.execute_kill_switch(symbol="BTCUSDT")

            assert success_client.cancel_all_orders_calls == ["BTCUSDT"]
            assert success_client.close_all_positions_calls == ["BTCUSDT"]
            assert result["Alpha"]["cancelled_orders"] == {"cancelled_orders": ["o-1"]}
            assert result["Alpha"]["closed_positions"] == ["BTCUSDT"]
            assert result["Alpha"]["failed_position_closures"] == ["ETHUSDT"]

            assert failure_client.cancel_all_orders_calls == ["BTCUSDT"]
            assert failure_client.close_all_positions_calls == []
            assert "error" in result["Beta"]
            assert "cancel failed" in result["Beta"]["error"]
            return result
        finally:
            await asyncio.gather(fetcher.close(), success_client.close(), failure_client.close())

    asyncio.run(run_scenario())
