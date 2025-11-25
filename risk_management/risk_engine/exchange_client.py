"""Adapters that wrap account clients with one-way safe operations."""

from __future__ import annotations

import logging
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from risk_management.account_clients import AccountClientProtocol

from .metrics import MetricRegistry, Timer

logger = logging.getLogger(__name__)

ONE_WAY_ERROR_CODES = {"-4061", "-4164"}


class ExchangeClientAdapter:
    """Thin adapter around :class:`AccountClientProtocol` with Binance one-way guards."""

    def __init__(
        self, client: AccountClientProtocol, *, one_way_mode: bool = True, metrics: Optional[MetricRegistry] = None
    ) -> None:
        self._client = client
        self._one_way_mode = one_way_mode
        self._metrics = metrics or MetricRegistry()

    @property
    def name(self) -> str:
        return self._client.config.name

    @property
    def config(self):  # pragma: no cover - passthrough for legacy callers
        return self._client.config

    async def fetch(self) -> Mapping[str, Any]:
        with Timer(self._metrics, "exchange_api_latency_seconds", labels={"exchange": self.name, "op": "fetch"}):
            try:
                return await self._client.fetch()
            except Exception as exc:  # pragma: no cover - metrics/logging
                self._metrics.inc(
                    "exchange_api_errors_total", labels={"exchange": self.name, "op": "fetch", "code": _error_code(exc)}
                )
                logger.error(
                    "Failed to fetch account snapshot",
                    extra={"exchange": self.name, "error": str(exc), "op": "fetch"},
                )
                raise

    async def kill_switch(self, symbol: Optional[str] = None) -> Mapping[str, Any]:
        with Timer(self._metrics, "exchange_api_latency_seconds", labels={"exchange": self.name, "op": "kill_switch"}):
            return await self._client.kill_switch(symbol)

    async def close_all_positions(self, symbol: Optional[str] = None) -> Mapping[str, Any]:
        with Timer(
            self._metrics,
            "exchange_api_latency_seconds",
            labels={"exchange": self.name, "op": "close_all_positions"},
        ):
            return await self._client.close_all_positions(symbol)

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Mapping[str, Any]:
        with Timer(
            self._metrics,
            "exchange_api_latency_seconds",
            labels={"exchange": self.name, "op": "cancel_all_orders"},
        ):
            return await self._client.cancel_all_orders(symbol)

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Place an order while stripping ``positionSide`` in one-way mode.

        Binance one-way mode rejects orders when ``positionSide`` is provided. We
        defensively drop the parameter and retry when ccxt propagates -4061 or
        -4164 errors to avoid "position side does not match" failures.
        """

        cleaned_params: MutableMapping[str, Any] = {}
        if isinstance(params, Mapping):
            cleaned_params.update(params)
        if self._one_way_mode:
            cleaned_params.pop("positionSide", None)
        labels = {"exchange": self.name, "op": "create_order"}
        with Timer(self._metrics, "exchange_api_latency_seconds", labels=labels):
            try:
                return await self._client.create_order(
                    symbol, order_type, side, amount, price, params=cleaned_params
                )
            except Exception as exc:
                code = _error_code(exc)
                message = str(exc)
                if self._one_way_mode and code in ONE_WAY_ERROR_CODES:
                    logger.warning(
                        "Retrying order without positionSide in one-way mode",
                        extra={"exchange": self.name, "symbol": symbol, "side": side, "error": message},
                    )
                    cleaned_params.pop("positionSide", None)
                    return await self._client.create_order(
                        symbol, order_type, side, amount, price, params=cleaned_params
                    )
                self._metrics.inc("exchange_api_errors_total", labels={**labels, "code": code})
                logger.error(
                    "Exchange order failed",
                    extra={"exchange": self.name, "symbol": symbol, "side": side, "error_code": code, "error": message},
                )
                raise

    async def close_position(self, symbol: str) -> Mapping[str, Any]:
        with Timer(self._metrics, "exchange_api_latency_seconds", labels={"exchange": self.name, "op": "close_position"}):
            return await self._client.close_position(symbol)

    async def list_order_types(self) -> Sequence[str]:
        return await self._client.list_order_types()


def _error_code(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, (int, str)):
        code_str = str(code)
        if code_str:
            return code_str
    message = str(exc)
    for token in ONE_WAY_ERROR_CODES:
        if token in message:
            return token
    return "unknown"
