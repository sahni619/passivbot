"""Binance exchange adapter leveraging CCXT client."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from risk_management.models import ExchangeId

from .base import _CcxtAdapter


class BinanceExchangeAdapter(_CcxtAdapter):
    """One-way mode adapter for Binance futures accounts."""

    def __init__(
        self,
        name: str,
        client: Any,
        *,
        settle_currency: str = "USDT",
        params: Optional[Mapping[str, Mapping[str, object]]] = None,
    ) -> None:
        super().__init__(
            name,
            client,
            ExchangeId.BINANCE,
            settle_currency,
            params=params,
        )
