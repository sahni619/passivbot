"""OKX exchange adapter leveraging CCXT client."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from risk_management.models import ExchangeId

from .base import _CcxtAdapter


class OKXExchangeAdapter(_CcxtAdapter):
    """One-way mode adapter for OKX perpetual instruments."""

    def __init__(
        self,
        name: str,
        client: Any,
        *,
        settle_currency: str = "USDT",
        params: Optional[Mapping[str, Mapping[str, object]]] = None,
    ) -> None:
        merged_params = {"positions": {"instType": "SWAP"}, **(params or {})}
        super().__init__(
            name,
            client,
            ExchangeId.OKX,
            settle_currency,
            params=merged_params,
        )

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        if not hasattr(self.client, "set_leverage"):
            return
        params = dict(self._params.leverage)
        params.setdefault("mgnMode", "cross")
        await self.client.set_leverage(leverage, symbol, params)
