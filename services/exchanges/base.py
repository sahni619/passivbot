"""Exchange adapter interface enforcing one-way mode account operations."""

from __future__ import annotations
import abc
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

from risk_management.account_clients import _extract_position_details
from risk_management.models import AccountState, Balance, ExchangeId, Position
from risk_management.realtime import _extract_balance, _parse_position

logger = logging.getLogger(__name__)


class HedgedPositionViolation(RuntimeError):
    """Raised when hedged (dual-side) positions are discovered on an account."""

    def __init__(self, exchange: ExchangeId, violations: Sequence[Mapping[str, Any]]):
        self.exchange = exchange
        self.violations = tuple(violations)
        symbols = ", ".join(str(item.get("symbol")) for item in self.violations)
        message = f"{exchange.value} returned hedged positions for symbols: {symbols}"
        super().__init__(message)


class AccountStateAdapter(abc.ABC):
    """Asynchronous interface for exchange account interactions.

    Implementations must operate in one-way (non-hedged) position mode. Any detected
    hedged positions should be rejected with :class:`HedgedPositionViolation`.
    """

    exchange: ExchangeId
    settle_currency: str

    @abc.abstractmethod
    async def fetch_account_state(self) -> AccountState:
        """Return the current account state, including balance and open positions."""

    @abc.abstractmethod
    async def close_positions(self, symbols: Optional[Sequence[str]] = None) -> Sequence[str]:
        """Close open positions for ``symbols`` (or all when ``None``)."""

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for ``symbol`` when supported by the adapter."""
        raise NotImplementedError


@dataclass
class _CcxtParams:
    balance: MutableMapping[str, Any]
    positions: MutableMapping[str, Any]
    close: MutableMapping[str, Any]
    leverage: MutableMapping[str, Any]


class _CcxtAdapter(AccountStateAdapter):
    """Shared CCXT-backed adapter logic for specific exchanges."""

    def __init__(
        self,
        name: str,
        client: Any,
        exchange: ExchangeId,
        settle_currency: str = "USDT",
        *,
        params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        self.name = name
        self.client = client
        self.exchange = exchange
        self.settle_currency = settle_currency
        self._markets_lock: Optional[asyncio.Lock] = None
        base_params = params or {}
        self._params = _CcxtParams(
            balance=dict(base_params.get("balance", {})),
            positions=dict(base_params.get("positions", {})),
            close=dict(base_params.get("close", {})),
            leverage=dict(base_params.get("leverage", {})),
        )

    async def _ensure_markets(self) -> None:
        lock = self._markets_lock
        if lock is None:
            lock = asyncio.Lock()
            self._markets_lock = lock
        async with lock:
            if getattr(self.client, "markets", None):
                return
            if hasattr(self.client, "load_markets"):
                await self.client.load_markets()

    def _validate_positions(self, positions: Iterable[Mapping[str, Any]]) -> None:
        violations: list[Mapping[str, Any]] = []
        for position in positions or []:
            position_side, position_idx = _extract_position_details(position)
            if position_side in {"LONG", "SHORT"} or position_idx in {1, 2}:
                symbol = position.get("symbol") or position.get("id") or "unknown"
                violations.append(
                    {"symbol": symbol, "positionSide": position_side, "positionIdx": position_idx}
                )
        if violations:
            logger.error("Detected hedged positions on %s: %s", self.exchange.value, violations)
            raise HedgedPositionViolation(self.exchange, violations)

    async def _fetch_positions(self) -> Sequence[Mapping[str, Any]]:
        if not hasattr(self.client, "fetch_positions"):
            return []
        positions = await self.client.fetch_positions(params=self._params.positions)
        self._validate_positions(positions or [])
        return list(positions or [])

    def _parse_positions(self, positions_raw: Iterable[Mapping[str, Any]], balance: float) -> list[Position]:
        parsed: list[Position] = []
        for position in positions_raw:
            normalized = _parse_position(position, balance)
            if normalized is None:
                continue
            position_side, position_idx = _extract_position_details(position)
            if position_side:
                normalized["position_side"] = position_side
            if position_idx is not None:
                normalized["position_idx"] = position_idx
            parsed.append(Position(**normalized))
        return parsed

    async def fetch_account_state(self) -> AccountState:
        await self._ensure_markets()
        balance_raw = await self.client.fetch_balance(params=self._params.balance)
        balance_value = _extract_balance(balance_raw, self.settle_currency)
        positions_raw = await self._fetch_positions()
        positions = self._parse_positions(positions_raw, balance_value)
        balance_model = Balance(balance=balance_value, currency=self.settle_currency)
        return AccountState(
            name=self.name,
            balance=balance_model,
            positions=positions,
            exchange=self.exchange,
        )

    async def close_positions(self, symbols: Optional[Sequence[str]] = None) -> Sequence[str]:
        await self._ensure_markets()
        positions = await self._fetch_positions()
        closed: list[str] = []
        for position in positions:
            size_raw = position.get("contracts") or position.get("size") or position.get("amount")
            symbol = position.get("symbol") or position.get("id")
            if not symbol:
                continue
            if symbols and symbol not in symbols:
                continue
            try:
                size = float(size_raw)
            except (TypeError, ValueError):
                continue
            if abs(size) < 1e-12:
                continue
            side = "sell" if size > 0 else "buy"
            amount = abs(size)
            params = dict(self._params.close)
            position_side, position_idx = _extract_position_details(position)
            if position_side:
                params.setdefault("positionSide", position_side)
            if position_idx is not None:
                params.setdefault("positionIdx", position_idx)
            params.setdefault("reduceOnly", True)
            try:
                await self.client.create_order(symbol, "market", side, amount, params=params)
                closed.append(symbol)
            except Exception:
                logger.warning("Failed to close position %s on %s", symbol, self.exchange.value, exc_info=True)
        return closed

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        if not hasattr(self.client, "set_leverage"):
            logger.debug("set_leverage not supported on %s", self.exchange.value)
            return
        params = dict(self._params.leverage)
        await self.client.set_leverage(leverage, symbol, params)
