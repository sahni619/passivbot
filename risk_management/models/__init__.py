from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool:
    return bool(value)


class ExchangeId(str, Enum):
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    BITGET = "bitget"
    GATEIO = "gateio"
    HYPERLIQUID = "hyperliquid"
    KUCOIN = "kucoin"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object) -> Optional["ExchangeId"]:  # pragma: no cover - defensive
        if isinstance(value, str):
            return cls.OTHER
        return None


@dataclass
class Balance:
    total: float
    equity: Optional[float] = None
    available: Optional[float] = None
    currency: str = "USDT"

    def __init__(
        self,
        balance: Any,
        equity: Any | None = None,
        available: Any | None = None,
        currency: str = "USDT",
    ) -> None:
        self.total = float(balance) if balance not in (None, "") else 0.0
        self.equity = _coerce_float(equity)
        self.available = _coerce_float(available)
        self.currency = str(currency or "USDT")

    def __float__(self) -> float:  # pragma: no cover - convenience
        return self.total


@dataclass
class RiskLimits:
    wallet_exposure_pct: float = 0.6
    position_wallet_exposure_pct: float = 0.25
    max_drawdown_pct: float = 0.3
    loss_threshold_pct: float = -0.12


@dataclass
class PnLBreakdown:
    unrealized: float = 0.0
    realized: float = 0.0
    daily_realized: float = 0.0


class Order:
    def __init__(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str | None = None,
        price: Any = None,
        amount: Any = None,
        remaining: Any = None,
        status: str = "",
        reduce_only: Any = False,
        stop_price: Any = None,
        notional: Any = None,
        order_id: str | None = None,
        created_at: str | None = None,
        type: str | None = None,
        stopPrice: Any = None,
        remaining_amount: Any = None,
        orderId: Any = None,
        createdAt: Any = None,
    ) -> None:
        self.symbol = str(symbol)
        self.side = str(side or "").lower()
        chosen_type = order_type if order_type not in (None, "") else type
        self.order_type = str(chosen_type or "").lower()
        self.price = _coerce_float(price)
        self.amount = _coerce_float(amount)
        remaining_value = remaining if remaining is not None else remaining_amount
        self.remaining = _coerce_float(remaining_value)
        self.status = str(status or "")
        self.reduce_only = _coerce_bool(reduce_only)
        stop_value = stop_price if stop_price is not None else stopPrice
        self.stop_price = _coerce_float(stop_value)
        self.notional = _coerce_float(notional)
        if order_id is None and orderId is not None:
            order_id = orderId
        self.order_id = str(order_id) if order_id not in (None, "") else None
        if created_at is None and createdAt is not None:
            created_at = createdAt
        self.created_at = str(created_at) if created_at not in (None, "") else None


class Position:
    def __init__(
        self,
        *,
        symbol: str,
        side: str,
        notional: Any,
        entry_price: Any,
        mark_price: Any,
        liquidation_price: Any | None = None,
        wallet_exposure_pct: Any | None = None,
        unrealized_pnl: Any = 0.0,
        max_drawdown_pct: Any | None = None,
        take_profit_price: Any | None = None,
        stop_loss_price: Any | None = None,
        size: Any | None = None,
        signed_notional: Any | None = None,
        volatility: Mapping[str, float] | None = None,
        funding_rates: Mapping[str, float] | None = None,
        daily_realized_pnl: Any = 0.0,
        position_side: str | None = None,
        position_idx: Any | None = None,
        positionSide: str | None = None,
        positionIdx: Any | None = None,
    ) -> None:
        self.symbol = str(symbol)
        self.side = str(side or "").lower()
        self.notional = float(notional)
        self.entry_price = float(entry_price)
        self.mark_price = float(mark_price)
        self.liquidation_price = _coerce_float(liquidation_price)
        self.wallet_exposure_pct = _coerce_float(wallet_exposure_pct)
        self.unrealized_pnl = float(unrealized_pnl)
        self.max_drawdown_pct = _coerce_float(max_drawdown_pct)
        self.take_profit_price = _coerce_float(take_profit_price)
        self.stop_loss_price = _coerce_float(stop_loss_price)
        self.size = _coerce_float(size)
        self.signed_notional = _coerce_float(signed_notional)
        self.volatility = volatility
        self.funding_rates = funding_rates
        self.daily_realized_pnl = float(daily_realized_pnl)
        side_value = position_side if position_side is not None else positionSide
        self.position_side = side_value.upper() if side_value else None
        idx_value = position_idx if position_idx is not None else positionIdx
        self.position_idx = int(idx_value) if idx_value not in (None, "") else None

    def exposure_relative_to(self, balance: float) -> float:
        if balance == 0:
            return 0.0
        return abs(self.notional) / balance

    def pnl_pct(self, balance: float) -> float:
        if balance == 0:
            return 0.0
        return self.unrealized_pnl / balance


class AccountState:
    def __init__(
        self,
        *,
        name: str,
        balance: Any,
        positions: Sequence[Position] | Sequence[Mapping[str, Any]],
        orders: Sequence[Order] | Sequence[Mapping[str, Any]] = (),
        pnl: PnLBreakdown | Mapping[str, Any] | None = None,
        risk_limits: RiskLimits | Mapping[str, Any] | None = None,
        exchange: ExchangeId | str | None = None,
    ) -> None:
        self.name = str(name)
        self.balance = self._coerce_balance(balance)
        self.positions = tuple(
            pos if isinstance(pos, Position) else Position(**pos) for pos in positions or []
        )
        self.orders = tuple(order if isinstance(order, Order) else Order(**order) for order in orders or [])
        if isinstance(pnl, PnLBreakdown):
            self.pnl = pnl
        elif isinstance(pnl, Mapping):
            self.pnl = PnLBreakdown(
                unrealized=float(pnl.get("unrealized", 0.0)),
                realized=float(pnl.get("realized", 0.0)),
                daily_realized=float(pnl.get("daily_realized", 0.0)),
            )
        else:
            self.pnl = None
        if isinstance(risk_limits, RiskLimits):
            self.risk_limits = risk_limits
        elif isinstance(risk_limits, Mapping):
            self.risk_limits = RiskLimits(**risk_limits)
        else:
            self.risk_limits = None
        self.exchange = ExchangeId(exchange) if exchange else None

    @staticmethod
    def _coerce_balance(value: Any) -> Balance:
        if isinstance(value, Balance):
            return value
        if isinstance(value, Mapping):
            if "balance" in value and "total" not in value:
                return Balance(**value)
            if "total" in value:
                return Balance(
                    balance=value.get("total", value.get("balance")),
                    equity=value.get("equity"),
                    available=value.get("available"),
                    currency=value.get("currency", "USDT"),
                )
        try:
            total = float(value)
        except (TypeError, ValueError):
            total = 0.0
        return Balance(balance=total)

    def total_abs_notional(self) -> float:
        return sum(abs(p.notional) for p in self.positions)

    def total_unrealized(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions)

    def total_daily_realized(self) -> float:
        return sum(p.daily_realized_pnl for p in self.positions)

    def exposure_pct(self) -> float:
        if self.balance.total == 0:
            return 0.0
        return self.total_abs_notional() / self.balance.total

    def net_notional(self) -> float:
        total = 0.0
        for position in self.positions:
            if position.signed_notional is not None:
                total += position.signed_notional
            else:
                total += position.notional if position.side.lower() == "long" else -position.notional
        return total

    def gross_exposure_pct(self) -> float:
        return self.exposure_pct()

    def net_exposure_pct(self) -> float:
        if self.balance.total == 0:
            return 0.0
        return self.net_notional() / self.balance.total

    def exposures_by_symbol(self) -> dict[str, dict[str, float]]:
        exposures: dict[str, dict[str, float]] = {}
        for position in self.positions:
            signed = (
                position.signed_notional
                if position.signed_notional is not None
                else position.notional if position.side.lower() == "long" else -position.notional
            )
            data = exposures.setdefault(position.symbol, {"gross": 0.0, "net": 0.0})
            data["gross"] += abs(signed)
            data["net"] += signed
        return exposures

    @property
    def daily_realized_pnl(self) -> float:
        if self.pnl is not None:
            return self.pnl.daily_realized
        return self.total_daily_realized()


__all__ = [
    "AccountState",
    "Balance",
    "ExchangeId",
    "Order",
    "PnLBreakdown",
    "Position",
    "RiskLimits",
]
