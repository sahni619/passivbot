from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from pydantic import BaseModel, Field, root_validator, validator


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


class _Model(BaseModel):
    class Config:
        allow_population_by_field_name = True
        extra = "ignore"


class Balance(_Model):
    total: float = Field(..., alias="balance")
    equity: Optional[float] = None
    available: Optional[float] = None
    currency: str = "USDT"

    @validator("total", "equity", "available", pre=True)
    def _coerce_numeric(cls, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    def __float__(self) -> float:  # pragma: no cover - convenience
        return self.total


class RiskLimits(_Model):
    wallet_exposure_pct: float = 0.6
    position_wallet_exposure_pct: float = 0.25
    max_drawdown_pct: float = 0.3
    loss_threshold_pct: float = -0.12


class PnLBreakdown(_Model):
    unrealized: float = 0.0
    realized: float = 0.0
    daily_realized: float = 0.0


class Order(_Model):
    symbol: str
    side: str
    order_type: str = Field(..., alias="type")
    price: Optional[float] = None
    amount: Optional[float] = None
    remaining: Optional[float] = None
    status: str = ""
    reduce_only: bool = False
    stop_price: Optional[float] = None
    notional: Optional[float] = None
    order_id: Optional[str] = None
    created_at: Optional[str] = None

    @validator("side", pre=True)
    def _normalize_side(cls, value: Any) -> str:
        return str(value or "").lower()

    @validator("order_type", pre=True)
    def _normalize_type(cls, value: Any) -> str:
        return str(value or "").lower()

    @validator(
        "price",
        "amount",
        "remaining",
        "stop_price",
        "notional",
        pre=True,
    )
    def _coerce_float(cls, value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    @validator("reduce_only", pre=True)
    def _coerce_bool(cls, value: Any) -> bool:
        return bool(value)

    @root_validator(pre=True)
    def _handle_aliases(cls, values: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        if "orderId" in values and "order_id" not in values:
            values["order_id"] = values.get("orderId")
        if "createdAt" in values and "created_at" not in values:
            values["created_at"] = values.get("createdAt")
        return values


class Position(_Model):
    symbol: str
    side: str
    notional: float
    entry_price: float
    mark_price: float
    liquidation_price: Optional[float] = None
    wallet_exposure_pct: Optional[float] = None
    unrealized_pnl: float
    max_drawdown_pct: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    size: Optional[float] = None
    signed_notional: Optional[float] = None
    volatility: Optional[Mapping[str, float]] = None
    funding_rates: Optional[Mapping[str, float]] = None
    daily_realized_pnl: float = 0.0
    position_side: Optional[str] = Field(default=None, alias="positionSide")
    position_idx: Optional[int] = Field(default=None, alias="positionIdx")

    @root_validator(pre=True)
    def _enforce_position_side(cls, values: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        legacy = values.get("position_side")
        alias = values.get("positionSide")
        if legacy is not None and alias is None:
            values["positionSide"] = legacy
        elif legacy is not None and alias is not None:
            if str(legacy).upper() != str(alias).upper():
                raise ValueError("Conflicting positionSide and position_side provided")
        return values

    @validator("side", pre=True)
    def _normalize_side(cls, value: Any) -> str:
        return str(value or "").lower()

    @validator(
        "notional",
        "entry_price",
        "mark_price",
        "liquidation_price",
        "wallet_exposure_pct",
        "max_drawdown_pct",
        "take_profit_price",
        "stop_loss_price",
        "size",
        "signed_notional",
        "unrealized_pnl",
        "daily_realized_pnl",
        pre=True,
    )
    def _coerce_float(cls, value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    @validator("position_side", pre=True)
    def _normalize_position_side(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        candidate = str(value).strip().upper()
        if candidate not in {"LONG", "SHORT", "BOTH"}:
            raise ValueError("positionSide must be LONG, SHORT, BOTH, or omitted")
        return candidate

    @validator("position_idx", pre=True)
    def _coerce_idx(cls, value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            idx = int(float(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None
        return idx

    def exposure_relative_to(self, balance: float) -> float:
        if balance == 0:
            return 0.0
        return abs(self.notional) / balance

    def pnl_pct(self, balance: float) -> float:
        if balance == 0:
            return 0.0
        return self.unrealized_pnl / balance


@dataclass
class AccountState(_Model):  # type: ignore[misc]
    name: str
    balance: Balance
    positions: Sequence[Position]
    orders: Sequence[Order] = ()
    pnl: Optional[PnLBreakdown] = None
    risk_limits: Optional[RiskLimits] = None
    exchange: Optional[ExchangeId] = None

    def __init__(self, **data: Any) -> None:  # type: ignore[override]
        super().__init__(**data)

    @validator("exchange", pre=True)
    def _coerce_exchange(cls, value: Any) -> Optional[ExchangeId]:
        if value is None:
            return None
        try:
            return ExchangeId(value)
        except ValueError:
            return ExchangeId.OTHER

    @validator("balance", pre=True)
    def _coerce_balance(cls, value: Any) -> Balance:
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

    @validator("positions", pre=True)
    def _coerce_positions(cls, value: Any) -> Sequence[Position]:
        return tuple(Position(**item) if not isinstance(item, Position) else item for item in value or [])

    @validator("orders", pre=True)
    def _coerce_orders(cls, value: Any) -> Sequence[Order]:
        return tuple(Order(**item) if not isinstance(item, Order) else item for item in value or [])

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
