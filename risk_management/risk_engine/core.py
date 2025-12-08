from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Position:
    """Simple immutable representation of an open position."""

    exchange: str
    symbol: str
    quantity: float
    entry_price: float
    mark_price: float
    leverage: float = 1.0

    def notional_value(self) -> float:
        return abs(self.quantity) * self.mark_price

    def unrealized_pnl(self) -> float:
        return (self.mark_price - self.entry_price) * self.quantity


@dataclass(frozen=True)
class CashFlowEvent:
    """Represents a balance-changing event unrelated to PnL."""

    timestamp: int
    amount: float  # positive for deposit, negative for withdrawal
    description: str | None = None


@dataclass(frozen=True)
class ExchangeDrawdown:
    exchange: str
    drawdown: float
    limit: float | None

    @property
    def breached(self) -> bool:
        return self.limit is not None and self.drawdown > self.limit


@dataclass(frozen=True)
class LimitBreach:
    exchange: str
    symbol: str
    kind: str
    value: float
    limit: float


def total_unrealized_pnl(positions: Iterable[Position]) -> float:
    """Return the sum of unrealized PnL across all positions."""

    return sum(position.unrealized_pnl() for position in positions)


def evaluate_exchange_drawdowns(
    current_equity: Dict[str, float],
    peak_equity: Dict[str, float],
    limits: Dict[str, float | None] | None = None,
) -> List[ExchangeDrawdown]:
    """Calculate drawdowns per exchange and whether they breach configured limits.

    Drawdown is computed as (peak - current) / peak when peak is positive.
    Exchanges absent from peak_equity are treated as having zero peak.
    Missing limits indicate no cap for the exchange and therefore no breach.
    """

    limits = limits or {}
    drawdowns: List[ExchangeDrawdown] = []
    exchanges = set(current_equity) | set(peak_equity) | set(limits)
    for exchange in exchanges:
        peak = peak_equity.get(exchange, 0.0)
        current = current_equity.get(exchange, 0.0)
        limit = limits.get(exchange)
        if peak <= 0:
            drawdown = 0.0
        else:
            drawdown = max(0.0, (peak - current) / peak)
        drawdowns.append(ExchangeDrawdown(exchange, drawdown, limit))
    return drawdowns


def evaluate_position_limits(
    positions: Iterable[Position],
    notional_limits: Dict[str, float] | None = None,
    leverage_limits: Dict[str, float] | None = None,
) -> List[LimitBreach]:
    """Check positions against notional and leverage caps per exchange.

    The limit dictionaries are keyed by exchange. Missing entries imply no cap.
    """

    notional_limits = notional_limits or {}
    leverage_limits = leverage_limits or {}
    breaches: List[LimitBreach] = []
    for position in positions:
        notional_cap = notional_limits.get(position.exchange)
        leverage_cap = leverage_limits.get(position.exchange)

        if notional_cap is not None and position.notional_value() > notional_cap:
            breaches.append(
                LimitBreach(
                    exchange=position.exchange,
                    symbol=position.symbol,
                    kind="notional",
                    value=position.notional_value(),
                    limit=notional_cap,
                )
            )

        if leverage_cap is not None and position.leverage > leverage_cap:
            breaches.append(
                LimitBreach(
                    exchange=position.exchange,
                    symbol=position.symbol,
                    kind="leverage",
                    value=position.leverage,
                    limit=leverage_cap,
                )
            )
    return breaches


def validate_one_way_mode(positions: Iterable[Position]) -> List[Tuple[str, str]]:
    """Detect symbol conflicts that violate one-way mode.

    Returns a list of (exchange, symbol) pairs where both long and short
    exposure exists simultaneously.
    """

    exposures: Dict[Tuple[str, str], List[float]] = {}
    for position in positions:
        key = (position.exchange, position.symbol)
        exposures.setdefault(key, []).append(position.quantity)

    conflicts: List[Tuple[str, str]] = []
    for (exchange, symbol), quantities in exposures.items():
        has_long = any(q > 0 for q in quantities)
        has_short = any(q < 0 for q in quantities)
        if has_long and has_short:
            conflicts.append((exchange, symbol))
    return conflicts


def separate_pnl_from_cash_flow(
    starting_balance: float,
    ending_balance: float,
    cash_flow_events: Sequence[CashFlowEvent] | None = None,
) -> Tuple[float, float]:
    """Return (net_pnl, net_cash_flow).

    Cash flow is the net of deposits (positive) and withdrawals (negative).
    Net PnL is derived as the balance change minus cash flow.
    """

    cash_flow_events = cash_flow_events or []
    net_cash_flow = sum(event.amount for event in cash_flow_events)
    balance_change = ending_balance - starting_balance
    net_pnl = balance_change - net_cash_flow
    return net_pnl, net_cash_flow
