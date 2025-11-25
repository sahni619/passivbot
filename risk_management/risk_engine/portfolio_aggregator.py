"""Aggregate cross-exchange snapshots into a portfolio view."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)


@dataclass
class PositionBreakdown:
    symbol: str
    notional: float
    unrealized_pnl: float

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AccountBreakdown:
    name: str
    balance: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    positions: List[PositionBreakdown] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["positions"] = [position.to_payload() for position in self.positions]
        return payload


@dataclass
class PortfolioView:
    total_balance: float
    total_equity: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    net_cashflow: float
    accounts: List[AccountBreakdown] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["accounts"] = [account.to_payload() for account in self.accounts]
        return payload


class PortfolioAggregator:
    """Compute a consolidated portfolio view from account snapshots."""

    def aggregate(
        self, accounts_payload: Iterable[Mapping[str, Any]], cashflow_events: Iterable[Mapping[str, Any]]
    ) -> PortfolioView:
        account_breakdowns: List[AccountBreakdown] = []
        total_balance = 0.0
        total_unrealized = 0.0
        total_realized = 0.0
        for account in accounts_payload:
            if not isinstance(account, Mapping):
                continue
            name = str(account.get("name", ""))
            balance = _coerce_float(account.get("balance"), fallback=0.0)
            unrealized = self._sum_unrealized(account.get("positions"))
            realized = _coerce_float(account.get("realized_pnl"), fallback=0.0)
            total_balance += balance
            total_unrealized += unrealized
            total_realized += realized
            equity = balance + unrealized + realized
            positions = self._positions(account.get("positions"))
            account_breakdowns.append(
                AccountBreakdown(
                    name=name,
                    balance=balance,
                    equity=equity,
                    unrealized_pnl=unrealized,
                    realized_pnl=realized,
                    positions=positions,
                )
            )
        net_cashflow = self._net_cashflow(cashflow_events)
        total_equity = total_balance + total_unrealized + total_realized
        return PortfolioView(
            total_balance=total_balance,
            total_equity=total_equity,
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=total_realized,
            net_cashflow=net_cashflow,
            accounts=account_breakdowns,
        )

    def _sum_unrealized(self, positions: Any) -> float:
        unrealized = 0.0
        for position in positions or []:
            if not isinstance(position, Mapping):
                continue
            unrealized += _coerce_float(position.get("unrealized_pnl") or position.get("pnl"), fallback=0.0)
        return unrealized

    def _positions(self, positions: Any) -> List[PositionBreakdown]:
        breakdowns: List[PositionBreakdown] = []
        for position in positions or []:
            if not isinstance(position, Mapping):
                continue
            symbol = str(position.get("symbol") or position.get("pair") or "").strip()
            if not symbol:
                continue
            notional = _coerce_float(position.get("signed_notional") or position.get("notional"), fallback=0.0)
            unrealized = _coerce_float(position.get("unrealized_pnl") or position.get("pnl"), fallback=0.0)
            breakdowns.append(PositionBreakdown(symbol=symbol, notional=abs(notional), unrealized_pnl=unrealized))
        return breakdowns

    def _net_cashflow(self, events: Iterable[Mapping[str, Any]]) -> float:
        net = 0.0
        for event in events or []:
            if not isinstance(event, Mapping):
                continue
            try:
                amount = float(event.get("amount", 0.0))
            except (TypeError, ValueError):
                continue
            event_type = str(event.get("type") or "").lower()
            if event_type == "deposit":
                net += amount
            elif event_type == "withdrawal":
                net -= amount
        return net


def _coerce_float(value: Any, *, fallback: float = 0.0) -> float:
    try:
        if value is None:
            return fallback
        return float(value)
    except (TypeError, ValueError):
        logger.debug("Failed to coerce %r to float; using fallback %s", value, fallback)
        return fallback
