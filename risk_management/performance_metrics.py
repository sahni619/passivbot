"""Utilities for deriving performance metrics from historical balances."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import fmean, pstdev
from typing import Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class DrawdownStats:
    """Container describing a maximum drawdown event."""

    amount: float
    percentage: Optional[float]
    peak_date: Optional[str]
    peak_balance: Optional[float]
    trough_date: Optional[str]
    trough_balance: Optional[float]

    def to_dict(self) -> dict[str, Optional[float | str]]:
        return {
            "amount": self.amount,
            "percentage": self.percentage,
            "peak_date": self.peak_date,
            "peak_balance": self.peak_balance,
            "trough_date": self.trough_date,
            "trough_balance": self.trough_balance,
        }


def _normalise_series(
    series: Sequence[Mapping[str, Optional[float | str]]]
) -> List[dict[str, Optional[float | str]]]:
    normalised: List[dict[str, Optional[float | str]]] = []
    for entry in series:
        date_value = entry.get("date")
        balance_value = entry.get("balance")
        if date_value is None or balance_value is None:
            continue
        try:
            balance = float(balance_value)
        except (TypeError, ValueError):
            continue
        normalised.append(
            {
                "date": str(date_value),
                "balance": balance,
                "timestamp": entry.get("timestamp"),
            }
        )
    normalised.sort(key=lambda item: item["date"])
    return normalised


def _calculate_daily_returns(
    equity_curve: Sequence[Mapping[str, Optional[float | str]]]
) -> List[dict[str, float | str | None]]:
    returns: List[dict[str, float | str | None]] = []
    previous_balance: Optional[float] = None
    for entry in equity_curve:
        balance = entry.get("balance")
        if not isinstance(balance, (int, float)):
            previous_balance = None
            continue
        if previous_balance is None or previous_balance == 0:
            previous_balance = float(balance)
            continue
        daily_return = (float(balance) - previous_balance) / previous_balance
        returns.append(
            {
                "date": entry.get("date"),
                "value": daily_return,
                "timestamp": entry.get("timestamp"),
            }
        )
        previous_balance = float(balance)
    return returns


def _calculate_max_drawdown(
    equity_curve: Sequence[Mapping[str, Optional[float | str]]]
) -> DrawdownStats:
    peak_balance: Optional[float] = None
    peak_date: Optional[str] = None
    max_drawdown_amount = 0.0
    max_drawdown_pct: Optional[float] = 0.0
    trough_balance: Optional[float] = None
    trough_date: Optional[str] = None

    for entry in equity_curve:
        balance = entry.get("balance")
        if not isinstance(balance, (int, float)):
            continue
        balance = float(balance)
        if peak_balance is None or balance > peak_balance:
            peak_balance = balance
            peak_date = entry.get("date")
            trough_balance = balance
            trough_date = entry.get("date")
        drawdown_amount = 0.0 if peak_balance is None else peak_balance - balance
        if peak_balance in (None, 0):
            drawdown_pct = None
        else:
            drawdown_pct = drawdown_amount / peak_balance
        should_update = False
        if drawdown_pct is None:
            should_update = False
        elif max_drawdown_pct is None or drawdown_pct > max_drawdown_pct:
            should_update = True
        elif drawdown_pct == max_drawdown_pct and drawdown_amount > max_drawdown_amount:
            should_update = True
        if should_update:
            max_drawdown_amount = drawdown_amount
            max_drawdown_pct = drawdown_pct
            trough_balance = balance
            trough_date = entry.get("date")

    if max_drawdown_pct is None:
        percentage = None
    else:
        percentage = float(max_drawdown_pct)

    return DrawdownStats(
        amount=float(max_drawdown_amount),
        percentage=percentage,
        peak_date=peak_date,
        peak_balance=float(peak_balance) if peak_balance is not None else None,
        trough_date=trough_date,
        trough_balance=float(trough_balance) if trough_balance is not None else None,
    )


def _calculate_sharpe_ratio(returns: Iterable[float], risk_free_rate: float = 0.0) -> Optional[float]:
    realised = [float(value) for value in returns if isinstance(value, (int, float))]
    if not realised:
        return None
    daily_rf = risk_free_rate / 252.0
    adjusted = [value - daily_rf for value in realised]
    if len(adjusted) < 2:
        return None
    mean_return = fmean(adjusted)
    std_dev = pstdev(adjusted)
    if std_dev == 0:
        return None
    return (mean_return / std_dev) * sqrt(252.0)


def build_performance_metrics(
    series: Sequence[Mapping[str, Optional[float | str]]],
    *,
    risk_free_rate: float = 0.0,
) -> dict[str, object]:
    """Return equity statistics derived from a historical balance series."""

    equity_curve = _normalise_series(series)
    if not equity_curve:
        return {
            "equity_curve": [],
            "latest_snapshot": None,
            "daily_returns": [],
            "max_drawdown": DrawdownStats(
                amount=0.0,
                percentage=None,
                peak_date=None,
                peak_balance=None,
                trough_date=None,
                trough_balance=None,
            ).to_dict(),
            "sharpe_ratio": None,
            "statistics": {
                "num_points": 0,
                "start_date": None,
                "end_date": None,
                "start_balance": None,
                "end_balance": None,
                "total_return": None,
                "total_return_pct": None,
            },
        }

    daily_returns = _calculate_daily_returns(equity_curve)
    sharpe_ratio = _calculate_sharpe_ratio(
        [entry["value"] for entry in daily_returns if isinstance(entry.get("value"), (int, float))],
        risk_free_rate=risk_free_rate,
    )
    drawdown = _calculate_max_drawdown(equity_curve)

    start_balance = equity_curve[0]["balance"]
    end_balance = equity_curve[-1]["balance"]
    if start_balance == 0:
        total_return_pct: Optional[float] = None
    else:
        total_return_pct = (float(end_balance) - float(start_balance)) / float(start_balance)

    return {
        "equity_curve": equity_curve,
        "latest_snapshot": equity_curve[-1],
        "daily_returns": daily_returns,
        "max_drawdown": drawdown.to_dict(),
        "sharpe_ratio": sharpe_ratio,
        "statistics": {
            "num_points": len(equity_curve),
            "start_date": equity_curve[0].get("date"),
            "end_date": equity_curve[-1].get("date"),
            "start_balance": float(start_balance),
            "end_balance": float(end_balance),
            "total_return": float(end_balance) - float(start_balance),
            "total_return_pct": total_return_pct,
        },
    }

