"""Risk policy definitions and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


@dataclass(frozen=True)
class RiskViolation:
    """A structured policy violation result."""

    policy: str
    message: str
    severity: str = "warning"
    subject: str | None = None
    data: Mapping[str, Any] | None = None

    def as_dict(self) -> MutableMapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "policy": self.policy,
            "message": self.message,
            "severity": self.severity,
        }
        if self.subject is not None:
            payload["subject"] = self.subject
        if self.data is not None:
            payload["data"] = dict(self.data)
        return payload


@dataclass
class RiskPolicy:
    """Configuration for a risk policy rule."""

    type: str
    threshold: float
    name: str | None = None
    severity: str = "warning"
    exchange: str | None = None
    symbol: str | None = None
    side: str | None = None
    description: str | None = None
    extra: MutableMapping[str, Any] = field(default_factory=dict)

    def evaluate(self, snapshot: Mapping[str, Any]) -> list[RiskViolation]:
        """Evaluate the policy against a snapshot payload."""

        evaluators = {
            "combined_pnl": _evaluate_combined_pnl,
            "exchange_drawdown": _evaluate_exchange_drawdown,
            "notional_cap": _evaluate_notional_cap,
            "one_way": _evaluate_one_way,
        }
        evaluator = evaluators.get(self.type)
        if evaluator is None:
            return []
        return evaluator(self, snapshot)


def evaluate_policies(
    policies: Iterable[RiskPolicy], snapshot: Mapping[str, Any]
) -> list[RiskViolation]:
    """Evaluate ``policies`` against ``snapshot`` and collect violations."""

    violations: list[RiskViolation] = []
    for policy in policies:
        violations.extend(policy.evaluate(snapshot))
    return violations


def _accounts(snapshot: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    accounts = snapshot.get("accounts") if isinstance(snapshot, Mapping) else None
    if isinstance(accounts, Sequence):
        return [account for account in accounts if isinstance(account, Mapping)]
    return []


def _positions_for_account(account: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    positions = account.get("positions") if isinstance(account, Mapping) else None
    if isinstance(positions, Sequence):
        return [pos for pos in positions if isinstance(pos, Mapping)]
    return []


def _evaluate_combined_pnl(policy: RiskPolicy, snapshot: Mapping[str, Any]) -> list[RiskViolation]:
    total_pnl = 0.0
    for account in _accounts(snapshot):
        total_pnl += float(account.get("daily_realized_pnl", 0.0))
        for position in _positions_for_account(account):
            total_pnl += float(position.get("unrealized_pnl", 0.0))
            total_pnl += float(position.get("daily_realized_pnl", 0.0))
    if total_pnl < policy.threshold:
        name = policy.name or "combined_pnl"
        message = f"Combined PnL {total_pnl:.2f} breached limit {policy.threshold:.2f}"
        return [RiskViolation(name, message, severity=policy.severity, data={"pnl": total_pnl})]
    return []


def _evaluate_exchange_drawdown(policy: RiskPolicy, snapshot: Mapping[str, Any]) -> list[RiskViolation]:
    violations: list[RiskViolation] = []
    for account in _accounts(snapshot):
        exchange = str(account.get("exchange") or "")
        if policy.exchange and exchange.lower() != policy.exchange.lower():
            continue
        balance = float(account.get("balance", 0.0))
        if balance <= 0:
            continue
        pnl_total = float(account.get("daily_realized_pnl", 0.0))
        for position in _positions_for_account(account):
            pnl_total += float(position.get("unrealized_pnl", 0.0))
            pnl_total += float(position.get("daily_realized_pnl", 0.0))
        drawdown_pct = pnl_total / balance
        limit = -abs(policy.threshold)
        if drawdown_pct <= limit:
            name = policy.name or "exchange_drawdown"
            message = (
                f"{exchange or 'unknown exchange'} drawdown {drawdown_pct:.2%} exceeds {limit:.2%}"
            )
            violations.append(
                RiskViolation(
                    name,
                    message,
                    severity=policy.severity,
                    subject=exchange or None,
                    data={"drawdown_pct": drawdown_pct},
                )
            )
    return violations


def _evaluate_notional_cap(policy: RiskPolicy, snapshot: Mapping[str, Any]) -> list[RiskViolation]:
    total_notional = 0.0
    for account in _accounts(snapshot):
        if policy.exchange and str(account.get("exchange") or "").lower() != policy.exchange.lower():
            continue
        for position in _positions_for_account(account):
            if policy.symbol and str(position.get("symbol") or "").lower() != policy.symbol.lower():
                continue
            total_notional += abs(float(position.get("notional", 0.0)))
    if total_notional > policy.threshold:
        name = policy.name or "notional_cap"
        subject = policy.symbol or policy.exchange or None
        message = (
            f"Notional {total_notional:.2f} exceeds cap {policy.threshold:.2f}"
        )
        return [
            RiskViolation(
                name,
                message,
                severity=policy.severity,
                subject=subject,
                data={"notional": total_notional},
            )
        ]
    return []


def _evaluate_one_way(policy: RiskPolicy, snapshot: Mapping[str, Any]) -> list[RiskViolation]:
    net_notional = 0.0
    for account in _accounts(snapshot):
        if policy.exchange and str(account.get("exchange") or "").lower() != policy.exchange.lower():
            continue
        for position in _positions_for_account(account):
            if policy.symbol and str(position.get("symbol") or "").lower() != policy.symbol.lower():
                continue
            side = str(position.get("side") or "").lower()
            notional = float(position.get("notional", 0.0))
            signed = float(position.get("signed_notional", 0.0)) if position.get("signed_notional") is not None else None
            if signed is None:
                if side == "short":
                    signed = -abs(notional)
                elif side:
                    signed = abs(notional)
                else:
                    signed = notional
            net_notional += signed
    allowed_side = (policy.side or "").lower()
    if allowed_side == "long" and net_notional < 0:
        direction = "short"
    elif allowed_side == "short" and net_notional > 0:
        direction = "long"
    else:
        return []
    name = policy.name or "one_way"
    message = (
        f"Net exposure {net_notional:.2f} violates one-way {allowed_side} policy by leaning {direction}"
    )
    return [
        RiskViolation(
            name,
            message,
            severity=policy.severity,
            subject=policy.symbol or policy.exchange or None,
            data={"net_notional": net_notional},
        )
    ]

