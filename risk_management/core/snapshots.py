"""Shared snapshot parsing and alert evaluation utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .domain import Account, AlertThresholds, Order, Position

__all__ = [
    "evaluate_alerts",
    "normalise_policy_summary",
    "parse_snapshot",
]


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _require_float(value: Any, field: str) -> float:
    result = _safe_float(value)
    if result is None:
        raise ValueError(f"{field} must be numeric")
    return result


def _parse_position(raw: Mapping[str, Any]) -> Position:
    required = [
        "symbol",
        "side",
        "notional",
        "entry_price",
        "mark_price",
        "unrealized_pnl",
    ]
    missing = [field for field in required if field not in raw]
    if missing:
        raise ValueError(f"Position missing required fields: {missing}")

    size_raw = raw.get("size")
    signed_notional_raw = raw.get("signed_notional")
    volatility = (
        {str(key): float(value) for key, value in raw.get("volatility", {}).items()}
        if isinstance(raw.get("volatility"), Mapping)
        else None
    )
    funding_rates = (
        {str(key): float(value) for key, value in raw.get("funding_rates", {}).items()}
        if isinstance(raw.get("funding_rates"), Mapping)
        else None
    )
    liquidity_raw = raw.get("liquidity") if isinstance(raw.get("liquidity"), Mapping) else None
    liquidity: Optional[Dict[str, Any]] = None
    liquidity_warnings: List[str] = []
    if isinstance(liquidity_raw, Mapping):
        liquidity = {}
        for key, value in liquidity_raw.items():
            if key == "warnings" and isinstance(value, Iterable):
                liquidity_warnings = [str(item) for item in value if isinstance(item, str)]
                liquidity["warnings"] = list(liquidity_warnings)
                continue
            if key in {
                "filled_size",
                "filled_notional",
                "average_price",
                "slippage_pct",
                "unfilled_size",
                "coverage_pct",
                "reference_price",
                "warning_threshold_pct",
                "timestamp",
            }:
                try:
                    liquidity[key] = float(value) if value is not None else None
                except (TypeError, ValueError):
                    liquidity[key] = None
            else:
                liquidity[key] = value
    extra_warnings = raw.get("liquidity_warnings")
    if isinstance(extra_warnings, Iterable):
        for warning in extra_warnings:
            if isinstance(warning, str) and warning not in liquidity_warnings:
                liquidity_warnings.append(warning)
    return Position(
        symbol=str(raw["symbol"]),
        side=str(raw.get("side", "")),
        notional=_require_float(raw["notional"], "notional"),
        entry_price=_require_float(raw["entry_price"], "entry_price"),
        mark_price=_require_float(raw["mark_price"], "mark_price"),
        liquidation_price=_safe_float(raw.get("liquidation_price")),
        wallet_exposure_pct=_safe_float(raw.get("wallet_exposure_pct")),
        unrealized_pnl=_require_float(raw["unrealized_pnl"], "unrealized_pnl"),
        max_drawdown_pct=_safe_float(raw.get("max_drawdown_pct")),
        take_profit_price=_safe_float(raw.get("take_profit_price")),
        stop_loss_price=_safe_float(raw.get("stop_loss_price")),
        size=_safe_float(size_raw),
        signed_notional=_safe_float(signed_notional_raw),
        volatility=volatility,
        funding_rates=funding_rates,
        daily_realized_pnl=_safe_float(raw.get("daily_realized_pnl", 0.0)) or 0.0,
        liquidity=liquidity,
        liquidity_warnings=tuple(liquidity_warnings),
    )


def _parse_order(raw: Mapping[str, Any]) -> Order:
    symbol = str(raw.get("symbol", ""))
    side = str(raw.get("side", "")).lower()
    order_type = str(raw.get("type") or raw.get("order_type") or "").lower()
    price = _safe_float(raw.get("price"))
    amount = _safe_float(raw.get("amount"))
    remaining = _safe_float(raw.get("remaining" if "remaining" in raw else "remaining_amount"))
    reduce_only_raw = raw.get("reduce_only") if "reduce_only" in raw else raw.get("reduceOnly")
    stop_price = _safe_float(raw.get("stop_price") if "stop_price" in raw else raw.get("stopPrice"))
    notional = _safe_float(raw.get("notional"))
    created_at = raw.get("created_at") if "created_at" in raw else raw.get("createdAt")
    order_id = raw.get("order_id") if "order_id" in raw else raw.get("orderId") or raw.get("id")
    return Order(
        symbol=symbol,
        side=side,
        order_type=order_type,
        price=price,
        amount=amount,
        remaining=remaining,
        status=str(raw.get("status", "")),
        reduce_only=bool(reduce_only_raw),
        stop_price=stop_price,
        notional=notional,
        order_id=str(order_id) if order_id not in (None, "") else None,
        created_at=str(created_at) if created_at not in (None, "") else None,
    )


def _parse_thresholds(raw: Mapping[str, Any]) -> AlertThresholds:
    thresholds = AlertThresholds()
    for key in (
        "wallet_exposure_pct",
        "position_wallet_exposure_pct",
        "max_drawdown_pct",
        "loss_threshold_pct",
    ):
        if key in raw:
            setattr(thresholds, key, _require_float(raw[key], key))
    return thresholds


def _parse_account(raw: Mapping[str, Any]) -> Account:
    if "name" not in raw or "balance" not in raw:
        raise ValueError("Account entries must include 'name' and 'balance'.")

    positions_raw = raw.get("positions", [])
    positions = [_parse_position(pos) for pos in positions_raw]
    orders_raw = raw.get("open_orders") or raw.get("orders") or []
    orders = [_parse_order(order) for order in orders_raw]
    daily_realized_raw = raw.get("daily_realized_pnl")
    if daily_realized_raw is None:
        daily_realized = sum(position.daily_realized_pnl for position in positions)
    else:
        try:
            daily_realized = float(daily_realized_raw)
        except (TypeError, ValueError):
            daily_realized = sum(position.daily_realized_pnl for position in positions)
    metadata: Dict[str, Any] = {}
    metadata_raw = raw.get("metadata")
    if isinstance(metadata_raw, Mapping):
        metadata = {str(key): value for key, value in metadata_raw.items()}
    for key in (
        "counterparty_rating",
        "exposure_limits",
        "concentration",
        "limit_breaches",
        "scores",
    ):
        if key in raw and key not in metadata:
            metadata[key] = raw[key]
    return Account(
        name=str(raw["name"]),
        balance=_require_float(raw["balance"], "balance"),
        positions=positions,
        orders=orders,
        daily_realized_pnl=float(daily_realized),
        metadata=metadata or None,
    )


def parse_snapshot(
    data: Mapping[str, Any],
) -> tuple[datetime, Sequence[Account], AlertThresholds, Sequence[str]]:
    generated_at_raw = data.get("generated_at")
    if generated_at_raw:
        try:
            generated_at = datetime.fromisoformat(str(generated_at_raw).replace("Z", "+00:00"))
        except ValueError:
            generated_at = datetime.now(timezone.utc)
    else:
        generated_at = datetime.now(timezone.utc)

    accounts = [_parse_account(acc) for acc in data.get("accounts", [])]
    thresholds = _parse_thresholds(data.get("alert_thresholds", {}))
    notifications = [str(channel) for channel in data.get("notification_channels", [])]
    return generated_at, tuple(accounts), thresholds, tuple(notifications)


def evaluate_alerts(accounts: Sequence[Account], thresholds: AlertThresholds) -> list[str]:
    alerts: list[str] = []
    for account in accounts:
        exposure = account.exposure_pct()
        if exposure > thresholds.wallet_exposure_pct:
            alerts.append(
                f"{account.name}: wallet exposure {exposure:.2%} exceeds limit {thresholds.wallet_exposure_pct:.2%}"
            )

        balance = account.balance
        unrealized_pct = account.total_unrealized() / balance if balance else 0.0
        if unrealized_pct < thresholds.loss_threshold_pct:
            alerts.append(
                f"{account.name}: unrealized PnL {unrealized_pct:.2%} below loss limit {thresholds.loss_threshold_pct:.2%}"
            )

        for position in account.positions:
            pos_exposure = position.exposure_relative_to(balance)
            if pos_exposure > thresholds.position_wallet_exposure_pct:
                alerts.append(
                    f"{account.name} {position.symbol}: exposure {pos_exposure:.2%} exceeds {thresholds.position_wallet_exposure_pct:.2%}"
                )
            if (
                position.max_drawdown_pct is not None
                and position.max_drawdown_pct > thresholds.max_drawdown_pct
            ):
                alerts.append(
                    f"{account.name} {position.symbol}: drawdown {position.max_drawdown_pct:.2%} exceeds {thresholds.max_drawdown_pct:.2%}"
                )

            liquidity = position.liquidity or {}
            raw_warnings = list(position.liquidity_warnings)
            if isinstance(liquidity, Mapping):
                warning_entries = liquidity.get("warnings")
                if isinstance(warning_entries, Iterable):
                    raw_warnings.extend([str(item) for item in warning_entries if isinstance(item, str)])
            seen: set[str] = set()
            for warning in raw_warnings:
                if warning in seen:
                    continue
                seen.add(warning)
                if warning == "depth_unavailable":
                    alerts.append(
                        f"{account.name} {position.symbol}: order book depth unavailable; liquidity estimates rely on fallbacks"
                    )
                elif warning == "insufficient_depth":
                    coverage = None
                    if isinstance(liquidity, Mapping):
                        coverage = liquidity.get("coverage_pct")
                        if isinstance(coverage, (int, float)):
                            coverage = float(coverage)
                    if isinstance(coverage, float):
                        alerts.append(
                            f"{account.name} {position.symbol}: order book covers only {coverage:.0%} of position size"
                        )
                    else:
                        alerts.append(
                            f"{account.name} {position.symbol}: insufficient order book depth to exit position"
                        )
                elif warning == "slippage_threshold_exceeded":
                    slippage = None
                    threshold_value = None
                    if isinstance(liquidity, Mapping):
                        slippage = liquidity.get("slippage_pct")
                        threshold_value = liquidity.get("warning_threshold_pct")
                    slippage_str = None
                    if isinstance(slippage, (int, float)):
                        slippage_str = f"{float(slippage):.2%}"
                    threshold_str = None
                    if isinstance(threshold_value, (int, float)):
                        threshold_str = f"{float(threshold_value):.2%}"
                    if slippage_str and threshold_str:
                        alerts.append(
                            f"{account.name} {position.symbol}: estimated slippage {slippage_str} exceeds threshold {threshold_str}"
                        )
                    elif slippage_str:
                        alerts.append(
                            f"{account.name} {position.symbol}: estimated slippage {slippage_str} exceeds configured threshold"
                        )
                    else:
                        alerts.append(
                            f"{account.name} {position.symbol}: estimated slippage exceeds configured threshold"
                        )
                elif warning == "position_size_undefined":
                    alerts.append(
                        f"{account.name} {position.symbol}: unable to determine position size for liquidity analysis"
                    )
    return alerts


def normalise_policy_summary(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return None
    summary: Dict[str, Any] = {}
    evaluations_raw = payload.get("evaluations", [])
    evaluations: list[Dict[str, Any]] = []
    if isinstance(evaluations_raw, Iterable):
        for entry in evaluations_raw:
            if isinstance(entry, Mapping):
                evaluations.append(dict(entry))
    summary["evaluations"] = evaluations

    for key in ("active", "pending_actions", "manual_overrides"):
        values_raw = payload.get(key)
        items: list[Dict[str, Any]] = []
        if isinstance(values_raw, Iterable):
            for entry in values_raw:
                if isinstance(entry, Mapping):
                    items.append(dict(entry))
        if items:
            summary[key] = items
    return summary
