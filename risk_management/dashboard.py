"""Terminal dashboard for monitoring trading portfolios.

The module consumes a JSON snapshot describing one or more accounts along with
alert thresholds. It renders a textual dashboard summarising exposure, profit
and loss, and risk metrics while also highlighting any triggered alerts.

The command line interface can either read a static snapshot file or, when
configured with realtime credentials, fetch fresh account information from the
supported exchanges on a configurable interval.

Additional helpers in this module support rendering and interacting with the
CLI risk dashboard."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .configuration import CustomEndpointSettings, load_realtime_config
from .domain.models import (
    Account,
    AlertThresholds,
    Order,
    Position,
    Scenario,
    ScenarioResult,
    ScenarioShock,
)


logger = logging.getLogger(__name__)


def _select_configured_scenarios(
    configured: Sequence[Scenario],
    requested: Optional[Sequence[str]],
) -> List[Scenario]:
    if not configured:
        if requested:
            names = ", ".join(str(name) for name in requested)
            raise ValueError(
                f"No configured scenarios available to match: {names}",
            )
        return []

    if not requested:
        return list(configured)

    index: Dict[str, Scenario] = {}
    for scenario in configured:
        keys = {scenario.name.lower()}
        if scenario.id:
            keys.add(scenario.id.lower())
        for key in keys:
            index.setdefault(key, scenario)

    selected: List[Scenario] = []
    missing: List[str] = []

    for entry in requested:
        key = str(entry).strip().lower()
        if not key:
            continue
        match = index.get(key)
        if match is None:
            missing.append(str(entry))
            continue
        if match not in selected:
            selected.append(match)

    if missing:
        raise ValueError(f"Unknown scenario(s): {', '.join(missing)}")

    return selected


def _parse_ad_hoc_scenario(values: Optional[Sequence[str]]) -> Optional[Scenario]:
    if not values:
        return None

    shocks: List[ScenarioShock] = []
    for raw in values:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(
                "Ad-hoc shocks must use the format SYMBOL:PCT (for example BTCUSDT:-0.10).",
            )
        symbol_part, pct_part = text.split(":", 1)
        symbol = symbol_part.strip().upper()
        if not symbol:
            raise ValueError("Shock definitions must include a symbol name before ':'.")
        try:
            pct = float(pct_part)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Shock '{text}' must include a numeric percentage change.",
            ) from exc
        shocks.append(ScenarioShock(symbol=symbol, price_pct=pct))

    if not shocks:
        return None

    description = ", ".join(
        f"{shock.symbol} {shock.price_pct:+.2%}" for shock in shocks
    )
    return Scenario(
        id="adhoc",
        name="Ad-hoc shock",
        description=description,
        shocks=tuple(shocks),
    )


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _format_pct(value: float) -> str:
    return f"{value * 100:6.2f}%"


def _format_price(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:,.2f}"


def _format_simple_number(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:,.4f}"


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_position(raw: Dict[str, Any]) -> Position:
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
        notional=float(raw["notional"]),
        entry_price=float(raw["entry_price"]),
        mark_price=float(raw["mark_price"]),
        liquidation_price=(
            float(raw["liquidation_price"])
            if raw.get("liquidation_price") is not None
            else None
        ),
        wallet_exposure_pct=(
            float(raw["wallet_exposure_pct"])
            if raw.get("wallet_exposure_pct") is not None
            else None
        ),
        unrealized_pnl=float(raw["unrealized_pnl"]),
        max_drawdown_pct=(
            float(raw["max_drawdown_pct"])
            if raw.get("max_drawdown_pct") is not None
            else None
        ),
        take_profit_price=(
            float(raw["take_profit_price"])
            if raw.get("take_profit_price") is not None
            else None
        ),
        stop_loss_price=(
            float(raw["stop_loss_price"])
            if raw.get("stop_loss_price") is not None
            else None
        ),
        size=float(size_raw) if size_raw not in (None, "") else None,
        signed_notional=(
            float(signed_notional_raw)
            if signed_notional_raw not in (None, "")
            else None
        ),
        volatility=volatility,
        funding_rates=funding_rates,
        daily_realized_pnl=float(raw.get("daily_realized_pnl", 0.0)),
        liquidity=liquidity,
        liquidity_warnings=tuple(liquidity_warnings),
    )


def _parse_account(raw: Mapping[str, Any]) -> Account:
    name_raw = raw.get("name")
    if name_raw in (None, "") and "account" in raw:
        name_raw = raw.get("account")

    balance_raw = raw.get("balance")

    if name_raw in (None, "") or balance_raw in (None, ""):
        raise ValueError("Account entries must include 'name' and 'balance'.")

    try:
        balance_value = float(balance_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Account balance must be numeric.") from exc

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
        name=str(name_raw),
        balance=balance_value,
        positions=positions,
        orders=orders,
        daily_realized_pnl=float(daily_realized),
        metadata=metadata or None,
    )


def _parse_order(raw: Mapping[str, Any]) -> Order:
    symbol = str(raw.get("symbol", ""))
    side = str(raw.get("side", "")).lower()
    order_type = str(raw.get("type") or raw.get("order_type") or "").lower()
    price = raw.get("price")
    amount = raw.get("amount")
    remaining = raw.get("remaining" if "remaining" in raw else "remaining_amount")
    reduce_only_raw = raw.get("reduce_only") if "reduce_only" in raw else raw.get("reduceOnly")
    stop_price = raw.get("stop_price") if "stop_price" in raw else raw.get("stopPrice")
    notional = raw.get("notional")
    created_at = raw.get("created_at") if "created_at" in raw else raw.get("createdAt")
    order_id = raw.get("order_id") if "order_id" in raw else raw.get("orderId") or raw.get("id")
    return Order(
        symbol=symbol,
        side=side,
        order_type=order_type,
        price=float(price) if price not in (None, "") else None,
        amount=float(amount) if amount not in (None, "") else None,
        remaining=float(remaining) if remaining not in (None, "") else None,
        status=str(raw.get("status", "")),
        reduce_only=bool(reduce_only_raw),
        stop_price=float(stop_price) if stop_price not in (None, "") else None,
        notional=float(notional) if notional not in (None, "") else None,
        order_id=str(order_id) if order_id not in (None, "") else None,
        created_at=str(created_at) if created_at not in (None, "") else None,
    )


def _parse_thresholds(raw: Dict[str, Any]) -> AlertThresholds:
    thresholds = AlertThresholds()
    for key in (
        "wallet_exposure_pct",
        "position_wallet_exposure_pct",
        "max_drawdown_pct",
        "loss_threshold_pct",
    ):
        if key in raw:
            setattr(thresholds, key, float(raw[key]))
    return thresholds


def load_snapshot(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def parse_snapshot(data: Dict[str, Any]) -> tuple[datetime, Sequence[Account], AlertThresholds, Sequence[str]]:
    generated_at_raw = data.get("generated_at")
    if generated_at_raw:
        try:
            generated_at = datetime.fromisoformat(generated_at_raw.replace("Z", "+00:00"))
        except ValueError:
            generated_at = datetime.now(timezone.utc)
    else:
        generated_at = datetime.now(timezone.utc)

    accounts: List[Account] = []
    accounts_raw = data.get("accounts", [])

    indexed_accounts: List[Tuple[int, Any]] = []
    key_lookup: Dict[int, Any] = {}

    if isinstance(accounts_raw, Mapping):
        for idx, (key, value) in enumerate(accounts_raw.items()):
            key_lookup[idx] = key
            indexed_accounts.append((idx, value))
    elif isinstance(accounts_raw, Iterable) and not isinstance(
        accounts_raw, (str, bytes, bytearray)
    ):
        indexed_accounts = list(enumerate(accounts_raw))
    else:
        if accounts_raw not in (None, []):
            logger.warning(
                "Skipping accounts payload because it is not an iterable of mappings; "
                "entries must be mappings but received %r (not a mapping).",
                accounts_raw,
            )
        indexed_accounts = []

    for index, raw_account in indexed_accounts:
        key_hint = key_lookup.get(index)
        if not isinstance(raw_account, Mapping):
            if key_hint is not None:
                logger.warning(
                    "Skipping account at index %s (key %r) because entry is not a mapping: %r",
                    index,
                    key_hint,
                    raw_account,
                )
            else:
                logger.warning(
                    "Skipping account at index %s because entry is not a mapping: %r",
                    index,
                    raw_account,
                )
            continue
        try:
            account = _parse_account(raw_account)
        except (TypeError, ValueError) as exc:
            name = raw_account.get("name", "<unknown>")
            if key_hint is not None:
                logger.warning(
                    "Skipping account %s at index %s (key %r) due to parse error: %s",
                    name,
                    index,
                    key_hint,
                    exc,
                )
            else:
                logger.warning(
                    "Skipping account %s at index %s due to parse error: %s",
                    name,
                    index,
                    exc,
                )
            continue
        accounts.append(account)
    thresholds = _parse_thresholds(data.get("alert_thresholds", {}))
    notifications = [str(channel) for channel in data.get("notification_channels", [])]
    return generated_at, accounts, thresholds, notifications


def evaluate_alerts(accounts: Sequence[Account], thresholds: AlertThresholds) -> List[str]:
    alerts: List[str] = []
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


def _normalise_policy_summary(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return None
    summary: Dict[str, Any] = {}
    evaluations_raw = payload.get("evaluations", [])
    evaluations: List[Dict[str, Any]] = []
    if isinstance(evaluations_raw, Iterable):
        for entry in evaluations_raw:
            if isinstance(entry, Mapping):
                evaluations.append(dict(entry))
    summary["evaluations"] = evaluations

    for key in ("active", "pending_actions", "manual_overrides"):
        values_raw = payload.get(key)
        items: List[Dict[str, Any]] = []
        if isinstance(values_raw, Iterable):
            for entry in values_raw:
                if isinstance(entry, Mapping):
                    items.append(dict(entry))
        if items:
            summary[key] = items
    return summary


def render_dashboard(
    generated_at: datetime,
    accounts: Sequence[Account],
    alerts: Sequence[str],
    notifications: Sequence[str],
    account_messages: Optional[Mapping[str, str]] = None,

    policy_summary: Optional[Mapping[str, Any]] = None,

    scenario_results: Optional[Sequence[ScenarioResult]] = None,

) -> str:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("Risk Management Dashboard")
    lines.append(f"Snapshot generated at: {generated_at.astimezone(timezone.utc).isoformat()}")
    lines.append("=" * 80)
    lines.append("")

    account_messages = account_messages or {}

    if not accounts:
        lines.append("No accounts available in the snapshot.")
    for account in accounts:
        lines.append(f"Account: {account.name}")
        lines.append(f"  Balance: {_format_currency(account.balance)}")
        lines.append(f"  Exposure: {_format_pct(account.exposure_pct())}")
        lines.append(f"  Unrealized PnL: {_format_currency(account.total_unrealized())}")
        lines.append(f"  Daily realized PnL: {_format_currency(account.daily_realized_pnl)}")
        metadata = getattr(account, "metadata", None) or {}
        scores = metadata.get("scores") if isinstance(metadata, Mapping) else {}
        concentration = metadata.get("concentration") if isinstance(metadata, Mapping) else {}
        limits = metadata.get("exposure_limits") if isinstance(metadata, Mapping) else {}
        breaches = metadata.get("limit_breaches") if isinstance(metadata, Mapping) else {}
        rating = None
        if isinstance(scores, Mapping) and scores.get("counterparty_rating"):
            rating = scores.get("counterparty_rating")
        elif isinstance(metadata, Mapping):
            rating = metadata.get("counterparty_rating")
        if rating:
            lines.append(f"  Counterparty rating: {rating}")

        def _append_limit_line(metric_key: str, label: str, value: Optional[float], detail: Optional[str] = None) -> None:
            if value is None:
                return
            line = f"  {label}: {_format_pct(value)}"
            if detail:
                line += f" ({detail})"
            breach_info = breaches.get(metric_key) if isinstance(breaches, Mapping) else None
            limit_value = None
            if breach_info and breach_info.get("limit") is not None:
                limit_value = breach_info.get("limit")
            elif isinstance(limits, Mapping):
                limit_value = limits.get(metric_key)
            status = None
            if breach_info:
                status = "BREACHED" if breach_info.get("breached") else "within limit"
            if limit_value is not None:
                try:
                    limit_float = float(limit_value)
                except (TypeError, ValueError):
                    limit_float = None
                if limit_float is not None:
                    line += f" (limit {_format_pct(limit_float)})"
            if status:
                line += f" [{status}]"
            lines.append(line)

        if isinstance(concentration, Mapping):
            venue_value = concentration.get("venue_concentration_pct")
            _append_limit_line("venue_concentration_pct", "Venue concentration", venue_value)
            asset_value = concentration.get("asset_concentration_pct")
            top_asset = concentration.get("top_asset")
            detail = f"top: {top_asset}" if top_asset else None
            _append_limit_line("asset_concentration_pct", "Asset concentration", asset_value, detail)
        status_message = account_messages.get(account.name)
        if status_message:
            lines.append(f"  Status: {status_message}")
        lines.append("")
        if account.positions:
            header = (
                f"    {'Symbol':<10}{'Side':<6}{'Exposure':>10}{'PnL':>12}{'Entry':>12}{'Mark':>12}"
                f"{'Liq.':>12}{'Max DD':>10}{'TP':>12}{'SL':>12}"
            )
            lines.append(header)
            lines.append("    " + "-" * (len(header) - 4))
            for position in account.positions:
                balance = account.balance
                pos_exposure = position.exposure_relative_to(balance)
                pnl_pct = position.pnl_pct(balance)
                lines.append(
                    "    "
                    + f"{position.symbol:<10}{position.side:<6}{_format_pct(pos_exposure):>10}"
                    + f"{_format_pct(pnl_pct):>12}{_format_price(position.entry_price):>12}"
                    + f"{_format_price(position.mark_price):>12}{_format_price(position.liquidation_price):>12}"
                    + f"{_format_pct(position.max_drawdown_pct or 0.0):>10}"
                    + f"{_format_price(position.take_profit_price):>12}{_format_price(position.stop_loss_price):>12}"
                )
                liquidity = position.liquidity if isinstance(position.liquidity, Mapping) else None
                if liquidity:
                    details: List[str] = []
                    coverage = liquidity.get("coverage_pct")
                    if isinstance(coverage, (int, float)):
                        details.append(f"coverage {float(coverage):.0%}")
                    slippage = liquidity.get("slippage_pct")
                    if isinstance(slippage, (int, float)):
                        details.append(f"slippage {float(slippage):.2%}")
                    average_price = liquidity.get("average_price")
                    if isinstance(average_price, (int, float)):
                        details.append(f"avg {float(average_price):,.2f}")
                    source = liquidity.get("source")
                    if isinstance(source, str) and source:
                        details.append(f"source {source}")
                    if details:
                        lines.append("      Liquidity: " + ", ".join(details))
                    warnings = list(position.liquidity_warnings)
                    warning_entries = liquidity.get("warnings")
                    if isinstance(warning_entries, Iterable):
                        warnings.extend([str(item) for item in warning_entries if isinstance(item, str)])
                    if warnings:
                        unique = list(dict.fromkeys(warnings))
                        lines.append("      Liquidity warnings: " + ", ".join(unique))
        else:
            lines.append("    No open positions.")
        lines.append("")

    if scenario_results:
        lines.append("Scenario analysis")
        lines.append("-" * 80)
        for result in scenario_results:
            scenario = result.scenario
            identifier = scenario.name
            if scenario.id and scenario.id.lower() != scenario.name.lower():
                identifier = f"{identifier} [{scenario.id}]"
            lines.append(identifier)
            if scenario.description:
                lines.append(f"  {scenario.description}")
            if scenario.shocks:
                shocks_summary = ", ".join(
                    f"{shock.symbol}: {_format_pct(shock.price_pct)}"
                    for shock in scenario.shocks
                )
                lines.append(f"  Shocks: {shocks_summary}")
            portfolio = result.portfolio
            lines.append(
                "  "
                + f"Portfolio PnL: {_format_currency(portfolio.pnl)} | "
                + f"Balance: {_format_currency(portfolio.balance_after)} "
                + f"(from {_format_currency(portfolio.balance_before)})"
            )
            lines.append(
                "  "
                + f"Gross exposure: {_format_currency(portfolio.gross_exposure)} "
                + f"({_format_pct(portfolio.gross_exposure_pct)})"
            )
            lines.append(
                "  "
                + f"Net exposure: {_format_currency(portfolio.net_exposure)} "
                + f"({_format_pct(portfolio.net_exposure_pct)})"
            )
            top_symbols = list(portfolio.symbols)[:3]
            if top_symbols:
                lines.append("  Top exposures:")
                for symbol in top_symbols:
                    lines.append(
                        "    "
                        + f"{symbol.symbol}: gross {_format_currency(symbol.gross_notional)} "
                        + f"({_format_pct(symbol.gross_pct)}) net {_format_currency(symbol.net_notional)} "
                        + f"({_format_pct(symbol.net_pct)}) pnl {_format_currency(symbol.pnl)}"
                    )
            lines.append("")

    lines.append("Alerts")
    lines.append("-" * 80)
    if alerts:
        for item in alerts:
            lines.append(f"• {item}")
    else:
        lines.append("No active alerts. All monitored metrics are within thresholds.")
    lines.append("")

    lines.append("Policies")
    lines.append("-" * 80)
    evaluations = (
        list(policy_summary.get("evaluations", []))
        if isinstance(policy_summary, Mapping)
        else []
    )
    if not evaluations:
        lines.append("No automated risk policies are configured.")
    else:
        for evaluation in evaluations:
            name = str(evaluation.get("name", "Unnamed policy"))
            status_bits: List[str] = []
            if evaluation.get("triggered"):
                status_bits.append("ACTIVE")
            else:
                status_bits.append("idle")
            if evaluation.get("cooldown_active"):
                status_bits.append("cooldown")
            if evaluation.get("override_active"):
                status_bits.append("override")
            status = ", ".join(status_bits)
            lines.append(f"• {name} [{status}]")
            description = evaluation.get("description")
            if description:
                lines.append(f"    {description}")
            metric = evaluation.get("metric")
            operator = evaluation.get("operator")
            threshold_value = _format_simple_number(
                _safe_float(evaluation.get("threshold"))
            )
            current_value = _format_simple_number(
                _safe_float(evaluation.get("value"))
            )
            lines.append(
                f"    Metric: {metric} {operator} {threshold_value} | current {current_value}"
            )
            actions = evaluation.get("actions", [])
            if isinstance(actions, Iterable):
                for action in actions:
                    if not isinstance(action, Mapping):
                        continue
                    action_type = action.get("type", "action")
                    action_status = action.get("status", "idle")
                    action_message = action.get("message")
                    suffix = f" – {action_message}" if action_message else ""
                    lines.append(
                        f"    - {action_type} [{action_status}]{suffix}"
                    )
            manual_override = evaluation.get("manual_override")
            if isinstance(manual_override, Mapping) and manual_override.get("allowed"):
                state = "active" if manual_override.get("active") else "available"
                lines.append(f"    Manual override {state}")
                instructions = manual_override.get("instructions")
                if instructions:
                    lines.append(f"      {instructions}")
    lines.append("")

    pending_actions = (
        list(policy_summary.get("pending_actions", []))
        if isinstance(policy_summary, Mapping)
        else []
    )
    if pending_actions:
        lines.append("Pending policy actions")
        lines.append("-" * 80)
        for pending in pending_actions:
            if not isinstance(pending, Mapping):
                continue
            policy_name = pending.get("policy", "Unknown policy")
            message = pending.get("message") or "Awaiting operator confirmation"
            lines.append(f"• {policy_name}: {message}")
        lines.append("")

    if notifications:
        lines.append("Notification channels")
        lines.append("-" * 80)
        for channel in notifications:
            lines.append(f"• {channel}")
    else:
        lines.append("No notification channels configured.")

    lines.append("=" * 80)
    return "\n".join(lines)


def run_dashboard(config_path: Path) -> str:
    snapshot = load_snapshot(config_path)
    return build_dashboard(snapshot)


def build_dashboard(
    snapshot: Dict[str, Any],
    *,
    scenario_results: Optional[Sequence[ScenarioResult]] = None,
) -> str:
    generated_at, accounts, thresholds, notifications = parse_snapshot(snapshot)
    alerts = evaluate_alerts(accounts, thresholds)
    account_messages = snapshot.get("account_messages", {}) if isinstance(snapshot, Mapping) else {}

    policy_summary = None
    if isinstance(snapshot, Mapping):
        policy_summary = _normalise_policy_summary(snapshot.get("policies"))

    return render_dashboard(
        generated_at,
        accounts,
        alerts,
        notifications,
        account_messages=account_messages,

        policy_summary=policy_summary,

        scenario_results=scenario_results,

    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render the risk management dashboard")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("dashboard_config.json"),
        help="Path to the JSON snapshot used for the dashboard.",
    )
    parser.add_argument(
        "--realtime-config",
        type=Path,
        default=None,
        help="Path to a realtime configuration file. Overrides --config when provided.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Refresh interval in seconds. Set to >0 to continuously monitor the file.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of times to render the dashboard when --interval is set. Use 0 to loop indefinitely.",
    )
    parser.add_argument(
        "--custom-endpoints",
        help=(
            "Override custom endpoint behaviour. Provide a JSON file path to reuse the same "
            "proxy configuration as the trading system, 'auto' to enable auto-discovery, or 'none' to "
            "disable overrides."
        ),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenario_names",
        help=(
            "Name or ID of a configured scenario to evaluate. Repeat to include multiple scenarios."
        ),
    )
    parser.add_argument(
        "--shock",
        action="append",
        dest="shocks",
        metavar="SYMBOL:PCT",
        help=(
            "Ad-hoc price shock expressed as a decimal percentage. Example: --shock BTCUSDT:-0.10"
        ),
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.interval < 0:
        parser.error("--interval must be >= 0")
    if args.iterations < 0:
        parser.error("--iterations must be >= 0")

    try:
        return asyncio.run(_run_cli(args))
    except FileNotFoundError:
        parser.error(f"Snapshot file not found: {args.config}")
    except json.JSONDecodeError as exc:
        parser.error(f"Snapshot file is not valid JSON: {exc}")
    except ValueError as exc:
        parser.error(str(exc))
    return 1


async def _run_cli(args: argparse.Namespace) -> int:
    from .services import RiskService

    realtime_service: Optional[RiskService] = None
    configured_scenarios: Sequence[Scenario] = []
    if args.realtime_config:
        realtime_config = load_realtime_config(Path(args.realtime_config))
        override = args.custom_endpoints
        if override is not None:
            override_normalized = override.strip()
            if not override_normalized:
                realtime_config.custom_endpoints = None
            else:
                lowered = override_normalized.lower()
                if lowered in {"none", "off", "disable"}:
                    realtime_config.custom_endpoints = CustomEndpointSettings(
                        path=None, autodiscover=False
                    )
                elif lowered in {"auto", "autodiscover", "default"}:
                    realtime_config.custom_endpoints = CustomEndpointSettings(
                        path=None, autodiscover=True
                    )
                else:
                    realtime_config.custom_endpoints = CustomEndpointSettings(
                        path=override_normalized, autodiscover=False
                    )
        configured_scenarios = list(realtime_config.scenarios)
        realtime_service = RiskService.from_config(realtime_config)
        logger.info("Starting realtime dashboard using %s", args.realtime_config)

    try:
        selected_configured = _select_configured_scenarios(
            configured_scenarios,
            getattr(args, "scenario_names", None),
        )
        ad_hoc = _parse_ad_hoc_scenario(getattr(args, "shocks", None))
        scenarios_to_run: List[Scenario] = list(selected_configured)
        if ad_hoc is not None:
            scenarios_to_run.append(ad_hoc)
        simulate = None
        if scenarios_to_run:
            from .stress import simulate_scenarios

            simulate = simulate_scenarios

        iteration = 0
        while True:
            if realtime_service is not None:
                snapshot = await realtime_service.fetch_snapshot()
            else:
                snapshot = load_snapshot(Path(args.config))
            scenario_results: Sequence[ScenarioResult] = []
            if simulate is not None:
                scenario_results = simulate(snapshot, scenarios_to_run)
            dashboard = build_dashboard(snapshot, scenario_results=scenario_results)
            print(dashboard)
            iteration += 1
            if args.iterations and iteration >= args.iterations:
                break
            if args.interval == 0:
                break
            await asyncio.sleep(args.interval)
        return 0
    finally:
        if realtime_service is not None:
            await realtime_service.close()


if __name__ == "__main__":
    sys.exit(main())

