"""Dashboard rendering helpers for CLI and web interfaces."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from ..core.domain import Account, ScenarioResult
from ..core.snapshots import evaluate_alerts, normalise_policy_summary, parse_snapshot
from .dashboard_formatting import (
    format_currency,
    format_pct,
    format_price,
    format_simple_number,
)

__all__ = ["build_dashboard", "render_dashboard"]


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_dashboard(
    snapshot: Mapping[str, Any],
    *,
    scenario_results: Optional[Sequence[ScenarioResult]] = None,
) -> str:
    generated_at, accounts, thresholds, notifications = parse_snapshot(snapshot)
    alerts = evaluate_alerts(accounts, thresholds)
    account_messages = snapshot.get("account_messages", {}) if isinstance(snapshot, Mapping) else {}

    policy_summary = None
    if isinstance(snapshot, Mapping):
        policy_summary = normalise_policy_summary(snapshot.get("policies"))

    return render_dashboard(
        generated_at,
        accounts,
        alerts,
        notifications,
        account_messages=account_messages,
        policy_summary=policy_summary,
        scenario_results=scenario_results,
    )


def render_dashboard(
    generated_at: datetime,
    accounts: Sequence[Account],
    alerts: Sequence[str],
    notifications: Sequence[str],
    account_messages: Optional[Mapping[str, str]] = None,
    *,
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
        lines.append(f"  Balance: {format_currency(account.balance)}")
        lines.append(f"  Exposure: {format_pct(account.exposure_pct())}")
        lines.append(f"  Unrealized PnL: {format_currency(account.total_unrealized())}")
        lines.append(f"  Daily realized PnL: {format_currency(account.daily_realized_pnl)}")
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
            line = f"  {label}: {format_pct(value)}"
            if detail:
                line += f" ({detail})"
            breach_info = breaches.get(metric_key) if isinstance(breaches, Mapping) else None
            limit_value = None
            if isinstance(limits, Mapping):
                limit_value = limits.get(metric_key)
            elif isinstance(metadata, Mapping):
                limit_value = metadata.get(metric_key)
            status = None
            if breach_info:
                status = "BREACHED" if breach_info.get("breached") else "within limit"
            if limit_value is not None:
                try:
                    limit_float = float(limit_value)
                except (TypeError, ValueError):
                    limit_float = None
                if limit_float is not None:
                    line += f" (limit {format_pct(limit_float)})"
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
                    + f"{position.symbol:<10}{position.side:<6}{format_pct(pos_exposure):>10}"
                    + f"{format_pct(pnl_pct):>12}{format_price(position.entry_price):>12}"
                    + f"{format_price(position.mark_price):>12}{format_price(position.liquidation_price):>12}"
                    + f"{format_pct(position.max_drawdown_pct or 0.0):>10}"
                    + f"{format_price(position.take_profit_price):>12}{format_price(position.stop_loss_price):>12}"
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
                    f"{shock.symbol}: {format_pct(shock.price_pct)}"
                    for shock in scenario.shocks
                )
                lines.append(f"  Shocks: {shocks_summary}")
            portfolio = result.portfolio
            lines.append(
                "  "
                + f"Portfolio PnL: {format_currency(portfolio.pnl)} | "
                + f"Balance: {format_currency(portfolio.balance_after)} "
                + f"(from {format_currency(portfolio.balance_before)})"
            )
            lines.append(
                "  "
                + f"Gross exposure: {format_currency(portfolio.gross_exposure)} "
                + f"({format_pct(portfolio.gross_exposure_pct)})"
            )
            lines.append(
                "  "
                + f"Net exposure: {format_currency(portfolio.net_exposure)} "
                + f"({format_pct(portfolio.net_exposure_pct)})"
            )
            top_symbols = list(portfolio.symbols)[:3]
            if top_symbols:
                lines.append("  Top exposures:")
                for symbol in top_symbols:
                    lines.append(
                        "    "
                        + f"{symbol.symbol}: gross {format_currency(symbol.gross_notional)} "
                        + f"({format_pct(symbol.gross_pct)}) net {format_currency(symbol.net_notional)} "
                        + f"({format_pct(symbol.net_pct)}) pnl {format_currency(symbol.pnl)}"
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
            threshold_value = format_simple_number(
                _safe_float(evaluation.get("threshold"))
            )
            current_value = format_simple_number(
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
            policy_name = pending.get("policy", "policy")
            message = pending.get("message") or pending.get("description")
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
