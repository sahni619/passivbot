"""Stress testing helpers for the risk management dashboard."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from .core.domain import (
    Scenario,
    ScenarioAccountImpact,
    ScenarioResult,
    ScenarioShock,
    ScenarioSymbolExposure,
)


def simulate_scenarios(
    snapshot: Mapping[str, Any],
    scenarios: Sequence[Scenario],
) -> List[ScenarioResult]:
    """Return scenario simulation results for ``snapshot``."""

    if not scenarios:
        return []

    from .presentation.snapshot_builder import (
        MAX_ACCOUNTS_PAGE_SIZE,
        build_presentable_snapshot,
    )

    presentable = build_presentable_snapshot(
        snapshot,
        page=1,
        page_size=MAX_ACCOUNTS_PAGE_SIZE,
    )
    accounts = presentable.get("accounts", [])
    if not isinstance(accounts, Sequence):
        return []

    results: List[ScenarioResult] = []
    for scenario in scenarios:
        results.append(_simulate_scenario(accounts, scenario))
    return results


def scenario_results_to_dict(results: Sequence[ScenarioResult]) -> List[Dict[str, Any]]:
    """Serialise scenario results into dictionaries suitable for JSON payloads."""

    return [asdict(result) for result in results]


def _simulate_scenario(
    accounts: Sequence[Mapping[str, Any]],
    scenario: Scenario,
) -> ScenarioResult:
    shock_lookup = {
        _normalise_symbol(shock.symbol): shock for shock in scenario.shocks
    }

    account_results: List[ScenarioAccountImpact] = []
    portfolio_symbols: Dict[str, MutableMapping[str, float]] = {}
    portfolio_balance_before = 0.0
    portfolio_pnl = 0.0

    for account in accounts:
        impact = _simulate_account(account, shock_lookup)
        account_results.append(impact)
        portfolio_balance_before += impact.balance_before
        portfolio_pnl += impact.pnl
        for symbol in impact.symbols:
            entry = portfolio_symbols.setdefault(
                symbol.symbol,
                {"gross": 0.0, "net": 0.0, "pnl": 0.0},
            )
            entry["gross"] += symbol.gross_notional
            entry["net"] += symbol.net_notional
            entry["pnl"] += symbol.pnl

    balance_after = portfolio_balance_before + portfolio_pnl
    portfolio_symbol_entries: List[ScenarioSymbolExposure] = []
    portfolio_gross = 0.0
    portfolio_net = 0.0

    for symbol, values in portfolio_symbols.items():
        gross = float(values.get("gross", 0.0))
        net = float(values.get("net", 0.0))
        pnl = float(values.get("pnl", 0.0))
        portfolio_gross += gross
        portfolio_net += net
        portfolio_symbol_entries.append(
            ScenarioSymbolExposure(
                symbol=symbol,
                gross_notional=gross,
                net_notional=net,
                gross_pct=_safe_ratio(gross, balance_after),
                net_pct=_safe_ratio(net, balance_after),
                pnl=pnl,
            )
        )

    portfolio_symbol_entries.sort(key=lambda entry: entry.gross_notional, reverse=True)

    portfolio_impact = ScenarioAccountImpact(
        name="Portfolio",
        balance_before=portfolio_balance_before,
        balance_after=balance_after,
        pnl=portfolio_pnl,
        gross_exposure=portfolio_gross,
        gross_exposure_pct=_safe_ratio(portfolio_gross, balance_after),
        net_exposure=portfolio_net,
        net_exposure_pct=_safe_ratio(portfolio_net, balance_after),
        symbols=tuple(portfolio_symbol_entries),
    )

    return ScenarioResult(
        scenario=scenario,
        portfolio=portfolio_impact,
        accounts=tuple(account_results),
    )


def _simulate_account(
    account: Mapping[str, Any],
    shock_lookup: Mapping[str, ScenarioShock],
) -> ScenarioAccountImpact:
    positions = account.get("positions")
    if not isinstance(positions, Sequence):
        positions = []

    balance_before = _to_float(account.get("balance"), 0.0)
    symbol_impacts: Dict[str, MutableMapping[str, float]] = {}
    total_pnl = 0.0
    gross_exposure = 0.0
    net_exposure = 0.0

    for position in positions:
        if not isinstance(position, Mapping):
            continue
        symbol = _normalise_symbol(position.get("symbol"))
        shock = shock_lookup.get(symbol)
        price_factor = 1.0 + (shock.price_pct if shock else 0.0)
        mark_price = _to_float(position.get("mark_price"), 0.0)
        entry_price = _to_float(position.get("entry_price"), 0.0)
        size = _resolve_position_size(position, mark_price)
        if size <= 0.0:
            continue
        direction = _resolve_direction(position)

        new_mark_price = mark_price * price_factor
        new_notional = size * new_mark_price
        signed_notional = new_notional * direction
        pnl = (new_mark_price - entry_price) * size * direction

        total_pnl += pnl
        gross_exposure += abs(signed_notional)
        net_exposure += signed_notional

        entry = symbol_impacts.setdefault(
            symbol,
            {"gross": 0.0, "net": 0.0, "pnl": 0.0},
        )
        entry["gross"] += abs(signed_notional)
        entry["net"] += signed_notional
        entry["pnl"] += pnl

    balance_after = balance_before + total_pnl
    symbol_entries: List[ScenarioSymbolExposure] = []
    for symbol, values in symbol_impacts.items():
        gross = float(values.get("gross", 0.0))
        net = float(values.get("net", 0.0))
        pnl = float(values.get("pnl", 0.0))
        symbol_entries.append(
            ScenarioSymbolExposure(
                symbol=symbol,
                gross_notional=gross,
                net_notional=net,
                gross_pct=_safe_ratio(gross, balance_after),
                net_pct=_safe_ratio(net, balance_after),
                pnl=pnl,
            )
        )
    symbol_entries.sort(key=lambda entry: entry.gross_notional, reverse=True)

    return ScenarioAccountImpact(
        name=str(account.get("name", "")),
        balance_before=balance_before,
        balance_after=balance_after,
        pnl=total_pnl,
        gross_exposure=gross_exposure,
        gross_exposure_pct=_safe_ratio(gross_exposure, balance_after),
        net_exposure=net_exposure,
        net_exposure_pct=_safe_ratio(net_exposure, balance_after),
        symbols=tuple(symbol_entries),
    )


def _resolve_position_size(position: Mapping[str, Any], mark_price: float) -> float:
    size_value = _to_optional_float(position.get("size"))
    if size_value is not None:
        return abs(size_value)

    if mark_price:
        signed_notional = _to_optional_float(position.get("signed_notional"))
        if signed_notional is not None:
            return abs(signed_notional) / abs(mark_price)
        notional = _to_optional_float(position.get("notional"))
        if notional is not None:
            return abs(notional) / abs(mark_price)

    return 0.0


def _resolve_direction(position: Mapping[str, Any]) -> float:
    signed_notional = _to_optional_float(position.get("signed_notional"))
    if signed_notional is not None and signed_notional != 0.0:
        return 1.0 if signed_notional > 0.0 else -1.0

    side = str(position.get("side", "")).lower()
    if side in {"short", "sell"}:
        return -1.0
    return 1.0


def _normalise_symbol(symbol: Any) -> str:
    if symbol is None:
        return ""
    return str(symbol).upper()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _to_float(value: Any, default: float) -> float:
    parsed = _to_optional_float(value)
    return parsed if parsed is not None else default


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["simulate_scenarios", "scenario_results_to_dict"]

