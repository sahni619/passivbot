"""Lightweight orchestrators bridging services and presentation layers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, Optional, Protocol, Sequence

from ..risk_engine.core import LimitBreach, Position as EnginePosition, evaluate_position_limits

from ..snapshot_utils import build_presentable_snapshot
from ..dashboard import parse_snapshot


class SnapshotProvider(Protocol):
    async def fetch_snapshot(self) -> Mapping[str, Any]:
        ...


Notifier = Callable[[Sequence[str], Mapping[str, Any]], Awaitable[None]]


@dataclass
class RiskEvaluation:
    alerts: Sequence[str]
    breaches: Sequence[LimitBreach]


class DashboardController:
    """Coordinate snapshot retrieval, risk evaluation, and persistence."""

    def __init__(
        self,
        provider: SnapshotProvider,
        *,
        history_store: Any = None,
        report_manager: Any = None,
        notifier: Optional[Notifier] = None,
    ) -> None:
        self.provider = provider
        self.history_store = history_store
        self.report_manager = report_manager
        self.notifier = notifier

    async def fetch_cli_view(self) -> tuple[Any, Sequence[Any], Any, Sequence[str], Mapping[str, str]]:
        snapshot = await self.provider.fetch_snapshot()
        generated_at, accounts, thresholds, notifications = parse_snapshot(dict(snapshot))
        account_messages = snapshot.get("account_messages", {}) if isinstance(snapshot, Mapping) else {}
        alerts = build_presentable_snapshot(snapshot).get("alerts", [])
        await self._maybe_notify(alerts, snapshot)
        await self._persist_snapshot(snapshot)
        return generated_at, accounts, thresholds, notifications, account_messages

    async def fetch_presentable_snapshot(self) -> Dict[str, Any]:
        snapshot = await self.provider.fetch_snapshot()
        view_model = build_presentable_snapshot(snapshot)
        view_model["risk_assessment"] = self._evaluate_risk_engine(view_model)
        await self._maybe_notify(view_model.get("alerts", []), view_model)
        await self._persist_snapshot(view_model)
        return view_model

    async def list_order_types(self, account_name: str) -> Sequence[str]:
        if hasattr(self.provider, "list_order_types"):
            return await self.provider.list_order_types(account_name)  # type: ignore[call-arg]
        raise ValueError(f"No exchange provider available for account {account_name}")

    async def place_order(
        self,
        account_name: str,
        *,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        if not hasattr(self.provider, "place_order"):
            raise RuntimeError("Order placement not supported by the configured provider")
        return await self.provider.place_order(  # type: ignore[attr-defined]
            account_name,
            symbol=symbol,
            order_type=order_type,
            side=side,
            amount=amount,
            price=price,
            params=params,
        )

    async def cancel_order(
        self,
        account_name: str,
        order_id: str,
        *,
        symbol: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        if not hasattr(self.provider, "cancel_order"):
            raise RuntimeError("Order cancellation not supported by the configured provider")
        return await self.provider.cancel_order(  # type: ignore[attr-defined]
            account_name, order_id, symbol=symbol, params=params
        )

    async def close_position(self, account_name: str, symbol: str) -> Mapping[str, Any]:
        if not hasattr(self.provider, "close_position"):
            raise RuntimeError("Position close not supported by the configured provider")
        return await self.provider.close_position(account_name, symbol)  # type: ignore[attr-defined]

    async def trigger_kill_switch(
        self, account_name: Optional[str] = None, *, symbol: Optional[str] = None
    ) -> Any:
        if not hasattr(self.provider, "execute_kill_switch"):
            raise RuntimeError("Kill switch not supported by the configured provider")
        return await self.provider.execute_kill_switch(account_name, symbol)  # type: ignore[attr-defined]

    async def get_portfolio_stop_loss(self) -> Optional[Mapping[str, Any]]:
        if hasattr(self.provider, "get_portfolio_stop_loss"):
            state = self.provider.get_portfolio_stop_loss()  # type: ignore[attr-defined]
            return dict(state) if state else None
        return None

    async def set_portfolio_stop_loss(self, threshold_pct: float) -> Mapping[str, Any]:
        if not hasattr(self.provider, "set_portfolio_stop_loss"):
            raise RuntimeError("Stop loss not supported by the configured provider")
        return await self.provider.set_portfolio_stop_loss(threshold_pct)  # type: ignore[attr-defined]

    async def clear_portfolio_stop_loss(self) -> None:
        if hasattr(self.provider, "clear_portfolio_stop_loss"):
            await self.provider.clear_portfolio_stop_loss()  # type: ignore[attr-defined]

    def list_conditional_stop_losses(self) -> list[Dict[str, Any]]:
        if hasattr(self.provider, "get_conditional_stop_losses"):
            return list(self.provider.get_conditional_stop_losses())  # type: ignore[attr-defined]
        return []

    async def add_conditional_stop_loss(
        self, name: str, metric: str, threshold: float, operator: str
    ) -> Mapping[str, Any]:
        if not hasattr(self.provider, "add_conditional_stop_loss"):
            raise RuntimeError("Conditional stop losses are not supported")
        return await self.provider.add_conditional_stop_loss(  # type: ignore[attr-defined]
            name, metric, threshold, operator
        )

    async def clear_conditional_stop_losses(self, name: Optional[str] = None) -> None:
        if hasattr(self.provider, "clear_conditional_stop_losses"):
            await self.provider.clear_conditional_stop_losses(name=name)  # type: ignore[attr-defined]

    async def generate_report(self, account_name: str, snapshot: Mapping[str, Any]) -> Any:
        if self.report_manager is None:
            raise RuntimeError("Report generation is not configured")
        view_model = build_presentable_snapshot(snapshot)
        return await self.report_manager.create_account_report(account_name, view_model)

    async def _maybe_notify(self, alerts: Sequence[str], payload: Mapping[str, Any]) -> None:
        if not alerts or self.notifier is None:
            return
        await self.notifier(alerts, payload)

    async def _persist_snapshot(self, payload: Mapping[str, Any]) -> None:
        if self.history_store is None:
            return
        try:
            if asyncio.iscoroutinefunction(self.history_store.record_async):  # type: ignore[attr-defined]
                await self.history_store.record_async(payload)  # type: ignore[attr-defined]
            elif hasattr(self.history_store, "record"):
                self.history_store.record(payload)  # type: ignore[attr-defined]
        except Exception:
            # Persistence failures should not break the user experience.
            return

    def _evaluate_risk_engine(self, view_model: Mapping[str, Any]) -> Mapping[str, Any]:
        breaches = self._evaluate_limits(view_model)
        return {
            "limit_breaches": [
                {
                    "exchange": breach.exchange,
                    "symbol": breach.symbol,
                    "kind": breach.kind,
                    "value": breach.value,
                    "limit": breach.limit,
                }
                for breach in breaches
            ]
        }

    def _evaluate_limits(self, view_model: Mapping[str, Any]) -> Sequence[LimitBreach]:
        accounts = view_model.get("accounts", []) if isinstance(view_model, Mapping) else []
        positions: list[EnginePosition] = []
        for account in accounts:
            exchange = account.get("exchange") or account.get("name", "") if isinstance(account, Mapping) else ""
            for position in account.get("positions", []) if isinstance(account, Mapping) else []:
                try:
                    mark_price = float(position.get("mark_price") or 0.0)
                    entry_price = float(position.get("entry_price") or mark_price)
                    notional = float(position.get("notional") or 0.0)
                except (TypeError, ValueError):
                    continue
                if mark_price == 0:
                    continue
                side = str(position.get("side", "")).lower()
                quantity = notional / mark_price if mark_price else 0.0
                if side == "short":
                    quantity *= -1
                positions.append(
                    EnginePosition(
                        exchange=exchange,
                        symbol=str(position.get("symbol", "")),
                        quantity=quantity,
                        entry_price=entry_price,
                        mark_price=mark_price,
                        leverage=float(position.get("leverage")) if position.get("leverage") else 1.0,
                    )
                )
        return evaluate_position_limits(positions)


class AdminController:
    """Isolate persistence and runtime reload concerns from the web layer."""

    def __init__(self, config) -> None:
        self.config = config

    def load_config_payload(self) -> Dict[str, Any]:
        if self.config.config_path is None:
            raise RuntimeError("Realtime configuration path is not available for editing")
        try:
            return json.loads(self.config.config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(f"Realtime configuration file not found: {exc}") from exc

    def save_config_payload(self, payload: Mapping[str, Any]) -> None:
        if self.config.config_path is None:
            raise RuntimeError("Realtime configuration path is not available for editing")
        self.config.config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


import json

