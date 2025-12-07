"""Realtime data gathering for the risk management dashboard."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from types import TracebackType
from pathlib import Path

from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


from custom_endpoint_overrides import (
    CustomEndpointConfigError,
    configure_custom_endpoint_loader,
    load_custom_endpoint_config,
)

try:  # pragma: no cover - optional dependency when running tests
    from ccxt.base.errors import AuthenticationError
except (ModuleNotFoundError, ImportError):  # pragma: no cover - ccxt is optional for tests

    class AuthenticationError(Exception):
        """Fallback authentication error used when ccxt is unavailable."""

        pass

from risk_engine.policies import RiskViolation

from .account_clients import AccountClientProtocol, CCXTAccountClient
from .configuration import CustomEndpointSettings, RealtimeConfig
from .email_notifications import EmailAlertSender
from .telegram_notifications import TelegramNotifier
from services.telemetry import ResiliencePolicy, Telemetry
from .realtime_components import (
    ClientOrchestrator,
    KillSwitchExecutor,
    NotificationDispatcher,
    ResilientExecutor,
    SnapshotPolicyEvaluator,
)

logger = logging.getLogger(__name__)

def _exception_info(
    exc: BaseException,
) -> tuple[type[BaseException], BaseException, TracebackType | None]:
    """Return a ``logging`` compatible ``exc_info`` tuple for ``exc``."""

    return (type(exc), exc, exc.__traceback__)


def _build_search_paths(config_root: Path | None) -> tuple[str, ...]:
    """Return candidate custom endpoint paths prioritising the config directory."""

    candidates: list[str] = []
    if config_root is not None:
        candidate = (config_root / "custom_endpoints.json").resolve()
        candidates.append(str(candidate))
    default_path = os.path.join("configs", "custom_endpoints.json")
    if default_path not in candidates:
        candidates.append(default_path)
    # Remove duplicates while preserving order
    ordered = list(dict.fromkeys(candidates))
    return tuple(ordered)


def _configure_custom_endpoints(
    settings: Optional[CustomEndpointSettings], config_root: Optional[Path]
) -> None:
    """Initialise custom endpoint overrides before creating ccxt clients."""

    search_paths = _build_search_paths(config_root)

    if settings is None or (not settings.path and settings.autodiscover):
        preloaded = None
        try:
            preloaded = load_custom_endpoint_config(search_paths=search_paths)
        except CustomEndpointConfigError as exc:
            logger.warning("Failed to load custom endpoint config via discovery: %s", exc)
        configure_custom_endpoint_loader(None, autodiscover=True, preloaded=preloaded)
        source = preloaded.source_path if preloaded else None
        if source:
            logger.info("Using custom endpoints from %s", source)
        else:
            logger.info("No custom endpoint overrides found; using exchange defaults")
        return

    path = settings.path
    autodiscover = settings.autodiscover
    preloaded = None

    if path:
        try:
            preloaded = load_custom_endpoint_config(path)
        except CustomEndpointConfigError as exc:
            raise ValueError(f"Failed to load custom endpoint config '{path}': {exc}") from exc

    configure_custom_endpoint_loader(path, autodiscover=autodiscover, preloaded=preloaded)
    if path:
        logger.info("Using custom endpoints from %s", path)


class RealtimeDataFetcher:
    """Fetch realtime snapshots across multiple accounts."""

    def __init__(
        self,
        config: RealtimeConfig,
        account_clients: Optional[Sequence[AccountClientProtocol]] = None,
        telemetry: Optional[Telemetry] = None,
        *,
        policy_evaluator: Optional[PolicyEvaluator] = None,
        notification_handler: Optional[NotificationHandler] = None,
        kill_switch_handler: Optional[KillSwitchHandler] = None,
        orchestrator: Optional[ClientOrchestrator] = None,
        notification_dispatcher: Optional[NotificationDispatcher] = None,
        kill_switch_executor: Optional[KillSwitchExecutor] = None,
        executor: Optional[ResilientExecutor] = None,
    ) -> None:
        self.config = config
        self.telemetry = telemetry or Telemetry(policy=config.resilience)
        self.resilience_policy: ResiliencePolicy = config.resilience
        _configure_custom_endpoints(config.custom_endpoints, config.config_root)
        if account_clients is None:
            clients: List[AccountClientProtocol] = []
            for account in config.accounts:
                try:
                    clients.append(CCXTAccountClient(account))
                except RuntimeError as exc:
                    raise RuntimeError(
                        "Unable to create realtime clients. Install ccxt or provide custom account clients."
                    ) from exc
                except Exception as exc:
                    logger.error(
                        "Failed to initialise account client for %s: %s", account.name, exc, exc_info=True
                    )
                    raise
            self._account_clients = clients
        else:
            self._account_clients = list(account_clients)
            for account, client in zip(config.accounts, self._account_clients):
                if not hasattr(client, "config"):
                    client.config = account  # type: ignore[attr-defined]
            self.resilience_policy = ResiliencePolicy(
                request_timeout=config.resilience.request_timeout,
                max_retries=0,
                retry_backoff=0,
                circuit_breaker_threshold=config.resilience.circuit_breaker_threshold,
                circuit_breaker_reset_s=config.resilience.circuit_breaker_reset_s,
            )
            self.telemetry.policy = self.resilience_policy
        if config.debug_api_payloads:
            logger.info(
                "Exchange API payload debug logging enabled for realtime fetcher"
            )
        for account in config.accounts:
            if account.debug_api_payloads and not config.debug_api_payloads:
                logger.info(
                    "Debug API payload logging enabled for account %s", account.name
                )
        self._executor = executor or ResilientExecutor(self.telemetry, self.resilience_policy)
        self._orchestrator = orchestrator or ClientOrchestrator(
            self._account_clients,
            self._executor,
            account_messages=config.account_messages,
        )
        self._email_sender = EmailAlertSender(config.email) if config.email else None
        self._email_recipients = self._extract_email_recipients()
        self._telegram_targets = self._extract_telegram_targets()
        self._telegram_notifier = TelegramNotifier() if self._telegram_targets else None
        self._portfolio_stop_loss: Optional[Dict[str, Any]] = None
        self._last_portfolio_balance: Optional[float] = None
        self._conditional_stop_losses: list[Dict[str, Any]] = []
        self._policy_evaluator: PolicyEvaluator = policy_evaluator or SnapshotPolicyEvaluator()
        self._notification_dispatcher = notification_dispatcher or NotificationDispatcher(
            self._executor,
            email_sender=self._email_sender,
            email_recipients=self._email_recipients,
            telegram_notifier=self._telegram_notifier,
            telegram_targets=self._telegram_targets,
        )
        self._notification_handler: NotificationHandler = notification_handler or self._notification_dispatcher.dispatch
        self._kill_switch_handler = kill_switch_handler
        self._kill_switch_executor = kill_switch_executor or KillSwitchExecutor(
            self._account_clients, self._executor
        )

    async def _resilient_call(self, name: str, func: Callable[[], Any]) -> Any:
        return await self._executor.execute(name, func)

    async def _resilient_threaded_call(self, name: str, func: Callable[[], Any]) -> Any:
        return await self._executor.execute_threaded(name, lambda: asyncio.to_thread(func))

    def _extract_email_recipients(self) -> List[str]:
        recipients: List[str] = []
        for channel in self.config.notification_channels:
            if not isinstance(channel, str):
                continue
            if channel.lower().startswith("email:"):
                address = channel.split(":", 1)[1].strip()
                if address:
                    recipients.append(address)
        return recipients

    def _extract_telegram_targets(self) -> List[tuple[str, str]]:
        targets: List[tuple[str, str]] = []
        for channel in self.config.notification_channels:
            if not isinstance(channel, str):
                continue
            if not channel.lower().startswith("telegram:"):
                continue
            payload = channel.split(":", 1)[1]
            token = ""
            chat_id = ""
            if "@" in payload:
                token, _, chat_id = payload.partition("@")
            elif "/" in payload:
                token, _, chat_id = payload.partition("/")
            else:
                parts = payload.split(":", 1)
                if len(parts) == 2:
                    token, chat_id = parts
            token = token.strip()
            chat_id = chat_id.strip()
            if token and chat_id:
                targets.append((token, chat_id))
        return targets

    async def fetch_snapshot(self) -> Dict[str, Any]:
        accounts_payload, account_messages = await self._orchestrator.fetch()
        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "accounts": accounts_payload,
            "alert_thresholds": self.config.alert_thresholds,
            "notification_channels": self.config.notification_channels,
        }
        if account_messages:
            snapshot["account_messages"] = account_messages
        portfolio_balance = sum(
            float(account.get("balance", 0.0)) for account in accounts_payload
        )
        unrealized_total = sum(float(account.get("unrealized_pnl", 0.0)) for account in accounts_payload)
        portfolio_equity = portfolio_balance + unrealized_total
        self._last_portfolio_balance = portfolio_balance
        stop_loss_state = self._update_portfolio_stop_loss_state(portfolio_balance)
        if stop_loss_state:
            snapshot["portfolio_stop_loss"] = stop_loss_state
        conditional_state = self._evaluate_conditional_stop_losses(
            portfolio_balance, portfolio_equity, unrealized_total
        )
        if conditional_state:
            snapshot["conditional_stop_losses"] = conditional_state

        violations = list(self._policy_evaluator(snapshot))
        if violations:
            snapshot["policy_violations"] = [violation.as_dict() for violation in violations]
        if self._kill_switch_handler is not None:
            try:
                await self._kill_switch_handler(violations, snapshot)
            except Exception:
                logger.exception("Kill switch handler failed", exc_info=True)
        notification_result = self._notification_handler(violations, snapshot)
        if asyncio.iscoroutine(notification_result):
            await notification_result

        return snapshot

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self._account_clients))

    async def execute_kill_switch(
        self, account_name: Optional[str] = None, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        return await self._kill_switch_executor.execute(account_name, symbol)


    def _update_portfolio_stop_loss_state(
        self, portfolio_balance: float
    ) -> Optional[Dict[str, Any]]:
        if self._portfolio_stop_loss is None:
            return None
        state = dict(self._portfolio_stop_loss)
        state.setdefault("active", True)
        state.setdefault("triggered", False)
        state.setdefault("threshold_pct", 0.0)
        if state.get("baseline_balance") is None and portfolio_balance:
            state["baseline_balance"] = portfolio_balance
        baseline = state.get("baseline_balance")
        threshold_pct = state.get("threshold_pct")
        drawdown: Optional[float] = None
        if baseline and baseline > 0:
            drawdown = max(0.0, (baseline - portfolio_balance) / baseline)
        state["current_balance"] = portfolio_balance
        state["current_drawdown_pct"] = drawdown
        if (
            isinstance(threshold_pct, (int, float))
            and threshold_pct > 0
            and drawdown is not None
            and drawdown >= float(threshold_pct) / 100.0
            and not state.get("triggered")
        ):
            state["triggered"] = True
            state["triggered_at"] = datetime.now(timezone.utc).isoformat()
        self._portfolio_stop_loss = state
        return dict(state)

    def get_portfolio_stop_loss(self) -> Optional[Dict[str, Any]]:
        if self._portfolio_stop_loss is None:
            return None
        return dict(self._portfolio_stop_loss)

    async def set_portfolio_stop_loss(self, threshold_pct: float) -> Dict[str, Any]:
        if threshold_pct <= 0:
            raise ValueError("Portfolio stop loss threshold must be greater than zero.")
        state = {
            "threshold_pct": float(threshold_pct),
            "baseline_balance": self._last_portfolio_balance,
            "triggered": False,
            "triggered_at": None,
            "active": True,
        }
        self._portfolio_stop_loss = state
        return dict(state)

    async def clear_portfolio_stop_loss(self) -> None:
        self._portfolio_stop_loss = None

    def _evaluate_conditional_stop_losses(
        self, portfolio_balance: float, portfolio_equity: float, unrealized: float
    ) -> list[Dict[str, Any]]:
        if not self._conditional_stop_losses:
            return []

        updated: list[Dict[str, Any]] = []
        for condition in self._conditional_stop_losses:
            entry = dict(condition)
            metric = str(entry.get("metric", "")).lower()
            threshold = entry.get("threshold")
            operator = str(entry.get("operator", "lte")).lower()
            value: Optional[float] = None
            if metric == "balance_below":
                value = portfolio_balance
            elif metric == "equity_below":
                value = portfolio_equity
            elif metric == "equity_drawdown_pct":
                baseline = entry.get("baseline") or portfolio_equity
                entry["baseline"] = baseline
                if baseline:
                    value = max(0.0, (baseline - portfolio_equity) / baseline * 100.0)
            elif metric == "unrealized_loss_pct":
                value = (unrealized / portfolio_equity * 100.0) if portfolio_equity else None

            entry["last_value"] = value
            triggered = bool(entry.get("triggered"))
            if value is not None and threshold is not None:
                if operator in {"lt", "lte", "below", "<="}:
                    condition_met = value <= float(threshold)
                else:
                    condition_met = value >= float(threshold)
                if condition_met and not triggered:
                    entry["triggered"] = True
                    entry["triggered_at"] = datetime.now(timezone.utc).isoformat()
            updated.append(entry)

        self._conditional_stop_losses = updated
        return [dict(item) for item in updated]

    def get_conditional_stop_losses(self) -> list[Dict[str, Any]]:
        return [dict(item) for item in self._conditional_stop_losses]

    async def add_conditional_stop_loss(
        self,
        name: str,
        metric: str,
        threshold: float,
        operator: str = "lte",
    ) -> Dict[str, Any]:
        metric_normalized = metric.strip().lower()
        if metric_normalized not in {"balance_below", "equity_below", "equity_drawdown_pct", "unrealized_loss_pct"}:
            raise ValueError("metric must be one of balance_below, equity_below, equity_drawdown_pct, unrealized_loss_pct")
        if threshold is None or float(threshold) <= 0:
            raise ValueError("threshold must be greater than zero")
        operator_normalized = operator.strip().lower() if operator else "lte"
        condition = {
            "name": name.strip() or metric,
            "metric": metric_normalized,
            "threshold": float(threshold),
            "operator": operator_normalized,
            "triggered": False,
            "triggered_at": None,
        }
        self._conditional_stop_losses = [
            entry for entry in self._conditional_stop_losses if entry.get("name") != condition["name"]
        ]
        self._conditional_stop_losses.append(condition)
        return dict(condition)

    async def clear_conditional_stop_losses(self, *, name: Optional[str] = None) -> None:
        if name is None:
            self._conditional_stop_losses = []
            return
        self._conditional_stop_losses = [
            entry for entry in self._conditional_stop_losses if entry.get("name") != name
        ]

    def _resolve_account_client(self, account_name: str) -> AccountClientProtocol:
        for client in self._account_clients:
            if client.config.name == account_name:
                return client
        raise ValueError(f"Account '{account_name}' is not configured for realtime monitoring.")

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
        client = self._resolve_account_client(account_name)
        normalized_amount = float(amount)
        normalized_price = float(price) if price is not None else None
        return await self._resilient_call(
            f"account:{account_name}:create_order",
            lambda: client.create_order(
                symbol, order_type, side, normalized_amount, normalized_price, params=params
            ),
        )

    async def cancel_order(
        self,
        account_name: str,
        order_id: str,
        *,
        symbol: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        client = self._resolve_account_client(account_name)
        normalized_id = str(order_id)
        return await self._resilient_call(
            f"account:{account_name}:cancel_order",
            lambda: client.cancel_order(normalized_id, symbol, params=params),
        )

    async def close_position(self, account_name: str, symbol: str) -> Mapping[str, Any]:
        client = self._resolve_account_client(account_name)
        return await self._resilient_call(
            f"account:{account_name}:close_position",
            lambda: client.close_position(symbol),
        )

    async def list_order_types(self, account_name: str) -> Sequence[str]:
        client = self._resolve_account_client(account_name)
        return await self._resilient_call(
            f"account:{account_name}:list_order_types",
            lambda: client.list_order_types(),
        )


def _extract_balance(balance: Mapping[str, Any], settle_currency: str) -> float:
    """Extract a numeric balance from ccxt balance payloads."""

    if not isinstance(balance, Mapping):
        return 0.0

    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    aggregate_keys = (
        "totalMarginBalance",
        "totalEquity",
        "totalWalletBalance",
        "marginBalance",
        "totalBalance",
    )

    def _find_nested_aggregate(value: Any) -> Optional[float]:
        if isinstance(value, Mapping):
            for key in aggregate_keys:
                candidate = _to_float(value.get(key))
                if candidate is not None:
                    return candidate
            for child in value.values():
                result = _find_nested_aggregate(child)
                if result is not None:
                    return result
        elif isinstance(value, (list, tuple)):
            for child in value:
                result = _find_nested_aggregate(child)
                if result is not None:
                    return result
        return None

    # Some exchanges expose aggregate balances directly on the top-level payload.
    for key in (*aggregate_keys, "equity"):
        candidate = _to_float(balance.get(key))
        if candidate is not None:
            return candidate

    info = balance.get("info")
    if isinstance(info, Mapping):
        for key in (*aggregate_keys, "equity"):
            candidate = _to_float(info.get(key))
            if candidate is not None:
                return candidate
        nested = _find_nested_aggregate(info)
        if nested is not None:
            return nested

    total = balance.get("total")
    if isinstance(total, Mapping) and total:
        if settle_currency in total:
            candidate = _to_float(total.get(settle_currency))
            if candidate is not None:
                return candidate
        summed = 0.0
        found_value = False
        for value in total.values():
            candidate = _to_float(value)
            if candidate is None:
                continue
            summed += candidate
            found_value = True
        if found_value:
            return summed

    for currency_key in (settle_currency, "USDT"):
        entry = balance.get(currency_key)
        if isinstance(entry, Mapping):
            for key in ("total", "free", "used"):
                candidate = _to_float(entry.get(key))
                if candidate is not None:
                    return candidate
        else:
            candidate = _to_float(entry)
            if candidate is not None:
                return candidate

    return 0.0


def _parse_position(position: Mapping[str, Any], balance: float) -> Optional[Dict[str, Any]]:
    size = _first_float(
        position.get("contracts"),
        position.get("size"),
        position.get("amount"),
        position.get("info", {}).get("positionAmt") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("size") if isinstance(position.get("info"), Mapping) else None,
    )
    if size is None or abs(size) < 1e-12:
        return None
    side = "long" if size > 0 else "short"
    entry_price = _first_float(
        position.get("entryPrice"),
        position.get("entry_price"),
        position.get("info", {}).get("entryPrice") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("avgEntryPrice") if isinstance(position.get("info"), Mapping) else None,
    )
    mark_price = _first_float(
        position.get("markPrice"),
        position.get("mark_price"),
        position.get("info", {}).get("markPrice") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("last") if isinstance(position.get("info"), Mapping) else None,
    )
    liquidation_price = _first_float(
        position.get("liquidationPrice"),
        position.get("info", {}).get("liquidationPrice") if isinstance(position.get("info"), Mapping) else None,
    )
    unrealized = _first_float(
        position.get("unrealizedPnl"),
        position.get("info", {}).get("unRealizedProfit") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("unrealisedPnl") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("upl") if isinstance(position.get("info"), Mapping) else None,
    ) or 0.0
    realized = _first_float(
        position.get("dailyRealizedPnl"),
        position.get("realizedPnl"),
        position.get("realisedPnl"),
        position.get("info", {}).get("dailyRealizedPnl")
        if isinstance(position.get("info"), Mapping)
        else None,
        position.get("info", {}).get("realizedPnl") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("realisedPnl") if isinstance(position.get("info"), Mapping) else None,
    ) or 0.0
    contract_size = _first_float(
        position.get("contractSize"),
        position.get("info", {}).get("contractSize") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("ctVal") if isinstance(position.get("info"), Mapping) else None,
    ) or 1.0
    notional = _first_float(
        position.get("notional"),
        position.get("notionalValue"),
        position.get("info", {}).get("notionalValue") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("notionalUsd") if isinstance(position.get("info"), Mapping) else None,
    )
    if notional is None:
        reference_price = mark_price or entry_price or 0.0
        notional = abs(size) * contract_size * reference_price
    notional_value = float(notional or 0.0)
    if size < 0 and notional_value > 0:
        signed_notional = -abs(notional_value)
    elif size > 0 and notional_value < 0:
        signed_notional = abs(notional_value)
    else:
        signed_notional = notional_value
    abs_notional = abs(signed_notional)
    take_profit = _first_float(
        position.get("takeProfitPrice"),
        position.get("tpPrice"),
        position.get("info", {}).get("takeProfitPrice") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("tpTriggerPx") if isinstance(position.get("info"), Mapping) else None,
    )
    stop_loss = _first_float(
        position.get("stopLossPrice"),
        position.get("slPrice"),
        position.get("info", {}).get("stopLossPrice") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("slTriggerPx") if isinstance(position.get("info"), Mapping) else None,
    )
    wallet_exposure = None
    if balance:
        wallet_exposure = abs_notional / balance if balance else None
    return {
        "symbol": str(position.get("symbol") or position.get("id") or "unknown"),
        "side": side,
        "notional": abs_notional,
        "entry_price": float(entry_price or 0.0),
        "mark_price": float(mark_price or 0.0),
        "liquidation_price": float(liquidation_price) if liquidation_price is not None else None,
        "wallet_exposure_pct": float(wallet_exposure) if wallet_exposure is not None else None,
        "unrealized_pnl": float(unrealized),
        "daily_realized_pnl": float(realized),
        "max_drawdown_pct": None,
        "take_profit_price": float(take_profit) if take_profit is not None else None,
        "stop_loss_price": float(stop_loss) if stop_loss is not None else None,
        "size": float(size),
        "signed_notional": signed_notional,
    }


def _parse_order(order: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(order, Mapping):
        return None
    symbol = order.get("symbol") or order.get("id")
    if not symbol:
        return None
    price = _first_float(
        order.get("price"),
        order.get("triggerPrice"),
        order.get("stopPrice"),
        order.get("info", {}).get("price") if isinstance(order.get("info"), Mapping) else None,
    )
    amount = _first_float(
        order.get("amount"),
        order.get("contracts"),
        order.get("size"),
        order.get("info", {}).get("origQty") if isinstance(order.get("info"), Mapping) else None,
    )
    if amount is None:
        return None
    remaining = _first_float(
        order.get("remaining"),
        order.get("remainingAmount"),
        order.get("info", {}).get("leavesQty") if isinstance(order.get("info"), Mapping) else None,
    )
    reduce_only_raw = order.get("reduceOnly")
    if isinstance(order.get("info"), Mapping):
        reduce_only_raw = reduce_only_raw or order["info"].get("reduceOnly")
    reduce_only = bool(reduce_only_raw)
    stop_price = _first_float(
        order.get("stopPrice"),
        order.get("triggerPrice"),
        order.get("info", {}).get("stopPrice") if isinstance(order.get("info"), Mapping) else None,
    )
    timestamp_raw = order.get("timestamp")
    created_at = None
    if isinstance(timestamp_raw, (int, float)):
        created_at = datetime.fromtimestamp(float(timestamp_raw) / 1000, timezone.utc).isoformat()
    else:
        datetime_str = order.get("datetime")
        if isinstance(datetime_str, str) and datetime_str:
            created_at = datetime_str
    notional = price * amount if price is not None else None
    return {
        "order_id": str(order.get("id") or order.get("clientOrderId") or ""),
        "symbol": str(symbol),
        "side": str(order.get("side") or "").lower(),
        "type": str(order.get("type") or "").lower(),
        "price": price,
        "amount": amount,
        "remaining": remaining,
        "status": str(order.get("status") or ""),
        "reduce_only": reduce_only,
        "stop_price": stop_price,
        "notional": notional,
        "created_at": created_at,
    }


def _first_float(*values: Any) -> Optional[float]:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None

PolicyEvaluator = Callable[[Mapping[str, Any]], Sequence[RiskViolation]]
NotificationHandler = Callable[[Sequence[RiskViolation], Mapping[str, Any]], None]
KillSwitchHandler = Callable[[Sequence[RiskViolation], Mapping[str, Any]], Awaitable[None]]

