"""Realtime data gathering for the risk management dashboard."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta
from types import TracebackType
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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

from ._notifications import NotificationCoordinator
from ._parsing import (
    extract_balance as _extract_balance,
    parse_order as _parse_order,
    parse_position as _parse_position,
)
from .account_clients import AccountClientProtocol, CCXTAccountClient

from .configuration import AccountConfig, CustomEndpointSettings, RealtimeConfig

from .audit import get_audit_logger
from .performance import PerformanceTracker
from .policies import PolicyEvaluationResult, PolicyEvaluator
from .risk_engine import (
    ActionExecutor,
    ExchangeClientAdapter,
    FileStateStore,
    PortfolioAggregator,
    RiskEngineConfig,
    RiskRulesEngine,
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
    ) -> None:
        self.config = config
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
        self._last_auth_errors: Dict[str, str] = {}
        if config.debug_api_payloads:
            logger.info(
                "Exchange API payload debug logging enabled for realtime fetcher"
            )
        for account in config.accounts:
            if account.debug_api_payloads and not config.debug_api_payloads:
                logger.info(
                    "Debug API payload logging enabled for account %s", account.name
                )

        audit_logger = get_audit_logger(config.audit)
        self._notifications = NotificationCoordinator(
            config,
            audit_logger=audit_logger,
        )
        self._policy_evaluator: Optional[PolicyEvaluator]
        if config.policies:
            self._policy_evaluator = PolicyEvaluator(config.policies)
        else:
            self._policy_evaluator = None

        self._portfolio_stop_loss: Optional[Dict[str, Any]] = None
        self._last_portfolio_balance: Optional[float] = None
        self._account_stop_losses: Dict[str, Dict[str, Any]] = {}
        self._last_account_balances: Dict[str, float] = {}
        reports_dir = config.reports_dir
        if reports_dir is None:
            base_root = Path(__file__).resolve().parent
            reports_dir = base_root / "reports"
        self._performance_tracker = PerformanceTracker(Path(reports_dir))
        self._risk_config = RiskEngineConfig.from_realtime_config(config)
        state_path = self._risk_config.state_path or Path(reports_dir) / "risk_state.json"
        self._state_store = FileStateStore(state_path)
        self._portfolio_aggregator = PortfolioAggregator()
        self._risk_rules_engine = RiskRulesEngine(self._risk_config)
        self._exchange_clients = [ExchangeClientAdapter(client) for client in self._account_clients]
        self._action_executor = ActionExecutor(
            self._risk_config,
            state_store=self._state_store,
            notification_coordinator=self._notifications,
            account_clients=self._exchange_clients,
        )

    def _ensure_portfolio_aggregator(self) -> None:
        """Instantiate a portfolio aggregator if the attribute is missing."""

        if not hasattr(self, "_portfolio_aggregator"):
            logger.debug(
                "Portfolio aggregator attribute missing; creating a new instance for fetch cycle"
            )
            self._portfolio_aggregator = PortfolioAggregator()

    def _ensure_risk_rules_engine(self) -> None:
        """Instantiate a risk rules engine if the attribute is missing."""

        if not hasattr(self, "_risk_rules_engine"):
            logger.debug(
                "Risk rules engine attribute missing; creating a new instance for fetch cycle"
            )
            self._risk_rules_engine = RiskRulesEngine(self._risk_config)

    async def fetch_snapshot(self) -> Dict[str, Any]:
        self._ensure_portfolio_aggregator()
        self._ensure_risk_rules_engine()
        tasks = [client.fetch() for client in self._account_clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        accounts_payload: List[Dict[str, Any]] = []
        account_messages: Dict[str, str] = dict(self.config.account_messages)
        cashflow_events: List[Dict[str, Any]] = []
        account_balances: Dict[str, float] = {}
        account_entries: List[tuple[AccountConfig, Dict[str, Any], float]] = []
        for account_config, result in zip(self.config.accounts, results):
            payload: Dict[str, Any]
            balance_value = 0.0
            if isinstance(result, Exception):
                if isinstance(result, AuthenticationError):
                    message = (
                        f"{account_config.name}: authentication failed - {result}"
                    )

                    error_message = str(result)
                    previous_error = self._last_auth_errors.get(account_config.name)
                    if previous_error != error_message:
                        logger.warning(
                            "Authentication failed for %s: %s",
                            account_config.name,
                            result,
                        )
                        self._last_auth_errors[account_config.name] = error_message
                    else:
                        logger.debug(
                            "Authentication failure for %s unchanged: %s",
                            account_config.name,
                            result,
                        )

                else:
                    message = f"{account_config.name}: {result}"
                    logger.error(
                        "Failed to fetch snapshot for %s",
                        account_config.name,
                        exc_info=_exception_info(result),
                    )
                account_messages[account_config.name] = message
                payload = {"name": account_config.name, "balance": 0.0, "positions": []}
            else:
                if isinstance(result, Mapping):
                    payload = dict(result)
                else:
                    payload = {
                        "name": account_config.name,
                        "balance": 0.0,
                        "positions": [],
                    }
                try:
                    balance_value = float(payload.get("balance", 0.0))
                except (TypeError, ValueError):
                    balance_value = 0.0
                if account_config.name in self._last_auth_errors:
                    logger.info(
                        "Authentication for %s restored", account_config.name
                    )
                    self._last_auth_errors.pop(account_config.name, None)

            metadata_raw = payload.get("metadata")
            metadata: Dict[str, Any]
            if isinstance(metadata_raw, Mapping):
                metadata = dict(metadata_raw)
            else:
                metadata = {}
            if account_config.counterparty_rating:
                metadata.setdefault("counterparty_rating", account_config.counterparty_rating)
            if account_config.exposure_limits:
                metadata.setdefault("exposure_limits", dict(account_config.exposure_limits))
            payload["metadata"] = metadata

            raw_cashflows = payload.pop("cashflows", None)
            if raw_cashflows:
                extracted = self._extract_cashflow_events(account_config, raw_cashflows)
                if extracted:
                    cashflow_events.extend(extracted)

            accounts_payload.append(payload)
            account_balances[account_config.name] = balance_value
            account_entries.append((account_config, payload, balance_value))
        portfolio_balance = sum(account_balances.values())

        venue_concentration: Dict[str, float] = {}
        portfolio_asset_totals: Dict[str, float] = {}

        for account_config, payload, balance_value in account_entries:
            metadata = payload.get("metadata") or {}
            concentration: Dict[str, Any]
            if isinstance(metadata.get("concentration"), Mapping):
                concentration = dict(metadata["concentration"])
            else:
                concentration = {}

            ratio = balance_value / portfolio_balance if portfolio_balance else 0.0
            venue_concentration[account_config.name] = ratio
            concentration["venue_concentration_pct"] = ratio

            positions = payload.get("positions")
            symbol_totals: Dict[str, float] = {}
            total_abs_notional = 0.0
            if isinstance(positions, Iterable):
                for position in positions:
                    if not isinstance(position, Mapping):
                        continue
                    symbol_raw = position.get("symbol")
                    symbol = str(symbol_raw).strip()
                    if not symbol:
                        continue
                    signed_notional = position.get("signed_notional")
                    notional_value: Optional[float] = None
                    if signed_notional not in (None, ""):
                        try:
                            notional_value = abs(float(signed_notional))
                        except (TypeError, ValueError):
                            notional_value = None
                    if notional_value is None:
                        try:
                            notional_value = abs(float(position.get("notional", 0.0)))
                        except (TypeError, ValueError):
                            continue
                    if notional_value == 0.0:
                        continue
                    symbol_totals[symbol] = symbol_totals.get(symbol, 0.0) + notional_value
                    total_abs_notional += notional_value
                    portfolio_asset_totals[symbol] = portfolio_asset_totals.get(symbol, 0.0) + notional_value

            asset_breakdown: Dict[str, float] = {}
            top_asset = None
            top_value = 0.0
            if total_abs_notional > 0:
                for symbol, value in symbol_totals.items():
                    pct = value / total_abs_notional
                    asset_breakdown[symbol] = pct
                    if value > top_value:
                        top_asset = symbol
                        top_value = value
            asset_concentration = top_value / total_abs_notional if total_abs_notional > 0 else 0.0
            concentration["asset_concentration_pct"] = asset_concentration
            concentration["top_asset"] = top_asset
            if asset_breakdown:
                concentration["asset_breakdown"] = asset_breakdown

            scores: Dict[str, Any]
            if isinstance(metadata.get("scores"), Mapping):
                scores = dict(metadata["scores"])
            else:
                scores = {}
            rating = metadata.get("counterparty_rating")
            if rating and "counterparty_rating" not in scores:
                scores["counterparty_rating"] = rating
            scores["venue_concentration_pct"] = ratio
            scores["asset_concentration_pct"] = asset_concentration
            metadata["scores"] = scores

            limits_raw = metadata.get("exposure_limits")
            exposure_limits: Dict[str, float] = {}
            if isinstance(limits_raw, Mapping):
                for key, value in limits_raw.items():
                    try:
                        exposure_limits[str(key)] = float(value)
                    except (TypeError, ValueError):
                        logger.debug(
                            "Ignored non-numeric exposure limit %s=%r for account %s",
                            key,
                            value,
                            account_config.name,
                        )
                metadata["exposure_limits"] = exposure_limits

            breaches: Dict[str, Dict[str, Any]] = {}
            for metric, limit_value in exposure_limits.items():
                try:
                    limit_float = float(limit_value)
                except (TypeError, ValueError):
                    continue
                current_value: Optional[float]
                if metric == "venue_concentration_pct":
                    current_value = ratio
                elif metric == "asset_concentration_pct":
                    current_value = asset_concentration
                else:
                    current_value = concentration.get(metric) if isinstance(concentration, Mapping) else None
                if current_value is None:
                    continue
                breaches[metric] = {
                    "breached": current_value > limit_float,
                    "value": current_value,
                    "limit": limit_float,
                }
            if breaches:
                metadata["limit_breaches"] = breaches
            else:
                metadata.pop("limit_breaches", None)

            metadata["concentration"] = concentration
            payload["metadata"] = metadata

        generated_at_dt = datetime.now(timezone.utc)
        snapshot = {
            "generated_at": generated_at_dt.isoformat(),
            "accounts": accounts_payload,
            "alert_thresholds": self.config.alert_thresholds,
            "notification_channels": self.config.notification_channels,
        }
        if account_messages:
            snapshot["account_messages"] = account_messages
        self._last_account_balances = account_balances
        self._last_portfolio_balance = portfolio_balance
        if venue_concentration or portfolio_asset_totals:
            total_asset_notional = sum(portfolio_asset_totals.values())
            asset_ratios = {}
            if total_asset_notional > 0:
                asset_ratios = {
                    symbol: value / total_asset_notional
                    for symbol, value in sorted(
                        portfolio_asset_totals.items(), key=lambda item: item[1], reverse=True
                    )
                }
            snapshot["concentration"] = {
                "venues": venue_concentration,
                "assets": asset_ratios,
            }
        stop_loss_state = self._update_portfolio_stop_loss_state(portfolio_balance)
        if stop_loss_state:
            snapshot["portfolio_stop_loss"] = stop_loss_state
        account_stop_losses: Dict[str, Dict[str, Any]] = {}
        for account_name, balance in account_balances.items():
            state = self._update_account_stop_loss_state(account_name, balance)
            if state:
                account_stop_losses[account_name] = state
        if account_stop_losses:
            snapshot["account_stop_losses"] = account_stop_losses
        if cashflow_events:
            snapshot["cashflows"] = self._summarise_cashflows(cashflow_events)
        performance_summary = self._performance_tracker.record(
            generated_at=generated_at_dt,
            portfolio_balance=portfolio_balance,
            account_balances=account_balances,
        )
        if performance_summary:
            snapshot["performance"] = performance_summary
        portfolio_view = self._portfolio_aggregator.aggregate(accounts_payload, cashflow_events)
        snapshot["risk_portfolio"] = portfolio_view.to_payload()
        risk_decision = self._risk_rules_engine.evaluate(portfolio_view, self._state_store)
        snapshot["risk_decision"] = risk_decision.to_payload()
        policy_result = None
        if self._policy_evaluator is not None:
            policy_result = self._policy_evaluator.evaluate(snapshot)
            payload = policy_result.to_payload()
            if payload["evaluations"]:
                snapshot["policies"] = payload
        await self._action_executor.execute(risk_decision, portfolio_view)
        self._dispatch_notifications(snapshot, portfolio_balance, policy_result)
        return snapshot

    def _maybe_send_daily_balance_snapshot(
        self, snapshot: Mapping[str, Any], portfolio_balance: float
    ) -> None:
        try:
            self._notifications.send_daily_snapshot(snapshot, portfolio_balance)
        except Exception as exc:  # pragma: no cover - notification failures should not crash
            logger.debug(
                "Skipping daily balance snapshot notification due to error: %s", exc, exc_info=True
            )

    def _dispatch_notifications(
        self,
        snapshot: Mapping[str, Any],
        portfolio_balance: Optional[float] = None,
        policy_result: Optional["PolicyEvaluationResult"] = None,
    ) -> None:
        """Send relevant notifications for the snapshot.

        Historically this method was invoked with only the snapshot payload.
        To remain backward compatible while avoiding ``TypeError`` during web
        requests, the optional parameters fall back to safe defaults when not
        provided.
        """

        # Short-circuit if nothing is configured, allowing callers to invoke
        # notification dispatch unconditionally without extra guards.
        if not self.config.notification_channels and policy_result is None:
            return

        resolved_balance = portfolio_balance
        if resolved_balance is None:
            try:
                accounts = snapshot.get("accounts") if isinstance(snapshot, Mapping) else None
                if isinstance(accounts, Iterable):
                    resolved_balance = sum(
                        float(account.get("balance", 0.0))
                        for account in accounts
                        if isinstance(account, Mapping)
                    )
            except Exception:  # pragma: no cover - defensive fallback
                resolved_balance = None

        if resolved_balance is None:
            resolved_balance = self._last_portfolio_balance or 0.0

        self._maybe_send_daily_balance_snapshot(snapshot, resolved_balance)
        self._notifications.dispatch_alerts(snapshot)
        if policy_result is not None:
            self._notifications.handle_policy_evaluations(policy_result)

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self._account_clients))

    async def execute_kill_switch(
        self, account_name: Optional[str] = None, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        scope = account_name or "all accounts"
        symbol_desc = f" for {symbol}" if symbol else ""
        logger.info("Kill switch requested for %s%s", scope, symbol_desc)
        targets: List[AccountClientProtocol] = []
        for client in self._account_clients:
            if account_name is None or client.config.name == account_name:
                targets.append(client)
        if account_name is not None and not targets:
            raise ValueError(f"Account '{account_name}' is not configured for realtime monitoring.")
        results: Dict[str, Any] = {}
        for client in targets:
            try:
                results[client.config.name] = await client.kill_switch(symbol)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Kill switch failed for %s", client.config.name, exc_info=True)
                results[client.config.name] = {"error": str(exc)}
        logger.info("Kill switch completed for %s", scope)
        return results

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

    def _update_account_stop_loss_state(
        self, account_name: str, balance: float
    ) -> Optional[Dict[str, Any]]:
        state = self._account_stop_losses.get(account_name)
        if state is None:
            return None
        state = dict(state)
        state.setdefault("active", True)
        state.setdefault("triggered", False)
        state.setdefault("threshold_pct", 0.0)
        if state.get("baseline_balance") is None and balance:
            state["baseline_balance"] = balance
        baseline = state.get("baseline_balance")
        drawdown: Optional[float] = None
        if baseline and baseline > 0:
            drawdown = max(0.0, (baseline - balance) / baseline)
        state["current_balance"] = balance
        state["current_drawdown_pct"] = drawdown
        threshold_pct = state.get("threshold_pct")
        if (
            isinstance(threshold_pct, (int, float))
            and threshold_pct > 0
            and drawdown is not None
            and drawdown >= float(threshold_pct) / 100.0
            and not state.get("triggered")
        ):
            state["triggered"] = True
            state["triggered_at"] = datetime.now(timezone.utc).isoformat()
        self._account_stop_losses[account_name] = state
        return dict(state)

    def _extract_cashflow_events(
        self,
        account_config: AccountConfig,
        raw_events: Any,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if isinstance(raw_events, Mapping):
            candidates = raw_events.get("events")
            if isinstance(candidates, Iterable) and not isinstance(candidates, (str, bytes)):
                iterable = candidates
            else:
                iterable = [raw_events]
        elif isinstance(raw_events, Iterable) and not isinstance(raw_events, (str, bytes)):
            iterable = raw_events
        else:
            return events

        for entry in iterable:
            if not isinstance(entry, Mapping):
                continue
            normalised = self._normalise_cashflow_event(account_config, entry)
            if normalised is not None:
                events.append(normalised)
        return events

    def _normalise_cashflow_event(
        self,
        account_config: AccountConfig,
        entry: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        flow_type = str(entry.get("type", "")).strip().lower()
        if flow_type not in {"deposit", "withdrawal"}:
            return None

        try:
            amount = float(entry.get("amount", 0.0))
        except (TypeError, ValueError):
            return None
        if amount <= 0:
            return None

        currency_raw = entry.get("currency") or entry.get("code") or account_config.settle_currency
        currency = str(currency_raw).upper().strip() if currency_raw else ""
        if not currency:
            currency = "UNKNOWN"

        timestamp_ms = entry.get("timestamp_ms")
        timestamp_dt: Optional[datetime] = None
        if isinstance(timestamp_ms, (int, float)):
            timestamp_ms = int(timestamp_ms)
            timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        else:
            timestamp_raw = entry.get("timestamp") or entry.get("datetime")
            if isinstance(timestamp_raw, (int, float)):
                timestamp_ms = int(timestamp_raw)
                timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            elif isinstance(timestamp_raw, str) and timestamp_raw.strip():
                candidate = timestamp_raw.strip()
                if candidate.endswith("Z"):
                    candidate = candidate[:-1] + "+00:00"
                try:
                    parsed = datetime.fromisoformat(candidate)
                except ValueError:
                    parsed = None
                if parsed is not None:
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    timestamp_dt = parsed.astimezone(timezone.utc)
                    timestamp_ms = int(timestamp_dt.timestamp() * 1000)
        if timestamp_dt is None:
            timestamp_dt = datetime.now(timezone.utc)
            timestamp_ms = int(timestamp_dt.timestamp() * 1000)

        status_raw = entry.get("status")
        txid_raw = entry.get("txid") or entry.get("id")
        note_raw = entry.get("note")

        account_name = str(entry.get("account", account_config.name))
        exchange_name = str(entry.get("exchange", account_config.exchange))

        return {
            "account": account_name,
            "exchange": exchange_name,
            "type": flow_type,
            "amount": amount,
            "currency": currency,
            "timestamp": timestamp_dt.isoformat(),
            "timestamp_ms": int(timestamp_ms),
            "status": str(status_raw) if status_raw not in (None, "") else "",
            "txid": str(txid_raw) if txid_raw not in (None, "") else "",
            "note": str(note_raw) if isinstance(note_raw, str) else "",
        }

    def _summarise_cashflows(self, events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        ordered = sorted(
            (
                event
                for event in events
                if isinstance(event, Mapping) and event.get("timestamp_ms") is not None
            ),
            key=lambda item: int(item.get("timestamp_ms", 0)),
            reverse=True,
        )

        max_events = 200
        trimmed: List[Dict[str, Any]] = []
        for event in ordered[:max_events]:
            cleaned = {key: value for key, value in event.items() if key != "timestamp_ms"}
            trimmed.append(cleaned)

        now = datetime.now(timezone.utc)
        windows = {
            "7d": timedelta(days=7),
            "14d": timedelta(days=14),
            "21d": timedelta(days=21),
            "30d": timedelta(days=30),
        }
        summary: Dict[str, Dict[str, Any]] = {}
        for label, delta in windows.items():
            cutoff = now - delta
            per_currency: Dict[str, Dict[str, Any]] = {}
            deposit_count = 0
            withdrawal_count = 0
            for event in ordered:
                timestamp_ms = event.get("timestamp_ms")
                if not isinstance(timestamp_ms, (int, float)):
                    continue
                event_dt = datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc)
                if event_dt < cutoff:
                    continue
                currency = str(event.get("currency") or "").upper()
                if not currency:
                    currency = "UNKNOWN"
                bucket = per_currency.setdefault(
                    currency,
                    {
                        "currency": currency,
                        "deposits": 0.0,
                        "withdrawals": 0.0,
                        "net": 0.0,
                        "deposit_count": 0,
                        "withdrawal_count": 0,
                    },
                )
                try:
                    amount = float(event.get("amount", 0.0))
                except (TypeError, ValueError):
                    continue
                if event.get("type") == "deposit":
                    bucket["deposits"] += amount
                    bucket["net"] += amount
                    bucket["deposit_count"] += 1
                    deposit_count += 1
                else:
                    bucket["withdrawals"] += amount
                    bucket["net"] -= amount
                    bucket["withdrawal_count"] += 1
                    withdrawal_count += 1
            currencies = []
            total_deposits = 0.0
            total_withdrawals = 0.0
            for currency, data in per_currency.items():
                data["currency"] = currency
                data["deposits"] = float(data["deposits"])
                data["withdrawals"] = float(data["withdrawals"])
                data["net"] = float(data["net"])
                currencies.append(data)
                total_deposits += data["deposits"]
                total_withdrawals += data["withdrawals"]
            currencies.sort(key=lambda item: item["currency"])
            summary[label] = {
                "currencies": currencies,
                "totals": {
                    "deposit_count": deposit_count,
                    "withdrawal_count": withdrawal_count,
                    "deposits": total_deposits,
                    "withdrawals": total_withdrawals,
                    "net": total_deposits - total_withdrawals,
                },
            }
        return {"summary": summary, "events": trimmed}

    def get_account_stop_loss(self, account_name: str) -> Optional[Dict[str, Any]]:
        self._resolve_account_client(account_name)
        state = self._account_stop_losses.get(account_name)
        return dict(state) if state is not None else None

    async def set_account_stop_loss(self, account_name: str, threshold_pct: float) -> Dict[str, Any]:
        if threshold_pct <= 0:
            raise ValueError("Account stop loss threshold must be greater than zero.")
        self._resolve_account_client(account_name)
        baseline = self._last_account_balances.get(account_name)
        state = {
            "threshold_pct": float(threshold_pct),
            "baseline_balance": baseline,
            "triggered": False,
            "triggered_at": None,
            "active": True,
        }
        self._account_stop_losses[account_name] = state
        return dict(state)

    async def clear_account_stop_loss(self, account_name: str) -> None:
        self._resolve_account_client(account_name)
        self._account_stop_losses.pop(account_name, None)

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
        return await client.create_order(
            symbol, order_type, side, normalized_amount, normalized_price, params=params
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
        return await client.cancel_order(normalized_id, symbol, params=params)

    async def close_position(self, account_name: str, symbol: str) -> Mapping[str, Any]:
        client = self._resolve_account_client(account_name)
        return await client.close_position(symbol)

    async def list_order_types(self, account_name: str) -> Sequence[str]:
        client = self._resolve_account_client(account_name)
        return await client.list_order_types()

    async def cancel_all_orders(
        self, account_name: str, symbol: Optional[str] = None
    ) -> Mapping[str, Any]:
        client = self._resolve_account_client(account_name)
        return await client.cancel_all_orders(symbol)

    async def close_all_positions(
        self, account_name: str, symbol: Optional[str] = None
    ) -> Mapping[str, Any]:
        client = self._resolve_account_client(account_name)
        return await client.close_all_positions(symbol)
