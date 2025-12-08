"""Composable realtime components with clear interfaces for testing."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, date, time, timezone
from typing import Any, Callable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from risk_engine.policies import RiskViolation

from .dashboard import evaluate_alerts, parse_snapshot
from .email_notifications import EmailAlertSender
from .telegram_notifications import TelegramNotifier
from .account_clients import AccountClientProtocol
from services.telemetry import ResiliencePolicy, Telemetry

logger = logging.getLogger(__name__)


class ResilientExecutor:
    """Adapter that wraps ``Telemetry.execute_with_resilience`` for DI."""

    def __init__(self, telemetry: Telemetry, policy: Optional[ResiliencePolicy] = None) -> None:
        self.telemetry = telemetry
        self.policy = policy or telemetry.policy

    async def execute(self, name: str, func: Callable[[], Any]) -> Any:
        async def _invoke() -> Any:
            result = func()
            if asyncio.iscoroutine(result):
                return await result
            return result

        return await self.telemetry.execute_with_resilience(name, _invoke, policy=self.policy)

    async def execute_threaded(self, name: str, func: Callable[[], Any]) -> Any:
        return await self.execute(name, lambda: asyncio.to_thread(func))


class ClientOrchestrator:
    """Fetch snapshots from clients while handling resilience concerns."""

    def __init__(
        self,
        clients: Sequence[AccountClientProtocol],
        executor: ResilientExecutor,
        *,
        account_messages: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._clients = list(clients)
        self._executor = executor
        self._last_auth_errors: dict[str, str] = {}
        self._account_messages = dict(account_messages or {})

    async def fetch(self) -> tuple[list[dict[str, Any]], dict[str, str]]:
        tasks = [
            self._executor.execute(
                f"account:{client.config.name}:fetch", lambda client=client: client.fetch()
            )
            for client in self._clients
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        accounts_payload: list[dict[str, Any]] = []
        account_messages: dict[str, str] = dict(self._account_messages)
        for client, result in zip(self._clients, results):
            if isinstance(result, Exception):
                message = self._format_error_message(client.config.name, result)
                account_messages[client.config.name] = message
                accounts_payload.append(
                    {
                        "name": client.config.name,
                        "balance": 0.0,
                        "positions": [],
                        "exchange": client.config.exchange,
                    }
                )
            else:
                payload = dict(result)
                payload.setdefault("exchange", client.config.exchange)
                accounts_payload.append(payload)
                self._reset_auth_error(client.config.name)

        return accounts_payload, account_messages

    def _format_error_message(self, account_name: str, exc: Exception) -> str:
        auth_error_names = {"AuthenticationError"}
        try:
            from ccxt.base.errors import AuthenticationError as CCXTAuthenticationError
        except Exception:  # pragma: no cover - optional dependency
            CCXTAuthenticationError = None

        is_auth_error = exc.__class__.__name__ in auth_error_names
        if CCXTAuthenticationError is not None and isinstance(exc, CCXTAuthenticationError):
            is_auth_error = True

        if is_auth_error:
            message = f"{account_name}: authentication failed - {exc}"
            error_message = str(exc)
            previous_error = self._last_auth_errors.get(account_name)
            if previous_error != error_message:
                logger.warning("Authentication failed for %s: %s", account_name, exc)
                self._last_auth_errors[account_name] = error_message
            else:
                logger.debug("Authentication failure for %s unchanged: %s", account_name, exc)
            return message

        logger.error(
            "Failed to fetch snapshot for %s", account_name, exc_info=(type(exc), exc, exc.__traceback__)
        )
        return f"{account_name}: {exc}"

    def _reset_auth_error(self, account_name: str) -> None:
        if account_name in self._last_auth_errors:
            logger.info("Authentication for %s restored", account_name)
            self._last_auth_errors.pop(account_name, None)


class SnapshotPolicyEvaluator:
    """Evaluate alert policies for a snapshot."""

    def __call__(self, snapshot: Mapping[str, Any]) -> Sequence[RiskViolation]:
        try:
            _, accounts, thresholds, _ = parse_snapshot(dict(snapshot))
            alerts = evaluate_alerts(accounts, thresholds)
        except Exception:  # pragma: no cover - defensive fall back
            return []
        return [RiskViolation("alert_threshold", alert) for alert in alerts]


class NotificationDispatcher:
    """Send notifications while deduplicating alerts and scheduling snapshots."""

    def __init__(
        self,
        executor: ResilientExecutor,
        *,
        email_sender: Optional[EmailAlertSender] = None,
        email_recipients: Optional[Sequence[str]] = None,
        telegram_notifier: Optional[TelegramNotifier] = None,
        telegram_targets: Optional[Sequence[tuple[str, str]]] = None,
        timezone_name: str = "America/New_York",
    ) -> None:
        self._executor = executor
        self._email_sender = email_sender
        self._email_recipients = list(email_recipients or [])
        self._telegram_notifier = telegram_notifier
        self._telegram_targets = list(telegram_targets or [])
        self._active_alerts: set[str] = set()
        self._tz = ZoneInfo(timezone_name)
        self._daily_snapshot_sent_date: Optional[date] = None

    async def dispatch(self, violations: Sequence[RiskViolation], snapshot: Mapping[str, Any]) -> None:
        await self._maybe_send_daily_snapshot(snapshot)
        await self._send_alerts(violations, snapshot)

    async def _send_alerts(self, violations: Sequence[RiskViolation], snapshot: Mapping[str, Any]) -> None:
        if not (self._email_sender or self._telegram_notifier):
            return
        alerts = [violation.message for violation in violations]
        alerts_set = set(alerts)
        new_alerts = [alert for alert in alerts if alert not in self._active_alerts]
        self._active_alerts = alerts_set
        if not new_alerts:
            return
        generated_at = snapshot.get("generated_at")
        timestamp = generated_at if isinstance(generated_at, str) else datetime.now(timezone.utc).isoformat()
        lines = [f"Exposure thresholds were exceeded at {timestamp}.", "", "Alerts:"]
        lines.extend(f"- {alert}" for alert in new_alerts)
        body = "\n".join(lines)
        subject = "Risk alert: exposure threshold breached"
        if self._email_sender and self._email_recipients:
            await self._executor.execute_threaded(
                "notification:email", lambda: self._email_sender.send(subject, body, self._email_recipients)
            )
        if self._telegram_notifier and self._telegram_targets:
            message = f"Exposure alert at {timestamp}\n" + "\n".join(new_alerts)
            await asyncio.gather(
                *[
                    self._executor.execute_threaded(
                        f"notification:telegram:{chat_id}",
                        lambda token=token, chat_id=chat_id: self._telegram_notifier.send(token, chat_id, message),
                    )
                    for token, chat_id in self._telegram_targets
                ]
            )

    async def _maybe_send_daily_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        if not self._email_sender or not self._email_recipients:
            return
        now_ny = datetime.now(self._tz)
        current_date = now_ny.date()
        if self._daily_snapshot_sent_date and current_date > self._daily_snapshot_sent_date:
            self._daily_snapshot_sent_date = None
        if now_ny.time() < time(16, 0):
            return
        if self._daily_snapshot_sent_date == current_date:
            return

        accounts = snapshot.get("accounts", [])
        portfolio_balance = sum(float(account.get("balance", 0.0)) for account in accounts if isinstance(account, Mapping))
        lines = [
            f"Daily portfolio snapshot ({now_ny.strftime('%Y-%m-%d')} 16:00 ET)",
            f"Total balance: ${portfolio_balance:,.2f}",
            "",
            "Accounts:",
        ]
        for account in accounts or []:
            if not isinstance(account, Mapping):
                continue
            name = str(account.get("name", "unknown"))
            balance = float(account.get("balance", 0.0))
            realised = float(account.get("daily_realized_pnl", 0.0))
            lines.append(f"- {name}: balance ${balance:,.2f}, daily realised PnL ${realised:,.2f}")
        body = "\n".join(lines)
        subject = "Daily portfolio balance snapshot"
        await self._executor.execute_threaded(
            "notification:email.daily", lambda: self._email_sender.send(subject, body, self._email_recipients)
        )
        self._daily_snapshot_sent_date = current_date


class KillSwitchExecutor:
    """Execute kill-switch actions across account clients."""

    def __init__(self, clients: Sequence[AccountClientProtocol], executor: ResilientExecutor) -> None:
        self._clients = list(clients)
        self._executor = executor

    async def execute(self, account_name: Optional[str] = None, symbol: Optional[str] = None) -> dict[str, Any]:
        scope = account_name or "all accounts"
        symbol_desc = f" for {symbol}" if symbol else ""
        logger.info("Kill switch requested for %s%s", scope, symbol_desc)
        targets: list[AccountClientProtocol] = []
        for client in self._clients:
            if account_name is None or client.config.name == account_name:
                targets.append(client)
        if account_name is not None and not targets:
            raise ValueError(f"Account '{account_name}' is not configured for realtime monitoring.")
        results: dict[str, Any] = {}
        for client in targets:
            try:
                results[client.config.name] = await self._executor.execute(
                    f"account:{client.config.name}:kill_switch", lambda client=client: client.kill_switch(symbol)
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Kill switch failed for %s", client.config.name, exc_info=True)
                results[client.config.name] = {"error": str(exc)}
        logger.info("Kill switch completed for %s", scope)
        return results


async def _noop_async(*_: Any, **__: Any) -> None:  # pragma: no cover - helper for defaults
    return None


