"""Side-effectful execution of risk decisions."""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable

from risk_management._notifications import NotificationCoordinator
from risk_management.account_clients import AccountClientProtocol

from .config import RiskEngineConfig
from .metrics import MetricRegistry
from .portfolio_aggregator import PortfolioView
from .risk_rules import RiskDecision, RiskDecisionLevel
from .state_store import StateStore

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Perform actions with safety rails and optional dry-run."""

    def __init__(
        self,
        config: RiskEngineConfig,
        *,
        state_store: StateStore,
        notification_coordinator: NotificationCoordinator,
        account_clients: Iterable[AccountClientProtocol],
        metrics: MetricRegistry | None = None,
    ) -> None:
        self._config = config
        self._state_store = state_store
        self._notifications = notification_coordinator
        self._account_clients = list(account_clients)
        self._metrics = metrics or MetricRegistry()
        self._failure_counts = {}

    async def execute(self, decision: RiskDecision, view: PortfolioView) -> None:
        logger.info(
            "Executing risk decision",
            extra={"decision": decision.level.value, "drawdown": decision.drawdown, "rationale": decision.rationale},
        )
        if decision.level is RiskDecisionLevel.NONE:
            return
        if decision.level is RiskDecisionLevel.ALERT:
            self._emit(decision)
            return
        if decision.level is RiskDecisionLevel.CLOSE_POSITIONS:
            await self._maybe_close_positions(decision, view)
            return
        if decision.level is RiskDecisionLevel.KILL_BOTS:
            await self._maybe_kill_bots(decision)
            return

    def _emit(self, decision: RiskDecision) -> None:
        subject = f"Risk alert: drawdown {decision.drawdown:.2%}"
        body = decision.rationale
        self._notifications.send_risk_signal(subject=subject, body=body, severity="warning")

    async def _maybe_close_positions(self, decision: RiskDecision, view: PortfolioView) -> None:
        action_key = f"close:{decision.breach_id}"
        cooldown = self._config.actions.close_cooldown_seconds
        if not self._state_store.should_execute(action_key, cooldown):
            logger.info(
                "Close positions already executed recently",
                extra={"breach": decision.breach_id, "cooldown_seconds": cooldown},
            )
            return
        if self._config.actions.dry_run:
            logger.warning(
                "[DRY-RUN] Would close positions",
                extra={"breach": decision.breach_id, "rationale": decision.rationale},
            )
            self._emit(decision)
            return
        if self._config.actions.max_close_notional is not None:
            total_notional = sum(position.notional for account in view.accounts for position in account.positions)
            if total_notional > self._config.actions.max_close_notional:
                logger.error(
                    "Aborting close positions due to notional limit",
                    extra={
                        "breach": decision.breach_id,
                        "limit": self._config.actions.max_close_notional,
                        "total_notional": total_notional,
                    },
                )
                return
            logger.info(
                "Within max close notional", extra={"total_notional": total_notional, "limit": self._config.actions.max_close_notional}
            )
        tasks = []
        for client in self._account_clients:
            client_name = self._client_name(client)
            logger.info(
                "Closing positions",
                extra={"exchange": client_name, "rationale": decision.rationale, "breach": decision.breach_id},
            )
            tasks.append(self._close_with_backoff(client, action_key))
        if tasks:
            await asyncio.gather(*tasks)
        self._state_store.record_action(action_key)
        self._emit(decision)

    async def _maybe_kill_bots(self, decision: RiskDecision) -> None:
        action_key = f"kill:{decision.breach_id}"
        cooldown = self._config.actions.kill_cooldown_seconds
        if not self._state_store.should_execute(action_key, cooldown):
            logger.info("Kill switch already executed recently for breach %s", decision.breach_id)
            return
        if self._config.actions.dry_run:
            logger.warning("[DRY-RUN] Would trigger kill switch due to %s", decision.rationale)
            self._emit(decision)
            return
        tasks = []
        for client in self._account_clients:
            client_name = self._client_name(client)
            logger.critical("Executing kill switch for %s: %s", client_name, decision.rationale)
            tasks.append(client.kill_switch())
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as exc:  # pragma: no cover
                logger.error("Error while executing kill switch: %s", exc, exc_info=True)
        self._state_store.record_action(action_key)
        self._emit(decision)

    async def _close_with_backoff(self, client: AccountClientProtocol, action_key: str) -> None:
        attempts = self._config.actions.close_retry_attempts
        client_name = self._client_name(client)
        for attempt in range(1, attempts + 1):
            try:
                await client.close_all_positions()
                self._failure_counts.pop(client_name, None)
                return
            except Exception as exc:  # pragma: no cover - defensive
                failures = self._failure_counts.get(client_name, 0) + 1
                self._failure_counts[client_name] = failures
                backoff = min(
                    self._config.actions.backoff_seconds * failures,
                    self._config.actions.max_backoff_seconds,
                )
                self._metrics.inc(
                    "exchange_api_errors_total",
                    labels={"exchange": client_name, "op": "close_all_positions", "code": getattr(exc, "code", "unknown")},
                )
                logger.error(
                    "Close positions failed",
                    extra={
                        "exchange": client_name,
                        "failure_count": failures,
                        "backoff_seconds": backoff,
                        "action": action_key,
                        "error": str(exc),
                        "attempt": attempt,
                        "max_attempts": attempts,
                    },
                    exc_info=True,
                )
                if attempt >= attempts:
                    logger.error(
                        "Exhausted close position retries",
                        extra={"exchange": client_name, "action": action_key, "attempts": attempts},
                    )
                    return
                await asyncio.sleep(backoff)

    def _client_name(self, client: AccountClientProtocol) -> str:
        cfg = getattr(client, "config", None)
        if cfg and getattr(cfg, "name", None):
            return str(cfg.name)
        return getattr(client, "name", "unknown")
