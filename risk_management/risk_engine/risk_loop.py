"""Top-level orchestration for the risk engine loop."""

from __future__ import annotations

import logging
import time
from typing import Dict, Mapping, MutableSequence, Optional, Tuple

from .action_executor import ActionExecutor
from .config import RiskEngineConfig
from .exchange_client import ExchangeClientAdapter
from .metrics import MetricRegistry, Timer
from .portfolio_aggregator import PortfolioAggregator, PortfolioView
from .risk_rules import RiskDecision, RiskRulesEngine
from .state_store import StateStore

logger = logging.getLogger(__name__)


async def risk_loop(
    clients: Dict[str, ExchangeClientAdapter],
    rules_engine: RiskRulesEngine,
    action_executor: ActionExecutor,
    state_store: StateStore,
    config: RiskEngineConfig,
    *,
    aggregator: Optional[PortfolioAggregator] = None,
    metrics: Optional[MetricRegistry] = None,
) -> Tuple[RiskDecision, PortfolioView]:
    """Run a single iteration of the risk pipeline.

    Steps:
    1. Fetch snapshots from each exchange client (dropping any that fail).
    2. Aggregate snapshots into a portfolio view while separating cashflows from PnL.
    3. Evaluate drawdown rules using the provided ``rules_engine`` and ``state_store``.
    4. Execute resulting actions via ``action_executor`` with its configured safety rails.
    5. Return the decision and aggregated portfolio view for observability.
    """

    aggregator = aggregator or PortfolioAggregator()
    metrics = metrics or MetricRegistry()
    account_payloads: MutableSequence[Mapping[str, object]] = []
    cashflow_events: MutableSequence[Mapping[str, object]] = []

    loop_start = time.perf_counter()
    for name, client in clients.items():
        try:
            with Timer(metrics, "exchange_api_latency_seconds", labels={"exchange": name, "op": "fetch_snapshot"}):
                snapshot = await client.fetch()
        except Exception as exc:  # pragma: no cover - defensive logging
            metrics.inc("exchange_api_errors_total", labels={"exchange": name, "op": "fetch_snapshot", "code": getattr(exc, "code", "unknown") or "unknown"})
            logger.error(
                "Failed to fetch snapshot",
                extra={"exchange": name, "error": str(exc), "op": "fetch_snapshot"},
                exc_info=True,
            )
            continue

        if not isinstance(snapshot, Mapping):
            metrics.inc(
                "exchange_api_errors_total",
                labels={"exchange": name, "op": "fetch_snapshot", "code": "malformed"},
            )
            logger.warning(
                "Ignoring malformed snapshot",
                extra={"exchange": name, "type": str(type(snapshot)), "op": "fetch_snapshot"},
            )
            continue

        account_payload = snapshot.get("account") if "account" in snapshot else snapshot
        if isinstance(account_payload, Mapping):
            account_payloads.append(account_payload)
        else:
            metrics.inc(
                "exchange_api_errors_total",
                labels={"exchange": name, "op": "fetch_snapshot", "code": "missing_account"},
            )
            logger.warning(
                "Snapshot missing account payload",
                extra={"exchange": name, "op": "fetch_snapshot"},
            )

        cashflows = snapshot.get("cashflow_events") or snapshot.get("cashflows") or []
        if isinstance(cashflows, list):
            cashflow_events.extend([cf for cf in cashflows if isinstance(cf, Mapping)])

    portfolio_view = aggregator.aggregate(account_payloads, cashflow_events)
    decision = rules_engine.evaluate(portfolio_view, state_store)
    metrics.inc("risk_actions_total", labels={"action": decision.level.value})
    await action_executor.execute(decision, portfolio_view)
    duration = time.perf_counter() - loop_start
    metrics.observe("risk_loop_latency_seconds", duration)
    logger.info(
        "Risk loop completed",
        extra={
            "decision": decision.level.value,
            "drawdown": round(decision.drawdown, 4),
            "equity": round(decision.adjusted_equity, 2),
            "duration": duration,
        },
    )
    return decision, portfolio_view
