import asyncio
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from risk_management.risk_engine.action_executor import ActionExecutor
import asyncio
from types import SimpleNamespace
from typing import Any, Dict, Optional

from risk_management.risk_engine.action_executor import ActionExecutor
from risk_management.risk_engine.config import ActionConfig, RiskEngineConfig
from risk_management.risk_engine.portfolio_aggregator import AccountBreakdown, PortfolioView
from risk_management.risk_engine.risk_rules import RiskDecision, RiskDecisionLevel
from risk_management.risk_engine.state_store import StateStore, RiskState


class InMemoryStore(StateStore):
    def __init__(self):
        self.actions: Dict[str, float] = {}

    def load(self) -> RiskState:
        return RiskState(baseline_equity=None, high_water=None, actions=self.actions)

    def save(self, *, baseline_equity: Optional[float], high_water: Optional[float]) -> None:
        return None

    def record_action(self, action_key: str) -> None:
        self.actions[action_key] = 0

    def should_execute(self, action_key: str, cooldown_seconds: int) -> bool:
        return action_key not in self.actions

    def breach_id(self, drawdown: float, high_water: float) -> str:
        return f"{drawdown:.2f}:{high_water:.2f}"


class DummyNotifier:
    def __init__(self):
        self.messages = []

    def send_risk_signal(self, *, subject: str, body: str, severity: str = "info") -> None:
        self.messages.append({"subject": subject, "body": body, "severity": severity})


class DummyClient:
    def __init__(self, name: str):
        self.config = SimpleNamespace(name=name)
        self.closed = 0
        self.killed = 0

    async def close_all_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        self.closed += 1
        return {"symbol": symbol}

    async def kill_switch(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        self.killed += 1
        return {"symbol": symbol}


def test_alert_emits_notification_only():
    notifier = DummyNotifier()
    store = InMemoryStore()
    config = RiskEngineConfig(actions=ActionConfig(dry_run=False))
    executor = ActionExecutor(config, state_store=store, notification_coordinator=notifier, account_clients=[])
    decision = RiskDecision(
        level=RiskDecisionLevel.ALERT,
        drawdown=0.05,
        adjusted_equity=900,
        high_water=1000,
        rationale="alert",
        breach_id="id1",
    )

    asyncio.run(executor.execute(decision, PortfolioView(0, 0, 0, 0, 0)))

    assert notifier.messages and notifier.messages[0]["severity"] == "warning"


def test_close_positions_respects_dry_run_and_cooldown():
    notifier = DummyNotifier()
    store = InMemoryStore()
    config = RiskEngineConfig(actions=ActionConfig(dry_run=True, close_cooldown_seconds=999))
    client = DummyClient("binance")
    executor = ActionExecutor(config, state_store=store, notification_coordinator=notifier, account_clients=[client])
    decision = RiskDecision(
        level=RiskDecisionLevel.CLOSE_POSITIONS,
        drawdown=0.1,
        adjusted_equity=800,
        high_water=1000,
        rationale="close",
        breach_id="breach",
    )

    asyncio.run(executor.execute(decision, PortfolioView(0, 0, 0, 0, 0)))
    asyncio.run(executor.execute(decision, PortfolioView(0, 0, 0, 0, 0)))

    assert client.closed == 0  # dry-run prevents execution
    assert len(notifier.messages) == 2  # both executions emit signal


def test_close_positions_executes_once_per_breach():
    notifier = DummyNotifier()
    store = InMemoryStore()
    config = RiskEngineConfig(actions=ActionConfig(dry_run=False, close_cooldown_seconds=999))
    client = DummyClient("bybit")
    executor = ActionExecutor(config, state_store=store, notification_coordinator=notifier, account_clients=[client])
    decision = RiskDecision(
        level=RiskDecisionLevel.CLOSE_POSITIONS,
        drawdown=0.2,
        adjusted_equity=500,
        high_water=1000,
        rationale="breach",
        breach_id="breach-1",
    )

    asyncio.run(executor.execute(decision, PortfolioView(0, 0, 0, 0, 0)))
    asyncio.run(executor.execute(decision, PortfolioView(0, 0, 0, 0, 0)))

    assert client.closed == 1
    assert any(msg["severity"] == "warning" for msg in notifier.messages)


def test_kill_bots_respects_cooldown():
    notifier = DummyNotifier()
    store = InMemoryStore()
    config = RiskEngineConfig(actions=ActionConfig(dry_run=False, kill_cooldown_seconds=999))
    client = DummyClient("okx")
    executor = ActionExecutor(config, state_store=store, notification_coordinator=notifier, account_clients=[client])
    decision = RiskDecision(
        level=RiskDecisionLevel.KILL_BOTS,
        drawdown=0.3,
        adjusted_equity=400,
        high_water=1000,
        rationale="kill",
        breach_id="breach-2",
    )

    asyncio.run(executor.execute(decision, PortfolioView(0, 0, 0, 0, 0)))
    asyncio.run(executor.execute(decision, PortfolioView(0, 0, 0, 0, 0)))

    assert client.killed == 1
    assert notifier.messages
