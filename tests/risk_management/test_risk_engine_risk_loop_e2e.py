import asyncio
import time
from dataclasses import dataclass

import pytest

from risk_management.risk_engine import (
    ActionExecutor,
    ExchangeClientAdapter,
    PortfolioAggregator,
    RiskDecisionLevel,
    RiskEngineConfig,
    RiskRulesEngine,
    risk_loop,
)
from risk_management.risk_engine.state_store import RiskState, StateStore


@dataclass
class _AccountConfig:
    name: str


class FakeNotificationCoordinator:
    def __init__(self) -> None:
        self.messages = []

    def send_risk_signal(self, *, subject: str, body: str, severity: str) -> None:
        self.messages.append({"subject": subject, "body": body, "severity": severity})


class FakeStateStore(StateStore):
    def __init__(self, baseline_equity=None, high_water=None) -> None:
        self._state = RiskState(baseline_equity=baseline_equity, high_water=high_water, actions={})

    def load(self) -> RiskState:
        return self._state

    def save(self, *, baseline_equity, high_water) -> None:
        if baseline_equity is not None:
            self._state.baseline_equity = baseline_equity
        if high_water is not None:
            self._state.high_water = high_water

    def record_action(self, action_key: str) -> None:
        self._state.actions[action_key] = time.time()

    def should_execute(self, action_key: str, cooldown_seconds: int) -> bool:
        last = self._state.actions.get(action_key)
        if last is None:
            return True
        return (time.time() - last) >= cooldown_seconds

    def breach_id(self, drawdown: float, high_water: float) -> str:
        return f"test|{drawdown:.4f}|{high_water:.2f}"


class FakeAccountClient:
    def __init__(self, name: str, balance: float, *, unrealized: float = 0.0, cashflows=None, fail_close=False):
        self.config = _AccountConfig(name=name)
        self.balance = balance
        self.unrealized = unrealized
        self.cashflows = cashflows or []
        self.fail_close = fail_close
        self.closes = 0

    async def fetch(self):
        return {
            "name": self.config.name,
            "balance": self.balance,
            "positions": [
                {"symbol": "BTCUSDT", "unrealized_pnl": self.unrealized, "signed_notional": 100}
            ],
            "realized_pnl": 0.0,
            "cashflow_events": list(self.cashflows),
        }

    async def close_all_positions(self, symbol=None):
        self.closes += 1
        if self.fail_close:
            raise Exception("-4061 position side does not match")
        return {"status": "closed"}

    async def kill_switch(self, symbol=None):
        return {}

    async def create_order(self, *args, **kwargs):  # pragma: no cover - unused
        return {}

    async def cancel_order(self, *args, **kwargs):  # pragma: no cover - unused
        return {}

    async def close_position(self, *args, **kwargs):  # pragma: no cover - unused
        return {}

    async def list_order_types(self):  # pragma: no cover - unused
        return []

    async def cancel_all_orders(self, *args, **kwargs):  # pragma: no cover - unused
        return {}

    async def close(self):  # pragma: no cover - unused
        return None


def _build_action_executor(config: RiskEngineConfig, store: StateStore, *clients):
    notifier = FakeNotificationCoordinator()
    return notifier, ActionExecutor(
        config,
        state_store=store,
        notification_coordinator=notifier,
        account_clients=clients,
    )


def test_drawdown_alert_triggers_notification_only():
    config = RiskEngineConfig()
    state_store = FakeStateStore(baseline_equity=100, high_water=100)
    client = FakeAccountClient("binance", balance=97)
    notifier, executor = _build_action_executor(config, state_store, client)
    aggregator = PortfolioAggregator()

    decision, view = asyncio.run(
        risk_loop(
            {"binance": ExchangeClientAdapter(client)},
            RiskRulesEngine(config),
            executor,
            state_store,
            config,
            aggregator=aggregator,
        )
    )

    assert decision.level is RiskDecisionLevel.ALERT
    assert notifier.messages, "Telegram alert should have been sent"
    assert client.closes == 0
    assert pytest.approx(view.total_equity, rel=1e-3) == 97


def test_close_positions_executes_for_all_clients():
    config = RiskEngineConfig()
    state_store = FakeStateStore(baseline_equity=200, high_water=200)
    client_a = FakeAccountClient("binance", balance=90)
    client_b = FakeAccountClient("okx", balance=92)
    notifier, executor = _build_action_executor(config, state_store, client_a, client_b)

    decision, _ = asyncio.run(
        risk_loop(
            {
                "binance": ExchangeClientAdapter(client_a),
                "okx": ExchangeClientAdapter(client_b),
            },
            RiskRulesEngine(config),
            executor,
            state_store,
            config,
        )
    )

    assert decision.level is RiskDecisionLevel.CLOSE_POSITIONS
    assert client_a.closes == 1
    assert client_b.closes == 1
    assert notifier.messages, "Alert should accompany close action"


def test_close_positions_logs_failure_and_continues(caplog):
    caplog.set_level("ERROR")
    config = RiskEngineConfig()
    state_store = FakeStateStore(baseline_equity=200, high_water=200)
    failing_client = FakeAccountClient("binance", balance=90, fail_close=True)
    healthy_client = FakeAccountClient("bybit", balance=91)
    notifier, executor = _build_action_executor(config, state_store, failing_client, healthy_client)

    decision, _ = asyncio.run(
        risk_loop(
            {
                "binance": ExchangeClientAdapter(failing_client),
                "bybit": ExchangeClientAdapter(healthy_client),
            },
            RiskRulesEngine(config),
            executor,
            state_store,
            config,
        )
    )

    assert decision.level is RiskDecisionLevel.CLOSE_POSITIONS
    assert healthy_client.closes == 1
    assert failing_client.closes >= config.actions.close_retry_attempts
    assert any("-4061" in message for message in caplog.text.splitlines())
    assert notifier.messages, "Alert should still be emitted"


def test_deposit_updates_baseline_without_triggering_breach():
    config = RiskEngineConfig()
    state_store = FakeStateStore()
    client = FakeAccountClient(
        "binance",
        balance=120,
        cashflows=[{"type": "deposit", "amount": 20}],
    )
    notifier, executor = _build_action_executor(config, state_store, client)
    aggregator = PortfolioAggregator()

    decision, view = asyncio.run(
        risk_loop(
            {"binance": ExchangeClientAdapter(client)},
            RiskRulesEngine(config),
            executor,
            state_store,
            config,
            aggregator=aggregator,
        )
    )

    assert decision.level is RiskDecisionLevel.NONE
    assert not notifier.messages
    assert pytest.approx(view.net_cashflow, rel=1e-6) == 20.0
    assert pytest.approx(state_store.load().high_water, rel=1e-6) == 100.0

