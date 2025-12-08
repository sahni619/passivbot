from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from risk_engine.policies import RiskViolation

from risk_management.configuration import AccountConfig
from risk_management.realtime import AuthenticationError
from risk_management.realtime_components import (
    ClientOrchestrator,
    KillSwitchExecutor,
    NotificationDispatcher,
    ResilientExecutor,
    SnapshotPolicyEvaluator,
)
from services.telemetry import ResiliencePolicy, Telemetry


class FakeEmailSender:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, tuple[str, ...]]] = []

    def send(self, subject: str, body: str, recipients: list[str]) -> None:
        self.sent.append((subject, body, tuple(recipients)))


class FakeTelegramNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send(self, token: str, chat_id: str, message: str) -> None:
        self.sent.append((token, chat_id, message))


class OutcomeClient:
    def __init__(self, config: AccountConfig, outcomes: list[Any]) -> None:
        self.config = config
        self._outcomes = list(outcomes)
        self.calls = 0
        self.kill_calls = 0

    async def fetch(self) -> Any:
        self.calls += 1
        if not self._outcomes:
            raise RuntimeError("no more outcomes configured")
        result = self._outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def kill_switch(self, symbol: str | None = None) -> dict[str, Any]:
        self.kill_calls += 1
        return {"symbol": symbol or "*"}


def test_resilient_executor_retries_and_succeeds() -> None:
    telemetry = Telemetry(policy=ResiliencePolicy(max_retries=1, retry_backoff=0))
    executor = ResilientExecutor(telemetry)

    attempts = 0

    async def flaky_call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient")
        return "ok"

    result = asyncio.run(executor.execute("flaky", flaky_call))

    assert result == "ok"
    assert attempts == 2


def test_client_orchestrator_handles_authentication_errors(caplog: pytest.LogCaptureFixture) -> None:
    config = AccountConfig(name="Auth", exchange="binance")
    client = OutcomeClient(config, [AuthenticationError("invalid"), AuthenticationError("invalid")])
    executor = ResilientExecutor(Telemetry(policy=ResiliencePolicy(max_retries=0)))
    orchestrator = ClientOrchestrator([client], executor)

    caplog.set_level("WARNING")

    accounts, messages = asyncio.run(orchestrator.fetch())
    _, messages_second = asyncio.run(orchestrator.fetch())

    assert messages["Auth"].startswith("Auth: authentication failed")
    assert messages_second["Auth"].startswith("Auth: authentication failed")
    assert accounts[0]["balance"] == 0
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1


def test_snapshot_policy_evaluator_emits_policy_violations() -> None:
    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "accounts": [
            {
                "name": "Test",
                "balance": 1000.0,
                "positions": [
                    {
                        "symbol": "BTC",
                        "side": "long",
                        "wallet_exposure_pct": 0.6,
                        "notional": 600,
                        "entry_price": 1.0,
                        "mark_price": 1.0,
                        "unrealized_pnl": 0.0,
                        "max_drawdown_pct": 0.0,
                    }
                ],
            }
        ],
        "alert_thresholds": {
            "wallet_exposure_pct": 0.5,
            "position_wallet_exposure_pct": 0.3,
            "max_drawdown_pct": 0.4,
            "loss_threshold_pct": -0.2,
        },
        "notification_channels": [],
    }

    evaluator = SnapshotPolicyEvaluator()
    violations = evaluator(snapshot)

    assert violations
    assert violations[0].policy == "alert_threshold"


def test_notification_dispatcher_deduplicates_and_schedules(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = Telemetry(policy=ResiliencePolicy(max_retries=0, retry_backoff=0))
    executor = ResilientExecutor(telemetry)
    email = FakeEmailSender()
    telegram = FakeTelegramNotifier()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return datetime(2024, 1, 1, 17, 0, tzinfo=tz or timezone.utc)

    monkeypatch.setattr("risk_management.realtime_components.datetime", FixedDateTime)

    dispatcher = NotificationDispatcher(
        executor,
        email_sender=email,
        email_recipients=["alert@example.com"],
        telegram_notifier=telegram,
        telegram_targets=[("token", "chat")],
        timezone_name="UTC",
    )
    violations = [RiskViolation("breach", "alert one")]
    snapshot = {
        "generated_at": FixedDateTime.now(timezone.utc).isoformat(),
        "accounts": [{"name": "A", "balance": 50.0, "positions": []}],
    }

    asyncio.run(dispatcher.dispatch(violations, snapshot))
    asyncio.run(dispatcher.dispatch(violations, snapshot))

    assert len(email.sent) == 2  # daily snapshot + initial alert
    assert len(telegram.sent) == 1


def test_kill_switch_executor_invokes_clients() -> None:
    config = AccountConfig(name="Kill", exchange="binance")
    client = OutcomeClient(config, [ {"name": "Kill", "balance": 0, "positions": []} ])
    executor = ResilientExecutor(Telemetry())
    kill_executor = KillSwitchExecutor([client], executor)

    result = asyncio.run(kill_executor.execute(symbol="BTCUSDT"))

    assert result["Kill"]["symbol"] == "BTCUSDT"
    assert client.kill_calls == 1

