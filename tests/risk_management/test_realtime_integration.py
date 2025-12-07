from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from risk_engine.policies import RiskViolation

from risk_management.configuration import AccountConfig, RealtimeConfig
from risk_management.realtime import AuthenticationError, RealtimeDataFetcher
from risk_management.realtime_components import (
    ClientOrchestrator,
    NotificationDispatcher,
    ResilientExecutor,
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
    def __init__(self, config: AccountConfig, outcomes: list[object]) -> None:
        self.config = config
        self._outcomes = list(outcomes)
        self.calls = 0
        self.kill_calls = 0

    async def fetch(self):
        self.calls += 1
        if not self._outcomes:
            raise RuntimeError("no more outcomes")
        result = self._outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def kill_switch(self, symbol: str | None = None):
        self.kill_calls += 1
        return {"symbol": symbol or "*"}

    async def close(self):  # pragma: no cover - required for interface
        return None


def test_realtime_components_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = Telemetry(policy=ResiliencePolicy(max_retries=1, retry_backoff=0))
    executor = ResilientExecutor(telemetry)

    clients = [
        OutcomeClient(
            AccountConfig(name="Retry", exchange="binance"),
            [
                RuntimeError("flaky"),
                {"name": "Retry", "balance": 100.0, "positions": []},
                {"name": "Retry", "balance": 100.0, "positions": []},
            ],
        ),
        OutcomeClient(
            AccountConfig(name="Auth", exchange="okx"),
            [AuthenticationError("denied"), AuthenticationError("denied"), AuthenticationError("denied"), AuthenticationError("denied")],
        ),
        OutcomeClient(
            AccountConfig(name="Healthy", exchange="binance"),
            [
                {"name": "Healthy", "balance": 50.0, "positions": []},
                {"name": "Healthy", "balance": 50.0, "positions": []},
            ],
        ),
    ]
    config = RealtimeConfig(
        accounts=[client.config for client in clients],
        alert_thresholds={
            "wallet_exposure_pct": 0.1,
            "position_wallet_exposure_pct": 0.1,
            "max_drawdown_pct": 0.1,
            "loss_threshold_pct": -0.1,
        },
        notification_channels=["email:ops@example.com", "telegram:token/chat"],
    )

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return datetime(2024, 1, 1, 17, 0, tzinfo=tz or timezone.utc)

    monkeypatch.setattr("risk_management.realtime_components.datetime", FixedDateTime)

    email_sender = FakeEmailSender()
    telegram_notifier = FakeTelegramNotifier()
    dispatcher = NotificationDispatcher(
        executor,
        email_sender=email_sender,
        email_recipients=["ops@example.com"],
        telegram_notifier=telegram_notifier,
        telegram_targets=[("token", "chat")],
        timezone_name="UTC",
    )

    orchestrator = ClientOrchestrator(clients, executor, account_messages=config.account_messages)

    def policy_eval(snapshot: dict) -> list[RiskViolation]:
        return [RiskViolation("breach", "exceeded threshold")]

    fetcher = RealtimeDataFetcher(
        config,
        account_clients=clients,
        telemetry=telemetry,
        policy_evaluator=policy_eval,
        notification_dispatcher=dispatcher,
        orchestrator=orchestrator,
        executor=executor,
    )

    snapshot_first = asyncio.run(fetcher.fetch_snapshot())
    asyncio.run(fetcher.fetch_snapshot())

    assert snapshot_first["accounts"][0]["balance"] == 100.0
    assert "Auth" in snapshot_first["account_messages"]
    assert clients[0].calls >= 3  # retried once and succeeded again on next fetch
    assert len(email_sender.sent) == 2  # daily + alert
    assert len(telegram_notifier.sent) == 1

    assert len(email_sender.sent) == 2  # no duplicates on second run
    result = asyncio.run(fetcher.execute_kill_switch(account_name="Healthy", symbol="BTCUSDT"))
    assert result["Healthy"]["symbol"] == "BTCUSDT"
    assert clients[2].kill_calls == 1

