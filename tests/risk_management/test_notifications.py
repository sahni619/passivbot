from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from risk_management._notifications import NotificationCoordinator
from risk_management.configuration import (
    AccountConfig,
    EmailSettings,
    PolicyActionConfig,
    PolicyConfig,
    PolicyTriggerConfig,
    RealtimeConfig,
)
from risk_management.policies import PolicyActionState, PolicyEvaluation, PolicyEvaluationResult


class RecordingSender:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.sent: List[Tuple[str, str, Sequence[str]]] = []

    def send(self, subject: str, body: str, recipients: Sequence[str]) -> None:
        self.sent.append((subject, body, tuple(recipients)))


class RecordingTelegram:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.sent: List[Tuple[str, str, str]] = []

    def send(self, token: str, chat_id: str, message: str) -> None:
        self.sent.append((token, chat_id, message))


class RecordingAudit:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def log(self, *, action: str, actor: str, details: Dict[str, Any]) -> str:
        record = {"action": action, "actor": actor, "details": dict(details)}
        self.records.append(record)
        return "hash"


def _make_config(channels: Sequence[str]) -> RealtimeConfig:
    email_settings = EmailSettings(host="smtp.local", sender="alerts@example.com")
    return RealtimeConfig(
        accounts=[AccountConfig(name="primary", exchange="binance")],
        alert_thresholds={},
        notification_channels=list(channels),
        email=email_settings,
    )


def test_channel_extraction_and_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    email_sender = RecordingSender()
    telegram_notifier = RecordingTelegram()

    monkeypatch.setattr("risk_management._notifications.EmailAlertSender", lambda *_: email_sender)
    monkeypatch.setattr("risk_management._notifications.TelegramNotifier", lambda *_: telegram_notifier)

    config = _make_config(
        [
            "email:alerts@example.com",
            "EMAIL: secondary@example.com",
            "telegram: token@chat",
            "telegram:token/chat",
            "telegram:token:room",
            "slack:#alerts",
            42,
        ]
    )

    coordinator = NotificationCoordinator(config)

    assert coordinator.email_recipients == (
        "alerts@example.com",
        "secondary@example.com",
    )
    assert coordinator.telegram_targets == (
        ("token", "chat"),
        ("token", "chat"),
        ("token", "room"),
    )


def _patch_alert_calculations(monkeypatch: pytest.MonkeyPatch, alerts: Sequence[str]) -> None:
    generated_at = datetime(2024, 5, 1, 12, 30, tzinfo=timezone.utc)

    def fake_parse(snapshot: Dict[str, Any]) -> tuple[Any, Sequence[Any], Any, Sequence[str]]:
        return generated_at, [], object(), []

    monkeypatch.setattr("risk_management._notifications.parse_snapshot", fake_parse)
    monkeypatch.setattr(
        "risk_management._notifications.evaluate_alerts",
        lambda accounts, thresholds: list(alerts),
    )


def test_dispatch_alerts_deduplicates_and_audits(monkeypatch: pytest.MonkeyPatch) -> None:
    email_sender = RecordingSender()
    telegram_notifier = RecordingTelegram()
    audit = RecordingAudit()

    monkeypatch.setattr("risk_management._notifications.EmailAlertSender", lambda *_: email_sender)
    monkeypatch.setattr("risk_management._notifications.TelegramNotifier", lambda *_: telegram_notifier)

    _patch_alert_calculations(monkeypatch, alerts=["breach"])

    config = _make_config(["email:ops@example.com", "telegram:token@chat"])
    coordinator = NotificationCoordinator(config, audit_logger=audit)

    snapshot = {
        "generated_at": "2024-05-01T12:30:00+00:00",
        "accounts": [
            {
                "name": "Alpha",
                "metadata": {
                    "limit_breaches": {
                        "venue_concentration_pct": {"breached": True, "value": 0.4, "limit": 0.25}
                    }
                },
            }
        ],
    }

    coordinator.dispatch_alerts(snapshot)

    assert len(email_sender.sent) == 1
    assert "Exposure thresholds were exceeded" in email_sender.sent[0][1]
    assert email_sender.sent[0][2] == ("ops@example.com",)
    assert len(telegram_notifier.sent) == 1
    assert audit.records[0]["action"] == "notification.email.alert"
    assert audit.records[1]["action"] == "notification.telegram.alert"

    coordinator.dispatch_alerts(snapshot)

    assert len(email_sender.sent) == 1
    assert len(telegram_notifier.sent) == 1
    assert len(audit.records) == 2


def test_audit_failures_are_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    class FailingAudit:
        def log(self, *, action: str, actor: str, details: Dict[str, Any]) -> str:
            raise RuntimeError("boom")

    email_sender = RecordingSender()

    monkeypatch.setattr("risk_management._notifications.EmailAlertSender", lambda *_: email_sender)
    monkeypatch.setattr("risk_management._notifications.TelegramNotifier", lambda *_: RecordingTelegram())

    _patch_alert_calculations(monkeypatch, alerts=["breach"])

    config = _make_config(["email:ops@example.com"])
    coordinator = NotificationCoordinator(config, audit_logger=FailingAudit())

    snapshot = {
        "generated_at": "2024-05-01T12:30:00+00:00",
        "accounts": [
            {
                "name": "Alpha",
                "limit_breaches": {
                    "asset_concentration_pct": {"breached": True, "value": 0.3, "limit": 0.25}
                },
            }
        ],
    }

    with caplog.at_level("WARNING"):
        coordinator.dispatch_alerts(snapshot)

    assert any("Failed to emit notification audit entry" in record.message for record in caplog.records)
    assert len(email_sender.sent) == 1


def test_policy_notifications_route_channels(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    email_sender = RecordingSender()
    telegram_notifier = RecordingTelegram()

    monkeypatch.setattr("risk_management._notifications.EmailAlertSender", lambda *_: email_sender)
    monkeypatch.setattr("risk_management._notifications.TelegramNotifier", lambda *_: telegram_notifier)

    config = _make_config(["email:ops@example.com", "telegram:token@chat"])
    coordinator = NotificationCoordinator(config)

    action_config = PolicyActionConfig(
        type="notify",
        message="Leverage exceeded",
        channels=["telegram", "email", "pagerduty"],
        severity="warning",
        subject="Policy triggered",
    )
    confirmation_config = PolicyActionConfig(
        type="require_confirmation",
        message="Manual approval required",
        channels=["email"],
        severity="error",
        requires_confirmation=True,
        confirmation_key="abc123",
    )

    evaluation = PolicyEvaluation(
        config=PolicyConfig(
            name="Leverage", trigger=PolicyTriggerConfig(type="metric", metric="x", operator=">", value=1.0)
        ),
        triggered=True,
        trigger_value=1.2,
        actions=[
            PolicyActionState(
                policy_name="Leverage",
                config=action_config,
                status="triggered",
                trigger_value=1.2,
                threshold=1.0,
                rendered_message="Leverage exceeded",
            ),
            PolicyActionState(
                policy_name="Leverage",
                config=confirmation_config,
                status="triggered",
                trigger_value=1.2,
                threshold=1.0,
                rendered_message=None,
            ),
        ],
    )

    result = PolicyEvaluationResult([evaluation])

    with caplog.at_level("INFO"):
        coordinator.handle_policy_evaluations(result)

    assert len(email_sender.sent) == 2
    assert email_sender.sent[0][0] == "Policy triggered"
    assert "Confirmation key: abc123" in email_sender.sent[1][1]
    assert len(telegram_notifier.sent) == 1
    assert telegram_notifier.sent[0][2].startswith("Leverage exceeded")
    assert any("requested unsupported notification channel" in record.message for record in caplog.records)
    assert any("awaiting confirmation" in record.message for record in caplog.records)
