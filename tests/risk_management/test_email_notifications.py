from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path
import sys
from typing import List, Optional, Sequence

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from risk_management.configuration import EmailSettings
from risk_management.email_notifications import EmailAlertSender


class StubSMTP:
    def __init__(self, host: str, port: int, timeout: int) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.started_tls = False
        self.login_calls: List[tuple[str, str]] = []
        self.messages: List[tuple[EmailMessage, Optional[str], Sequence[str]]] = []

    def __enter__(self) -> "StubSMTP":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - context manager protocol
        return None

    def starttls(self) -> None:
        self.started_tls = True

    def login(self, username: str, password: str) -> None:
        self.login_calls.append((username, password))

    def send_message(
        self, message: EmailMessage, *, from_addr: str | None, to_addrs: Sequence[str]
    ) -> None:
        self.messages.append((message, from_addr, tuple(to_addrs)))


class StubSMTPSSL(StubSMTP):
    pass


def test_tls_delivery_uses_sender_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    smtp_instance: StubSMTP | None = None

    def factory(host: str, port: int, timeout: int = 10) -> StubSMTP:
        nonlocal smtp_instance
        smtp_instance = StubSMTP(host, port, timeout)
        return smtp_instance

    monkeypatch.setattr("smtplib.SMTP", factory)

    settings = EmailSettings(
        host="smtp.local",
        port=2525,
        username="alerts@example.com",
        password="super-secret",
        use_tls=True,
        use_ssl=False,
        sender="risk@example.com",
    )

    sender = EmailAlertSender(settings)
    sender.send("Subject", "Body", ["ops@example.com", ""])

    assert smtp_instance is not None
    assert smtp_instance.host == "smtp.local"
    assert smtp_instance.port == 2525
    assert smtp_instance.started_tls is True
    assert smtp_instance.login_calls == [("alerts@example.com", "super-secret")]
    assert len(smtp_instance.messages) == 1
    message, from_addr, to_addrs = smtp_instance.messages[0]
    assert message["From"] == "risk@example.com"
    assert from_addr == "risk@example.com"
    assert message["To"] == "ops@example.com"
    assert tuple(addr for addr in to_addrs if addr) == ("ops@example.com",)


def test_ssl_delivery_skips_tls_and_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    smtp_instance: StubSMTPSSL | None = None

    def factory(host: str, port: int, timeout: int = 10) -> StubSMTPSSL:
        nonlocal smtp_instance
        smtp_instance = StubSMTPSSL(host, port, timeout)
        return smtp_instance

    monkeypatch.setattr("smtplib.SMTP_SSL", factory)

    settings = EmailSettings(
        host="smtp.local",
        port=465,
        use_tls=False,
        use_ssl=True,
    )

    sender = EmailAlertSender(settings)
    sender.send("Subject", "Body", ["dev@example.com"])

    assert smtp_instance is not None
    assert smtp_instance.started_tls is False
    assert smtp_instance.login_calls == []
    assert smtp_instance.messages[0][1] == "alerts@localhost"


@pytest.mark.parametrize(
    "settings_kwargs, expected_sender",
    [
        ({"sender": "configured@example.com"}, "configured@example.com"),
        ({"sender": None, "username": "user@example.com"}, "user@example.com"),
        ({"sender": None, "username": None}, "alerts@localhost"),
    ],
)
def test_sender_fallbacks(monkeypatch: pytest.MonkeyPatch, settings_kwargs, expected_sender) -> None:
    smtp_instance: StubSMTP | None = None

    def factory(host: str, port: int, timeout: int = 10) -> StubSMTP:
        nonlocal smtp_instance
        smtp_instance = StubSMTP(host, port, timeout)
        return smtp_instance

    monkeypatch.setattr("smtplib.SMTP", factory)

    settings = EmailSettings(host="smtp.local", port=2525, **settings_kwargs)
    sender = EmailAlertSender(settings)
    sender.send("Alert", "Body", ["team@example.com"])

    assert smtp_instance is not None
    message, from_addr, to_addrs = smtp_instance.messages[0]
    assert message["From"] == expected_sender
    assert from_addr == expected_sender
    assert to_addrs == ("team@example.com",)
