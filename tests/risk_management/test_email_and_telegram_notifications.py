import types

import pytest

pytest.importorskip("pydantic")

from risk_management.configuration import EmailSettings
from risk_management.email_notifications import EmailAlertSender
from risk_management.telegram_notifications import TelegramNotifier


class DummySMTP:
    def __init__(self):
        self.started_tls = False
        self.login_calls: list[tuple[str, str]] = []
        self.sent_messages: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def starttls(self):
        self.started_tls = True

    def login(self, username, password):
        self.login_calls.append((username, password))

    def send_message(self, message, from_addr=None, to_addrs=None):
        self.sent_messages.append((message, from_addr, tuple(to_addrs or ())))


class DummyResponse:
    def __init__(self, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text


class DummyClient:
    def __init__(self, response: DummyResponse):
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json):
        self.calls.append((url, json))
        return self.response


@pytest.mark.parametrize("use_ssl", [True, False])
def test_email_sender_dispatches_with_auth(monkeypatch, use_ssl):
    sender = DummySMTP()

    if use_ssl:
        monkeypatch.setattr(
            "risk_management.email_notifications.smtplib.SMTP_SSL",
            lambda host, port, timeout=None: sender,
        )
    else:
        monkeypatch.setattr(
            "risk_management.email_notifications.smtplib.SMTP",
            lambda host, port, timeout=None: sender,
        )

    settings = EmailSettings(
        host="localhost",
        port=2525,
        username="alerts@example.com",
        password="secret",
        use_tls=not use_ssl,
        use_ssl=use_ssl,
        sender=None,
    )
    alert_sender = EmailAlertSender(settings)
    alert_sender.send("Subject", "Body", ["user@example.com", ""])

    if use_ssl:
        assert sender.started_tls is False
    else:
        assert sender.started_tls is True
    assert sender.login_calls == [("alerts@example.com", "secret")]
    assert sender.sent_messages
    message, from_addr, recipients = sender.sent_messages[0]
    assert message["Subject"] == "Subject"
    assert from_addr == "alerts@example.com"
    assert recipients == ("user@example.com",)


def test_email_sender_uses_explicit_sender(monkeypatch):
    sender = DummySMTP()
    monkeypatch.setattr(
        "risk_management.email_notifications.smtplib.SMTP",
        lambda host, port, timeout=None: sender,
    )
    alert_sender = EmailAlertSender(
        EmailSettings(host="localhost", sender="robot@example.com", use_tls=False)
    )
    alert_sender.send("Notice", "Hi", [])
    assert sender.sent_messages == []

    alert_sender.send("Notice", "Hi", ["user@example.com"])
    message, from_addr, _ = sender.sent_messages[0]
    assert from_addr == "robot@example.com"
    assert message["From"] == "robot@example.com"


def test_telegram_notifier_posts_and_logs(monkeypatch, caplog):
    response = DummyResponse(status_code=500, text="fail")
    client = DummyClient(response)

    def client_factory(timeout):
        assert timeout == 5.0
        return client

    fake_httpx = types.SimpleNamespace(Client=client_factory)
    monkeypatch.setattr("risk_management.telegram_notifications.httpx", fake_httpx)

    notifier = TelegramNotifier(timeout=5.0)
    notifier.send("token", "123", "hello")

    assert client.calls == [
        ("https://api.telegram.org/bottoken/sendMessage", {"chat_id": "123", "text": "hello"})
    ]
    assert any("status 500" in record.message for record in caplog.records)


def test_telegram_notifier_ignores_missing_inputs(monkeypatch):
    called = False

    def _unexpected_client(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("HTTP client should not be invoked")

    monkeypatch.setattr("risk_management.telegram_notifications.httpx", types.SimpleNamespace(Client=_unexpected_client))

    notifier = TelegramNotifier(timeout=1)
    notifier.send("", "", "")
    assert called is False


