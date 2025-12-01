import asyncio

from services.notifications import send_email_message, send_telegram_message


def test_telegram_retries_and_reports_error():
    attempts = 0

    async def failing_request(url, payload):  # noqa: ARG001
        nonlocal attempts
        attempts += 1
        raise RuntimeError("network down")

    result = asyncio.run(
        send_telegram_message(
            "token",
            "chat",
            "hello",
            max_retries=2,
            backoff_seconds=0,
            request_func=failing_request,
        )
    )

    assert not result.success
    assert attempts == 3
    assert result.attempts == 3
    assert result.error is not None
    assert "network down" in result.error.reason
    assert not result.error.retryable


def test_email_retries_then_succeeds():
    attempts = 0

    def flaky_sender():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("transient")
        return "sent"

    result = asyncio.run(
        send_email_message(
            smtp_server="smtp.test",
            smtp_port=25,
            from_address="from@example.com",
            to_addresses=["to@example.com"],
            subject="hello",
            body="world",
            max_retries=2,
            backoff_seconds=0,
            send_func=flaky_sender,
        )
    )

    assert result.success
    assert attempts == 2
    assert result.attempts == 2
    assert result.error is None
