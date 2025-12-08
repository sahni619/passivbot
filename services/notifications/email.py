from __future__ import annotations

import asyncio
import logging
import smtplib
from email.mime.text import MIMEText
from typing import Iterable, Optional

from .types import NotificationError, NotificationResult

logger = logging.getLogger(__name__)


async def send_email_message(
    *,
    smtp_server: str,
    smtp_port: int,
    from_address: str,
    to_addresses: Iterable[str],
    subject: str,
    body: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_tls: bool = True,
    max_retries: int = 2,
    backoff_seconds: float = 0.5,
    timeout: float = 10.0,
    send_func=None,
) -> NotificationResult:
    """Send an email with retries and structured error reporting.

    ``send_func`` can be provided for testing to replace the actual SMTP call.
    """

    async def _dispatch() -> None:
        if send_func is not None:
            result = send_func()
            if asyncio.iscoroutine(result):
                await result
            return
        await asyncio.to_thread(
            _send_email_sync,
            smtp_server,
            smtp_port,
            from_address,
            tuple(to_addresses),
            subject,
            body,
            username,
            password,
            use_tls,
            timeout,
        )

    return await _execute_with_retries(
        "email",
        _dispatch,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )


def _send_email_sync(
    smtp_server: str,
    smtp_port: int,
    from_address: str,
    to_addresses: Iterable[str],
    subject: str,
    body: str,
    username: Optional[str],
    password: Optional[str],
    use_tls: bool,
    timeout: float,
) -> None:
    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = from_address
    message["To"] = ", ".join(to_addresses)

    with smtplib.SMTP(smtp_server, smtp_port, timeout=timeout) as smtp:
        if use_tls:
            smtp.starttls()
        if username and password:
            smtp.login(username, password)
        smtp.sendmail(from_address, list(to_addresses), message.as_string())


async def _execute_with_retries(
    channel: str,
    operation,
    *,
    max_retries: int,
    backoff_seconds: float,
) -> NotificationResult:
    attempts = 0
    last_error: Optional[Exception] = None
    while attempts <= max_retries:
        attempts += 1
        try:
            await operation()
            return NotificationResult(channel=channel, success=True, attempts=attempts)
        except Exception as exc:  # pragma: no cover - resilience branch
            last_error = exc
            logger.debug("%s notification attempt %s failed: %s", channel, attempts, exc)
            if attempts > max_retries:
                break
            await asyncio.sleep(backoff_seconds * attempts)

    assert last_error is not None  # nosec - guarded by loop condition
    error = NotificationError(
        channel=channel,
        reason=str(last_error),
        retryable=False,
        details={"attempts": attempts},
    )
    return NotificationResult(channel=channel, success=False, attempts=attempts, error=error)
