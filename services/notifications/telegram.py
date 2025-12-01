from __future__ import annotations
import asyncio
import json
import logging
import urllib.request
from typing import Any, Mapping, Optional

from .types import NotificationError, NotificationResult

logger = logging.getLogger(__name__)


async def send_telegram_message(
    token: str,
    chat_id: str,
    message: str,
    *,
    parse_mode: str = "Markdown",
    request_func=None,
    max_retries: int = 2,
    backoff_seconds: float = 0.5,
    timeout: float = 10.0,
) -> NotificationResult:
    """Send a Telegram message with retry and backoff policies."""

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": parse_mode}

    async def _dispatch() -> Mapping[str, Any]:
        if request_func is not None:
            return await request_func(url, payload)
        return await asyncio.to_thread(_post_message_sync, url, payload, timeout)

    return await _execute_with_retries(
        "telegram",
        _dispatch,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )


def _post_message_sync(url: str, payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
        content = response.read()
        if response.status >= 400:
            raise RuntimeError(f"telegram responded with {response.status}: {content!r}")
        data = json.loads(content.decode("utf-8"))
        if not data.get("ok", True):
            raise RuntimeError(f"telegram returned failure: {data}")
        return data


async def _execute_with_retries(
    channel: str,
    operation,
    *,
    max_retries: int,
    backoff_seconds: float,
) -> NotificationResult:
    attempts = 0
    last_error: Optional[Exception] = None
    payload: Optional[Mapping[str, Any]] = None
    while attempts <= max_retries:
        attempts += 1
        try:
            payload = await operation()
            return NotificationResult(
                channel=channel, success=True, attempts=attempts, payload=payload
            )
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
