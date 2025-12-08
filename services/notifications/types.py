from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class NotificationError:
    channel: str
    reason: str
    retryable: bool
    details: Optional[Mapping[str, Any]] = None


@dataclass
class NotificationResult:
    channel: str
    success: bool
    attempts: int
    error: Optional[NotificationError] = None
    payload: Optional[Mapping[str, Any]] = None
