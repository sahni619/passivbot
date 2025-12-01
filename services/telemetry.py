"""Structured telemetry utilities for logging, metrics, and resiliency."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResiliencePolicy:
    """Default resilience configuration for external calls."""

    request_timeout: float = 10.0
    max_retries: int = 2
    retry_backoff: float = 0.5
    circuit_breaker_threshold: int = 3
    circuit_breaker_reset_s: float = 30.0

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "ResiliencePolicy":
        if not payload:
            return cls()
        kwargs: Dict[str, Any] = {}
        for key in (
            "request_timeout",
            "max_retries",
            "retry_backoff",
            "circuit_breaker_threshold",
            "circuit_breaker_reset_s",
        ):
            if key in payload:
                kwargs[key] = payload[key]
        return cls(**kwargs)


@dataclass
class CircuitBreakerState:
    """Track failures for a service and expose a simple circuit breaker."""

    threshold: int
    reset_seconds: float
    failure_count: int = 0
    opened_at: Optional[float] = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if (time.time() - self.opened_at) >= self.reset_seconds:
            self.failure_count = 0
            self.opened_at = None
            return False
        return True

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.threshold and self.opened_at is None:
            self.opened_at = time.time()

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_at = None


@dataclass
class ServiceStatus:
    status: str
    reason: Optional[str] = None
    last_success: Optional[datetime] = None
    last_checked: Optional[datetime] = None
    failures: int = 0
    successes: int = 0


class Telemetry:
    """Collect structured logs, metrics, and health data."""

    def __init__(self, *, policy: Optional[ResiliencePolicy] = None) -> None:
        self.policy = policy or ResiliencePolicy()
        self.metrics: Dict[str, Any] = {}
        self.service_status: Dict[str, ServiceStatus] = {}
        self._circuit_breakers: Dict[str, CircuitBreakerState] = {}

    def _breaker_for(self, name: str) -> CircuitBreakerState:
        breaker = self._circuit_breakers.get(name)
        if breaker is None:
            breaker = CircuitBreakerState(
                threshold=self.policy.circuit_breaker_threshold,
                reset_seconds=self.policy.circuit_breaker_reset_s,
            )
            self._circuit_breakers[name] = breaker
        return breaker

    def mark_service_healthy(self, name: str) -> None:
        status = self.service_status.get(name)
        now = datetime.now(timezone.utc)
        if status is None:
            status = ServiceStatus(status="healthy")
            self.service_status[name] = status
        status.status = "healthy"
        status.reason = None
        status.last_success = now
        status.last_checked = now
        status.successes += 1

    def mark_service_degraded(self, name: str, reason: str) -> None:
        status = self.service_status.get(name)
        now = datetime.now(timezone.utc)
        if status is None:
            status = ServiceStatus(status="degraded")
            self.service_status[name] = status
        status.status = "degraded"
        status.reason = reason
        status.last_checked = now
        status.failures += 1

    def record_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value

    async def execute_with_resilience(
        self,
        name: str,
        func: Callable[[], Any],
        *,
        policy: Optional[ResiliencePolicy] = None,
    ) -> Any:
        """Execute ``func`` with timeouts, retries, and circuit breaking."""

        selected_policy = policy or self.policy
        breaker = self._breaker_for(name)
        if breaker.is_open():
            reason = "circuit_open"
            logger.warning("Circuit breaker open for %s", name)
            self.mark_service_degraded(name, reason)
            raise RuntimeError(f"Circuit open for {name}")

        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= selected_policy.max_retries:
            attempt += 1
            started = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    func() if asyncio.iscoroutinefunction(func) else func(),
                    timeout=selected_policy.request_timeout,
                )
                duration_ms = (time.perf_counter() - started) * 1000
                self.record_metric(f"{name}.latency_ms", round(duration_ms, 2))
                breaker.record_success()
                self.mark_service_healthy(name)
                return result
            except Exception as exc:  # pragma: no cover - resilience branch
                duration_ms = (time.perf_counter() - started) * 1000
                self.record_metric(f"{name}.latency_ms", round(duration_ms, 2))
                breaker.record_failure()
                last_exc = exc
                self.mark_service_degraded(name, str(exc))
                if attempt > selected_policy.max_retries:
                    break
                backoff = selected_policy.retry_backoff * attempt
                await asyncio.sleep(backoff)

        if last_exc:
            raise last_exc
        raise RuntimeError(f"Unknown failure while executing {name}")

    def health_snapshot(self) -> Dict[str, Any]:
        services: Dict[str, Any] = {}
        for name, status in self.service_status.items():
            services[name] = {
                "status": status.status,
                "reason": status.reason,
                "last_success": status.last_success.isoformat() if status.last_success else None,
                "last_checked": status.last_checked.isoformat() if status.last_checked else None,
                "failures": status.failures,
                "successes": status.successes,
            }
        overall = "healthy"
        for status in self.service_status.values():
            if status.status != "healthy":
                overall = "degraded"
                break
        return {"status": overall, "services": services}

    def readiness_snapshot(self) -> Dict[str, Any]:
        snapshot = self.health_snapshot()
        ready = True
        for status in self.service_status.values():
            if status.last_success is None or status.status != "healthy":
                ready = False
                break
        snapshot["ready"] = ready
        return snapshot


__all__ = ["Telemetry", "ResiliencePolicy", "CircuitBreakerState"]
