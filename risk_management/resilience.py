"""Resilience patterns for institutional-grade error handling.

This module implements robust resilience patterns including:
- Retry logic with exponential backoff
- Circuit breakers for external service protection
- Bulkheads for resource isolation
- Fallback mechanisms
- Rate limiting
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union

from .exceptions import (
    CircuitBreakerOpenError,
    ExchangeRateLimitError,
    RiskManagementError,
    is_recoverable,
)
from .logging_config import get_logger, log_performance

logger = get_logger(__name__)

T = TypeVar("T")


# ============================================================================
# Circuit Breaker
# ============================================================================

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    
    # Number of failures before opening circuit
    failure_threshold: int = 5
    
    # Time window for counting failures (seconds)
    failure_window_seconds: float = 60.0
    
    # How long to wait before trying again (seconds)
    cooldown_seconds: float = 30.0
    
    # Number of successful calls needed in half-open state
    success_threshold: int = 2
    
    # Consider these exceptions as failures
    failure_exceptions: tuple[type[Exception], ...] = (Exception,)
    
    # Don't count these exceptions as failures (they're expected)
    excluded_exceptions: tuple[type[Exception], ...] = ()


class CircuitBreaker:
    """Circuit breaker pattern implementation.
    
    Protects external services from cascading failures by:
    1. Tracking failure rates
    2. Opening circuit (blocking calls) when failures exceed threshold
    3. Allowing test calls after cooldown period
    4. Closing circuit when service recovers
    
    Example:
        >>> breaker = CircuitBreaker("exchange_api")
        >>> async with breaker:
        ...     result = await fetch_from_exchange()
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        """Initialize circuit breaker.
        
        Args:
            name: Identifier for this circuit breaker
            config: Configuration parameters
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._opened_at: Optional[datetime] = None
        self._failure_times: list[datetime] = []
        
        self._logger = get_logger(f"{__name__}.{name}")
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking calls)."""
        return self._state == CircuitState.OPEN
    
    def _clean_old_failures(self) -> None:
        """Remove failures outside the time window."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.config.failure_window_seconds)
        self._failure_times = [t for t in self._failure_times if t > cutoff]
        self._failure_count = len(self._failure_times)
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._opened_at is None:
            return False
        
        now = datetime.now(timezone.utc)
        cooldown_elapsed = (now - self._opened_at).total_seconds()
        return cooldown_elapsed >= self.config.cooldown_seconds
    
    def _transition_to_half_open(self) -> None:
        """Transition from open to half-open state."""
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0
        self._logger.info(
            "Circuit breaker '%s' entering HALF_OPEN state (testing recovery)",
            self.name
        )
    
    def _record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._close_circuit()
        elif self._state == CircuitState.CLOSED:
            # Clean up old failures on success
            self._clean_old_failures()
    
    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        # Check if this exception should be excluded
        if isinstance(exception, self.config.excluded_exceptions):
            return
        
        now = datetime.now(timezone.utc)
        self._failure_times.append(now)
        self._last_failure_time = now
        self._clean_old_failures()
        
        if self._state == CircuitState.HALF_OPEN:
            # Failure in half-open state opens circuit again
            self._open_circuit()
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._open_circuit()
    
    def _open_circuit(self) -> None:
        """Open the circuit (block calls)."""
        self._state = CircuitState.OPEN
        self._opened_at = datetime.now(timezone.utc)
        self._logger.warning(
            "Circuit breaker '%s' OPENED after %d failures",
            self.name,
            self._failure_count,
            extra={
                "circuit_name": self.name,
                "failure_count": self._failure_count,
                "cooldown_seconds": self.config.cooldown_seconds,
            }
        )
    
    def _close_circuit(self) -> None:
        """Close the circuit (allow calls)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._failure_times.clear()
        self._opened_at = None
        self._logger.info(
            "Circuit breaker '%s' CLOSED (service recovered)",
            self.name
        )
    
    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute a function through the circuit breaker.
        
        Args:
            func: Async function to execute
        
        Returns:
            Result from the function
        
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception raised by the function
        """
        # Check if we should attempt reset
        if self._state == CircuitState.OPEN and self._should_attempt_reset():
            self._transition_to_half_open()
        
        # Block if circuit is still open
        if self._state == CircuitState.OPEN:
            elapsed = 0.0
            if self._opened_at:
                elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
            
            remaining = max(0, self.config.cooldown_seconds - elapsed)
            
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN",
                circuit_name=self.name,
                failure_count=self._failure_count,
                cooldown_seconds=int(remaining),
            )
        
        # Execute the function
        try:
            result = await func()
            self._record_success()
            return result
        except Exception as exc:
            self._record_failure(exc)
            raise
    
    async def __aenter__(self) -> CircuitBreaker:
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if exc_type is not None and exc_val is not None:
            self._record_failure(exc_val)
        else:
            self._record_success()


# ============================================================================
# Retry Logic with Exponential Backoff
# ============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    # Maximum number of retry attempts
    max_attempts: int = 3
    
    # Initial delay between retries (seconds)
    initial_delay: float = 1.0
    
    # Maximum delay between retries (seconds)
    max_delay: float = 60.0
    
    # Exponential backoff multiplier
    backoff_multiplier: float = 2.0
    
    # Add random jitter to delays (reduces thundering herd)
    jitter: bool = True
    
    # Maximum jitter as fraction of delay (0.0 to 1.0)
    jitter_fraction: float = 0.1
    
    # Only retry these exception types
    retriable_exceptions: tuple[type[Exception], ...] = (Exception,)
    
    # Never retry these exception types
    non_retriable_exceptions: tuple[type[Exception], ...] = ()


class RetryHandler:
    """Retry logic with exponential backoff and jitter.
    
    Example:
        >>> retry = RetryHandler("fetch_balance")
        >>> result = await retry.execute(fetch_balance_from_exchange)
    """
    
    def __init__(
        self,
        operation_name: str,
        config: Optional[RetryConfig] = None,
    ) -> None:
        """Initialize retry handler.
        
        Args:
            operation_name: Name of the operation (for logging)
            config: Retry configuration
        """
        self.operation_name = operation_name
        self.config = config or RetryConfig()
        self._logger = get_logger(__name__)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number.
        
        Args:
            attempt: Current attempt number (0-indexed)
        
        Returns:
            Delay in seconds
        """
        # Exponential backoff: delay = initial * (multiplier ^ attempt)
        delay = self.config.initial_delay * (
            self.config.backoff_multiplier ** attempt
        )
        
        # Cap at max_delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_fraction
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay + jitter)
        
        return delay
    
    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry after this exception.
        
        Args:
            exception: The exception that occurred
            attempt: Current attempt number
        
        Returns:
            True if we should retry, False otherwise
        """
        # Check attempt limit
        if attempt >= self.config.max_attempts:
            return False
        
        # Check if exception is non-retriable
        if isinstance(exception, self.config.non_retriable_exceptions):
            return False
        
        # Check if exception is explicitly retriable
        if not isinstance(exception, self.config.retriable_exceptions):
            return False
        
        # Use custom logic for RiskManagementError
        if isinstance(exception, RiskManagementError):
            return is_recoverable(exception)
        
        # Default: retry
        return True
    
    async def execute(
        self,
        func: Callable[[], Awaitable[T]],
        *,
        on_retry: Optional[Callable[[Exception, int, float], None]] = None,
    ) -> T:
        """Execute function with retry logic.
        
        Args:
            func: Async function to execute
            on_retry: Optional callback called before each retry
        
        Returns:
            Result from the function
        
        Raises:
            Exception: The last exception if all retries exhausted
        """
        last_exception: Optional[Exception] = None
        
        for attempt in range(self.config.max_attempts):
            try:
                with log_performance(
                    f"{self.operation_name}:attempt_{attempt + 1}",
                    self._logger,
                    attempt=attempt + 1,
                    max_attempts=self.config.max_attempts,
                ):
                    result = await func()
                
                # Success!
                if attempt > 0:
                    self._logger.info(
                        "Operation '%s' succeeded on attempt %d/%d",
                        self.operation_name,
                        attempt + 1,
                        self.config.max_attempts,
                    )
                
                return result
                
            except Exception as exc:
                last_exception = exc
                
                # Check if we should retry
                if not self._should_retry(exc, attempt):
                    self._logger.error(
                        "Operation '%s' failed with non-retriable error: %s",
                        self.operation_name,
                        exc,
                        extra={
                            "operation": self.operation_name,
                            "attempt": attempt + 1,
                            "exception_type": type(exc).__name__,
                        },
                    )
                    raise
                
                # This was the last attempt
                if attempt + 1 >= self.config.max_attempts:
                    self._logger.error(
                        "Operation '%s' failed after %d attempts: %s",
                        self.operation_name,
                        self.config.max_attempts,
                        exc,
                        extra={
                            "operation": self.operation_name,
                            "attempts": self.config.max_attempts,
                            "exception_type": type(exc).__name__,
                        },
                    )
                    raise
                
                # Calculate delay and log retry
                delay = self._calculate_delay(attempt)
                
                self._logger.warning(
                    "Operation '%s' failed on attempt %d/%d, retrying in %.2fs: %s",
                    self.operation_name,
                    attempt + 1,
                    self.config.max_attempts,
                    delay,
                    exc,
                    extra={
                        "operation": self.operation_name,
                        "attempt": attempt + 1,
                        "max_attempts": self.config.max_attempts,
                        "retry_delay": delay,
                        "exception_type": type(exc).__name__,
                    },
                )
                
                # Call retry callback if provided
                if on_retry:
                    try:
                        on_retry(exc, attempt, delay)
                    except Exception:  # pragma: no cover
                        self._logger.exception("Retry callback failed")
                
                # Wait before retrying
                await asyncio.sleep(delay)
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Operation '{self.operation_name}' failed without exception")


# ============================================================================
# Combined Resilience Manager
# ============================================================================

@dataclass
class ResilienceConfig:
    """Combined configuration for resilience patterns."""
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: Optional[CircuitBreakerConfig] = None


class ResilientExecutor:
    """Execute operations with combined resilience patterns.
    
    Combines retry logic, circuit breakers, and performance logging.
    
    Example:
        >>> executor = ResilientExecutor("exchange_api")
        >>> result = await executor.execute(fetch_data)
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[ResilienceConfig] = None,
    ) -> None:
        """Initialize resilient executor.
        
        Args:
            name: Identifier for this executor
            config: Resilience configuration
        """
        self.name = name
        self.config = config or ResilienceConfig()
        
        self.retry_handler = RetryHandler(name, self.config.retry)
        
        if self.config.circuit_breaker:
            self.circuit_breaker: Optional[CircuitBreaker] = CircuitBreaker(
                name, self.config.circuit_breaker
            )
        else:
            self.circuit_breaker = None
        
        self._logger = get_logger(__name__)
    
    async def execute(
        self,
        func: Callable[[], Awaitable[T]],
        *,
        fallback: Optional[Callable[[], Awaitable[T]]] = None,
    ) -> T:
        """Execute function with full resilience protection.
        
        Args:
            func: Async function to execute
            fallback: Optional fallback function if all attempts fail
        
        Returns:
            Result from the function (or fallback)
        
        Raises:
            Exception: If function fails and no fallback provided
        """
        async def _execute_with_circuit_breaker() -> T:
            if self.circuit_breaker:
                return await self.circuit_breaker.call(func)
            else:
                return await func()
        
        try:
            return await self.retry_handler.execute(_execute_with_circuit_breaker)
        except Exception as exc:
            # Try fallback if available
            if fallback:
                self._logger.warning(
                    "Using fallback for operation '%s' after failure: %s",
                    self.name,
                    exc,
                )
                try:
                    return await fallback()
                except Exception as fallback_exc:
                    self._logger.error(
                        "Fallback also failed for operation '%s': %s",
                        self.name,
                        fallback_exc,
                    )
            
            raise


# ============================================================================
# Rate Limiter
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter for API calls.
    
    Example:
        >>> limiter = RateLimiter(requests_per_second=5)
        >>> async with limiter:
        ...     await make_api_call()
    """
    
    def __init__(
        self,
        requests_per_second: float,
        *,
        burst_size: Optional[int] = None,
    ) -> None:
        """Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst size (defaults to requests_per_second)
        """
        self.rate = requests_per_second
        self.burst_size = burst_size or int(requests_per_second)
        
        self._tokens = float(self.burst_size)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire
        """
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_update
                
                # Add tokens based on elapsed time
                self._tokens = min(
                    self.burst_size,
                    self._tokens + elapsed * self.rate
                )
                self._last_update = now
                
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                
                # Calculate wait time
                deficit = tokens - self._tokens
                wait_time = deficit / self.rate
                await asyncio.sleep(wait_time)
    
    async def __aenter__(self) -> RateLimiter:
        """Context manager entry."""
        await self.acquire()
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """Context manager exit."""
        pass


__all__ = [
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "RetryConfig",
    "RetryHandler",
    "ResilienceConfig",
    "ResilientExecutor",
    "RateLimiter",
]
