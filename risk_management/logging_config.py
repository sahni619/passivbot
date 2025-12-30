"""Institutional-grade logging infrastructure with structured logging and audit trails.

This module provides comprehensive logging facilities including:
- Structured logging with JSON formatting
- Correlation ID tracking for request tracing
- Performance metrics logging
- Audit logging for critical operations
- Log rotation and retention
- Multiple log levels and handlers
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .exceptions import extract_error_context


# ============================================================================
# Context Variables for Correlation Tracking
# ============================================================================

# Correlation ID for tracing requests across the system
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)

# Account name for context-aware logging
_account_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "account_context", default=None
)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for the current context.
    
    Args:
        correlation_id: Optional correlation ID. If None, generates a new UUID.
    
    Returns:
        The correlation ID that was set
    """
    cid = correlation_id or str(uuid.uuid4())
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return _correlation_id.get()


def set_account_context(account_name: Optional[str]) -> None:
    """Set account context for logging."""
    _account_context.set(account_name)


def get_account_context() -> Optional[str]:
    """Get the current account context."""
    return _account_context.get()


def clear_context() -> None:
    """Clear all context variables."""
    _correlation_id.set(None)
    _account_context.set(None)


# ============================================================================
# Structured JSON Formatter
# ============================================================================

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging with context injection."""
    
    def __init__(
        self,
        *,
        include_timestamp: bool = True,
        include_context: bool = True,
        include_exc_info: bool = True,
    ) -> None:
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_context = include_context
        self.include_exc_info = include_exc_info
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with context."""
        log_data: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add timestamp
        if self.include_timestamp:
            log_data["timestamp"] = datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat()
        
        # Add correlation ID if available
        if self.include_context:
            correlation_id = get_correlation_id()
            if correlation_id:
                log_data["correlation_id"] = correlation_id
            
            # Add account context if available
            account = get_account_context()
            if account:
                log_data["account"] = account
        
        # Add source location
        log_data["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName", "relativeCreated",
                "thread", "threadName", "exc_info", "exc_text", "stack_info",
            }:
                extra_fields[key] = value
        
        if extra_fields:
            log_data["extra"] = extra_fields
        
        # Add exception information
        if self.include_exc_info and record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            if exc_value:
                log_data["exception"] = extract_error_context(exc_value)
                log_data["exception"]["traceback"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


# ============================================================================
# Performance Logging
# ============================================================================

class PerformanceLogger:
    """Context manager for logging operation performance metrics."""
    
    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        *,
        log_level: int = logging.INFO,
        threshold_seconds: Optional[float] = None,
        **context: Any,
    ) -> None:
        """Initialize performance logger.
        
        Args:
            operation_name: Name of the operation being measured
            logger: Logger to use (defaults to root logger)
            log_level: Logging level for performance metrics
            threshold_seconds: Only log if duration exceeds this threshold
            **context: Additional context to include in logs
        """
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.log_level = log_level
        self.threshold_seconds = threshold_seconds
        self.context = context
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> PerformanceLogger:
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End timing and log performance."""
        self.end_time = time.perf_counter()
        
        if self.start_time is None:
            return
        
        duration = self.end_time - self.start_time
        
        # Check threshold
        if self.threshold_seconds and duration < self.threshold_seconds:
            return
        
        # Log performance metrics
        extra = {
            "operation": self.operation_name,
            "duration_seconds": duration,
            "duration_ms": duration * 1000,
            "success": exc_type is None,
            **self.context,
        }
        
        if exc_type is None:
            self.logger.log(
                self.log_level,
                f"Operation '{self.operation_name}' completed in {duration:.3f}s",
                extra=extra,
            )
        else:
            self.logger.warning(
                f"Operation '{self.operation_name}' failed after {duration:.3f}s",
                extra=extra,
                exc_info=(exc_type, exc_val, exc_tb),
            )
    
    def get_duration(self) -> Optional[float]:
        """Get the measured duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time


# ============================================================================
# Audit Logger
# ============================================================================

class AuditLogger:
    """Specialized logger for audit trail of critical operations.
    
    Audit logs are immutable records of security-relevant events including:
    - Authentication/authorization
    - Configuration changes
    - Manual interventions
    - Kill switch activations
    - Risk limit breaches
    """
    
    def __init__(self, logger_name: str = "risk_management.audit") -> None:
        self.logger = logging.getLogger(logger_name)
    
    def log_event(
        self,
        event_type: str,
        description: str,
        *,
        severity: str = "INFO",
        user: Optional[str] = None,
        account: Optional[str] = None,
        **details: Any,
    ) -> None:
        """Log an audit event.
        
        Args:
            event_type: Type of event (e.g., "AUTH", "CONFIG_CHANGE", "KILL_SWITCH")
            description: Human-readable description
            severity: Event severity ("INFO", "WARNING", "ERROR", "CRITICAL")
            user: Username associated with event
            account: Account name associated with event
            **details: Additional event-specific details
        """
        extra = {
            "audit_event": True,
            "event_type": event_type,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        if user:
            extra["user"] = user
        if account:
            extra["account"] = account
        
        if details:
            extra["details"] = details
        
        level = getattr(logging, severity.upper(), logging.INFO)
        self.logger.log(level, f"[AUDIT] {event_type}: {description}", extra=extra)
    
    def log_authentication(
        self,
        user: str,
        success: bool,
        *,
        ip_address: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Log authentication attempt."""
        self.log_event(
            event_type="AUTHENTICATION",
            description=f"User '{user}' authentication {'succeeded' if success else 'failed'}",
            severity="INFO" if success else "WARNING",
            user=user,
            success=success,
            ip_address=ip_address,
            failure_reason=reason if not success else None,
        )
    
    def log_config_change(
        self,
        config_key: str,
        old_value: Any,
        new_value: Any,
        *,
        user: Optional[str] = None,
    ) -> None:
        """Log configuration change."""
        self.log_event(
            event_type="CONFIG_CHANGE",
            description=f"Configuration '{config_key}' changed",
            severity="INFO",
            user=user,
            config_key=config_key,
            old_value=str(old_value),
            new_value=str(new_value),
        )
    
    def log_kill_switch(
        self,
        account: Optional[str],
        symbol: Optional[str],
        *,
        user: Optional[str] = None,
        reason: Optional[str] = None,
        success: bool = True,
    ) -> None:
        """Log kill switch activation."""
        scope = account or "all_accounts"
        symbol_desc = f" for {symbol}" if symbol else ""
        
        self.log_event(
            event_type="KILL_SWITCH",
            description=f"Kill switch activated for {scope}{symbol_desc}",
            severity="CRITICAL" if success else "ERROR",
            user=user,
            account=account,
            symbol=symbol,
            reason=reason,
            success=success,
        )
    
    def log_cashflow(
        self,
        flow_type: str,
        amount: float,
        account: str,
        *,
        currency: str = "USDT",
        detection_method: Optional[str] = None,
    ) -> None:
        """Log detected cashflow event."""
        self.log_event(
            event_type="CASHFLOW_DETECTED",
            description=f"{flow_type.capitalize()} of {amount} {currency} on {account}",
            severity="INFO",
            account=account,
            flow_type=flow_type,
            amount=amount,
            currency=currency,
            detection_method=detection_method,
        )
    
    def log_risk_breach(
        self,
        breach_type: str,
        account: str,
        *,
        limit_value: Optional[float] = None,
        current_value: Optional[float] = None,
        **details: Any,
    ) -> None:
        """Log risk limit breach."""
        self.log_event(
            event_type="RISK_BREACH",
            description=f"Risk limit breach: {breach_type} on {account}",
            severity="WARNING",
            account=account,
            breach_type=breach_type,
            limit_value=limit_value,
            current_value=current_value,
            **details,
        )


# ============================================================================
# Logging Configuration
# ============================================================================

def configure_logging(
    log_dir: Optional[Union[str, Path]] = None,
    *,
    log_level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 10,
    audit_enabled: bool = True,
) -> None:
    """Configure logging for the risk management system.
    
    Args:
        log_dir: Directory for log files (default: risk_reports/logs)
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Enable console logging
        file_output: Enable file logging
        json_format: Use JSON formatting for structured logs
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        audit_enabled: Enable separate audit log file
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path("risk_reports") / "logs"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Choose formatter
    if json_format:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Use simpler format for console
        if json_format:
            console_formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
        else:
            console_formatter = formatter
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output:
        log_file = log_dir / "risk_management.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Error log file (only errors and critical)
    if file_output:
        error_log_file = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
    
    # Audit log file
    if audit_enabled and file_output:
        audit_log_file = log_dir / "audit.log"
        audit_handler = logging.handlers.RotatingFileHandler(
            audit_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(StructuredFormatter() if json_format else formatter)
        
        # Add handler only to audit logger
        audit_logger = logging.getLogger("risk_management.audit")
        audit_logger.addHandler(audit_handler)
        audit_logger.propagate = True  # Also send to root logger
    
    # Performance log file
    if file_output:
        perf_log_file = log_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        perf_handler.setLevel(logging.DEBUG)
        perf_handler.setFormatter(StructuredFormatter() if json_format else formatter)
        
        # Add handler only to performance logger
        perf_logger = logging.getLogger("risk_management.performance")
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False  # Don't send to root logger
    
    logging.info("Logging configured: level=%s, json_format=%s, log_dir=%s", log_level, json_format, log_dir)


# ============================================================================
# Convenience Functions
# ============================================================================

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_performance(
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    **context: Any,
) -> PerformanceLogger:
    """Create a performance logger context manager.
    
    Args:
        operation_name: Name of the operation to measure
        logger: Optional logger to use
        **context: Additional context for logging
    
    Returns:
        PerformanceLogger context manager
    
    Example:
        >>> with log_performance("fetch_balances", account="Binance"):
        ...     await fetch_balances()
    """
    return PerformanceLogger(operation_name, logger, **context)


__all__ = [
    "set_correlation_id",
    "get_correlation_id",
    "set_account_context",
    "get_account_context",
    "clear_context",
    "StructuredFormatter",
    "PerformanceLogger",
    "AuditLogger",
    "configure_logging",
    "get_logger",
    "log_performance",
]
