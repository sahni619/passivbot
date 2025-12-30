"""Custom exception hierarchy for institutional-grade error handling.

This module defines a comprehensive exception hierarchy that enables precise error
handling, recovery mechanisms, and proper logging throughout the risk management system.

Exception Categories:
    - Configuration Errors: Issues with system configuration
    - Data Errors: Problems with data validation or integrity
    - External Service Errors: Failures in external dependencies (exchanges, databases)
    - Notification Errors: Issues with alert dispatch systems
    - Security Errors: Authentication, authorization, or validation failures
    - Business Logic Errors: Rule violations or policy breaches
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ============================================================================
# Base Exception
# ============================================================================

class RiskManagementError(Exception):
    """Base exception for all risk management errors.
    
    Attributes:
        message: Human-readable error description
        error_code: Machine-readable error identifier
        context: Additional contextual information for debugging
        recoverable: Whether the error allows for retry/recovery
    """
    
    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.recoverable = recoverable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to structured dictionary for logging/reporting."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "recoverable": self.recoverable,
        }


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(RiskManagementError):
    """Base class for configuration-related errors."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Configuration file or settings are invalid or malformed."""
    
    def __init__(
        self,
        message: str,
        *,
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
    ) -> None:
        context = {}
        if config_key:
            context["config_key"] = config_key
        if expected_type:
            context["expected_type"] = expected_type
        if actual_value is not None:
            context["actual_value"] = str(actual_value)
        
        super().__init__(message, error_code="INVALID_CONFIG", context=context)


class MissingConfigurationError(ConfigurationError):
    """Required configuration is missing."""
    
    def __init__(self, config_key: str, message: Optional[str] = None) -> None:
        msg = message or f"Required configuration key '{config_key}' is missing"
        super().__init__(
            msg,
            error_code="MISSING_CONFIG",
            context={"config_key": config_key}
        )


# ============================================================================
# Data Errors
# ============================================================================

class DataError(RiskManagementError):
    """Base class for data-related errors."""
    pass


class DataValidationError(DataError):
    """Input data failed validation checks."""
    
    def __init__(
        self,
        message: str,
        *,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        validation_rule: Optional[str] = None,
    ) -> None:
        context = {}
        if field_name:
            context["field_name"] = field_name
        if invalid_value is not None:
            context["invalid_value"] = str(invalid_value)
        if validation_rule:
            context["validation_rule"] = validation_rule
        
        super().__init__(message, error_code="VALIDATION_FAILED", context=context)


class DataIntegrityError(DataError):
    """Data consistency or integrity check failed."""
    
    def __init__(
        self,
        message: str,
        *,
        expected_value: Optional[Any] = None,
        actual_value: Optional[Any] = None,
    ) -> None:
        context = {}
        if expected_value is not None:
            context["expected_value"] = str(expected_value)
        if actual_value is not None:
            context["actual_value"] = str(actual_value)
        
        super().__init__(
            message,
            error_code="DATA_INTEGRITY_FAILED",
            context=context,
            recoverable=False
        )


# ============================================================================
# External Service Errors
# ============================================================================

class ExternalServiceError(RiskManagementError):
    """Base class for external service errors."""
    
    def __init__(
        self,
        message: str,
        *,
        service_name: Optional[str] = None,
        error_code: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if service_name:
            context["service_name"] = service_name
        
        super().__init__(
            message,
            error_code=error_code or "EXTERNAL_SERVICE_ERROR",
            context=context,
            recoverable=True,  # Most external errors are recoverable
            **kwargs
        )


class ExchangeAPIError(ExternalServiceError):
    """Error communicating with exchange API."""
    
    def __init__(
        self,
        message: str,
        *,
        exchange: Optional[str] = None,
        account: Optional[str] = None,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        context = {}
        if exchange:
            context["exchange"] = exchange
        if account:
            context["account"] = account
        if endpoint:
            context["endpoint"] = endpoint
        if status_code:
            context["status_code"] = status_code
        
        super().__init__(
            message,
            service_name=exchange or "exchange",
            error_code="EXCHANGE_API_ERROR",
            context=context
        )


class ExchangeAuthenticationError(ExchangeAPIError):
    """Exchange API authentication failed."""
    
    def __init__(
        self,
        message: str,
        *,
        exchange: Optional[str] = None,
        account: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            exchange=exchange,
            account=account,
            error_code="EXCHANGE_AUTH_FAILED"
        )
        self.recoverable = False  # Auth errors are usually not recoverable


class ExchangeRateLimitError(ExchangeAPIError):
    """Exchange API rate limit exceeded."""
    
    def __init__(
        self,
        message: str,
        *,
        exchange: Optional[str] = None,
        retry_after: Optional[int] = None,
    ) -> None:
        context = {}
        if retry_after:
            context["retry_after_seconds"] = retry_after
        
        super().__init__(
            message,
            exchange=exchange,
            error_code="RATE_LIMIT_EXCEEDED",
            **{"context": context}
        )


class DatabaseError(ExternalServiceError):
    """Database operation failed."""
    
    def __init__(
        self,
        message: str,
        *,
        operation: Optional[str] = None,
        table: Optional[str] = None,
    ) -> None:
        context = {}
        if operation:
            context["operation"] = operation
        if table:
            context["table"] = table
        
        super().__init__(
            message,
            service_name="database",
            error_code="DATABASE_ERROR",
            context=context
        )


# ============================================================================
# Notification Errors
# ============================================================================

class NotificationError(RiskManagementError):
    """Base class for notification system errors."""
    
    def __init__(
        self,
        message: str,
        *,
        channel: Optional[str] = None,
        recipient: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if channel:
            context["channel"] = channel
        if recipient:
            context["recipient"] = recipient
        
        super().__init__(
            message,
            error_code="NOTIFICATION_ERROR",
            context=context,
            recoverable=True,  # Notification failures shouldn't stop monitoring
            **kwargs
        )


class EmailNotificationError(NotificationError):
    """Failed to send email notification."""
    
    def __init__(
        self,
        message: str,
        *,
        recipient: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> None:
        context = {}
        if subject:
            context["subject"] = subject
        
        super().__init__(
            message,
            channel="email",
            recipient=recipient,
            context=context
        )


class TelegramNotificationError(NotificationError):
    """Failed to send Telegram notification."""
    
    def __init__(
        self,
        message: str,
        *,
        chat_id: Optional[str] = None,
        bot_token_hint: Optional[str] = None,
    ) -> None:
        context = {}
        if bot_token_hint:
            context["bot_token_hint"] = bot_token_hint  # First few chars only
        
        super().__init__(
            message,
            channel="telegram",
            recipient=chat_id,
            context=context
        )


# ============================================================================
# Security Errors
# ============================================================================

class SecurityError(RiskManagementError):
    """Base class for security-related errors."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code="SECURITY_ERROR",
            recoverable=False,  # Security errors are typically not recoverable
            **kwargs
        )


class AuthenticationError(SecurityError):
    """User authentication failed."""
    
    def __init__(self, message: str, *, username: Optional[str] = None) -> None:
        context = {}
        if username:
            context["username"] = username
        
        super().__init__(message, error_code="AUTH_FAILED", context=context)


class AuthorizationError(SecurityError):
    """User lacks required permissions."""
    
    def __init__(
        self,
        message: str,
        *,
        username: Optional[str] = None,
        required_permission: Optional[str] = None,
    ) -> None:
        context = {}
        if username:
            context["username"] = username
        if required_permission:
            context["required_permission"] = required_permission
        
        super().__init__(message, error_code="AUTHZ_FAILED", context=context)


class APIKeyError(SecurityError):
    """API key is invalid or expired."""
    
    def __init__(
        self,
        message: str,
        *,
        key_id: Optional[str] = None,
        service: Optional[str] = None,
    ) -> None:
        context = {}
        if key_id:
            context["key_id"] = key_id
        if service:
            context["service"] = service
        
        super().__init__(message, error_code="INVALID_API_KEY", context=context)


# ============================================================================
# Business Logic Errors
# ============================================================================

class BusinessLogicError(RiskManagementError):
    """Base class for business rule violations."""
    pass


class RiskLimitExceededError(BusinessLogicError):
    """Risk limit or threshold was exceeded."""
    
    def __init__(
        self,
        message: str,
        *,
        limit_type: Optional[str] = None,
        limit_value: Optional[float] = None,
        current_value: Optional[float] = None,
        account: Optional[str] = None,
    ) -> None:
        context = {}
        if limit_type:
            context["limit_type"] = limit_type
        if limit_value is not None:
            context["limit_value"] = limit_value
        if current_value is not None:
            context["current_value"] = current_value
        if account:
            context["account"] = account
        
        super().__init__(
            message,
            error_code="RISK_LIMIT_EXCEEDED",
            context=context,
            recoverable=False
        )


class PositionLimitError(BusinessLogicError):
    """Position size exceeds allowed limits."""
    
    def __init__(
        self,
        message: str,
        *,
        symbol: Optional[str] = None,
        current_size: Optional[float] = None,
        max_size: Optional[float] = None,
    ) -> None:
        context = {}
        if symbol:
            context["symbol"] = symbol
        if current_size is not None:
            context["current_size"] = current_size
        if max_size is not None:
            context["max_size"] = max_size
        
        super().__init__(
            message,
            error_code="POSITION_LIMIT_EXCEEDED",
            context=context,
            recoverable=False
        )


class CircuitBreakerOpenError(BusinessLogicError):
    """Circuit breaker is open, blocking operation."""
    
    def __init__(
        self,
        message: str,
        *,
        circuit_name: Optional[str] = None,
        failure_count: Optional[int] = None,
        cooldown_seconds: Optional[int] = None,
    ) -> None:
        context = {}
        if circuit_name:
            context["circuit_name"] = circuit_name
        if failure_count is not None:
            context["failure_count"] = failure_count
        if cooldown_seconds is not None:
            context["cooldown_seconds"] = cooldown_seconds
        
        super().__init__(
            message,
            error_code="CIRCUIT_BREAKER_OPEN",
            context=context,
            recoverable=True  # Can retry after cooldown
        )


# ============================================================================
# Cashflow Detection Errors (New for deposit/withdrawal feature)
# ============================================================================

class CashflowError(RiskManagementError):
    """Base class for cashflow detection errors."""
    pass


class CashflowDetectionError(CashflowError):
    """Failed to detect or categorize cashflow event."""
    
    def __init__(
        self,
        message: str,
        *,
        account: Optional[str] = None,
        balance_change: Optional[float] = None,
    ) -> None:
        context = {}
        if account:
            context["account"] = account
        if balance_change is not None:
            context["balance_change"] = balance_change
        
        super().__init__(
            message,
            error_code="CASHFLOW_DETECTION_FAILED",
            context=context,
            recoverable=True
        )


class BalanceReconciliationError(CashflowError):
    """Balance reconciliation check failed."""
    
    def __init__(
        self,
        message: str,
        *,
        account: Optional[str] = None,
        expected_balance: Optional[float] = None,
        actual_balance: Optional[float] = None,
        discrepancy: Optional[float] = None,
    ) -> None:
        context = {}
        if account:
            context["account"] = account
        if expected_balance is not None:
            context["expected_balance"] = expected_balance
        if actual_balance is not None:
            context["actual_balance"] = actual_balance
        if discrepancy is not None:
            context["discrepancy"] = discrepancy
        
        super().__init__(
            message,
            error_code="BALANCE_RECONCILIATION_FAILED",
            context=context,
            recoverable=False  # Data integrity issue
        )


# ============================================================================
# Utility Functions
# ============================================================================

def is_recoverable(exception: Exception) -> bool:
    """Check if an exception is recoverable and can be retried.
    
    Args:
        exception: The exception to check
    
    Returns:
        True if the error is recoverable, False otherwise
    """
    if isinstance(exception, RiskManagementError):
        return exception.recoverable
    
    # By default, assume non-custom exceptions are recoverable
    # (e.g., network errors, timeouts)
    return True


def extract_error_context(exception: Exception) -> Dict[str, Any]:
    """Extract structured context from an exception for logging.
    
    Args:
        exception: The exception to extract context from
    
    Returns:
        Dictionary containing error context
    """
    if isinstance(exception, RiskManagementError):
        return exception.to_dict()
    
    # For standard exceptions, create basic context
    return {
        "error_type": exception.__class__.__name__,
        "message": str(exception),
        "recoverable": True,
    }


__all__ = [
    "RiskManagementError",
    "ConfigurationError",
    "InvalidConfigurationError",
    "MissingConfigurationError",
    "DataError",
    "DataValidationError",
    "DataIntegrityError",
    "ExternalServiceError",
    "ExchangeAPIError",
    "ExchangeAuthenticationError",
    "ExchangeRateLimitError",
    "DatabaseError",
    "NotificationError",
    "EmailNotificationError",
    "TelegramNotificationError",
    "SecurityError",
    "AuthenticationError",
    "AuthorizationError",
    "APIKeyError",
    "BusinessLogicError",
    "RiskLimitExceededError",
    "PositionLimitError",
    "CircuitBreakerOpenError",
    "CashflowError",
    "CashflowDetectionError",
    "BalanceReconciliationError",
    "is_recoverable",
    "extract_error_context",
]
