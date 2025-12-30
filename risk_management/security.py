"""Security utilities for institutional-grade input validation and secrets management.

This module provides:
- Input validation and sanitization
- Secrets management (avoiding hardcoded API keys)
- Rate limiting
- Authentication helpers
- Security headers
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .exceptions import DataValidationError, SecurityError, APIKeyError
from .logging_config import get_logger, AuditLogger

logger = get_logger(__name__)
audit_logger = AuditLogger()


# ============================================================================
# Input Validation
# ============================================================================

class InputValidator:
    """Validates and sanitizes external inputs."""
    
    # Regular expressions for common validation patterns
    EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{2,12}(/[A-Z0-9]{2,12})?:?[A-Z]*$")
    ACCOUNT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\s]{1,50}$")
    
    @classmethod
    def validate_email(cls, email: str) -> str:
        """Validate email address format.
        
        Args:
            email: Email address to validate
        
        Returns:
            Sanitized email address
        
        Raises:
            DataValidationError: If email format is invalid
        """
        if not isinstance(email, str):
            raise DataValidationError(
                "Email must be a string",
                field_name="email",
                invalid_value=type(email).__name__,
            )
        
        email = email.strip().lower()
        
        if not cls.EMAIL_PATTERN.match(email):
            raise DataValidationError(
                "Invalid email address format",
                field_name="email",
                invalid_value=email,
                validation_rule="EMAIL_PATTERN",
            )
        
        return email
    
    @classmethod
    def validate_symbol(cls, symbol: str) -> str:
        """Validate trading symbol format.
        
        Args:
            symbol: Trading symbol to validate (e.g., "BTC/USDT:USDT")
        
        Returns:
            Sanitized symbol
        
        Raises:
            DataValidationError: If symbol format is invalid
        """
        if not isinstance(symbol, str):
            raise DataValidationError(
                "Symbol must be a string",
                field_name="symbol",
                invalid_value=type(symbol).__name__,
            )
        
        symbol = symbol.strip().upper()
        
        if not cls.SYMBOL_PATTERN.match(symbol):
            raise DataValidationError(
                "Invalid symbol format",
                field_name="symbol",
                invalid_value=symbol,
                validation_rule="SYMBOL_PATTERN",
            )
        
        return symbol
    
    @classmethod
    def validate_account_name(cls, account_name: str) -> str:
        """Validate account name format.
        
        Args:
            account_name: Account name to validate
        
        Returns:
            Sanitized account name
        
        Raises:
            DataValidationError: If account name format is invalid
        """
        if not isinstance(account_name, str):
            raise DataValidationError(
                "Account name must be a string",
                field_name="account_name",
                invalid_value=type(account_name).__name__,
            )
        
        account_name = account_name.strip()
        
        if not cls.ACCOUNT_NAME_PATTERN.match(account_name):
            raise DataValidationError(
                "Invalid account name format (alphanumeric, spaces, hyphens, underscores only)",
                field_name="account_name",
                invalid_value=account_name,
                validation_rule="ACCOUNT_NAME_PATTERN",
            )
        
        return account_name
    
    @classmethod
    def validate_positive_number(
        cls,
        value: Union[int, float],
        field_name: str = "value",
        *,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        """Validate that a number is positive and within range.
        
        Args:
            value: Number to validate
            field_name: Name of the field (for error messages)
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
        
        Returns:
            Validated number as float
        
        Raises:
            DataValidationError: If validation fails
        """
        if not isinstance(value, (int, float)):
            raise DataValidationError(
                f"{field_name} must be a number",
                field_name=field_name,
                invalid_value=type(value).__name__,
            )
        
        value = float(value)
        
        if min_value is not None and value < min_value:
            raise DataValidationError(
                f"{field_name} must be at least {min_value}",
                field_name=field_name,
                invalid_value=value,
                validation_rule=f"min_value={min_value}",
            )
        
        if max_value is not None and value > max_value:
            raise DataValidationError(
                f"{field_name} must be at most {max_value}",
                field_name=field_name,
                invalid_value=value,
                validation_rule=f"max_value={max_value}",
            )
        
        return value
    
    @classmethod
    def sanitize_string(
        cls,
        value: str,
        field_name: str = "value",
        *,
        max_length: int = 1000,
        allow_newlines: bool = False,
    ) -> str:
        """Sanitize string input.
        
        Args:
            value: String to sanitize
            field_name: Name of the field (for error messages)
            max_length: Maximum allowed length
            allow_newlines: Whether to allow newline characters
        
        Returns:
            Sanitized string
        
        Raises:
            DataValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise DataValidationError(
                f"{field_name} must be a string",
                field_name=field_name,
                invalid_value=type(value).__name__,
            )
        
        # Remove leading/trailing whitespace
        value = value.strip()
        
        # Check length
        if len(value) > max_length:
            raise DataValidationError(
                f"{field_name} exceeds maximum length of {max_length}",
                field_name=field_name,
                invalid_value=f"length={len(value)}",
                validation_rule=f"max_length={max_length}",
            )
        
        # Remove newlines if not allowed
        if not allow_newlines:
            value = value.replace("\n", " ").replace("\r", " ")
        
        # Remove control characters
        value = "".join(char for char in value if ord(char) >= 32 or char in "\n\r\t")
        
        return value


# ============================================================================
# Secrets Management
# ============================================================================

@dataclass
class APICredentials:
    """API credentials with secure handling."""
    api_key: str
    api_secret: str
    exchange: str
    account_name: str
    
    def __repr__(self) -> str:
        """Safe representation that doesn't expose secrets."""
        return (
            f"APICredentials(exchange='{self.exchange}', "
            f"account_name='{self.account_name}', "
            f"api_key='***{self.api_key[-4:] if len(self.api_key) >= 4 else '***'}')"
        )
    
    def mask_secret(self, length: int = 4) -> str:
        """Return masked version of the secret."""
        if len(self.api_secret) >= length:
            return f"***{self.api_secret[-length:]}"
        return "***"


class SecretsManager:
    """Manages API keys and secrets securely.
    
    Priorities (in order):
    1. Environment variables (most secure)
    2. Secure secrets file (encrypted or restricted permissions)
    3. Configuration file (fallback, logs warning)
    
    Example:
        >>> manager = SecretsManager()
        >>> credentials = manager.get_credentials("binance_main")
    """
    
    def __init__(
        self,
        *,
        secrets_file: Optional[Path] = None,
        env_prefix: str = "RISK_MGMT",
    ) -> None:
        """Initialize secrets manager.
        
        Args:
            secrets_file: Path to secrets file (JSON format)
            env_prefix: Prefix for environment variables
        """
        self.secrets_file = secrets_file
        self.env_prefix = env_prefix
        self._credentials_cache: Dict[str, APICredentials] = {}
        self._logger = get_logger(__name__)
    
    def get_credentials(
        self,
        key_id: str,
        *,
        exchange: Optional[str] = None,
        fallback_config: Optional[Dict[str, Any]] = None,
    ) -> APICredentials:
        """Get API credentials for a key ID.
        
        Args:
            key_id: Identifier for the API key
            exchange: Exchange name (optional)
            fallback_config: Fallback configuration if not found elsewhere
        
        Returns:
            API credentials
        
        Raises:
            APIKeyError: If credentials not found or invalid
        """
        # Check cache first
        if key_id in self._credentials_cache:
            return self._credentials_cache[key_id]
        
        # Try environment variables first (most secure)
        api_key = os.getenv(f"{self.env_prefix}_{key_id.upper()}_API_KEY")
        api_secret = os.getenv(f"{self.env_prefix}_{key_id.upper()}_API_SECRET")
        
        if api_key and api_secret:
            self._logger.info(
                "Loaded credentials for '%s' from environment variables (secure)",
                key_id
            )
            credentials = APICredentials(
                api_key=api_key,
                api_secret=api_secret,
                exchange=exchange or "unknown",
                account_name=key_id,
            )
            self._credentials_cache[key_id] = credentials
            return credentials
        
        # Try secrets file (if provided)
        if self.secrets_file and self.secrets_file.exists():
            try:
                import json
                with open(self.secrets_file, "r") as f:
                    secrets_data = json.load(f)
                
                key_data = secrets_data.get(key_id, {})
                if key_data.get("api_key") and key_data.get("api_secret"):
                    self._logger.info(
                        "Loaded credentials for '%s' from secrets file",
                        key_id
                    )
                    credentials = APICredentials(
                        api_key=key_data["api_key"],
                        api_secret=key_data["api_secret"],
                        exchange=key_data.get("exchange", exchange or "unknown"),
                        account_name=key_id,
                    )
                    self._credentials_cache[key_id] = credentials
                    return credentials
            except Exception as exc:
                self._logger.error(
                    "Failed to load credentials from secrets file: %s",
                    exc
                )
        
        # Fallback to configuration (least secure - log warning)
        if fallback_config:
            api_key = fallback_config.get("api_key")
            api_secret = fallback_config.get("api_secret")
            
            if api_key and api_secret:
                self._logger.warning(
                    "Loaded credentials for '%s' from configuration file. "
                    "Consider using environment variables or secrets file for better security.",
                    key_id
                )
                audit_logger.log_event(
                    event_type="INSECURE_CREDENTIALS_LOAD",
                    description=f"Credentials for '{key_id}' loaded from config file",
                    severity="WARNING",
                    key_id=key_id,
                )
                credentials = APICredentials(
                    api_key=api_key,
                    api_secret=api_secret,
                    exchange=fallback_config.get("exchange", exchange or "unknown"),
                    account_name=key_id,
                )
                self._credentials_cache[key_id] = credentials
                return credentials
        
        # Not found anywhere
        raise APIKeyError(
            f"Credentials not found for key ID '{key_id}'",
            key_id=key_id,
            service=exchange,
        )
    
    def clear_cache(self) -> None:
        """Clear credentials cache (e.g., for key rotation)."""
        self._credentials_cache.clear()
        self._logger.info("Credentials cache cleared")


# ============================================================================
# Token Generation
# ============================================================================

class TokenGenerator:
    """Generates secure tokens for sessions, API keys, etc."""
    
    @staticmethod
    def generate_session_token(length: int = 32) -> str:
        """Generate a secure session token.
        
        Args:
            length: Token length in bytes
        
        Returns:
            Hex-encoded token
        """
        return secrets.token_hex(length)
    
    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """Generate a secure API key.
        
        Args:
            length: Key length in bytes
        
        Returns:
            URL-safe base64-encoded key
        """
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_hmac_signature(
        message: str,
        secret: str,
        algorithm: str = "sha256",
    ) -> str:
        """Generate HMAC signature for message authentication.
        
        Args:
            message: Message to sign
            secret: Secret key
            algorithm: Hash algorithm (sha256, sha512, etc.)
        
        Returns:
            Hex-encoded signature
        """
        hasher = hmac.new(
            secret.encode("utf-8"),
            message.encode("utf-8"),
            getattr(hashlib, algorithm),
        )
        return hasher.hexdigest()
    
    @staticmethod
    def verify_hmac_signature(
        message: str,
        signature: str,
        secret: str,
        algorithm: str = "sha256",
    ) -> bool:
        """Verify HMAC signature.
        
        Args:
            message: Original message
            signature: Signature to verify
            secret: Secret key
            algorithm: Hash algorithm
        
        Returns:
            True if signature is valid, False otherwise
        """
        expected = TokenGenerator.generate_hmac_signature(
            message, secret, algorithm
        )
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(signature, expected)


# ============================================================================
# Security Headers
# ============================================================================

class SecurityHeaders:
    """HTTP security headers for web dashboard."""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get recommended security headers.
        
        Returns:
            Dictionary of header name -> value
        """
        return {
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            
            # Enable XSS protection
            "X-XSS-Protection": "1; mode=block",
            
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            
            # HTTPS only (if using TLS)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self' data:; "
                "connect-src 'self'"
            ),
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Feature policy
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=()"
            ),
        }


__all__ = [
    "InputValidator",
    "APICredentials",
    "SecretsManager",
    "TokenGenerator",
    "SecurityHeaders",
]
