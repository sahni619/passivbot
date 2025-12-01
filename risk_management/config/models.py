from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Field, root_validator, validator


class _BaseModel(BaseModel):
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True


class CustomEndpointSettings(_BaseModel):
    """Settings controlling how custom endpoint overrides are loaded."""

    path: Optional[str] = None
    autodiscover: bool = True


class AccountConfig(_BaseModel):
    """Configuration for a single exchange account."""

    name: str
    exchange: str
    settle_currency: str = "USDT"
    api_key_id: Optional[str] = None
    credentials: Dict[str, Any] = Field(default_factory=dict)
    symbols: Optional[List[str]] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    debug_api_payloads: bool = False

    @validator("name", "exchange", "settle_currency")
    def _strip_values(cls, value: str) -> str:
        if isinstance(value, str):
            value = value.strip()
        if not value:
            raise ValueError("Fields 'name', 'exchange', and 'settle_currency' must be non-empty strings.")
        return value


class AuthConfig(_BaseModel):
    """Settings for session authentication in the web dashboard."""

    secret_key: str
    users: Mapping[str, str]
    session_cookie_name: str = "risk_dashboard_session"
    https_only: bool = True

    @validator("secret_key", "session_cookie_name")
    def _require_strings(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("Authentication configuration requires non-empty strings.")
        return str(value)

    @validator("users")
    def _ensure_users(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        if not value:
            raise ValueError("Authentication configuration requires at least one user entry.")
        return {str(username): str(password) for username, password in value.items()}


class EmailSettings(_BaseModel):
    """SMTP configuration used to dispatch alert emails."""

    host: str
    port: int = 587
    username: Optional[str] = None
    password: Optional[str] = None
    use_tls: bool = True
    use_ssl: bool = False
    sender: Optional[str] = None

    @validator("host")
    def _strip_host(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("Email settings must include a non-empty 'host'.")
        return str(value).strip()


class GrafanaDashboardConfig(_BaseModel):
    """Description of a Grafana dashboard or panel to embed."""

    title: str
    url: str
    description: Optional[str] = None
    height: Optional[int] = None

    @validator("title", "url")
    def _require_non_empty(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("Grafana dashboard entries require a non-empty 'title' and 'url'.")
        return str(value).strip()

    @validator("height")
    def _validate_height(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("Grafana dashboard 'height' must be greater than zero when provided.")
        return value


class GrafanaConfig(_BaseModel):
    """Settings for embedding Grafana dashboards in the web UI."""

    dashboards: List[GrafanaDashboardConfig] = Field(default_factory=list)
    default_height: int = 600
    theme: str = "dark"
    base_url: Optional[str] = None
    account_equity_template: Optional[str] = None

    @validator("default_height")
    def _validate_default_height(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Grafana 'default_height' must be greater than zero.")
        return value


class AlertLimits(_BaseModel):
    wallet_exposure_pct: float = 0.6
    position_wallet_exposure_pct: float = 0.25
    max_drawdown_pct: float = 0.3
    loss_threshold_pct: float = -0.12


class NotificationSettings(_BaseModel):
    channels: List[str] = Field(default_factory=list)

    @validator("channels", each_item=True)
    def _strip_channel(cls, value: str) -> str:
        return str(value).strip()


class RealtimeConfig(_BaseModel):
    """Top level realtime configuration."""

    accounts: List[AccountConfig]
    alert_thresholds: AlertLimits = Field(default_factory=AlertLimits)
    notification_channels: List[str] = Field(default_factory=list)
    auth: Optional[AuthConfig] = None
    account_messages: Dict[str, str] = Field(default_factory=dict)
    custom_endpoints: Optional[CustomEndpointSettings] = None
    email: Optional[EmailSettings] = None
    config_root: Optional[Path] = None
    debug_api_payloads: bool = False
    reports_dir: Optional[Path] = None
    grafana: Optional[GrafanaConfig] = None
    api_keys_path: Optional[Path] = None
    config_path: Optional[Path] = None

    @root_validator
    def _require_accounts(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        accounts = values.get("accounts") or []
        if not accounts:
            raise ValueError("Realtime configuration must include at least one enabled account entry.")
        return values


__all__ = [
    "AccountConfig",
    "AlertLimits",
    "AuthConfig",
    "CustomEndpointSettings",
    "EmailSettings",
    "GrafanaConfig",
    "GrafanaDashboardConfig",
    "NotificationSettings",
    "RealtimeConfig",
]
