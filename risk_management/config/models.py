from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from services.telemetry import ResiliencePolicy


@dataclass()
class CustomEndpointSettings:
    """Settings controlling how custom endpoint overrides are loaded."""

    path: Optional[str] = None
    autodiscover: bool = True


@dataclass()
class AccountConfig:
    """Configuration for a single exchange account."""

    name: str
    exchange: str
    settle_currency: str = "USDT"
    api_key_id: Optional[str] = None
    credentials: Dict[str, Any] = field(default_factory=dict)
    symbols: Optional[List[str]] = None
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    debug_api_payloads: bool = False


@dataclass()
class AuthConfig:
    """Settings for session authentication in the web dashboard."""

    secret_key: str
    users: Mapping[str, str]
    session_cookie_name: str = "risk_dashboard_session"
    https_only: bool = True


@dataclass()
class EmailSettings:
    """SMTP configuration used to dispatch alert emails."""

    host: str
    port: int = 587
    username: Optional[str] = None
    password: Optional[str] = None
    use_tls: bool = True
    use_ssl: bool = False
    sender: Optional[str] = None


@dataclass()
class GrafanaDashboardConfig:
    """Description of a Grafana dashboard or panel to embed."""

    title: str
    url: str
    description: Optional[str] = None
    height: Optional[int] = None


@dataclass()
class GrafanaConfig:
    """Settings for embedding Grafana dashboards in the web UI."""

    dashboards: List[GrafanaDashboardConfig] = field(default_factory=list)
    default_height: int = 600
    theme: str = "dark"
    base_url: Optional[str] = None
    account_equity_template: Optional[str] = None


@dataclass()
class AlertLimits:
    wallet_exposure_pct: float = 0.6
    position_wallet_exposure_pct: float = 0.25
    max_drawdown_pct: float = 0.3
    loss_threshold_pct: float = -0.12


@dataclass()
class NotificationSettings:
    channels: List[str] = field(default_factory=list)


@dataclass()
class RealtimeConfig:
    """Top level realtime configuration."""

    accounts: List[AccountConfig]
    alert_thresholds: AlertLimits = field(default_factory=AlertLimits)
    notification_channels: List[str] = field(default_factory=list)
    auth: Optional[AuthConfig] = None
    account_messages: Dict[str, str] = field(default_factory=dict)
    custom_endpoints: Optional[CustomEndpointSettings] = None
    email: Optional[EmailSettings] = None
    resilience: ResiliencePolicy = field(default_factory=ResiliencePolicy)
    config_root: Optional[Path] = None
    debug_api_payloads: bool = False
    reports_dir: Optional[Path] = None
    grafana: Optional[GrafanaConfig] = None
    api_keys_path: Optional[Path] = None
    config_path: Optional[Path] = None


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
