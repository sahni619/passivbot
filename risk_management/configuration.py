"""Utilities for loading realtime risk management configuration files."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Set

from risk_management.config.models import (
    AccountConfig,
    AlertLimits,
    AuthConfig,
    CustomEndpointSettings,
    EmailSettings,
    GrafanaConfig,
    GrafanaDashboardConfig,
    NotificationSettings,
    RealtimeConfig,
)

from services.telemetry import ResiliencePolicy


@lru_cache(maxsize=1)
def _passivbot_logging_configurator() -> Optional[Callable[..., Any]]:
    """Return Passivbot's logging configurator when the package is available."""

    try:
        import importlib
        import importlib.util
    except Exception:  # pragma: no cover - defensive guard
        return None

    spec = importlib.util.find_spec("logging_setup")
    if spec is None:  # pragma: no cover - Passivbot package missing in unit tests
        return None
    module = importlib.import_module("logging_setup")
    configurator = getattr(module, "configure_logging", None)
    if not callable(configurator):  # pragma: no cover - defensive guard
        return None
    return configurator



def _ensure_logger_level(logger: logging.Logger, level: int) -> None:
    """Ensure ``logger`` and its handlers are set to at most ``level``."""

    if logger.level in {logging.NOTSET} or logger.level > level:
        logger.setLevel(level)
    for handler in logger.handlers:
        if handler.level in {logging.NOTSET} or handler.level > level:
            handler.setLevel(level)


def _configure_default_logging(debug_level: int = 1) -> bool:
    """Provision Passivbot-style logging and enforce sensible defaults."""

    root_logger = logging.getLogger()
    already_configured = bool(root_logger.handlers)

    if not already_configured:
        configurator = _passivbot_logging_configurator()
        if configurator is not None:
            configurator(debug=debug_level)
        else:
            logging.basicConfig(level=_debug_to_logging_level(debug_level))

    desired_level = _debug_to_logging_level(debug_level)
    _ensure_logger_level(root_logger, desired_level)
    risk_logger = logging.getLogger("risk_management")
    _ensure_logger_level(risk_logger, desired_level)

    return not already_configured


def _ensure_debug_logging_enabled() -> None:
    """Raise logging verbosity when debug API payloads are requested."""

    _configure_default_logging(debug_level=2)

    root_logger = logging.getLogger()
    risk_logger = logging.getLogger("risk_management")
    _ensure_logger_level(root_logger, logging.DEBUG)
    _ensure_logger_level(risk_logger, logging.DEBUG)


def _debug_to_logging_level(debug_level: int) -> int:
    """Map a debug verbosity integer to a logging level."""

    if debug_level <= 0:
        return logging.WARNING
    if debug_level == 1:
        return logging.INFO
    return logging.DEBUG



def _load_json(path: Path) -> Dict[str, Any]:
    """Return parsed JSON payload from ``path`` with helpful error messages."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in configuration file {path}: {exc}") from exc


def _ensure_mapping(payload: Any, *, description: str) -> MutableMapping[str, Any]:
    """Return ``payload`` when it is a mapping, otherwise raise ``TypeError``."""

    if isinstance(payload, MutableMapping):
        return payload
    if isinstance(payload, Mapping):
        return dict(payload)
    raise TypeError(f"{description} must be a JSON object, not {type(payload).__name__}.")


def _resolve_path_relative_to(base: Path, candidate: Any) -> Path:
    """Return an absolute path for ``candidate`` relative to ``base`` when required."""

    path = Path(str(candidate)).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()
    return path


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Return a boolean for ``value`` supporting common string representations."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "default", "auto"}:
            return default
        if lowered in {"1", "true", "yes", "on", "enabled", "enable"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled", "disable"}:
            return False
    return bool(value)


def _normalise_credentials(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise credential keys to ccxt's expected names."""

    key_aliases = {
        "key": "apiKey",
        "apikey": "apiKey",
        "api_key": "apiKey",
        "api-key": "apiKey",
        "secret": "secret",
        "secret_key": "secret",
        "secretkey": "secret",
        "secret-key": "secret",
        "apisecret": "secret",
        "api_secret": "secret",
        "api-secret": "secret",
        "password": "password",
        "passphrase": "password",
        "pass_phrase": "password",
        "pass-phrase": "password",
        "uid": "uid",
        "user_id": "uid",
        "wallet_address": "walletAddress",
        "walletaddress": "walletAddress",
        "private_key": "privateKey",
        "privatekey": "privateKey",
        "ccxt_config": "ccxt",
        "ccxtconfig": "ccxt",
    }
    normalised: Dict[str, Any] = {}
    for raw_key, value in data.items():
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                continue
        key_lookup = raw_key.lower().replace(" ", "").replace("-", "_")
        key = key_aliases.get(key_lookup, raw_key)
        if key == "exchange":
            continue
        if key in {"headers", "options"} and isinstance(value, Mapping):
            existing = normalised.setdefault(key, {})
            existing.update(value)
            continue
        normalised[key] = value
    return normalised


def _merge_credentials(primary: Mapping[str, Any], secondary: Mapping[str, Any]) -> Dict[str, Any]:
    merged = _normalise_credentials(secondary)
    primary_normalised = _normalise_credentials(primary)
    for key, value in primary_normalised.items():
        if key in {"headers", "options"} and isinstance(value, Mapping):
            existing = merged.setdefault(key, {})
            existing.update(value)
        else:
            merged[key] = value
    return merged


def _iter_candidate_roots(config_root: Optional[Path]) -> Iterable[Path]:
    """Yield directories to inspect when auto-discovering shared files."""

    module_root = Path(__file__).resolve().parent
    repository_root = module_root.parent

    bases = [config_root, Path.cwd(), module_root, repository_root]

    seen: Set[Path] = set()
    for base in bases:
        if base is None:
            continue
        try:
            resolved = base.resolve()
        except FileNotFoundError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved
        for parent in resolved.parents:
            if parent in seen:
                continue
            seen.add(parent)
            yield parent


def _discover_api_keys_path(config_root: Optional[Path]) -> Optional[Path]:
    """Return the first ``api-keys.json`` found relative to common roots."""

    for root in _iter_candidate_roots(config_root):
        candidate = root / "api-keys.json"
        if candidate.is_file():
            return candidate
    return None


def _parse_custom_endpoints(settings: Any, *, base_dir: Path) -> Optional[CustomEndpointSettings]:
    """Return structured custom endpoint settings from ``settings``."""

    if settings is None:
        return None
    if isinstance(settings, Mapping):
        path_raw = settings.get("path")
        path = str(path_raw).strip() if path_raw not in (None, "") else None
        autodiscover = bool(settings.get("autodiscover", True))
        if path:
            path = str(_resolve_path_relative_to(base_dir, path))
        return CustomEndpointSettings(path=path or None, autodiscover=autodiscover)
    value = str(settings).strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"none", "off", "disable"}:
        return CustomEndpointSettings(path=None, autodiscover=False)
    resolved = str(_resolve_path_relative_to(base_dir, value))
    return CustomEndpointSettings(path=resolved, autodiscover=False)


def _parse_email_settings(settings: Any) -> Optional[EmailSettings]:
    """Return SMTP settings when provided in the realtime configuration."""

    if settings is None:
        return None
    if not isinstance(settings, Mapping):
        raise TypeError("Email settings must be provided as an object in the configuration file.")

    host_raw = settings.get("host")
    if not host_raw or not str(host_raw).strip():
        raise ValueError("Email settings must include a non-empty 'host'.")
    host = str(host_raw).strip()

    port_raw = settings.get("port", 587)
    try:
        port = int(port_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Email settings 'port' must be an integer.") from exc

    username = settings.get("username")
    password = settings.get("password")
    sender = settings.get("sender")
    use_tls = _coerce_bool(settings.get("use_tls"), True)
    use_ssl = _coerce_bool(settings.get("use_ssl"), False)

    return EmailSettings(
        host=host,
        port=port,
        username=str(username).strip() if username not in (None, "") else None,
        password=str(password).strip() if password not in (None, "") else None,
        sender=str(sender).strip() if sender not in (None, "") else None,
        use_tls=use_tls,
        use_ssl=use_ssl,
    )


def _parse_grafana_config(settings: Any) -> Optional[GrafanaConfig]:
    """Return Grafana embedding settings from ``settings``."""

    if settings is None:
        return None
    if not isinstance(settings, Mapping):
        raise TypeError(
            "Grafana settings must be provided as an object in the configuration file."
        )

    dashboards_raw = settings.get("dashboards")
    if dashboards_raw in (None, []):
        return None
    if not isinstance(dashboards_raw, Iterable):
        raise TypeError("Grafana 'dashboards' must be an array of dashboard definitions.")

    dashboards: list[GrafanaDashboardConfig] = []
    for entry in dashboards_raw:
        if not isinstance(entry, Mapping):
            raise TypeError(
                "Each Grafana dashboard entry must be an object with at least a title and url."
            )
        url_raw = entry.get("url")
        if not url_raw or not str(url_raw).strip():
            raise ValueError("Grafana dashboard entries require a non-empty 'url'.")
        title_raw = entry.get("title", "Grafana dashboard")
        description_raw = entry.get("description")
        height_raw = entry.get("height")

        height: Optional[int] = None
        if height_raw not in (None, ""):
            try:
                height = int(height_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Grafana dashboard 'height' must be an integer when provided."
                ) from exc
            if height <= 0:
                raise ValueError(
                    "Grafana dashboard 'height' must be greater than zero when provided."
                )

        dashboards.append(
            GrafanaDashboardConfig(
                title=str(title_raw).strip() or "Grafana dashboard",
                url=str(url_raw).strip(),
                description=(
                    str(description_raw).strip() if description_raw not in (None, "") else None
                ),
                height=height,
            )
        )

    default_height_raw = settings.get("default_height", 600)
    try:
        default_height = int(default_height_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Grafana 'default_height' must be an integer.") from exc
    if default_height <= 0:
        raise ValueError("Grafana 'default_height' must be greater than zero.")

    theme_raw = settings.get("theme", "dark")
    theme = str(theme_raw).strip() or "dark"

    base_url_raw = settings.get("base_url")
    base_url = str(base_url_raw).strip() if base_url_raw not in (None, "") else None
    account_equity_template_raw = settings.get("account_equity_template")
    account_equity_template = (
        str(account_equity_template_raw).strip()
        if account_equity_template_raw not in (None, "")
        else None
    )

    return GrafanaConfig(
        dashboards=dashboards,
        default_height=default_height,
        theme=theme,
        base_url=base_url,
        account_equity_template=account_equity_template,
    )


def _parse_accounts(
    accounts_raw: Iterable[Mapping[str, Any]],
    api_keys: Optional[Mapping[str, Mapping[str, Any]]],
    debug_api_payloads_default: bool = False,
) -> list[AccountConfig]:
    accounts: list[AccountConfig] = []
    for raw in accounts_raw:
        if not isinstance(raw, Mapping):
            raise TypeError("Account entries must be objects with account configuration fields.")
        if not raw.get("enabled", True):
            continue
        api_key_id = raw.get("api_key_id")
        credentials: Mapping[str, Any] = raw.get("credentials", {})
        exchange = raw.get("exchange")
        if api_key_id:
            if api_keys is None:
                raise ValueError(
                    f"Account '{raw.get('name')}' references api_key_id '{api_key_id}' but no api key file was provided"
                )
            if api_key_id not in api_keys:
                raise ValueError(
                    f"Account '{raw.get('name')}' references unknown api_key_id '{api_key_id}'"
                )
            key_payload = api_keys[api_key_id]
            if not exchange:
                exchange = key_payload.get("exchange")
            credentials = _merge_credentials(credentials, key_payload)
        else:
            credentials = _normalise_credentials(credentials)
        if not exchange:
            raise ValueError(
                f"Account '{raw.get('name')}' must specify an exchange either directly or via the api key entry."
            )
        debug_api_payloads = _coerce_bool(raw.get("debug_api_payloads"), debug_api_payloads_default)

        account = AccountConfig(
            name=str(raw.get("name", exchange)),
            exchange=str(exchange),
            settle_currency=str(raw.get("settle_currency", "USDT")),
            api_key_id=api_key_id,
            credentials=dict(credentials),
            symbols=list(raw.get("symbols") or []) or None,
            params=dict(raw.get("params", {})),
            enabled=bool(raw.get("enabled", True)),
            debug_api_payloads=debug_api_payloads,
        )
        accounts.append(account)
    if not accounts:
        raise ValueError("Realtime configuration must include at least one enabled account entry.")
    return accounts


def _parse_auth(auth_raw: Optional[Mapping[str, Any]]) -> Optional[AuthConfig]:
    if not auth_raw:
        return None
    secret_key = auth_raw.get("secret_key")
    if not secret_key:
        raise ValueError("Authentication configuration requires a 'secret_key'.")
    users_raw = auth_raw.get("users")
    if not users_raw:
        raise ValueError("Authentication configuration requires at least one user entry.")
    if isinstance(users_raw, Mapping):
        users = {str(username): str(password) for username, password in users_raw.items()}
    else:
        users = {}
        for entry in users_raw:
            if not isinstance(entry, Mapping):
                raise TypeError(
                    "Authentication 'users' entries must be objects with 'username' and 'password_hash'."
                )
            username = entry.get("username")
            password_hash = entry.get("password_hash")
            if not username or not password_hash:
                raise ValueError(
                    "Authentication 'users' entries must include both 'username' and 'password_hash'."
                )
            users[str(username)] = str(password_hash)
    session_cookie = str(auth_raw.get("session_cookie_name", "risk_dashboard_session"))
    https_only = _coerce_bool(auth_raw.get("https_only"), True)
    return AuthConfig(
        secret_key=str(secret_key),
        users=users,
        session_cookie_name=session_cookie,
        https_only=https_only,
    )


def load_realtime_payload(path: Path | str) -> tuple[MutableMapping[str, Any], Path]:
    """Load and return the raw realtime configuration mapping from disk."""

    path = Path(path).expanduser().resolve()
    payload = _load_json(path)
    return _ensure_mapping(payload, description="Realtime configuration"), path


def validate_realtime_config(
    config: Mapping[str, Any], *, source_path: Optional[Path] = None
) -> RealtimeConfig:
    """Validate and normalise a realtime configuration payload."""

    base_dir = source_path.parent.resolve() if source_path else Path.cwd()
    config_root = base_dir

    api_keys_file = config.get("api_keys_file")
    api_keys: Optional[Dict[str, Mapping[str, Any]]] = None
    api_keys_path: Optional[Path] = None
    if api_keys_file:
        api_keys_path = _resolve_path_relative_to(base_dir, api_keys_file)
    else:
        api_keys_path = _discover_api_keys_path(config_root)
    if api_keys_path:
        api_keys_payload = _load_json(api_keys_path)
        api_keys_raw = _ensure_mapping(api_keys_payload, description="API key configuration")
        flattened: Dict[str, Mapping[str, Any]] = {}
        for key, value in api_keys_raw.items():
            if key == "referrals" or not isinstance(value, Mapping):
                continue
            if key.lower() == "users":
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, Mapping):
                        flattened[sub_key] = sub_value
                continue
            flattened[key] = value
        api_keys = flattened

    accounts_raw = config.get("accounts")
    if not accounts_raw:
        raise ValueError("Realtime configuration must include at least one account entry.")
    if isinstance(accounts_raw, Mapping) or isinstance(accounts_raw, (str, bytes)):
        raise TypeError(
            "Realtime configuration 'accounts' must be an iterable of account definition objects."
        )
    debug_api_payloads_default = _coerce_bool(config.get("debug_api_payloads"), False)

    accounts = _parse_accounts(accounts_raw, api_keys, debug_api_payloads_default)

    alert_thresholds_raw = config.get("alert_thresholds") or {}
    if not isinstance(alert_thresholds_raw, Mapping):
        raise TypeError(
            "Realtime configuration 'alert_thresholds' must be an object when provided."
        )
    alert_thresholds = AlertLimits(**alert_thresholds_raw)

    notification_channels_raw = config.get("notification_channels", [])
    if notification_channels_raw is None:
        notification_channels_raw = []
    if isinstance(notification_channels_raw, Mapping) or isinstance(
        notification_channels_raw, (str, bytes)
    ):
        raise TypeError(
            "Realtime configuration 'notification_channels' must be an array of channel identifiers."
        )
    notification_settings = NotificationSettings(
        channels=[str(item) for item in notification_channels_raw]
    )

    auth = _parse_auth(config.get("auth"))
    custom_endpoints = _parse_custom_endpoints(config.get("custom_endpoints"), base_dir=base_dir)
    email_settings = _parse_email_settings(config.get("email"))
    resilience = ResiliencePolicy.from_mapping(config.get("resilience"))
    grafana_settings = _parse_grafana_config(config.get("grafana"))

    reports_dir_value = config.get("reports_dir")
    reports_dir: Optional[Path] = None
    if reports_dir_value:
        reports_dir = _resolve_path_relative_to(base_dir, reports_dir_value)

    account_messages_payload = config.get("account_messages", {})
    account_messages: Dict[str, str] = {}
    if account_messages_payload:
        messages_mapping = _ensure_mapping(
            account_messages_payload, description="Realtime configuration 'account_messages'"
        )
        for name, message in messages_mapping.items():
            if message is None:
                continue
            account_messages[str(name)] = str(message)

    return RealtimeConfig(
        accounts=accounts,
        alert_thresholds=alert_thresholds,
        notification_channels=notification_settings.channels,
        auth=auth,
        custom_endpoints=custom_endpoints,
        email=email_settings,
        resilience=resilience,
        config_root=config_root,
        debug_api_payloads=debug_api_payloads_default,
        reports_dir=reports_dir,
        grafana=grafana_settings,
        account_messages=account_messages,
        api_keys_path=api_keys_path,
        config_path=source_path,
    )


def load_realtime_config(path: Path | str) -> RealtimeConfig:
    """Load and validate a realtime configuration file from disk."""

    payload, resolved_path = load_realtime_payload(path)
    return validate_realtime_config(payload, source_path=resolved_path)


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
    "load_realtime_config",
    "load_realtime_payload",
    "validate_realtime_config",
    "_merge_credentials",
    "_normalise_credentials",
]
