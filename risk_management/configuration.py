"""Utilities for loading realtime risk management configuration files."""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
)


from .audit import (
    AuditSettings,
    AuditS3Settings,
    AuditSyslogSettings,
    DEFAULT_REDACT_FIELDS,
    get_audit_logger,
)

from .domain.models import Scenario, ScenarioShock



logger = logging.getLogger(__name__)


def _debug_to_logging_level(debug_level: int) -> int:
    """Translate Passivbot debug level values into logging module levels."""

    if debug_level <= 0:
        return logging.WARNING
    if debug_level == 1:
        return logging.INFO
    return logging.DEBUG


def _resolve_passivbot_logging_configurator() -> Optional[Callable[..., Any]]:
    """Return Passivbot's logging configurator when the package is available."""

    return _cached_passivbot_logging_configurator()


@lru_cache(maxsize=1)
def _cached_passivbot_logging_configurator() -> Optional[Callable[..., Any]]:
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
        configurator = _resolve_passivbot_logging_configurator()
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


def _parse_use_custom_endpoints(value: Any) -> Optional[bool]:
    """Return an optional boolean describing custom endpoint preferences."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "default", "auto", "inherit"}:
            return None
        if lowered in {"1", "true", "yes", "on", "enable", "enabled", "force"}:
            return True
        if lowered in {"0", "false", "no", "off", "disable", "disabled", "none"}:
            return False
        raise ValueError(
            "must be one of: auto, inherit, default, true, false, yes, no, on, off, enable, disable"
        )
    raise ValueError("custom endpoint preference must be a boolean or string value")


@dataclass()
class CustomEndpointSettings:
    """Settings controlling how custom endpoint overrides are loaded."""

    path: Optional[str] = None
    autodiscover: bool = True


@dataclass()
class LiquiditySettings:
    """Controls order-book collection and liquidity analytics."""

    fetch_order_book: bool = True
    depth: int = 25
    fallback_mode: str = "ticker"
    slippage_warning_pct: float = 0.02


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
    use_custom_endpoints: Optional[bool] = None

    counterparty_rating: Optional[str] = None
    exposure_limits: Dict[str, float] = field(default_factory=dict)

    liquidity: LiquiditySettings = field(default_factory=LiquiditySettings)



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


@dataclass()
class PolicyTriggerConfig:
    """Trigger definition describing when a policy should activate."""

    type: str
    metric: str
    operator: str
    value: float
    cooldown_seconds: Optional[int] = None
    lookback_seconds: Optional[int] = None


@dataclass()
class PolicyActionConfig:
    """Action executed when a policy fires."""

    type: str
    message: Optional[str] = None
    channels: List[str] = field(default_factory=list)
    severity: str = "info"
    requires_confirmation: bool = False
    confirmation_key: Optional[str] = None
    subject: Optional[str] = None


@dataclass()
class ManualOverrideConfig:
    """Describe how a policy can be manually overridden."""

    allowed: bool = False
    instructions: Optional[str] = None
    expires_after_seconds: Optional[int] = None


@dataclass()
class PolicyConfig:
    """Structured policy definition used by the realtime evaluator."""

    name: str
    trigger: PolicyTriggerConfig
    actions: List[PolicyActionConfig] = field(default_factory=list)
    description: Optional[str] = None
    manual_override: Optional[ManualOverrideConfig] = None


@dataclass()
class RealtimeConfig:
    """Top level realtime configuration."""

    accounts: List[AccountConfig]
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    notification_channels: List[str] = field(default_factory=list)
    auth: Optional[AuthConfig] = None
    account_messages: Dict[str, str] = field(default_factory=dict)
    scenarios: List[Scenario] = field(default_factory=list)
    custom_endpoints: Optional[CustomEndpointSettings] = None
    email: Optional[EmailSettings] = None
    config_root: Optional[Path] = None
    debug_api_payloads: bool = False
    reports_dir: Optional[Path] = None
    grafana: Optional[GrafanaConfig] = None
    liquidity: LiquiditySettings = field(default_factory=LiquiditySettings)

    policies: List[PolicyConfig] = field(default_factory=list)

    audit: Optional[AuditSettings] = None



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
    raise TypeError(
        f"{description} must be a JSON object, not {type(payload).__name__}."
    )


def _resolve_path_relative_to(base: Path, candidate: Any) -> Path:
    """Return an absolute path for ``candidate`` relative to ``base`` when required."""

    path = Path(str(candidate)).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()
    return path


def _parse_liquidity_settings(
    payload: Any,
    *,
    defaults: Optional[LiquiditySettings] = None,
    description: str = "Liquidity settings",
) -> LiquiditySettings:
    """Return merged :class:`LiquiditySettings` applying overrides from ``payload``."""

    settings = replace(defaults) if defaults is not None else LiquiditySettings()
    if payload is None:
        return settings
    if not isinstance(payload, Mapping):
        raise TypeError(f"{description} must be provided as an object")

    fetch_order_book = _coerce_bool(
        payload.get("fetch_order_book"), settings.fetch_order_book
    )
    settings.fetch_order_book = fetch_order_book

    depth_raw = payload.get("depth")
    if depth_raw is not None:
        try:
            depth = int(depth_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{description} depth must be a positive integer") from exc
        if depth <= 0:
            raise ValueError(f"{description} depth must be greater than zero")
        settings.depth = depth

    fallback_raw = payload.get("fallback_mode")
    if fallback_raw is None:
        fallback_raw = payload.get("fallback")
    if fallback_raw is not None:
        fallback_normalised = str(fallback_raw).strip().lower()
        fallback_aliases = {
            "ticker": "ticker",
            "tickers": "ticker",
            "book": "ticker",
            "order_book": "ticker",
            "mark": "mark",
            "mark_price": "mark",
            "none": "none",
            "off": "none",
            "disabled": "none",
        }
        if fallback_normalised not in fallback_aliases:
            raise ValueError(
                f"{description} fallback must be one of: ticker, mark, none"
            )
        settings.fallback_mode = fallback_aliases[fallback_normalised]

    warning_raw = payload.get("slippage_warning_pct")
    if warning_raw is not None:
        try:
            warning_value = float(warning_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{description} slippage_warning_pct must be a numeric value"
            ) from exc
        if warning_value <= 0:
            raise ValueError(
                f"{description} slippage_warning_pct must be greater than zero"
            )
        settings.slippage_warning_pct = warning_value

    return settings


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
            # ``exchange`` is metadata in api key files and should not be
            # treated as authentication input for ccxt clients.
            continue
        if key in {"headers", "options"} and isinstance(value, Mapping):
            existing = normalised.setdefault(key, {})
            existing.update(value)
            continue
        normalised[key] = value
    return normalised


def _merge_credentials(
    primary: Mapping[str, Any], secondary: Mapping[str, Any]
) -> Dict[str, Any]:
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


def _parse_custom_endpoints(settings: Any) -> Optional[CustomEndpointSettings]:
    """Return structured custom endpoint settings from ``settings``."""

    if settings is None:
        return None
    if isinstance(settings, Mapping):
        path_raw = settings.get("path")
        path = str(path_raw).strip() if path_raw not in (None, "") else None
        autodiscover = bool(settings.get("autodiscover", True))
        return CustomEndpointSettings(path=path or None, autodiscover=autodiscover)
    value = str(settings).strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"none", "off", "disable"}:
        return CustomEndpointSettings(path=None, autodiscover=False)
    return CustomEndpointSettings(path=value, autodiscover=False)


def _parse_email_settings(settings: Any) -> Optional[EmailSettings]:
    """Return SMTP settings when provided in the realtime configuration."""

    if settings is None:
        return None
    if not isinstance(settings, Mapping):
        raise TypeError(
            "Email settings must be provided as an object in the configuration file."
        )

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


def _parse_exposure_limits(value: Any) -> Dict[str, float]:
    """Normalise per-account exposure limits from configuration payloads."""

    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "Account 'exposure_limits' must be provided as an object mapping metric names to numeric limits."
        )

    limits: Dict[str, float] = {}
    for raw_key, raw_value in value.items():
        if raw_value in (None, ""):
            continue
        try:
            limits[str(raw_key)] = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Exposure limit '{raw_key}' must be a numeric value."
            ) from exc
    return limits


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
        raise TypeError(
            "Grafana 'dashboards' must be an array of dashboard definitions."
        )

    dashboards: List[GrafanaDashboardConfig] = []
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
                    str(description_raw).strip()
                    if description_raw not in (None, "")
                    else None
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

    return GrafanaConfig(
        dashboards=dashboards,
        default_height=default_height,
        theme=theme,
        base_url=base_url,
    )


def _parse_policy_trigger(
    payload: Mapping[str, Any], policy_name: str
) -> PolicyTriggerConfig:
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Policy '{policy_name}' trigger must be provided as an object."
        )

    trigger_type_raw = payload.get("type", "metric_threshold")
    trigger_type = str(trigger_type_raw).strip() or "metric_threshold"

    metric_raw = payload.get("metric")
    if metric_raw in (None, ""):
        raise ValueError(f"Policy '{policy_name}' trigger requires a 'metric'.")
    metric = str(metric_raw).strip()

    operator_raw = payload.get("operator", ">=")
    operator = str(operator_raw).strip()
    operator_aliases = {
        ">": ">",
        "gt": ">",
        ">=": ">=",
        "gte": ">=",
        "<": "<",
        "lt": "<",
        "<=": "<=",
        "lte": "<=",
        "==": "==",
        "=": "==",
        "eq": "==",
        "!=": "!=",
        "<>": "!=",
        "ne": "!=",
    }
    operator_normalised = operator_aliases.get(operator.lower(), operator)

    value_raw = payload.get("value", payload.get("threshold"))
    if value_raw in (None, ""):
        raise ValueError(f"Policy '{policy_name}' trigger requires a numeric 'value'.")
    try:
        value = float(value_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Policy '{policy_name}' trigger 'value' must be a number."
        ) from exc

    cooldown_raw = payload.get("cooldown_seconds")
    cooldown: Optional[int] = None
    if cooldown_raw not in (None, ""):
        try:
            cooldown = int(cooldown_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Policy '{policy_name}' trigger 'cooldown_seconds' must be an integer."
            ) from exc
        if cooldown < 0:
            raise ValueError(
                f"Policy '{policy_name}' trigger 'cooldown_seconds' must be >= 0."
            )

    lookback_raw = payload.get("lookback_seconds")
    lookback: Optional[int] = None
    if lookback_raw not in (None, ""):
        try:
            lookback = int(lookback_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Policy '{policy_name}' trigger 'lookback_seconds' must be an integer."
            ) from exc
        if lookback <= 0:
            raise ValueError(
                f"Policy '{policy_name}' trigger 'lookback_seconds' must be greater than zero."
            )

    return PolicyTriggerConfig(
        type=trigger_type,
        metric=metric,
        operator=operator_normalised,
        value=float(value),
        cooldown_seconds=cooldown,
        lookback_seconds=lookback,
    )


def _parse_policy_actions(
    payload: Any, policy_name: str
) -> List[PolicyActionConfig]:
    if payload in (None, []):
        return []
    if isinstance(payload, Mapping) or isinstance(payload, (str, bytes)):
        raise TypeError(
            f"Policy '{policy_name}' actions must be provided as an array of objects."
        )

    actions: List[PolicyActionConfig] = []
    for raw in payload:
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Policy '{policy_name}' action entries must be objects with action configuration fields."
            )
        action_type_raw = raw.get("type", "log")
        action_type = str(action_type_raw).strip().lower() or "log"

        message_raw = raw.get("message")
        message = str(message_raw) if message_raw not in (None, "") else None

        channels_raw = raw.get("channels", [])
        channels: List[str] = []
        if isinstance(channels_raw, str):
            if channels_raw.strip():
                channels = [channels_raw.strip()]
        elif isinstance(channels_raw, Iterable):
            for channel in channels_raw:
                if channel in (None, ""):
                    continue
                channels.append(str(channel).strip())
        elif channels_raw not in (None,):
            raise TypeError(
                f"Policy '{policy_name}' action 'channels' must be an array of strings."
            )

        severity_raw = raw.get("severity", "info")
        severity = str(severity_raw).strip().lower() or "info"

        confirmation_raw = raw.get("requires_confirmation")
        requires_confirmation = _coerce_bool(
            confirmation_raw,
            default=action_type == "require_confirmation",
        )

        confirmation_key_raw = raw.get("confirmation_key")
        confirmation_key = (
            str(confirmation_key_raw).strip()
            if confirmation_key_raw not in (None, "")
            else None
        )

        subject_raw = raw.get("subject")
        subject = str(subject_raw).strip() if subject_raw not in (None, "") else None

        actions.append(
            PolicyActionConfig(
                type=action_type,
                message=message,
                channels=channels,
                severity=severity,
                requires_confirmation=requires_confirmation,
                confirmation_key=confirmation_key,
                subject=subject,
            )
        )

    return actions


def _parse_policy_override(settings: Any, policy_name: str) -> Optional[ManualOverrideConfig]:
    if settings in (None, False, {}):
        return None
    if not isinstance(settings, Mapping):
        raise TypeError(
            f"Policy '{policy_name}' manual_override must be provided as an object."
        )

    allowed = _coerce_bool(settings.get("allowed"), False)
    instructions_raw = settings.get("instructions")
    instructions = (
        str(instructions_raw).strip() if instructions_raw not in (None, "") else None
    )

    expires_raw = settings.get("expires_after_seconds")
    expires: Optional[int] = None
    if expires_raw not in (None, ""):
        try:
            expires = int(expires_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Policy '{policy_name}' manual_override 'expires_after_seconds' must be an integer."
            ) from exc
        if expires <= 0:
            raise ValueError(
                f"Policy '{policy_name}' manual_override 'expires_after_seconds' must be greater than zero."
            )

    if not allowed and instructions is None and expires is None:
        return None

    return ManualOverrideConfig(
        allowed=allowed,
        instructions=instructions,
        expires_after_seconds=expires,
    )


def _parse_policies(payload: Any) -> List[PolicyConfig]:
    if payload in (None, []):
        return []
    if isinstance(payload, Mapping) or isinstance(payload, (str, bytes)):
        raise TypeError(
            "Realtime configuration 'policies' must be an array of policy objects."
        )

    policies: List[PolicyConfig] = []
    seen_names: Set[str] = set()
    for index, entry in enumerate(payload):
        if not isinstance(entry, Mapping):
            raise TypeError(
                "Realtime configuration policy entries must be objects with policy fields."
            )

        name_raw = entry.get("name") or f"Policy {index + 1}"
        name = str(name_raw).strip() or f"Policy {index + 1}"
        if name in seen_names:
            raise ValueError(f"Duplicate policy name '{name}' encountered in configuration.")
        seen_names.add(name)

        description_raw = entry.get("description")
        description = (
            str(description_raw).strip() if description_raw not in (None, "") else None
        )

        trigger_raw = entry.get("trigger")
        if not isinstance(trigger_raw, Mapping):
            raise TypeError(f"Policy '{name}' must include a trigger object.")
        trigger = _parse_policy_trigger(trigger_raw, name)

        actions = _parse_policy_actions(entry.get("actions"), name)
        manual_override = _parse_policy_override(entry.get("manual_override"), name)

        policies.append(
            PolicyConfig(
                name=name,
                description=description,
                trigger=trigger,
                actions=actions,
                manual_override=manual_override,
            )
        )

    return policies


def _parse_accounts(
    accounts_raw: Iterable[Mapping[str, Any]],
    api_keys: Optional[Mapping[str, Mapping[str, Any]]],
    debug_api_payloads_default: bool = False,
    *,
    liquidity_defaults: Optional[LiquiditySettings] = None,
    api_keys_source: Optional[str] = None,
) -> List[AccountConfig]:
    accounts: List[AccountConfig] = []
    debug_requested = False
    for raw in accounts_raw:
        if not isinstance(raw, Mapping):
            raise TypeError(
                "Account entries must be objects with account configuration fields."
            )
        if not raw.get("enabled", True):
            continue
        api_key_id = raw.get("api_key_id")
        credentials: Mapping[str, Any] = raw.get("credentials", {})
        exchange = raw.get("exchange")
        use_custom_endpoints_pref: Optional[bool] = None
        key_payload: Optional[Mapping[str, Any]] = None
        if api_key_id:
            if api_keys is None:
                raise ValueError(
                    f"Account '{raw.get('name')}' references api_key_id '{api_key_id}' but no api key file was provided"
                )
            if api_key_id not in api_keys:
                available_ids = ", ".join(sorted(api_keys)) if api_keys else "none"
                available_message = ""
                if api_keys_source:
                    available_message = f" Available api_key_id values in {api_keys_source}: {available_ids}."
                elif available_ids != "none":
                    available_message = f" Available api_key_id values: {available_ids}."
                raise ValueError(
                    f"Account '{raw.get('name')}' references unknown api_key_id '{api_key_id}'."
                    + available_message
                )
            key_payload = api_keys[api_key_id]
            if not exchange:
                exchange = key_payload.get("exchange")
            credentials = _merge_credentials(credentials, key_payload)
            key_pref_raw = key_payload.get("use_custom_endpoints") if isinstance(key_payload, Mapping) else None
            if key_pref_raw is not None:
                try:
                    key_pref = _parse_use_custom_endpoints(key_pref_raw)
                except ValueError as exc:
                    raise ValueError(
                        f"API key '{api_key_id}' has invalid 'use_custom_endpoints' value: {exc}"
                    ) from exc
                if key_pref is not None:
                    use_custom_endpoints_pref = key_pref
        else:
            credentials = _normalise_credentials(credentials)
        if not exchange:
            raise ValueError(
                f"Account '{raw.get('name')}' must specify an exchange either directly or via the api key entry."
            )
        debug_api_payloads = _coerce_bool(
            raw.get("debug_api_payloads"), debug_api_payloads_default
        )

        credentials_dict = dict(credentials)
        embedded_pref_raw = credentials_dict.pop("use_custom_endpoints", None)
        if embedded_pref_raw is not None:
            try:
                embedded_pref = _parse_use_custom_endpoints(embedded_pref_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Account '{raw.get('name')}' has invalid credential 'use_custom_endpoints' value: {exc}"
                ) from exc
            if embedded_pref is not None:
                use_custom_endpoints_pref = embedded_pref

        account_pref_raw = raw.get("use_custom_endpoints")
        if account_pref_raw is not None:
            try:
                account_pref = _parse_use_custom_endpoints(account_pref_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Account '{raw.get('name')}' has invalid 'use_custom_endpoints' value: {exc}"
                ) from exc
            use_custom_endpoints_pref = account_pref


        rating_raw = raw.get("counterparty_rating")
        rating = str(rating_raw).strip() if rating_raw not in (None, "") else None

        try:
            exposure_limits = _parse_exposure_limits(raw.get("exposure_limits"))
        except (TypeError, ValueError) as exc:
            raise type(exc)(
                f"Account '{raw.get('name')}' has invalid exposure limit configuration: {exc}"
            ) from exc

        liquidity_settings = _parse_liquidity_settings(
            raw.get("liquidity"),
            defaults=liquidity_defaults,
            description=f"Account '{raw.get('name', exchange)}' liquidity settings",
        )


        account = AccountConfig(
            name=str(raw.get("name", exchange)),
            exchange=str(exchange),
            settle_currency=str(raw.get("settle_currency", "USDT")),
            api_key_id=api_key_id,
            credentials=credentials_dict,
            symbols=list(raw.get("symbols") or []) or None,
            params=dict(raw.get("params", {})),
            enabled=bool(raw.get("enabled", True)),
            debug_api_payloads=debug_api_payloads,
            use_custom_endpoints=use_custom_endpoints_pref,

            counterparty_rating=rating,
            exposure_limits=exposure_limits,

            liquidity=liquidity_settings,

        )
        accounts.append(account)
        if debug_api_payloads:
            debug_requested = True
    if debug_requested:
        _ensure_debug_logging_enabled()
    return accounts


def _parse_auth(auth_raw: Optional[Mapping[str, Any]]) -> Optional[AuthConfig]:
    if not auth_raw:
        return None
    secret_key = auth_raw.get("secret_key")
    if not secret_key:
        raise ValueError("Authentication configuration requires a 'secret_key'.")
    users_raw = auth_raw.get("users")
    if not users_raw:
        raise ValueError(
            "Authentication configuration requires at least one user entry."
        )
    if isinstance(users_raw, Mapping):
        users = {
            str(username): str(password) for username, password in users_raw.items()
        }
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


def _parse_audit_settings(
    audit_raw: Optional[Mapping[str, Any]],
    *,
    base_dir: Path,
) -> Optional[AuditSettings]:
    if not audit_raw:
        return None
    mapping = _ensure_mapping(audit_raw, description="Realtime configuration 'audit'")
    enabled = _coerce_bool(mapping.get("enabled"), True)
    log_path_raw = mapping.get("log_path")
    if not log_path_raw:
        raise ValueError("Audit configuration requires a 'log_path'.")
    log_path = _resolve_path_relative_to(base_dir, log_path_raw)

    redact_fields_raw = mapping.get("redact_fields")
    redact_fields: Sequence[str]
    if redact_fields_raw is None:
        redact_fields = DEFAULT_REDACT_FIELDS
    else:
        if isinstance(redact_fields_raw, (str, bytes)):
            raise TypeError("Audit 'redact_fields' must be an iterable of field names.")
        if not isinstance(redact_fields_raw, Iterable):
            raise TypeError("Audit 'redact_fields' must be an iterable of field names.")
        processed: list[str] = []
        for field in redact_fields_raw:
            if field is None:
                continue
            field_str = str(field).strip()
            if field_str:
                processed.append(field_str)
        redact_fields = tuple(processed) if processed else DEFAULT_REDACT_FIELDS

    s3_settings: Optional[AuditS3Settings] = None
    s3_raw = mapping.get("s3")
    if s3_raw:
        s3_mapping = _ensure_mapping(s3_raw, description="Realtime configuration 'audit.s3'")
        bucket = s3_mapping.get("bucket")
        if not bucket:
            raise ValueError("Audit 's3' configuration requires a 'bucket'.")
        prefix = str(s3_mapping.get("prefix", ""))
        region_raw = s3_mapping.get("region_name")
        profile_raw = s3_mapping.get("profile_name")
        s3_settings = AuditS3Settings(
            bucket=str(bucket),
            prefix=prefix,
            region_name=str(region_raw) if region_raw else None,
            profile_name=str(profile_raw) if profile_raw else None,
        )

    syslog_settings: Optional[AuditSyslogSettings] = None
    syslog_raw = mapping.get("syslog")
    if syslog_raw:
        syslog_mapping = _ensure_mapping(
            syslog_raw, description="Realtime configuration 'audit.syslog'"
        )
        address = str(syslog_mapping.get("address", "localhost"))
        port_raw = syslog_mapping.get("port", 514)
        try:
            port = int(port_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Audit 'syslog.port' must be an integer") from exc
        facility = str(syslog_mapping.get("facility", "user"))
        syslog_settings = AuditSyslogSettings(address=address, port=port, facility=facility)

    return AuditSettings(
        log_path=log_path,
        enabled=enabled,
        redact_fields=redact_fields,
        s3=s3_settings,
        syslog=syslog_settings,
    )

def _parse_scenarios(payload: Any) -> List[Scenario]:
    if not payload:
        return []

    if isinstance(payload, Mapping):
        iterable = payload.values()
    elif isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        iterable = payload
    else:
        raise TypeError(
            "Realtime configuration 'scenarios' must be an iterable of scenario definitions.",
        )

    scenarios: List[Scenario] = []
    for raw in iterable:
        if not isinstance(raw, Mapping):
            raise TypeError("Scenario definitions must be JSON objects.")
        scenario_id = raw.get("id")
        name = raw.get("name") or scenario_id
        if not name:
            raise ValueError("Scenario definitions must include a 'name' or 'id'.")
        description_raw = raw.get("description")
        description = (
            str(description_raw).strip()
            if isinstance(description_raw, str) and description_raw.strip()
            else None
        )
        shocks_raw = raw.get("shocks")
        shocks = _parse_shock_definitions(shocks_raw, str(name))
        scenarios.append(
            Scenario(
                id=str(scenario_id) if scenario_id else None,
                name=str(name),
                description=description,
                shocks=tuple(shocks),
            )
        )
    return scenarios


def _parse_shock_definitions(payload: Any, scenario_name: str) -> List[ScenarioShock]:
    if not payload:
        raise ValueError(f"Scenario '{scenario_name}' must include at least one shock definition.")

    shocks: List[ScenarioShock] = []

    if isinstance(payload, Mapping):
        items = payload.items()
        for symbol, value in items:
            pct = _coerce_float(value, description=f"Scenario '{scenario_name}' shock for '{symbol}'")
            shocks.append(ScenarioShock(symbol=str(symbol), price_pct=pct))
    elif isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        for raw in payload:
            if not isinstance(raw, Mapping):
                raise TypeError(
                    f"Scenario '{scenario_name}' shock entries must be JSON objects.",
                )
            symbol = raw.get("symbol") or raw.get("pair") or raw.get("ticker")
            if not symbol:
                raise ValueError(
                    f"Scenario '{scenario_name}' shock entries must include a 'symbol'.",
                )
            pct_raw = (
                raw.get("price_pct")
                if raw.get("price_pct") is not None
                else raw.get("pct")
                if raw.get("pct") is not None
                else raw.get("percent")
            )
            pct = _coerce_float(
                pct_raw,
                description=f"Scenario '{scenario_name}' shock for '{symbol}'",
            )
            shocks.append(ScenarioShock(symbol=str(symbol), price_pct=pct))
    else:
        raise TypeError(
            f"Scenario '{scenario_name}' shocks must be provided as a mapping or list of definitions.",
        )

    if not shocks:
        raise ValueError(
            f"Scenario '{scenario_name}' must include at least one shock definition.",
        )

    return shocks


def _coerce_float(value: Any, *, description: str) -> float:
    if value is None:
        raise ValueError(f"{description} must be a number.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be a number.") from exc



def load_realtime_config(path: Path | str) -> RealtimeConfig:
    """Load a realtime configuration file.

    Parameters
    ----------
    path:
        Absolute or relative path to the realtime configuration JSON file. ``path``
        may be provided as either a :class:`pathlib.Path` instance or a string.

    Returns
    -------
    RealtimeConfig
        Structured configuration dataclass consumed by the realtime dashboard
        and supporting utilities.

    Raises
    ------
    FileNotFoundError
        Raised when the configuration file or any referenced api key files
        cannot be located.
    ValueError
        Raised when the configuration payload is incomplete or invalid.
    TypeError
        Raised when sections of the configuration are provided in unexpected
        formats.
    """

    _configure_default_logging(debug_level=1)

    path = Path(path)

    config_payload = _load_json(path)
    config = _ensure_mapping(config_payload, description="Realtime configuration")
    config_root = path.parent.resolve()
    api_keys_file = config.get("api_keys_file")
    api_keys: Optional[Dict[str, Mapping[str, Any]]] = None
    api_keys_path: Optional[Path] = None
    if api_keys_file:
        api_keys_path = _resolve_path_relative_to(path.parent, api_keys_file)
    else:
        api_keys_path = _discover_api_keys_path(config_root)
        if api_keys_path:
            logger.info("Using api keys from %s", api_keys_path)
    if api_keys_path:
        api_keys_payload = _load_json(api_keys_path)
        api_keys_raw = _ensure_mapping(
            api_keys_payload, description="API key configuration"
        )
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
        raise ValueError(
            "Realtime configuration must include at least one account entry."
        )
    if isinstance(accounts_raw, Mapping) or isinstance(accounts_raw, (str, bytes)):
        raise TypeError(
            "Realtime configuration 'accounts' must be an iterable of account definition objects."
        )
    debug_api_payloads_default = _coerce_bool(config.get("debug_api_payloads"), False)
    if debug_api_payloads_default:
        _ensure_debug_logging_enabled()

    liquidity_defaults = _parse_liquidity_settings(
        config.get("liquidity"),
        description="Realtime liquidity settings",
    )

    accounts = _parse_accounts(
        accounts_raw,
        api_keys,
        debug_api_payloads_default,
        liquidity_defaults=liquidity_defaults,
        api_keys_source=str(api_keys_path) if api_keys_path else None,
    )
    alert_thresholds = {
        str(k): float(v) for k, v in config.get("alert_thresholds", {}).items()
    }
    notification_channels = [
        str(item) for item in config.get("notification_channels", [])
    ]
    auth = _parse_auth(config.get("auth"))
    custom_endpoints = _parse_custom_endpoints(config.get("custom_endpoints"))
    email_settings = _parse_email_settings(config.get("email"))
    grafana_settings = _parse_grafana_config(config.get("grafana"))
    policies = _parse_policies(config.get("policies"))
    reports_dir_value = config.get("reports_dir")
    reports_dir: Optional[Path] = None
    if reports_dir_value:
        reports_dir = _resolve_path_relative_to(path.parent, reports_dir_value)

    if custom_endpoints and custom_endpoints.path:
        resolved_path = _resolve_path_relative_to(path.parent, custom_endpoints.path)
        custom_endpoints = CustomEndpointSettings(
            path=str(resolved_path),
            autodiscover=custom_endpoints.autodiscover,
        )

    account_messages_payload = config.get("account_messages", {})
    account_messages: Dict[str, str] = {}
    if account_messages_payload:
        messages_mapping = _ensure_mapping(
            account_messages_payload,
            description="Realtime configuration 'account_messages'",
        )
        for name, message in messages_mapping.items():
            if message is None:
                continue
            account_messages[str(name)] = str(message)


    audit_settings = _parse_audit_settings(config.get("audit"), base_dir=path.parent)
    audit_logger = get_audit_logger(audit_settings)
    if audit_logger:
        try:
            audit_logger.log(
                action="config.load",
                actor="system",
                details={
                    "path": str(path),
                    "accounts": [account.name for account in accounts],
                    "notification_channels": list(notification_channels),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to emit configuration audit entry: %s", exc)

    scenarios = _parse_scenarios(config.get("scenarios"))


    return RealtimeConfig(
        accounts=accounts,
        alert_thresholds=alert_thresholds,
        notification_channels=notification_channels,
        auth=auth,
        custom_endpoints=custom_endpoints,
        email=email_settings,
        config_root=config_root,
        debug_api_payloads=debug_api_payloads_default,
        reports_dir=reports_dir,
        grafana=grafana_settings,
        account_messages=account_messages,

        liquidity=liquidity_defaults,


        policies=policies,

        audit=audit_settings,

        scenarios=scenarios,


    )
