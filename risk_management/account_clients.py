"""Account client implementations for the risk management tooling."""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import math
import statistics
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency in some envs
    import ccxt.async_support as ccxt_async
    from ccxt.base.errors import AuthenticationError, BadRequest, BaseError, NotSupported
except ModuleNotFoundError:  # pragma: no cover - allow tests without ccxt
    ccxt_async = None  # type: ignore[assignment]

    class BaseError(Exception):
        """Fallback error when ccxt is unavailable."""
        pass

    class BadRequest(BaseError):
        """Fallback error matching ccxt BadRequest when unavailable."""
        pass

    class AuthenticationError(BaseError):
        """Fallback authentication error when ccxt is unavailable."""
        pass

    class NotSupported(BaseError):
        """Fallback not supported error when ccxt is unavailable."""
        pass

from custom_endpoint_overrides import (
    ResolvedEndpointOverride,
    apply_rest_overrides_to_ccxt,
    get_custom_endpoint_source,
    resolve_custom_endpoint_override,
)

from risk_management.configuration import AccountConfig
from ._utils import (
    coerce_float as _coerce_float,
    coerce_int as _coerce_int,
    extract_position_details as _extract_position_details,
    first_float as _first_float,
    stringify_payload as _stringify_payload,
)
from ._parsing import (
    extract_balance as _extract_balance,
    parse_order as _parse_order,
    parse_position as _parse_position,
)
from .realized_pnl import fetch_realized_pnl_history  # noqa: F401 (kept for external usage)
from .liquidity import calculate_position_liquidity, normalise_order_book

try:  # pragma: no cover - passivbot is optional when running tests
    from passivbot.utils import load_ccxt_instance, normalize_exchange_name
except (ModuleNotFoundError, ImportError):  # pragma: no cover - allow running without passivbot
    load_ccxt_instance = None  # type: ignore[assignment]

    def normalize_exchange_name(exchange: str) -> str:  # type: ignore[override]
        return exchange


logger = logging.getLogger(__name__)


class AccountClientProtocol(abc.ABC):
    """Abstract interface for realtime account clients."""

    config: "AccountConfig"

    @abc.abstractmethod
    async def fetch(self) -> Dict[str, Any]:
        """Return a mapping with account balance, positions, and metadata."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Close any open network connections."""

    @abc.abstractmethod
    async def kill_switch(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Cancel open orders and close positions for the account.

        When ``symbol`` is provided, only orders and positions for that market are
        touched. Otherwise every open position is targeted.
        """

    @abc.abstractmethod
    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Place an order on the underlying exchange."""

    @abc.abstractmethod
    async def cancel_order(
        self, order_id: str, symbol: Optional[str] = None, params: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        """Cancel an existing order."""

    @abc.abstractmethod
    async def close_position(self, symbol: str) -> Mapping[str, Any]:
        """Close the open position for ``symbol`` if one exists."""

    @abc.abstractmethod
    async def list_order_types(self) -> Sequence[str]:
        """Return supported order types for the account exchange."""

    @abc.abstractmethod
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Mapping[str, Any]:
        """Cancel every open order for the account, optionally filtered by symbol."""

    @abc.abstractmethod
    async def close_all_positions(self, symbol: Optional[str] = None) -> Mapping[str, Any]:
        """Close all open positions for the account, optionally filtered by symbol."""


def _set_exchange_field(client: Any, key: str, value: Any, aliases: Sequence[str]) -> None:
    """Assign ``value`` to ``key`` on ``client`` and store aliases when possible."""
    config = getattr(client, "config", None)
    keys = tuple(dict.fromkeys((key, *aliases)))  # preserve order, drop duplicates
    for attr in keys:
        try:
            setattr(client, attr, value)
        except Exception:
            logger.debug("Ignored unsupported credential attribute %s", attr)
        if isinstance(config, MutableMapping):
            try:
                config[attr] = value
            except Exception:
                logger.debug("Failed to persist credential %s in exchange config", attr)


def _format_header_placeholders(
    headers: MutableMapping[str, Any], values: Mapping[str, Any]
) -> Optional[Mapping[str, Any]]:
    """Expand placeholder tokens in ``headers`` using ``values`` as the source."""

    class _DefaultDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    alias_map = {
        "apiKey": ("api_key", "key"),
        "secret": ("apiSecret", "secret_key", "secretKey"),
        "password": ("passphrase", "pass_phrase"),
        "uid": ("user_id",),
        "walletAddress": ("wallet_address",),
        "privateKey": ("private_key",),
    }

    substitutions: Dict[str, str] = {}

    for key, value in values.items():
        if isinstance(value, (str, int, float, bool)):
            substitutions[key] = str(value)

    for canonical, aliases in alias_map.items():
        canonical_value = substitutions.get(canonical)
        if canonical_value is None:
            continue
        for alias in aliases:
            substitutions.setdefault(alias, canonical_value)

    formatter = _DefaultDict(substitutions)

    updated = False
    for header_key, header_value in list(headers.items()):
        if isinstance(header_value, str) and "{" in header_value and "}" in header_value:
            try:
                formatted = header_value.format_map(formatter)
            except Exception:  # pragma: no cover - defensive against malformed format strings
                continue
            if formatted != header_value:
                headers[header_key] = formatted
                updated = True

    if updated:
        # ``headers`` may be an exchange-specific structure; normalise to ``dict`` to
        # avoid subtle mutation bugs when ccxt clones the mapping.
        return dict(headers)

    return None


def _apply_credentials(client: Any, credentials: Mapping[str, Any]) -> None:
    """Populate authentication fields on a ccxt client."""
    sensitive_fields = {"apiKey", "secret", "password", "uid", "login", "walletAddress", "privateKey"}
    alias_map = {
        "apiKey": ("api_key", "key"),
        "secret": ("apiSecret", "secret_key", "secretKey"),
        "password": ("passphrase",),
        "uid": (),
        "login": (),
        "walletAddress": ("wallet_address",),
        "privateKey": ("private_key",),
    }

    for key, value in credentials.items():
        if value is None:
            continue
        if key in sensitive_fields:
            aliases = alias_map.get(key, ())
            _set_exchange_field(client, key, value, aliases)
        elif key == "headers" and isinstance(value, Mapping):
            headers = getattr(client, "headers", {}) or {}
            headers.update(value)
            client.headers = headers
        elif key == "options" and isinstance(value, Mapping):
            options = getattr(client, "options", None)
            if isinstance(options, MutableMapping):
                options.update(value)
            else:
                setattr(client, "options", dict(value))
        elif key == "ccxt" and isinstance(value, Mapping):
            # Some configurations expose an explicit ``ccxt`` block mirroring
            # passivbot's "ccxt_config" support. Apply the known keys while
            # falling back to attribute assignment for any extras.
            _apply_credentials(client, value)
        else:
            try:
                setattr(client, key, value)
            except Exception:
                logger.debug("Ignored unsupported credential field %s", key)

    headers = getattr(client, "headers", None)
    if isinstance(headers, MutableMapping):
        formatted = _format_header_placeholders(headers, credentials)
        if formatted is not None:
            client.headers = formatted


def _disable_fetch_currencies(client: Any) -> None:
    """Disable ccxt currency lookups that require authenticated endpoints."""
    options = getattr(client, "options", None)
    if isinstance(options, MutableMapping):
        # ccxt exchanges often respect ``options['fetchCurrencies']`` when deciding
        # whether to hit authenticated endpoints while loading markets.
        options["fetchCurrencies"] = False
        # Suppress any warnings about skipping currency downloads without keys.
        options["warnOnFetchCurrenciesWithoutApiKey"] = False

    has = getattr(client, "has", None)
    if isinstance(has, MutableMapping):
        # Some exchange implementations consult ``has['fetchCurrencies']``
        # instead of the options flag, therefore toggle both to cover either
        # code path.
        has["fetchCurrencies"] = False


def _suppress_open_orders_warning(client: Any) -> None:
    """Prevent ccxt from escalating open-order symbol warnings to exceptions."""
    options = getattr(client, "options", None)
    if isinstance(options, MutableMapping):
        options["warnOnFetchOpenOrdersWithoutSymbol"] = False
    else:
        setattr(client, "options", {"warnOnFetchOpenOrdersWithoutSymbol": False})


def _is_symbol_specific_open_orders_warning(error: BaseError) -> bool:
    """Return ``True`` when ``error`` is the ccxt warning about missing symbols."""
    message = str(error)
    return (
        "fetchOpenOrders() WARNING" in message
        and "without specifying a symbol" in message
        and "warnOnFetchOpenOrdersWithoutSymbol" in message
    )


_DEFAULT_OVERRIDE_SENTINEL = object()


def _instantiate_ccxt_client(
    exchange_id: str,
    credentials: Mapping[str, Any],
    *,
    custom_endpoint_override: object = _DEFAULT_OVERRIDE_SENTINEL,
) -> Any:
    """Instantiate a ccxt async client honoring passivbot customisations when available."""
    normalized = normalize_exchange_name(exchange_id)
    rate_limited = bool(credentials.get("enableRateLimit", True))

    if load_ccxt_instance is not None:
        client = load_ccxt_instance(
            normalized,
            enable_rate_limit=rate_limited,
            apply_custom_endpoints=False,
        )
        _apply_credentials(client, credentials)
        _disable_fetch_currencies(client)
        _suppress_open_orders_warning(client)
        override: Optional[ResolvedEndpointOverride]
        if custom_endpoint_override is _DEFAULT_OVERRIDE_SENTINEL:
            override = resolve_custom_endpoint_override(normalized)
        else:
            override = custom_endpoint_override  # type: ignore[assignment]
        apply_rest_overrides_to_ccxt(client, override)
        return client

    if ccxt_async is None:
        raise RuntimeError(
            "ccxt is required to create realtime exchange clients. Install it via 'pip install ccxt'."
        )

    try:
        exchange_class = getattr(ccxt_async, normalized)
    except AttributeError as exc:  # pragma: no cover - configuration error
        raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.") from exc

    params: MutableMapping[str, Any] = dict(credentials)
    params.setdefault("enableRateLimit", rate_limited)
    client = exchange_class(params)
    _apply_credentials(client, credentials)
    _disable_fetch_currencies(client)
    _suppress_open_orders_warning(client)
    override = (
        resolve_custom_endpoint_override(normalized)
        if custom_endpoint_override is _DEFAULT_OVERRIDE_SENTINEL
        else custom_endpoint_override  # type: ignore[assignment]
    )
    apply_rest_overrides_to_ccxt(client, override)
    return client


class CCXTAccountClient(AccountClientProtocol):
    """Realtime account client backed by ccxt asynchronous exchanges."""

    def __init__(self, config: AccountConfig) -> None:
        if not isinstance(config, AccountConfig):  # pragma: no cover - defensive
            raise TypeError("config must be an AccountConfig instance")

        self.config = config
        self._normalized_exchange = normalize_exchange_name(config.exchange)
        credentials = dict(config.credentials)
        credentials.setdefault("enableRateLimit", True)
        available_override = resolve_custom_endpoint_override(self._normalized_exchange)
        apply_override: Optional[ResolvedEndpointOverride]
        if config.use_custom_endpoints is False:
            apply_override = None
            if available_override and not available_override.is_noop():
                logger.info(
                    "Custom endpoint override for %s disabled by account configuration",
                    self.config.name,
                )
        else:
            apply_override = available_override
            if config.use_custom_endpoints is True and not available_override:
                logger.info(
                    "Account %s requested custom endpoints but none are configured for %s",
                    self.config.name,
                    self._normalized_exchange,
                )

        self.client = _instantiate_ccxt_client(
            config.exchange,
            credentials,
            custom_endpoint_override=apply_override,
        )
        self._balance_params = dict(config.params.get("balance", {}))
        self._positions_params = dict(config.params.get("positions", {}))
        self._orders_params = dict(config.params.get("orders", {}))
        self._close_params = dict(config.params.get("close", {}))
        cashflow_params = config.params.get("cashflows", {})
        self._cashflow_params = dict(cashflow_params) if isinstance(cashflow_params, Mapping) else {}
        self._markets_loaded: Optional[asyncio.Lock] = None
        self._debug_api_payloads = bool(config.debug_api_payloads)
        self._custom_endpoint_override = apply_override
        self._using_custom_endpoints = bool(apply_override and not apply_override.is_noop())
        source_path = get_custom_endpoint_source()
        self._custom_endpoint_source = str(source_path) if source_path else None
        self._liquidity_settings = config.liquidity

    def _refresh_open_order_preferences(self) -> None:
        """Re-apply exchange options that silence noisy open-order warnings."""
        try:
            _suppress_open_orders_warning(self.client)
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "Failed to update open-order warning preference for %s", self.config.name
            )

    def _log_exchange_payload(
        self, operation: str, payload: Any, params: Optional[Mapping[str, Any]]
    ) -> None:
        if not self._debug_api_payloads:
            return
        params_repr = _stringify_payload(params or {}) if params else "{}"
        payload_repr = _stringify_payload(payload)
        logger.debug(
            "[%s] %s response params=%s payload=%s",
            self.config.name,
            operation,
            params_repr,
            payload_repr,
        )

    async def _ensure_markets(self) -> None:
        lock = self._markets_loaded
        if lock is None:
            lock = asyncio.Lock()
            self._markets_loaded = lock
        async with lock:
            if getattr(self.client, "markets", None):
                return
            await self.client.load_markets()

    async def _fetch_realized_pnl(
        self,
        positions: Sequence[Mapping[str, Any]],
        *,
        now_ms: Optional[int] = None,
    ) -> Optional[float]:
        """Return realised PnL for supported venues within a configurable window."""
        params_cfg_raw = self.config.params.get("realized_pnl")
        params_cfg: Mapping[str, Any]
        if isinstance(params_cfg_raw, Mapping):
            params_cfg = params_cfg_raw
        else:
            return None
        if params_cfg.get("enabled", True) is False:
            return None

        lookback_ms = _coerce_int(params_cfg.get("lookback_ms"))
        if lookback_ms is None:
            lookback_hours = _coerce_int(params_cfg.get("lookback_hours"))
            if lookback_hours is not None:
                lookback_ms = lookback_hours * 60 * 60 * 1000
        if lookback_ms is None:
            lookback_minutes = _coerce_int(params_cfg.get("lookback_minutes"))
            if lookback_minutes is not None:
                lookback_ms = lookback_minutes * 60 * 1000
        if lookback_ms is None or lookback_ms <= 0:
            lookback_ms = 24 * 60 * 60 * 1000

        now_ms = now_ms or int(time.time() * 1000)
        since = (
            _coerce_int(params_cfg.get("since_ms"))
            or _coerce_int(params_cfg.get("start_time_ms"))
            or _coerce_int(params_cfg.get("since"))
        )
        if since is None:
            since = max(0, now_ms - lookback_ms)
        until = (
            _coerce_int(params_cfg.get("until_ms"))
            or _coerce_int(params_cfg.get("end_time_ms"))
            or _coerce_int(params_cfg.get("until"))
        )
        if until is None or until <= 0:
            until = now_ms
        if since >= until:
            return 0.0

        params_base = (
            dict(params_cfg.get("params", {}))
            if isinstance(params_cfg.get("params"), Mapping)
            else {}
        )

        try:
            exchange_id = self._normalized_exchange
            if exchange_id in {"binanceusdm", "binancecoinm", "binancecm"}:
                fetch_income = getattr(self.client, "fetch_income", None)
                if fetch_income is None:
                    return None
                request = dict(params_base)
                request.setdefault("incomeType", "REALIZED_PNL")
                request.setdefault("startTime", since)
                request.setdefault("endTime", until)
                symbol_override = params_cfg.get("symbol")
                if isinstance(symbol_override, str) and symbol_override:
                    request.setdefault("symbol", symbol_override)
                elif self.config.symbols and len(self.config.symbols) == 1:
                    request.setdefault("symbol", self.config.symbols[0])
                incomes = await fetch_income(params=request)
                total = 0.0
                for entry in incomes or []:
                    amount = _coerce_float(entry.get("amount"))
                    if amount is None and isinstance(entry.get("info"), Mapping):
                        info = entry["info"]
                        amount = _first_float(
                            info.get("amount"),
                            info.get("income"),
                            info.get("realizedPnl"),
                            info.get("realisedPnl"),
                        )
                    if amount is None:
                        continue
                    total += float(amount)
                return total

            if exchange_id == "bybit":
                fetch_closed_pnl = getattr(
                    self.client, "private_get_v5_position_closed_pnl", None
                )
                if fetch_closed_pnl is None:
                    return None
                limit = _coerce_int(params_cfg.get("limit")) or _coerce_int(
                    params_base.get("limit")
                )
                if limit is None or limit <= 0:
                    limit = 200
                total = 0.0
                cursor: Optional[str] = None
                while True:
                    request = dict(params_base)
                    request.setdefault("startTime", since)
                    request.setdefault("endTime", until)
                    request.setdefault("limit", limit)
