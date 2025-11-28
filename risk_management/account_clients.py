"""Account client implementations for the risk management tooling."""

from __future__ import annotations

import abc
import asyncio
import inspect
import json
import logging
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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


def _normalise_order_book_depth(exchange: str, requested: int) -> int:
    """Return a depth limit supported by the exchange."""

    exchange_key = (exchange or "").strip().lower()
    if exchange_key:
        exchange_key = exchange_key.replace("-", "").replace("_", "")

    allowed_depths_map = {
        "binance": (5, 10, 20, 50, 100, 500, 1000),
        "binanceusdm": (5, 10, 20, 50, 100, 500, 1000),
        "binancecoinm": (5, 10, 20, 50, 100, 500, 1000),
    }

    allowed_depths = allowed_depths_map.get(exchange_key)
    if allowed_depths is None:
        if exchange_key.startswith("binance") and exchange_key.endswith("coinm"):
            allowed_depths = allowed_depths_map["binancecoinm"]
        elif exchange_key.startswith("binance"):
            allowed_depths = allowed_depths_map["binanceusdm"]

    if not allowed_depths:
        return max(int(requested), 1)

    if requested <= 0:
        return allowed_depths[0]

    for depth in allowed_depths:
        if requested <= depth:
            if requested != depth:
                logger.debug(
                    "Normalising unsupported order book depth %s to %s for %s",
                    requested,
                    depth,
                    exchange,
                )
            return depth

    logger.debug(
        "Normalising unsupported order book depth %s to %s for %s",
        requested,
        allowed_depths[-1],
        exchange,
    )
    return allowed_depths[-1]


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
        self._balance_params = self._normalise_params(config.params.get("balance"))
        self._positions_params = self._normalise_params(config.params.get("positions"))
        self._orders_params = self._normalise_params(config.params.get("orders"))
        self._close_params = self._normalise_params(config.params.get("close"))
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

    @staticmethod
    def _normalise_params(params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """Return a copy of ``params`` when it is a mapping, otherwise an empty dict."""

        if isinstance(params, Mapping):
            return dict(params)
        if params is None:
            return {}
        logger.warning("Unsupported params type %s; ignoring", type(params).__name__)
        return {}

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

        symbol_override = params_cfg.get("symbol")
        symbols: Optional[Sequence[Optional[str]]] = None
        if isinstance(symbol_override, str) and symbol_override:
            symbols = [symbol_override]
        elif self.config.symbols:
            symbols = [symbol for symbol in self.config.symbols if isinstance(symbol, str)]

        limit = _coerce_int(params_cfg.get("limit")) or _coerce_int(params_base.get("limit"))

        try:
            return await fetch_realized_pnl_history(
                self._normalized_exchange,
                self.client,
                since=since,
                until=until,
                params=params_base,
                limit=limit,
                symbols=symbols,
                account_name=self.config.name,
                log=logger,
                debug_api_payloads=self._debug_api_payloads,
            )
        except BaseError as exc:
            logger.debug(
                "[%s] Failed to fetch realised PnL via history endpoints: %s",
                self.config.name,
                exc,
                exc_info=self._debug_api_payloads,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug(
                "[%s] Unexpected error while fetching realised PnL via history: %s",
                self.config.name,
                exc,
                exc_info=self._debug_api_payloads,
            )

        return None

    async def _collect_symbol_metrics(
        self, symbols: Sequence[str]
    ) -> Dict[str, Mapping[str, Any]]:
        """Fetch optional market data used to enrich position analytics."""
        metrics: Dict[str, Mapping[str, Any]] = {}
        if not symbols:
            return metrics
        settings = self._liquidity_settings
        if not getattr(settings, "fetch_order_book", True):
            return metrics
        raw_depth = getattr(settings, "depth", 25)
        depth = _coerce_int(raw_depth) or 25
        exchange_id = getattr(self, "_normalized_exchange", None)
        if not exchange_id:
            exchange_id = normalize_exchange_name(getattr(self.config, "exchange", ""))
        depth = _normalise_order_book_depth(exchange_id, depth)
        for symbol in dict.fromkeys(symbols):
            if not symbol:
                continue
            try:
                order_book = await self.client.fetch_order_book(symbol, limit=depth)
            except NotSupported:
                logger.debug("[%s] Order book not supported for %s", self.config.name, symbol)
                continue
            except BaseError as exc:
                logger.debug(
                    "[%s] Failed to fetch order book for %s: %s",
                    self.config.name,
                    symbol,
                    exc,
                    exc_info=self._debug_api_payloads,
                )
                continue
            normalised = normalise_order_book(order_book, depth=depth)
            if normalised:
                metrics[symbol] = {"order_book": normalised}
        return metrics

    async def _fetch_cashflows(self) -> List[Mapping[str, Any]]:
        """Return recent deposit and withdrawal events when configured."""
        params_cfg = self._cashflow_params or {}
        normalized_exchange = getattr(
            self, "_normalized_exchange", normalize_exchange_name(getattr(self.config, "exchange", ""))
        )
        now_ms = int(time.time() * 1000)
        lookback_ms = _coerce_int(params_cfg.get("lookback_ms"))
        if lookback_ms is None:
            lookback_days = _coerce_int(params_cfg.get("lookback_days"))
            if lookback_days is None:
                lookback_days = 30
            lookback_ms = int(timedelta(days=lookback_days).total_seconds() * 1000)
        since = _coerce_int(params_cfg.get("since_ms")) or now_ms - lookback_ms
        until = _coerce_int(params_cfg.get("until_ms")) or now_ms
        if since >= until:
            since = max(0, until - lookback_ms)

        base_params = (
            dict(params_cfg.get("params", {}))
            if isinstance(params_cfg.get("params"), Mapping)
            else {}
        )
        limit = _coerce_int(params_cfg.get("limit"))
        code = params_cfg.get("code") or self.config.settle_currency

        account_types_cfg = params_cfg.get("account_types")
        if isinstance(account_types_cfg, Sequence) and not isinstance(account_types_cfg, (str, bytes)):
            account_types: Sequence[Optional[str]] = list(account_types_cfg)  # type: ignore[list-item]
        elif account_types_cfg is None:
            if normalized_exchange == "bybit":
                account_types = [None, "UNIFIED", "CONTRACT", "SPOT"]
            else:
                account_types = [None]
        else:
            account_types = [account_types_cfg]

        async def _fetch_side(
            fetcher,
            side: str,
            account_type: Optional[str],
        ) -> List[Mapping[str, Any]]:
            events: List[Mapping[str, Any]] = []
            ranges: List[Tuple[int, int]] = [(since, until)]

            def _build_params(start: int, end: int) -> Dict[str, Any]:
                params = dict(base_params)
                if normalized_exchange == "okx":
                    params.setdefault("from", start)
                    params.setdefault("to", end)
                    params.setdefault("after", start)
                    params.setdefault("before", end)
                else:
                    params.setdefault("startTime", start)
                    params.setdefault("endTime", end)
                    params.setdefault("until", end)
                if account_type:
                    params.setdefault("accountType", account_type)
                if limit:
                    params.setdefault("limit", limit)
                return params

            while ranges:
                start, end = ranges.pop(0)
                if start >= end:
                    continue
                params = _build_params(start, end)
                try:
                    payload = await fetcher(
                        code=code,
                        since=start,
                        limit=limit,
                        params=params,
                    )
                except BadRequest as exc:
                    message = str(exc)
                    if "interval between the startTime and endTime is incorrect" in message:
                        window = end - start
                        if window <= 60 * 60 * 1000:
                            logger.debug(
                                "[%s] Cashflow window too small to split further: %s",
                                self.config.name,
                                exc,
                            )
                            continue
                        midpoint = start + window // 2
                        ranges.insert(0, (midpoint, end))
                        ranges.insert(0, (start, midpoint))
                        continue
                    raise self._translate_ccxt_error(exc)
                except BaseError as exc:
                    raise self._translate_ccxt_error(exc)

                for entry in payload or []:
                    amount = _coerce_float(entry.get("amount"))
                    timestamp = _coerce_int(entry.get("timestamp")) or _coerce_int(entry.get("time"))
                    currency = entry.get("currency") or entry.get("code")
                    events.append(
                        {
                            "type": side,
                            "id": entry.get("id") or entry.get("txid") or entry.get("txId"),
                            "amount": float(amount or 0.0),
                            "currency": currency,
                            "timestamp": timestamp,
                            "info": entry,
                        }
                    )
            return events

        client = _instantiate_ccxt_client(
            self.config.exchange,
            self.config.credentials,
            custom_endpoint_override=self._custom_endpoint_override,
        )
        try:
            events: List[Mapping[str, Any]] = []
            for account_type in account_types:
                fetch_deposits = getattr(client, "fetch_deposits", None)
                if callable(fetch_deposits):
                    events.extend(await _fetch_side(fetch_deposits, "deposit", account_type))
                fetch_withdrawals = getattr(client, "fetch_withdrawals", None)
                if callable(fetch_withdrawals):
                    events.extend(await _fetch_side(fetch_withdrawals, "withdrawal", account_type))
            seen: set[tuple[Any, Any, Any]] = set()
            unique_events: List[Mapping[str, Any]] = []
            for event in events:
                key = (event.get("type"), event.get("id"), event.get("timestamp"))
                if key in seen:
                    continue
                seen.add(key)
                unique_events.append(event)
            return unique_events
        finally:
            closer = getattr(client, "close", None)
            if closer is not None:
                try:
                    result = closer()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.debug("[%s] Failed to close cashflow client", self.config.name)

    def _translate_ccxt_error(self, error: BaseError) -> Exception:
        """Map ccxt exceptions to richer runtime errors."""
        message = str(error)
        payload_text: Optional[str] = None
        json_payload: Optional[Mapping[str, Any]] = None
        if "{" in message and "}" in message:
            payload_text = message[message.find("{") : message.rfind("}") + 1]
            try:
                json_payload = json.loads(payload_text)
            except Exception:
                json_payload = None

        extracted_message = None
        source = None
        if isinstance(json_payload, Mapping):
            extracted_message = (
                json_payload.get("msg")
                or json_payload.get("message")
                or json_payload.get("error")
            )
            source = json_payload.get("source") or json_payload.get("sourceId")

        detail = extracted_message or message
        parts = [detail]
        if source:
            parts.append(f"source: {source}")
        if self._using_custom_endpoints and self._custom_endpoint_source:
            parts.append(f"custom endpoint source: {self._custom_endpoint_source}")
        summary = " | ".join(part for part in parts if part)

        lower = summary.lower()
        auth_tokens = ("api key", "apikey", "auth", "permission", "sign", "invalid key")
        is_auth = any(token in lower for token in auth_tokens)

        if not is_auth:
            summary = f"{summary} — See logs for additional details."

        message_prefix = f"[{self.config.name}] "
        final_message = message_prefix + summary

        if is_auth:
            return AuthenticationError(final_message)
        return RuntimeError(final_message)

    async def fetch(self) -> Dict[str, Any]:
        """Return the latest account snapshot."""
        try:
            await self._ensure_markets()
            if self._balance_params:
                balance_payload = await self.client.fetch_balance(params=self._balance_params)
            else:
                balance_payload = await self.client.fetch_balance()
        except BaseError as exc:
            raise self._translate_ccxt_error(exc)

        balance_value = _extract_balance(balance_payload or {}, self.config.settle_currency)

        try:
            if self._positions_params:
                raw_positions = await self.client.fetch_positions(params=self._positions_params)
            else:
                raw_positions = await self.client.fetch_positions()
        except NotSupported:
            raw_positions = []
        except BaseError as exc:
            raise self._translate_ccxt_error(exc)

        parsed_positions: List[Dict[str, Any]] = []
        for position in raw_positions or []:
            parsed = _parse_position(position, balance_value)
            if not parsed:
                continue
            hedge_side, position_idx, side_explicit = _extract_position_details(position)
            parsed["hedge_side"] = hedge_side
            parsed["position_idx"] = position_idx
            parsed["hedge_side_explicit"] = side_explicit
            size_value = _first_float(
                position.get("contracts"),
                position.get("size"),
                position.get("amount"),
                position.get("info", {}).get("positionAmt") if isinstance(position.get("info"), Mapping) else None,
            )
            if size_value is not None:
                parsed["size"] = float(size_value)
            parsed_positions.append(parsed)

        symbol_metrics = await self._collect_symbol_metrics([p["symbol"] for p in parsed_positions])
        for position in parsed_positions:
            metrics = symbol_metrics.get(position["symbol"])
            if metrics and "order_book" in metrics:
                position["liquidity"] = calculate_position_liquidity(
                    {
                        "side": position.get("side"),
                        "size": position.get("size"),
                        "notional": position.get("notional"),
                        "entry_price": position.get("entry_price"),
                        "mark_price": position.get("mark_price"),
                    },
                    metrics["order_book"],
                    fallback_price=position.get("mark_price") or position.get("entry_price"),
                    warning_threshold=self._liquidity_settings.slippage_warning_pct,
                )
                position.setdefault("market_data", {}).update(metrics)

        try:
            if self._orders_params:
                open_orders_payload = await self.client.fetch_open_orders(params=self._orders_params)
            else:
                open_orders_payload = await self.client.fetch_open_orders()
        except BaseError as exc:
            if _is_symbol_specific_open_orders_warning(exc):
                self._refresh_open_order_preferences()
                open_orders_payload = []
            else:
                raise self._translate_ccxt_error(exc)

        open_orders = []
        for order in open_orders_payload or []:
            parsed_order = _parse_order(order)
            if parsed_order:
                open_orders.append(parsed_order)

        realized = await self._fetch_realized_pnl(parsed_positions)
        if realized is None:
            realized = sum(position.get("daily_realized_pnl", 0.0) for position in parsed_positions)

        total_unrealized = sum(position.get("unrealized_pnl", 0.0) for position in parsed_positions)
        total_notional = sum(position.get("notional", 0.0) for position in parsed_positions)

        cashflow_events: Optional[List[Mapping[str, Any]]] = None
        cashflow_settings = getattr(self, "_cashflow_params", {}) or {}
        if not hasattr(self, "_cashflow_params"):
            self._cashflow_params = dict(cashflow_settings)
        exchange_name = getattr(self.config, "exchange", None)
        if cashflow_settings.get("enabled", True) and exchange_name:
            try:
                cashflow_events = await self._fetch_cashflows()
            except AuthenticationError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[%s] Failed to fetch cash flow events: %s",
                    self.config.name,
                    exc,
                )
                if self._debug_api_payloads:
                    logger.debug("[%s] Cash flow fetch error", self.config.name, exc_info=True)

        exchange_name = exchange_name or ""
        snapshot: Dict[str, Any] = {
            "account": self.config.name,
            "name": self.config.name,
            "exchange": exchange_name,
            "balance": balance_value,
            "balance_raw": balance_payload,
            "positions": parsed_positions,
            "orders": open_orders,
            "unrealized_pnl": total_unrealized,
            "notional": total_notional,
            "daily_realized_pnl": realized,
            "using_custom_endpoints": getattr(self, "_using_custom_endpoints", False),
        }
        snapshot["order_books"] = symbol_metrics
        custom_source = getattr(self, "_custom_endpoint_source", None)
        if custom_source:
            snapshot["custom_endpoint_source"] = custom_source
        if cashflow_events is not None:
            snapshot["cashflows"] = {"events": list(cashflow_events)}
        return snapshot

    async def close(self) -> None:
        closer = getattr(self.client, "close", None)
        if closer is None:
            return
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("[%s] Failed to close client cleanly", self.config.name)

    async def kill_switch(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        logger.info(
            "[%s] Executing kill switch%s",
            self.config.name,
            f" for {symbol}" if symbol else "",
        )

        result: Dict[str, Any] = {}
        cancel_result = await self.cancel_all_orders(symbol)
        result["cancelled_orders"] = cancel_result

        close_result = await self.close_all_positions(symbol)
        if isinstance(close_result, Mapping):
            result["closed_positions"] = close_result.get("closed_positions", [])
            if "failed_position_closures" in close_result:
                result["failed_position_closures"] = close_result.get("failed_position_closures", [])
        else:
            result["closed_positions"] = close_result

        if isinstance(cancel_result, Mapping) and "failed_order_cancellations" in cancel_result:
            result["failed_order_cancellations"] = cancel_result.get("failed_order_cancellations", [])

        if result.get("failed_position_closures") or result.get("failed_order_cancellations"):
            logger.debug("[%s] Kill switch details: %s", self.config.name, result)
        logger.info("[%s] Kill switch completed", self.config.name)
        return result

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        try:
            return await self.client.create_order(symbol, order_type, side, amount, price, params)
        except BaseError as exc:
            raise self._translate_ccxt_error(exc)

    async def cancel_order(
        self,
        order_id: str,
        symbol: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        try:
            return await self.client.cancel_order(order_id, symbol, params)
        except BaseError as exc:
            raise self._translate_ccxt_error(exc)

    async def close_position(self, symbol: str) -> Mapping[str, Any]:
        closer = getattr(self.client, "close_position", None)
        if not callable(closer):
            raise NotSupported("close_position is not supported by this exchange")
        try:
            return await closer(symbol)
        except BaseError as exc:
            raise self._translate_ccxt_error(exc)

    async def list_order_types(self) -> Sequence[str]:
        has = getattr(self.client, "has", {})
        if isinstance(has, Mapping):
            supported = [key.replace("createOrder", "order") for key, value in has.items() if key.startswith("createOrder") and value]
            if supported:
                return tuple(sorted(dict.fromkeys(supported)))
        return ("limit", "market")

    @staticmethod
    def _is_missing_symbol_error(error: BaseError) -> bool:
        """Return True when ``error`` indicates a required symbol parameter."""

        message = str(error).lower()
        if not message:
            return False
        mentions_symbol = any(token in message for token in ("symbol", "market", "pair"))
        mentions_required = any(token in message for token in ("required", "missing", "argument"))
        return mentions_symbol and mentions_required

    @staticmethod
    def _extract_order_symbols(orders: Sequence[Mapping[str, Any]]) -> List[str]:
        symbols: List[str] = []
        for order in orders or []:
            if not isinstance(order, Mapping):
                continue
            raw_symbol = order.get("symbol")
            if raw_symbol:
                symbols.append(str(raw_symbol))
        return list(dict.fromkeys(symbols))

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Mapping[str, Any]:
        canceller = getattr(self.client, "cancel_all_orders", None)
        if not callable(canceller):
            if symbol:
                raise NotSupported("cancel_all_orders is not supported by this exchange")
            return {}

        kwargs: Dict[str, Any] = {}
        if self._close_params:
            kwargs["params"] = self._close_params

        if symbol:
            try:
                return await canceller(symbol, **kwargs)
            except BaseError as exc:
                raise self._translate_ccxt_error(exc)

        fetch_open_orders = getattr(self.client, "fetch_open_orders", None)
        try:
            return await canceller(None, **kwargs)
        except BaseError as exc:
            if symbol is not None or not self._is_missing_symbol_error(exc):
                raise self._translate_ccxt_error(exc)
            if not callable(fetch_open_orders):
                raise RuntimeError(
                    f"[{self.config.name}] Exchange requires a symbol to cancel orders,"
                    " but open orders could not be inspected to determine targets."
                )

        try:
            open_orders = await fetch_open_orders()
        except BaseError as exc:
            raise self._translate_ccxt_error(exc)

        symbols = self._extract_order_symbols(open_orders if isinstance(open_orders, Sequence) else [])
        if not symbols:
            return {"cancelled_orders": [], "failed_order_cancellations": []}

        cancelled: List[Mapping[str, Any]] = []
        failures: List[Mapping[str, Any]] = []
        for market_symbol in symbols:
            try:
                result = await canceller(market_symbol, **kwargs)
                cancelled.append({"symbol": market_symbol, "result": result})
            except BaseError as exc:
                failures.append({"symbol": market_symbol, "error": str(self._translate_ccxt_error(exc))})

        response: Dict[str, Any] = {"cancelled_orders": cancelled}
        if failures:
            response["failed_order_cancellations"] = failures
        return response

    async def close_all_positions(self, symbol: Optional[str] = None) -> Mapping[str, Any]:
        closer = getattr(self.client, "close_all_positions", None)
        if callable(closer):
            try:
                return await closer(symbol)
            except NotSupported:
                logger.info(
                    "[%s] close_all_positions not supported natively; falling back to per-position closes",
                    self.config.name,
                )
            except BaseError as exc:
                raise self._translate_ccxt_error(exc)
        fetch_positions = getattr(self.client, "fetch_positions", None)
        if not callable(fetch_positions):
            if symbol:
                raise NotSupported("close_position is not supported by this exchange")
            raise NotSupported("close_all_positions is not supported by this exchange")

        try:
            positions_payload = await fetch_positions(self._positions_params or {})
        except BaseError as exc:
            raise self._translate_ccxt_error(exc)

        closed: List[Mapping[str, Any]] = []
        failed: List[Mapping[str, Any]] = []

        for position in positions_payload or []:
            raw_symbol = position.get("symbol") or position.get("id")
            if not raw_symbol:
                continue
            position_symbol = str(raw_symbol)
            if symbol and position_symbol != symbol:
                continue

            size = _coerce_float(
                _first_float(
                    position.get("contracts"),
                    position.get("size"),
                    position.get("amount"),
                    position.get("info", {}).get("positionAmt") if isinstance(position.get("info"), Mapping) else None,
                )
            )
            if not size:
                continue

            order_side = "sell" if size > 0 else "buy"
            try:
                ticker = await self.client.fetch_ticker(position_symbol)
            except BaseError as exc:
                failed.append({"symbol": position_symbol, "error": str(exc)})
                continue

            bid = _first_float(ticker.get("bid"), ticker.get("info", {}).get("bid"))
            ask = _first_float(ticker.get("ask"), ticker.get("info", {}).get("ask"))
            last = _first_float(ticker.get("last"), ticker.get("close"), ticker.get("info", {}).get("last") or ticker.get("info", {}).get("close"))
            price = bid if order_side == "sell" else ask
            if price is None:
                price = last
            if price is None:
                failed.append({"symbol": position_symbol, "error": "No price available to close position"})
                continue

            params = dict(self._close_params or {})
            provided_reduce_only = any(str(key).lower() == "reduceonly" for key in params)
            params = {key: value for key, value in params.items() if str(key).lower() != "reduceonly"}
            position_side, position_idx, side_explicit = _extract_position_details(position)
            if not provided_reduce_only and not side_explicit:
                params["reduceOnly"] = True

            if position_side:
                params.setdefault("positionSide", position_side)
            if position_idx is not None:
                params.setdefault("positionIdx", position_idx)

            try:
                await self.client.create_order(
                    position_symbol,
                    "market",
                    order_side,
                    abs(size),
                    price,
                    params,
                )
                closed.append({"symbol": position_symbol})
            except BaseError as exc:
                failed.append({"symbol": position_symbol, "error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive
                failed.append({"symbol": position_symbol, "error": str(exc)})

        if symbol and not closed and not failed:
            raise NotSupported("close_position is not supported by this exchange")

        return {"closed_positions": closed, "failed_position_closures": failed}
