"""Utilities for analysing order-book liquidity and slippage."""

from __future__ import annotations

"""Utilities for analysing order-book liquidity and slippage."""

from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

__all__ = ["normalise_order_book", "calculate_position_liquidity"]


def _to_float(value: Any) -> Optional[float]:
    """Attempt to coerce ``value`` to ``float`` returning ``None`` on failure."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return float(candidate)
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_side(levels: Any, depth: Optional[int]) -> List[Tuple[float, float]]:
    """Return a normalised ``[(price, size), ...]`` list for an order-book side."""

    if not isinstance(levels, Iterable):
        return []
    normalised: List[Tuple[float, float]] = []
    for level in levels:
        price: Optional[float] = None
        amount: Optional[float] = None
        if isinstance(level, Mapping):
            price = _to_float(level.get("price"))
            amount = _to_float(level.get("amount")) or _to_float(level.get("size"))
        elif isinstance(level, Sequence):
            sequence: Sequence[Any] = level
            if len(sequence) >= 2:
                price = _to_float(sequence[0])
                amount = _to_float(sequence[1])
        if price is None or amount is None:
            continue
        if amount <= 0:
            continue
        normalised.append((price, amount))
        if depth is not None and depth > 0 and len(normalised) >= depth:
            break
    return normalised


def normalise_order_book(
    order_book: Mapping[str, Any], *, depth: Optional[int] = None
) -> Optional[MutableMapping[str, Any]]:
    """Normalise a ccxt style order-book payload."""

    if not isinstance(order_book, Mapping):
        return None

    bids = _normalise_side(order_book.get("bids"), depth)
    asks = _normalise_side(order_book.get("asks"), depth)
    if not bids and not asks:
        return None

    timestamp = _to_float(order_book.get("timestamp"))
    datetime_raw = order_book.get("datetime")
    datetime_value = str(datetime_raw) if isinstance(datetime_raw, str) and datetime_raw else None

    payload: MutableMapping[str, Any] = {
        "bids": [[price, size] for price, size in bids],
        "asks": [[price, size] for price, size in asks],
        "timestamp": timestamp,
    }
    if datetime_value:
        payload["datetime"] = datetime_value
    if bids:
        payload["best_bid"] = bids[0][0]
    if asks:
        payload["best_ask"] = asks[0][0]
    payload["depth"] = {"bids": len(bids), "asks": len(asks)}
    return payload


def _resolve_reference_price(position: Mapping[str, Any], fallback: Optional[float]) -> Optional[float]:
    for key in ("mark_price", "markPrice", "mark", "last", "last_price"):
        value = position.get(key)
        price = _to_float(value)
        if price:
            return price
    for key in ("entry_price", "entryPrice", "entry"):
        value = position.get(key)
        price = _to_float(value)
        if price:
            return price
    return fallback


def calculate_position_liquidity(
    position: Mapping[str, Any],
    order_book: Optional[Mapping[str, Any]],
    *,
    fallback_price: Optional[float] = None,
    warning_threshold: float = 0.02,
) -> MutableMapping[str, Any]:
    """Estimate liquidity metrics for ``position`` using ``order_book``."""

    size = _to_float(position.get("size"))
    notional = _to_float(position.get("notional"))
    reference_price = _resolve_reference_price(position, fallback_price)

    if size is None or size == 0:
        if reference_price and reference_price > 0 and notional:
            size = abs(notional / reference_price)
        elif notional and reference_price is None:
            reference_price = _to_float(position.get("mark_price")) or _to_float(
                position.get("entry_price")
            )
            if reference_price:
                size = abs(notional / reference_price) if reference_price else None
    if size is None or size <= 0:
        return {
            "filled_size": 0.0,
            "filled_notional": 0.0,
            "average_price": None,
            "slippage_pct": None,
            "unfilled_size": 0.0,
            "coverage_pct": 0.0,
            "reference_price": reference_price,
            "warnings": ["position_size_undefined"],
            "source": "unavailable",
            "warning_threshold_pct": warning_threshold,
        }

    position_size = abs(size)
    side = str(position.get("side", "")).lower()
    exit_side = "sell" if side == "long" else "buy"

    coverage_pct = 0.0
    filled_size = 0.0
    filled_notional = 0.0
    slippage_pct: Optional[float] = None
    average_price: Optional[float] = None
    warnings: List[str] = []
    source = "order_book"

    if order_book and isinstance(order_book, Mapping):
        levels_key = "bids" if exit_side == "sell" else "asks"
        levels = order_book.get(levels_key)
        normalised_levels = _normalise_side(levels, None)
        for price, available in normalised_levels:
            take = min(available, position_size - filled_size)
            if take <= 0:
                break
            filled_size += take
            filled_notional += take * price
        if filled_size > 0:
            average_price = filled_notional / filled_size
            if reference_price is None:
                reference_price = normalised_levels[0][0]
            coverage_pct = min(1.0, filled_size / position_size)
            if reference_price and reference_price > 0 and average_price is not None:
                slippage_pct = (average_price - reference_price) / reference_price
        else:
            coverage_pct = 0.0
            warnings.append("depth_unavailable")
        if filled_size < position_size:
            warnings.append("insufficient_depth")
        unfilled_size = max(0.0, position_size - filled_size)
        timestamp = order_book.get("timestamp") if isinstance(order_book, Mapping) else None
        datetime_value = order_book.get("datetime") if isinstance(order_book, Mapping) else None
    else:
        unfilled_size = position_size
        source = "fallback" if fallback_price is not None else "unavailable"
        warnings.append("depth_unavailable")
        timestamp = None
        datetime_value = None
        if fallback_price is not None:
            average_price = fallback_price
            reference_price = reference_price or fallback_price
            coverage_pct = 0.0
    if slippage_pct is not None and abs(slippage_pct) >= warning_threshold:
        warnings.append("slippage_threshold_exceeded")

    payload: MutableMapping[str, Any] = {
        "filled_size": float(filled_size),
        "filled_notional": float(filled_notional),
        "average_price": average_price,
        "slippage_pct": slippage_pct,
        "unfilled_size": float(unfilled_size),
        "coverage_pct": float(coverage_pct),
        "reference_price": reference_price,
        "warnings": warnings,
        "side": exit_side,
        "source": source,
        "warning_threshold_pct": warning_threshold,
    }
    if timestamp is not None:
        payload["timestamp"] = timestamp
    if datetime_value:
        payload["datetime"] = datetime_value
    return payload
