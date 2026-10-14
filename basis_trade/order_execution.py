"""Post-only (maker) order execution with automatic re-pricing and a bounded-window taker
fallback. Adapted directly from portfolio/live/order_execution.py's proven pattern (same
maker-first + depth-aware slicing + bounded taker fallback) -- genuinely generic ccxt
order-execution logic with no portfolio/blend-specific coupling, kept as a per-module copy
here rather than a cross-directory import because that matches how this repo's other
standalone strategy modules (funding_arb/, momentum/, directional/) are already each
self-contained rather than importing from one another.

Fallback policy: re-price as maker for up to `timeout_minutes`, then cross as taker for
whatever quantity remains unfilled -- a deliberate, bounded exception to "always maker,"
not a silent compromise. Every taker fallback is printed explicitly for later audit.

dry_run=True (the default everywhere this is called from) places NO real orders -- it logs
what would have been placed (including the depth/slicing decision) and returns immediately.
"""
import time

import ccxt


def _round_to_tick(price: float, tick_size: float) -> float:
    if not tick_size:
        return price
    return round(round(price / tick_size) * tick_size, 10)


def _safe_maker_price(side: str, best_bid: float, best_ask: float, tick_size: float) -> float:
    """Posting exactly at best_bid/best_ask leaves zero margin: if the book ticks against
    us in the round-trip between fetching it and the order reaching the exchange, Binance
    rejects the post-only order as immediately fillable (confirmed live on a real BTC/USDT
    order during real-account testing). Backing off one extra tick from the opposing side
    gives a one-tick buffer against that race, at the cost of being one tick less
    aggressive as a maker."""
    step = tick_size or 0.0
    if side == "buy":
        return _round_to_tick(min(best_bid, best_ask - step), tick_size)
    return _round_to_tick(max(best_ask, best_bid + step), tick_size)


def estimate_depth_notional(book: dict, side: str, levels: int) -> float:
    """A SELL order takes liquidity from resting BIDS, a BUY order takes liquidity from
    resting ASKS. Returns 0.0 for a missing/empty book side -- callers treat 0.0 as "no
    depth information available" and fall back to not constraining the order."""
    book_side = book.get("bids") if side == "sell" else book.get("asks")
    if not book_side:
        return 0.0
    return sum(price * qty for price, qty in book_side[:levels])


def size_slice_for_depth(remaining_qty: float, price: float, depth_notional: float,
                          max_participation_pct: float) -> float:
    """Caps an order slice at `max_participation_pct` of visible depth notional. Falls back
    to `remaining_qty` unconstrained when depth_notional or price is 0 (no depth data)."""
    if depth_notional <= 0 or price <= 0:
        return remaining_qty
    depth_qty = (depth_notional * max_participation_pct) / price
    return min(remaining_qty, depth_qty)


def _log_depth(symbol: str, side: str, qty: float, price: float, depth_notional: float,
                slice_qty: float, prefix: str = "") -> None:
    requested_notional = qty * price
    low_liquidity = depth_notional < requested_notional
    flag = "  <<< LOW LIQUIDITY (order exceeds visible depth)" if low_liquidity else ""
    print(f"{prefix}[DEPTH] {symbol} {side}: requested={qty:.6f} "
          f"(${requested_notional:,.2f}) depth=${depth_notional:,.2f} -> "
          f"slice={slice_qty:.6f}{flag}")


def place_and_fill_maker_first(
    ex, symbol: str, side: str, qty: float,
    order_params: dict = None,
    reprice_interval_seconds: float = 60.0, timeout_minutes: float = 45.0,
    book_depth_levels: int = 20, max_book_participation_pct: float = 0.20,
    dry_run: bool = True,
) -> dict:
    """side: 'buy' or 'sell'. qty: positive quantity to trade. order_params: extra params
    merged into every create_order call (e.g. {"positionSide": "SHORT"} for a PM perp
    hedge-mode order) -- see basis_trade/manager.py for how the two legs differ here.
    """
    order_params = dict(order_params or {})
    market = ex.market(symbol)
    tick_size = market.get("precision", {}).get("price")
    # A dust remainder below the exchange's minimum tradable amount can never be placed
    # as its own order (confirmed live: a 0.000009 BTC leftover after a partial fill hit
    # Binance's InvalidOrder for being under the 0.00001 minimum) -- treat it as filled
    # rather than retrying it forever or crashing the pass.
    min_amount = (market.get("limits", {}).get("amount", {}).get("min") or 0.0)

    if dry_run:
        book = ex.fetch_order_book(symbol, limit=book_depth_levels)
        best_bid, best_ask = book["bids"][0][0], book["asks"][0][0]
        price = _safe_maker_price(side, best_bid, best_ask, tick_size)
        depth_notional = estimate_depth_notional(book, side, book_depth_levels)
        slice_qty = size_slice_for_depth(qty, price, depth_notional, max_book_participation_pct)
        print(f"[DRY RUN] would place post-only {side} {qty} {symbol} @ ~{price} "
              f"params={order_params} (maker-first, up to {timeout_minutes}min, then taker fallback)")
        _log_depth(symbol, side, qty, price, depth_notional, slice_qty, prefix="[DRY RUN] ")
        return {"symbol": symbol, "side": side, "requested_qty": qty, "filled_qty": 0.0,
                "avg_price": None, "taker_fallback": False, "dry_run": True}

    remaining = qty
    filled_qty = 0.0
    fills = []
    order_id = None
    deadline = time.time() + timeout_minutes * 60

    while remaining > max(1e-9, min_amount) and time.time() < deadline:
        book = ex.fetch_order_book(symbol, limit=book_depth_levels)
        best_bid, best_ask = book["bids"][0][0], book["asks"][0][0]
        price = _safe_maker_price(side, best_bid, best_ask, tick_size)
        depth_notional = estimate_depth_notional(book, side, book_depth_levels)
        slice_qty = size_slice_for_depth(remaining, price, depth_notional, max_book_participation_pct)
        _log_depth(symbol, side, remaining, price, depth_notional, slice_qty)

        if order_id is not None:
            try:
                ex.cancel_order(order_id, symbol)
            except Exception:
                pass  # may have already filled or been cancelled -- next fetch_order confirms which
            order_id = None

        params = {**order_params, "postOnly": True}
        try:
            order = ex.create_order(symbol, "limit", side, slice_qty, price, params=params)
        except ccxt.OrderImmediatelyFillable:
            # Book moved between fetch and submit despite the one-tick buffer above --
            # Binance never placed the order, so there is nothing to cancel. Just loop
            # around and reprice off a fresh book rather than aborting the whole pass.
            print(f"[REPRICE] {symbol} {side}: post-only would have crossed, refreshing book")
            continue
        order_id = order["id"]
        time.sleep(reprice_interval_seconds)

        fetched = ex.fetch_order(order_id, symbol)
        cumulative_filled = float(fetched.get("filled", 0.0))
        new_fill = cumulative_filled - filled_qty
        if new_fill > 1e-9:
            fills.append({"price": fetched.get("average") or price, "qty": new_fill})
        filled_qty = cumulative_filled
        remaining = qty - filled_qty

    taker_fallback = False
    if remaining > max(1e-9, min_amount):
        taker_fallback = True
        if order_id is not None:
            try:
                ex.cancel_order(order_id, symbol)
            except Exception:
                pass
        print(f"[TAKER FALLBACK] {symbol} {side} {remaining:.6f} did not fill as maker within "
              f"{timeout_minutes}min -- crossing the spread")
        order = ex.create_order(symbol, "market", side, remaining, params=order_params)
        taker_fill = float(order.get("filled", remaining))
        fills.append({"price": order.get("average"), "qty": taker_fill})
        filled_qty += taker_fill

    avg_price = (sum(f["price"] * f["qty"] for f in fills) / filled_qty) if filled_qty > 0 else None
    return {"symbol": symbol, "side": side, "requested_qty": qty, "filled_qty": filled_qty,
            "avg_price": avg_price, "taker_fallback": taker_fallback, "dry_run": False}
