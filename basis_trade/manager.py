"""Target computation and rebalance decisions for the live basis trade. One check-and-act
pass per call to `run_once` -- the caller (run_live.py) decides the polling cadence.

Rebalance trigger is at the PORTFOLIO level, not per-coin: total basis-trade notional
(summed across coins, read from the spot leg as the delta-neutral reference) is compared
against the exposure band as a fraction of account equity. Inside the band, nothing is
touched, even if an individual coin has drifted a little -- this avoids paying round-trip
fees to chase small equity/price fluctuations. Outside the band, every enabled coin is
resized back to its target weight in the same pass.

Disabling a coin (or the master switch) unwinds that coin/everything to flat regardless of
the band check -- "off" always means off, not "off, eventually."
"""
from dataclasses import dataclass

from config import BasisTradeConfig
from exchange import PMBasisExchange
from order_execution import place_and_fill_maker_first

DELTA_NEUTRALITY_WARN_PCT = 0.05  # warn if spot and perp notional diverge beyond this


@dataclass
class CoinState:
    coin: str
    price: float
    spot_qty: float
    spot_notional: float
    perp_qty: float  # signed, negative = short
    perp_notional: float


def read_coin_state(ex: PMBasisExchange, coin: str, quote: str) -> CoinState:
    spot_symbol = f"{coin}/{quote}"
    perp_symbol = f"{coin}/{quote}:{quote}"
    ticker = ex.perp.fetch_ticker(perp_symbol)
    price = float(ticker["last"])

    spot_qty = ex.spot_holding(coin)
    spot_notional = spot_qty * price

    pos = ex.perp_position(perp_symbol)
    perp_qty = pos["qty"] if pos else 0.0
    perp_notional = abs(perp_qty) * price

    if spot_notional > 0 and abs(spot_notional - perp_notional) / spot_notional > DELTA_NEUTRALITY_WARN_PCT:
        print(f"[WARN] {coin}: spot/perp notional diverged beyond "
              f"{DELTA_NEUTRALITY_WARN_PCT:.0%} (spot=${spot_notional:,.2f} "
              f"perp=${perp_notional:,.2f}) -- check for a partial fill or stuck leg")

    return CoinState(coin, price, spot_qty, spot_notional, perp_qty, perp_notional)


def unimmr_tier(cfg: BasisTradeConfig, unimmr: float) -> str:
    """"normal" / "caution" / "derisk" / "emergency", in increasing order of severity.
    See config.py for why these sit well above Binance's own real thresholds (1.5/1.2/1.05)."""
    if unimmr <= cfg.unimmr_emergency:
        return "emergency"
    if unimmr <= cfg.unimmr_derisk:
        return "derisk"
    if unimmr <= cfg.unimmr_caution:
        return "caution"
    return "normal"


RECONCILE_MIN_NOTIONAL = 10.0  # matches resize_coin's own min-worth-a-round-trip floor


def reconcile_coin(ex: PMBasisExchange, cfg: BasisTradeConfig, state: CoinState) -> bool:
    """If a previous pass crashed between the spot and perp legs of resize_coin (confirmed
    live: a post-only reject partway through left spot bought with the matching perp short
    never placed), the two legs can be left at different notionals with no automatic path
    back to delta-neutral -- the normal target-vs-current-spot delta in resize_coin only
    reacts to the TARGET, not to this kind of divergence, so it would never close on its
    own. Always pulls the AHEAD leg down to match the LAGGING leg (never grows either leg),
    so calling this every pass can only shrink mismatch, never add net exposure. Returns
    whether any correction was placed, so the caller knows to re-read state afterward."""
    divergence = state.spot_notional - state.perp_notional
    threshold = max(RECONCILE_MIN_NOTIONAL, DELTA_NEUTRALITY_WARN_PCT * max(state.spot_notional, state.perp_notional))
    if abs(divergence) <= threshold or state.price <= 0:
        return False

    spot_symbol = f"{state.coin}/{cfg.quote}"
    perp_symbol = f"{state.coin}/{cfg.quote}:{cfg.quote}"
    kwargs = dict(
        reprice_interval_seconds=cfg.reprice_interval_seconds,
        timeout_minutes=cfg.timeout_minutes,
        book_depth_levels=cfg.book_depth_levels,
        max_book_participation_pct=cfg.max_book_participation_pct,
        dry_run=cfg.dry_run,
    )
    if divergence > 0:
        qty = min(divergence / state.price, ex.spot_holding(state.coin))
        if qty <= 0:
            return False
        place_and_fill_maker_first(ex.spot, spot_symbol, "sell", qty, **kwargs)
        print(f"[RECONCILE] {state.coin}: sold back ${divergence:,.2f} of unhedged excess spot")
    else:
        pos = ex.perp_position(perp_symbol)
        shortable = abs(pos["qty"]) if pos else 0.0
        qty = min(abs(divergence) / state.price, shortable)
        if qty <= 0:
            return False
        place_and_fill_maker_first(ex.perp, perp_symbol, "buy", qty,
                                    order_params={"positionSide": "SHORT"}, **kwargs)
        print(f"[RECONCILE] {state.coin}: bought back ${abs(divergence):,.2f} of excess short")
    return True


def _min_amount(market: dict) -> float:
    return float(market.get("limits", {}).get("amount", {}).get("min") or 0.0)


def resize_coin(ex: PMBasisExchange, cfg: BasisTradeConfig, state: CoinState, target_notional: float) -> bool:
    """Moves one coin's spot+perp pair from its current notional to target_notional.
    Opens the spot leg before the perp leg when increasing (matches the order validated
    live), and the perp leg before the spot leg when decreasing -- close the short first
    so the position is never net-long with no hedge, even transiently. Returns whether it
    actually placed anything, so callers can log what really happened rather than assuming
    every call resized."""
    spot_symbol = f"{state.coin}/{cfg.quote}"
    perp_symbol = f"{state.coin}/{cfg.quote}:{cfg.quote}"
    delta_notional = target_notional - state.spot_notional
    if abs(delta_notional) < max(10.0, 0.01 * max(target_notional, state.spot_notional)):
        return False  # below min-notional / not worth a round trip

    qty_delta = abs(delta_notional) / state.price

    if delta_notional > 0:
        # Perp futures markets commonly use a coarser minimum lot size than the matching
        # spot market (confirmed live: BTC/USDT:USDT requires >=0.001 BTC while BTC/USDT
        # spot allows 0.00001) -- a target too small to satisfy the PERP leg's minimum
        # can never be hedged, no matter how many times it's retried. Checked before
        # touching either leg so a too-small target never buys spot it can't hedge.
        perp_min_qty = _min_amount(ex.perp.market(perp_symbol))
        if qty_delta < perp_min_qty:
            print(f"[SKIP] {state.coin}: target increase {qty_delta:.6f} is below the perp "
                  f"market's minimum tradable amount {perp_min_qty:.6f} -- cannot be hedged "
                  f"at this account size, not opening either leg")
            return False
    kwargs = dict(
        reprice_interval_seconds=cfg.reprice_interval_seconds,
        timeout_minutes=cfg.timeout_minutes,
        book_depth_levels=cfg.book_depth_levels,
        max_book_participation_pct=cfg.max_book_participation_pct,
        dry_run=cfg.dry_run,
    )

    if delta_notional > 0:
        # increasing exposure: set leverage for the NEW notional first (only affects
        # margin for notional opened after this call, not the existing position's
        # maintenance margin -- see exchange.py), then open spot, then the matching short
        if not cfg.dry_run:
            ex.set_perp_leverage(state.coin, cfg.quote, cfg.leverage)
        place_and_fill_maker_first(ex.spot, spot_symbol, "buy", qty_delta, **kwargs)
        place_and_fill_maker_first(ex.perp, perp_symbol, "sell", qty_delta,
                                    order_params={"positionSide": "SHORT"}, **kwargs)
        return True
    else:
        # decreasing exposure: close the short first, then sell the now-unhedged spot.
        # Spot qty is read fresh (not assumed from nominal target) -- trading fees are
        # deducted from the asset received on a buy, so actual holdings run slightly
        # below whatever was nominally bought; confirmed live during connector testing.
        actual_spot_qty = ex.spot_holding(state.coin)
        close_qty = min(qty_delta, actual_spot_qty, abs(state.perp_qty))
        if close_qty <= 0:
            return False
        place_and_fill_maker_first(ex.perp, perp_symbol, "buy", close_qty,
                                    order_params={"positionSide": "SHORT"}, **kwargs)
        # re-read: a market/taker fallback on the perp leg settles near-instantly, but a
        # maker-first spot close should only ever sell what's confirmed free right now
        sellable = ex.spot_holding(state.coin)
        place_and_fill_maker_first(ex.spot, spot_symbol, "sell", min(close_qty, sellable), **kwargs)
        return True


def run_once(cfg: BasisTradeConfig, ex: PMBasisExchange) -> dict:
    account = ex.account_state()
    equity, unimmr = account["equity"], account["unimmr"]
    tier = unimmr_tier(cfg, unimmr)

    weights = cfg.enabled_coin_weights() if cfg.enabled else {}
    all_coins = list(cfg.coins.keys())

    states = {c: read_coin_state(ex, c, cfg.quote) for c in all_coins}
    reconciled = [c for c in all_coins if reconcile_coin(ex, cfg, states[c])]
    if reconciled:
        states = {c: read_coin_state(ex, c, cfg.quote) for c in all_coins}

    total_current_notional = sum(s.spot_notional for s in states.values())
    current_pct = (total_current_notional / equity) if equity > 0 else 0.0

    report = {
        "equity": equity,
        "unimmr": unimmr,
        "unimmr_tier": tier,
        "current_exposure_pct": current_pct,
        "band": (cfg.min_exposure_pct, cfg.max_exposure_pct),
        "actions": [f"{c}: reconciled spot/perp divergence" for c in reconciled],
    }

    # uniMMR emergency tier overrides everything else, including the master enable
    # switch -- flatten immediately regardless of band/disabled-coin bookkeeping below.
    if tier == "emergency":
        for coin in all_coins:
            s = states[coin]
            if s.spot_notional > 1.0 or s.perp_notional > 1.0:
                resize_coin(ex, cfg, s, target_notional=0.0)
        report["actions"].append(
            f"EMERGENCY: uniMMR={unimmr:.3f} <= {cfg.unimmr_emergency} -- flattened all coins"
        )
        return report

    disabled_coins = [c for c in all_coins if c not in weights]
    for coin in disabled_coins:
        s = states[coin]
        if s.spot_notional > 1.0 or s.perp_notional > 1.0:
            resize_coin(ex, cfg, s, target_notional=0.0)
            report["actions"].append(f"{coin}: unwound (disabled)")

    if not cfg.enabled:
        report["actions"].append("master switch disabled -- fully unwound" if disabled_coins else "already flat")
        return report

    if tier == "derisk":
        # Actively reduce now, regardless of band: one step down to the band floor per
        # pass. If uniMMR is still in this tier next pass, this fires again and keeps
        # cutting -- deliberately gradual (maker-first, orderly) rather than an
        # immediate full flatten, which is reserved for the emergency tier above.
        total_target_notional = equity * cfg.min_exposure_pct
        for coin, weight in weights.items():
            target = min(total_target_notional * weight, states[coin].spot_notional)
            resize_coin(ex, cfg, states[coin], target)
        report["actions"].append(
            f"DERISK: uniMMR={unimmr:.3f} <= {cfg.unimmr_derisk} -- reduced toward band floor "
            "(per-coin skips, if any, logged separately above)"
        )
        return report

    in_band = cfg.min_exposure_pct <= current_pct <= cfg.max_exposure_pct
    if in_band:
        report["actions"].append(f"within band ({current_pct:.1%}), no rebalance")
        return report

    total_target_notional = equity * cfg.target_exposure_pct
    for coin, weight in weights.items():
        target = total_target_notional * weight
        if tier == "caution" and target > states[coin].spot_notional:
            # caution tier: never increase exposure, even if the band logic alone
            # would call for it -- decreases (band logic calling for less exposure)
            # are still allowed through unchanged.
            report["actions"].append(
                f"{coin}: skipped increase (uniMMR={unimmr:.3f} in caution tier)"
            )
            continue
        acted = resize_coin(ex, cfg, states[coin], target)
        if acted:
            report["actions"].append(f"{coin}: resized toward ${target:,.2f} (was ${states[coin].spot_notional:,.2f})")

    return report
