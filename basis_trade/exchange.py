"""Binance Portfolio Margin exchange wrapper for live execution of the long-spot /
short-perp basis trade. Needs TWO ccxt clients sharing one PM account -- a spot/margin
client for the long leg and a USDT-M futures client for the short leg -- both with
Binance's unified ccxt.binance class routing to papi* endpoints via a single
`options["portfolioMargin"]` flag, confirmed directly against a real PM account (see
src/exchanges/binance_portfolio.py for the full validation history this reuses).

Not a reuse of passivbot's own exchange classes, same reasoning as
portfolio/live/exchange.py: those are tightly entangled with the Passivbot god-class
and its grid-specific config schema. Uses plain synchronous ccxt (not ccxt.async_support)
-- this strategy holds a static, always-on position and rebalances on drift against a
10-15% exposure band, not a continuously-managed grid, so async buys nothing here.

Two things confirmed as REAL BUGS during live testing against this exact account type,
both avoided below rather than rediscovered:
- ccxt's unified fetch_balance() reads only the UM-futures sub-ledger for a PM account
  and is legitimately 0 on an account with real, unallocated collateral. Balance here
  reads papiGetAccount()'s accountEquity field directly instead.
- ccxt's unified fetch_balance()['free'] is not populated for PM spot/margin holdings
  even when the raw papi data confirms the asset is fully free to trade. Spot holdings
  here are read from the raw papiGetBalance() response's crossMarginFree field, not
  ccxt's derived 'free'.

Credentials are read from the repo's existing api-keys.json by account label (same
convention passivbot and portfolio/live both use), never embedded in code, printed, or
logged.
"""
import json
import os

import ccxt

API_KEYS_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "api-keys.json")


def load_api_keys(account_label: str, api_keys_path: str = None) -> dict:
    path = api_keys_path or API_KEYS_PATH_DEFAULT
    with open(path) as f:
        keys = json.load(f)
    if account_label not in keys:
        available = [k for k, v in keys.items() if isinstance(v, dict) and "exchange" in v]
        raise KeyError(f"{account_label!r} not found in {path} -- available accounts: {available}")
    entry = keys[account_label]
    if entry.get("exchange") != "binance":
        raise ValueError(f"{account_label!r} is a {entry.get('exchange')!r} account, not binance")
    if str(entry.get("account_type", "")).lower() not in ("portfolio_margin", "pm"):
        raise ValueError(
            f"{account_label!r} is not marked account_type: portfolio_margin in api-keys.json -- "
            "this module only works against a real PM account."
        )
    return entry


def _creds_kwargs(creds: dict) -> dict:
    return {
        "apiKey": creds.get("apiKey") or creds.get("key"),
        "secret": creds.get("secret"),
        "enableRateLimit": True,
    }


class PMBasisExchange:
    """Owns both legs' ccxt clients for one PM account."""

    def __init__(self, account_label: str, api_keys_path: str = None):
        creds = load_api_keys(account_label, api_keys_path)
        self.account_label = account_label

        self.spot = ccxt.binance(_creds_kwargs(creds))
        self.spot.options["portfolioMargin"] = True
        self.spot.load_markets()

        self.perp = ccxt.binanceusdm(_creds_kwargs(creds))
        self.perp.options["defaultType"] = "swap"
        self.perp.options["portfolioMargin"] = True
        self.perp.load_markets()

    def account_state(self) -> dict:
        """Single papiGetAccount() call -- equity and uniMMR are read together since both
        come off the same response and callers generally want both per pass."""
        acct = self.perp.papiGetAccount()
        return {
            "equity": float(acct["accountEquity"]),
            # uniMMR = adjusted equity / maintenance margin. Binance returns "99999999"
            # (not a real ratio) when maintenance margin is ~0, i.e. positions too small
            # to be meaningfully at risk -- confirmed live on a flat/near-flat account.
            # Treated as "effectively infinite," not clamped or special-cased further.
            "unimmr": float(acct["uniMMR"]),
        }

    def account_equity(self) -> float:
        """papiGetAccount()'s accountEquity -- confirmed to include unrealized PnL (both
        it and actualEquity do) and to match totalAvailableBalance when flat. Single
        account-wide figure, correct even for multi-asset collateral by construction."""
        return self.account_state()["equity"]

    def set_perp_leverage(self, coin: str, quote: str, leverage: int):
        """Sets leverage via set_leverage(), which ccxt routes to papiPostUmLeverage once
        options["portfolioMargin"] is set -- confirmed live during connector validation.
        Only affects margin required for NEW notional opened after this call; does not
        retroactively change an existing position's maintenance margin requirement (that
        is determined by Binance's notional-bracket table, not the leverage setting)."""
        symbol = f"{coin}/{quote}:{quote}"
        return self.perp.set_leverage(leverage, symbol)

    def spot_holding(self, coin: str) -> float:
        """Actual free quantity of `coin` held in the PM account's spot/margin ledger.
        Reads the raw papiGetBalance() response directly -- ccxt's derived
        fetch_balance()['free'] is NOT reliable here, confirmed empty against real,
        genuinely-free collateral during live testing."""
        for entry in self.perp.papiGetBalance():
            if entry.get("asset") == coin:
                return float(entry.get("crossMarginFree", 0.0))
        return 0.0

    def perp_position(self, symbol: str) -> dict | None:
        """symbol e.g. 'BTC/USDT:USDT'. Returns None if flat, else a dict with signed
        qty (negative = short) and entry/mark price."""
        for p in self.perp.fetch_positions_risk():
            info = p.get("info", {})
            if info.get("symbol") == self.perp.market(symbol)["id"]:
                qty = float(info.get("positionAmt", 0.0) or 0.0)
                if qty != 0.0:
                    return {
                        "qty": qty,
                        "entry_price": float(info.get("entryPrice", 0.0) or 0.0),
                        "mark_price": float(info.get("markPrice", 0.0) or 0.0),
                        "unrealized_pnl": float(info.get("unRealizedProfit", 0.0) or 0.0),
                    }
        return None

    def close(self):
        self.spot.close() if hasattr(self.spot, "close") else None
        self.perp.close() if hasattr(self.perp, "close") else None
