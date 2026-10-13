import asyncio

from exchanges.binance import BinanceBot
from exchanges.ccxt_bot import format_exchange_config_response
from passivbot import logging
from utils import symbol_to_coin


class BinancePortfolioBot(BinanceBot):
    """Binance Portfolio Margin (PM) account support.

    PM is a distinct Binance account model (unified risk across spot + margin + UM
    futures + CM futures + options), reachable only through the papi* endpoint family,
    not the regular fapi* endpoints BinanceBot uses. See
    docs/plans/binance_portfolio_margin_plan.md for the full design rationale.

    Confirmed directly from the installed ccxt source (checked against both 4.5.48 and
    4.5.66, not guessed from docs): ccxt's `binance` class -- which `binanceusdm`
    inherits from unmodified -- already has built-in papi routing for most unified
    methods, gated behind a single `options["portfolioMargin"]` flag: fetch_balance,
    fetch_positions_risk, create_order, cancel_order, cancel_all_orders,
    fetch_open_orders/fetch_orders, fetch_my_trades, set_leverage, and
    set_position_mode all branch internally to the matching papi* endpoint once that
    flag is set, with no other call-site changes needed. ccxt's own response-shape
    comments show the papi position-risk payload uses the SAME field names as the
    regular futures response (symbol, positionSide, positionAmt, entryPrice,
    markPrice, unRealizedProfit, leverage, ...), so BinanceBot's existing
    _normalize_positions / _build_order_params / order-fill parsing are reused
    unchanged below.

    Four things ccxt does NOT bridge, all overridden below:
    - set_margin_mode has no papi branch at all (PM risk is unified account-wide;
      there is no per-symbol isolated/cross toggle to set). update_exchange_config_by_symbols
      drops the set_margin_mode call and keeps only set_leverage.
    - fetch_position_mode ALSO has no papi branch (confirmed in both ccxt versions
      checked) -- it unconditionally calls fapiPrivateGetPositionSideDual, the
      regular futures wallet's hedge-mode setting, not PM's UM position mode. This
      matters because BinanceBot._prepare_protective_account() calls
      fetch_position_mode() as a startup safety gate and raises if it doesn't report
      hedged mode; inheriting it unchanged would check the wrong account state on a
      PM account and could either falsely block startup or falsely pass. Overridden
      below to call the raw papi* endpoint directly. Confirmed against a real PM
      test account: the endpoint works and returns the expected dualSidePosition
      field, currently false (one-way mode) on that fresh account.
    - _get_balance / _do_fetch_balance: confirmed as a REAL BUG against a real PM
      test account, not just inferred. ccxt's unified fetch_balance() resolves
      `type` to 'linear' (passivbot sets defaultType=swap; binanceusdm defaults
      defaultSubType=linear), which makes parse_balance_custom's PM branch read
      only umWalletBalance/umUnrealizedPNL into total[quote] -- the UM-futures
      sub-ledger, not overall usable PM collateral. Live capture: an account with
      $500 of real, transferred-in collateral showed fetch_balance()'s
      total["USDT"] as exactly 0.0 (umWalletBalance was 0, nothing had been
      allocated to UM trading yet), while the SAME account's raw info showed
      crossMarginFree=500 and papiGetAccount()'s accountEquity=499.32. The original
      version of this class read fetch_balance()'s total[quote] directly and would
      have treated this account as having zero balance. Fixed below by calling
      papiGetAccount() directly instead (same shape of fix as OKX's totalEq-based
      _get_balance -- a single, precomputed, account-wide equity figure).
    - _normalize_balance_diagnostics (inherited from BinanceBot) assumes
      fetch_balance()'s ccxt-unified total/free/used shape; since _do_fetch_balance
      now returns papiGetAccount()'s raw shape instead, that diagnostics hook will
      report "malformed/missing_total" for PM accounts. This is a monitoring-only
      cosmetic gap, not a trading-correctness issue (nothing in the strategy path
      reads it) -- left as a known gap rather than reworked, to keep this change
      scoped to what actually affects trading.

    NOT YET FULLY VALIDATED. Live-tested against a real (currently flat,
    zero-position) PM test sub-account: load_markets, papiGetBalance,
    fetch_positions_risk, papiGetUmPositionSideDual, and papiGetAccount all
    confirmed working with real data. Specifically still open:
    1. fetch_positions_risk() calls self.load_leverage_brackets() before issuing the
       request; whether that sub-call behaves correctly (or is even meaningful) for
       a PM account is unverified.
    2. _get_balance now reads papiGetAccount()'s accountEquity field. On the one
       real account tested, accountEquity (499.32) and actualEquity (499.37)
       differed slightly, and the account was flat (zero positions, no unrealized
       PnL) at the time, so there's no way yet to confirm which field does/doesn't
       include uPNL. Regular BinanceBot's _get_balance deliberately excludes uPNL
       (totalCrossWalletBalance); accountEquity is the closer-named match to that
       convention, but re-confirm once the test account has an open position with
       real floating PnL -- whichever field stays flat as uPNL moves is the
       uPNL-excluded one.
    3. Whether PM's "margin mode / hedge mode already set" case returns the same
       error code BinanceBot already handles (-4059 for hedge mode via the inherited
       update_exchange_config; MarginModeAlreadySet is skipped entirely here rather
       than assumed) is unverified -- if PM uses a different code, update_exchange_config
       could raise on every restart once hedge mode is already correctly set.
    4. No order placement/cancellation has been tested against the real account yet
       (read-only validation only, deliberately -- no leverage/order changes were
       made to real capital during this round of testing).
    """

    def __init__(self, config: dict):
        super().__init__(config)

    def create_ccxt_sessions(self):
        super().create_ccxt_sessions()
        self.cca.options["portfolioMargin"] = True
        if self.ccp is not None:
            self.ccp.options["portfolioMargin"] = True
        logging.info(f"{self.exchange}: Portfolio Margin mode enabled (papi endpoints)")

    def can_watch_orders(self) -> bool:
        """PM has no ccxt.pro websocket class of its own, and ccxt.pro's regular
        binanceusdm watchOrders() stream is backed by the non-PM listen-key flow, which
        would not reflect a PM account's real order/position updates. Fall back to REST
        polling (an existing, first-class supported path, not a hack) rather than
        building new raw websocket handling in the same change that introduces the
        account type -- see design doc Testing & Rollout for the reasoning."""
        return False

    async def _do_fetch_positions(self) -> list:
        """PM: ccxt's fetch_positions_risk() routes to papiGetUmPositionRisk once the
        portfolioMargin option is set (done once in create_ccxt_sessions above); the
        response uses the same field names as the regular positionrisk endpoint, so
        BinanceBot._normalize_positions handles it unchanged."""
        return await self.cca.fetch_positions_risk()

    async def _do_fetch_balance(self) -> dict:
        """PM: call papiGetAccount() directly rather than ccxt's unified
        fetch_balance() -- see class docstring for the confirmed-real bug this
        avoids (fetch_balance()'s total[quote] read as 0 against a real account
        with real collateral)."""
        return await self.cca.papiGetAccount()

    def _get_balance(self, fetched: dict) -> float:
        """PM: papiGetAccount()'s accountEquity field -- a single, precomputed,
        account-wide equity figure, confirmed against a real account. See class
        docstring Open Item 2 for the still-unresolved accountEquity vs
        actualEquity question.

        Known gap: this is a single account-wide number by construction, so it
        already handles multi-asset collateral correctly (unlike reading a single
        asset's entry from papiGetBalance would) -- this resolves what was
        previously flagged as an open gap for BTC-collateral-style accounts.
        """
        if "accountEquity" not in fetched:
            raise KeyError(f"{self.exchange}: papiGetAccount response missing accountEquity")
        return float(fetched["accountEquity"])

    async def update_exchange_config_by_symbols(self, symbols):
        """PM: no per-symbol margin-mode toggle exists -- risk is unified account-wide,
        and ccxt's set_margin_mode has no papi branch to route through at all. Only set
        leverage (which does have papi routing via the global option)."""
        coros_to_call_lev = {}
        for symbol in symbols:
            coros_to_call_lev[symbol] = asyncio.create_task(
                self.cca.set_leverage(self._calc_leverage_for_symbol(symbol), symbol=symbol)
            )
        for symbol in symbols:
            log_symbol = symbol_to_coin(symbol, verbose=False) or symbol
            try:
                res = await coros_to_call_lev[symbol]
                logging.info(f"{log_symbol}: leverage={format_exchange_config_response(res)}")
            except Exception as e:
                logging.error(f"{log_symbol}: error setting leverage (PM) {e}")

    async def _prepare_protective_account(self):
        """PM: fetch_position_mode() has no papi routing in ccxt (confirmed in both
        4.5.48 and 4.5.66) and would check the regular futures wallet's hedge-mode
        setting, not PM's UM position mode -- inheriting BinanceBot's version here
        would validate the wrong account state. Call the raw papi endpoint directly.

        Response field name (dualSidePosition) is inferred from the regular
        Get-Current-Position-Mode endpoint's documented shape, not captured from a
        real papi response -- validate this against the real PM test account before
        trusting it to gate live startup.
        """
        response = await self.cca.papiGetUmPositionSideDual()
        dual_side = response.get("dualSidePosition")
        if dual_side is not True and str(dual_side).lower() != "true":
            raise RuntimeError(
                "Binance PM protective startup requires existing hedge position mode"
            )
