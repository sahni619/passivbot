"""Configuration for the live long-spot / short-perp basis trade on a Binance Portfolio
Margin account.

Exposure model: a target band (min_exposure_pct to max_exposure_pct of account equity),
not a single fixed number -- the position is left alone as long as total basis-trade
notional stays inside the band, and only resized back to target_exposure_pct when it
drifts outside. This avoids rebalancing (and paying round-trip fees) on every small
equity/price fluctuation, the same reasoning passivbot's own grid uses for its
wallet-exposure tolerances.

Per-coin weight splits the target exposure across coins; each coin can be individually
disabled without affecting the others' weights (weights are renormalized across
currently-enabled coins, not reinterpreted as a fraction of the original full set).
"""
from dataclasses import dataclass, field


@dataclass
class CoinConfig:
    enabled: bool = True
    weight: float = 1.0  # relative weight among enabled coins; renormalized, not absolute


@dataclass
class BasisTradeConfig:
    enabled: bool = True  # master on/off switch; False unwinds all open positions
    account_label: str = "binance_pm_test"

    target_exposure_pct: float = 0.125  # of account equity, e.g. 0.125 = 12.5%
    min_exposure_pct: float = 0.10
    max_exposure_pct: float = 0.15

    coins: dict = field(default_factory=lambda: {
        "BTC": CoinConfig(enabled=True, weight=0.40),
        "ETH": CoinConfig(enabled=True, weight=0.35),
        # SOL's economics under this always-on approach are unconfirmed (only BTC/ETH
        # were backtested with always-on specifically) -- enabled here per the original
        # 3-coin scope, but worth validating against real data before sizing it up.
        "SOL": CoinConfig(enabled=True, weight=0.25),
    })

    quote: str = "USDT"

    # Perp-leg leverage. Capital efficiency only -- the position is delta-neutral by
    # construction (long spot + short perp of equal notional), so this does not add
    # directional price risk. The real risk leverage adds is margin-buffer risk: a
    # higher leverage setting needs less margin posted per unit of perp notional,
    # which leaves a thinner equity cushion (lower uniMMR) against the same dollar
    # move in account equity from any other source (funding payments, basis-spread
    # widening, a temporarily unhedged leg during a resize). uniMMR monitoring below
    # exists specifically to manage that risk, not price risk.
    leverage: int = 10  # user-specified ceiling; validated live only up to 2x so far

    # uniMMR ("Unified Account Maintenance Margin Ratio") safety tiers. Binance's own
    # real, documented thresholds (confirmed directly from Binance's FAQ pages, not
    # inferred): uniMMR > 1.5 = healthy/free to trade; <= 1.5 = margin call warning
    # (notification only); <= 1.2 = forced reduce-only mode (new positions/margin
    # loans blocked); <= 1.05 = liquidation triggers (system cancels cross-margin
    # orders, then force-closes until it recovers).
    #
    # The tiers below are this module's OWN, deliberately much more conservative
    # thresholds -- the standard quant/risk-system pattern (same shape as this
    # repo's own HSL tiered drawdown system) is to never operate anywhere near an
    # exchange's actual forced-action thresholds. Each tier here sits a full
    # multiple above Binance's corresponding real one, so this module de-risks via
    # its own orderly (maker-first) execution long before Binance's reduce-only or
    # liquidation mechanics would ever engage.
    unimmr_caution: float = 3.0   # stop increasing exposure further (still > 2x Binance's 1.5 warning)
    unimmr_derisk: float = 2.0    # actively reduce exposure now (still > Binance's 1.5 warning)
    unimmr_emergency: float = 1.5  # flatten everything immediately -- this IS Binance's own warning level;
                                   # reaching it means the tiers above failed to de-risk in time

    # execution (passed through to order_execution.place_and_fill_maker_first)
    reprice_interval_seconds: float = 60.0
    timeout_minutes: float = 45.0
    book_depth_levels: int = 20
    max_book_participation_pct: float = 0.20

    dry_run: bool = True  # must be explicitly set False to place real orders

    def __post_init__(self):
        if not 0.0 <= self.min_exposure_pct <= self.target_exposure_pct <= self.max_exposure_pct:
            raise ValueError(
                f"exposure band must satisfy 0 <= min <= target <= max, got "
                f"min={self.min_exposure_pct} target={self.target_exposure_pct} "
                f"max={self.max_exposure_pct}"
            )

    def enabled_coin_weights(self) -> dict:
        """Coin -> normalized weight (sums to 1.0) across currently-enabled coins only."""
        enabled = {c: cfg.weight for c, cfg in self.coins.items() if cfg.enabled}
        total = sum(enabled.values())
        if total <= 0:
            return {}
        return {c: w / total for c, w in enabled.items()}
