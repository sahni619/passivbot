# Binance Portfolio Margin Support — Design Plan

## Status (updated after a first implementation pass)

A first-draft implementation now exists: `src/exchanges/binance_portfolio.py`
(`BinancePortfolioBot(BinanceBot)`), wired into `setup_bot()` in `src/passivbot.py` and
into the `ccxt_contracts` fixture-capture tool. Existing tests pass unmodified; a new
test (`tests/test_setup_bot_fallback.py::test_setup_bot_binance_portfolio_margin_account_type`)
covers the new dispatch branch.

Writing it surfaced something that changes the scope of this whole plan for the
better: **ccxt's installed `binance.py` already has built-in Portfolio Margin routing
for most unified methods**, confirmed by reading the actual installed package source
(not docs, not guessed) — see the revised "Why this doesn't already work" and
"Proposed Shape" sections below. The real diff needed is much smaller than originally
scoped here. This is still a first draft, **not validated against a real PM account**
— see the narrowed Open Questions at the bottom.

## Purpose

Add support for running a Binance account in Portfolio Margin (PM) mode, so that the
existing grid-trading engine and a new standing spot+perp basis trade can run on the
*same* sub-account.

This is client-driven: a live account wants to add a delta-neutral BTC/ETH funding-rate
basis sleeve (long spot, short perp, collects funding) alongside its existing grid
strategy, on one sub-account rather than two. Binance PM is account-wide — a sub-account
is either fully on PM or fully on regular margin, there is no per-position mix — so
running both strategies on one account requires the grid engine itself to work under PM,
not just a new connector for the basis leg.

## Why this doesn't already work

`src/exchanges/ccxt_bot.py` gets its exchange connection via
`getattr(ccxt_async, ccxt_id)` — for Binance this resolves to `ccxt.binanceusdm`
(`to_ccxt_exchange_id` in `src/utils.py` hardcodes `"binance" -> "binanceusdm"`).
Portfolio Margin is a structurally separate account model on Binance's side (unified
risk across spot + margin + UM futures + CM futures + options), reachable through a
parallel `papi*` endpoint family — `setup_bot()` never set an account up to use it, and
nothing in `BinanceBot` asked for it.

**Correction from the original version of this doc:** this is *not* because ccxt lacks
PM support. Confirmed directly from the installed `ccxt/binance.py` source:
`binanceusdm` is `class binanceusdm(binance, ImplicitAPI)` — a thin subclass that
inherits `binance`'s methods unmodified, and `binance.py` already has PM routing built
into most of its unified methods (`fetch_balance`, `fetch_positions_risk`,
`create_order`, `cancel_order`, `cancel_all_orders`, `fetch_open_orders`/`fetch_orders`,
`fetch_my_trades`, `set_leverage`, `set_position_mode`) gated behind a single
`options["portfolioMargin"]` flag (`handle_option_and_params_2(params, <method>, "papi",
"portfolioMargin", False)` in each). Once set, each of those methods branches
internally to the matching `papi*` endpoint (`papiGetBalance`, `papiGetUmPositionRisk`,
`papiPostUmOrder`, `papiPostMarginOrder`, `papiPostUmLeverage`,
`papiPostUmPositionSideDual`, ...) with no other call-site changes required. ccxt's own
response-shape doc comments show the papi position-risk payload uses the **same field
names** as the regular endpoint (`symbol`, `positionSide`, `positionAmt`, `entryPrice`,
`markPrice`, `unRealizedProfit`, `leverage`, ...), so `BinanceBot`'s existing
`_normalize_positions` / `_build_order_params` / order-fill parsing are reused
unchanged in `BinancePortfolioBot`.

Two things ccxt does *not* bridge:
- `set_margin_mode` has no `papi` branch at all — PM risk is unified account-wide, so
  there's no per-symbol isolated/cross toggle to set. Confirmed by reading the method
  body directly; it unconditionally calls the regular `fapiPrivatePostMarginType`.
- `_get_balance`: regular futures reads a single `info.totalCrossWalletBalance` field.
  PM's `papiGetBalance` returns one entry *per collateral asset* instead (confirmed via
  ccxt's `parse_balance_custom`), so this hook needs real PM-aware logic — this is the
  one place the original scoping in this doc was right that real work is needed.

## Coin-universe partition: how the basis trade and the grid engine avoid colliding

Confirmed design (applies identically on the OKX plan): the basis trade runs on 2-3
specific coins only (e.g. BTC, ETH, SOL), excluded from passivbot's own grid universe on
this account via the existing `live.ignored_coins.long` config field — no new code,
already-supported mechanism used throughout this fleet. Passivbot's mean-reversion grid
keeps running on every other approved coin untouched.

This is a structural guarantee, not a trust assumption: the grid engine and the basis
trade never hold a position in the same symbol, by config. The only thing they actually
share on this account is the PM margin pool itself — which is the entire point of this
plan — not any individual order or position.

## Non-Goals

- No change to trading logic, risk, unstuck, or HSL behavior. This is a connector-layer
  change only — same "Rust owns order behavior, Python owns exchange I/O" boundary as
  `exchange_connector_event_contract_plan.md`.
- No behavior change whatsoever for existing non-PM accounts on any exchange. PM support
  is additive and opt-in per config, not a modification to the default Binance path.
- v1 does not include the basis-trade strategy logic itself (entry/exit/rebalance rules)
  — that's a separate, smaller piece once the account can actually place both kinds of
  orders. This plan is scoped to making the account connector work, not the strategy.
- v1 does not include CM futures or options under PM — only UM futures (for the grid
  engine, unchanged symbol universe) and margin/spot (for the basis trade's long leg).
- v1 does not build custom PM websocket streaming. See Testing & Rollout below.

## Proposed Shape (implemented as a first draft — see Status above)

Not a new hook-by-hook rewrite of `binance.py`. Instead, `BinancePortfolioBot(BinanceBot)`
in a new `src/exchanges/binance_portfolio.py`, dispatched from `setup_bot()` when the
account's `api-keys.json` entry has `"account_type": "portfolio_margin"` (an
account-level property, parallel to the existing `"exchange"` field — deliberately
*not* a new `live.*` schema field, since account type isn't a strategy parameter and
adding one would mean touching `config/schema.py` + `config/validate.py` +
`config/coerce.py` for something that belongs in the credentials file `load_user_info`
already passes through untouched).

| Hook | Regular futures (`BinanceBot`) | PM override (`BinancePortfolioBot`) |
|---|---|---|
| `create_ccxt_sessions()` | — | sets `self.cca.options["portfolioMargin"] = True` once; everything below rides on this single flag |
| `_do_fetch_positions()` | raw `fapiprivatev3_get_positionrisk()` call | `self.cca.fetch_positions_risk()` (unified; routes to `papiGetUmPositionRisk` via the flag) |
| `_normalize_positions()` | — | **inherited unchanged** — field names match (confirmed from ccxt's response-shape comments) |
| order creation / cancellation / `_build_order_params()` | fapi `create_order`/`cancel_order` | **inherited unchanged** — same flag routes `create_order`/`cancel_order`/`cancel_all_orders` to `papiPostUmOrder`/`papiDeleteUmOrder`/etc., same response shape |
| `fetch_open_orders` / `fetch_my_trades` / pnl fetching | fapi | **inherited unchanged** — same flag, same routing mechanism |
| `_get_balance(fetched)` | reads `info.totalCrossWalletBalance` | **real override, not inherited.** `papiGetBalance` returns one entry per collateral asset; reads the quote asset's own folded `total[quote]` entry (via ccxt's `parse_balance_custom`). Does **not** yet convert non-quote collateral into quote terms the way OKX's `totalEq`-based `_get_balance` does — flagged as an open gap, not silently assumed to work. |
| `update_exchange_config_by_symbols()` | sets margin mode + leverage | **real override.** Drops the `set_margin_mode` call (no `papi` branch exists); keeps `set_leverage` (which does route via the same flag) |
| `update_exchange_config()` (hedge mode) | `set_position_mode(True)` | **inherited unchanged** — `set_position_mode` also routes via the same flag, confirmed from source |
| `can_watch_orders()` | CCXT `watchOrders` capability check | **forced `False`.** No ccxt.pro PM websocket class exists, and the regular `binanceusdm` watch stream is backed by the non-PM listen-key flow — would silently watch the wrong account state. Falls back to REST polling (first-class supported path, not a hack). |

The basis-trade spot/margin order-placement work (new surface area for the engine, not
covered by any of the above) remains out of scope for this connector piece — see
Non-Goals.

Everything else in the grid engine (signal generation, order sizing, wallet-exposure
math, HSL/orchestrator) is unchanged — it already only talks to the hook layer, not
directly to ccxt or to Binance's REST surface.

## Testing & Rollout

This repo already has the right tool for this: `src/ccxt_contracts.py` +
`capture_ccxt_contracts` / `diff_ccxt_contracts` (see
`docs/ai/ccxt_upgrade_workflow.md`). Same workflow applies here even though it's a new
account type rather than a ccxt version bump — capture real PM response snapshots
(balance, positions, order ack, cancel ack) from a real (small-capital) PM sub-account,
build offline fixtures from them, and get the `_normalize_*`/`_get_*` hooks covered by
`tests/ccxt_upgrade`-style regression tests before anything touches meaningful capital.

**v1 explicitly skips custom PM websocket support** and relies on `can_watch_orders()`
returning `False` for PM accounts — already a first-class supported path (REST polling,
not a hack), per `watch_orders()`'s existing fallback. Reasoning: this session already
hit one real, hard-to-diagnose bug from ccxt.pro's websocket layer silently dropping
Binance's private stream, on the *already-supported* account type. Building new raw
websocket handling for PM, which has no ccxt.pro class to begin with, in the same PR
that introduces the account type itself, stacks two unproven things on top of each
other. Ship polling first, add PM streaming as a follow-up once the REST path is
validated live.

Rollout order (1 done, 2 is next):
1. ~~Implement the hook overrides, PM behind the explicit opt-in `account_type` field.~~
   Done: `src/exchanges/binance_portfolio.py` + `setup_bot()`/`ccxt_contracts.py`
   wiring. Built by reading ccxt's own source for routing/shape confirmation, not
   against a real account — treat as a first draft.
2. **Next:** capture real PM API response shapes (balance, positions, orders) against a
   real small-capital PM sub-account, using `capture_contract_snapshot()` in
   `src/ccxt_contracts.py` (already works against `BinancePortfolioBot` once the
   account's `api-keys.json` entry has `account_type: portfolio_margin` — no further
   tooling needed). This is what actually answers Open Questions 1–2 above.
3. Offline fixture tests (`tests/ccxt_upgrade`-equivalent) built from those real
   captures for every overridden hook — not written yet, deliberately, since fixtures
   built before a real capture would just encode the same unverified assumptions.
4. Live validation on the same small-capital PM sub-account: confirm balance reads,
   order placement/cancellation (both UM and margin legs), position tracking, and that
   `account_type != "portfolio_margin"` paths are provably untouched (existing test
   suite is green as of this draft; no shared code was modified in a way that changes
   default behavior — confirmed by running the existing suite, not assumed).
5. PR to upstream, scoped to exactly the hooks above — no strategy logic included.
6. Basis-trade strategy logic (separate, smaller piece) layers on top once the account
   connector is merged.

## Open Questions

Resolved by reading ccxt's own source rather than guessing (no longer open):

- ~~Does `papiGetUmPositionRisk`/order response schema match the regular endpoint?~~
  For positions: yes, confirmed field-name parity from ccxt's response-shape comments.
  For balance: no — PM's response shape genuinely differs (one entry per collateral
  asset), which is why `_get_balance` has a real override rather than being inherited.
- ~~Preferred shape for the opt-in flag?~~ `api-keys.json`'s per-user `account_type`
  field, read in `setup_bot()` — not a new `live.*` schema field. Account type is a
  property of the credentials, same as `exchange` already is; this avoids touching
  `config/schema.py`/`validate.py`/`coerce.py` for a non-strategy parameter.
- ~~Subclass vs. mode flag?~~ Subclass (`BinancePortfolioBot(BinanceBot)`), matching
  how `exchanges/` already handles every other exchange variant. Keeps the regular
  futures path's diff surface at zero.

Still genuinely open (need a real account, not more reading):

1. Whether `fetch_positions_risk()`'s internal `self.load_leverage_brackets()` call
   (made before issuing the request, in both the regular and PM branches) behaves
   correctly — or is even meaningful — for a PM account. Unverified.
2. `_get_balance`'s handling of *multi-asset* collateral (e.g. BTC posted as margin
   rather than USDT). Current implementation only correctly sums the quote asset's own
   entry; a true cross-asset equity calculation (OKX's `totalEq`-style) would need a
   real multi-asset PM account to build and verify against.
3. Any existing internal PM testing account/sandbox, or do we need to stand up a fresh
   small-capital PM sub-account specifically for fixture capture (`capture_contract_snapshot`
   in `src/ccxt_contracts.py` already works against it once `account_type` is set in
   `api-keys.json` — no further tooling work needed, just a real account to point it at)
   and live validation?
4. Nothing in this draft has executed against Binance's real `papi*` endpoints. Treat
   every claim above as "true per ccxt's source," not "true in production," until a
   real capture confirms it.
