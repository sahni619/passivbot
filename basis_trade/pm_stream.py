"""Real-time Binance Portfolio Margin user-data-stream watcher.

Not a REST replacement -- run_once() still reads real state via REST (equity, positions,
uniMMR) exactly as before; this module's only job is LATENCY. Previously, a leg failure or
an exchange-side event (e.g. an unexpected position change) could sit undetected for up to
one full poll interval (default 5 minutes) before the next scheduled run_once() noticed it.
At a test account's few-dollar positions that's an inconvenience; at real deployment size
it's real, undetected naked exposure for however long that gap is. This watcher signals
"something happened, check now" the moment Binance pushes an event, cutting that gap to
roughly however long one run_once() pass takes (seconds), not however long is left on the
poll timer.

Endpoint confirmed real and working, not just a successful-handshake trap (see
order_execution.py's docstring for the real sibling-project incident where a *different*
Binance URL did exactly that): `wss://fstream.binance.com/pm/ws/<listenKey>` is grounded in
ccxt's own source (ccxt/pro/binance.py's `urls['api']['ws']['papi']` and its listenKey
auth flow for isPortfolioMargin), then independently confirmed live (2026-10-14) against the
real PM test account -- a real control-test order produced real `executionReport` /
`outboundAccountPosition` / `openOrderLoss` frames within ~1 second of being placed. PM
blends BOTH spot-style event names (`executionReport`, `outboundAccountPosition`) AND
futures-style names (`ORDER_TRADE_UPDATE`, `ACCOUNT_UPDATE`) on this ONE shared connection,
since a single PM account trades both legs -- this parses both families defensively rather
than assuming only one is possible.

listenKey lifecycle (proactive keepalive running on its own clock independent of message
arrival; reset-to-clean-slate on ANY failure rather than retrying identical stale state) is
adapted from a proven, independently-battle-tested implementation
(plezna-strategy-production/portfolio/live/order_stream.py, a sibling project, NOT part of
this repo) built for regular Binance USDM futures, here adapted for Portfolio Margin's papi
listenKey + pm/ws endpoint instead.

Thread-safety: this watcher owns its OWN dedicated ccxt client for listenKey REST calls --
it does NOT share the main loop's PMBasisExchange instance, because ccxt's synchronous
clients are not guaranteed safe for concurrent use from multiple threads at once. Callbacks
invoked from the watcher's background thread must only set thread-safe signals (e.g. a
threading.Event), never call into the main thread's ccxt clients directly.
"""
import asyncio
import json
import logging
import threading
import time

import ccxt
import websockets

from exchange import load_api_keys, _creds_kwargs

PM_WS_BASE_URL = "wss://fstream.binance.com/pm/ws"
LISTEN_KEY_KEEPALIVE_SECONDS = 30 * 60  # Binance listenKey validity is 60 min; refresh at half that
KEEPALIVE_CHECK_INTERVAL_SECONDS = 60  # independent of message arrival, see docstring above

# Event types that mean "something happened, re-check real state now" -- spans both the
# spot/margin-style names and the futures-style names PM can push on this one connection.
ACTIONABLE_EVENT_TYPES = {
    "executionReport",         # spot/margin order update
    "ORDER_TRADE_UPDATE",      # futures order update
    "outboundAccountPosition", # spot/margin balance change
    "ACCOUNT_UPDATE",          # futures balance/position change
    "openOrderLoss",           # PM-specific: potential loss on open orders
    "MARGIN_CALL",             # risk event
}


class PMAccountStreamWatcher:
    def __init__(self, account_label: str, api_keys_path: str = None):
        creds = load_api_keys(account_label, api_keys_path)
        self._listen_key_client = ccxt.binanceusdm(_creds_kwargs(creds))
        self._listen_key_client.options["defaultType"] = "swap"
        self._listen_key_client.options["portfolioMargin"] = True
        self._ws = None
        self._listen_key = None
        self._listen_key_created_at = 0.0
        self._heartbeat_task = None

    def _create_listen_key(self) -> str:
        resp = self._listen_key_client.papiPostListenKey()
        self._listen_key_created_at = time.time()
        return resp["listenKey"]

    def _keepalive_listen_key(self) -> None:
        self._listen_key_client.papiPutListenKey()
        self._listen_key_created_at = time.time()

    async def _keepalive_heartbeat_loop(self) -> None:
        """Runs on its own clock, independent of message arrival -- a quiet account (no
        order activity) must not let the listenKey silently expire just because nothing
        else triggered a recheck. Real incident this specifically prevents, documented in
        the reference implementation: a reactive-only keepalive check let the key expire on
        a predictable ~70-minute cycle during quiet stretches in production."""
        try:
            while True:
                await asyncio.sleep(KEEPALIVE_CHECK_INTERVAL_SECONDS)
                if self._listen_key is None:
                    continue
                if time.time() - self._listen_key_created_at > LISTEN_KEY_KEEPALIVE_SECONDS:
                    try:
                        await asyncio.to_thread(self._keepalive_listen_key)
                        logging.info("[pm_stream] proactive listenKey keepalive sent")
                    except Exception as e:
                        logging.warning("[pm_stream] proactive keepalive failed, will retry "
                                        "next cycle: %s: %s", type(e).__name__, e)
        except asyncio.CancelledError:
            pass

    async def _ensure_connected(self) -> None:
        """Any failure resets BOTH self._ws and self._listen_key to None before re-raising,
        so the next attempt always starts from a clean slate (fresh listenKey, fresh
        connection) rather than retrying identical stale state."""
        try:
            if self._ws is not None:
                if time.time() - self._listen_key_created_at > LISTEN_KEY_KEEPALIVE_SECONDS:
                    await asyncio.to_thread(self._keepalive_listen_key)
                return
            if self._listen_key is None:
                self._listen_key = await asyncio.to_thread(self._create_listen_key)
            url = f"{PM_WS_BASE_URL}/{self._listen_key}"
            self._ws = await websockets.connect(url)
            logging.info("[pm_stream] connected (raw websockets) to %s", PM_WS_BASE_URL)
            if self._heartbeat_task is None or self._heartbeat_task.done():
                self._heartbeat_task = asyncio.ensure_future(self._keepalive_heartbeat_loop())
        except Exception:
            self._ws = None
            self._listen_key = None
            raise

    def _handle_disconnect(self) -> None:
        self._ws = None

    async def watch_event(self):
        """Blocks until the next frame. Returns a small summary dict for an actionable
        event type, or None for a frame type to ignore. Raises on disconnect -- the caller
        (_watch_loop below) reconnects with backoff on the next call."""
        await self._ensure_connected()
        try:
            raw = await self._ws.recv()
        except Exception:
            self._handle_disconnect()
            raise

        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as e:
            logging.warning("[pm_stream] unparseable frame, skipping: %r", e)
            return None

        event_type = payload.get("e")
        if event_type == "listenKeyExpired":
            logging.warning("[pm_stream] listenKey expired -- forcing full reconnect with a fresh key")
            self._listen_key = None
            self._handle_disconnect()
            raise ConnectionError("listenKey expired")

        if event_type not in ACTIONABLE_EVENT_TYPES:
            return None
        return {"type": event_type, "raw": payload}

    async def close(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


async def _watch_loop(account_label: str, on_event, stop_event: threading.Event):
    watcher = PMAccountStreamWatcher(account_label)
    consecutive_failures = 0
    try:
        while not stop_event.is_set():
            try:
                event = await watcher.watch_event()
                consecutive_failures = 0
                if event is not None:
                    on_event(event)
            except Exception as e:
                consecutive_failures += 1
                backoff = min(60.0, 2 ** (consecutive_failures - 1))
                logging.warning("[pm_stream] disconnected (%s: %s), reconnecting in %.1fs",
                                 type(e).__name__, e, backoff)
                await asyncio.sleep(backoff)
    finally:
        await watcher.close()


def run_watcher_in_thread(account_label: str, on_event, stop_event: threading.Event) -> threading.Thread:
    """Runs the async watcher loop in its own event loop inside a dedicated background
    thread, so the rest of basis_trade (synchronous ccxt, simple polling loop) doesn't need
    to become async. `on_event` is called from THIS thread, not the main one -- it must
    only set a thread-safe signal (e.g. stop_event-style threading.Event), never touch the
    main thread's synchronous ccxt clients directly."""
    def _runner():
        asyncio.run(_watch_loop(account_label, on_event, stop_event))
    thread = threading.Thread(target=_runner, daemon=True, name="pm-stream-watcher")
    thread.start()
    return thread
