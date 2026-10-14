"""Live-loop entrypoint for the basis trade: runs manager.run_once on a fixed interval,
PLUS (by default) a real-time websocket watcher (pm_stream.py) that triggers an immediate
out-of-cycle run_once() the moment Binance pushes an order/account/risk event, rather than
waiting out the rest of the poll timer. dry_run defaults to True everywhere upstream
(config.py, order_execution.py) -- pass --live explicitly to place real orders.

The poll interval (default 5 minutes) is still the steady-state cadence for ordinary
target-band rebalancing -- that part is deliberately coarse, not a placeholder (a static,
always-on allocation doesn't need to be re-evaluated every second). The real-time watcher
exists for a different, narrower reason: a leg failure or an exchange-side event (fill,
ADL, margin call) leaving the position mismatched should be caught in roughly however long
one run_once() pass takes, not however long is left on the poll timer -- confirmed live
(2026-10-14) to matter: two real mid-resize failures this session left a coin briefly
naked, self-healed only on the NEXT scheduled poll, up to several minutes later.

Usage:
  python run_live.py <account_label>                  # dry run, prints what it would do
  python run_live.py <account_label> --live            # places real orders
  python run_live.py <account_label> --live --once     # single pass, for cron/manual runs
  python run_live.py <account_label> --live --no-realtime  # poll-only, no websocket watcher
"""
import argparse
import json
import threading
import time
from datetime import datetime, timezone

from config import BasisTradeConfig
from exchange import PMBasisExchange
from manager import run_once
from pm_stream import run_watcher_in_thread


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("account_label")
    parser.add_argument("--live", action="store_true", help="place real orders (default: dry run)")
    parser.add_argument("--once", action="store_true", help="run a single pass and exit")
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--target-exposure-pct", type=float, default=0.125)
    parser.add_argument("--min-exposure-pct", type=float, default=0.10)
    parser.add_argument("--max-exposure-pct", type=float, default=0.15)
    parser.add_argument("--disabled", action="store_true", help="master switch off: unwind everything")
    parser.add_argument("--timeout-minutes", type=float, default=45.0,
                         help="maker-first timeout before taker fallback, per order slice")
    parser.add_argument("--reprice-interval-seconds", type=float, default=60.0)
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--no-realtime", action="store_true",
                         help="disable the websocket watcher; poll on --poll-seconds only")
    args = parser.parse_args()

    cfg = BasisTradeConfig(
        enabled=not args.disabled,
        account_label=args.account_label,
        target_exposure_pct=args.target_exposure_pct,
        min_exposure_pct=args.min_exposure_pct,
        max_exposure_pct=args.max_exposure_pct,
        timeout_minutes=args.timeout_minutes,
        reprice_interval_seconds=args.reprice_interval_seconds,
        leverage=args.leverage,
        dry_run=not args.live,
    )
    ex = PMBasisExchange(args.account_label)

    mode = "LIVE" if args.live else "DRY RUN"
    print(f"=== basis_trade run_live starting ({mode}) account={args.account_label} ===")
    print(f"exposure band: {cfg.min_exposure_pct:.1%}-{cfg.max_exposure_pct:.1%} "
          f"(target {cfg.target_exposure_pct:.1%}), enabled={cfg.enabled}")

    urgent = threading.Event()
    stop_watcher = threading.Event()
    watcher_thread = None
    if not args.no_realtime and not args.once:
        def _on_event(event):
            # Runs on the watcher's own background thread -- only ever touches this one
            # thread-safe flag, never the main thread's ccxt clients (see pm_stream.py).
            print(f"[REALTIME] {event['type']} -- triggering immediate check")
            urgent.set()
        watcher_thread = run_watcher_in_thread(args.account_label, _on_event, stop_watcher)
        print("real-time websocket watcher started (pass --no-realtime to disable)")

    try:
        while True:
            ts = datetime.now(timezone.utc).isoformat()
            try:
                report = run_once(cfg, ex)
                print(f"[{ts}] {json.dumps(report, default=str)}")
            except Exception as e:
                print(f"[{ts}] ERROR during run_once: {type(e).__name__}: {e}")
            if args.once:
                break
            # Sleep in short slices rather than one long call, so an urgent signal from the
            # watcher thread is noticed within ~1s instead of waiting out the full interval.
            urgent.clear()
            deadline = time.time() + args.poll_seconds
            while time.time() < deadline and not urgent.is_set():
                time.sleep(1.0)
    finally:
        stop_watcher.set()
        if watcher_thread is not None:
            watcher_thread.join(timeout=5.0)
        ex.close()


if __name__ == "__main__":
    main()
