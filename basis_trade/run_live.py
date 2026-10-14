"""Live-loop entrypoint for the basis trade: runs manager.run_once on a fixed interval
forever. dry_run defaults to True everywhere upstream (config.py, order_execution.py) --
pass --live explicitly to place real orders. Nothing about this loop is adaptive/urgent;
the strategy is a static, always-on allocation rebalanced only on band drift, so a
coarse poll interval (default 5 minutes) is intentional, not a placeholder.

Usage:
  python run_live.py <account_label>                  # dry run, prints what it would do
  python run_live.py <account_label> --live            # places real orders
  python run_live.py <account_label> --live --once     # single pass, for cron/manual runs
"""
import argparse
import json
import time
from datetime import datetime, timezone

from config import BasisTradeConfig
from exchange import PMBasisExchange
from manager import run_once


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
            time.sleep(args.poll_seconds)
    finally:
        ex.close()


if __name__ == "__main__":
    main()
