"""Phase 1 sanity check: confirms API credentials work and reports current account state
for both legs. Read-only -- places NO orders, changes NO state. Safe to run at any time.

Usage: python check_connectivity.py <account_label>   (e.g. binance_pm_test, see api-keys.json)
"""
import argparse
import json

from config import BasisTradeConfig
from exchange import PMBasisExchange
from manager import read_coin_state, unimmr_tier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("account_label", help="Account label from api-keys.json, e.g. binance_pm_test")
    args = parser.parse_args()

    print(f"Connecting to Binance Portfolio Margin as {args.account_label!r} (read-only)...")
    ex = PMBasisExchange(args.account_label)
    cfg = BasisTradeConfig(account_label=args.account_label)

    account = ex.account_state()
    print(f"\naccount_equity: {account['equity']}")
    print(f"uniMMR: {account['unimmr']}  (tier: {unimmr_tier(cfg, account['unimmr'])})")

    states = {}
    for coin in cfg.coins:
        states[coin] = read_coin_state(ex, coin, cfg.quote)

    print(json.dumps(
        {c: vars(s) for c, s in states.items()},
        indent=2, default=str,
    ))
    ex.close()


if __name__ == "__main__":
    main()
