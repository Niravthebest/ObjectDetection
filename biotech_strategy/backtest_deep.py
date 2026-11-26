"""Backtest runner for strategy_deep_capitulation.py -- identical
mechanics to backtest.py, just swaps which strategy module supplies
compute_signals, so the base and refined strategies can be run and
compared side by side without overwriting either's results.

Usage:
    python3 backtest_deep.py --data-dir data --out-dir results \
        --hold-days 30 --test-start 2017-01-01 --test-end 2025-12-31
"""
import argparse
import os

import pandas as pd

from backtest import load_universe, simulate_trades, summarize, MIN_HISTORY_BARS
from strategy_deep_capitulation import compute_signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--hold-days", type=int, default=30)
    ap.add_argument("--test-start", default=None)
    ap.add_argument("--test-end", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    universe = load_universe(args.data_dir)
    latest_date = max(g["date"].max() for g in universe.values())
    test_end = pd.Timestamp(args.test_end) if args.test_end else latest_date
    universe = {sym: g[g["date"] <= test_end].reset_index(drop=True) for sym, g in universe.items()}
    test_start = (
        pd.Timestamp(args.test_start)
        if args.test_start
        else test_end - pd.DateOffset(years=3)
    )

    all_trades, all_daily_returns = [], []
    skipped = 0
    for symbol, df in universe.items():
        if len(df) < MIN_HISTORY_BARS:
            skipped += 1
            continue
        sig_df = compute_signals(df)
        trades, daily_returns = simulate_trades(sig_df, symbol, test_start, args.hold_days)
        all_trades.extend(trades)
        all_daily_returns.extend(daily_returns)

    trades_df = pd.DataFrame(all_trades)
    summary, equity = summarize(trades_df, all_daily_returns) if len(trades_df) else ({"total_trades": 0}, pd.Series(dtype=float))
    summary["hold_days"] = args.hold_days
    summary["test_start"] = str(test_start.date())
    summary["test_end"] = str(test_end.date())

    print(f"Universe: {len(universe)} symbols ({skipped} skipped for insufficient history)")
    print(f"Test window: {test_start.date()} .. {test_end.date()}  |  hold: {args.hold_days} trading days (deep-capitulation variant)")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    import json
    suffix = f"deep_hold{args.hold_days}d_{args.test_start}_{test_end.date()}"
    trades_path = os.path.join(args.out_dir, f"trades_{suffix}.csv")
    trades_df.to_csv(trades_path, index=False)
    with open(os.path.join(args.out_dir, f"summary_{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if len(equity):
        equity.rename("equity").to_csv(os.path.join(args.out_dir, f"equity_{suffix}.csv"), index_label="date")
    print(f"\nWrote {trades_path}")


if __name__ == "__main__":
    main()
