"""Sweep hold_days to pick a sensible fixed holding period, then report
the chosen one's results as a $10k initial investment.

Usage:
    python3 sweep_hold_days.py --data-dir data --out-dir results \
        --test-start 2017-01-01 --test-end 2025-12-31 \
        --hold-days-options 5,10,15,20,30 --initial-capital 10000
"""
import argparse
import os

import numpy as np
import pandas as pd

from backtest import load_universe, simulate_trades, summarize, MIN_HISTORY_BARS
from strategy import compute_signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--test-end", default=None)
    ap.add_argument("--hold-days-options", default="5,10,15,20,30")
    ap.add_argument("--initial-capital", type=float, default=10000.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    universe = load_universe(args.data_dir)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end) if args.test_end else max(g["date"].max() for g in universe.values())

    # clip each symbol's data to the requested test_end so later history doesn't leak in
    universe = {sym: g[g["date"] <= test_end].reset_index(drop=True) for sym, g in universe.items()}

    signal_cache = {}
    skipped = 0
    for symbol, df in universe.items():
        if len(df) < MIN_HISTORY_BARS:
            skipped += 1
            continue
        signal_cache[symbol] = compute_signals(df)
    print(f"Universe: {len(universe)} symbols ({skipped} skipped for insufficient history as of {test_end.date()})")

    hold_options = [int(x) for x in args.hold_days_options.split(",")]
    results = []
    best = None
    for hold_days in hold_options:
        all_trades, all_daily_returns = [], []
        for symbol, sig_df in signal_cache.items():
            trades, daily_returns = simulate_trades(sig_df, symbol, test_start, hold_days)
            all_trades.extend(trades)
            all_daily_returns.extend(daily_returns)
        trades_df = pd.DataFrame(all_trades)
        summary, equity = summarize(trades_df, all_daily_returns) if len(trades_df) else ({"total_trades": 0}, pd.Series(dtype=float))
        summary["hold_days"] = hold_days
        results.append(summary)
        label = f"hold={hold_days}d"
        print(f"{label:10s} trades={summary.get('total_trades',0):5d}  "
              f"win_rate={summary.get('win_rate',float('nan')):.1%}  "
              f"avg_ret={summary.get('avg_return_per_trade',float('nan')):+.2%}  "
              f"PF={summary.get('profit_factor',float('nan')):.2f}  "
              f"CAGR={summary.get('portfolio_cagr',float('nan')):+.1%}  "
              f"MaxDD={summary.get('portfolio_max_drawdown',float('nan')):.1%}  "
              f"Sharpe={summary.get('portfolio_sharpe_annualized',float('nan')):.2f}")
        if best is None or (summary.get("portfolio_sharpe_annualized", -np.inf) or -np.inf) > (best[1].get("portfolio_sharpe_annualized", -np.inf) or -np.inf):
            best = (hold_days, summary, trades_df, equity)

    grid_df = pd.DataFrame(results)
    grid_path = os.path.join(args.out_dir, "hold_days_sweep.csv")
    grid_df.to_csv(grid_path, index=False)
    print(f"\nWrote {grid_path}")

    hold_days, summary, trades_df, equity = best
    print(f"\n=== Best by Sharpe: hold_days={hold_days} ===")

    trades_path = os.path.join(args.out_dir, f"trades_hold{hold_days}d_{args.test_start}_{test_end.date()}.csv")
    trades_df.to_csv(trades_path, index=False)

    if len(equity):
        dollar_equity = equity * args.initial_capital
        eq_path = os.path.join(args.out_dir, f"equity_hold{hold_days}d_{args.test_start}_{test_end.date()}.csv")
        dollar_equity.rename("portfolio_value").to_csv(eq_path, index_label="date")
        final_value = dollar_equity.iloc[-1]
        peak_value = dollar_equity.cummax().iloc[-1] if False else dollar_equity.cummax().max()
        max_dd_dollar = (dollar_equity - dollar_equity.cummax()).min()
    else:
        final_value = args.initial_capital
        max_dd_dollar = 0.0

    print(f"Initial investment: ${args.initial_capital:,.2f}")
    print(f"Final portfolio value: ${final_value:,.2f}")
    print(f"Total P&L: ${final_value - args.initial_capital:,.2f}")
    print(f"Total return: {(final_value/args.initial_capital - 1):+.1%}")
    print(f"CAGR: {summary.get('portfolio_cagr', float('nan')):+.1%}")
    print(f"Max drawdown (dollar): ${max_dd_dollar:,.2f}")
    print(f"Max drawdown (pct): {summary.get('portfolio_max_drawdown', float('nan')):.1%}")
    print(f"Win rate: {summary.get('win_rate', float('nan')):.1%}")
    print(f"Profit factor: {summary.get('profit_factor', float('nan')):.2f}")
    print(f"Total trades: {summary.get('total_trades', 0)} across {summary.get('unique_symbols', 0)} symbols")
    print(f"Avg concurrent positions: {summary.get('avg_concurrent_positions', float('nan')):.1f}")


if __name__ == "__main__":
    main()
