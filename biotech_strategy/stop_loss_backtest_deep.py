"""Stop-loss sweep for the deep-capitulation strategy, 30-day hold.

The hold-days sweep showed every holding period produces a brutal
-85% to -91% max drawdown, driven by individual trades that fall hard
before (if ever) recovering -- e.g. MTRI -75.6%, SLS -85.7% on a
single position. This tests whether capping the loss on any one
trade with a stop-loss (checked via each day's low, no take-profit
cap so winners still run the full hold) tames that portfolio-level
drawdown, and at what cost to the upside that made the strategy worth
testing in the first place.

Usage:
    python3 stop_loss_backtest_deep.py --data-dir data --out-dir results \
        --hold-days 30 --test-start 2017-01-01 --test-end 2025-12-31 \
        --stops 0.05,0.10,0.15,0.20,0.30
"""
import argparse
import os

import numpy as np
import pandas as pd

from backtest import load_universe, max_drawdown, build_portfolio_equity_curve, MIN_HISTORY_BARS
from strategy_deep_capitulation import compute_signals


def simulate_trades_stop(df: pd.DataFrame, symbol: str, test_start: pd.Timestamp,
                          max_hold_days: int, stop_pct) -> tuple:
    """Same entry timing as backtest.py (next bar's open after a signal).
    If stop_pct is None, this degenerates to the plain fixed-hold exit.
    Otherwise exits early if a day's low crosses the stop price, exiting
    at the open if it gapped through, else at the stop price itself;
    no take-profit cap, so winners ride to the full hold."""
    trades = []
    daily_returns = []
    in_position_until = -1
    n = len(df)
    for t in range(n):
        if t <= in_position_until:
            continue
        if not df["entry_signal"].iloc[t]:
            continue
        if df["date"].iloc[t] < test_start:
            continue
        entry_idx = t + 1
        if entry_idx >= n:
            continue
        entry_price = df["open"].iloc[entry_idx]
        if pd.isna(entry_price) or entry_price <= 0:
            continue
        stop_price = entry_price * (1 - stop_pct) if stop_pct is not None else None
        last_possible = min(entry_idx + max_hold_days - 1, n - 1)
        if last_possible < entry_idx:
            continue

        exit_idx, exit_price, exit_reason = None, None, None
        for i in range(entry_idx, last_possible + 1):
            o, l = df["open"].iloc[i], df["low"].iloc[i]
            if pd.isna(o) or pd.isna(l):
                break
            if stop_price is not None and l <= stop_price:
                exit_price = o if o < stop_price else stop_price
                exit_idx, exit_reason = i, "stop"
                break
        if exit_idx is None:
            exit_idx = last_possible
            exit_price = df["close"].iloc[exit_idx]
            exit_reason = "time"
        if pd.isna(exit_price):
            continue

        ret = exit_price / entry_price - 1.0
        trades.append({
            "symbol": symbol,
            "signal_date": df["date"].iloc[t],
            "entry_date": df["date"].iloc[entry_idx],
            "exit_date": df["date"].iloc[exit_idx],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return": ret,
            "exit_reason": exit_reason,
        })
        prev_price = entry_price
        for i in range(entry_idx, exit_idx + 1):
            price = df["close"].iloc[i] if i < exit_idx else exit_price
            if pd.isna(price) or prev_price <= 0:
                break
            daily_returns.append((df["date"].iloc[i], price / prev_price - 1.0))
            prev_price = price
        in_position_until = exit_idx
    return trades, daily_returns


def summarize(trades_df: pd.DataFrame, daily_returns: list) -> tuple:
    n = len(trades_df)
    if n == 0:
        return {"total_trades": 0}, pd.Series(dtype=float)
    wins = trades_df[trades_df["return"] > 0]
    losses = trades_df[trades_df["return"] <= 0]
    gross_win = wins["return"].sum()
    gross_loss = -losses["return"].sum()
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf

    equity = build_portfolio_equity_curve(daily_returns)
    years_span = max(len(equity) / 252.0, 1e-6)
    cagr = equity.iloc[-1] ** (1 / years_span) - 1.0 if len(equity) else np.nan
    daily_ret_series = equity.pct_change().dropna()
    sharpe = (
        daily_ret_series.mean() / daily_ret_series.std() * np.sqrt(252)
        if daily_ret_series.std() and daily_ret_series.std() > 0
        else np.nan
    )
    reason_counts = trades_df["exit_reason"].value_counts(normalize=True).to_dict()

    return {
        "total_trades": n,
        "win_rate": len(wins) / n,
        "avg_return_per_trade": trades_df["return"].mean(),
        "median_return_per_trade": trades_df["return"].median(),
        "avg_win": wins["return"].mean() if len(wins) else np.nan,
        "avg_loss": losses["return"].mean() if len(losses) else np.nan,
        "profit_factor": profit_factor,
        "pct_stopped_out": reason_counts.get("stop", 0),
        "pct_time_exit": reason_counts.get("time", 0),
        "portfolio_total_return": equity.iloc[-1] - 1.0 if len(equity) else np.nan,
        "portfolio_cagr": cagr,
        "portfolio_max_drawdown": max_drawdown(equity) if len(equity) else np.nan,
        "portfolio_sharpe_annualized": sharpe,
    }, equity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--hold-days", type=int, default=30)
    ap.add_argument("--test-start", default="2017-01-01")
    ap.add_argument("--test-end", default="2025-12-31")
    ap.add_argument("--stops", default="0.05,0.10,0.15,0.20,0.30")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    universe = load_universe(args.data_dir)
    test_end = pd.Timestamp(args.test_end)
    universe = {sym: g[g["date"] <= test_end].reset_index(drop=True) for sym, g in universe.items()}
    test_start = pd.Timestamp(args.test_start)

    signal_dfs = {}
    skipped = 0
    for symbol, df in universe.items():
        if len(df) < MIN_HISTORY_BARS:
            skipped += 1
            continue
        signal_dfs[symbol] = compute_signals(df)
    print(f"Universe: {len(signal_dfs)} symbols ({skipped} skipped for insufficient history)")

    stops = [None] + [float(x) for x in args.stops.split(",")]  # None = no-stop baseline
    results = []
    for stop_pct in stops:
        all_trades, all_daily_returns = [], []
        for symbol, sig_df in signal_dfs.items():
            trades, daily_returns = simulate_trades_stop(sig_df, symbol, test_start, args.hold_days, stop_pct)
            all_trades.extend(trades)
            all_daily_returns.extend(daily_returns)
        trades_df = pd.DataFrame(all_trades)
        summary, equity = summarize(trades_df, all_daily_returns)
        summary["stop_pct"] = stop_pct if stop_pct is not None else "none"
        results.append(summary)
        label = f"stop={'none' if stop_pct is None else f'{stop_pct:.0%}'}"
        print(f"{label:10s} trades={summary.get('total_trades',0):4d}  "
              f"win_rate={summary.get('win_rate',float('nan')):.1%}  "
              f"avg_ret={summary.get('avg_return_per_trade',float('nan')):+.2%}  "
              f"PF={summary.get('profit_factor',float('nan')):.2f}  "
              f"CAGR={summary.get('portfolio_cagr',float('nan')):+.1%}  "
              f"MaxDD={summary.get('portfolio_max_drawdown',float('nan')):.1%}  "
              f"Sharpe={summary.get('portfolio_sharpe_annualized',float('nan')):.2f}  "
              f"stopped_out={summary.get('pct_stopped_out',float('nan')):.1%}")

        suffix = f"stop{'none' if stop_pct is None else int(stop_pct*100)}_hold{args.hold_days}d"
        trades_df.to_csv(os.path.join(args.out_dir, f"trades_deep_{suffix}.csv"), index=False)
        if len(equity):
            equity.rename("equity").to_csv(os.path.join(args.out_dir, f"equity_deep_{suffix}.csv"), index_label="date")

    grid_df = pd.DataFrame(results)
    grid_path = os.path.join(args.out_dir, f"stop_sweep_deep_hold{args.hold_days}d.csv")
    grid_df.to_csv(grid_path, index=False)
    print(f"\nWrote {grid_path}")


if __name__ == "__main__":
    main()
